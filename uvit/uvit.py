import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
from . import modules_uvit as modules
import evaluate
import loss
import pandas as pd


class UNetViTModel(keras.Model):
	def __init__(self, flags):
		super().__init__()
		self.flags = flags

		self.experiment_name = self.flags.name
		self.samples_dir = self.flags.sample_dir
		self.models_dir = self.flags.checkpoints_dir
		self.hist_dir = self.flags.hist_path
		self.image_size = self.flags.crop_size
		self.batch_size = self.flags.batch_size

		self.timesteps = self.flags.timesteps
		self.img_size = self.flags.crop_size
		self.widths = [self.flags.first_conv_channels * mult for mult in self.flags.channel_multiplier]

		self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
		self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")

		self.vgg_feature_loss_coeff = 1 #flags.vgg_feature_loss_coeff
		self.ssim_loss_coeff = flags.ssim_loss_coeff

		self.vgg_loss = loss.VGGFeatureMatchingLoss()

		self.optimizer = keras.optimizers.Adam(learning_rate=self.flags.gen_lr)


		self.network = self.build_model(self.img_size,
							self.flags.img_channels,
							self.widths,
							self.flags.has_attention,
							self.flags.num_res_blocks,
							self.flags.norm_groups)

		
	@property
	def metrics(self):
		return [
			self.vgg_loss_tracker,
			self.ssim_loss_tracker
		]
	

	def build_model(
			self,
			img_size,
			img_channels,
			widths,
			has_attention,
			num_res_blocks=2,
			norm_groups=8,
			interpolation="nearest",
			activation_fn=keras.activations.swish,
			first_conv_channels = 32,
	):
		image_input = layers.Input(
			shape=(img_size, img_size, img_channels), name="image_input"
		)
		time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
		
		x = layers.Conv2D(
			first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=modules.kernel_init(1.0),
		)(image_input)
		
		temb = modules.TimeEmbedding(dim=first_conv_channels * 4)(time_input)
		temb = modules.TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
		
		skips = [x]
		
		# DownBlock
		for i in range(len(widths)):
			for _ in range(num_res_blocks):
				x = modules.ResidualBlockLayer(
					widths[i], groups=norm_groups, activation_fn=activation_fn
				)([x, temb])
				if has_attention[i]:
					x = modules.AttentionBlock(widths[i], groups=norm_groups)(x)
				skips.append(x)
			
			if widths[i] != widths[-1]:
				x = modules.DownSample(widths[i])(x)
				skips.append(x)
		
		# MiddleBlock
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		x = modules.AttentionBlock(widths[-1], groups=norm_groups)(x)
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		
		# UpBlock
		for i in reversed(range(len(widths))):
			for _ in range(num_res_blocks + 1):
				x = layers.Concatenate(axis=-1)([x, skips.pop()])
				x = modules.ResidualBlockLayer(
					widths[i], groups=norm_groups, activation_fn=activation_fn
				)([x, temb])
				if has_attention[i]:
					x = modules.AttentionBlock(widths[i], groups=norm_groups)(x)
			
			if i != 0:
				x = modules.UpSample(widths[i], interpolation=interpolation)(x)
		
		# End block
		x = layers.GroupNormalization(groups=norm_groups)(x)
		x = activation_fn(x)
		x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=modules.kernel_init(0.0))(x)
		x = layers.Activation('tanh')(x)
		return keras.Model(inputs=[image_input, time_input], outputs=x, name="unetvit")
	
	def compile(self, **kwargs):
		super().compile(**kwargs)
	
	def call(self, inputs):
		image_input, time_input = inputs
		return self.network([image_input, time_input])
	
	def save_model(self, path):
		self.network.save(path)
	
	def train_step(self, data):
		source, target = data
		batch_size = tf.shape(source)[0]
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		with tf.GradientTape() as tape:
			pred_ = self.network([source, t], training=True)

			vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(target, pred_)
			ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(target, pred_)
			total_loss = vgg_loss + ssim_loss
		
		gradients = tape.gradient(total_loss, self.network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

		# Report progress.
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		return {"vgg_loss": vgg_loss, "ssim_loss": ssim_loss}
	
	def test_step(self, data):
		source, target = data
		batch_size = tf.shape(source)[0]
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		pred_ = self.network([source, t])
		
		vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(target, pred_)
		ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(target, pred_)
		total_loss = vgg_loss + ssim_loss

		# Report progress.
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)
		
		return {"vgg_loss": vgg_loss, "ssim_loss": ssim_loss}
	
	def generate_images(self, source, num_images=8):
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
		)
		
		return self.network([source[:num_images], t])
	
	def plot_images(
			self, val_dataset, logs=None, num_rows=2, num_cols=4, figsize=(12, 5)
	):
		self.val_images = val_dataset
		generated_samples = self.generate_images(self.val_images[0], num_images=num_rows * num_cols)
		
		_, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
		for i, image in enumerate(generated_samples.squeeze()):
			if num_rows == 1:
				ax[i].imshow(image, cmap='gray')
				ax[i].axis("off")
			else:
				ax[i // num_cols, i % num_cols].imshow(image)
				ax[i // num_cols, i % num_cols].axis("off")
		
		plt.tight_layout()
		plt.show()


	def model_evaluate(self, test_dataset, epoch=0):


		results = []
		
		num_batches = len(test_data[0]//self.batch_size)

		test_data = next(iter(test_dataset))

		for i in range(0, num_batches, self.batch_size):
			ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]


		#for ct, mri in test_data:
			
			fake_mri = self.generate_images(ct)

			# normalize to values between 0 and 1
			mri = (mri + 1.0) / 2.0
			fake_mri = (fake_mri + 1.0) / 2.0
			

			fid = evaluate.calculate_fid(mri, fake_mri, 
				input_shape=(self.img_size, self.img_size, 3))

			mse, mae, cs, psnr, ssim = evaluate.get_metrics(mri, fake_mri)

			results.append([fid, mse, mae, cs, psnr, ssim])
			print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))

		results = np.array(results, dtype=object).mean(axis=0)

		filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
		results_dir = os.path.join(self.flags.result_logs, self.flags.name)
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		log_file = os.path.join(results_dir, filename)
		np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")

	
	def plot_losses(self, hist):
		
		exp_path = self.hist_dir + self.experiment_name
		
		if not os.path.exists(exp_path):
			os.makedirs(exp_path)
			
		# save history to csv   
		hist_df = pd.DataFrame(hist) 
		hist_df.to_csv(exp_path + '/hist.csv')
		
		losses = ['vgg', 'ssim']
		
		# plot losses
		for loss in losses:
			plt.figure()
			plt.plot(hist[loss + '_loss'])
			plt.plot(hist['val_' + loss + '_loss'])
			plt.legend([loss + '_loss','val_' + loss + '_loss'],loc='upper right')
			plt.savefig(exp_path + '/uvit_' + loss + '.png')

