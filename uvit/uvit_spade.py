import matplotlib.pyplot as plt
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import loss
from . import modules
import evaluate
import pandas as pd
#import tensorflow_addons as tfa

class UNetViTModel(kr.Model):
	def __init__(self, flags):
		super().__init__()

		self.flags = flags
		self.experiment_name = self.flags.name
		self.samples_dir = self.flags.sample_dir
		self.models_dir = self.flags.checkpoints_dir
		self.hist_dir = self.flags.hist_path

		self.batch_size = self.flags.batch_size
		self.timesteps = self.flags.timesteps
		self.img_size = self.flags.crop_size
		self.img_channels = self.flags.img_channels
		self.first_conv_channels = self.flags.first_conv_channels
		self.widths = [self.first_conv_channels * mult for mult in self.flags.channel_multiplier]
		self.has_attention = self.flags.has_attention
		self.num_res_blocks = self.flags.num_res_blocks
		self.norm_groups = self.flags.norm_groups
		self.latent_dim = flags.latent_dim
		self.mask_shape = (flags.crop_size, flags.crop_size, 2)
		
		self.vgg_loss = loss.VGGFeatureMatchingLoss()
		self.optimizer = kr.optimizers.Adam(learning_rate=self.flags.gen_lr)
		self.discriminator_loss = loss.DiscriminatorLoss()

		self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
		self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")

		self.vgg_feature_loss_coeff = 1 #flags.vgg_feature_loss_coeff
		self.ssim_loss_coeff = flags.ssim_loss_coeff
		self.discriminator = modules.Discriminator(self.flags)
		self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
		self.discriminator_optimizer = kr.optimizers.Adam(self.flags.disc_lr, 
														  beta_1=self.flags.disc_beta_1,
														  beta_2=self.flags.disc_beta_2)
		
		self.sampler = modules.GaussianSampler(self.batch_size, self.latent_dim)
		
		self.network = self.build_model(self.img_size,
							self.img_channels,
							self.widths,
							self.has_attention,
							self.num_res_blocks,
							self.norm_groups)
		#self.network.compile(optimizer=self.optimizer, loss=self.loss)
	
	@property
	def metrics(self):
		return [
			self.vgg_loss_tracker,
			self.ssim_loss_tracker,
			self.disc_loss_tracker
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
			activation_fn=kr.activations.swish,
	):
		image_input = kr.layers.Input(
			shape=(img_size, img_size, img_channels), name="image_input"
		)
		time_input = kr.Input(shape=(), dtype=tf.int64, name="time_input")
		
		mask_input = kr.Input(shape=self.mask_shape, dtype=tf.int64, name="mask_input")

		## Encoder
		x = kr.layers.Conv2D(
			self.first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=modules.kernel_init(1.0),
		)(image_input)
		
		temb = modules.TimeEmbedding(dim=self.first_conv_channels * 4)(time_input)
		temb = modules.TimeMLP(units=self.first_conv_channels * 4, activation_fn=activation_fn)(temb)
		
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
		
		## NA
		# MiddleBlock
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		x = modules.AttentionBlock(widths[-1], groups=norm_groups)(x)
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		
        # Obtain latent vector for decoder
		x = kr.layers.Flatten()(x)
		mean = kr.layers.Dense(self.latent_dim)(x)
		variance = kr.layers.Dense(self.latent_dim)(x)
	
		latent_vector = self.sampler([mean, variance])
		
		x = kr.layers.Dense(self.latent_dim * 32 * 32)(latent_vector)
		x = kr.layers.Reshape((32, 32, self.latent_dim))(x)

		## Decoder
		# UpBlock
		# SPADE filters: 512, 256, 256, 128, 64, 32
		for i in reversed(range(len(widths))):
			for _ in range(num_res_blocks + 1):
				x = kr.layers.Concatenate(axis=-1)([x, skips.pop()])
				x = modules.ResidualBlockLayerSpade(
					self.flags, widths[i], groups=norm_groups, activation_fn=activation_fn
				)([x, temb, mask_input])
				if has_attention[i]:
					x = modules.AttentionBlock(widths[i], groups=norm_groups)(x)
			
			if i != 0:
				x = modules.UpSample(widths[i], interpolation=interpolation)(x)
		
		# End block
		x = kr.layers.GroupNormalization(groups=norm_groups)(x)
		x = activation_fn(x)
		x = kr.layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=modules.kernel_init(0.0))(x)
		x = kr.layers.Activation('tanh')(x)
		return kr.Model(inputs=[image_input, time_input, mask_input], outputs=x, name="unetvit")
	
	def compile(self, **kwargs):
		super().compile(**kwargs)
	
	def call(self, inputs):
		image_input, time_input, mask = inputs
		return self.network([image_input, time_input, mask])
	
	def save_model(self, flags):
		self.network.save(flags.model_path + flags.exp_name)
	
    
	def train_discriminator(self, time_input, ct, mri, mask):
		self.discriminator.trainable = True
		
		fake_images = self.network([ct, time_input, mask], training=False)

		
		with tf.GradientTape() as gradient_tape:
			pred_fake = self.discriminator([ct, fake_images])[-1]  # check
			pred_real = self.discriminator([ct, mri])[-1]  # check
			loss_fake = self.discriminator_loss(False, pred_fake)
			loss_real = self.discriminator_loss(True, pred_real)
			total_loss = 0.5 * (loss_fake + loss_real)
		
		self.discriminator.trainable = True
		gradients = gradient_tape.gradient(
			total_loss, self.discriminator.trainable_variables
		)
		
		self.discriminator_optimizer.apply_gradients(
			zip(gradients, self.discriminator.trainable_variables)
		)
		return total_loss
	
	



	def train_step(self, data):
		
		source, target, mask = data
		batch_size = tf.shape(source)[0]
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		


		discriminator_loss = self.train_discriminator(t, source, target, mask)
		i_discriminator_loss = 0.5* discriminator_loss

		self.discriminator.trainable = False
		with tf.GradientTape() as tape:
			pred_ = self.network([source, t, mask], training=True)
			vgg_loss = self.vgg_loss(target, pred_)
			ssim_loss = loss.SSIMLoss(target, pred_)
			total_loss = vgg_loss + ssim_loss + i_discriminator_loss
		
		gradients = tape.gradient(total_loss, self.network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

		# Report progress.
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)
		self.disc_loss_tracker.update_state(discriminator_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
		#return {"vgg_loss": vgg_loss, "ssim_loss": ssim_loss}
	
	def test_step(self, data):
		source, target, mask = data
		batch_size = tf.shape(source)[0]
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		pred_ = self.network([source, t, mask])
		vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(target, pred_)
		ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(target, pred_)
		#total_loss = vgg_loss + ssim_loss

		# Report progress.
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
		
		#return {"vgg_loss": vgg_loss, "ssim_loss": ssim_loss}
	
	def generate_images(self, source, mask, num_images=8):
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
		)
		
		return self.network([source[:num_images], t, mask])
	
	# def plot_images(
	# 		self, source, target, logs=None, num_rows=2, num_cols=4, figsize=(12, 5)
	# ):
		
	# 	generated_samples = self.generate_images(source, num_images=num_rows * num_cols)
		
	# 	_, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
	# 	for i, image in enumerate(generated_samples.squeeze()):
	# 		if num_rows == 1:
	# 			ax[i].imshow(image, cmap='gray')
	# 			ax[i].axis("off")
	# 		else:
	# 			ax[i // num_cols, i % num_cols].imshow(image)
	# 			ax[i // num_cols, i % num_cols].axis("off")
		
	# 	plt.tight_layout()
	# 	plt.show()
        
    
    
	def model_evaluate(self, test_dataset, epoch=0):


		results = []

		#test_data = next(iter(test_dataset))
        
		# num_batches = len(test_data[0]//self.batch_size)

		# for i in range(0, num_batches, self.batch_size):
		# 	ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]


		for ct, mri in test_dataset:
			
			fake_mri = self.generate_images(ct, num_images=self.batch_size)

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

