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


class GAN_UVIT(keras.Model):
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
		self.img_channels = self.flags.img_channels
		self.widths = [self.flags.first_conv_channels * mult for mult in self.flags.channel_multiplier]
		
		self.discriminator = modules.Discriminator(self.flags)
		self.generator = modules.uvit_generator(self.flags)

		self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
		self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")
		self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")

		self.vgg_feature_loss_coeff = 1 #flags.vgg_feature_loss_coeff
		self.ssim_loss_coeff = flags.ssim_loss_coeff

		self.vgg_loss = loss.VGGFeatureMatchingLoss()
		self.discriminator_loss = loss.DiscriminatorLoss()

		self.generator_optimizer = keras.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1,
																									beta_2=self.flags.gen_beta_2)
		self.discriminator_optimizer = keras.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1,
																											beta_2=self.flags.disc_beta_2)
		
		self.patch_size, self.network = self.build_combined_model()

		
	@property
	def metrics(self):
		return [
			self.vgg_loss_tracker,
			self.ssim_loss_tracker
		]
	

	def build_combined_model(self):
		
		self.discriminator.trainable = False
		image_input = layers.Input(
			shape=(self.img_size, self.img_size, self.img_channels), name="image_input"
		)
		time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
		generated_image = self.generator([image_input, time_input])
		discriminator_output = self.discriminator([image_input, generated_image])
		
		patch_size = discriminator_output[-1].shape[1]
		combined_model = keras.Model(
			[image_input, time_input],
			[discriminator_output, generated_image],
		)
		return patch_size, combined_model
	
	def compile(self, **kwargs):
		super().compile(**kwargs)


	def train_discriminator(self, time_input, ct, mri):
		
		fake_images = self.generator([ct, time_input])
		
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
	
	
	def train_generator(self, time_input, ct, mri):
		
		self.discriminator.trainable = False


		with tf.GradientTape(persistent=True) as tape:


			real_d_output = self.discriminator([ct, mri])
			fake_d_output, fake_image = self.network(
				[ct, time_input]
			)
			pred = fake_d_output[-1]

			
			# Compute generator losses.
			vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(ct, fake_image)
			ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(ct, fake_image)
			total_loss = vgg_loss + ssim_loss

		
		all_trainable_variables = (
				self.network.trainable_variables
		)
		
		gradients = tape.gradient(total_loss, all_trainable_variables)

		self.generator_optimizer.apply_gradients(
			zip(gradients, all_trainable_variables)
		)
		

		return vgg_loss, ssim_loss
	
	
	def train_step(self, data):
		ct, mri = data
		batch_size = tf.shape(ct)[0]
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)

		discriminator_loss = self.train_discriminator(
			t, ct, mri
		)
		(vgg_loss, ssim_loss) = self.train_generator(
			t, ct, mri
		)
		
		# Report progress.
		self.disc_loss_tracker.update_state(discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		
		return results
	

	def test_step(self, data):
		ct, mri = data
		batch_size = tf.shape(ct)[0]
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		fake_images = self.generator([ct, t])


		# Calculate the losses.
		pred_fake = self.discriminator([ct, fake_images])[-1]
		pred_real = self.discriminator([ct, mri])[-1]
		loss_fake = self.discriminator_loss(False, pred_fake)
		loss_real = self.discriminator_loss(True, pred_real)
		total_discriminator_loss = 0.5 * (loss_fake + loss_real)
		
		real_d_output = self.discriminator([ct, mri])
		fake_d_output, fake_image  = self.network([ct, t])
		
		pred = fake_d_output[-1]
		

		vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(mri, fake_image)
		ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(mri, fake_image)
		#total_loss = vgg_loss + ssim_loss

		# Report progress.
		self.disc_loss_tracker.update_state(total_discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)
		
		results = {m.name: m.result() for m in self.metrics}
		return results
	
	
	def call(self, inputs):
		image_input, time_input = inputs
		return self.network([image_input, time_input])
	
	def save_model(self, path):
		self.network.save(path)
		
	
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

