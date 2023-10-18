# PCxGAN
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from . import modules_no_mask as modules
import loss
import evaluate
from datetime import datetime
import os
import pandas as pd
from  matplotlib import pyplot as plt
from datetime import datetime


class PCxGAN(kr.Model):
	def __init__(
			self,
			flags,
			**kwargs
	):
		super().__init__(**kwargs)
		self.flags = flags
		self.experiment_name = flags.name
		self.hist_path = flags.hist_path
		self.samples_dir = flags.sample_dir
		self.models_dir = flags.checkpoints_dir
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.image_size = flags.crop_size
		self.latent_dim = flags.latent_dim
		self.batch_size = flags.batch_size

		self.vgg_feature_loss_coeff = 1 #flags.vgg_feature_loss_coeff
		self.kl_divergence_loss_coeff = 50*flags.kl_divergence_loss_coeff
		self.ssim_loss_coeff = flags.ssim_loss_coeff
		
		self.discriminator = modules.Discriminator(self.flags)
		self.decoder = modules.Decoder(self.flags)
		self.encoder = modules.Encoder(self.flags)
		self.sampler = modules.GaussianSampler(self.batch_size, self.latent_dim)
		self.patch_size, self.combined_model = self.build_combined_model()
		
		self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
		self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
		self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
		self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")
		
		self.en_optimizer = kr.optimizers.Adam()
		self.generator_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1,
																									beta_2=self.flags.gen_beta_2)
		self.discriminator_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1,
																											beta_2=self.flags.disc_beta_2)
		self.discriminator_loss = loss.DiscriminatorLoss()
		self.vgg_loss = loss.VGGFeatureMatchingLoss()
	
	@property
	def metrics(self):
		return [
			self.disc_loss_tracker,
			self.vgg_loss_tracker,
			self.kl_loss_tracker,
			self.ssim_loss_tracker
		]
	
	def build_combined_model(self):
		
		self.discriminator.trainable = False
		image_input = kr.Input(shape=self.image_shape, name="image")
		latent_input = kr.Input(shape=self.latent_dim, name="latent")
		generated_image = self.decoder([latent_input, image_input])
		discriminator_output = self.discriminator([image_input, generated_image])
		
		patch_size = discriminator_output[-1].shape[1]
		combined_model = kr.Model(
			[latent_input, image_input],
			[discriminator_output, generated_image],
		)
		return patch_size, combined_model
	
	def compile(self, **kwargs):
		
		super().compile(**kwargs)
	
	def train_discriminator(self, latent_vector, segmentation_map, real_image):
		
		fake_images = self.decoder([latent_vector, segmentation_map])
		
		with tf.GradientTape() as gradient_tape:
			pred_fake = self.discriminator([segmentation_map, fake_images])[-1]  # check
			pred_real = self.discriminator([segmentation_map, real_image])[-1]  # check
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
	
	
	def train_generator(self, latent_vector, segmentation_map, image):
		
		self.discriminator.trainable = False


		with tf.GradientTape(persistent=True) as en_tape:
			
			mean, variance = self.encoder(image)

			
			# Compute generator losses.
			kl_loss = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean, variance)

		
		en_trainable_variables = (
				self.encoder.trainable_variables
		)
		
		en_gradients = en_tape.gradient(kl_loss, en_trainable_variables)

		self.en_optimizer.apply_gradients(
			zip(en_gradients, en_trainable_variables)
		)
		



		with tf.GradientTape(persistent=True) as tape:


			real_d_output = self.discriminator([segmentation_map, image])
			fake_d_output, fake_image = self.combined_model(
				[latent_vector, segmentation_map]
			)
			pred = fake_d_output[-1]

			
			# Compute generator losses.
			vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image, fake_image)
			ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(image, fake_image)
			total_loss = vgg_loss + ssim_loss

		
		all_trainable_variables = (
				self.combined_model.trainable_variables
		)
		
		gradients = tape.gradient(total_loss, all_trainable_variables)

		self.generator_optimizer.apply_gradients(
			zip(gradients, all_trainable_variables)
		)
		

		return kl_loss, vgg_loss, ssim_loss
	
	
	def train_step(self, data):
		ct, mri = data

		# Obtain the learned moments of the real image distribution.
		mean, variance = self.encoder(mri)
		
		# Sample a latent from the distribution defined by the learned moments.
		latent_vector = self.sampler([mean, variance])

		discriminator_loss = self.train_discriminator(
			latent_vector, ct, mri
		)
		(kl_loss, vgg_loss, ssim_loss) = self.train_generator(
			latent_vector, ct, mri
		)
		
		# Report progress.
		self.disc_loss_tracker.update_state(discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
	def test_step(self, data):
		ct, mri = data
		mean, variance = self.encoder(mri)
		latent_vector = self.sampler([mean, variance])
		fake_images = self.decoder([latent_vector, ct])
		
		# Calculate the losses.
		pred_fake = self.discriminator([ct, fake_images])[-1]
		pred_real = self.discriminator([ct, mri])[-1]
		loss_fake = self.discriminator_loss(False, pred_fake)
		loss_real = self.discriminator_loss(True, pred_real)
		total_discriminator_loss = 0.5 * (loss_fake + loss_real)
		
		real_d_output = self.discriminator([ct, mri])
		fake_d_output, fake_image = self.combined_model(
			[latent_vector, ct]
		)
		pred = fake_d_output[-1]
		
		kl_loss = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean, variance)
		vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(mri, fake_image)
		ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(mri, fake_image)
		#total_generator_loss = kl_loss + vgg_loss + ssim_loss
		
		# Report progress.
		self.disc_loss_tracker.update_state(total_discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
	def call(self, inputs):
		latent_vectors, ct = inputs
		return self.decoder([latent_vectors, ct])
	
	def model_evaluate(self, data, epoch=0):
		results = []
		
		for ct, mri in data:

			# Sample latent from a normal distribution.
			latent_vector = tf.random.normal(
				shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=2.0
			)
			fake_image = self.decoder([latent_vector, ct])
			
			# normalize to values between 0 and 1
			mri = (mri + 1.0) / 2.0
			fake_image = (fake_image + 1.0) / 2.0
			
			fid = evaluate.calculate_fid(mri, fake_image,
																	 input_shape=(
																		 self.flags.crop_size,
																		 self.flags.crop_size, 3))
			mse, mae, cs, psnr, ssim = evaluate.get_metrics(mri, fake_image)
			
			results.append([fid, mse, mae, cs, psnr, ssim])
			print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))
		

		results = np.array(results, dtype=object).mean(axis=0)
		
		filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
		results_dir = os.path.join(self.flags.result_logs, self.flags.name)
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		log_file = os.path.join(results_dir, filename)
		np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")

	
	
	def save_model(self):
		self.encoder.save(self.flags.model_path + self.experiment_name + '_e')
		self.decoder.save(self.flags.model_path + self.experiment_name + '_d')
		self.discriminator.save(self.flags.model_path + self.experiment_name + '_disc')


	def plot_losses(self, hist):
		
		exp_path = self.hist_path + self.experiment_name
		
		if not os.path.exists(exp_path):
			os.makedirs(exp_path)
			
		# save history to csv   
		hist_df = pd.DataFrame(hist) 
		hist_df.to_csv(exp_path + '/hist.csv')
		
		losses = ['disc', 'kl', 'vgg', 'ssim']
		
		# plot losses
		for loss in losses:
			plt.figure()
			plt.plot(hist[loss + '_loss'])
			plt.plot(hist['val_' + loss + '_loss'])
			plt.legend([loss + '_loss','val_' + loss + '_loss'],loc='upper right')
			plt.savefig(exp_path + '/pcxgan_' + loss + '.png')



