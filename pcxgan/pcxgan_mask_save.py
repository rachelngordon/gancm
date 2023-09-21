# PCxGAN
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from . import modules_mask_save as modules
import loss_save
import evaluate
from datetime import datetime
import os
import pandas as pd
from  matplotlib import pyplot as plt
from datetime import datetime

@kr.saving.register_keras_serializable(package="MyLayers")
class PCxGAN_mask(kr.Model):
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
		self.mask_shape = (flags.crop_size, flags.crop_size, 2)

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
		self.discriminator_loss = loss_save.DiscriminatorLoss()
		#self.feature_matching_loss = loss.FeatureMatchingLoss()
		self.vgg_loss = loss_save.VGGFeatureMatchingLoss()
		#self.mae_loss = loss.MAE()
	
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
		mask_input = kr.Input(shape=self.mask_shape, name="mask")
		image_input = kr.Input(shape=self.image_shape, name="image")
		latent_input = kr.Input(shape=self.latent_dim, name="latent")
		generated_image = self.decoder([latent_input, mask_input, image_input])
		discriminator_output = self.discriminator([image_input, generated_image])
		
		patch_size = discriminator_output[-1].shape[1]
		combined_model = kr.Model(
			[latent_input, mask_input, image_input],
			[discriminator_output, generated_image],
		)
		return patch_size, combined_model
	
	def compile(self, **kwargs):
		
		super().compile(**kwargs)
	
	def train_discriminator(self, latent_vector, segmentation_map, real_image, labels):
		
		fake_images = self.decoder([latent_vector, labels, segmentation_map])
		
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
	
	
	def train_generator(self, latent_vector, segmentation_map, labels, image):
		
		self.discriminator.trainable = False


		with tf.GradientTape(persistent=True) as en_tape:
			
			mean, variance = self.encoder(image)

			
			# Compute generator losses.
			kl_loss = self.kl_divergence_loss_coeff * loss_save.kl_divergence_loss(mean, variance)

		
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
				[latent_vector, labels, segmentation_map]
			)
			pred = fake_d_output[-1]

			
			# Compute generator losses.
			vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image, fake_image)
			ssim_loss = self.ssim_loss_coeff * loss_save.SSIMLoss(image, fake_image)
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
		ct, mri, labels = data

		# Obtain the learned moments of the real image distribution.
		mean, variance = self.encoder(mri)
		
		# Sample a latent from the distribution defined by the learned moments.
		latent_vector = self.sampler([mean, variance])

		discriminator_loss = self.train_discriminator(
			latent_vector, ct, mri, labels
		)
		(kl_loss, vgg_loss, ssim_loss) = self.train_generator(
			latent_vector, ct, labels, mri
		)
		
		# Report progress.
		self.disc_loss_tracker.update_state(discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
	def test_step(self, data):
		ct, mri, labels = data
		mean, variance = self.encoder(mri)
		latent_vector = self.sampler([mean, variance])
		fake_images = self.decoder([latent_vector, labels, ct])
		
		# Calculate the losses.
		pred_fake = self.discriminator([ct, fake_images])[-1]
		pred_real = self.discriminator([ct, mri])[-1]
		loss_fake = self.discriminator_loss(False, pred_fake)
		loss_real = self.discriminator_loss(True, pred_real)
		total_discriminator_loss = 0.5 * (loss_fake + loss_real)
		
		real_d_output = self.discriminator([ct, mri])
		fake_d_output, fake_image = self.combined_model(
			[latent_vector, labels, ct]
		)
		pred = fake_d_output[-1]
		
		kl_loss = self.kl_divergence_loss_coeff * loss_save.kl_divergence_loss(mean, variance)
		vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(mri, fake_image)
		ssim_loss = self.ssim_loss_coeff * loss_save.SSIMLoss(mri, fake_image)
		#total_generator_loss = kl_loss + vgg_loss + ssim_loss
		
		# Report progress.
		self.disc_loss_tracker.update_state(total_discriminator_loss)
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
	def call(self, inputs):
		latent_vectors, labels, ct = inputs
		return self.decoder([latent_vectors, labels, ct])
	
	

	def get_config(self):
		base_config = super().get_config()
		config = {
            "flags": kr.saving.serialize_keras_object(self.flags),
			"image_shape": kr.saving.serialize_keras_object(self.image_shape),
			"mask_shape": kr.saving.serialize_keras_object(self.mask_shape),

			"discriminator": kr.saving.serialize_keras_object(self.discriminator),
			"decoder": kr.saving.serialize_keras_object(self.decoder),
			"encoder": kr.saving.serialize_keras_object(self.encoder),
			"sampler": kr.saving.serialize_keras_object(self.sampler),
			"combined_model": kr.saving.serialize_keras_object(self.combined_model),

			"disc_loss_tracker": kr.saving.serialize_keras_object(self.disc_loss_tracker),
			"vgg_loss_tracker": kr.saving.serialize_keras_object(self.vgg_loss_tracker),
			"kl_loss_tracker": kr.saving.serialize_keras_object(self.kl_loss_tracker),
			"ssim_loss_tracker": kr.saving.serialize_keras_object(self.ssim_loss_tracker),

			"en_optimizer": kr.saving.serialize_keras_object(self.en_optimizer),
			"generator_optimizer": kr.saving.serialize_keras_object(self.generator_optimizer),
			"discriminator_optimizer": kr.saving.serialize_keras_object(self.discriminator_optimizer),

			"discriminator_loss": kr.saving.serialize_keras_object(self.discriminator_loss),
			"vgg_loss": kr.saving.serialize_keras_object(self.vgg_loss),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		flags_config = config.pop("flags")
		flags = kr.saving.deserialize_keras_object(flags_config)
		image_shape_config = config.pop("image_shape")
		image_shape = kr.saving.deserialize_keras_object(image_shape_config)
		mask_shape_config = config.pop("mask_shape")
		mask_shape = kr.saving.deserialize_keras_object(mask_shape_config)

		discriminator_config = config.pop("discriminator")
		discriminator = kr.saving.deserialize_keras_object(discriminator_config)
		decoder_config = config.pop("decoder")
		decoder = kr.saving.deserialize_keras_object(decoder_config)
		encoder_config = config.pop("encoder")
		encoder = kr.saving.deserialize_keras_object(encoder_config)
		sampler_config = config.pop("sampler")
		sampler = kr.saving.deserialize_keras_object(sampler_config)
		combined_model_config = config.pop("combined_model")
		combined_model = kr.saving.deserialize_keras_object(combined_model_config)

		disc_loss_tracker_config = config.pop("disc_loss_tracker")
		disc_loss_tracker = kr.saving.deserialize_keras_object(disc_loss_tracker_config)
		vgg_loss_tracker_config = config.pop("vgg_loss_tracker")
		vgg_loss_tracker = kr.saving.deserialize_keras_object(vgg_loss_tracker_config)
		kl_loss_tracker_config = config.pop("kl_loss_tracker")
		kl_loss_tracker = kr.saving.deserialize_keras_object(kl_loss_tracker_config)
		ssim_loss_tracker_config = config.pop("ssim_loss_tracker")
		ssim_loss_tracker = kr.saving.deserialize_keras_object(ssim_loss_tracker_config)

		en_optimizer_config = config.pop("en_optimizer")
		en_optimizer = kr.saving.deserialize_keras_object(en_optimizer_config)
		generator_optimizer_config = config.pop("generator_optimizer")
		generator_optimizer = kr.saving.deserialize_keras_object(generator_optimizer_config)
		discriminator_optimizer_config = config.pop("discriminator_optimizer")
		discriminator_optimizer = kr.saving.deserialize_keras_object(discriminator_optimizer_config)

		discriminator_loss_config = config.pop("discriminator_loss")
		discriminator_loss = kr.saving.deserialize_keras_object(discriminator_loss_config)
		vgg_loss_config = config.pop("vgg_loss")
		vgg_loss = kr.saving.deserialize_keras_object(vgg_loss_config)
		
		return cls(flags, image_shape, mask_shape, discriminator, decoder, encoder, sampler, combined_model, 
			 disc_loss_tracker, vgg_loss_tracker, kl_loss_tracker, ssim_loss_tracker, en_optimizer, 
			 generator_optimizer, discriminator_optimizer, discriminator_loss, vgg_loss, **config)
	
	
	def model_evaluate(self, data, epoch=0):
		results = []
		
		for ct, mri, label in data:

			# Sample latent from a normal distribution.
			latent_vector = tf.random.normal(
				shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=2.0
			)
			fake_image = self.decoder([latent_vector, label, ct])

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



