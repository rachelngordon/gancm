# PCxGAN
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from . import modules
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
		self.mask_shape = (flags.crop_size, flags.crop_size, 2)

		self.vgg_feature_loss_coeff = 1 #flags.vgg_feature_loss_coeff
		self.kl_divergence_loss_coeff = 50*flags.kl_divergence_loss_coeff
		self.ssim_loss_coeff = flags.ssim_loss_coeff

        # define discriminators
		self.dp_ct = modules.Discriminator(self.flags)
		self.dp_mri = modules.Discriminator(self.flags)
		self.dc_ct = modules.Discriminator(self.flags)
		self.dc_mri = modules.Discriminator(self.flags)
		
        # define generator encoders and decoders
		self.de_ct = modules.Decoder(self.flags)
		self.en_ct = modules.Encoder(self.flags)
		self.de_mri = modules.Decoder(self.flags)
		self.en_mri = modules.Encoder(self.flags)
		
        # define generator and discriminator optimizers
		self.mri_g_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
		self.ct_g_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
		
		self.ct_en_optimizer = kr.optimizers.Adam()
		self.mri_en_optimizer = kr.optimizers.Adam()
		
		self.ct_dp_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.mri_dp_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.ct_dc_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.mri_dc_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
        
		self.sampler = modules.GaussianSampler(self.batch_size, self.latent_dim)
		self.patch_size, self.combined_model = self.build_combined_model()
		
		self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
		self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
		self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
		self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")
		
		self.discriminator_loss = loss.DiscriminatorLoss()
		self.feature_matching_loss = loss.FeatureMatchingLoss()
		self.vgg_loss = loss.VGGFeatureMatchingLoss()
		self.mae_loss = loss.MAE()
	
	@property
	def metrics(self):
		return [
			self.disc_loss_tracker,
			self.vgg_loss_tracker,
			self.kl_loss_tracker,
			self.ssim_loss_tracker
		]
	
	def build_combined_model(self):
		
		self.dp_mri.trainable = False
		self.dp_ct.trainable = False
		self.dc_mri.trainable = False
		self.dc_ct.trainable = False
        
		ct_input = kr.Input(shape=self.image_shape, name="ct")
		mri_input = kr.Input(shape=self.image_shape, name="mri")
		mask_input = kr.Input(shape=self.mask_shape, name="mask")
		latent_input = kr.Input(shape=self.latent_dim, name="latent")

		generated_mri = self.de_mri([latent_input, mask_input, ct_input])
		generated_ct = self.de_ct([latent_input, mask_input, mri_input])
		id_mri = self.de_mri(mri_input)
		id_ct = self.de_ct(ct_input)
		cycled_mri = self.de_mri(generated_ct)
		cycled_ct = self.de_ct(generated_mri)

           
		combined_model = kr.Model(
            [ct_input, mri_input, mask_input, latent_input],
            [generated_mri, generated_ct,
             id_mri, id_ct,
             cycled_mri, cycled_ct
             ])
		
		
		dp_mri_output = self.dp_mri([ct_input, generated_mri])
		dp_ct_output = self.dp_ct([mri_input, generated_ct])
		dc_mri_output = self.dc_mri([mri_input, generated_mri])
		dc_ct_output = self.dc_ct([ct_input, generated_ct])
		
		patch_size = dp_mri_output[-1].shape[1]
		
		combined_model = kr.Model(
			[ct_input, mri_input, mask_input, latent_input],
			[discriminator_output, generated_image],
		)
		return patch_size, combined_model
	
	def compile(self, **kwargs):
		
		super().compile(**kwargs)
	
	def train_discriminators(self, latent_vector, real_mri, mri_mask, real_ct, ct_mask):
		
		fake_mri = self.de_mri([latent_vector, mri_mask, real_ct])
		fake_ct = self.de_mri([latent_vector, ct_mask, real_ct])
		
		with tf.GradientTape() as dp_mri_tape:
			pred_fake_mri_pair = self.dp_mri([real_ct, fake_mri])[-1] 
			pred_real_mri_pair = self.dp_mri([real_ct, real_mri])[-1]
			loss_fake_mri_pair = self.discriminator_loss(False, pred_fake_mri_pair)
			loss_real_mri_pair = self.discriminator_loss(True, pred_real_mri_pair)
			total_loss_mri_pair = 0.5 * (loss_fake_mri_pair + loss_real_mri_pair)
			
		with tf.GradientTape() as dp_ct_tape:
			pred_fake_ct_pair = self.dp_ct([real_mri, fake_ct])[-1] 
			pred_real_ct_pair = self.dp_ct([real_mri, real_ct])[-1]
			loss_fake_ct_pair = self.discriminator_loss(False, pred_fake_ct_pair)
			loss_real_ct_pair = self.discriminator_loss(True, pred_real_ct_pair)
			total_loss_ct_pair = 0.5 * (loss_fake_ct_pair + loss_real_ct_pair)
			
		with tf.GradientTape() as dc_mri_tape:
			pred_fake_mri_cycle = self.dc_mri([real_mri, fake_mri])[-1] 
			pred_real_mri_cycle = self.dp_mri([real_mri, real_mri])[-1]
			loss_fake_mri_cycle = self.discriminator_loss(False, pred_fake_mri_cycle)
			loss_real_mri_cycle = self.discriminator_loss(True, pred_real_mri_cycle)
			total_loss_mri_cycle = 0.5 * (loss_fake_mri_cycle + loss_real_mri_cycle)
			
		with tf.GradientTape() as dc_ct_tape:
			pred_fake_ct_cycle = self.dp_ct([real_ct, fake_ct])[-1] 
			pred_real_ct_cycle = self.dp_ct([real_ct, real_ct])[-1]
			loss_fake_ct_cycle = self.discriminator_loss(False, pred_fake_ct_cycle)
			loss_real_ct_cycle = self.discriminator_loss(True, pred_real_ct_cycle)
			total_loss_ct_cycle = 0.5 * (loss_fake_ct_cycle + loss_real_ct_cycle)
		

		gradients_mri_pair = dp_mri_tape.gradient(
            total_loss_mri_pair, self.dp_mri.trainable_variables) 

		self.mri_dp_optimizer.apply_gradients(
            zip(gradients_mri_pair, self.dp_mri.trainable_variables)
        )

		gradients_ct_pair = dp_ct_tape.gradient(
            total_loss_ct_pair, self.dp_ct.trainable_variables) 

		self.ct_dp_optimizer.apply_gradients(
            zip(gradients_ct_pair, self.dp_ct.trainable_variables)
        )
		
		gradients_mri_cycle = dc_mri_tape.gradient(
            total_loss_mri_cycle, self.dc_mri.trainable_variables) 

		self.mri_dc_optimizer.apply_gradients(
            zip(gradients_mri_cycle, self.dc_mri.trainable_variables)
        )

		gradients_ct_cycle = dc_ct_tape.gradient(
            total_loss_ct_cycle, self.dc_ct.trainable_variables) 

		self.ct_dc_optimizer.apply_gradients(
            zip(gradients_ct_cycle, self.dc_ct.trainable_variables)
        )
	
		return total_loss_mri_pair, total_loss_mri_cycle, total_loss_ct_pair, total_loss_ct_cycle
	
	
	
	def train_generators(self, latent_vector, ct, labels, mri):
		
		self.dp_mri.trainable = False
		self.dp_ct.trainable = False
		self.dc_mri.trainable = False
		self.dc_ct.trainable = False


		with tf.GradientTape(persistent=True) as en_mri_tape:
			
			mean_mri, variance_mri = self.en_mri(mri)

			
			# Compute generator losses.
			kl_loss_mri = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_mri, variance_mri)
			
		with tf.GradientTape(persistent=True) as en_ct_tape:
			
			mean_ct, variance_ct = self.en_ct(ct)

			
			# Compute generator losses.
			kl_loss_ct = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_ct, variance_ct)

		
		en_mri_trainable_variables = (
				self.en_mri.trainable_variables
		)
		
		en_mri_gradients = en_mri_tape.gradient(kl_loss_mri, en_mri_trainable_variables)

		self.mri_en_optimizer.apply_gradients(
			zip(en_mri_gradients, en_mri_trainable_variables)
		)
		
		en_ct_trainable_variables = (
				self.en_ct.trainable_variables
		)
		
		en_ct_gradients = en_ct_tape.gradient(kl_loss_ct, en_ct_trainable_variables)

		self.ct_en_optimizer.apply_gradients(
			zip(en_ct_gradients, en_ct_trainable_variables)
		)
		



		with tf.GradientTape(persistent=True) as mri_tape:


			real_d_mri_output = self.dp_mri([ct, mri])
			#fake_d_mri_output, fake_image = self.combined_model(
				#[latent_vector, labels, segmentation_map])
			#pred = fake_d_output[-1]

			
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
		latent_vectors, labels, ct = inputs
		return self.decoder([latent_vectors, labels, ct])
	
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



