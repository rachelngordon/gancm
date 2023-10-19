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
import cv2


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
		self.cycle_loss_coeff = self.flags.cycle_loss_coeff
		self.identity_loss_coeff = self.flags.identity_loss_coeff

        # Define discriminators.
		self.dp_ct = modules.Discriminator(self.flags)
		self.dp_mri = modules.Discriminator(self.flags)
		self.dc_ct = modules.Discriminator(self.flags)
		self.dc_mri = modules.Discriminator(self.flags)
		
        # Define generator encoders and decoders.
		self.de_ct = modules.Decoder(self.flags)
		self.en_ct = modules.Encoder(self.flags)
		self.de_mri = modules.Decoder(self.flags)
		self.en_mri = modules.Encoder(self.flags)
		
        # Define encoder and decoder optimizers.
		self.mri_de_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
		self.ct_de_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
		self.ct_en_optimizer = kr.optimizers.Adam()
		self.mri_en_optimizer = kr.optimizers.Adam()
		
		# Define discriminator optimizers.
		self.ct_dp_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.mri_dp_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.ct_dc_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
		self.mri_dc_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.disc_beta_1)
        
		self.sampler = modules.GaussianSampler(self.batch_size, self.latent_dim)
		self.mask_generator = modules.MaskGenerationLayer()
		self.patch_size, self.combined_model = self.build_combined_model()
		
		# Define MRI loss trackers.
		self.disc_pair_loss_mri_tracker = tf.keras.metrics.Mean(name="disc_pair_loss_mri")
		self.disc_cycle_loss_mri_tracker = tf.keras.metrics.Mean(name="disc_cycle_loss_mri")
		self.kl_loss_mri_tracker = tf.keras.metrics.Mean(name="kl_loss_mri")
		self.ssim_loss_mri_tracker = tf.keras.metrics.Mean(name="ssim_loss_mri")
		self.vgg_loss_mri_tracker = tf.keras.metrics.Mean(name="vgg_loss_mri")
		self.cycle_loss_mri_tracker = tf.keras.metrics.Mean(name="cycle_loss_mri")
		self.identity_loss_mri_tracker = tf.keras.metrics.Mean(name="identity_loss_mri")

		# Define CT loss trackers.
		self.disc_pair_loss_ct_tracker = tf.keras.metrics.Mean(name="disc_pair_loss_ct")
		self.disc_cycle_loss_ct_tracker = tf.keras.metrics.Mean(name="disc_cycle_loss_ct")
		self.kl_loss_ct_tracker = tf.keras.metrics.Mean(name="kl_loss_ct")
		self.ssim_loss_ct_tracker = tf.keras.metrics.Mean(name="ssim_loss_ct")
		self.vgg_loss_ct_tracker = tf.keras.metrics.Mean(name="vgg_loss_ct")
		self.cycle_loss_ct_tracker = tf.keras.metrics.Mean(name="cycle_loss_ct")
		self.identity_loss_ct_tracker = tf.keras.metrics.Mean(name="identity_loss_ct")
		
		# Define losses.
		self.discriminator_loss = loss.DiscriminatorLoss()
		self.feature_matching_loss = loss.FeatureMatchingLoss()
		self.vgg_loss = loss.VGGFeatureMatchingLoss()
		self.mae_loss = loss.MAE()
	
	@property
	def metrics(self):
		return [
			self.disc_pair_loss_mri_tracker,
			self.disc_cycle_loss_mri_tracker,
			self.kl_loss_mri_tracker,
			self.ssim_loss_mri_tracker,
			self.vgg_loss_mri_tracker,
			self.cycle_loss_mri_tracker,
			self.identity_loss_mri_tracker,

			self.disc_pair_loss_ct_tracker,
			self.disc_cycle_loss_ct_tracker,
			self.kl_loss_ct_tracker,
			self.ssim_loss_ct_tracker,
			self.vgg_loss_ct_tracker,
			self.cycle_loss_ct_tracker,
			self.identity_loss_ct_tracker
		]
	
	
	def build_combined_model(self):
		
		self.dp_mri.trainable = False
		self.dp_ct.trainable = False
		self.dc_mri.trainable = False
		self.dc_ct.trainable = False
        
		# Define model inputs.
		mri_input = kr.Input(shape=self.image_shape, name="mri")
		mri_mask = kr.Input(shape=self.mask_shape, name="mri_mask")
		mri_latent = kr.Input(shape=self.latent_dim, name="mri_latent")
		ct_input = kr.Input(shape=self.image_shape, name="ct")
		ct_mask = kr.Input(shape=self.mask_shape, name="ct_mask")
		ct_latent = kr.Input(shape=self.latent_dim, name="ct_latent")

		# Generate MRI and CT.
		generated_mri = self.de_mri([mri_latent, ct_mask, ct_input])
		generated_ct = self.de_ct([ct_latent, mri_mask, mri_input])

		# Generate MRI and CT for identity cycle.
		id_mri = self.de_mri([mri_latent, mri_mask, mri_input])
		id_ct = self.de_ct([ct_latent, ct_mask, ct_input])

		# Get segmentation masks from generated images.
		gen_ct_mask = self.mask_generator(generated_ct)
		gen_mri_mask = self.mask_generator(generated_mri)

		# Generate MRI and CT for forward/backward cycle.
		cycled_mri = self.de_mri([mri_latent, gen_ct_mask, generated_ct])
		cycled_ct = self.de_ct([ct_latent, gen_mri_mask, generated_mri])


        # Define model inputs and outputs.
		combined_model = kr.Model(
            [mri_input, mri_mask, mri_latent, ct_input, ct_mask, ct_latent],
            [generated_mri, generated_ct,
             id_mri, id_ct,
             cycled_mri, cycled_ct
             ])
		

		# Obtain patch size from Pix2Pix discriminator.
		dp_mri_output = self.dp_mri([ct_input, generated_mri])
		patch_size = dp_mri_output[-1].shape[1]

		
		return patch_size, combined_model
	

	def compile(self, **kwargs):
		
		super().compile(**kwargs)
	

	def train_discriminators(self, mri_latent, real_mri, mri_mask, ct_latent, real_ct, ct_mask):

		#real_ct = tf.squeeze(real_ct, axis=0)
		#ct_mask = tf.squeeze(ct_mask, axis=0)
		#real_mri = tf.squeeze(real_mri, axis=0)
		#mri_mask = tf.squeeze(mri_mask, axis=0)

		fake_mri = self.de_mri([mri_latent, ct_mask, real_ct])
		fake_ct = self.de_mri([ct_latent, mri_mask, real_mri])
		
		# Train MRI discriminator pair.
		with tf.GradientTape() as dp_mri_tape:
			pred_fake_mri_pair = self.dp_mri([real_ct, fake_mri])[-1] 
			pred_real_mri_pair = self.dp_mri([real_ct, real_mri])[-1]
			loss_fake_mri_pair = self.discriminator_loss(False, pred_fake_mri_pair)
			loss_real_mri_pair = self.discriminator_loss(True, pred_real_mri_pair)
			total_loss_mri_pair = 0.5 * (loss_fake_mri_pair + loss_real_mri_pair)
		
		# Train CT discriminator pair.
		with tf.GradientTape() as dp_ct_tape:
			pred_fake_ct_pair = self.dp_ct([real_mri, fake_ct])[-1] 
			pred_real_ct_pair = self.dp_ct([real_mri, real_ct])[-1]
			loss_fake_ct_pair = self.discriminator_loss(False, pred_fake_ct_pair)
			loss_real_ct_pair = self.discriminator_loss(True, pred_real_ct_pair)
			total_loss_ct_pair = 0.5 * (loss_fake_ct_pair + loss_real_ct_pair)
		
		# Train MRI discriminator cycle.
		with tf.GradientTape() as dc_mri_tape:
			pred_fake_mri_cycle = self.dc_mri([real_mri, fake_mri])[-1] 
			pred_real_mri_cycle = self.dp_mri([real_mri, real_mri])[-1]
			loss_fake_mri_cycle = self.discriminator_loss(False, pred_fake_mri_cycle)
			loss_real_mri_cycle = self.discriminator_loss(True, pred_real_mri_cycle)
			total_loss_mri_cycle = 0.5 * (loss_fake_mri_cycle + loss_real_mri_cycle)
			
		# Train CT discriminator cycle.
		with tf.GradientTape() as dc_ct_tape:
			pred_fake_ct_cycle = self.dp_ct([real_ct, fake_ct])[-1] 
			pred_real_ct_cycle = self.dp_ct([real_ct, real_ct])[-1]
			loss_fake_ct_cycle = self.discriminator_loss(False, pred_fake_ct_cycle)
			loss_real_ct_cycle = self.discriminator_loss(True, pred_real_ct_cycle)
			total_loss_ct_cycle = 0.5 * (loss_fake_ct_cycle + loss_real_ct_cycle)
		
		# Calculate gradients.
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
	
	
	
	def train_generators(self, mri, mri_mask, mri_latent, ct, ct_mask, ct_latent):
		
		self.dp_mri.trainable = False
		self.dp_ct.trainable = False
		self.dc_mri.trainable = False
		self.dc_ct.trainable = False

		# Train MRI encoder.
		with tf.GradientTape(persistent=True) as en_mri_tape:
			mean_mri, variance_mri = self.en_mri(mri)
			kl_loss_mri = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_mri, variance_mri)
			
		# Train CT encoder.
		with tf.GradientTape(persistent=True) as en_ct_tape:
			mean_ct, variance_ct = self.en_ct(ct)
			kl_loss_ct = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_ct, variance_ct)

		
		# Calculate encoder gradients.
		en_mri_trainable_variables = (
				self.en_mri.trainable_variables)
		en_mri_gradients = en_mri_tape.gradient(kl_loss_mri, en_mri_trainable_variables)
		self.mri_en_optimizer.apply_gradients(
			zip(en_mri_gradients, en_mri_trainable_variables))
		
		en_ct_trainable_variables = (
				self.en_ct.trainable_variables)
		en_ct_gradients = en_ct_tape.gradient(kl_loss_ct, en_ct_trainable_variables)
		self.ct_en_optimizer.apply_gradients(
			zip(en_ct_gradients, en_ct_trainable_variables))
		

		#ct = tf.squeeze(ct, axis=0)
		#ct_mask = tf.squeeze(ct_mask, axis=0)
		#mri = tf.squeeze(mri, axis=0)
		#mri_mask = tf.squeeze(mri_mask, axis=0)


		# Train MRI decoder.
		with tf.GradientTape(persistent=True) as mri_tape:
			generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model([mri, mri_mask, 
																							mri_latent, ct, 
																							ct_mask, ct_latent])

			ssim_loss_mri = self.ssim_loss_coeff * loss.SSIMLoss(mri, generated_mri)
			vgg_loss_mri = self.vgg_feature_loss_coeff * self.vgg_loss(mri, generated_mri)
			cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri, cycled_mri)
			cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
			identity_loss_mri = self.identity_loss_coeff * loss.mae(mri, id_mri)
			total_loss_mri = ssim_loss_mri + vgg_loss_mri + cycle_loss_mri + cycle_loss_ct + identity_loss_mri


		# Train CT decoder.
		with tf.GradientTape(persistent=True) as ct_tape:
			generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model([mri, mri_mask, 
																							mri_latent, ct, 
																							ct_mask, ct_latent])

			ssim_loss_ct = self.ssim_loss_coeff * loss.SSIMLoss(ct, generated_ct)
			vgg_loss_ct = self.vgg_feature_loss_coeff * self.vgg_loss(ct, generated_ct)
			cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri, cycled_mri)
			cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
			identity_loss_ct = self.identity_loss_coeff * loss.mae(ct, id_ct)
			total_loss_ct = ssim_loss_ct + vgg_loss_ct + cycle_loss_mri + cycle_loss_ct + identity_loss_ct
	

		# Compute decoder gradients.
		mri_de_variables = self.de_mri.trainable_variables
		mri_de_gradients = mri_tape.gradient(total_loss_mri, mri_de_variables)
		self.mri_de_optimizer.apply_gradients(zip(mri_de_gradients , mri_de_variables))

		ct_de_variables = self.de_ct.trainable_variables
		ct_de_gradients = ct_tape.gradient(total_loss_ct, ct_de_variables)
		self.ct_de_optimizer.apply_gradients(zip(ct_de_gradients , ct_de_variables))
	

		losses__ = (kl_loss_mri, ssim_loss_mri, vgg_loss_mri, cycle_loss_mri, identity_loss_mri,
                    kl_loss_ct, ssim_loss_ct, vgg_loss_ct, cycle_loss_ct, identity_loss_ct)
		
		
		return losses__
	
	
	def train_step(self, data):

		ct, ct_mask, mri, mri_mask = data

		# Obtain the learned moments of the real image distribution.
		mean_mri, variance_mri = self.en_mri(mri)
		mean_ct, variance_ct = self.en_ct(ct)

		# Sample a latent from the distribution defined by the learned moments.
		latent_mri = self.sampler([mean_mri, variance_mri])
		latent_ct = self.sampler([mean_ct, variance_ct])

		total_loss_mri_pair, total_loss_mri_cycle, total_loss_ct_pair, total_loss_ct_cycle = self.train_discriminators(
			latent_mri, mri, mri_mask, latent_ct, ct, ct_mask)
		
		losses = self.train_generators(
			mri, mri_mask, latent_mri, ct, ct_mask, latent_ct
		)

		(kl_loss_mri, ssim_loss_mri, vgg_loss_mri, cycle_loss_mri, identity_loss_mri,
                    kl_loss_ct, ssim_loss_ct, vgg_loss_ct, cycle_loss_ct, identity_loss_ct) = losses


		# Report progress.
		self.disc_pair_loss_mri_tracker.update_state(total_loss_mri_pair)
		self.disc_cycle_loss_mri_tracker.update_state(total_loss_mri_cycle)
		self.kl_loss_mri_tracker.update_state(kl_loss_mri)
		self.ssim_loss_mri_tracker.update_state(ssim_loss_mri)
		self.vgg_loss_mri_tracker.update_state(vgg_loss_mri)
		self.cycle_loss_mri_tracker.update_state(cycle_loss_mri)
		self.identity_loss_mri_tracker.update_state(identity_loss_mri)

		self.disc_pair_loss_ct_tracker.update_state(total_loss_ct_pair)
		self.disc_cycle_loss_ct_tracker.update_state(total_loss_ct_cycle)
		self.kl_loss_ct_tracker.update_state(kl_loss_ct)
		self.ssim_loss_ct_tracker.update_state(ssim_loss_ct)
		self.vgg_loss_ct_tracker.update_state(vgg_loss_ct)
		self.cycle_loss_ct_tracker.update_state(cycle_loss_ct)
		self.identity_loss_ct_tracker.update_state(identity_loss_ct)


		results = {m.name: m.result() for m in self.metrics}
		return results
	
	
	def test_step(self, data):
		ct, ct_mask, mri, mri_mask = data

		# Obtain the learned moments of the real image distribution.
		mean_mri, variance_mri = self.en_mri(mri)
		mean_ct, variance_ct = self.en_ct(ct)

		# Sample a latent from the distribution defined by the learned moments.
		latent_mri = self.sampler([mean_mri, variance_mri])
		latent_ct = self.sampler([mean_ct, variance_ct])


		fake_mri = self.de_mri([latent_mri, ct_mask, ct])
		fake_ct = self.de_mri([latent_ct, mri_mask, mri])

		
		# Calculate MRI discriminator pair losses.
		pred_fake_mri_pair = self.dp_mri([ct, fake_mri])[-1] 
		pred_real_mri_pair = self.dp_mri([ct, mri])[-1]
		loss_fake_mri_pair = self.discriminator_loss(False, pred_fake_mri_pair)
		loss_real_mri_pair = self.discriminator_loss(True, pred_real_mri_pair)
		total_loss_mri_pair = 0.5 * (loss_fake_mri_pair + loss_real_mri_pair)
			
		# Calculate CT discriminator pair losses.
		pred_fake_ct_pair = self.dp_ct([mri, fake_ct])[-1] 
		pred_real_ct_pair = self.dp_ct([mri, ct])[-1]
		loss_fake_ct_pair = self.discriminator_loss(False, pred_fake_ct_pair)
		loss_real_ct_pair = self.discriminator_loss(True, pred_real_ct_pair)
		total_loss_ct_pair = 0.5 * (loss_fake_ct_pair + loss_real_ct_pair)
			
		# Calculate MRI discriminator cycle losses.
		pred_fake_mri_cycle = self.dc_mri([mri, fake_mri])[-1] 
		pred_real_mri_cycle = self.dp_mri([mri, mri])[-1]
		loss_fake_mri_cycle = self.discriminator_loss(False, pred_fake_mri_cycle)
		loss_real_mri_cycle = self.discriminator_loss(True, pred_real_mri_cycle)
		total_loss_mri_cycle = 0.5 * (loss_fake_mri_cycle + loss_real_mri_cycle)
			
		# Calculate CT discriminator cycle losses.
		pred_fake_ct_cycle = self.dp_ct([ct, fake_ct])[-1] 
		pred_real_ct_cycle = self.dp_ct([ct, ct])[-1]
		loss_fake_ct_cycle = self.discriminator_loss(False, pred_fake_ct_cycle)
		loss_real_ct_cycle = self.discriminator_loss(True, pred_real_ct_cycle)
		total_loss_ct_cycle = 0.5 * (loss_fake_ct_cycle + loss_real_ct_cycle)


		
		generated_mri, generated_ct, id_mri, id_ct, cycled_mri, cycled_ct = self.combined_model(
			[mri, mri_mask, latent_mri, ct, ct_mask, latent_ct]
		)

		# Calculate MRI generator losses.
		kl_loss_mri = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_mri, variance_mri)
		vgg_loss_mri = self.vgg_feature_loss_coeff * self.vgg_loss(mri, generated_mri)
		ssim_loss_mri = self.ssim_loss_coeff * loss.SSIMLoss(mri, generated_mri)
		cycle_loss_mri = self.cycle_loss_coeff * loss.mae(mri, cycled_mri)
		identity_loss_mri = self.identity_loss_coeff * loss.mae(mri, id_mri)

		# Calculate CT generator losses.
		kl_loss_ct = self.kl_divergence_loss_coeff * loss.kl_divergence_loss(mean_ct, variance_ct)
		vgg_loss_ct = self.vgg_feature_loss_coeff * self.vgg_loss(ct, generated_ct)
		ssim_loss_ct = self.ssim_loss_coeff * loss.SSIMLoss(ct, generated_ct)
		cycle_loss_ct = self.cycle_loss_coeff * loss.mae(ct, cycled_ct)
		identity_loss_ct = self.identity_loss_coeff * loss.mae(ct, id_ct)

		
		# Report progress.
		self.disc_pair_loss_mri_tracker.update_state(total_loss_mri_pair)
		self.disc_cycle_loss_mri_tracker.update_state(total_loss_mri_cycle)
		self.kl_loss_mri_tracker.update_state(kl_loss_mri)
		self.ssim_loss_mri_tracker.update_state(ssim_loss_mri)
		self.vgg_loss_mri_tracker.update_state(vgg_loss_mri)
		self.cycle_loss_mri_tracker.update_state(cycle_loss_mri)
		self.identity_loss_mri_tracker.update_state(identity_loss_mri)

		self.disc_pair_loss_ct_tracker.update_state(total_loss_ct_pair)
		self.disc_cycle_loss_ct_tracker.update_state(total_loss_ct_cycle)
		self.kl_loss_ct_tracker.update_state(kl_loss_ct)
		self.ssim_loss_ct_tracker.update_state(ssim_loss_ct)
		self.vgg_loss_ct_tracker.update_state(vgg_loss_ct)
		self.cycle_loss_ct_tracker.update_state(cycle_loss_ct)
		self.identity_loss_ct_tracker.update_state(identity_loss_ct)


		results = {m.name: m.result() for m in self.metrics}
		return results
	
	def call(self, inputs):
		latent_vector, mask, ct = inputs
		return self.de_mri([latent_vector, mask, ct])
	
	def model_evaluate(self, data, epoch=0):
		results = []
		
		for ct, ct_mask, mri, mri_mask in data:

			# Sample latent from a normal distribution.
			latent_vector = tf.random.normal(
				shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=2.0
			)
			fake_image = self.de_mri([latent_vector, ct_mask, ct])

			# Normalize to values between 0 and 1.
			mri = (mri + 1.0) / 2.0
			fake_image = (fake_image + 1.0) / 2.0
			
			# Calculate metrics.
			fid = evaluate.calculate_fid(mri, fake_image,
																	 input_shape=(
																		 self.flags.crop_size,
																		 self.flags.crop_size, 3))
			mse, mae, cs, psnr, ssim = evaluate.get_metrics(mri, fake_image)
			
			results.append([fid, mse, mae, cs, psnr, ssim])
			print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))
		

		# Save results.
		results = np.array(results, dtype=object).mean(axis=0)
		filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
		results_dir = os.path.join(self.flags.result_logs, self.flags.name)
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		log_file = os.path.join(results_dir, filename)
		np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")

	
	
	def save_model(self):
		self.en_mri.save(self.flags.model_path + self.experiment_name + '_e_mri')
		self.de_mri.save(self.flags.model_path + self.experiment_name + '_d_mri')
		self.en_ct.save(self.flags.model_path + self.experiment_name + '_e_ct')
		self.de_ct.save(self.flags.model_path + self.experiment_name + '_d_ct')


	def plot_losses(self, hist):
		
		exp_path = self.hist_path + self.experiment_name
		
		if not os.path.exists(exp_path):
			os.makedirs(exp_path)
			
		# Save history to a file.  
		hist_df = pd.DataFrame(hist) 
		hist_df.to_csv(exp_path + '/hist.csv')
		
		
		losses = ['disc_pair_loss_mri', 'disc_cycle_loss_mri', 'kl_loss_mri', 'ssim_loss_mri', 'vgg_loss_mri', 
			'cycle_loss_mri', 'identity_loss_mri', 'disc_pair_loss_ct', 'disc_cycle_loss_ct', 'kl_loss_ct', 'ssim_loss_ct', 
			'vgg_loss_ct', 'cycle_loss_ct', 'identity_loss_ct']
		
		
		# Plot losses.
		for loss in losses:
			plt.figure()
			plt.plot(hist[loss + '_loss'])
			plt.plot(hist['val_' + loss + '_loss'])
			plt.legend([loss + '_loss','val_' + loss + '_loss'],loc='upper right')
			plt.savefig(exp_path + '/pcxgan_' + loss + '.png')



