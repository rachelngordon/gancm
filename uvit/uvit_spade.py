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
		self.timesteps = self.flags.timesteps
		self.img_channels = self.flags.img_channels
		self.first_conv_channels = self.flags.first_conv_channels
		self.widths = [self.first_conv_channels * mult for mult in self.flags.channel_multiplier]
		self.has_attention = self.flags.has_attention
		self.num_res_blocks = self.flags.num_res_blocks
		self.norm_groups = self.flags.norm_groups
		
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
		image_input = kr.layers.Input(shape=self.image_shape, name="image_input")
		time_input = kr.Input(shape=(), dtype=tf.int64, name="time_input")
		mask_input = kr.Input(shape=self.mask_shape, dtype=tf.int64, name="mask_input")
		latent_input = kr.Input(shape=self.latent_dim, name="latent")

		_, _, temb, skips = self.encoder([image_input, time_input])
		self.decoder = self.decoder.build_graph(time_input.shape, len(skips))

		generated_image = self.decoder([latent_input, temb, mask_input, skips])
		discriminator_output = self.discriminator([image_input, generated_image])
		
		patch_size = discriminator_output[-1].shape[1]
		combined_model = kr.Model(
			[latent_input, temb, mask_input, skips],
			[discriminator_output, generated_image],
		)
		return patch_size, combined_model


	def compile(self, **kwargs):
		super().compile(**kwargs)
	
	def call(self, inputs):
		latent_vector, temb, mask_input, skips = inputs
		return self.combined_model([latent_vector, temb, mask_input, skips])
	
	def save_model(self, flags):
		self.combined_model.save(flags.model_path + flags.exp_name)
	
    
	def train_discriminator(self, time_input, ct, mri, mask):
		self.discriminator.trainable = True
		
		fake_images = self.combined_model([ct, time_input, mask], training=False)

		
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
	
	

	def train_generator(self, latent_vector, segmentation_map, labels, image, time_input):
		
		self.discriminator.trainable = False


		with tf.GradientTape(persistent=True) as en_tape:
			
			mean, variance, temb, skips = self.encoder([image, time_input])

			
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
			fake_d_output, fake_image = self.combined_model([latent_vector, temb, labels, skips])
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
		
		source, target, mask = data
		batch_size = tf.shape(source)[0]
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		# Obtain the learned moments of the real image distribution.
		mean, variance, temb, skips = self.encoder([target, t])
		
		# Sample a latent from the distribution defined by the learned moments.
		latent_vector = self.sampler([mean, variance])

		discriminator_loss = self.train_discriminator(
			temb, source, target, mask
		)
		(kl_loss, vgg_loss, ssim_loss) = self.train_generator(
			latent_vector, source, mask, target, t
		)
		

		discriminator_loss = self.train_discriminator(t, source, target, mask)
		i_discriminator_loss = 0.5* discriminator_loss

		self.discriminator.trainable = False
		with tf.GradientTape() as tape:
			pred_ = self.combined_model([latent_vector, temb, mask, skips], training=True)
			vgg_loss = self.vgg_loss(target, pred_)
			ssim_loss = loss.SSIMLoss(target, pred_)
			total_loss = vgg_loss + ssim_loss + i_discriminator_loss
		
		gradients = tape.gradient(total_loss, self.combined_model.trainable_variables)
		self.gen_optimizer.apply_gradients(zip(gradients, self.combined_model.trainable_variables))

		# Report progress.
		self.vgg_loss_tracker.update_state(vgg_loss)
		self.ssim_loss_tracker.update_state(ssim_loss)
		self.disc_loss_tracker.update_state(discriminator_loss)

		results = {m.name: m.result() for m in self.metrics}
		return results
	
	
	def test_step(self, data):
		source, target, mask = data
		batch_size = tf.shape(source)[0]
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)

		# Obtain the learned moments of the real image distribution.
		mean, variance, temb, skips = self.encoder([target, t])
		
		# Sample a latent from the distribution defined by the learned moments.
		latent_vector = self.sampler([mean, variance])
		
		pred_ = self.combined_model([latent_vector, temb, mask, skips])
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
		latent_vector = tf.random.normal(
				shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=2.0
			)
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
		)

		# for testing we need the encoder because it outputs the skips!! and t --> temb
		
		return self.combined_model([latent_vector, temb, mask, skips])
	
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
				input_shape=(self.image_size, self.image_size, 3))

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

