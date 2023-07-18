import tensorflow.keras as kr
import tensorflow as tf
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import data_loader
import flags
import random
#import tensorflow_addons as tfa
#from tensorflow_addons.layers import InstanceNormalization


class SPADE(kr.layers.Layer):
	def __init__(self, filters, flags, **kwargs):
		super().__init__(**kwargs)
		self.epsilon = flags.s_epsilon
		self.conv = kr.layers.Conv2D(128, 3, padding="same", activation="relu")
		self.conv_gamma = kr.layers.Conv2D(filters, flags.s_gamma_filter_size, padding="same")
		self.conv_beta = kr.layers.Conv2D(filters, flags.s_beta_filter_size, padding="same")
	
	def build(self, input_shape):
		self.resize_shape = input_shape[1:3]
	
	def call(self, input_tensor, raw_ct):
		mask = tf.image.resize(raw_ct, self.resize_shape, method="nearest")
		x = self.conv(mask)
		gamma = self.conv_gamma(x)
		beta = self.conv_beta(x)
		mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
		std = tf.sqrt(var + self.epsilon)
		normalized = (input_tensor - mean) / std
		output = gamma * normalized + beta
		return output


class ResBlock(kr.layers.Layer):
	def __init__(self, flags, filters, **kwargs):
		super().__init__(**kwargs)
		self.filters = filters
		self.flags = flags
	
	def build(self, input_shape):
		input_filter = input_shape[-1]
		self.spade_1 = SPADE(input_filter, self.flags)
		self.spade_2 = SPADE(self.filters, self.flags)
		self.conv_1 = kr.layers.Conv2D(self.filters, self.flags.s_gamma_filter_size, padding="same")
		self.conv_2 = kr.layers.Conv2D(self.filters, self.flags.s_gamma_filter_size, padding="same")
		self.learned_skip = False
		
		if self.filters != input_filter:
			self.learned_skip = True
			self.spade_3 = SPADE(input_filter, self.flags)
			self.conv_3 = kr.layers.Conv2D(self.filters, self.flags.s_gamma_filter_size, padding="same")
	
	def call(self, input_tensor, ct):
		x = self.spade_1(input_tensor, ct)
		x = self.conv_1(tf.nn.leaky_relu(x, 0.2))
		x = self.spade_2(x, ct)
		x = self.conv_2(tf.nn.leaky_relu(x, 0.2))
		skip = (
			self.conv_3(tf.nn.leaky_relu(self.spade_3(input_tensor, ct), 0.2))
			if self.learned_skip
			else input_tensor
		)
		output = skip + x
		return output


class GaussianSampler(kr.layers.Layer):
	def __init__(self, batch_size, latent_dim, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = 1 #batch_size
		self.latent_dim = latent_dim
	
	def call(self, inputs):
		means, variance = inputs
		epsilon = tf.random.normal(
			shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=1.0, seed=1234
		)
		samples = means + tf.exp(0.5 * variance) * epsilon
		return samples


class DownsampleModule(kr.layers.Layer):
	def __init__(self, channels, filter_size, apply_norm=True, **kwargs):
		super().__init__(**kwargs)
		gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1234)
		self.block = kr.Sequential()
		self.strides = 2
		self.block.add(
			kr.layers.Conv2D(
				channels,
				filter_size,
				strides=self.strides,
				padding="same",
				use_bias=False
			)
		)

		if apply_norm:
			self.block.add(kr.layers.GroupNormalization(groups=channels, gamma_initializer=gamma_init))
			#self.block.add(InstanceNormalization())

		self.block.add(kr.layers.LeakyReLU(0.2))
	
	def call(self, inputs__):
		return self.block(inputs__)


class UpsampleModule(kr.layers.Layer):
	def __init__(self, channels, filter_size, batch_norm=True, dropout=True,
							 apply_activation=True, **kwargs):
		super().__init__(**kwargs)
		self.block = kr.Sequential()
		self.strides = 2
		self.apply_activation = apply_activation
		
		# self.block.add(kr.layers.UpSampling2D((2, 2)))
		
		self.block.add(kr.layers.Conv2DTranspose(channels, filter_size, strides=self.strides, padding='same',
																						 kernel_initializer=kr.initializers.RandomNormal(stddev=0.02, seed=123),
																						 kernel_regularizer=kr.regularizers.l1_l2(l1=1e-5, l2=1e-5),
																						 activity_regularizer=kr.regularizers.l2(1e-5)))
		
		if batch_norm:
			self.block.add(kr.layers.BatchNormalization())
		if dropout:
			self.block.add(kr.layers.Dropout(0.5))
		if self.apply_activation:
			self.block.add(kr.layers.LeakyReLU(0.2))
	
	def call(self, inputs_):
		return self.block(inputs_)


class Encoder(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		n_filters = flags.e_n_filters
		filter_size = flags.e_filter_size
		self.downsample1 = DownsampleModule(n_filters, filter_size=filter_size, apply_norm=False)
		self.downsample2 = DownsampleModule(2 * n_filters, filter_size)
		self.downsample3 = DownsampleModule(4 * n_filters, filter_size)
		self.downsample4 = DownsampleModule(8 * n_filters, filter_size)
		self.downsample5 = DownsampleModule(8 * n_filters, filter_size)
		# self.downsample6 = DownsampleModule(8 * n_filters, filter_size)
		
		self.flatten = kr.layers.Flatten()
		self.mean = kr.layers.Dense(self.latent_dim)
		self.variance = kr.layers.Dense(self.latent_dim)
	
	def call(self, input_, **kwargs):
		x = self.downsample1(input_)
		x = self.downsample2(x)
		x = self.downsample3(x)
		x = self.downsample4(x)
		x = self.downsample5(x)
		# x = self.downsample6(x)
		x = self.flatten(x)
		return [self.mean(x), self.variance(x)]
	
	def build_graph(self):
		x = kr.layers.Input(shape=self.image_shape)
		return kr.Model(inputs=[x], outputs=self.call(x))


class Decoder(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		#self.mask_shape = (flags.crop_size, flags.crop_size, 2)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		res_filters = flags.d_res_filters
		self.dense1 = kr.layers.Dense(self.latent_dim * 4 * 4)
		self.reshape = kr.layers.Reshape((4, 4, self.latent_dim))
		self.resblock1 = ResBlock(flags, filters=res_filters)
		self.upsample1 = kr.layers.UpSampling2D((2, 2))
		self.resblock2 = ResBlock(flags, filters=res_filters)
		self.upsample2 = kr.layers.UpSampling2D((2, 2))
		self.resblock3 = ResBlock(flags, filters=res_filters)
		self.upsample3 = kr.layers.UpSampling2D((2, 2))
		self.resblock4 = ResBlock(flags, filters=res_filters / 2)
		self.upsample4 = kr.layers.UpSampling2D((2, 2))
		self.resblock5 = ResBlock(flags, filters=res_filters / 4)
		self.upsample5 = kr.layers.UpSampling2D((2, 2))
		self.resblock6 = ResBlock(flags, filters=res_filters / 8)
		self.upsample6 = kr.layers.UpSampling2D((2, 2))
		# self.resblock7 = ResBlock(flags, filters=res_filters / 16)
		# self.upsample7 = kr.layers.UpSampling2D((2, 2))
		self.activation = kr.layers.LeakyReLU(0.2)
		self.out_image = kr.layers.Conv2D(1, kernel_size=4, padding='same', activation='sigmoid')
	
	def build_graph(self):
		#m = kr.layers.Input(shape=self.mask_shape)
		l = kr.layers.Input(shape=self.latent_dim)
		c = kr.layers.Input(shape=self.image_shape)
		return kr.Model(inputs=[l, c], outputs=self.call([l, c]))
	
	def call(self, inputs_, **kwargs):
		latent, ct = inputs_
		x = self.dense1(latent)
		x = self.reshape(x)
		x = self.resblock1(x, ct)
		x = self.upsample1(x)
		x = self.resblock2(x, ct)
		x = self.upsample2(x)
		x = self.resblock3(x, ct)
		x = self.upsample3(x)
		x = self.resblock4(x, ct)
		x = self.upsample4(x)
		x = self.resblock5(x, ct)
		x = self.upsample5(x)
		x = self.resblock6(x, ct)
		x = self.upsample6(x)
		# x = self.resblock7(x, mask)
		# x = self.upsample7(x)
		x = self.activation(x)
		return self.out_image(x)


class Discriminator(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		n_filters = flags.disc_n_filters
		filter_size = flags.disc_filter_size
		self.merged = kr.layers.Concatenate()
		self.downsample1 = DownsampleModule(n_filters, filter_size, apply_norm=False)
		self.downsample2 = DownsampleModule(2 * n_filters, filter_size)
		self.downsample3 = DownsampleModule(4 * n_filters, filter_size)
		self.downsample4 = DownsampleModule(8 * n_filters, filter_size)
		self.conv = kr.layers.Conv2D(self.image_shape[-1], kernel_size=filter_size, padding='same')
	
	def call(self, inputs_, **kwargs):
		x = self.merged([inputs_[0], inputs_[1]])
		x1 = self.downsample1(x)
		x2 = self.downsample2(x1)
		x3 = self.downsample3(x2)
		x4 = self.downsample4(x3)
		x5 = self.conv(x4)
		return [x1, x2, x3, x4, x5]
	
	def build_graph(self):
		x1 = kr.layers.Input(shape=self.image_shape)
		x2 = kr.layers.Input(shape=self.image_shape)
		return kr.Model(inputs=[x1, x2], outputs=self.call([x1, x2]))


class GanMonitor(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags):

		'''
		self.val_images = next(iter(val_dataset))
		if flags.batch_size > 3:
			self.n_samples = 3
		else:
			self.n_samples = 1
		'''
		self.val_images = next(iter(val_dataset))
		self.n_samples = 1

		self.epoch_interval = flags.epoch_interval
		self.checkpoints_path = os.path.join(flags.checkpoints_dir, flags.name)
		self.hist_path = os.path.join(flags.hist_path, flags.name)
		self.sample_dir = os.path.join(flags.sample_dir, flags.name)
		self.losses = {'disc_loss': [], 'kl_loss': [], 'vgg_loss': [], 'ssim_loss': []} 
		self.flags = flags
		
		# create directories if needed
		if not os.path.exists(self.checkpoints_path):
			os.makedirs(self.checkpoints_path)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
		if not os.path.exists(self.hist_path):
			os.makedirs(self.hist_path)
	
	# self.save_models(self.checkpoints_path)
	# self.model.model_evaluate(val_dataset)
	
	def infer(self, batch=False):
		latent_vector = tf.random.normal(
			shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0, seed=500
		)
		'''
		# get random images if the batch size is larger than 3
		if batch == True:
			#self.val_images= self.batch_images
			# print("Val Images: ", tf.shape(self.val_images))
			# print("Val Images 0: ", tf.shape(self.val_images[0]))
			# print("Val Images 1: ", tf.shape(self.val_images[1]))
			indices = np.random.permutation(self.flags.batch_size)
			#self.n_masks = self.val_images[2].numpy()[indices]

			self.n_cts = self.val_images[0].numpy()[indices]
			self.n_mris = self.val_images[1].numpy()[indices]
			# print("CTs: ", tf.shape(self.n_cts))
			# print("MRIs: ", tf.shape(self.n_mris))

		else:
			#self.n_masks = self.val_images[2]
			self.n_cts = self.val_images[0]
			self.n_mris = self.val_images[1]
		'''
		return self.model.predict([latent_vector, self.val_images[0]])
	
	def save_models(self, epoch):
		e_name = f"encoder_epoch_{epoch}"
		d_name = f"decoder_epoch_{epoch}"
		self.model.encoder.save(os.path.join(self.checkpoints_path, e_name))
		self.model.decoder.save(os.path.join(self.checkpoints_path, d_name))
	
	def on_epoch_end(self, epoch, logs=None):
		if epoch > 0 and epoch % self.epoch_interval == 0:
			'''
			# get predicted images
			if self.n_samples == 3:
				generated_images = self.infer(batch=True)
			else:
				generated_images = self.infer()
			'''
			generated_images = self.infer()
			# plot training samples
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					#r = random.randrange(self.flags.batch_size)
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((self.val_images[0][row].squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.val_images[1][row].squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("rMRI", fontsize=20)
					ax[2].imshow((generated_images[row].squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("PCxGAN sMRI", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)


				self.losses['disc_loss'].append(logs['disc_loss']) 
				self.losses['kl_loss'].append(logs['kl_loss']) 
				self.losses['vgg_loss'].append(logs['vgg_loss']) 
				self.losses['ssim_loss'].append(logs['ssim_loss']) 

				# Plot losses
				plt.figure()
				plt.plot(self.losses['disc_loss'], label='Discriminator Loss')
				plt.plot(self.losses['kl_loss'], label='KL Loss')
				plt.plot(self.losses['vgg_loss'], label='VGG Loss')
				plt.plot(self.losses['ssim_loss'], label='SSIM Loss')
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.legend()
				plt.title('PCxGAN Losses')
				plt.savefig(os.path.join(self.hist_path, 'losses.png'))
				plt.close()


				for loss in self.losses.keys():
					plt.figure()
					plt.plot(self.losses[loss])
					plt.title(loss)
					plt.savefig(self.hist_path + '/pcxgan_' + loss + '.png')
					plt.close()