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
import tensorflow_addons as tfa
import keras

@keras.saving.register_keras_serializable(package="MyLayers")
class SPADE(kr.layers.Layer):
	def __init__(self, filters, flags, **kwargs):
		super().__init__(**kwargs)
		self.epsilon = flags.s_epsilon
		self.conv = kr.layers.Conv2D(128, 3, padding="same", activation="relu")
		self.conv_gamma = kr.layers.Conv2D(filters, flags.s_gamma_filter_size, padding="same")
		self.conv_beta = kr.layers.Conv2D(filters, flags.s_beta_filter_size, padding="same")
	
	def build(self, input_shape):
		self.resize_shape = input_shape[1:3]
	
	def call(self, input_tensor, raw_mask, raw_ct):
		merge = kr.layers.concatenate([raw_mask, raw_ct])
		mask = tf.image.resize(merge, self.resize_shape, method="nearest")
		x = self.conv(mask)
		gamma = self.conv_gamma(x)
		beta = self.conv_beta(x)
		mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
		std = tf.sqrt(var + self.epsilon)
		normalized = (input_tensor - mean) / std
		output = gamma * normalized + beta
		return output
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "conv": kr.saving.serialize_keras_object(self.conv),
			"conv_gamma": kr.saving.serialize_keras_object(self.conv_gamma),
			"conv_beta": kr.saving.serialize_keras_object(self.conv_beta),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		conv_config = config.pop("conv")
		conv = kr.saving.deserialize_keras_object(conv_config)
		conv_gamma_config = config.pop("conv_gamma")
		conv_gamma = kr.saving.deserialize_keras_object(conv_gamma_config)
		conv_beta_config = config.pop("conv_beta")
		conv_beta = kr.saving.deserialize_keras_object(conv_beta_config)
		return cls(conv, conv_gamma, conv_beta, **config)
	


@kr.saving.register_keras_serializable(package="MyLayers")
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
	
	def call(self, input_tensor, mask, ct):
		x = self.spade_1(input_tensor, mask, ct)
		x = self.conv_1(tf.nn.leaky_relu(x, 0.2))
		x = self.spade_2(x, mask, ct)
		x = self.conv_2(tf.nn.leaky_relu(x, 0.2))
		skip = (
			self.conv_3(tf.nn.leaky_relu(self.spade_3(input_tensor, mask, ct), 0.2))
			if self.learned_skip
			else input_tensor
		)
		output = skip + x
		return output
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "spade_1": kr.saving.serialize_keras_object(self.spade_1),
			"spade_2": kr.saving.serialize_keras_object(self.spade_2),
			"spade_3": kr.saving.serialize_keras_object(self.spade_3),
			"conv_1": kr.saving.serialize_keras_object(self.spade_1),
			"conv_2": kr.saving.serialize_keras_object(self.spade_2),
			"conv_3": kr.saving.serialize_keras_object(self.spade_3),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		spade_1_config = config.pop("spade_1")
		spade_1 = kr.saving.deserialize_keras_object(spade_1_config)
		spade_2_config = config.pop("spade_2")
		spade_2 = kr.saving.deserialize_keras_object(spade_2_config)
		spade_3_config = config.pop("spade_3")
		spade_3 = kr.saving.deserialize_keras_object(spade_3_config)
		conv_1_config = config.pop("conv_1")
		conv_1 = kr.saving.deserialize_keras_object(conv_1_config)
		conv_2_config = config.pop("conv_2")
		conv_2 = kr.saving.deserialize_keras_object(conv_2_config)
		conv_3_config = config.pop("conv_3")
		conv_3 = kr.saving.deserialize_keras_object(conv_3_config)
		return cls(spade_1, spade_2, spade_3, conv_1, conv_2, conv_3, **config)
	

@kr.saving.register_keras_serializable(package="MyLayers")
class GaussianSampler(kr.layers.Layer):
	def __init__(self, batch_size, latent_dim, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.latent_dim = latent_dim
	
	def call(self, inputs):
		means, variance = inputs
		epsilon = tf.random.normal(
			shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=1.0, seed=1234
		)
		samples = means + tf.exp(0.5 * variance) * epsilon
		return samples
	

@kr.saving.register_keras_serializable(package="MyLayers")
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
			self.block.add(tfa.layers.GroupNormalization(groups=channels, gamma_initializer=gamma_init))

		self.block.add(kr.layers.LeakyReLU(0.2))
	
	def call(self, inputs__):
		return self.block(inputs__)
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "gamma_init": kr.saving.serialize_keras_object(self.gamma_init),
			"block": kr.saving.serialize_keras_object(self.block),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		gamma_init_config = config.pop("gamma_init")
		gamma_init = kr.saving.deserialize_keras_object(gamma_init_config)
		block_config = config.pop("block")
		block = kr.saving.deserialize_keras_object(block_config)
		return cls(gamma_init, block, **config)

@kr.saving.register_keras_serializable(package="MyLayers")
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
	
	def get_config(self):
		base_config = super().get_config()
		config = {
			"block": kr.saving.serialize_keras_object(self.block),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		block_config = config.pop("block")
		block = kr.saving.deserialize_keras_object(block_config)
		return cls(block, **config)

@kr.saving.register_keras_serializable(package="MyLayers")
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
	

	def get_config(self):
		base_config = super().get_config()
		config = {
			"image_shape": kr.saving.serialize_keras_object(self.image_shape),

			"downsample1": kr.saving.serialize_keras_object(self.downsample1),
			"downsample2": kr.saving.serialize_keras_object(self.downsample2),
			"downsample3": kr.saving.serialize_keras_object(self.downsample3),
			"downsample4": kr.saving.serialize_keras_object(self.downsample4),
			"downsample5": kr.saving.serialize_keras_object(self.downsample5),

			"flatten": kr.saving.serialize_keras_object(self.flatten),
			"mean": kr.saving.serialize_keras_object(self.mean),
			"variance": kr.saving.serialize_keras_object(self.variance),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):

		image_shape_config = config.pop("image_shape")
		image_shape = kr.saving.deserialize_keras_object(image_shape_config)

		downsample1_config = config.pop("downsample1")
		downsample1 = kr.saving.deserialize_keras_object(downsample1_config)
		downsample2_config = config.pop("downsample2")
		downsample2 = kr.saving.deserialize_keras_object(downsample2_config)
		downsample3_config = config.pop("downsample3")
		downsample3 = kr.saving.deserialize_keras_object(downsample3_config)
		downsample4_config = config.pop("downsample4")
		downsample4 = kr.saving.deserialize_keras_object(downsample4_config)
		downsample5_config = config.pop("downsample5")
		downsample5 = kr.saving.deserialize_keras_object(downsample5_config)

		flatten_config = config.pop("flatten")
		flatten = kr.saving.deserialize_keras_object(flatten_config)
		mean_config = config.pop("mean")
		mean = kr.saving.deserialize_keras_object(mean_config)
		variance_config = config.pop("variance")
		variance = kr.saving.deserialize_keras_object(variance_config)

		return cls(image_shape, downsample1, downsample2, downsample3, downsample4, downsample5, 
			 flatten, mean, variance, **config)


@kr.saving.register_keras_serializable(package="MyLayers")
class Decoder(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		self.mask_shape = (flags.crop_size, flags.crop_size, 2)
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
		self.out_image = kr.layers.Conv2D(1, kernel_size=4, padding='same', activation='tanh')
	
	def build_graph(self):
		m = kr.layers.Input(shape=self.mask_shape)
		l = kr.layers.Input(shape=self.latent_dim)
		c = kr.layers.Input(shape=self.image_shape)
		return kr.Model(inputs=[l, m, c], outputs=self.call([l, m, c]))
	
	def call(self, inputs_, **kwargs):
		latent, mask, ct = inputs_
		x = self.dense1(latent)
		x = self.reshape(x)
		x = self.resblock1(x, mask, ct)
		x = self.upsample1(x)
		x = self.resblock2(x, mask, ct)
		x = self.upsample2(x)
		x = self.resblock3(x, mask, ct)
		x = self.upsample3(x)
		x = self.resblock4(x, mask, ct)
		x = self.upsample4(x)
		x = self.resblock5(x, mask, ct)
		x = self.upsample5(x)
		x = self.resblock6(x, mask, ct)
		x = self.upsample6(x)
		# x = self.resblock7(x, mask)
		# x = self.upsample7(x)
		x = self.activation(x)
		return self.out_image(x)
	

	def get_config(self):
		base_config = super().get_config()
		config = {
			"mask_shape": kr.saving.serialize_keras_object(self.mask_shape),
			"image_shape": kr.saving.serialize_keras_object(self.image_shape),
			"dense1": kr.saving.serialize_keras_object(self.dense1),
			"reshape": kr.saving.serialize_keras_object(self.reshape),

			"resblock1": kr.saving.serialize_keras_object(self.resblock1),
			"upsample1": kr.saving.serialize_keras_object(self.upsample1),
			"resblock2": kr.saving.serialize_keras_object(self.resblock2),
			"upsample2": kr.saving.serialize_keras_object(self.upsample2),
			"resblock3": kr.saving.serialize_keras_object(self.resblock3),
			"upsample3": kr.saving.serialize_keras_object(self.upsample3),
			"resblock4": kr.saving.serialize_keras_object(self.resblock4),
			"upsample4": kr.saving.serialize_keras_object(self.upsample4),
			"resblock5": kr.saving.serialize_keras_object(self.resblock5),
			"upsample5": kr.saving.serialize_keras_object(self.upsample5),
			"resblock6": kr.saving.serialize_keras_object(self.resblock6),
			"upsample6": kr.saving.serialize_keras_object(self.upsample6),

			"activation": kr.saving.serialize_keras_object(self.activation),
			"out_image": kr.saving.serialize_keras_object(self.out_image),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		mask_shape_config = config.pop("mask_shape")
		mask_shape = kr.saving.deserialize_keras_object(mask_shape_config)
		image_shape_config = config.pop("image_shape")
		image_shape = kr.saving.deserialize_keras_object(image_shape_config)

		dense1_config = config.pop("dense1")
		dense1 = kr.saving.deserialize_keras_object(dense1_config)
		reshape_config = config.pop("reshape")
		reshape = kr.saving.deserialize_keras_object(reshape_config)

		resblock1_config = config.pop("resblock1")
		resblock1 = kr.saving.deserialize_keras_object(resblock1_config)
		upsample1_config = config.pop("upsample1")
		upsample1 = kr.saving.deserialize_keras_object(upsample1_config)
		resblock2_config = config.pop("resblock2")
		resblock2 = kr.saving.deserialize_keras_object(resblock2_config)
		upsample2_config = config.pop("upsample2")
		upsample2 = kr.saving.deserialize_keras_object(upsample2_config)
		resblock3_config = config.pop("resblock3")
		resblock3 = kr.saving.deserialize_keras_object(resblock3_config)
		upsample3_config = config.pop("upsample3")
		upsample3 = kr.saving.deserialize_keras_object(upsample3_config)
		resblock4_config = config.pop("resblock4")
		resblock4 = kr.saving.deserialize_keras_object(resblock4_config)
		upsample4_config = config.pop("upsample4")
		upsample4 = kr.saving.deserialize_keras_object(upsample4_config)
		resblock5_config = config.pop("resblock5")
		resblock5 = kr.saving.deserialize_keras_object(resblock5_config)
		upsample5_config = config.pop("upsample5")
		upsample5 = kr.saving.deserialize_keras_object(upsample5_config)
		resblock6_config = config.pop("resblock6")
		resblock6 = kr.saving.deserialize_keras_object(resblock6_config)
		upsample6_config = config.pop("upsample6")
		upsample6 = kr.saving.deserialize_keras_object(upsample6_config)

		activation_config = config.pop("activation")
		activation = kr.saving.deserialize_keras_object(activation_config)
		out_image_config = config.pop("out_image")
		out_image = kr.saving.deserialize_keras_object(out_image_config)

		return cls(mask_shape, image_shape, dense1, reshape, 
			 resblock1, upsample1, resblock2, upsample2, resblock3, upsample3, resblock4, upsample4,
			  resblock5, upsample5, resblock6, upsample6, activation, out_image, **config)

@kr.saving.register_keras_serializable(package="MyLayers")
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
	
	def get_config(self):
		base_config = super().get_config()
		config = {
			"image_shape": kr.saving.serialize_keras_object(self.image_shape),
			"merged": kr.saving.serialize_keras_object(self.merged),

			"downsample1": kr.saving.serialize_keras_object(self.downsample1),
			"downsample2": kr.saving.serialize_keras_object(self.downsample2),
			"downsample3": kr.saving.serialize_keras_object(self.downsample3),
			"downsample4": kr.saving.serialize_keras_object(self.downsample4),

			"conv": kr.saving.serialize_keras_object(self.conv),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):

		image_shape_config = config.pop("image_shape")
		image_shape = kr.saving.deserialize_keras_object(image_shape_config)
		merged_config = config.pop("merged")
		merged = kr.saving.deserialize_keras_object(merged_config)

		downsample1_config = config.pop("downsample1")
		downsample1 = kr.saving.deserialize_keras_object(downsample1_config)
		downsample2_config = config.pop("downsample2")
		downsample2 = kr.saving.deserialize_keras_object(downsample2_config)
		downsample3_config = config.pop("downsample3")
		downsample3 = kr.saving.deserialize_keras_object(downsample3_config)
		downsample4_config = config.pop("downsample4")
		downsample4 = kr.saving.deserialize_keras_object(downsample4_config)

		conv_config = config.pop("conv")
		conv = kr.saving.deserialize_keras_object(conv_config)

		return cls(image_shape, merged, downsample1, downsample2, downsample3, downsample4, conv, **config)


class GanMonitor(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags):

		'''
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
			self.val_images=self.batch_images

			indices = np.random.permutation(self.flags.batch_size)
			self.n_masks = self.val_images[2].numpy()[indices]
			self.n_cts = self.val_images[0].numpy()[indices]
			self.n_mris = self.val_images[1].numpy()[indices]

		else:
			self.n_masks = self.val_images[2]
			self.n_cts = self.val_images[0]
			self.n_mris = self.val_images[1]
		'''

		return self.model.predict([latent_vector, self.val_images[2], self.val_images[0]])
	
	def save_models(self, epoch):
		e_name = f"encoder_epoch_{epoch}"
		d_name = f"decoder_epoch_{epoch}"
		self.model.encoder.save(os.path.join(self.checkpoints_path, e_name))
		self.model.decoder.save(os.path.join(self.checkpoints_path, d_name))
	
	def on_epoch_end(self, epoch, logs=None):
		if epoch > 0 and epoch % self.epoch_interval == 0:
			'''
			#self.save_models(epoch)
			
			# get predicted images
			if self.n_samples == 3:
				self.batch_images = next(iter(self.val_images))
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
					ax[0].imshow((self.val_images[0][row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.val_images[1][row].numpy().squeeze() + 1) / 2, cmap='gray')
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
					plt.savefig(self.hist_path + '/pcxgan_' + loss + '_loss.png')
					plt.close()