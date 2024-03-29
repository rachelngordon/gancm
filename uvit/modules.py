import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


# Kernel initializer to use
def kernel_init(scale):
	scale = max(scale, 1e-10)
	return kr.initializers.VarianceScaling(
		scale, mode="fan_avg", distribution="uniform"
	)


class AttentionBlock(kr.layers.Layer):
	"""Applies self-attention.

	Args:
		units: Number of units in the dense layers
		groups: Number of groups to be used for GroupNormalization layer
	"""
	
	def __init__(self, units, groups=8, **kwargs):
		self.units = units
		self.groups = groups
		super().__init__(**kwargs)
		
		self.norm = tfa.layers.GroupNormalization(groups=groups)
		self.query = kr.layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.key = kr.layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.value = kr.layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.proj = kr.layers.Dense(units, kernel_initializer=kernel_init(0.0))
	
	def call(self, inputs):
		batch_size = tf.shape(inputs)[0]
		height = tf.shape(inputs)[1]
		width = tf.shape(inputs)[2]
		scale = tf.cast(self.units, tf.float32) ** (-0.5)
		
		inputs = self.norm(inputs)
		q = self.query(inputs)
		k = self.key(inputs)
		v = self.value(inputs)
		
		attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
		attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
		
		attn_score = tf.nn.softmax(attn_score, -1)
		attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
		
		proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
		proj = self.proj(proj)
		return inputs + proj


class TimeEmbedding(kr.layers.Layer):
	def __init__(self, dim, **kwargs):
		super().__init__(**kwargs)
		self.dim = dim
		self.half_dim = dim // 2
		self.emb = math.log(10000) / (self.half_dim - 1)
		self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * - self.emb)
	
	def call(self, inputs):
		inputs = tf.cast(inputs, dtype=tf.float32)
		emb = inputs[:, None] * self.emb[None, :]
		emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
		return emb



def ResidualBlock(width, groups=8, activation_fn=kr.activations.swish):
	def apply(inputs):
		x, t = inputs
		input_width = x.shape[3]
		
		if input_width == width:
			residual = x
		else:
			residual = kr.layers.Conv2D(
				width, kernel_size=1, kernel_initializer=kernel_init(1.0)
			)(x)
		
		#temb = activation_fn(t)
		temb = kr.layers.Dense(width, kernel_initializer=kernel_init(1.0))(t)[
						:, None, None, :
						]
		
		x = tfa.layers.GroupNormalization(groups=groups)(x)
		x = activation_fn(x)
		x = kr.layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
		)(x)
		
		x = kr.layers.Add()([x, temb])
		x = tfa.layers.GroupNormalization(groups=groups)(x)
		x = activation_fn(x)
		
		x = kr.layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
		)(x)
		x = kr.layers.Add()([x, residual])
		return x
	
	return apply
	

def DownSample(width, activation_fn='relu'):
	def apply(x):
		x = kr.layers.Conv2D(
			width,
			kernel_size=3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_init(1.0),activation=activation_fn
		)(x)
		return x
	
	return apply


def UpSample(width, interpolation="nearest", activation_fn='relu'):
	def apply(x):
		x = kr.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
		x = kr.layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation=activation_fn
		)(x)
		return x
	
	return apply


def TimeMLP(units, activation_fn=kr.activations.swish):
	def apply(inputs):
		temb = kr.layers.Dense(
			units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
		)(inputs)
		temb = kr.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
		return temb
	
	return apply


class ResidualBlockLayer(kr.layers.Layer):
	def __init__(self, width, groups=8, activation_fn=kr.activations.swish, **kwargs):
		super(ResidualBlockLayer, self).__init__(**kwargs)
		self.width = width
		self.groups = groups
		self.activation_fn = activation_fn
	
	def build(self, input_shape):

	
		input_width = input_shape[0][-1]
		
		if input_width == self.width:
			self.residual_layer = kr.layers.Lambda(lambda x: x)
		else:
			self.residual_layer = kr.layers.Conv2D(self.width, kernel_size=1,
												kernel_initializer=kernel_init(0.0))
		
		self.temb_layer = kr.layers.Dense(self.width, kernel_initializer=kernel_init(0.0))
		self.group_norm1_layer = tfa.layers.GroupNormalization(groups=self.groups)
		self.conv1_layer = kr.layers.Conv2D(self.width, kernel_size=3, padding="same",
										 kernel_initializer=kernel_init(0.0))
		self.add_layer = kr.layers.Add()
		self.group_norm2_layer = tfa.layers.GroupNormalization(groups=self.groups)
		self.conv2_layer = kr.layers.Conv2D(self.width, kernel_size=3, padding="same",
										 kernel_initializer=kernel_init(0.0))
		self.add2_layer = kr.layers.Add()
	
	def call(self, inputs):
		x, t = inputs
		residual = self.residual_layer(x)
		t = self.activation_fn(t)
		temb = self.temb_layer(t)[:, None, None, :]

		x = self.group_norm1_layer(x)
		x = self.activation_fn(x)
		x = self.conv1_layer(x)
		
		x = self.add_layer([x, temb])
		x = self.group_norm2_layer(x)
		x = self.activation_fn(x)
		x = self.conv2_layer(x)
		
		return self.add2_layer([x, residual])
	
	def get_config(self):
		config = super(ResidualBlockLayer, self).get_config()
		config.update({'width': self.width,
					   'groups': self.groups,
					   'activation_fn': kr.activations.serialize(self.activation_fn)})
		return config








class SPADE(kr.layers.Layer):
	def __init__(self, filters, flags, **kwargs):
		super().__init__(**kwargs)
		self.epsilon = flags.s_epsilon
		self.conv = kr.layers.Conv2D(128, 3, padding="same", activation="relu")
		self.conv_gamma = kr.layers.Conv2D(filters, flags.s_gamma_filter_size, padding="same")
		self.conv_beta = kr.layers.Conv2D(filters, flags.s_beta_filter_size, padding="same")
	
	def build(self, input_shape):
		self.resize_shape = input_shape[1:3]
	
	def call(self, input_tensor, raw_mask):
		raw_mask = tf.cast(raw_mask, tf.float32)
		# 256, 256, 2
		mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
		# 32, 32, 2
		x = self.conv(mask)
		# 32, 32, 128
		gamma = self.conv_gamma(x)
		# 32, 32, 512
		beta = self.conv_beta(x)
		# 32, 32, 512
		mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
		std = tf.sqrt(var + self.epsilon)
		normalized = (input_tensor - mean) / std

		# 32, 32, 512
		output = gamma * normalized + beta
		return output



class GaussianSampler(kr.layers.Layer):
	def __init__(self, batch_size, latent_dim, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.latent_dim = latent_dim
	
	def call(self, inputs):
		means, variance= inputs
		epsilon = tf.random.normal(
			shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=1.0, seed=1234
		)

		samples = means + tf.exp(0.5 * variance) * epsilon


		return samples
	


class ResidualBlockLayerSpade(kr.layers.Layer):
	def __init__(self, flags, width, groups=8, activation_fn=kr.activations.swish, **kwargs):
		super(ResidualBlockLayerSpade, self).__init__(**kwargs)
		self.width = width
		self.groups = groups
		self.activation_fn = activation_fn
		#self.filters = filters
		self.flags = flags
	
	def build(self, input_shape):
		
		input_width = input_shape[0][-1]

		self.spade_1 = SPADE(input_width, self.flags)
		self.spade_2 = SPADE(input_width, self.flags)

		if input_width == self.width:
			self.residual_layer = kr.layers.Lambda(lambda x: x)
		else:
			self.residual_layer = kr.layers.Conv2D(self.width, kernel_size=1,
												kernel_initializer=kernel_init(0.0))
		
		self.temb_layer = kr.layers.Dense(self.width, kernel_initializer=kernel_init(0.0))
		self.group_norm1_layer = tfa.layers.GroupNormalization(groups=self.groups)
		self.conv1_layer = kr.layers.Conv2D(self.width, kernel_size=3, padding="same",
										 kernel_initializer=kernel_init(0.0))
		self.add_layer = kr.layers.Add()
		self.group_norm2_layer = tfa.layers.GroupNormalization(groups=self.groups)
		self.conv2_layer = kr.layers.Conv2D(self.width, kernel_size=3, padding="same",
										 kernel_initializer=kernel_init(0.0))
		self.add2_layer = kr.layers.Add()
	
	def call(self, inputs):
		x, t, mask = inputs
		residual = self.residual_layer(x)
		t = self.activation_fn(t)
		temb = self.temb_layer(t)[:, None, None, :]
		
		# 32, 32, 256
		x = self.group_norm1_layer(x)
		# 32, 32, 256
		x = self.spade_1(x, mask)
		# 32, 32, 256
		x = self.activation_fn(x)
		# 32, 32, 256
		x = self.conv1_layer(x)
		# 32, 32, 256
		
		x = self.add_layer([x, temb])
		# 1, 32, 32, 256
		x = self.group_norm2_layer(x)
		# 1, 32, 32, 256
		x = self.activation_fn(x)
		# 1, 32, 32, 256

		#x = self.spade_2(x, mask)
		x = self.activation_fn(x)
		x = self.conv2_layer(x)

		return self.add2_layer([x, residual])
	
	def get_config(self):
		config = super(ResidualBlockLayer, self).get_config()
		config.update({'width': self.width,
					   'groups': self.groups,
					   'activation_fn': kr.activations.serialize(self.activation_fn)})
		return config



class GanMonitor(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags, n_samples=1, epoch_interval=5):
		
		self.val_images = next(iter(val_dataset))
		self.source = self.val_images[0]
		self.target = self.val_images[1]
		
		self.n_samples = n_samples
		self.epoch_interval = flags.epoch_interval
		self.losses = {'vgg_loss': [], 'ssim_loss': []} 
		self.hist_path = os.path.join(flags.hist_path, flags.name)
		self.sample_dir = flags.sample_dir + '/' + flags.exp_name
			
		if not os.path.exists(self.hist_path):
			os.makedirs(self.hist_path)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
# 	def sample_data(self):
# 		indices = np.random.permutation(self.source.shape[0])[:self.n_samples]
# 		return self.source[indices], self.target[indices]
		
	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.epoch_interval == 0:
# 			s, t = self.sample_data()
			generated_images = self.model.generate_images(self.source, num_images=self.n_samples)
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((self.source[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.target[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("Ground Truth", fontsize=20)
					ax[2].imshow((np.array(generated_images[row]).squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Generated", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)
				

				self.losses['vgg_loss'].append(logs['vgg_loss']) 
				self.losses['ssim_loss'].append(logs['ssim_loss']) 

				# Plot losses
				plt.figure()
				plt.plot(self.losses['vgg_loss'], label='VGG Loss')
				plt.plot(self.losses['ssim_loss'], label='SSIM Loss')
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.legend()
				plt.title('UVIT Losses')
				plt.savefig(os.path.join(self.hist_path, 'losses.png'))
				plt.close()
								

				for loss in self.losses.keys():
					plt.figure()
					plt.plot(self.losses[loss])
					plt.title(loss)
					plt.savefig(self.hist_path + '/uvit_' +  loss + '.png')
					plt.close()
		
		# if epoch > 50 and epoch % (self.epoch_interval * 2) == 0:
		# 	model_dir = f"saved_models/uvit/{epoch}"
		# 	if not os.path.exists(model_dir):
		# 		os.makedirs(model_dir)
		# 	self.model.save_model(model_dir)


class GanMonitorMask(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags, n_samples=1, epoch_interval=5):
		
		self.val_images = next(iter(val_dataset))
		self.source = self.val_images[0]
		self.target = self.val_images[1]
		self.mask = self.val_images[2]
		
		self.n_samples = n_samples
		self.epoch_interval = flags.epoch_interval
		self.losses = {'vgg_loss': [], 'ssim_loss': []} 
		self.hist_path = os.path.join(flags.hist_path, flags.name)
		self.sample_dir = flags.sample_dir + '/' + flags.exp_name
			
		if not os.path.exists(self.hist_path):
			os.makedirs(self.hist_path)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
# 	def sample_data(self):
# 		indices = np.random.permutation(self.source.shape[0])[:self.n_samples]
# 		return self.source[indices], self.target[indices]
		
	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.epoch_interval == 0:
# 			s, t = self.sample_data()
			generated_images = self.model.generate_images(self.source, self.mask, num_images=self.n_samples)
			print(f"----------------------------\n{generated_images.shape}")
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((self.source[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.target[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("Ground Truth", fontsize=20)
					ax[2].imshow((np.array(generated_images[row]).squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Generated", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)
				

				self.losses['vgg_loss'].append(logs['vgg_loss']) 
				self.losses['ssim_loss'].append(logs['ssim_loss']) 

				# Plot losses
				plt.figure()
				plt.plot(self.losses['vgg_loss'], label='VGG Loss')
				plt.plot(self.losses['ssim_loss'], label='SSIM Loss')
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.legend()
				plt.title('UVIT Losses')
				plt.savefig(os.path.join(self.hist_path, 'losses.png'))
				plt.close()
								

				for loss in self.losses.keys():
					plt.figure()
					plt.plot(self.losses[loss])
					plt.title(loss)
					plt.savefig(self.hist_path + '/uvit_' +  loss + '.png')
					plt.close()
		
		# if epoch > 50 and epoch % (self.epoch_interval * 2) == 0:
		# 	model_dir = f"saved_models/uvit/{epoch}"
		# 	if not os.path.exists(model_dir):
		# 		os.makedirs(model_dir)
		# 	self.model.save_model(model_dir)
					

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
			#self.block.add(InstanceNormalization())

		self.block.add(kr.layers.LeakyReLU(0.2))
	
	def call(self, inputs__):
		return self.block(inputs__)
	


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



class Encoder(kr.Model):
	def __init__(self, flags):
		super().__init__()

		self.flags = flags
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		self.first_conv_channels = self.flags.first_conv_channels
		self.widths = [self.first_conv_channels * mult for mult in self.flags.channel_multiplier]
		self.has_attention = self.flags.has_attention
		self.num_res_blocks = self.flags.num_res_blocks
		self.norm_groups = self.flags.norm_groups
		self.activation_fn=kr.activations.swish
		self.batch_size = flags.batch_size

		#define the layers
		self.conv1 = kr.layers.Conv2D(
			self.first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=kernel_init(1.0),
		)
		self.TM = TimeEmbedding(dim=self.first_conv_channels * 4)

		self.TMLP1 = kr.layers.Dense(
			self.first_conv_channels * 4, activation=self.activation_fn, kernel_initializer=kernel_init(1.0)
		)
		self.TMLP2 = kr.layers.Dense(self.first_conv_channels * 4, kernel_initializer=kernel_init(1.0))


		'''
		self.res = [[]*len(self.widths)]
		self.att = [[]*len(self.widths)]
		for i in range(len(self.widths)):
			print(i,"\t#############################")
			# self.res[i] =[[]*self.num_res_blocks]
			# self.att[i] =[[]*self.num_res_blocks]
			for j in range(self.num_res_blocks):
				self.res[i].append(ResidualBlockLayer(self.widths[i], groups=self.norm_groups, activation_fn=self.activation_fn))
				if self.has_attention[i]:
					self.att[i].append(AttentionBlock(self.widths[i], groups=self.norm_groups))
				else:
					self.att[i].append(False)
		'''
		self.mid_res = [ResidualBlockLayer(self.widths[-1], groups=self.norm_groups, activation_fn=self.activation_fn) for i in range(2)]
		self.mid_att = AttentionBlock(self.widths[-1], groups=self.norm_groups)
		self.flatten = kr.layers.Flatten()
		self.mean_dense = kr.layers.Dense(self.latent_dim)
		self.variance_dense = kr.layers.Dense(self.latent_dim)

		#self.att_1 = [ResidualBlockLayer(self.widths[i], groups=self.norm_groups, activation_fn=self.activation_fn) for i in range(self.widths) if self.has_attention[i]]

		self.res1 = ResidualBlockLayer(self.widths[0], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res2 = ResidualBlockLayer(self.widths[1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res3 = ResidualBlockLayer(self.widths[2], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res4 = ResidualBlockLayer(self.widths[3], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res11 = ResidualBlockLayer(self.widths[0], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res22 = ResidualBlockLayer(self.widths[1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res33 = ResidualBlockLayer(self.widths[2], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res44 = ResidualBlockLayer(self.widths[3], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.att1 = AttentionBlock(self.widths[2], groups=self.norm_groups)
		self.att2 = AttentionBlock(self.widths[3], groups=self.norm_groups)
		self.att11 = AttentionBlock(self.widths[2], groups=self.norm_groups)
		self.att22 = AttentionBlock(self.widths[3], groups=self.norm_groups)

		self.down1 = kr.layers.Conv2D(self.widths[0],
			kernel_size=3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_init(1.0),activation='relu'
		)
		self.down2 = kr.layers.Conv2D(self.widths[1],
			kernel_size=3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_init(1.0),activation='relu'
		)
		self.down3 = kr.layers.Conv2D(self.widths[2],
			kernel_size=3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_init(1.0),activation='relu'
		)
		self.dense_1 = kr.layers.Dense(1000, activation="relu")
		self.dense_2 = kr.layers.Dense(1000, activation="relu")




	def call(self, input, **kwargs):
		
		image_input = input[0]
		time_input = input[1]

		x = self.conv1(image_input)
		
		temb = self.TM(time_input)
		temb = self.TMLP1(temb)
		temb = self.TMLP2(temb)
		
		x = self.res1([x, temb])
		x = self.res11([x, temb])
		x = self.down1(x)
		x = self.res2([x, temb])
		x = self.res22([x, temb])
		x = self.down2(x)
		x = self.res3([x, temb])
		x = self.att1(x)
		x = self.res33([x, temb])
		x = self.att11(x)
		x = self.down3(x)
		x = self.res4([x, temb])
		x = self.att2(x)
		x = self.res44([x, temb])
		x = self.att22(x)


		#skips = [x]
		'''
		# DownBlock
		for i in range(len(self.widths)):
			for j in range(self.num_res_blocks):
				x = self.res[i][j]([x, temb])

				if self.att[i]:
					x = self.att[i][j](x)
				#skips.append(x)
			
			if self.widths[i] != self.widths[-1]:
				x = DownSample(self.widths[i])(x)
				#skips.append(x)
		'''	
		## NA
		# MiddleBlock
		for i in range(len(self.mid_res)):
			x = self.mid_res[i]([x, temb])
			if i ==0:
				x=self.mid_att(x)
	

		# Obtain latent vector for decoder
		x = self.flatten(x)
		mean = self.mean_dense(x)
		variance = self.variance_dense(x)
		# mean_t = self.dense_1(x)
		# variance_t = self.dense_2(x)

		return [mean, variance]
	
	def build_graph(self):
		image_input = kr.layers.Input(shape=self.image_shape, name="image_input")
		time_input = kr.Input(shape=(), dtype=tf.int64, name="time_input")
		return kr.Model(inputs=[image_input, time_input], outputs=self.call([image_input, time_input]))
	

class Decoder(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)

		self.flags = flags 
		self.mask_shape = (flags.crop_size, flags.crop_size, 2)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		self.first_conv_channels = flags.first_conv_channels
		self.widths = [self.first_conv_channels * mult for mult in flags.channel_multiplier]
		self.has_attention = flags.has_attention
		self.num_res_blocks = flags.num_res_blocks
		self.norm_groups = flags.norm_groups
		self.interpolation="nearest"
		self.activation_fn=kr.activations.swish
		self.batch_size = flags.batch_size

		#define the layers
		self.TM = TimeEmbedding(dim=self.first_conv_channels * 4)
		self.TMLP1 = kr.layers.Dense(
			self.first_conv_channels * 4, activation=self.activation_fn, kernel_initializer=kernel_init(1.0)
		)
		self.TMLP2 = kr.layers.Dense(self.first_conv_channels * 4, kernel_initializer=kernel_init(1.0))

		self.dense = kr.layers.Dense(self.latent_dim * 32 * 32)
		self.reshape = kr.layers.Reshape((32, 32, self.latent_dim))

		self.res1 = ResidualBlockLayerSpade(self.flags, self.widths[3], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res2 = ResidualBlockLayerSpade(self.flags, self.widths[2], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res3 = ResidualBlockLayerSpade(self.flags, self.widths[1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res4 = ResidualBlockLayerSpade(self.flags, self.widths[0], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res11 = ResidualBlockLayerSpade(self.flags, self.widths[3], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res22 = ResidualBlockLayerSpade(self.flags, self.widths[2], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res33 = ResidualBlockLayerSpade(self.flags, self.widths[1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res44 = ResidualBlockLayerSpade(self.flags, self.widths[0], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res111 = ResidualBlockLayerSpade(self.flags, self.widths[3], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res222 = ResidualBlockLayerSpade(self.flags, self.widths[2], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res333 = ResidualBlockLayerSpade(self.flags, self.widths[1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.res444 = ResidualBlockLayerSpade(self.flags, self.widths[0], groups=self.norm_groups, activation_fn=self.activation_fn)

		self.att1 = AttentionBlock(self.widths[3], groups=self.norm_groups)
		self.att2 = AttentionBlock(self.widths[2], groups=self.norm_groups)
		self.att11 = AttentionBlock(self.widths[3], groups=self.norm_groups)
		self.att22 = AttentionBlock(self.widths[2], groups=self.norm_groups)
		self.att111 = AttentionBlock(self.widths[3], groups=self.norm_groups)
		self.att222 = AttentionBlock(self.widths[2], groups=self.norm_groups)

		self.up1 = kr.layers.UpSampling2D(size=2, interpolation=self.interpolation)
		self.conv1 = kr.layers.Conv2D(
			self.widths[3], kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation='relu')
		self.up2 = kr.layers.UpSampling2D(size=2, interpolation=self.interpolation)
		self.conv2 = kr.layers.Conv2D(
			self.widths[2], kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation='relu')
		self.up3 = kr.layers.UpSampling2D(size=2, interpolation=self.interpolation)
		self.conv3 = kr.layers.Conv2D(
			self.widths[1], kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation='relu')
		

		self.group_norm = tfa.layers.GroupNormalization(groups=self.norm_groups)
		self.conv = kr.layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))
		self.tanh = kr.layers.Activation('tanh')

	def call(self, inputs, **kwargs):
		latent_vector, time_input, mask_input = inputs

		temb = self.TM(time_input)
		temb = self.TMLP1(temb)
		temb = self.TMLP2(temb)


		x = self.dense(latent_vector)
		x = self.reshape(x)

		x = self.res1([x, temb, mask_input])
		x = self.att1(x)
		x = self.res11([x, temb, mask_input])
		x = self.att11(x)
		x = self.res111([x, temb, mask_input])
		x = self.att111(x)
		x = self.up1(x)
		x = self.res2([x, temb, mask_input])
		x = self.att2(x)
		x = self.res22([x, temb, mask_input])
		x = self.att22(x)
		x = self.res222([x, temb, mask_input])
		x = self.att222(x)
		x = self.up2(x)
		x = self.res3([x, temb, mask_input])
		x = self.res33([x, temb, mask_input])
		x = self.res333([x, temb, mask_input])
		x = self.up3(x)
		x = self.res4([x, temb, mask_input])
		x = self.res44([x, temb, mask_input])
		x = self.res444([x, temb, mask_input])
		
		'''
		for i in reversed(range(len(self.widths))):
			for _ in range(self.num_res_blocks + 1):
				#x = kr.layers.Concatenate(axis=-1)([x, skips.pop()])
				x = ResidualBlockLayerSpade(
					self.flags, self.widths[i], groups=self.norm_groups, activation_fn=self.activation_fn
				)([x, temb, mask_input])
				if self.has_attention[i]:
					x = AttentionBlock(self.widths[i], groups=self.norm_groups)(x)
			
			if i != 0:
				x = UpSample(self.widths[i], interpolation=self.interpolation)(x)
		'''

		# End block
		x = self.group_norm(x)
		x = self.activation_fn(x)
		x = self.conv(x)
		x = self.tanh(x)

		return x

	def build_graph(self):
		latent_vector = kr.layers.Input(shape=self.latent_dim)
		time_input = kr.layers.Input(shape=())
		mask_input = kr.layers.Input(shape=self.mask_shape)
		#skips = kr.layers.Input(shape=skips_shape)
		return kr.Model(inputs=[latent_vector, time_input, mask_input], 
				  outputs=self.call([latent_vector, time_input, mask_input]))