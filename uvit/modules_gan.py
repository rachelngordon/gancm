import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime


# Kernel initializer to use
def kernel_init(scale):
	scale = max(scale, 1e-10)
	return keras.initializers.VarianceScaling(
		scale, mode="fan_avg", distribution="uniform"
	)


class AttentionBlock(layers.Layer):
	"""Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """
	
	def __init__(self, units, groups=8, **kwargs):
		self.units = units
		self.groups = groups
		super().__init__(**kwargs)
		
		self.norm = layers.GroupNormalization(groups=groups)
		self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
		self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))
	
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


class TimeEmbedding(layers.Layer):
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
	

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
	def apply(inputs):
		x, t = inputs
		input_width = x.shape[3]
		
		if input_width == width:
			residual = x
		else:
			residual = layers.Conv2D(
				width, kernel_size=1, kernel_initializer=kernel_init(1.0)
			)(x)
		
		#temb = activation_fn(t)
		temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(t)[
						:, None, None, :
						]
		
		x = layers.GroupNormalization(groups=groups)(x)
		x = activation_fn(x)
		x = layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
		)(x)
		
		x = layers.Add()([x, temb])
		x = layers.GroupNormalization(groups=groups)(x)
		x = activation_fn(x)
		
		x = layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
		)(x)
		x = layers.Add()([x, residual])
		return x
	
	return apply

	
	


class ResidualBlockLayer(layers.Layer):
    def __init__(self, width, groups=8, activation_fn=tf.keras.activations.swish, **kwargs):
        super(ResidualBlockLayer, self).__init__(**kwargs)
        self.width = width
        self.groups = groups
        self.activation_fn = activation_fn
    
    def build(self, input_shape):
        input_width = input_shape[0][-1]
        
        if input_width == self.width:
            self.residual_layer = layers.Lambda(lambda x: x)
        else:
            self.residual_layer = layers.Conv2D(self.width, kernel_size=1,
                                                kernel_initializer=kernel_init(0.0))
        
        self.temb_layer = layers.Dense(self.width, kernel_initializer=kernel_init(0.0))
        self.group_norm1_layer = layers.GroupNormalization(groups=self.groups)
        self.conv1_layer = layers.Conv2D(self.width, kernel_size=3, padding="same",
                                         kernel_initializer=kernel_init(0.0))
        self.add_layer = layers.Add()
        self.group_norm2_layer = layers.GroupNormalization(groups=self.groups)
        self.conv2_layer = layers.Conv2D(self.width, kernel_size=3, padding="same",
                                         kernel_initializer=kernel_init(0.0))
        self.add2_layer = layers.Add()
    
    def call(self, inputs):
        x, t = inputs
        residual = self.residual_layer(x)
        t = self.activation_fn(t)
        temb_t = self.temb_layer(t)[:, None, None, :]
        
        x = self.group_norm1_layer(x)
        x = self.activation_fn(x)
        x = self.conv1_layer(x)
        
        x = self.add_layer([x, temb_t])
        x = self.group_norm2_layer(x)
        x = self.activation_fn(x)
        x = self.conv2_layer(x)
        
        return self.add2_layer([x, residual])
    
    def get_config(self):
        config = super(ResidualBlockLayer, self).get_config()
        config.update({'width': self.width,
                       'groups': self.groups,
                       'activation_fn': tf.keras.activations.serialize(self.activation_fn)})
        return config
	

class uvit_generator(keras.Model):

	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)

		self.flags = flags
		self.img_size = self.flags.crop_size
		self.img_channels = self.flags.img_channels
		self.first_conv_channels = self.flags.first_conv_channels
		self.has_attention = self.flags.has_attention
		self.num_res_blocks = self.flags.num_res_blocks
		self.norm_groups = self.flags.norm_groups
		self.widths = [self.first_conv_channels * mult for mult in self.flags.channel_multiplier]
		self.interpolation="nearest"
		self.activation_fn=keras.activations.swish
		self.conv = layers.Conv2D(
			self.first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=kernel_init(1.0),
		)
		self.temb_layer = TimeEmbedding(dim=self.first_conv_channels * 4)
		self.tmlp1 = layers.Dense(self.first_conv_channels * 4, activation=self.activation_fn, kernel_initializer=kernel_init(1.0))
		self.tmlp2 = layers.Dense(self.first_conv_channels * 4, kernel_initializer=kernel_init(1.0))
		self.group_norm = layers.GroupNormalization(groups=self.norm_groups)
		self.conv1 = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))
		self.activation = layers.Activation('tanh')

	def TimeMLP(self, units, activation_fn, inputs):

		temb = layers.Dense(
			units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
		)(inputs)
		temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)

		return temb
	
	def DownSample(self, width, x, activation_fn='relu'):

		x = layers.Conv2D(
			width,
			kernel_size=3,
			strides=2,
			padding="same",
			kernel_initializer=kernel_init(1.0),activation=activation_fn
		)(x)

		return x
	

	

	def UpSample(self, width, x, interpolation="nearest", activation_fn='relu'):

		x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
		x = layers.Conv2D(
			width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation=activation_fn
		)(x)
		return x
	

	def DownBlock(self, x, temb_x):

		skips = [x]

		# DownBlock
		for i in range(len(self.widths)):
			for _ in range(self.num_res_blocks):
				x = ResidualBlockLayer(
					self.widths[i], groups=self.norm_groups, activation_fn=self.activation_fn
				)([x, temb_x])
				if self.has_attention[i]:
					x = AttentionBlock(self.widths[i], groups=self.norm_groups)(x)
				skips.append(x)
			
			if self.widths[i] != self.widths[-1]:
				x = self.DownSample(self.widths[i], x)
				skips.append(x)
		
		return x, skips
	
	def MiddleBlock(self, x, temb_x):

		# MiddleBlock
		x = ResidualBlockLayer(self.widths[-1], groups=self.norm_groups, activation_fn=self.activation_fn)(
			[x, temb_x]
		)
		x = AttentionBlock(self.widths[-1], groups=self.norm_groups)(x)
		x = ResidualBlockLayer(self.widths[-1], groups=self.norm_groups, activation_fn=self.activation_fn)(
			[x, temb_x]
		)

		return x
	
	def UpBlock(self, x, temb_x, skips):

		# UpBlock
		for i in reversed(range(len(self.widths))):
			for _ in range(self.num_res_blocks + 1):
				x = layers.Concatenate(axis=-1)([x, skips.pop()])
				x = ResidualBlockLayer(
					self.widths[i], groups=self.norm_groups, activation_fn=self.activation_fn
				)([x, temb_x])
				if self.has_attention[i]:
					x = AttentionBlock(self.widths[i], groups=self.norm_groups)(x)
			
			if i != 0:
				x = self.UpSample(self.widths[i], x, interpolation=self.interpolation)

	def call(self, inputs__):
		
		image_input, time_input = inputs__

		x = self.conv(image_input)
		
		temb_x = self.temb_layer(time_input)
		temb_x1 = self.TimeMLP(units=self.first_conv_channels * 4, activation_fn=self.activation_fn, inputs=temb_x)
		
		x, skips = self.DownBlock(x, temb_x1)
		
		x = self.MiddleBlock(x, temb_x1)

		x = self.UpBlock(x, temb_x1, skips)

	
		# End block
		x = self.group_norm(x)
		x = self.activation_fn(x)
		x = self.conv1(x)

		return self.activation(x)
	
	def build_graph(self):
		image_input = layers.Input(
			shape=(self.img_size, self.img_size, self.img_channels), name="image_input"
		)
		time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
		return keras.Model(inputs=[image_input, time_input], outputs=self.call([image_input, time_input]))



class DownsampleModule(layers.Layer):
	def __init__(self, channels, filter_size, apply_norm=True, **kwargs):
		super().__init__(**kwargs)
		gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1234)
		self.block = keras.Sequential()
		self.strides = 2
		self.block.add(
			layers.Conv2D(
				channels,
				filter_size,
				strides=self.strides,
				padding="same",
				use_bias=False
			)
		)

		if apply_norm:
			self.block.add(layers.GroupNormalization(groups=channels, gamma_initializer=gamma_init))
			#self.block.add(InstanceNormalization())

		self.block.add(layers.LeakyReLU(0.2))
	
	def call(self, inputs__):
		return self.block(inputs__)
	


class Discriminator(keras.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		self.latent_dim = flags.latent_dim
		n_filters = flags.disc_n_filters
		filter_size = flags.disc_filter_size
		self.merged = layers.Concatenate()
		self.downsample1 = DownsampleModule(n_filters, filter_size, apply_norm=False)
		self.downsample2 = DownsampleModule(2 * n_filters, filter_size)
		self.downsample3 = DownsampleModule(4 * n_filters, filter_size)
		self.downsample4 = DownsampleModule(8 * n_filters, filter_size)
		self.conv = layers.Conv2D(self.image_shape[-1], kernel_size=filter_size, padding='same')
	
	def call(self, inputs_, **kwargs):
		x = self.merged([inputs_[0], inputs_[1]])
		x1 = self.downsample1(x)
		x2 = self.downsample2(x1)
		x3 = self.downsample3(x2)
		x4 = self.downsample4(x3)
		x5 = self.conv(x4)
		return [x1, x2, x3, x4, x5]
	
	def build_graph(self):
		x1 = layers.Input(shape=self.image_shape)
		x2 = layers.Input(shape=self.image_shape)
		return keras.Model(inputs=[x1, x2], outputs=self.call([x1, x2]))



class GanMonitor(keras.callbacks.Callback):
	def __init__(self, val_dataset, flags, n_samples=3):
		
		self.flags = flags
		self.val_images = next(iter(val_dataset))
		self.n_samples = n_samples
		self.epoch_interval = flags.epoch_interval
		self.sample_dir = flags.sample_dir
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)

			
	# def sample_data(self):
	# 	indices = np.random.permutation(self.source.shape[0])[:self.n_samples]
	# 	return self.source[indices], self.target[indices]
	

	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.epoch_interval == 0:
			generated_images = self.model.generate_images(self.val_images[0], num_images=self.n_samples)
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((self.val_images[0][row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.val_images[1][row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("Ground Truth", fontsize=20)
					ax[2].imshow((np.array(generated_images[row]).squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Generated", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file_path = self.sample_dir + f"{self.flags.exp_name}/"
				if not os.path.exists(sample_file_path):
					os.makedirs(sample_file_path)
				sample_file = os.path.join(sample_file_path, filename)
				plt.savefig(sample_file)
		
		if epoch > 50 and epoch % (self.epoch_interval * 2) == 0:
			model_dir = self.flags.model_path + self.flags.exp_name + f'/{epoch}'
			if not os.path.exists(model_dir):
				os.makedirs(model_dir)
			self.model.save_model(model_dir)

