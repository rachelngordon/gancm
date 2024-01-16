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
                       'activation_fn': tf.keras.activations.serialize(self.activation_fn)})
        return config
	
	
class GanMonitor(keras.callbacks.Callback):
	def __init__(self, source, target, n_samples=3, epoch_interval=5):
		
		self.source = source
		self.target = target
		self.n_samples = n_samples
		self.epoch_interval = epoch_interval
		self.sample_dir = "samples_figures"
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
	def sample_data(self):
		indices = np.random.permutation(self.source.shape[0])[:self.n_samples]
		return self.source[indices], self.target[indices]
        
	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.epoch_interval == 0:
			s, t = self.sample_data()
			generated_images = self.model.generate_images(s, num_images=self.n_samples)
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((s[row].squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((t[row].squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("Ground Truth", fontsize=20)
					ax[2].imshow((np.array(generated_images[row]).squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Generated", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)
		
		if epoch > 50 and epoch % (self.epoch_interval * 2) == 0:
			model_dir = f"saved_models/uvit/{epoch}"
			if not os.path.exists(model_dir):
				os.makedirs(model_dir)
			self.model.save_model(model_dir)

