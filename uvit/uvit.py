import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
from . import modules_uvit as modules


class UNetViTModel(keras.Model):
	def __init__(self, timesteps, learning_rate,
							 img_size, img_channels, widths,
							 has_attention, num_res_blocks=2,
							 norm_groups=8):
		super().__init__()
		self.timesteps = timesteps
		self.loss = keras.losses.MeanSquaredError()
		self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
		self.network = self.build_model(img_size,
							img_channels,
							widths,
							has_attention,
							num_res_blocks,
							norm_groups)
		self.network.compile(optimizer=self.optimizer, loss=self.loss)
	
	def ResidualBlock(self, width, groups=8, activation_fn=keras.activations.swish):
		def apply(inputs):
			x, t = inputs
			input_width = x.shape[3]
			
			if input_width == width:
				residual = x
			else:
				residual = layers.Conv2D(
					width, kernel_size=1, kernel_initializer=modules.kernel_init(1.0)
				)(x)
			
			#temb = activation_fn(t)
			temb = layers.Dense(width, kernel_initializer=modules.kernel_init(1.0))(t)[
						 :, None, None, :
						 ]
			
			x = layers.GroupNormalization(groups=groups)(x)
			x = activation_fn(x)
			x = layers.Conv2D(
				width, kernel_size=3, padding="same", kernel_initializer=modules.kernel_init(1.0)
			)(x)
			
			x = layers.Add()([x, temb])
			x = layers.GroupNormalization(groups=groups)(x)
			x = activation_fn(x)
			
			x = layers.Conv2D(
				width, kernel_size=3, padding="same", kernel_initializer=modules.kernel_init(0.0)
			)(x)
			x = layers.Add()([x, residual])
			return x
		
		return apply
	
	def DownSample(self, width, activation_fn='relu'):
		def apply(x):
			x = layers.Conv2D(
				width,
				kernel_size=3,
				strides=2,
				padding="same",
				kernel_initializer=modules.kernel_init(1.0),activation=activation_fn
			)(x)
			return x
		
		return apply
	
	def UpSample(self, width, interpolation="nearest", activation_fn='relu'):
		def apply(x):
			x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
			x = layers.Conv2D(
				width, kernel_size=3, padding="same", kernel_initializer=modules.kernel_init(1.0),activation=activation_fn
			)(x)
			return x
		
		return apply
	
	def TimeMLP(self, units, activation_fn=keras.activations.swish):
		def apply(inputs):
			temb = layers.Dense(
				units, activation=activation_fn, kernel_initializer=modules.kernel_init(1.0)
			)(inputs)
			temb = layers.Dense(units, kernel_initializer=modules.kernel_init(1.0))(temb)
			return temb
		
		return apply
	
	def build_model(
			self,
			img_size,
			img_channels,
			widths,
			has_attention,
			num_res_blocks=2,
			norm_groups=8,
			interpolation="nearest",
			activation_fn=keras.activations.swish,
			first_conv_channels = 32,
	):
		image_input = layers.Input(
			shape=(img_size, img_size, img_channels), name="image_input"
		)
		time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
		
		x = layers.Conv2D(
			first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=modules.kernel_init(1.0),
		)(image_input)
		
		temb = modules.TimeEmbedding(dim=first_conv_channels * 4)(time_input)
		temb = self.TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
		
		skips = [x]
		
		# DownBlock
		for i in range(len(widths)):
			for _ in range(num_res_blocks):
				x = modules.ResidualBlockLayer(
					widths[i], groups=norm_groups, activation_fn=activation_fn
				)([x, temb])
				if has_attention[i]:
					x = modules.AttentionBlock(widths[i], groups=norm_groups)(x)
				skips.append(x)
			
			if widths[i] != widths[-1]:
				x = self.DownSample(widths[i])(x)
				skips.append(x)
		
		# MiddleBlock
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		x = modules.AttentionBlock(widths[-1], groups=norm_groups)(x)
		x = modules.ResidualBlockLayer(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
			[x, temb]
		)
		
		# UpBlock
		for i in reversed(range(len(widths))):
			for _ in range(num_res_blocks + 1):
				x = layers.Concatenate(axis=-1)([x, skips.pop()])
				x = modules.ResidualBlockLayer(
					widths[i], groups=norm_groups, activation_fn=activation_fn
				)([x, temb])
				if has_attention[i]:
					x = modules.AttentionBlock(widths[i], groups=norm_groups)(x)
			
			if i != 0:
				x = self.UpSample(widths[i], interpolation=interpolation)(x)
		
		# End block
		x = layers.GroupNormalization(groups=norm_groups)(x)
		x = activation_fn(x)
		x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=modules.kernel_init(0.0))(x)
		x = layers.Activation('tanh')(x)
		return keras.Model(inputs=[image_input, time_input], outputs=x, name="unetvit")
	
	def compile(self, **kwargs):
		super().compile(**kwargs)
	
	def call(self, inputs):
		image_input, time_input = inputs
		return self.network([image_input, time_input])
	
	def save_model(self, path):
		self.network.save(path)
	
	def train_step(self, data):
		source, target = data
		batch_size = tf.shape(source)[0]
		
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		with tf.GradientTape() as tape:
			pred_ = self.network([source, t], training=True)
			loss = self.loss(target, pred_)
		
		gradients = tape.gradient(loss, self.network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
		return {"loss": loss}
	
	def test_step(self, data):
		source, target = data
		batch_size = tf.shape(source)[0]
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
		)
		
		pred_ = self.network([source, t])
		loss = self.loss(target, pred_)
		
		return {"loss": loss}
	
	def generate_images(self, source, num_images=8):
		t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(num_images,), dtype=tf.int64
		)
		
		return self.network([source[:num_images], t])
	
	def plot_images(
			self, source, target, logs=None, num_rows=2, num_cols=4, figsize=(12, 5)
	):
		
		generated_samples = self.generate_images(source, num_images=num_rows * num_cols)
		
		_, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
		for i, image in enumerate(generated_samples.squeeze()):
			if num_rows == 1:
				ax[i].imshow(image, cmap='gray')
				ax[i].axis("off")
			else:
				ax[i // num_cols, i % num_cols].imshow(image)
				ax[i // num_cols, i % num_cols].axis("off")
		
		plt.tight_layout()
		plt.show()
