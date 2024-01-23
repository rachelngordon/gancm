import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np 
import matplotlib.pyplot as plt


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
		
		self.norm = kr.layers.GroupNormalization(groups=groups)
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
        
        x = kr.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = kr.layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        
        x = kr.layers.Add()([x, temb])
        x = kr.layers.GroupNormalization(groups=groups)(x)
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
        self.group_norm1_layer = kr.layers.GroupNormalization(groups=self.groups)
        self.conv1_layer = kr.layers.Conv2D(self.width, kernel_size=3, padding="same",
                                         kernel_initializer=kernel_init(0.0))
        self.add_layer = kr.layers.Add()
        self.group_norm2_layer = kr.layers.GroupNormalization(groups=self.groups)
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

