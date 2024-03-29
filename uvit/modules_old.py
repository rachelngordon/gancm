import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np 
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa

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
	

class DownSample(kr.layers.Layer):
        def __init__(self, width, activation_fn='relu', **kwargs):
            super().__init__(**kwargs)
            self.conv = kr.layers.Conv2D(
                        width,
                        kernel_size=3,
                        strides=2,
                        padding="same",
                        kernel_initializer=kernel_init(1.0),activation=activation_fn)
                    
        def call(self, input):
              return self.conv(input)


class UpSample(kr.layers.Layer):
        def __init__(self, width, interpolation="nearest", activation_fn='relu', **kwargs):
            super().__init__(**kwargs)

            self.upsample = kr.layers.UpSampling2D(size=2, interpolation=interpolation)
            self.conv = kr.layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0),activation=activation_fn)

        def call(self, input):
              x = self.upsample(input)
              return self.conv(x)


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


class DownBlock(kr.layers.Layer):
    def __init__(self, widths, norm_groups, activation_fn, num_res_blocks, has_attention, **kwargs):
        super().__init__(**kwargs)

        self.down_layers = []

        # DownBlock
        for i in range(len(widths)):
            for _ in range(num_res_blocks):
                self.down_layers.append(ResidualBlockLayer(
                    widths[i], groups=norm_groups, activation_fn=activation_fn
                ))
                if has_attention[i]:
                    self.down_layers.append(AttentionBlock(widths[i], groups=norm_groups))
                    

            if widths[i] != widths[-1]:
                self.down_layers.append(DownSample(widths[i]))

				
    def call(self, inputs__):
        x, temb = inputs__
        skips = [x]


        for i in range(len(self.down_layers)):
            
            if isinstance(self.down_layers[i], ResidualBlockLayer):
                x = self.down_layers[i]([x, temb])

                if not isinstance(self.down_layers[i+1], AttentionBlock):
                    skips.append(x)
                    
            else:
                x = self.down_layers[i](x)
                skips.append(x)

            
        return x, skips
    

class UpBlock(kr.layers.Layer):
    def __init__(self, widths, norm_groups, activation_fn, num_res_blocks, has_attention, interpolation="nearest", **kwargs):
        super().__init__(**kwargs)
        self.up_layers = []

        # UpBlock
        for i in reversed(range(len(widths))):
            for _ in range(num_res_blocks + 1):
                self.up_layers.append(kr.layers.Concatenate(axis=-1))
                self.up_layers.append((ResidualBlockLayer(
					widths[i], groups=norm_groups, activation_fn=activation_fn
				)))
                if has_attention[i]:
                    self.up_layers.append((AttentionBlock(widths[i], groups=norm_groups)))
			
            if i != 0:
                self.up_layers.append((UpSample(widths[i], interpolation=interpolation)))
		
				
    def call(self, inputs__, skips):
        x, temb = inputs__
        for layer in self.up_layers:
            
            if isinstance(layer, kr.layers.Concatenate):
                x = layer([x, skips.pop()])
                
            elif isinstance(layer, ResidualBlockLayer):
                x = layer([x, temb])
                
            else:
                x = layer(x)

        return x

class uvit_generator(kr.Model):

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
		self.activation_fn=kr.activations.swish
		self.conv = kr.layers.Conv2D(
			self.first_conv_channels,
			kernel_size=(3, 3),
			padding="same",
			kernel_initializer=kernel_init(1.0),
		)
		self.temb_layer = TimeEmbedding(dim=self.first_conv_channels * 4)
		
		# TimeMLP Layer
		self.tmlp1 = kr.layers.Dense(self.first_conv_channels * 4, activation=self.activation_fn, kernel_initializer=kernel_init(1.0))
		self.tmlp2 = kr.layers.Dense(self.first_conv_channels * 4, kernel_initializer=kernel_init(1.0))

		# Middle Block Layers
		self.res_block = ResidualBlockLayer(self.widths[-1], groups=self.norm_groups, activation_fn=self.activation_fn)
		self.attention = AttentionBlock(self.widths[-1], groups=self.norm_groups)

		self.down_block = DownBlock(self.widths, self.norm_groups, self.activation_fn, self.num_res_blocks, self.has_attention)
		self.up_block = UpBlock(self.widths, self.norm_groups, self.activation_fn, self.num_res_blocks, self.has_attention)
      
		# End Block Layers
		self.group_norm = kr.layers.GroupNormalization(groups=self.norm_groups)
		self.conv1 = kr.layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))
		self.activation = kr.layers.Activation('tanh')


	# def TimeMLP(self, units, activation_fn=kr.activations.swish):
	# 	def apply(inputs):
	# 		temb = kr.layers.Dense(
	# 			units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
	# 		)(inputs)
	# 		temb = kr.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
	# 		return temb
		
	# 	return apply


	def call(self, inputs__):
		
		image_input, time_input = inputs__

		x = self.conv(image_input)

		temb = self.temb_layer(time_input)
		temb1 = self.tmlp1(temb)
		temb2 = self.tmlp1(temb1)

		skips = [x]
            
        # Down Block
		x_down, skips = self.down_block([x, temb2])
            
        # Middle Block
		x = self.res_block([x_down, temb2])
		x = self.attention(x)
		x_mid = self.res_block([x, temb2])
            
		# Up Block
		x_up = self.up_block([x_mid, temb2], skips)
            
		# End block
		x = self.group_norm(x_up)
		x = self.activation_fn(x)
		x_end = self.conv1(x)

		return self.activation(x_end)
	

	def build_graph(self):
		image_input = kr.layers.Input(
			shape=(self.img_size, self.img_size, self.img_channels), name="image_input"
		)
		time_input = kr.layers.Input(shape=(), dtype=tf.int64, name="time_input")
		return kr.Model(inputs=[image_input, time_input], outputs=self.call([image_input, time_input]))



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
	def __init__(self, val_dataset, flags, n_samples=1, epoch_interval=5):
		
		self.val_images = next(iter(val_dataset))
		self.source = self.val_images[0]
		self.target = self.val_images[1]
        
		self.timesteps = flags.timesteps
		self.t = tf.random.uniform(
			minval=0, maxval=self.timesteps, shape=(flags.batch_size,), dtype=tf.int64
		)
        
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
			# s, t = self.sample_data()
			generated_images = self.model([self.source, self.t])
			for s_ in range(self.n_samples):
				grid_row = min(generated_images.shape[0], 3)
				#grid_row = min(len(generated_images), 3)
				f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
				for row in range(grid_row):
					ax = axarr if grid_row == 1 else axarr[row]
					ax[0].imshow((self.source[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[0].axis("off")
					ax[0].set_title("CT", fontsize=20)
					ax[1].imshow((self.target[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[1].axis("off")
					ax[1].set_title("Ground Truth", fontsize=20)
					ax[2].imshow((tf.squeeze(generated_images[row]).numpy() + 1) / 2, cmap='gray')
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

