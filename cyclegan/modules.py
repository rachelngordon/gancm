import matplotlib.pyplot as plt
import tensorflow.keras as kr
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import tensorflow_addons as tfa



class Residual(kr.layers.Layer): 
    def __init__(self, 
                 num_channels, 
                 use_1x1conv=False, 
                 strides=1):
        super().__init__()
        gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1234)
        self.conv1 = kr.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides, use_bias=False,
            kernel_initializer = kr.initializers.GlorotNormal())
        self.conv2 = kr.layers.Conv2D(
            num_channels, kernel_size=3, padding='same', use_bias=False,
            kernel_initializer = kr.initializers.GlorotNormal())
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = kr.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides, use_bias=False,
                kernel_initializer = kr.initializers.GlorotNormal())
        self.bn1 = tfa.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)
        self.bn2 = tfa.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)

    def call(self, X):
        Y = kr.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return kr.activations.relu(Y)


class ResidualT(kr.layers.Layer): 
    def __init__(self, 
                 num_channels, 
                 strides=2):
        super().__init__()
        
        gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=123)

        self.conv1 = kr.layers.Conv2DTranspose(
            num_channels, padding='same', kernel_size=4, strides=strides,
            kernel_initializer = kr.initializers.GlorotNormal())
       
        self.bn1 = tfa.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)
     

    def call(self, X):
        return kr.activations.relu(self.bn1(self.conv1(X)))
       

class ResnetBlock(kr.layers.Layer):
    def __init__(self, 
                 num_channels, 
                 num_residuals, 
                 first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                x = Residual(num_channels, use_1x1conv=True, strides=2)
                self.residual_layers.append(x)
            else:
                x = Residual(num_channels)
                self.residual_layers.append(x)

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


class ResnetBlockT(kr.layers.Layer):
    def __init__(self, 
                 num_channels, 
                 num_residuals, 
                 strides=2,
                 **kwargs):
        super(ResnetBlockT, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            x = ResidualT(num_channels, strides=strides)
            self.residual_layers.append(x)

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


class EncoderModule(kr.Model): 
    def __init__(self, 
                 channels, 
                 filter_size,  
                 image_shape,
                 **kwargs):
        
        super().__init__(**kwargs)

        gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=500)

        #define encoder
        self.encoder = kr.models.Sequential()
        self.encoder._name = "Encoder"

        #first conv
        self.b1 = [kr.layers.Conv2D(channels, kernel_size=5, strides=2, padding='same', 
                                    input_shape=image_shape, name="Conv1"),
                    tfa.layers.GroupNormalization(groups=channels, name='BN1', gamma_initializer=gamma_init),
                   #kr.layers.BatchNormalization(name='BN1'),
                   kr.layers.Activation('relu', name='Relu'),
                   kr.layers.MaxPool2D(pool_size=3, strides=2, 
                                       padding='same', name='MaxPool')]
        
        for layer in self.b1:
            self.encoder.add(layer)

        #second block
        b2 = ResnetBlock(channels, 2, first_block=True)
        b2._name ='ResBlock_2'
        self.encoder.add(b2)

        for i in range(1, 4):
            b = ResnetBlock(int(2 ** (i + 6)), 2)
            b._name = f"ResBlock_{i+2}"
            self.encoder.add(b)

    def call(self, inputs__):
        return self.encoder(inputs__)


class DecoderModule(kr.Model): 
    def __init__(self, 
                 channels, 
                 filter_size,  
                 **kwargs):
        
        super().__init__(**kwargs)

        #define encoder
        self.decoder = kr.models.Sequential()
        self.decoder._name = "Decoder"
        self.decoder.add(ResnetBlockT(channels//4, 1))
        self.decoder.add(ResnetBlockT(channels//2, 1))
        self.decoder.add(ResnetBlockT(channels//2, 1))
        self.decoder.add(ResnetBlockT(channels, 1))



    def call(self, inputs__):
        return self.decoder(inputs__)


class cyclegan_generator(kr.Model):
    def __init__(self, flags, **kwargs):
        
        super().__init__(**kwargs)
        e_filter_size = flags.e_filter_size
        e_n_filters = flags.e_n_filters
        d_n_filters = flags.d_n_filters
        d_filter_size = flags.d_filter_size
 
        self.encoder = EncoderModule(channels=e_n_filters, 
                                        filter_size=e_filter_size,
                                        image_shape=(flags.crop_size, flags.crop_size, 1))
        
        self.decoder = DecoderModule(channels=d_n_filters, 
                                        filter_size=d_filter_size)
        
        self.convT = kr.layers.Conv2DTranspose(1, 
                                               5, 
                                               strides=2, 
                                               padding='same')
        

    def call(self, input__):
        return kr.activations.tanh(self.convT(self.decoder(self.encoder(input__))))


#This Discriminator is different from the p2p
class Discriminator(kr.Model):
	def __init__(self, flags, **kwargs):
		super().__init__(**kwargs)
		self.image_shape = (flags.crop_size, flags.crop_size, 1)
		n_filters = flags.disc_n_filters
		filter_size = flags.disc_filter_size
		self.merged = kr.layers.Concatenate()
		self.downsample1 = Residual(n_filters, use_1x1conv=True, strides=2)
		self.downsample2 = ResnetBlock(2*n_filters, 1)
		self.downsample3 = ResnetBlock(4*n_filters, 1)
		self.downsample4 = ResnetBlock(8*n_filters, 1)
		self.conv = kr.layers.Conv2D(self.image_shape[-1], kernel_size=filter_size, padding='same')
        
	
	def call(self, inputs_, **kwargs):
		x = self.merged([inputs_[0], inputs_[1]])
		x = self.downsample1(x)
		x = self.downsample2(x)
		x = self.downsample3(x)
		x = self.downsample4(x)
		return self.conv(x)

        


class CycleMonitor(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags, my_strategy=False):
		self.val_images = next(iter(val_dataset))
		self.n_samples = 3
		self.epoch_interval = flags.epoch_interval
		self.checkpoints_path = os.path.join(flags.checkpoints_dir, flags.name)
		self.sample_dir = os.path.join(flags.sample_dir, flags.name)

		if not os.path.exists(self.checkpoints_path):
			os.makedirs(self.checkpoints_path)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)

	def infer(self):
		return self.model(self.val_images[0])

	def on_epoch_end(self, epoch, logs=None):
		if epoch > 0 and epoch % self.epoch_interval == 0:
			#self.save_models()
			generated_images = self.infer()
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
					ax[2].imshow((generated_images[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Generated", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)
				#plt.show()


