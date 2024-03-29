import matplotlib.pyplot as plt
import tensorflow.keras as kr
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
#import tensorflow_addons as tfa



class Residual(kr.layers.Layer): 
    def __init__(self, 
                 num_channels, 
                 use_1x1conv=False, 
                 strides=1):
        super().__init__()
				
        gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=1234)
        self.conv1 = kr.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides,
					use_bias=False, kernel_initializer=kr.initializers.GlorotNormal())
        self.conv2 = kr.layers.Conv2D(
            num_channels, kernel_size=3, padding='same', use_bias=False,
            kernel_initializer = kr.initializers.GlorotNormal())
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = kr.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides,use_bias=False,
                kernel_initializer = kr.initializers.GlorotNormal())
			
        self.bn1 = kr.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)
        self.bn2 = kr.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)

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
            num_channels, padding='same', kernel_size=3, strides=strides,use_bias=False,
            kernel_initializer = kr.initializers.GlorotNormal())
       
        self.bn1 = kr.layers.GroupNormalization(groups=num_channels, gamma_initializer=gamma_init)
     

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
                 **kwargs):
        super(ResnetBlockT, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            x = ResidualT(num_channels, strides=2)
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

        #define encoder
        self.encoder = kr.models.Sequential()
        self.encoder._name = "Encoder"
        gamma_init = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=500)

        #first conv
        self.b1 = [kr.layers.Conv2D(channels, kernel_size=4, strides=2, padding='same',
																		use_bias=False, input_shape=image_shape, name="Conv1", activation='relu'),
									 kr.layers.Conv2D(channels, kernel_size=4, strides=2, padding='same',
																		use_bias=False,input_shape=image_shape, name="Conv2"),
                   kr.layers.GroupNormalization(groups=channels, gamma_initializer=gamma_init, name='IN1'),
                   kr.layers.Activation('relu', name='Relu')
                   ]
        
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

        #define decoder
        self.decoder = kr.models.Sequential()
        self.decoder._name = "Decoder"
        self.decoder.add(ResnetBlockT(channels, 1))
        self.decoder.add(ResnetBlockT(channels//2, 1))
        self.decoder.add(ResnetBlockT(channels//2, 1))
        self.decoder.add(ResnetBlockT(channels//4, 1))


    def call(self, inputs__):
        return self.decoder(inputs__)


class p2p_generator(kr.Model):
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
                                               4,
                                               strides=2, 
                                               padding='same')
        

    def call(self, input__):
        return kr.activations.tanh(self.convT(self.decoder(self.encoder(input__))))


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
		#print(f"{'=='.join(['#' for i in range (10)])}\n{inputs_[0].shape},\n {inputs_[1].shape}")
		x = self.merged([inputs_[0], inputs_[1]])
		x1 = self.downsample1(x)
		x2 = self.downsample2(x1)
		x3 = self.downsample3(x2)
		x4 = self.downsample4(x3)
		x5 = self.conv(x4)
		return [x1, x2, x3, x4, x5]


class P2PMonitor(kr.callbacks.Callback):
	def __init__(self, val_dataset, flags, my_strategy=False):

		self.val_images = next(iter(val_dataset))
		self.n_samples = 1
		self.epoch_interval = flags.epoch_interval
		self.checkpoints_path = os.path.join(flags.checkpoints_dir, flags.name)
		self.hist_path = os.path.join(flags.hist_path, flags.name)
		self.sample_dir = os.path.join(flags.sample_dir, flags.name)
		self.losses = {'disc_loss': [], 'vgg_loss': [], 'ssim_loss': []} 

		if not os.path.exists(self.checkpoints_path):
			os.makedirs(self.checkpoints_path)
		if not os.path.exists(self.sample_dir):
			os.makedirs(self.sample_dir)
		if not os.path.exists(self.hist_path):
			os.makedirs(self.hist_path)

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
					ax[1].set_title("rMRI", fontsize=20)
					ax[2].imshow((generated_images[row].numpy().squeeze() + 1) / 2, cmap='gray')
					ax[2].axis("off")
					ax[2].set_title("Pix2Pix sMRI", fontsize=20)
				filename = "sample_{}_{}_{}.png".format(epoch, s_, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
				sample_file = os.path.join(self.sample_dir, filename)
				plt.savefig(sample_file)
				#plt.show()


				self.losses['disc_loss'].append(logs['disc_loss']) 
				self.losses['vgg_loss'].append(logs['vgg_loss']) 
				self.losses['ssim_loss'].append(logs['ssim_loss']) 

				# Plot losses
				plt.figure()
				plt.plot(self.losses['disc_loss'], label='Discriminator Loss')
				plt.plot(self.losses['vgg_loss'], label='VGG Loss')
				plt.plot(self.losses['ssim_loss'], label='SSIM Loss')
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.legend()
				plt.title('Pix2Pix Losses')
				plt.savefig(os.path.join(self.hist_path, 'losses.png'))
				plt.close()
                                

				for loss in self.losses.keys():
					plt.figure()
					plt.plot(self.losses[loss])
					plt.title(loss)
					plt.savefig(self.hist_path + '/pix2pix_' +  loss + '.png')
					plt.close()


