import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime



class downsample(tf.keras.layers.Layer):
	def __init__(self, filters, size, apply_batchnorm=True, **kwargs):
		super().__init__(**kwargs)
		initializer = tf.random_normal_initializer(0., 0.02)

		self.block = tf.keras.Sequential()
    
		self.block.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))

		if apply_batchnorm:
			self.block.add(tf.keras.layers.BatchNormalization())

		self.block.add(tf.keras.layers.LeakyReLU())
  
	
	def call(self, inputs__):
		return self.block(inputs__)
    


class upsample(tf.keras.layers.Layer):
	def __init__(self, filters, size, apply_dropout=False, **kwargs):
		super().__init__(**kwargs)
		initializer = tf.random_normal_initializer(0., 0.02)

		self.block = tf.keras.Sequential()
		self.block.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

		self.block.add(tf.keras.layers.BatchNormalization())

		if apply_dropout:
			self.block.add(tf.keras.layers.Dropout(0.5))

		self.block.add(tf.keras.layers.ReLU())
	
	def call(self, inputs__):
		return self.block(inputs__)



class Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.down_stack = [
          downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
          downsample(128, 4),  # (batch_size, 64, 64, 128)
          downsample(256, 4),  # (batch_size, 32, 32, 256)
          downsample(512, 4),  # (batch_size, 16, 16, 512)
          downsample(512, 4),  # (batch_size, 8, 8, 512)
          downsample(512, 4),  # (batch_size, 4, 4, 512)
          downsample(512, 4),  # (batch_size, 2, 2, 512)
          downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        self.up_stack = [
          upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
          upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
          upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
          upsample(512, 4),  # (batch_size, 16, 16, 1024)
          upsample(256, 4),  # (batch_size, 32, 32, 512)
          upsample(128, 4),  # (batch_size, 64, 64, 256)
          upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(1, 4,
                                              strides=2,
                                              padding='same',
                                              kernel_initializer=initializer,
                                              activation='tanh')  # (batch_size, 256, 256, 3)

        

    def call(self, x):
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
          x = down(x)
          skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
          x = up(x)
          x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last(x)

        return x



class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.merged = tf.keras.layers.Concatenate()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = downsample(64, 4, False)  # (batch_size, 128, 128, 64)
        self.down2 = downsample(128, 4)  # (batch_size, 64, 64, 128)
        self.down3 = downsample(256, 4) # (batch_size, 32, 32, 256)

        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()  # (batch_size, 34, 34, 256)
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)  # (batch_size, 31, 31, 512)

        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()  # (batch_size, 33, 33, 512)

        self.last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)  # (batch_size, 30, 30, 1)

        

    def call(self, inputs__):
        
        x = self.merged([inputs__[0], inputs__[1]])  # (batch_size, 256, 256, channels*2)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.zero_pad1(x3)
        x5 = self.conv(x4)
        x6 = self.batchnorm1(x5)
        x7 = self.leaky_relu(x6)
        x8 = self.zero_pad2(x7)
        x9 = self.last(x8)
        return x9
    

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
					ax[2].set_title("UNet sMRI", fontsize=20)
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
				plt.title('UNet Losses')
				plt.savefig(os.path.join(self.hist_path, 'losses.png'))
				plt.close()
                                

				for loss in self.losses.keys():
					plt.figure()
					plt.plot(self.losses[loss])
					plt.title(loss)
					plt.savefig(self.hist_path + '/unet_' +  loss + '.png')
					plt.close()


