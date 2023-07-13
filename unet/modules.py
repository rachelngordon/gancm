import tensorflow as tf
import os 
import matplotlib.pyplot as plt



def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator(flags):
  inputs = tf.keras.layers.Input(shape=[flags.crop_size, flags.crop_size, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)




def Discriminator(flags):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[flags.crop_size, flags.crop_size, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[flags.crop_size, flags.crop_size, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  print(inp.shape)
  print(tar.shape)
  print(x.shape)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)

  print(down1.shape)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  print(down2.shape)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
  print(down3.shape)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)

  print(zero_pad1.shape)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
  print(conv.shape)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  print(batchnorm1.shape)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  print(leaky_relu.shape)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
  print(zero_pad2.shape)
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
  print(last.shape)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)



## LOSSES

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(gen_output, target):

  #gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  #total_gen_loss = gan_loss + (100 * l1_loss)

  return l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss



class GanMonitor(tf.keras.callbacks.Callback):
	def __init__(self, val_dataset, flags, my_strategy=False):

		self.val_images = next(iter(val_dataset))
		self.n_samples = 1
		self.epoch_interval = flags.epoch_interval
		self.checkpoints_path = os.path.join(flags.checkpoints_dir, flags.name)
		self.hist_path = os.path.join(flags.hist_path, flags.name)
		self.sample_dir = os.path.join(flags.sample_dir, flags.name)
		self.losses = {'disc_loss': []} 

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

				# Plot losses
				plt.figure()
				plt.plot(self.losses['disc_loss'], label='Discriminator Loss')
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


