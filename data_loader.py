import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
import math
import cv2


# data generator for pcxgan (normalized mask data) and cross validation with test fold ready
class DataGenerator_Ready(kr.utils.Sequence):
  def __init__(self, flags, data_path, if_train = True, **kwargs):
    
    super().__init__(**kwargs)

    
    self.data_path = data_path
    self.batch_size = flags.batch_size
    
	# load data
    x, y, z = self.load_data(flags, self.data_path, if_train=if_train)
    
	# create dataset
    self.dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
    self.dataset.shuffle(buffer_size=10, seed=42, reshuffle_each_iteration = not if_train)
    self.dataset = self.dataset.map(
    lambda x, y, z: (x, y, tf.one_hot(tf.squeeze(tf.cast(z, tf.int32)), 2)), num_parallel_calls=tf.data.AUTOTUNE)
    
	
	
  def load_data(self, flags, data_path, if_train=True):
	
	# concatenate training folds
    if if_train:
      
      folds = list(range(1,6))
      
	  # remove test fold
      folds.remove(flags.test_fold)
      
      for i in folds:
        path = f"{data_path}{i}.npz"
        if i == folds[0]:
          data = np.load(path)
          x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
        else:
          data = np.load(path)
          x = np.concatenate((x, data['arr_0']), axis=0)
          y = np.concatenate((y, data['arr_1']), axis=0)
          z = np.concatenate((z, data['arr_2']), axis=0)
	
      return x, y, z

	# load test fold
    else: 
        path = f"{data_path}{flags.test_fold}.npz"
        data = np.load(path)
        x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
        return x, y, z
    
  def __getitem__(self, idx):
    return self.dataset.batch(self.batch_size, drop_remainder=True)
  
  def load(self):
      return self.dataset.batch(self.batch_size, drop_remainder=True)
	
	

# data generator for pix2pix (normalized paired data) and cross validation with test fold ready
class DataGenerator_PairedReady(kr.utils.Sequence):

	def __init__(self, flags, data_path, if_train = True, **kwargs):
		
		super().__init__(**kwargs)

		self.batch_size = flags.batch_size
		self.data_path = data_path

		# load data
		x, y = self.load_data(flags, self.data_path, if_train=if_train)

		# create dataset
		self.dataset = tf.data.Dataset.from_tensor_slices((x, y))
		self.dataset.shuffle(buffer_size=10, seed=42, reshuffle_each_iteration=False)
		
	
	def load_data(self, flags, data_path, if_train):

		# concatenate training folds
		if if_train:

			folds = list(range(1,6))

			# remove test fold
			folds.remove(flags.test_fold)
			
			for i in folds:

				path = f"{data_path}{i}.npz"

				if i == folds[0]:
					data = np.load(path)
					x, y = data['arr_0'], data['arr_1']
			
				else:
					data = np.load(path)
					x = np.concatenate((x, data['arr_0']), axis=0)
					y = np.concatenate((y, data['arr_1']), axis=0)
		
			return x, y
		
		# load test fold
		else: 
			path = f"{data_path}{flags.test_fold}.npz"
			data = np.load(path)
			x, y = data['arr_0'], data['arr_1']
			
			return x, y

	

	def __getitem__(self, idx):
		return self.dataset.batch(self.batch_size, drop_remainder=True)
	
	def load(self):
		return self.dataset.batch(self.batch_size, drop_remainder=True)




# data generator for pcxgan (mask data) without test set ready
class DataGenerator(kr.utils.Sequence):
	def __init__(self, flags, return_labels=True, is_train=True, test_idx=[], **kwargs):
		
		super().__init__(**kwargs)

		self.test_idx = test_idx
		x, y, z = self.load_data(flags, is_train, return_labels)
		self.dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
		
		if is_train:
			self.dataset = self.dataset.cache().map(
				self.random_jitter, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
				self.buffer_size)
		else:
			self.dataset = self.dataset.cache().map(
				self.resize, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
				self.buffer_size)
		
		if return_labels:
			self.dataset = self.dataset.map(
				lambda x, y, z: (x, y, tf.one_hot(tf.squeeze(tf.cast(z, tf.int32)), 2)), num_parallel_calls=tf.data.AUTOTUNE)

		
		#print(self.dataset.element_spec)
	
	def load_data(self, flags, is_train, return_labels):
		data = np.load(flags.data_path)
		
		x, y = data['arr_0'], data['arr_1']
		self.batch_size = flags.batch_size
		self.image_shape = x.shape[1:]
		self.image_size = self.image_shape[0]
		self.crop_size = (flags.crop_size, flags.crop_size)  # can be changed
		self.threshold = flags.edge_threshold
		self.num_batches = math.ceil(len(x) / self.batch_size)
		self.buffer_size = len(x)
		self.batch_idx = np.array_split(range(len(x)), self.num_batches)
		
		if flags.remove_bad_images:
			# bad images list
			# you need to check to what file this applys
			try:
				bad_image_list = [37, 42, 111, 907, 908, 936, 1108, 1110, 1116, 1117,
													1130, 1138, 1545, 1557, 2006, 2012, 2021, 1925,
													1926, 1932, 1938, 950]
				x = np.delete(x, bad_image_list, 0)
				y = np.delete(y, bad_image_list, 0)
			except:
				print("Couldn't delete the images")
				

		
		#self.test_idx = list(range(len(x)-test_size, len(x)))
		test_size = int(len(x) * flags.data_test_rate)

		if is_train == True:
			# suffle x, y
			p = np.random.permutation(len(x))
			self.train_idx, self.test_idx = p[:-test_size], p[-test_size:]
			x, y = x[self.train_idx], y[self.train_idx]
		else:
			x, y = x[self.test_idx], y[self.test_idx]

		
		if flags.apply_normalization:
			x, y = (self.x - 0.5) / 0.5, (self.y - 0.5) / 0.5
		
		x = tf.cast(x, tf.float32)
		y = tf.cast(y, tf.float32)
		
		if return_labels:
			
			z = tf.sqrt(tf.math.reduce_sum(tf.image.sobel_edges(x) ** 2, axis=-1))
			z = self.normalize(z.numpy())
			z = tf.cast(z, tf.float32)
		else:
			z=x
		# print(self.x.shape, self.y.shape, self.mask.shape)
		
		return x, y, z
	
	@tf.function()
	def resize(self, input_1, input_2, input_3, resize_size=320):
		input_1 = tf.image.resize(input_1, [resize_size, resize_size],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		input_2 = tf.image.resize(input_2, [resize_size, resize_size],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		input_3 = tf.image.resize(input_3, [resize_size, resize_size],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		return input_1, input_2, input_3
	
	@tf.function()
	def random_crop(self, input_1, input_2, input_3, crop_size=512):
		input_1 = tf.image.random_crop(input_1, [crop_size, crop_size, 1])
		input_2 = tf.image.random_crop(input_2, [crop_size, crop_size, 1])
		input_3 = tf.image.random_crop(input_3, [crop_size, crop_size, 1])
		
		return input_1, input_2, input_3
	
	@tf.function()
	def random_jitter(self, input_1, input_2, input_3):
		input_1, input_2, input_3 = self.resize(input_1, input_2, input_3, 320)
		input_1, input_2, input_3 = self.random_crop(input_1, input_2, input_3)
		"""
		if tf.random.uniform(()) > 0.5:
			input_1 = tf.image.flip_left_right(input_1)
			input_2 = tf.image.flip_left_right(input_2)
			input_3 = tf.image.flip_left_right(input_3)
		"""
		
		return input_1, input_2, input_3
	
	def normalize(self, x):
		x = (x - x.min()) / (x.max() - x.min())
		return (x > self.threshold) * 1
	
	def __len__(self):
		return self.num_batches  # len(self.batch_idx)
	
	def __getitem__(self, idx):
		return self.dataset.batch(self.batch_size, drop_remainder=True)
	
	def load(self):
		return self.test_idx, self.dataset.batch(self.batch_size, drop_remainder=True)


# data generator for pix2pix without test set ready
class DataGenerator_Paired(kr.utils.Sequence):
	def __init__(self, flags, is_train=True, is_test=False, test_idx=[], **kwargs):
		super().__init__(**kwargs)
		self.test_idx = test_idx
		x, y = self.load_data(flags, is_train, is_test)
		self.dataset = tf.data.Dataset.from_tensor_slices((x, y))
		
		if is_train:
			self.dataset = self.dataset.cache().map(
				lambda x, y: self.random_jitter(x, y, image_size=flags.crop_size), num_parallel_calls=tf.data.AUTOTUNE)
		else:
			self.dataset = self.dataset.cache().map(
				lambda x, y: self.resize(x, y, resize_size=flags.crop_size), num_parallel_calls=tf.data.AUTOTUNE)
		
	

	def load_data(self, flags, is_train, is_test):

		if is_test:
			data = np.load(flags.test_data_path)
		else:
			data = np.load(flags.data_path)
		
		x, y = data['arr_0'], data['arr_1']
		self.batch_size = flags.batch_size
		self.image_shape = x.shape[1:]
		self.image_size = self.image_shape[0]
		self.num_batches = math.ceil(len(x) / self.batch_size)
		self.buffer_size = len(x)
		self.batch_idx = np.array_split(range(len(x)), self.num_batches)
		
		if flags.remove_bad_images:
			# bad images list
			# you need to check to what file this applys
			try:
				bad_image_list = [37, 42, 111, 907, 908, 936, 1108, 1110, 1116, 1117,
													1130, 1138, 1545, 1557, 2006, 2012, 2021, 1925,
													1926, 1932, 1938, 950]
				x = np.delete(x, bad_image_list, 0)
				y = np.delete(y, bad_image_list, 0)
			except:
				print("Couldn't delete the images")
		
		test_size = int(len(x) * flags.data_test_rate)
		
		if is_train == True:
			p = np.random.permutation(len(x))
			self.train_idx, self.test_idx = p[:-test_size], p[-test_size:]
			x, y = x[self.train_idx], y[self.train_idx]
		else:
			x, y = x[self.test_idx], y[self.test_idx]
		
		if flags.apply_normalization:
			x, y = (self.x - 0.5) / 0.5, (self.y - 0.5) / 0.5
		
		x = tf.cast(x, tf.float32)
		y = tf.cast(y, tf.float32)
		
		return x, y
	
	@tf.function()
	def resize(self, input_1, input_2, resize_size=256):
		input_1 = tf.image.resize(input_1, [resize_size, resize_size],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		input_2 = tf.image.resize(input_2, [resize_size, resize_size],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		return input_1, input_2
	
	@tf.function()
	def random_crop(self, input_1, input_2,crop_size=256):
		input_1 = tf.image.random_crop(input_1, [crop_size, crop_size, 1])
		input_2 = tf.image.random_crop(input_2, [crop_size, crop_size, 1])
	
		return input_1, input_2
	
	@tf.function()
	def random_jitter(self, input_1, input_2, image_size=256):
		input_1, input_2 = self.resize(input_1, input_2, image_size)
		input_1, input_2 = self.random_crop(input_1, input_2, image_size)

		return input_1, input_2
	

	def __len__(self):
		return self.num_batches  # len(self.batch_idx)
	
	def __getitem__(self, idx):
		return self.dataset.batch(self.batch_size, drop_remainder=True)
	
	def load(self):
		return self.test_idx, self.dataset.batch(self.batch_size, drop_remainder=True)