from pcxgan.pcxgan import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr



class DataGenerator_Ready(kr.utils.Sequence):
  def __init__(self, flags, data_path, if_train = True, **kwargs):
    
    super().__init__(**kwargs)

    
    self.data_path = data_path
    self.batch_size = flags.batch_size
    x, y, z = self.load_data(flags, self.data_path, if_train=if_train)
    self.dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
    self.dataset.shuffle(buffer_size=10, seed=42, reshuffle_each_iteration=False)
    self.dataset = self.dataset.map(
    lambda x, y, z: (x, y, tf.one_hot(tf.squeeze(tf.cast(z, tf.int32)), 2)), num_parallel_calls=tf.data.AUTOTUNE)

    
	
	
  def load_data(self, flags, data_path, if_train=True):
		
    if if_train:
      
      for i in [1,2,3,4]:
        path = f"{data_path}{i}.npz"
        if i == 1:
          data = np.load(path)
          x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
        else:
          data = np.load(path)
          x = np.concatenate((x, data['arr_0']), axis=0)
          y = np.concatenate((y, data['arr_1']), axis=0)
          z = np.concatenate((z, data['arr_2']), axis=0)
	
      return x, y, z

    else: 
        path = f"{data_path}.npz"
        data = np.load(path)
        x, y, z = data['arr_0'], data['arr_1'], data['arr_2']
        return x, y, z
    
  def __getitem__(self, idx):
    return self.dataset.batch(self.batch_size, drop_remainder=True)
  
  def load(self):
      return self.dataset.batch(self.batch_size, drop_remainder=True)
	




def main(flags):

  data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold"
  test_data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold5"

  train_data = DataGenerator_Ready(flags, data_path, if_train=True).load()
  test_data = DataGenerator_Ready(flags, test_data_path, if_train=False).load()

  #Build and train the model
  model = PCxGAN(flags)
  model.compile()
  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor(test_data, flags)],
  )
  
  
  model.save_model()
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)