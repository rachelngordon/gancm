from pcxgan.pcxgan import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules as modules
import numpy as np
import math
import tensorflow as tf

def main(flags):

  data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold"
  test_data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold5.npz"
	

  for i in [1,2,3,4]:
    path = f"{data_path}{i}.npz"
    if i == 1:
      data = np.load(path)
      x_train, y_train, z_train = data['arr_0'], data['arr_1'], data['arr_2']
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['arr_0']), axis=0)
      y_train = np.concatenate((y_train, data['arr_1']), axis=0)
      z_train = np.concatenate((z_train, data['arr_2']), axis=0)

  data_test = np.load(test_data_path)
  x_test, y_test, z_test = data_test['arr_0'], data_test['arr_1'], data_test['arr_2']

  
  def batch_dataset(x, y, z):
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.expand_dims(z, axis=-1)

    batch_size = flags.batch_size
    dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
    dataset = dataset.map(
        lambda x, y, z: (x, y, tf.one_hot(tf.squeeze(tf.cast(z, tf.int32)), 2)),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
  

  train_data = batch_dataset(x_train, y_train, z_train)
  test_data = batch_dataset(x_test, y_test, z_test)
  
  '''
  # Get the shape of the first batch
  first_batch = next(iter(train_data))
  batch_shape = tf.shape(first_batch)

  # Print the shape
  print(batch_shape)
  print(first_batch.numpy())
  '''


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
  
  
  model.save_model(flags)
  model.model_evaluate((x_test, y_test))
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)