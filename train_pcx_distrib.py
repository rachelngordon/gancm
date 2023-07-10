from pcxgan_distrib import PCxGAN, VGG
from flags import Flags
from pcxgan import modules_no_mask as modules
import data_loader
import time
import tensorflow as tf
import tensorflow.keras as kr
from contextlib import suppress
import numpy as np

def get_strategy_scope():
  if len(tf.config.list_physical_devices("GPU")) > 1:
    print(tf.config.list_physical_devices("GPU"))
    return tf.distribute.MirroredStrategy().scope()
  else:
      suppress()


  
def main(flags):
  
  #train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()

  # load test data without augmentation
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()
  x_test, y_test = test_data

  # play with rotation, shift, zoom
  
  # Create an ImageDataGenerator for CT images
  ct_datagen = kr.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
  )

  # Create an ImageDataGenerator for MRI images
  mri_datagen = kr.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
  )
  

  # remove test fold
  folds = list(range(1,6))
  folds.remove(flags.test_fold)
  
  for i in folds:

    path = f"{flags.data_path}{i}.npz"

    if i == folds[0]:
      data = np.load(path)
      x_train, y_train = data['arr_0'], data['arr_1']
  
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['arr_0']), axis=0)
      y_train = np.concatenate((y_train, data['arr_1']), axis=0)


  # do we do data augmentation on both ct and mri?                                   
  ct_datagen.fit(x_train)
  mri_datagen.fit(y_train)

    # Create the generator for training data
  train_generator = zip(
      ct_datagen.flow(x_train, batch_size=flags.batch_size, shuffle=True, subset='training'),
      mri_datagen.flow(y_train, batch_size=flags.batch_size, shuffle=True, subset='training')
  )

  start_time = time.time()
  
  with get_strategy_scope() as s:
    number_devices = s.num_replicas_in_sync
    print("Number of devices: {}".format(number_devices))

    #Build the model
    flags.batch_size = number_devices
    vgg_model = VGG()
    model = PCxGAN(flags, vgg_model, s.num_replicas_in_sync)
    model.compile()

  print("Batch size: ", flags.batch_size)
  history = model.fit(
    train_generator,
    validation_data=(x_test, y_test),
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor(test_data, flags)],
  )

  end_time = time.time()

  # Calculate the training duration
  training_duration = end_time - start_time

  # Print the training time
  print("Training time: {:.2f} seconds".format(training_duration))

  '''
  # Save the training time to a file
  filename = '/grand/EVITA/ct-mri/exp_results/time_train/' + flags.exp_name + "_train_time.txt"
  with open(filename, "w") as file:
      file.write("Training time: {:.2f} seconds".format(training_duration))
      print("Training time saved to", filename)
  '''
  
  
  model.save_model()
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
