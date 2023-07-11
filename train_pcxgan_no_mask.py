from pcxgan.pcxgan_no_mask import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules_no_mask as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr
import time




def main(flags):


  #train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()

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
  seed = 1                                 
  ct_datagen.fit(x_train, augment=True, seed=seed)
  mri_datagen.fit(y_train, augment=True, seed=seed)

  ct_generator = ct_datagen.flow(x_train, batch_size=flags.batch_size, shuffle=True, seed=seed, subset='training'),
  mri_generator = mri_datagen.flow(y_train, batch_size=flags.batch_size, shuffle=True, seed=seed, subset='training')

    # Create the generator for training data
  train_generator = zip(ct_generator, mri_generator)



  # Start the timer
  start_time = time.time()


  #Build and train the model
  model = PCxGAN(flags)
  model.compile()
  history = model.fit(
    train_generator,
    validation_data=test_data,
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

  # Save the training time to a file
  filename = '/grand/EVITA/ct-mri/exp_results/time_train/' + flags.exp_name + "_train_time.txt"
  with open(filename, "w") as file:
      file.write("Training time: {:.2f} seconds".format(training_duration))
      print("Training time saved to", filename)


  model.save_model()
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)