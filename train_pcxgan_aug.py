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

  # load augmented training data
  # get gpus visible to tensorflow
  gpus = tf.config.list_physical_devices('GPU')
  
  # load data using gpu 0
  #with tf.device(gpus[0]):
  train_data = data_loader.DataGeneratorAug(flags, flags.data_path, if_train=True).load()

  # load test data without augmentation
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


  # Start the timer
  start_time = time.time()


  #Build and train the model
  # run the model using different gpu
  #with tf.device(gpus[1]):
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