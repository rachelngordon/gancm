from pcxgan.pcxgan_mask import PCxGAN_mask
from flags import Flags
import data_loader
import pcxgan.modules_mask as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr
import time
import os


def main(flags):

  # pass path to data in flags

  train_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=False).load()


  # Start the timer
  start_time = time.time()


  #Build and train the model
  model = PCxGAN_mask(flags)
  model.compile()

  checkpoint_path = flags.model_path + flags.exp_name + '.h5'

  if os.path.exists(checkpoint_path):
     model.load_weights(checkpoint_path)

  checkpoint_callback = kr.callbacks.ModelCheckpoint(
     filepath = checkpoint_path,
     monitor = 'val_vgg_loss',
     save_best_only = True,
     save_weights_only = True,
     verbose = 1,
  )

  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor(test_data, flags), checkpoint_callback],
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