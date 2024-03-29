from p2p.pix2pix import Pix2Pix
from flags import Flags
import numpy as np
import p2p.modules as modules
import data_loader
import time
import tensorflow as tf

def main(flags):


  train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


  start_time = time.time()
  
  #Build the model
  model = Pix2Pix(flags)
  model.compile()

  # # Define a sample input shape (batch_size, height, width, channels) for your model
  # sample_input_ct = tf.keras.layers.Input(shape=(flags.crop_size, flags.crop_size, 1))
  # sample_input_mri = tf.keras.layers.Input(shape=(flags.crop_size, flags.crop_size, 1))
  # sample_input_mask = tf.keras.layers.Input(shape=(flags.crop_size, flags.crop_size, 2))
  # sample_latent_vector = tf.keras.layers.Input(shape=(flags.latent_dim,))

  # # Call the build method with the sample input shapes
  # model.build(input_shape=sample_input_ct.shape)


  # print(model.summary())

  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.P2PMonitor(test_data, flags)],
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
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

