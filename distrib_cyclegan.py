import os
import tensorflow as tf
import json
from cyclegan.cyclegan import CycleGAN
from flags import Flags
import data_loader
import cyclegan.modules as modules
import numpy as np

def main(flags):

  # Define the configuration argument (replace with your actual configuration)
  #config = '<your_config>'

  # Set the environment variables for distributed TensorFlow
  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': ['worker1:12345', 'worker2:12345', 'worker3:12345', 'worker4:12345']
      },
      'task': {'type': 'worker', 'index': int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))}
  })

  # Create a TensorFlow distribution strategy
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  # Define the function to build and train your TensorFlow model
  def build_and_train_model():

      # Build and compile your TensorFlow model here
      model = CycleGAN(flags)
      model.compile()
    

      # Define the training data and parameters
      train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
      test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()

      # Create an instance of the tf.distribute.MultiWorkerMirroredStrategy
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

      # Open a strategy scope to distribute the model and training
      with strategy.scope():
          # Train your model with the distributed strategy
          history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=flags.epochs,
            verbose=1,
            batch_size = flags.batch_size,
            callbacks=[modules.CycleMonitor(test_data, flags)],
          )

  # Call the function to build and train your TensorFlow model
  build_and_train_model()


if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
