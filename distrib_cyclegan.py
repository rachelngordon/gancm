import os
import tensorflow as tf
import json
from cyclegan.cyclegan import CycleGAN
from flags import Flags
import data_loader
import cyclegan.modules as modules
import numpy as np


def main(flags):

  # Read the list of worker nodes from the PBS_NODEFILE
  with open(os.environ['PBS_NODEFILE'], 'r') as f:
      nodes = f.read().splitlines()

  # Set the environment variables for distributed TensorFlow
  os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': nodes
      },
      'task': {'type': 'worker', 'index': int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))}
  })

  # Create a TensorFlow distribution strategy
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


  def build_and_train_model(rank, num_nodes, train_fold, test_fold, flags):

      # Build and compile your TensorFlow model here
      model = CycleGAN(flags)
      model.compile()
    

      # Define the training data and parameters
      train_data = data_loader.DataGenerator_Distrib(flags, flags.data_path, if_train=True).load()
      test_data = data_loader.DataGenerator_Distrib(flags, flags.data_path, if_train=False).load()

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

      
      # Save the model
      model.save_model(flags)

      # Evaluate the model on test data
      model.model_evaluate(test_data)

      # Plot the losses
      model.plot_losses(history.history)



  # get training folds and remove test fold
  train_folds = list(range(1,6))
  train_folds.remove(flags.test_fold)

  num_nodes = 4
  

  # Execute the script on each worker node in parallel with one fold per node for training
  for rank, train_fold in enumerate(train_folds):
     build_and_train_model(rank, num_nodes, train_fold, flags.test_fold, flags)



if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
