
from cyclegan.cyclegan import CycleGAN
from flags import Flags
import data_loader
import cyclegan.modules as modules
import os
import numpy as np
import tensorflow as tf

def main(flags):


  train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()

  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():

    #Build and train the model
    model = CycleGAN(flags)
    model.compile()
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=flags.epochs,
        verbose=1,
        batch_size = flags.batch_size,
        callbacks=[modules.CycleMonitor(test_data, flags)],
    )
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)