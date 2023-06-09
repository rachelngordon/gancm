from pcxgan.pcxgan import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr




def main(flags):

  data_path = "/media/aisec-102/DATA31/rachel/data/CV/mask_data/norm_mask_neg1pos1_fold"
  test_data_path = "/media/aisec-102/DATA31/rachel/data/CV/mask_data/norm_mask_neg1pos1_fold5"

  train_data = data_loader.DataGenerator_Ready(flags, data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_Ready(flags, test_data_path, if_train=False).load()

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