import data_loader
from pcxgan.pcxgan import PCxGAN
from flags import Flags
import pcxgan.modules as modules
import numpy as np
import math
import tensorflow as tf


def main(flags):

  data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold1.npz"
  test_data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold5.npz"
	

  (x_train, y_train, z_train) = data_loader.DataGenerator_Ready(flags, data_path)
  (x_test, y_test, z_test) = data_loader.DataGenerator_Ready(flags, test_data_path)

  #Build and train the model
  model = PCxGAN(flags)
  model.compile()
  history = model.fit(
    (x_train, y_train, z_train),
    validation_data=(x_test, y_test, z_test),
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor((x_test, y_test, z_test), flags)],
  )
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)