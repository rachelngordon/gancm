# import sys
# sys.path.append('/media/aisec-102/DATA3/rachel/pcxgan/cyclegan')

from cyclegan.cyclegan import CycleGAN
from flags import Flags
import data_loader
import cyclegan.modules as modules
import os
import numpy as np

def main(flags):

  if flags.equalized:
    data_path = "/media/aisec-102/DATA3/rachel/data/CV/eq_paired/normalized_neg1pos1_fold"
    test_data_path = f"/media/aisec-102/DATA3/rachel/data/CV/eq_paired/normalized_neg1pos1_fold{flags.test_fold}"
  else:
    data_path = "/media/aisec-102/DATA3/rachel/data/CV/no_eq_paired/no_eq_neg1pos1_fold"
    test_data_path = f"/media/aisec-102/DATA3/rachel/data/CV/no_eq_paired/no_eq_neg1pos1_fold{flags.test_fold}"



  train_data = data_loader.DataGenerator_PairedReady(flags, data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, test_data_path, if_train=False).load()

  
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