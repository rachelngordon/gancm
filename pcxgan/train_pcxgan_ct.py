from pcxgan.pcxgan_ct import PCxGAN_ct
from flags import Flags
import data_loader
import pcxgan.modules_ct as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr



def main(flags):

  if flags.equalized == True:
    data_path = "/grand/EVITA/ct-mri/data/CV/eq_mask/norm_mask_neg1pos1_fold"
    test_data_path = f"/grand/EVITA/ct-mri/data/CV/eq_mask/norm_mask_neg1pos1_fold{flags.test_fold}"
  else:
    data_path = "/grand/EVITA/ct-mri/data/CV/no_eq_mask/no_eq_mask_neg1pos1_fold"
    test_data_path = f"/grand/EVITA/ct-mri/data/CV/no_eq_mask/no_eq_mask_neg1pos1_fold2{flags.test_fold}"



  train_data = data_loader.DataGenerator_Ready(flags, data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_Ready(flags, test_data_path, if_train=False).load()

  #Build and train the model
  model = PCxGAN_ct(flags)
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