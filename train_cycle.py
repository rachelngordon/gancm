# import sys
# sys.path.append('/media/aisec-102/DATA3/rachel/pcxgan/cyclegan')

from cyclegan.cyclegan import CycleGAN
from flags import Flags
import data_loader
import cyclegan.modules as modules
import os
import numpy as np

def main(flags):

  data_path = "/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold"
  test_data_path = f"/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold{flags.test_fold}.npz"
	
  folds = list(range(1,6))
  folds.remove(flags.test_fold)

  for i in folds:
    path = f"{data_path}{i}.npz"
    
    if i == folds[0]:
      data = np.load(path)
      x_train, y_train = data['arr_0'], data['arr_1']
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['arr_0']), axis=0)
      y_train = np.concatenate((y_train, data['arr_1']), axis=0)

  data_test = np.load(test_data_path)
  x_test, y_test = data_test['arr_0'], data_test['arr_1']

  
  #Build and train the model
  model = CycleGAN(flags)
  model.compile()
  history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.CycleMonitor((x_test[5:8], y_test[5:8]), flags)],
  )
  
  
  model.save_model(flags)
  model.model_evaluate((x_test, y_test))
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)