from flags import Flags
import numpy as np
import p2p.modules as modules
import tensorflow.keras as kr
import re

def main(flags):

  data_path = "/grand/EVITA/ct-mri/data/CV/normalized_neg1pos1_fold"
  test_data_path = f"/grand/EVITA/ct-mri/data/CV/normalized_neg1pos1_fold{flags.test_fold}.npz"
	
  folds = list(range(1,6))
  folds.remove(flags.test_fold)

  for i in folds:
    path = f"{data_path}{i}.npz"
    
    if i == 1:
      data = np.load(path)
      x_train, y_train = data['arr_0'], data['arr_1']
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['arr_0']), axis=0)
      y_train = np.concatenate((y_train, data['arr_1']), axis=0)

  data_test = np.load(test_data_path)
  x_test, y_test = data_test['arr_0'], data_test['arr_1']



  # load and evaluate the model

  # get model path
  path = "/grand/EVITA/ct-mri/pcxgan/models/" + flags.name

  model = kr.models.load_model(path)
  model.model_evaluate((x_test, y_test))
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

