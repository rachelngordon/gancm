from flags import Flags
import numpy as np
import pcxgan.modules as modules
import tensorflow.keras as kr
import evaluate

def main(flags):


  test_data_path = f"/grand/EVITA/ct-mri/data/mask_data/norm_mask_neg1pos1_fold{flags.test_fold}.npz"
	

  data_test = np.load(test_data_path)
  x_test, y_test = data_test['arr_0'], data_test['arr_1']



  # load and evaluate the model

  # get model path
  path = "/grand/EVITA/ct-mri/pcxgan/models/" + flags.name

  model = kr.models.load_model(path)
  evaluate.model_evaluate(flags, model, (x_test, y_test))
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

