from ..flags import Flags
import numpy as np
from ..pcxgan import modules
import tensorflow.keras as kr
from . import evaluate
from . import data_loader

def main(flags):

  test_data_path = f"/grand/EVITA/ct-mri/data/mask_data/norm_mask_neg1pos1_fold{flags.test_fold}"
  test_data = data_loader.DataGenerator_Ready(flags, test_data_path, if_train=False).load()


  # load and evaluate the model

  # get model path
  path = "/grand/EVITA/ct-mri/pcxgan/models/" + flags.name + '_d'

  model = kr.models.load_model(path)
  evaluate.pcxgan_evaluate(flags, model, test_data)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

