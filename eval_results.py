from flags import Flags
import numpy as np
from p2p import modules
import tensorflow.keras as kr
import evaluate
import data_loader

def main(flags):


  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()
	

  # load and evaluate the model

  # get model path
  #path = "/grand/EVITA/ct-mri/exp_results/models/" + flags.name
  path = "/media/aisec-102/DATA3/rachel/experiments/models/uvit_models/" + flags.name

  model = kr.models.load_model(path)
  evaluate.uvit_evaluate(flags, model, test_data)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

