from flags import Flags
import numpy as np
from p2p import modules
import tensorflow.keras as kr
import evaluate
import data_loader

def main(flags):


  test_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=False).load()
	

  # load and evaluate the model

  # get model path
  #path = "/grand/EVITA/ct-mri/exp_results/models/" + flags.name
  #path = "/grand/EVITA/ct-mri/uvit_results/uvit_models/" + flags.name
  #path = "~/uvit_models/" + flags.name
  path = "/media/aisec-102/DATA3/rachel/experiments/models/new_edge_threshold_exp/models/" + flags.exp_name + '_d'

  #path = "/media/aisec-102/DATA3/rachel/experiments/models/4000_models/" + flags.exp_name + '_d'

  # test data path: "/media/aisec-102/DATA3/rachel/data/test_data/IMAGE-DataSet#1/avg_eq_seg_test.npz" 

  model = kr.models.load_model(path)
  evaluate.pcxgan_evaluate(flags, model, test_data)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

