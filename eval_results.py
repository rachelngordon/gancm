from flags import Flags
import numpy as np
from p2p import modules
import tensorflow.keras as kr
import evaluate
import data_loader
import tensorflow as tf

def main(flags):


  #test_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=False).load()
	

  # load and evaluate the model

  # get model path
  #path = "/grand/EVITA/ct-mri/exp_results/models/" + flags.name
  #path = "/grand/EVITA/ct-mri/uvit_results/uvit_models/" + flags.name
  #path = "~/uvit_models/" + flags.name
  eqs = ['eq', 'no_eq', 'avg_eq']
  masks = ['bin_edge', 'seg']

  for e in eqs:
    for m in masks:
      for test_fold in range(1,6):
        
        #data_path = f'/media/aisec-102/DATA3/rachel/data/CV/{e}_{m}/norm_mask_neg1pos1_fold{test_fold}'
        data_path = f'/eagle/EVITA/ct-mri/data/CV/{e}_{m}/norm_mask_neg1pos1_fold{test_fold}'

        test_data = data_loader.DataGenerator_Ready(flags, data_path, if_train=False).load()

        folds = list(range(1,6))
        folds.remove(test_fold)
        folds_str = ''.join(map(str, folds))
      
        #path = "/media/aisec-102/DATA3/rachel/experiments/models/new_edge_threshold_exp/models/" + flags.exp_name + '_d'
        exp_name = f'uvit_spade_{m}_{e}_{folds_str}'

        path = f'~/models/{exp_name}_d'
        print("exp_name: ", exp_name)
        print("test fold: ", test_fold)

        #path = "/media/aisec-102/DATA3/rachel/experiments/models/4000_models/" + flags.exp_name + '_d'

        # test data path: "/media/aisec-102/DATA3/rachel/data/test_data/IMAGE-DataSet#1/avg_eq_seg_test.npz" 

        #model = kr.models.load_model(path)
        model = tf.saved_model.load(path)
        evaluate.pcxgan_evaluate(flags, model, exp_name, test_data)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

