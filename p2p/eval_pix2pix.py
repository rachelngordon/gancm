from flags import Flags
import numpy as np
import p2p.modules as modules
import tensorflow.keras as kr
import evaluate
import data_loader

def main(flags):


  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()
	

  # load and evaluate the model

  # get model path
  path = "/grand/EVITA/ct-mri/pcxgan/models/" + flags.name

  model = kr.models.load_model(path)
  evaluate.pix2pix_evaluate(test_data)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

