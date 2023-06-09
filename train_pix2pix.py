from p2p.pix2pix import Pix2Pix
from flags import Flags
import numpy as np
import p2p.modules as modules
import data_loader

def main(flags):

  if flags.equalized:
    data_path = "/grand/EVITA/ct-mri/data/CV/eq_paired/normalized_neg1pos1_fold"
    test_data_path = f"/grand/EVITA/ct-mri/data/CV/eq_paired/normalized_neg1pos1_fold{flags.test_fold}"
  else:
    data_path = "/grand/EVITA/ct-mri/data/CV/no_eq_paired/no_eq_neg1pos1_fold"
    test_data_path = f"/grand/EVITA/ct-mri/data/CV/no_eq_paired/no_eq_neg1pos1_fold{flags.test_fold}"



  train_data = data_loader.DataGenerator_PairedReady(flags, data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, test_data_path, if_train=False).load()



  
  #Build and train the model
  model = Pix2Pix(flags)
  model.compile()
  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.P2PMonitor(test_data, flags)],
  )
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

