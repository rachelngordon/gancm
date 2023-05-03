from p2p.pix2pix import Pix2Pix
from flags import Flags
import numpy as np
import p2p.modules as modules

def main(flags):
  #train_dataset = data_loader.DataGenerator_PairedReady(flags, flags.data_path).load()
  #test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()
  
  data_train = np.load(flags.data_path)
  x_train, y_train = data_train['x'], data_train['y']
  data_test = np.load(flags.test_data_path)
  x_test, y_test = data_test['x'], data_test['y']

  #Build and train the model
  model = Pix2Pix(flags)
  model.compile()
  history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=flags.epochs,
    verbose=1,
    callbacks=[modules.P2PMonitor((x_test[5:8], y_test[5:8]), flags)],
  )
  
  
  model.save_model(flags)
  model.model_evaluate(test_dataset)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

