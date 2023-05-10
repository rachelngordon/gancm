from p2p.pix2pix import Pix2Pix
from flags import Flags
import numpy as np
import p2p.modules as modules

def main(flags):

  data_path = "/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold"
  test_data_path = "/media/aisec-102/DATA3/rachel/data/CV/normalized_neg1pos1_fold5.npz"
	

  for i in [1,2,3,4]:
    path = f"{data_path}{i}.npz"
    if i == 1:
      data = np.load(path)
      x_train, y_train = data['x'], data['y']
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['x']), axis=0)
      y_train = np.concatenate((y_train, data['y']), axis=0)

  data_test = np.load(test_data_path)
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
  model.model_evaluate((x_test, y_test))
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)

