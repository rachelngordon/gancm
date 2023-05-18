from pcxgan.pcxgan import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules as modules
import numpy as np

def main(flags):

  data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold"
  test_data_path = "/media/aisec-102/DATA31/rachel/data/CV/norm_mask_neg1pos1_fold5.npz"
	

  for i in [1,2,3,4]:
    path = f"{data_path}{i}.npz"
    if i == 1:
      data = np.load(path)
      x_train, y_train, mask_train = data['arr_0'], data['arr_1'], data['arr_2']
    else:
      data = np.load(path)
      x_train = np.concatenate((x_train, data['arr_0']), axis=0)
      y_train = np.concatenate((y_train, data['arr_1']), axis=0)

  data_test = np.load(test_data_path)
  x_test, y_test, mask_test = data_test['arr_0'], data_test['arr_1'], data_test['arr_2']

  
  #Build and train the model
  model = PCxGAN(flags)
  model.compile()
  history = model.fit(
    x_train, y_train, mask_train,
    validation_data=(x_test, y_test, mask_test),
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size
    callbacks=[modules.GanMonitor((x_test[5:8], y_test[5:8], mask_test[5:8]), flags)],
  )
  
  
  model.save_model(flags)
  model.model_evaluate((x_test, y_test))
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)