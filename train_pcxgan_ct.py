from pcxgan.pcxgan_ct_mask import PCxGAN_ct
from flags import Flags
import data_loader
import pcxgan.modules_ct_mask as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr



def main(flags):

  # pass path to data in flags

  train_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=False).load()

  #Build and train the model
  model = PCxGAN_ct(flags)

  # Retrieve the conv2d_6 layer and its kernel
  conv2d_6_layer = model.get_layer('conv2d_6')
  conv2d_6_kernel = conv2d_6_layer.kernel

  # Print the trainable attribute of conv2d_6_kernel
  print(conv2d_6_kernel.trainable)



  model.compile()
  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor(test_data, flags)],
  )
  
  
  model.save_model()
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)