from pcx_aug_distrib import PCxGAN, VGG
from flags import Flags
import gancm.modules_just_ct as modules
import data_loader
import time
import tensorflow as tf
import tensorflow.keras as kr
from contextlib import suppress
import numpy as np

def get_strategy_scope():
  if len(tf.config.list_physical_devices("GPU")) > 1:
    print(tf.config.list_physical_devices("GPU"))
    return tf.distribute.MirroredStrategy().scope()
  else:
      suppress()


  
def main(flags):
  
  train_data = data_loader.DataGeneratorAug(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()



  start_time = time.time()
  
  with get_strategy_scope() as s:
    number_devices = s.num_replicas_in_sync
    print("Number of devices: {}".format(number_devices))

    #Build the model
    flags.batch_size = number_devices
    vgg_model = VGG()
    model = PCxGAN(flags, vgg_model, s.num_replicas_in_sync)
    model.compile()

    # Set the batch size for each replica (device)
    flags.batch_size //= number_devices

    print("Batch size: ", flags.batch_size)
    
  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.GanMonitor(test_data, flags)],
  )

  end_time = time.time()

  # Calculate the training duration
  training_duration = end_time - start_time

  # Print the training time
  print("Training time: {:.2f} seconds".format(training_duration))


  # Save the training time to a file
  filename = '/grand/EVITA/ct-mri/exp_results/time_train/' + flags.exp_name + "_train_time.txt"
  with open(filename, "w") as file:
      file.write("Training time: {:.2f} seconds".format(training_duration))
      print("Training time saved to", filename)

  
  
  model.save_model()
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
