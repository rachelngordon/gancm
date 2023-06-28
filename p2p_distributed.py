from distributed.pix2pix_distrib import Pix2Pix
from flags import Flags
import distributed.modules_distrib as modules
import data_loader
import time
import tensorflow as tf
import loss

def main(flags):


  train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


  start_time = time.time()
  
  # define the distribution strategy
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope() as s:
    #Build the model
    model = Pix2Pix(flags)
    feature_matching_loss = loss.FeatureMatchingLoss(reduction=tf.keras.losses.Reduction.NONE)
    vgg_loss = loss.VGGFeatureMatchingLoss(reduction=tf.keras.losses.Reduction.NONE)
    model.compile(feature_matching_loss, vgg_loss)


  history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=flags.epochs,
    verbose=1,
    batch_size = flags.batch_size,
    callbacks=[modules.P2PMonitor(test_data, flags)],
  )

  end_time = time.time()

  # Calculate the training duration
  training_duration = end_time - start_time

  # Print the training time
  print("Training time: {:.2f} seconds".format(training_duration))

  # Save the training time to a file
  filename = '/grand/EVITA/ct-mri/exp_results/training_time/' + flags.exp_name + "_train_time.txt"
  with open(filename, "w") as file:
      file.write("Training time: {:.2f} seconds".format(training_duration))
      print("Training time saved to", filename)
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
