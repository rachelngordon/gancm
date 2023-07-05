from pix2pix_distrib import Pix2Pix
from flags import Flags
import modules_distrib as modules
import data_loader
import time
import tensorflow as tf
import tensorflow.keras as kr

def main(flags):


  train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
  test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


  start_time = time.time()
  
  # define the distribution strategy
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope() as s:
    # define VGG model for VGG loss
    encoder_layers = [
						"block1_conv1",
						"block2_conv1",
						"block3_conv1",
						"block4_conv1",
						"block5_conv1",
				]
    weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    vgg = kr.applications.VGG19(include_top=False, weights="imagenet")
    layer_outputs = [vgg.get_layer(x).output for x in encoder_layers]
    vgg_model = kr.Model(vgg.input, layer_outputs, name="VGG")

    #Build the model
    model = Pix2Pix(flags, vgg_model, weights, strategy.num_replicas_in_sync)
    model.compile()


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

  '''
  # Save the training time to a file
  filename = '/grand/EVITA/ct-mri/exp_results/time_train/' + flags.exp_name + "_train_time.txt"
  with open(filename, "w") as file:
      file.write("Training time: {:.2f} seconds".format(training_duration))
      print("Training time saved to", filename)
  '''
  
  
  model.save_model(flags)
  model.model_evaluate(test_data)
  model.plot_losses(history.history)
  
  
if __name__ == '__main__':
  flags = Flags().parse()
  main(flags)
