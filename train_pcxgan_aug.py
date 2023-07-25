from pcxgan.pcxgan_no_mask import PCxGAN
from flags import Flags
import data_loader
import pcxgan.modules_no_mask as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr
import time


def main(flags):
    # Get the available GPUs
    #gpus = tf.config.experimental.list_physical_devices('GPU')

    # Check if at least two GPUs are available
    #if len(gpus) < 2:
        # RuntimeError("At least two GPUs are required for this script.")

    # Get the name of the first GPU without "physical_device:"
    #gpu_name_1 = gpus[0].name.replace("physical_device:", "")

    # Load augmented training data on the first GPU
    #with tf.device(gpu_name_1):
    train_data = data_loader.DataGeneratorAug(flags, flags.data_path, if_train=True).load()

    # Load test data without augmentation
    test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()

    # Start the timer
    start_time = time.time()

    # Get the name of the second GPU without "physical_device:"
    #gpu_name_2 = gpus[1].name.replace("physical_device:", "")

    # Build and train the model on the second GPU
    #with tf.device(gpu_name_2):
    model = PCxGAN(flags)
    model.compile()

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=flags.epochs,
        verbose=1,
        batch_size=flags.batch_size,
        callbacks=[modules.GanMonitor(test_data, flags)],
    )

    end_time = time.time()

    # Calculate the training duration
    training_duration = end_time - start_time

    # Print the training time
    print("Training time: {:.2f} seconds".format(training_duration))

    # Save the training time to a file
    filename = '/media/aisec-102/DATA3/rachel/pcxgan/time_train/' + flags.exp_name + "_train_time.txt"
    with open(filename, "w") as file:
        file.write("Training time: {:.2f} seconds".format(training_duration))
        print("Training time saved to", filename)

    model.save_model()
    model.model_evaluate(test_data)
    model.plot_losses(history.history)


if __name__ == '__main__':
    flags = Flags().parse()
    main(flags)