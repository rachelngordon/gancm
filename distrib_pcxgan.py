import os
from pcxgan.pcxgan_mask import PCxGAN_mask
from flags import Flags
import data_loader
import pcxgan.modules_no_mask as modules
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as kr
import time

def main(flags):
    # Set the CUDA visible devices to limit TensorFlow to the available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust based on the GPUs you want to use

    # Define the MirroredStrategy for data parallelism across GPUs
    strategy = tf.distribute.MirroredStrategy()

    # Pass path to data in flags
    train_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=True).load()
    test_data = data_loader.DataGenerator_Ready(flags, flags.data_path, if_train=False).load()

    # Start the timer
    start_time = time.time()

    # Build and train the model within the strategy's scope
    with strategy.scope():
        model = PCxGAN_mask(flags)
        model.compile()

        # Define batch size per GPU
        per_gpu_batch_size = flags.batch_size // strategy.num_replicas_in_sync

        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=flags.epochs,
            verbose=1,
            batch_size=per_gpu_batch_size,
            callbacks=[modules.GanMonitor(test_data, flags)],
        )

    end_time = time.time()

    # Calculate the training duration
    training_duration = end_time - start_time

    # Print the training time
    print("Training time: {:.2f} seconds".format(training_duration))

    # Save the training time to a file
    filename = './training_time/' + flags.exp_name + "_train_time.txt"
    with open(filename, "w") as file:
        file.write("Training time: {:.2f} seconds".format(training_duration))
        print("Training time saved to", filename)

    model.save_model()
    model.model_evaluate(test_data)
    model.plot_losses(history.history)

if __name__ == '__main__':
    flags = Flags().parse()
    main(flags)
