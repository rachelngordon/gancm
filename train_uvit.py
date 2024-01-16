import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
from uvit import uvit
from uvit import modules_uvit as modules
import time
import data_loader
from flags import Flags



def main(flags):

    # pass path to data in flags

    train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
    test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


    # Start the timer
    start_time = time.time()



    # Build the unet model
    model = uvit.UNetViTModel(flags)

    # Compile the model
    # model.compile(optimizer=model.optimizer, loss=model.loss)
    model.compile()

    # Train the model
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=flags.epochs,
        batch_size=flags.batch_size,
        callbacks=[modules.GanMonitor(test_data, flags)],
    )


    end_time = time.time()

    # Calculate the training duration
    training_duration = end_time - start_time

    # Print the training time
    print("Training time: {:.2f} seconds".format(training_duration))

    # Save the training time to a file
    #filename = '/grand/EVITA/ct-mri/new_edge_results/time_train/' + flags.exp_name + "_train_time.txt"
    #   with open(filename, "w") as file:
    #       file.write("Training time: {:.2f} seconds".format(training_duration))
    #       print("Training time saved to", filename)


    model.save_model('/media/aisec-102/DATA3/rachel/experiments/models/uvit_test')
    model.model_evaluate(test_data)
    #model.plot_losses(history.history)


if __name__ == '__main__':
    flags = Flags().parse()
    main(flags)