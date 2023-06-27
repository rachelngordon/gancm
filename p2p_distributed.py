
import tensorflow as tf
from flags import Flags
from distributed.pix2pix_distrib import Pix2Pix
import data_loader
import time
import distributed.modules_distrib as modules


def main(flags):

    # define the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        train_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=True).load()
        test_data = data_loader.DataGenerator_PairedReady(flags, flags.data_path, if_train=False).load()


        start_time = time.time()

        #Build and train the model
        model = Pix2Pix(flags)
        model.compile()
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=flags.epochs,
            verbose=1,
            batch_size = flags.batch_size*strategy.num_replicas_in_sync,
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


if __name__ == "__main__":
    flags = Flags().parse()
    main(flags)