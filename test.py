import evaluate
import data_loader


flags = Flags().parse()

train_dataset = data_loader.DataGenerator_PairedReady(flags, flags.data_path).load()
test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()


model_path = '/media/aisec1/DATA3/rachel/pcxgan/models/Pix2Pix_test'

# generate test images
for ct, mri in test_dataset:
    evaluate.predict_p2p(model_path, modelname='Pix2Pix_test' ct)