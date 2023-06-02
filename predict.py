from flags import Flags
import data_loader
import evaluate

flags = Flags().parse()

'''
# get pix2pix generated images
test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()

model_path = "/media/aisec-102/DATA3/rachel/pcxgan/models/Pix2Pix_fold1245"

for ct, mri in test_dataset:
    evaluate.predict_p2p(model_path, "p2p_1245", ct)
'''

# get pcxgan generated images

test_data_path = "/grand/EVITA/ct-mri/data/mask_data/norm_mask_neg1pos1_fold5"

test_data = data_loader.DataGenerator_Ready(flags, test_data_path, if_train=False).load()

test_dataset = data_loader.DataGenerator_Ready(flags, flags.test_data_path).load()

model_path = "/media/aisec-102/DATA3/rachel/pcxgan/models/PCxGAN_fold1245_d"

for ct, mri, label in test_dataset:
    evaluate.predict_pcx(flags, model_path, ct, label)
