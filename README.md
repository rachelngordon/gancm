# gan-cm
CT-to-MRI Translation work


## Models

Models are separated into their own folders with modules and model files. 

GAN-CM involves several variations, for training with just the CT (gancm_just_ct.py), just the mask (gancm_just_mask.py), or mask and CT (gancm_mask_ct.py). 
These models are all trained on 256 x 256 images but there is also an implementation of GAN-CM for training with the original 512 x 512 images and conditioning on the mask and CT (gancm_512_mask_ct.py).

Note: tensorflow_addons is necessary for using group normalization on polaris but not on the aisec server.

For the UViT Models there are several variations as well: just the UViT generator (uvit.py), UViT GAN with generator and discriminator (uvit_gan.py), and UViT GAN with SPADE similar to GAN-CM (uvit_spade.py).


## Preprocessing Data

The files were saved separately depending on equalization and the mask used for training. The data was preprocessed using the equalize.ipynb notebook in the Resources folder (in the current parent directory on the server).

Segmentation mask was obtained from the original CT and the edge mask was obtained from the CT with whichever equalization method was applied.

The data is located at: /media/aisec-102/DATA3/rachel/data/CV/ and is named according to the equalization and mask used, where "paired" indicates no mask.
The original_folds folder includes the five folds before they had been preprocessed using the equalize notebook.

The edge_comparison.ipynb was used to obtain figures for comparing the different edge thresholds in the paper.

## Settings, Loading Data

Flags.py must stay the same! Only adjusted for specific experiments but these were ultimately the best values for each parameter we found unless otherwise hard-coded.
Loss.py, evaluate.py include standard functions for training and testing. 
Data_loader.py: different data generators for loading the data depending on specific use case. See below for details:

Parameter if_train determines if the data is the training or testing set. Assumes they are already split and must be passed/loaded separately:

DataGeneratorAug: perform augmentation on paired 256 x 256 CT and MRI without mask
DataGeneratorAug_Mask: perform augmentation on 256 x 256 CT, MRI, and mask
DataGeneratorAug_512Mask: perform augmentation on 512 x 512 CT, MRI, and mask
DataGenerator_Ready: Load 256 x 256 CT, MRI, and mask from file in format: {path}/{filename}{test_fold}.npz, file path must be passed without test fold or .npz extension to flags
DataGenerator_PairedReady: Load paired 256 x 256 CT and MRI from file in format: {path}/{filename}{test_fold}.npz, file path must be passed without test fold or .npz extension to flags
DataGenerator_512Ready: Load 512 x 512 CT, MRI, and mask from file in format: {path}/{filename}{test_fold}.npz, file path must be passed without test fold or .npz extension to flags

Splits the data into train and test set, assumes not already split:

DataGenerator: Load 256 x 256 CT, MRI, and mask from file, entire file path must be passed with extension
DataGenerator_Paired: Load 256 x 256 CT and MRI from file, entire file path must be passed with extension


Note: we typically use DataGenerator_Ready for GAN-CM with mask and DataGenerator_PairedReady for GAN-CM with just CT and all other models



## Evaluation and Results

The eval_results.py file is used to calculate quantitative metrics for a saved model on the validation dataset while test_data_eval is used to obtain results for a separate test dataset. 
These may need to be modified depending on the specific instance.
Similarly, the get_predictions.py is used to saved generated images produced by a saved model. The images are saved in the generated_test folder in a different directory.
There are corresponding notebooks for these in Resources to visualize easier.

The zoom_mri.py file was used to crop the MRI images around the middle so that the differences between rMRI and sMRI could be seen more clearly. There is a corresponding notebook in Resources to visualize easier.

The get_results.ipynb notebook in the Resources folder averages the results of the five folds for each experiment. It may need to be modified depending on the experiment names.




## Converting to DICOM (all located in Resources)

process_512_data.ipynb: process DICOM images and prepare them to be passed to the model
save_test_predictions.ipynb: generate predicted images from processed DICOMs and save as numpy arrays
convert_dicom_working.ipynb: convert numpy arrays to DICOM format and save


## One Time Experiments & Using Polaris

The experiments directory contains examples for running experiments on Argonne's Polaris supercomputer using TensorFlow distributed strategies. 
When running experiments, make copies of these files (example_exp.py and example_exp.sh if running standard experiments without augmentation and each on one GPU) and change desired parameters.
To run an experiment on Polaris, use the following command: qsub -q preemptable -A EVITA -l select=1:system=polaris -l place=scatter -l walltime=72:00:00 -l filesystems=home:eagle
Then, pass the name of the script you want to run, such as example_exp.sh
Feel free to change the walltime for shorter experiments. To run, a distributed training experiment, simply change select=2 or however many nodes you need.

To check the progress and status of your jobs: qstat -wan | head -n 5; qstat -wan | grep -e rgordon
Make sure to change rgordon to your username



The old_aug_pix_exp directory contains previous experiments I've run to attempt to reduce the pixelation in our generated images from GAN-CM trained with augmentation.
So far I have found that removing all cropping and resizing, including central cropping, from the augmentation data generator (DataGeneratorAug_Mask) seems to produce the best results with slightly reduced pixelation.




## Previous Attempts / In Progress

There are a few GAN-CM checkpointing methods as well for saving the model and then loading and continuing training (see: train_gancm_ckpt.py and train_gancm_aug_ckpt.py within gancm directory, checkpointing directory).
However, these methods have not yet been proven to work and were not used for the previous experiments.

The distributed directory also contains attempts to distribute training of Pix2Pix and GAN-CM (previously PCxGAN and sometimes referred to as pcx in old filenames).
However, these attempts were also not used in the previous experiments and are still in progress. The Pix2Pix method may have worked but was never actually used.
