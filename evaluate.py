import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import math
import sklearn.metrics as sk
from flags import Flags
from datetime import datetime


flags = Flags().parse()

# calculate frechet inception distance
def calculate_fid(y_true, y_pred, input_shape = (flags.crop_size,flags.crop_size,3)):
  
  model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
  
  y_true = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_true))
  y_pred = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_pred))

  y_true = preprocess_input(y_true)
  y_pred = preprocess_input(y_pred)
  
  y_true = model.predict(y_true)
  y_pred = model.predict(y_pred)

  mu1, sigma1 = y_true.mean(axis=0), np.cov(y_true, rowvar=False)
  mu2, sigma2 = y_pred.mean(axis=0), np.cov(y_pred, rowvar=False)
  
  ssdiff = np.sum((mu1 - mu2)**2.0)

  covmean = sqrtm(np.matrix(sigma1).dot(np.matrix(sigma2)))

  if np.iscomplexobj(covmean):
    covmean = covmean.real

  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid


def get_metrics(y_true, y_pred):
  
  
  y_true,y_pred = np.array(y_true), np.array(y_pred)
  mse=sk.mean_squared_error(y_true.flatten(),y_pred.flatten())
  mae=sk.mean_absolute_error(y_true.flatten(),y_pred.flatten())
  cs=sk.pairwise.cosine_similarity([y_true.flatten()], [y_pred.flatten()])
  
  if mse == 0:
    psnr =  float('inf')
  
  else:
    psnr=20 * math.log10(1 / math.sqrt(mse))
  
  y_true = (y_true + 1.0) / 2.0
  y_pred = (y_pred + 1.0) / 2.0
  
  im1 = tf.image.convert_image_dtype(y_true, tf.float32)
  im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
  ssim=float(tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                           filter_sigma=1.5, k1=0.01, k2=0.03)[0])
  

  return mse, mae, cs, psnr,ssim


# plot the generated image
def show_plot_generated(image, name, step):

    f = pyplot.figure(figsize=(8,8))
    pyplot.axis('off')
    pyplot.imshow(np.squeeze(image),  cmap='gray')
    sample_dir_ = 'generated_test/%s/' %name
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    
    filename = '%s%s_plot_%04d.png' % (sample_dir_, name, step)
    pyplot.savefig(filename)
    pyplot.close()


def predict_pcx(flags, decoder_file, ct, label):
  
  decoder = load_model(decoder_file)
  modelname = decoder_file.split('/')[-1]
  latent_vector = tf.random.normal(
      shape=(flags.batch_size, flags.latent_dim), mean=0.0, stddev=2.0)
  
  generated = decoder([latent_vector, label, ct])
  
  for idx, image in enumerate(generated):
    show_plot_generated(image, modelname, idx+1)
    
    
  
  return generated


def predict_p2p(model_path, modelname, ct):
  

  generator = load_model(model_path)
  
  generated = generator(ct)
  
  data_name = 'test'
  
  counter = 0
  for idx, image in enumerate(generated):
    show_plot_generated(image, modelname, idx+1)
    counter += 1
  
  return generated


def pix2pix_evaluate(flags, generator, test_data, epoch=0):


        results = []
        
        #num_batches = len(test_data[0]//self.batch_size)

        #for i in range(0, num_batches, self.batch_size):
            #ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]


        for ct, mri in test_data:
            
            fake_mri = generator(ct)
            

            fid = calculate_fid(mri, fake_mri, 
				    input_shape=(flags.crop_size, flags.crop_size, 3))

            mse, mae, cs, psnr, ssim = get_metrics(mri, fake_mri)

            results.append([fid, mse, mae, cs, psnr, ssim])
            print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))

        results = np.array(results, dtype=object).mean(axis=0)

        filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        results_dir = os.path.join(flags.result_logs, flags.name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        log_file = os.path.join(results_dir, filename)
        np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")





def pcxgan_evaluate(flags, decoder, test_data, epoch=0):
  results = []
  
  for ct, mri, label in test_data:

    # Sample latent from a normal distribution.
    latent_vector = tf.random.normal(
      shape=(flags.batch_size, flags.latent_dim), mean=0.0, stddev=2.0
    )
    fake_image = decoder([latent_vector, label, ct])
    
    mri = (mri + 1.0) / 2.0
    fake_image = (fake_image + 1.0) / 2.0
    
    fid = calculate_fid(mri, fake_image,
                                  input_shape=(
                                    flags.crop_size,
                                    flags.crop_size, 3))
    mse, mae, cs, psnr, ssim = get_metrics(mri, fake_image)
    
    results.append([fid, mse, mae, cs, psnr, ssim])
    print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))
  

  results = np.array(results, dtype=object).mean(axis=0)
  
  filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
  results_dir = os.path.join(flags.result_logs, flags.name)
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  log_file = os.path.join(results_dir, filename)
  np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")