import os
import numpy as np
import tensorflow as tf
#import tensorflow.io as tfio
from tensorflow.keras.models import load_model
from matplotlib import pyplot
#import plot
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import math
import sklearn.metrics as sk
from flags import Flags
#import data_loader
#import modules

flags = Flags().parse()

# calculate frechet inception distance
def calculate_fid(y_true, y_pred, input_shape = (flags.crop_size,flags.crop_size,3)):
  
  model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
  
  y_true = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_true))
  y_pred = tf.image.grayscale_to_rgb(tf.convert_to_tensor(y_pred))
  
  # pre-process images
  y_true = preprocess_input(y_true)
  y_pred = preprocess_input(y_pred)
  
  y_true = model.predict(y_true)
  y_pred = model.predict(y_pred)

  # calculate mean and covariance statistics
  mu1, sigma1 = y_true.mean(axis=0), np.cov(y_true, rowvar=False)
  mu2, sigma2 = y_pred.mean(axis=0), np.cov(y_pred, rowvar=False)
  
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)

  # calculate sqrt of product between cov
  covmean = sqrtm(np.matrix(sigma1).dot(np.matrix(sigma2)))

  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
    covmean = covmean.real

  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid


def get_metrics(y_true, y_pred):
  
  
  y_true,y_pred = y_true.numpy(), y_pred.numpy()
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


# plot the image, the translation, and the reconstruction
def show_plot_generated(image, name = None, dataname = None, step = None):

    image = (image + 1) / 2.0
    f = pyplot.figure(figsize=(8,8))
    pyplot.axis('off')
    pyplot.imshow(np.squeeze(np.squeeze(image)),  cmap='gray')
    sample_dir_ = 'generated_test/'+name+'/'+dataname+'/'
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    filename1 = sample_dir_+'%s_plot_%04d.png' % (name, (step+1))
    pyplot.savefig(filename1)
    pyplot.show()
    pyplot.close()


def predict_pcx(flags, decoder_file, label):
  
  #encoder = load_model(encoder_file)
  decoder = load_model(decoder_file)
  latent_vector = tf.random.normal(
      shape=(flags.batch_size, flags.latent_dim), mean=0.0, stddev=2.0)
  

  generated = decoder([tf.convert_to_tensor(latent_vector), tf.convert_to_tensor(label)])
  
  modelname = 'PCxGAN_fold1245'
  print(modelname)
  
  data_name = 'test'
  
  counter = 0
  for image in generated:
    show_plot_generated(image, modelname, data_name + str(counter), 0)
    counter += 1
  
  return generated


def predict_p2p(model_path, modelname, ct):
  

  generator = load_model(model_path)
  
  generated = generator(ct)
  
  data_name = 'test'
  
  counter = 0
  for image in generated:
    show_plot_generated(image, modelname, data_name + str(counter), 0)
    counter += 1
  
  return generated



'''
def normalize_scale(x, file=''):
  ok_=True
  if x.max() - x.min()!=0:
    x1 = (x-x.min())/(x.max()-x.min())
  else:
    print(file)
    ok_=False
    x1 = np.zeros(x.shape)
  x1 = (x1 - 0.5) / 0.5
  return  x1, ok_


def load_image(image_path):
  image_bytes = tf.io.read_file(image_path)
  image_ = tfio.image.decode_dicom_image(image_bytes,dtype=tf.uint16)
  image_, _  = normalize_scale(np.squeeze(image_.numpy()))
  image_ = np.expand_dims(image_, axis=2)
  return image_
'''

'''
def predict_image (model_file, image):
  
  model_AtoB = load_model(model_file)
  sample = load_image(image)
  
  A_sample = np.expand_dims(sample, axis=0)
  B_generated  = model_AtoB.predict(A_sample)
  
  modelname = model_file.split('/')[-1][:-3]
  print(modelname)
  
  data_name = image.split('/')[-1][:-4]
  plot.show_plot_generated(B_generated[0]*3500, modelname, data_name, 0 )
'''
'''
def predict( model, image_ct ):
  #image_ct = 'dataset/dataset/Benters, T 0266877/IMAGE-DataSet#1/CT0009.dcm'
  #'dataset/dataset_test/test-sets/IMAGE-DataSet#1/CT0009.dcm'
  image_ = load_image(image_ct)
  f = pyplot.figure(figsize=(8,8))
  pyplot.imshow(np.squeeze(image_), cmap='gray')
  
  #model = 'models/Pix2Pix/new/model_1466520.h5'
  predict_image(model, image_ct)
'''


'''
def read_files(path):
  CT_IMAGES  = []
  MRI_IMAGES = []
  for root, dirs, files in os.walk(path):
    for file in files:
      if file.endswith(".dcm") and file.startswith("CT"):
        MRI_FILE = "MRI"+file[2:]
        if os.path.exists(os.path.join(root, MRI_FILE)):
          CT_IMAGES.append(os.path.join(root, file))
          MRI_IMAGES.append(os.path.join(root, MRI_FILE))
  
  return CT_IMAGES, MRI_IMAGES
'''
'''
# load all images in a directory into memory
def load_predict (path, model_path):
  
  model = load_model(model_path)
  
  dir_name = '/'.join(path.split('/')[2:])
  print(dir_name)
  
  dest_dir = 'dataset/dataset2/'+dir_name+'/'
  print(dest_dir)
  
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  
  CT_IMAGES, MRI_IMAGES = read_files(path)
  
  for ct, mri in zip(CT_IMAGES, MRI_IMAGES):
    filename_ct = (ct.split('/')[-1])[:-4]+'.png'
    filename_mri =( mri.split('/')[-1])[:-4]+'.png'
    
    image_bytes_CT = tf.io.read_file(ct)
    image_bytes_MRI = tf.io.read_file(mri)
    
    #print(image_bytes_CT.shape)
    
    image_CT = tfio.image.decode_dicom_image(image_bytes_CT,dtype=tf.uint16)
    image_MRI = tfio.image.decode_dicom_image(image_bytes_MRI,dtype=tf.uint16)
    
    image_CT  = np.squeeze(image_CT.numpy())
    image_MRI = np.squeeze(image_MRI.numpy())
    
    
    pyplot.image.imsave(dest_dir+filename_ct, image_CT, cmap=pyplot.cm.gray)
    pyplot.image.imsave(dest_dir+filename_mri, image_MRI, cmap=pyplot.cm.gray)
    
    image_CT, _  = data_helper.normalize_minmax(image_CT, ct)
    image_CT = np.expand_dims(image_CT, axis=2)
    #print(image_CT.shape)
    
    image_CT = np.expand_dims(image_CT, axis=0)
    
    generated_mri = model.predict(image_CT)
    
    #generated_mri = (np.squeeze(generated_mri[0]) + 1) / 2.0
    generated_mri = generated_mri[0]*3500
    
    generated_mri = (generated_mri + 1) / 2.0
    
    pyplot.image.imsave(dest_dir+"G-"+filename_mri, np.squeeze(generated_mri), cmap=pyplot.cm.gray)
'''