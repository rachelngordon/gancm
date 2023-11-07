import tensorflow as tf
import matplotlib.pyplot as plt

def crop_mris(rmri_path, smri_path):
    
    rmri = tf.io.read_file(rmri_path)
    print(rmri.shape)
    smri = tf.io.read_file(smri_path)

    rmri = tf.image.decode_image(rmri, channels=3)
    print(rmri.shape)
    smri = tf.image.decode_image(smri, channels=3)

    #x1, y1, height, width = 100, 150, 200, 200
    #x1, y1, height, width = 100, 100, 200, 200
    x1, y1, height, width = 150, 150, 150, 150

    cropped_rmri = tf.image.crop_to_bounding_box(rmri, y1, x1, height, width)
    print(cropped_rmri.shape)
    cropped_smri = tf.image.crop_to_bounding_box(smri, y1, x1, height, width)
    
    resize_rmri = tf.image.resize(cropped_rmri, [rmri.shape[0], rmri.shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    print(resize_rmri.shape)
    resize_smri = tf.image.resize(cropped_smri, [smri.shape[0], smri.shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    cropped_rmri_np = resize_rmri.numpy()
    print(cropped_rmri_np.shape)
    cropped_smri_np = resize_smri.numpy()
    
    plt.imshow(cropped_rmri_np)
    plt.axis("off")
    plt.imsave(rmri_path[:-4] + '_crop.png', cropped_rmri_np.astype('uint8'))

    plt.imshow(cropped_smri_np)
    plt.axis("off")
    plt.imsave(smri_path[:-4] + '_crop.png', cropped_smri_np.astype('uint8'))



crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/mri_plot_0003.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/pcx_ct_smri_plot_0003.png')
crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/mri_plot_0002.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/pcx_ct_smri_plot_0002.png')
crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/mri_plot_0001.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/avg_eq_images/pcx_ct_smri_plot_0001.png')


crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/rmri_plot_0003.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/pcx_aug_smri_plot_0003.png')
crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/rmri_plot_0002.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/pcx_aug_smri_plot_0002.png')
crop_mris('/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/rmri_plot_0001.png', 
          '/media/aisec-102/DATA3/rachel/gancm/generated_test/aug_4000/pcx_aug_smri_plot_0001.png')