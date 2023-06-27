
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from . import modules_distrib as modules
from . import loss_distrib as loss
import evaluate
import os
import pandas as pd
from  matplotlib import pyplot as plt
from datetime import datetime

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    
    # Pix2Pix
    class Pix2Pix(kr.Model):
        def __init__(self, flags,**kwargs):

            super().__init__(**kwargs)
            self.flags = flags

            self.experiment_name = self.flags.name
            self.samples_dir = self.flags.sample_dir
            self.models_dir = self.flags.checkpoints_dir
            self.hist_dir = self.flags.hist_path
            self.image_shape = (self.flags.crop_size, self.flags.crop_size, 1)
            self.image_size = self.flags.crop_size
            self.batch_size = self.flags.batch_size

            self.feature_loss_coeff = self.flags.feature_loss_coeff
            self.vgg_feature_loss_coeff = self.flags.vgg_feature_loss_coeff
            self.generator_loss_coeff = self.flags.generator_loss_coeff
            self.ssim_loss_coeff = self.flags.ssim_loss_coeff
            self.mae_loss_coeff = self.flags.mae_loss_coeff
            self.disc_loss_coeff = self.flags.disc_loss_coeff

            self.discriminator = modules.Discriminator(flags)
            self.generator = modules.p2p_generator(flags)
            self.patch_size, self.combined_model = self.build_combined_model()

            self.generator_optimizer = kr.optimizers.Adam(self.flags.gen_lr, beta_1=self.flags.gen_beta_1)
            self.discriminator_optimizer = kr.optimizers.Adam(self.flags.disc_lr, beta_1=self.flags.gen_beta_1)
            self.discriminator_loss = loss.DiscriminatorLoss()
            self.feature_matching_loss = loss.FeatureMatchingLoss()
            self.vgg_loss = loss.VGGFeatureMatchingLoss()
            self.mae_loss = loss.MAE()

            self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
            self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
            self.feat_loss_tracker = tf.keras.metrics.Mean(name="feat_loss")
            self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
            self.ssim_loss_tracker = tf.keras.metrics.Mean(name="ssim_loss")
            self.mae_loss_tracker = tf.keras.metrics.Mean(name="mae_loss")


        @property
        def metrics(self):
            return [
                self.disc_loss_tracker,
                self.gen_loss_tracker,
                self.feat_loss_tracker,
                self.vgg_loss_tracker,
                self.ssim_loss_tracker,
                self.mae_loss_tracker]


        def build_combined_model(self):

            self.discriminator.trainable = False
            ct_input = kr.Input(shape=self.image_shape, name="ct")
            mri_input = kr.Input(shape=self.image_shape, name="mri")

            generated_mri = self.generator(ct_input)

            discriminator_outputs = self.discriminator([ct_input, generated_mri])
            patch_size = discriminator_outputs[-1].shape[1]
            combined_model = kr.Model(
                [ct_input, mri_input],
                [discriminator_outputs, generated_mri])

            return patch_size, combined_model


        def compile(self, **kwargs):
            super().compile(**kwargs)


        def train_discriminator(self, ct, real_mri):
            
            fake_mri = self.generator(ct)
            with tf.GradientTape() as gradient_tape:
                pred_fake = self.discriminator([ct, fake_mri])[-1]  
                pred_real = self.discriminator([ct, real_mri])[-1]  
                loss_fake = self.discriminator_loss(False, pred_fake)
                loss_real = self.discriminator_loss(True, pred_real)
                total_loss = self.disc_loss_coeff * (loss_fake + loss_real)

            self.discriminator.trainable = True
            gradients = gradient_tape.gradient(
                total_loss, self.discriminator.trainable_variables)


            self.discriminator_optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

            return total_loss


        def train_generator(self, ct, mri__):

            self.discriminator.trainable = False
            with tf.GradientTape() as tape:
                fake_d_output, fake_mri = self.combined_model([ct, mri__])
                real_d_output = self.discriminator([ct, mri__])

                pred = fake_d_output[-1]
                
                # Compute generator loss
                g_loss = self.generator_loss_coeff*loss.generator_loss(pred)
                vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(mri__, fake_mri)
                feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
                    real_d_output, fake_d_output
                )
                ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(mri__, fake_mri)
                mae_loss = self.mae_loss_coeff * self.mae_loss(mri__, fake_mri)
                total_loss = g_loss + vgg_loss + feature_loss + ssim_loss + mae_loss
                
            all_trainable_variables = (
                self.combined_model.trainable_variables
            )

            gradients = tape.gradient(total_loss, all_trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(gradients, all_trainable_variables)
            )

            return total_loss, vgg_loss, feature_loss, ssim_loss, mae_loss


        def train_step(self, data):

            ct, mri = data
            
            discriminator_loss = self.train_discriminator(ct, mri)
            (generator_loss, vgg_loss, feature_loss, ssim_loss, mae_loss) = self.train_generator(ct, mri)

            # Report progress.
            self.disc_loss_tracker.update_state(discriminator_loss)
            self.gen_loss_tracker.update_state(generator_loss)
            self.feat_loss_tracker.update_state(feature_loss)
            self.vgg_loss_tracker.update_state(vgg_loss)
            self.ssim_loss_tracker.update_state(ssim_loss)
            self.mae_loss_tracker.update_state(mae_loss)


            results = {m.name: m.result() for m in self.metrics}

            return results


        def test_step(self, data):

            ct, mri = data
            fake_mri = self.generator(ct)

            # Calculate the losses.
            pred_fake = self.discriminator([ct, fake_mri])[-1]
            pred_real = self.discriminator([ct, mri])[-1]  
            loss_fake = self.discriminator_loss(False, pred_fake)
            loss_real = self.discriminator_loss(True, pred_real)
            total_discriminator_loss = self.disc_loss_coeff * (loss_fake + loss_real)

            real_d_output = self.discriminator([fake_mri, mri])
            fake_d_output, fake_image = self.combined_model([ct, mri])
            pred = fake_d_output[-1]
            g_loss = self.generator_loss_coeff*loss.generator_loss(pred)
            
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(mri, fake_image)
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
                real_d_output, fake_d_output)
            ssim_loss = self.ssim_loss_coeff * loss.SSIMLoss(mri, fake_image)
            mae_loss = self.mae_loss_coeff * self.mae_loss(mri, fake_mri)
            total_generator_loss = g_loss + vgg_loss + feature_loss + ssim_loss + mae_loss

            # Report progress.
            self.disc_loss_tracker.update_state(total_discriminator_loss)
            self.gen_loss_tracker.update_state(total_generator_loss)
            self.feat_loss_tracker.update_state(feature_loss)
            self.vgg_loss_tracker.update_state(vgg_loss)
            self.ssim_loss_tracker.update_state(ssim_loss)
            self.mae_loss_tracker.update_state(mae_loss)

            results = {m.name: m.result() for m in self.metrics}

            return results

        def call(self, inputs):
            return self.generator(inputs)

        
        def model_evaluate(self, test_data, epoch=0):


            results = []
            
            #num_batches = len(test_data[0]//self.batch_size)

            #for i in range(0, num_batches, self.batch_size):
                #ct, mri = test_data[0][i:i+self.batch_size], test_data[1][i:i+self.batch_size]


            for ct, mri in test_data:
                
                fake_mri = self.generator(ct)

                mri = (mri + 1.0) / 2.0
                fake_mri = (fake_mri + 1.0) / 2.0
                

                fid = evaluate.calculate_fid(mri, fake_mri, 
                    input_shape=(self.flags.crop_size, self.flags.crop_size, 3))

                mse, mae, cs, psnr, ssim = evaluate.get_metrics(mri, fake_mri)

                results.append([fid, mse, mae, cs, psnr, ssim])
                print("metrics: {}{}{}{}{}".format(fid, mse, mae, cs, psnr, ssim))

            results = np.array(results, dtype=object).mean(axis=0)

            filename = "results_{}_{}.log".format(epoch, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            results_dir = os.path.join(self.flags.result_logs, self.flags.name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            log_file = os.path.join(results_dir, filename)
            np.savetxt(log_file, [results], fmt='%.6f', header="fid, mse, mae, cs, psnr,ssim", delimiter=",")


        def save_model(self, flags):
            self.generator.save(self.flags.model_path + self.experiment_name)


        def plot_losses(self, hist):
            
            exp_path = self.hist_dir + self.experiment_name

            if not os.path.exists(exp_path):
                os.makedirs(exp_path)


            # save history to csv   
            hist_df = pd.DataFrame(hist) 
            hist_df.to_csv(exp_path + '/hist.csv')

            losses = ['disc', 'gen', 'feat', 'vgg', 'ssim', 'mae']

            # plot losses
            for loss in losses:
                plt.figure()
                plt.plot(hist[loss + '_loss'])
                plt.plot(hist['val_' + loss + '_loss'])
                plt.legend([loss + '_loss','val_' + loss + '_loss'],loc='upper right')
                plt.savefig(exp_path + '/pix2pix_' + loss + '_loss.png')


