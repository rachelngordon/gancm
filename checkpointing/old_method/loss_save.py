import tensorflow as tf
import tensorflow.keras as kr
import numpy as np

@kr.saving.register_keras_serializable(package="my_package", name="SSIMLoss")
def SSIMLoss(y_true, y_pred):
	y_true = (y_true + 1.0) / 2.0
	y_pred = (y_pred + 1.0) / 2.0
	return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

@kr.saving.register_keras_serializable(package="my_package", name="ssim_l2_a_loss")
def ssim_l2_a_loss(y_true, y_pred):
	yy_true = (y_true + 1.0) / 2.0
	yy_pred = (y_pred + 1.0) / 2.0
	return (1 - tf.reduce_mean(tf.image.ssim(yy_true, yy_pred, 1.0))) + 0.00005*tf.reduce_sum(tf.math.squared_difference(yy_true,yy_pred))

@kr.saving.register_keras_serializable(package="my_package", name="l1_l2_loss")
def l1_l2_loss(y_true, y_pred):
	return tf.reduce_mean(tf.abs(y_true - y_pred)) + tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

@kr.saving.register_keras_serializable(package="my_package", name="mae")
def mae(y_true, y_pred):
	y_true = (y_true + 1.0) / 2.0
	y_pred = (y_pred + 1.0) / 2.0
	return tf.reduce_mean(tf.abs(y_true - y_pred)) 


@kr.saving.register_keras_serializable(package="my_package", name="kl_divergence_loss")
def kl_divergence_loss(mean, variance):
	return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))

@kr.saving.register_keras_serializable(package="my_package", name="generator_loss")
def generator_loss(y):
	y = (y + 1.0) / 2.0
	return tf.reduce_mean(y)

@kr.saving.register_keras_serializable(package="my_package", name="FeatureMatchingLoss")
class FeatureMatchingLoss(kr.losses.Loss):
	def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.mae = kr.losses.MeanAbsoluteError()

	def call(self, y_true, y_pred):
			loss = 0
			y_true = [(y + 1.0) / 2.0 for y in y_true]
			y_pred = [(y + 1.0) / 2.0 for y in y_pred]
			#y_true = (y_true + 1.0) / 2.0
			#y_pred = (y_pred + 1.0) / 2.0
			
			for i in range(len(y_true) - 1):
					loss += self.mae(y_true[i], y_pred[i])
			return loss
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "mae": kr.saving.serialize_keras_object(self.mae),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		mae_config = config.pop("mae")
		mae = kr.saving.deserialize_keras_object(mae_config)
		return cls(mae, **config)

@kr.saving.register_keras_serializable(package="my_package", name="VGGFeatureMatchingLoss")
class VGGFeatureMatchingLoss(kr.losses.Loss):
	def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.encoder_layers = [
					"block1_conv1",
					"block2_conv1",
					"block3_conv1",
					"block4_conv1",
					"block5_conv1",
			]
			self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
			vgg = kr.applications.VGG19(include_top=False, weights="imagenet")
			layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
			self.vgg_model = kr.Model(vgg.input, layer_outputs, name="VGG")
			self.mae = kr.losses.MeanAbsoluteError()

	def call(self, y_true, y_pred):
			y_true = (y_true + 1.0) / 2.0
			y_pred = (y_pred + 1.0) / 2.0
			
			y_true = tf.image.grayscale_to_rgb(y_true)
			y_pred = tf.image.grayscale_to_rgb(y_pred)
			
			y_true = kr.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
			y_pred = kr.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
			real_features = self.vgg_model(y_true)
			fake_features = self.vgg_model(y_pred)
			loss = 0
			for i in range(len(real_features)):
					loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
			return loss
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "encoder_layers": kr.saving.serialize_keras_object(self.encoder_layers),
			"weights": kr.saving.serialize_keras_object(self.weights),
			"vgg": kr.saving.serialize_keras_object(self.vgg),
			"layer_outputs": kr.saving.serialize_keras_object(self.layer_outputs),
			"vgg_model": kr.saving.serialize_keras_object(self.vgg_model),
			"mae": kr.saving.serialize_keras_object(self.mae),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):

		encoder_layers_config = config.pop("encoder_layers")
		encoder_layers = kr.saving.deserialize_keras_object(encoder_layers_config)
		weights_config = config.pop("weights")
		weights = kr.saving.deserialize_keras_object(weights_config)
		vgg_config = config.pop("vgg")
		vgg = kr.saving.deserialize_keras_object(vgg_config)
		layer_outputs_config = config.pop("layer_outputs")
		layer_outputs = kr.saving.deserialize_keras_object(layer_outputs_config)
		vgg_model_config = config.pop("vgg_model")
		vgg_model = kr.saving.deserialize_keras_object(vgg_model_config)
		mae_config = config.pop("mae")
		mae = kr.saving.deserialize_keras_object(mae_config)

		return cls(encoder_layers, weights, vgg, layer_outputs, vgg_model, mae, **config)
	

@kr.saving.register_keras_serializable(package="my_package", name="DiscriminatorLoss")
class DiscriminatorLoss(kr.losses.Loss):
	def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.hinge_loss = kr.losses.Hinge()

	def call(self, is_real, y_pred):
			label = 1.0 if is_real else -1.0
			return self.hinge_loss(label, y_pred)
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "hinge_loss": kr.saving.serialize_keras_object(self.hinge_loss),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		hinge_loss_config = config.pop("hinge_loss")
		hinge_loss = kr.saving.deserialize_keras_object(hinge_loss_config)
		return cls(hinge_loss, **config)


@kr.saving.register_keras_serializable(package="my_package", name="MAE")
class MAE(kr.losses.Loss):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.mae = kr.losses.MeanAbsoluteError()

	def call(self, y_true, y_pred):
		y_true = (y_true + 1.0) / 2.0
		y_pred = (y_pred + 1.0) / 2.0
		return self.mae(y_true, y_pred)
	
	def get_config(self):
		base_config = super().get_config()
		config = {
            "mae": kr.saving.serialize_keras_object(self.mae),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		mae_config = config.pop("mae")
		mae = kr.saving.deserialize_keras_object(mae_config)
		return cls(mae, **config)
	

'''Pix2Pix Losses'''
class p2p_disc_loss(kr.losses.Loss):
	def __init__(self, **kwargs):
			super().__init__(**kwargs)
	
	def call(self, real_img, fake_img):
			real_loss = tf.reduce_mean(real_img)
			fake_loss = tf.reduce_mean(fake_img)
			return fake_loss - real_loss
