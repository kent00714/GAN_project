from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import mse , binary_crossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical , plot_model
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import cv2 , os


class VAE():

	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 2

		self.optimizer = SGD(lr=0.00001, momentum=0.9, decay=0.0001, nesterov=False)


		self.decoder = self.build_decoder()
		self.decoder.summary()

		self.encoder = self.build_encoder()
		self.encoder.summary()


		self.outputs = self.decoder(self.encoder(self.inputs)[2])

		self.vae = Model(self.inputs, self.outputs, name='VAE')
		plot_model(self.vae , to_file='model.png')


	def build_encoder(self):
		initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

		self.inputs = Input(shape=self.img_shape, name='Encoder_input')

		conv1_1 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(self.inputs)
		conv1_2 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_1)
		pool1 = MaxPooling2D(padding="same")(conv1_2)

		conv1_3 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(pool1)
		conv1_4 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_3)
		conv1_5 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_4)
		pool2 = MaxPooling2D(padding="same")(conv1_5)

		conv1_6 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(pool2)
		conv1_7 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_6)


		conv_flat = Flatten()(conv1_7)

		fc_1 = Dense(64,activation='relu',kernel_initializer=initializer)(conv_flat)

		self.z_mean = Dense(self.latent_dim, name='z_mean')(fc_1)
		self.z_log_var = Dense(self.latent_dim, name='z_log_var')(fc_1)

		z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

		return Model(self.inputs, [self.z_mean, self.z_log_var, z], name='encoder')

	def build_decoder(self):
		initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
		self.latent_inputs = Input(shape=(self.latent_dim,), name='Decoder_input')

		d_fc_1 = Dense(64, activation='relu')(self.latent_inputs)

		d_flat = Dense(3136, activation='relu')(d_fc_1)
		d_reshape = Reshape((7,7,64))(d_flat)

		d_conv1_1 = Conv2DTranspose(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_reshape)
		d_conv1_2 = Conv2DTranspose(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_conv1_1)
		upsamp1 = UpSampling2D((2,2))(d_conv1_2)

		d_conv1_3 = Conv2DTranspose(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(upsamp1)
		d_conv1_4 = Conv2DTranspose(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_conv1_3)
		d_conv1_5 = Conv2DTranspose(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_conv1_4)
		upsamp2 = UpSampling2D((2,2))(d_conv1_5)

		d_conv1_6 = Conv2DTranspose(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(upsamp2)
		d_conv1_7 = Conv2DTranspose(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_conv1_6)

		self.outputs = Conv2DTranspose(1 , kernel_size=(3, 3) , padding='same' ,activation='sigmoid')(d_conv1_7)


		return Model(self.latent_inputs, self.outputs, name='decoder')


	def sampling(self , args):

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]

		epsilon = K.random_normal(shape=(batch, dim))

		return z_mean + K.exp( z_log_var) * epsilon


	def vae_loss(self):

		xent_loss = 784 * binary_crossentropy(K.flatten(self.inputs), K.flatten(self.outputs))
		kl_loss = (-0.5 - 0.5*self.z_log_var + 0.5*K.square(self.z_mean) + 0.5*K.exp(self.z_log_var))
		kl_loss = K.sum(kl_loss, axis=-1)

		return K.mean(xent_loss + kl_loss)


	def train(self , epochs , batch_size , load_weights=None):

		(X_train, _), (X_test, _) = mnist.load_data()

		X_train = X_train.astype('float32') / 255
		X_test = X_test.astype('float32') / 255

		X_train = np.reshape(X_train, [-1, 28, 28, 1])
		X_test = np.reshape(X_test, [-1, 28, 28, 1])


		self.vae.add_loss(self.vae_loss())
		
		if (load_weights != None):
			self.vae.load_weights(load_weights)

		self.vae.compile(optimizer=self.optimizer)
		self.vae.summary()

		self.vae.fit(X_train,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(X_test, None))

		self.vae.save_weights('vae_mlp_mnist.h5')


	def sample_interval(self, epoch):
		r, c = 10, 10
		z = np.random.normal(size=(100, self.latent_dim))
		gen_imgs = self.decoder.predict(z)

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("vae_images/mnist_%d.png" % epoch)
		plt.close()


def plot_feature(model_path):
	(X_train,Y_train), (X_test, Y_test) = mnist.load_data()

	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255

	X_train = np.reshape(X_train, [-1, 28, 28, 1])
	X_test = np.reshape(X_test, [-1, 28, 28, 1])

	data = [X_train ,X_test]
	label = [Y_train , Y_test]

	vae = VAE()
	vaemodel = vae.vae
	vaemodel.load_weights(model_path)

	for i in range(2):
		z_mean, _, _ = vae.encoder.predict(data[i],
		                               batch_size=128)

		plt.figure(figsize=(12, 10))
		plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label[i])
		plt.colorbar()
		plt.xlabel("z[0]")
		plt.ylabel("z[1]")
		plt.show()


def plot_feature_map(model_path):
	vae = VAE()
	vaemodel = vae.vae
	vaemodel.load_weights(model_path)

	decoder = vae.decoder

	n = 30  
	digit_size = 28
	figure = np.zeros((digit_size * n, digit_size * n))
	grid_x = np.linspace(-1.5, 1.5, n)
	grid_y = np.linspace(-1.5, 1.5, n)

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])
			x_decoded = decoder.predict(z_sample)
			digit = x_decoded[0].reshape(digit_size, digit_size)
			figure[i * digit_size: (i + 1) * digit_size,
				j * digit_size: (j + 1) * digit_size] = digit

	plt.figure(figsize=(10, 10))
	plt.imshow(figure, cmap='Greys_r')
	plt.show()


if __name__ == '__main__':

	vae = VAE()
	vae.train(epochs=10000, batch_size=256, load_weights="vae_mlp_mnist.h5")
	# plot_feature('vae_mlp_mnist.h5')
	# plot_feature_map('vae_mlp_mnist.h5')


	