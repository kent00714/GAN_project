from __future__ import print_function, division


import tensorflow
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
from tensorflow.keras.utils import to_categorical,plot_model
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import cv2 , os


class AVAE():
	def __init__(self, vae_weight=None, D_weight=None):
		tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 2

		self.optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0001, nesterov=False)
		self.Doptimizer = SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False)



		self.discriminator = self.build_discriminator()
		self.discriminator.summary()
		self.discriminator.compile(loss=['binary_crossentropy'],optimizer=self.Doptimizer,metrics=['accuracy'])


		self.decoder = self.build_decoder()
		self.decoder.summary()

		self.encoder = self.build_encoder()
		self.encoder.summary()


		self.vae_outputs = self.decoder(self.encoder(self.inputs)[2])
		self.vae = Model(self.inputs, self.vae_outputs, name='VAE')

		self.discriminator.trainable = False

		self.fake = self.discriminator(self.decoder(self.encoder(self.inputs)[2]))
		self.valid = self.discriminator(self.inputs)
		
		self.avae = Model(self.inputs , [self.fake,self.valid] , name="avae")
		self.avae.add_loss(self.avae_loss())
		self.avae.compile(optimizer=self.optimizer)
		self.avae.summary()
		plot_model(self.avae , to_file='model.png')

	def build_encoder(self):
		initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

		self.inputs = Input(shape=self.img_shape, name='Encoder_input')

		conv1_1 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(self.inputs)
		conv1_2 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_1)
		pool1 = MaxPooling2D(padding="same")(conv1_2)

		conv1_3 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(pool1)
		conv1_4 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_3)
		pool2 = MaxPooling2D(padding="same")(conv1_4)

		conv_flat = Flatten()(pool2)

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

		d_conv1_3 = Conv2DTranspose(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(upsamp1)
		d_conv1_4 = Conv2DTranspose(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, padding='same' ,activation='relu')(d_conv1_3)
		upsamp2 = UpSampling2D((2,2))(d_conv1_4)

		self.outputs = Conv2DTranspose(1 , kernel_size=(3, 3) , padding='same' ,activation='sigmoid')(upsamp2)


		return Model(self.latent_inputs, self.outputs, name='decoder')

	def build_discriminator(self):


		self.img = Input(shape=self.img_shape, name='discriminator_input')
		flat = Flatten()(self.img)

		fc2_1 = Dense(256,activation='relu')(flat)
		fc2_2 = Dense(64,activation='relu')(fc2_1)
		fc2_3 = Dense(16,activation='relu')(fc2_2)

		self.validity = Dense(1, activation="sigmoid")(fc2_3)

		return Model(self.img, self.validity, name='discriminator')

	def sampling(self , args):

		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]

		epsilon = K.random_normal(shape=(batch, dim))

		return z_mean + K.exp( z_log_var) * epsilon

	def avae_loss(self):
		
		xent_loss = 784 * binary_crossentropy(K.flatten(self.inputs), K.flatten(self.vae_outputs))
		kl_loss = (-0.5 - 0.5*self.z_log_var + 0.5*K.square(self.z_mean) + 0.5*K.exp(self.z_log_var))
		kl_loss = K.sum(kl_loss, axis=-1)
		vae_loss = K.mean(xent_loss + kl_loss)

		valid = K.variable(np.ones((256, 1)))
		fake = K.variable(np.zeros((256, 1)))
		D_loss = K.mean(binary_crossentropy(self.valid , fake)+binary_crossentropy(self.fake , valid))

		w = 0.4
		loss = w*vae_loss +(1-w)*D_loss	
		
		return loss

	def train(self, epochs, batch_size=128):

		# --------Encoder data 60000
		(img, _), (_, _) = mnist.load_data()
		img = img.astype('float32') / 255
		img = np.reshape(img, [-1, 28, 28, 1])


		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		os.mkdir("images")
		os.mkdir("Model")
		
		for epoch in range(epochs+1):


			idx = np.random.randint(0, img.shape[0], batch_size)
			imgs = img[idx]
			fake_imgs = self.vae.predict(imgs)

			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			#  Train Generator

			g_loss = self.avae.train_on_batch(imgs)
			
				
			if epoch % 1000 == 0:
				print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
				self.avae.save_weights('Model/AVAE_model_%d.h5' %(epoch)) 
				self.sample_interval(epoch)

	def sample_interval(self, epoch):
		r, c = 5, 5
		z = np.random.normal(size=(25, self.latent_dim))
		gen_imgs = self.decoder.predict(z)

		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
				fig.savefig("images/mnist_%d.png" % epoch)
				plt.close()


def plot_feature(model_path):
	(X_train,Y_train), (X_test, Y_test) = mnist.load_data()

	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255

	X_train = np.reshape(X_train, [-1, 28, 28, 1])
	X_test = np.reshape(X_test, [-1, 28, 28, 1])

	data = [X_train ,X_test]
	label = [Y_train , Y_test]

	avae_class = AVAE()
	avaemodel = avae_class.avae
	avaemodel.load_weights(model_path)


	for i in range(2):
		z_mean, _, _ = bigan.encoder.predict(data[1],
		                               batch_size=128)

		plt.figure(figsize=(12, 10))
		plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label[1])
		plt.colorbar()
		plt.xlabel("z[0]")
		plt.ylabel("z[1]")
		plt.show()


def plot_feature_map(model_path):
	avae_class = AVAE()
	avaemodel = avae_class.avae
	avaemodel.load_weights(model_path)

	decoder = bigan.decoder

	n = 30  
	digit_size = 28
	figure = np.zeros((digit_size * n, digit_size * n))
	grid_x = np.linspace(-4, 4, n)
	grid_y = np.linspace(-4, 4, n)

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
	avae = AVAE()
	avae.train(epochs=2000000, batch_size=256)

	# plot_feature("Model/AVAE_model_2000000.h5")
	# plot_feature_map("Model/AVAE_model_2000000.h5")

