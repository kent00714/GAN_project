from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import numpy as np


class BIGAN():
	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 2

		self.optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False)
		self.initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)


		self.discriminator = self.build_discriminator()
		self.discriminator.summary()

		self.generator = self.build_generator()

		self.encoder = self.build_encoder()

		self.discriminator.trainable = False

		z = Input(shape=(self.latent_dim, ))
		img_ = self.generator(z)
		self.generator.summary()

		img = Input(shape=self.img_shape)
		z_ = self.encoder(img)
		self.encoder.summary()

		fake = self.discriminator([z, img_])
		valid = self.discriminator([z_, img])

		self.bigan_generator = Model([z, img], [fake, valid])
		self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
			optimizer=self.optimizer)


	def build_encoder(self):
		model = Sequential()

		model.add(Conv2D(64 , kernel_size=(3, 3) , padding='same',activation='tanh',kernel_initializer=self.initializer,input_shape=self.img_shape))
		model.add(Conv2D(64 , kernel_size=(3, 3) , padding='same',activation='tanh',kernel_initializer=self.initializer))
		model.add(MaxPooling2D(padding='same'))

		model.add(Conv2D(128, kernel_size=(3, 3) , padding='same',activation='tanh',kernel_initializer=self.initializer))
		model.add(Conv2D(128, kernel_size=(3, 3) , padding='same',activation='tanh',kernel_initializer=self.initializer))
		model.add(MaxPooling2D(padding='same'))

		model.add(Flatten())

		model.add(Dense(128,activation='tanh'))
		model.add(Dense(32,activation='tanh'))
		model.add(Dense(self.latent_dim, name='Encoder_output_layer'))


		img = Input(shape=self.img_shape, name='Encoder_input_layer')
		z = model(img)

		model.summary()
		return Model(img, z)

	def build_generator(self):
		model = Sequential()

		model.add(Dense(32, activation='relu' ,input_dim=self.latent_dim))
		model.add(Dense(128,activation='relu'))

		model.add(Dense(6272,activation='relu'))
		model.add(Reshape((7,7,128)))

		model.add(Conv2DTranspose(128 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer))
		model.add(LeakyReLU(alpha=0.5))
		model.add(Conv2DTranspose(128 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer))
		model.add(LeakyReLU(alpha=0.5))
		model.add(UpSampling2D((2, 2)))

		model.add(Conv2DTranspose(64 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer))
		model.add(LeakyReLU(alpha=0.5))
		model.add(Conv2DTranspose(64 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer))
		model.add(LeakyReLU(alpha=0.5))
		model.add(UpSampling2D((2, 2)))

		model.add(Conv2DTranspose(1 , kernel_size=(3, 3) , padding='same',activation='tanh',kernel_initializer=self.initializer, name='Generator_output_layer'))
		

		z = Input(shape=(self.latent_dim,), name='Generator_input_layer')
		gen_img = model(z)

		model.summary()
		return Model(z, gen_img)

	def build_discriminator(self):

		z = Input(shape=(self.latent_dim, ))
		img = Input(shape=self.img_shape)

		img_in = Conv2D(8 , kernel_size=(5, 5) , padding='same',kernel_initializer=self.initializer)(img)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		img_in = Conv2D(8 , kernel_size=(5, 5) , padding='same',kernel_initializer=self.initializer)(img_in)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		img_in = MaxPooling2D(padding='same')(img_in)

		img_in = Conv2D(2 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer)(img_in)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		img_in = Conv2D(2 , kernel_size=(3, 3) , padding='same',kernel_initializer=self.initializer)(img_in)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		img_in = MaxPooling2D(padding='same')(img_in)
		img_in = Flatten()(img_in)

		img_in = Dense(16 ,kernel_initializer=self.initializer)(img_in)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		img_in = Dense(2 ,kernel_initializer=self.initializer)(img_in)
		img_in = LeakyReLU(alpha=0.2)(img_in)
		
		model = concatenate([z, (img_in)])

		model = Dense(256)(model)
		model = LeakyReLU(alpha=0.2)(model)

		model = Dense(512)(model)
		model = LeakyReLU(alpha=0.2)(model)

		model = Dense(512)(model)
		model = LeakyReLU(alpha=0.2)(model)

		validity = Dense(1, activation="sigmoid")(model)

		return Model([z, img], validity)

	def train(self, epochs, batch_size=128, sample_interval=50):

		(X_train, _), (_, _) = mnist.load_data()

		X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		X_train = np.expand_dims(X_train, axis=3)

		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		self.discriminator.compile(loss=['binary_crossentropy'],
		optimizer=self.optimizer,
		metrics=['accuracy'])

		os.mkdir("model")
		os.mkdir("images")

		for epoch in range(epochs):

			z = np.random.randn(batch_size , self.latent_dim)/3
			imgs_ = self.generator.predict(z)

			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]
			z_ = self.encoder.predict(imgs)


			d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
			d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


			g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

			if epoch % 1000 == 0:
				print ("Iteration: %d [D loss: %.5f] [G loss: %.5f]" % (epoch, d_loss[0], g_loss[0]))
				
			if epoch % 10000 == 0:
				self.bigan_generator.save('model/BiGAN_model_%d.h5' %(epoch)) 

			if epoch % sample_interval == 0:
				self.sample_interval(epoch)

	def sample_interval(self, epoch):
		r, c = 5, 5
		z = np.random.normal(size=(25, self.latent_dim))
		gen_imgs = self.generator.predict(z)

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


if __name__ == '__main__':
	bigan = BIGAN()
	bigan.train(epochs=4000000, batch_size=128, sample_interval=1000)