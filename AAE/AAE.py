from __future__ import print_function, division

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU, GaussianNoise
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import mse , binary_crossentropy
from tensorflow.keras.initializers import RandomNormal, glorot_normal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical,plot_model
import tensorflow.keras.backend as K
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2 , os ,sys
class AdversarialAutoencoder():
    def __init__(self):
        tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 2

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        self.discriminator.trainable = False

        validity = self.discriminator(encoded_repr)

        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

        plot_model(self.adversarial_autoencoder , to_file='model.png')

    def build_encoder(self):

        initializer = glorot_normal()

        self.inputs = Input(shape=self.img_shape, name='Encoder_input')

        conv1_1 = Conv2D(128 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(self.inputs)
        conv1_2 = Conv2D(128 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_1)
        drop1 = Dropout(rate=0.8)(conv1_2)

        conv1_3 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(drop1)
        conv1_4 = Conv2D(64 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_3)
        pool1 = MaxPooling2D(padding="same")(conv1_4)
        drop2 = Dropout(rate=0.8)(pool1)

        conv1_3 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(drop2)
        conv1_4 = Conv2D(32 , kernel_size=(3, 3) ,kernel_initializer=initializer, strides=(1,1) , padding='same',activation='relu')(conv1_3)
        pool2 = MaxPooling2D(padding="same")(conv1_4)
        drop3 = Dropout(rate=0.8)(pool2)

        conv_flat = Flatten()(drop3)

        h = Dense(64,activation='relu',kernel_initializer=initializer)(conv_flat)

        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([mu, log_var])

        return Model(self.inputs, latent_repr)

    def build_decoder(self):

        initializer = glorot_normal()

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512,kernel_initializer=initializer,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512,kernel_initializer=initializer,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.img_shape),kernel_initializer=initializer, activation='sigmoid'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)


    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.8))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.8))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def sampling(self , args):

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + epsilon * K.exp( z_log_var / 2) 

    def train(self, epochs, batch_size=128, sample_interval=50):

        (X_train, _), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32)) / 255.0
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        os.mkdir("Model")
        os.mkdir("images")

        for epoch in range(epochs+1):


            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))


            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])


            if epoch % 1000 == 0:
                sys.stdout.write("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]\n" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
                sys.stdout.flush()
                self.adversarial_autoencoder.save_weights("Model/AAE_Model_%d.h5" %(epoch))



    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
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

    AAEclass = AdversarialAutoencoder()
    aaemodel = AAEclass.adversarial_autoencoder
    aaemodel.load_weights(model_path)


    for i in range(2):
        z_mean = AAEclass.encoder.predict(data[i],
                                       batch_size=128)

        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label[i])
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()


def plot_feature_map(model_path):
    AAEclass = AdversarialAutoencoder()
    aaemodel = AAEclass.adversarial_autoencoder
    aaemodel.load_weights(model_path)

    decoder = AAEclass.decoder

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

def write_feature(model_path):
    AAEclass = AdversarialAutoencoder()
    aaemodel = AAEclass.adversarial_autoencoder
    aaemodel.load_weights(model_path)

    folder = os.listdir("testing")
    for i in folder:
        file_list = os.listdir("testing/"+i)
        f=open("%s.txt" %(i) , "a+")

        for file in file_list:
            location = "testing/" + i + "/" + file
            img = cv2.imread(location , 0)
            img = img.astype('float32') / 255
            img = np.reshape(img , [-1 , 28 , 28, 1])
            feature = AAEclass.encoder.predict(img)[0]
            
            for j in range (len(feature)):
                f.write(str(feature[j])+"\t")
            f.write("\n")
            
        f.close()

if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=3000000, batch_size=128, sample_interval=1000)
    # plot_feature("AAE_Model_2379000.h5")
    # plot_feature_map("AAE_Model_2379000.h5")
    # write_feature("AAE_Model_2379000.h5")
