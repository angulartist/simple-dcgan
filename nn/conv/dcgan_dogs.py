import argparse
import os

import cv2
import numpy as np
from imutils import build_montages
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dcgan import DCGAN


class DCGANDemo(object):
    def __init__(self, input_path, epochs, nbatch, output_path):
        self.epoch = epochs
        self.nbatch = nbatch

        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        print('[INFO] Loading Dataset...')
        dataset = [
            cv2.imread(os.path.join(self.input_path, filename))
            for filename in os.listdir(self.input_path)
        ]
        train_dataset = np.array(dataset)
        train_dataset = (train_dataset.astype('float') - 127.5) / 127.5

        print('[INFO] Building Generator...')
        gen = DCGAN.build_generator(24, 64, channels=3)
        print(gen.summary())

        print('[INFO] Building Discriminator...')
        disc = DCGAN.build_discriminator(96, 96, 3)
        disc_opts = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / self.epoch)
        disc.compile(loss='binary_crossentropy', optimizer=disc_opts)
        disc.trainable = False

        print('[INFO] Building DCGAN...')
        gan_input = Input(shape=(100,))
        gan_output = disc(gen(gan_input))
        gan = Model(gan_input, gan_output)
        gan_opts = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / self.epoch)
        gan.compile(loss='binary_crossentropy', optimizer=gan_opts)

        print('[INFO] Starting Training...')
        benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

        for epoch in range(self.epoch):
            print(f'[INFO] starting epoch {epoch + 1} of {self.epoch}...')
            batches_per_epoch = int(train_dataset.shape[0] / self.nbatch)
            for i in range(0, batches_per_epoch):
                p = None

                image_batch = train_dataset[i * self.nbatch:(i + 1) * self.nbatch]
                noise = np.random.uniform(-1, 1, size=(self.nbatch, 100))
                gen_images = gen.predict(noise, verbose=0)

                x = np.concatenate((image_batch, gen_images))
                y = ([1] * self.nbatch) + ([0] * self.nbatch)
                (x, y) = shuffle(x, y)

                disc_loss = disc.train_on_batch(x, y)

                noise = np.random.uniform(-1, 1, (self.nbatch, 100))
                gan_loss = gan.train_on_batch(noise, [1] * self.nbatch)

                # Build output montage:

                if i == batches_per_epoch - 1:
                    p = [self.output_path, f'epoch_{str(epoch + 1).zfill(4)}_output.png']

                if p is not None:
                    print(f'[INFO] Step {epoch + 1}_{i}: '
                          f'discriminator_loss={disc_loss} '
                          f'adversarial_loss={gan_loss}')

                    images = gen.predict(benchmark_noise)
                    images = ((images * 127.5) + 127.5).astype('uint8')
                    vis = build_montages(images, (96, 96), (16, 16))[0]
                    p = os.path.sep.join(p)

                    cv2.imwrite(p, vis)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('-e', '--epochs', type=int, default=2500,
                    help='Max epochs to train for')

    ap.add_argument('-b', '--nbatch', type=int, default=128,
                    help='Batch Size >> Lower it on low memory')

    ap.add_argument('-i', '--input', required=True,
                    help='Path to input dataset')

    ap.add_argument('-o', '--output', required=True,
                    help='Path to output directory')

    args = vars(ap.parse_args())

    INPUT = args['input']

    EPOCHS = args['epochs']
    NBATCH = args['nbatch']
    OUTPUT = args['output']

    dcgan_demo = DCGANDemo(INPUT, EPOCHS, NBATCH, OUTPUT)
    dcgan_demo.run()
