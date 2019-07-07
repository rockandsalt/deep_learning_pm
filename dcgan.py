import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def make_discriminator_model(isize, nc, ndf, n_extra_layers=0):
    assert isize % 16 == 0, "isize has to be a multiple of 16"
    input = [isize]*3 + [nc]

    model = tf.keras.Sequential()

    model.add(layers.Conv3D(ndf, kernel_size=4, strides=2,
                            padding="same", use_bias=False, input_shape=input))
    model.add(layers.LeakyReLU(alpha=0.2))

    csize, cndf = isize / 2, ndf

    for _ in range(n_extra_layers):
        model.add(layers.Conv3D(cndf, kernel_size=3,
                                strides=1, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

    while csize > 4:
        out_feat = cndf*2
        model.add(layers.Conv3D(out_feat, kernel_size=4,
                                strides=2, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        cndf = cndf * 2
        csize = csize / 2

    model.add(layers.Conv3D(1, kernel_size=4, strides=1,
                            padding="valid", use_bias=False))
    model.add(layers.Activation("sigmoid"))

    return model


def make_generator_model(isize, nz, nc, ngf, n_extra_layers=0):
    assert isize % 16 == 0, "isize has to be a multiple of 16"
    cngf, tisize = ngf//2, 4

    while tisize != isize:
        cngf = cngf * 2
        tisize = tisize * 2

    input = [isize]*3 + [nz]

    model = tf.keras.Sequential()
    model.add(layers.Conv3DTranspose(cngf, kernel_size=4, strides=1,
                                     padding="valid", use_bias=False, input_shape=input))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    csize, cndf = 4, cngf
    while csize < isize//2:
        model.add(layers.Conv3DTranspose(cngf // 2, kernel_size=4,
                                         strides=2, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        cngf = cngf // 2
        csize = csize * 2

    for _ in range(n_extra_layers):
        model.add(layers.Conv3D(cndf, kernel_size=3,
                                strides=1, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(nc, kernel_size=4,
                                     strides=2, padding="same", use_bias=False))
    model.add(layers.Activation("tanh"))

    return model
