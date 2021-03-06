#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate
from keras.contrib.layers.normalization import InstanceNormalization # contributors: 사용자 정의 계층 가져오기
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    #인스턴스화 
    def __init__(self, width = 28, height= 28, channels = 1, starting_filters=64):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)        
        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'] )#mse: 평균 제곱 오차

        self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(self.SHAPE)
        
        up_layer_1 = Convolution2D(64, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

        up_layer_2 = Convolution2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        norm_layer_1 =  normalization()(up_layer_2)

        up_layer_3 = Convolution2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_1)
        norm_layer_2 = normalization()(up_layer_3)

        up_layer_4 = Convolution2D(64*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_2)
        norm_layer_3 = normalization()(up_layer_4)
        
        
        # 출력계층
        output_layer = Convolution2D(1, kernel_size=4, strides=1, padding='same')(norm_layer_3)
        output_layer_1 = Flatten(output_layer) #평평하게 만들기
        output_layer_2 = Dense(1, activation='sigmoid')(output_layer_1)
        
        return Model(input_layer,output_layer_2)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator, to_file='/out/Discriminator_Model.png')

