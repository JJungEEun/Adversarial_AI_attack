#!/usr/bin/env python
import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate
from keras.contrib.layers.normalization import InstanceNormalization #InstanceNormation 사용: 이미지 변환에 주로 사용됨
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D,Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam,Adamax
from keras import initializers
from keras.utils import plot_model

# 클래스 인스턴스화
class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (width,height,channels)

        self.Generator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER,metrics=['accuracy'])

        self.save_model()
        self.summary()

    def model(self):
        #down sampling(하향표폰추출): 해상도 낮추기
        input_layer = Input(shape=self.SHAPE)
        
        down_1 = Convolution2D(32, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
        norm_1 = InstanceNormalization()(down_1)

        down_2 = Convolution2D(32*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(down_1)
        norm_2 = InstanceNormalization()(down_2)

        down_3 = Convolution2D(32*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 = InstanceNormalization()(down_3)

        down_4 = Convolution2D(32*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 = InstanceNormalization()(down_4)

        #up sampling(상향표폰추출): 해상도 복구하기(높이기)
        upsample_1 = UpSampling2D()(norm_4)
        up_conv_1 = Convolution2D(32*4, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_1)
        norm_up_1 = InstanceNormalization()(up_conv_1)
        add_skip_1 = Concatenate()([norm_up_1,norm_3]


        upsample_2 = UpSampling2D()(add_skip_1)
        up_conv_2 = Convolution2D(32*2, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_2)
        norm_up_2 = InstanceNormalization()(up_conv_2)
        add_skip_2 = Concatenate()([norm_up_2,norm_2])

        upsample_3 = UpSampling2D()(add_skip_2)
        up_conv_3 = Convolution2D(32, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_3)
        norm_up_3 = InstanceNormalization()(up_conv_3)
        add_skip_3 = Concatenate()([norm_up_3,norm_1])

        # ** InstanceNormalization 계층 사용: 신경망에 화풍모사함수에 대한 일반화 제공
    
                                   
        #출력 계층과 구조
        last_upsample = UpSampling2D()(add_skip_3) #upsampling
        output_layer = Convolution2D(3, kernel_size=4, strides=1, padding='same',activation='tanh')(last_upsample) #출력 layer
        
        return Model(input_layer,output_layer) #모델 구축(입력과 출력 연결)

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator, to_file='/out/Generator_Model.png')

