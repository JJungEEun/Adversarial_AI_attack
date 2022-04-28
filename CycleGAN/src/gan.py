#!/usr/bin/env python3
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.utils import plot_model

class GAN(object):
    # 인스턴스화
    def __init__(self, model_inputs=[],model_outputs=[], lambda_cycle=10.0, lambda_id=1.0): #lambda값: 재구성 세대에 대한 손실함수들의 값
                                                                                            # id 파라미터는 cycle 변수의 10%
        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)
        self.inputs = model_inputs
        self.outputs = model_outputs
        
        # 전달된 입력 및 출력을 사용해 모델 만들기
        self.gan_model = Model(inputs = self.inputs, outputs = self.outputs) #self.input: 클래스에서 인스턴스화됨, 두 개의 케라스 입력 클래스의 배열로 표현
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        
        # 6개의 모델(4개의 생성기와 2개의 판별기)이 적대적 훈련에 쓸 수 있게 출력 배열을 이룸
        # 생성기: X->Y, Y->X, X->Y->X, Y->X->Y
        self.gan_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                            loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
                            optimizer=self.OPTIMIZER)
        
        self.save_model()
        self.summary()

    def model(self):
        model = Model()
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model, to_file='/out/GAN_Model.png')
