#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.layers import Input
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from PIL import Image
import random
import numpy as np

class Trainer:
    # 인스턴스화 
    def __init__(self, height = 256, width = 256, channels=3, epochs = 50000, batch = 1, checkpoint = 50, train_data_path = '',test_data_path=''):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint

        # 모든 데이터를 해당 클래스 변수에 적재하기
        self.X_train_A, self.H_A, self.W_A, self.C_A = self.load_data(train_data_path_A)
        self.X_train_B, self.H_B, self.W_B, self.C_B = self.load_data(train_data_path_B)
        self.X_test_A, self.H_A_test, self.W_A_test, self.C_A_test = self.load_data(test_data_path_A)
        self.X_test_B, self.H_A_test, self.W_A_test, self.C_A_test = self.load_data(test_data_path_B)        

        #  A->B, B->A 생성기
        self.generator_A_to_B = Generator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.generator_B_to_A = Generator(height=self.H_B, width=self.W_B, channels=self.C_B)

        self.orig_A = Input(shape=(self.W, self.H, self.C)) #원본 이미지 X
        self.orig_B = Input(shape=(self.W, self.H, self.C))

        self.fake_B = self.generator_A_to_B.Generator(self.orig_A) #가짜 이미지 생성 X -> Y
        self.fake_A = self.generator_B_to_A.Generator(self.orig_B)
        
        self.reconstructed_A = self.generator_B_to_A.Generator(self.fake_B) #재구성한 이미지 X -> Y -> X
        self.reconstructed_B = self.generator_A_to_B.Generator(self.fake_A)

        self.id_A = self.generator_B_to_A.Generator(self.orig_A) # 항등함수: 원본이미지를 가져와서 같은 화풍으로 변환 X -> X
        self.id_B = self.generator_A_to_B.Generator(self.orig_B)
        

        # 판별기
        self.discriminator_A = Discriminator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.discriminator_B = Discriminator(height=self.H_B, width=self.W_B, channels=self.C_B)
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.valid_A = self.discriminator.Discriminator(self.fake_A) 
        self.valid_B = self.discriminator.Discriminator(self.fake_B)

        # 적대적 모델 
        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid_A, self.valid_B, self.reconstructed_A, self.reconstructed_B, self.id_A, self.id_B]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs, lambda_cycle=10.0, lambda_id=1.0)
        
        
     # 훈련 메서드: 네 개의 판별기 훈련( A 이미지와 B 이미지 평가하는 판별기, fake_A와 fake_B 생성기를 검사하는 유효성 판별기)
    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)
            while min(len(X_train_A_temp), len(X_train_B_temp)) > self.Batch:
                # 배치들을 추적한다
                b = b + 1

                # 훈련 판별기
                # 이 훈련 배치에 대해 진짜 이미지들을 잡는다
                count_real_images = int(self.BATCH)
                
                starting_indexs = randint(0, (min(len(X_train_A_temp), len(X_train_B_temp))-count_real_images))
                real_images_raw_A = X_train_A_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                real_images_raw_B = X_train_B_temp[ starting_indexs : (starting_indexs + count_real_images) ]

                # 남는 게 없을 때까지 사용된 이미지를 삭제
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_indexs,(starting_indexs + count_real_images)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_indexs,(starting_indxes + count_real_images)),0)

                batch_A = real_images_raw_A.reshape( 1, self.W_A, self.H_A, self.C_A )
                batch_B = real_images_raw_B.reshape( 1, self.W_B, self.H_B, self.C_B )

                # 개별 판별기 훈련 - 배치 개발을 통해 훈련 과정에 레이블 잡음 도입
                if self.flipCoin():
                    x_batch_A = batch_A
                    x_batch_B = batch_B
                    y_batch_A = np.ones([count_real_images,1])
                    y_batch_B = np.ones([count_real_images,1])
                else:
                    x_batch_B = self.generator_A_to_B.Generator.predict(batch_A)
                    x_batch_A = self.generator_B_to_A.Generator.predict(batch_B)
                    y_batch_A = np.ones([self.BATCH, 1])
                    y_batch_B = np.ones([self.BATCH, 1])
                    
                # 새롭게 개발된 배치를 사용해 판별기 A와 B 작성
                
                self.discriminator_A.Discriminator.trainable = True
                discriminator_loss_A = self.discriminator_A.Discriminator.train_on_batch(x_batch_A, y_batch_A)[0]
                self.discriminator_A.Discriminator.trainable = False
                
                self.discriminator_B.Discriminator.trainable = True
                discriminator_loss_B = self.discriminator_B.Discriminator.train_on_batch(x_batch_B, y_batch_B)[0]
                self.discriminator_B.Discriminator.trainable = False
                
                # 손실 기록
                # 생성기 훈련시 수렴이 잘 될 수 있게 레이블 뒤집기
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels = np.ones([self.BATCH,1])

                generator_loss = self.gan.gan_model.train_on_batch([x_batch_A, x_batch_B],
                                                                   [y_generated_labels, y_generated_labels, 
                                                                    x_batch_A, x_batch_B, x_batch_A, x_batch_B])    

                # 배치의 출력을 주기적으로 그리면서 에포크 수준 점검
                print ('Batch: '+str(int(b))+ ',
                       [Discrimiator_A :: Loss: '+str(discrimiator_loss_A)+'],
                       [ Generator :: Loss: '+str(generator_loss)+']')
                       
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)

            print ('Epoch: '+str(int(e))+', [Discriminator_A :: Loss:'+str(discrimiator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                        
        return
                       
    
     def load_data(self,data_path):
         # load_data(): 폴더에 대한 경로를 나타내는 문자열을 예상하고 해당 폴더에 끝나는 특정 파일이 있는 모든 이미지를 읽는다
         # (numpy 배열 형태로 리턴)
        listOFFiles = self.grabListOfFiles(data_path,extension="jpg")
        imgs_temp = np.array(self.grabArrayOfImages(listOFFiles))
        imgs_A = []
        imgs_B = []
        for img in imgs_temp:
            imgs_A.append(img[:,:self.H])
            imgs_B.append(img[:,self.H:])

        imgs_A_out = self.norm_and_expand(np.array(imgs_A))
        imgs_B_out = self.norm_and_expand(np.array(imgs_B))

        return imgs_A_out, imgs_B_out

    def norm_and_expand(self,arr):
        arr = (arr.astype(np.float32) - 127.5)/127.5
        normed = np.expand_dims(arr, axis=3)
        return normed

    def grabListOfFiles(self,startingDirectory,extension=".webp"):
        # 파일 목록 입수
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))
        return listOfFiles
    
    def flipCoin(self, chance=0.5):
        # 동전 던지기 작업 수행
        return np.randombinomial(1, chance)

    def grabArrayOfImages(self,listOfFiles,gray=False):
        imageArr = []
        for f in listOfFiles:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")
            imData = np.asarray(im)
            imageArr.append(imData)
        return imageArr

    def plot_checkpoint(self,b):
        # 각 검사점에서, 테스트 집합에서 예를 가져오고 화풍을 A에서 B로 전송한 후 다시 A로 전송
        orig_filename = "/out/batch_check_"+str(b)+"_original.png"

        r, c = 3, 3
        random_inds = random.sample(range(len(self.X_test_A)),3)
        imgs_A = self.X_test_A[random_inds].reshape(3, self.W, self.H, self.C )
        imgs_B = self.X_test_B[random_inds].reshape( 3, self.W, self.H, self.C )
        fake_A = self.generator.Generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Style', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/out/batch_check_"+str(b)+".png")
        plt.close('all')

        return