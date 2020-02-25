# coding: utf-8

import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


train_txt_path = os.path.join("..", "..", "Data/train.txt")

CNum = 2000     # 挑选多少图片进行计算

img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)   # shuffle , 随机挑选图片

    for i in range(CNum):
        img_path = lines[i].rstrip().split()[0]         # 去掉字符串末尾的换行符，然后以空格分割字符串，获取图片本地存储地址

        img = cv2.imread(img_path)      #cv2库里面的imread读取图片
        img = cv2.resize(img, (img_h, img_w))   #cv.resize(img,(h,w))将图片数据变成特定高度和长度

        img = img[:, :, :, np.newaxis]      # 在相应位置上增加一个维度，这里维度（x,x,x,1)
        imgs = np.concatenate((imgs, img), axis=3)      # np.concatenate(元组，axis=0) 数组拼接函数，默认维度为0
        print(i)

imgs = imgs.astype(np.float32)/255.         # array.astype(np.float32) 将数据转化为32位浮点型数据


for i in range(3):
    pixels = imgs[:,:,i,:].ravel()  # 拉成一行
    means.append(np.mean(pixels))   
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

