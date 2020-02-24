# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
"""

import os
import glob
import random
import shutil

dataset_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")
train_dir = os.path.join("..", "..", "Data", "train")
valid_dir = os.path.join("..", "..", "Data", "valid")
test_dir = os.path.join("..", "..", "Data", "test")

train_per = 0.8     # 用于划分数据集的比例80%，训练集占80%
valid_per = 0.1
test_per = 0.1


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    for root, dirs, files in os.walk(dataset_dir):  # 可以得到一个三元tupple(dirpath, dirnames, filenames),——
                                                         #  此目录下所有文件夹路径，所有文件夹名字，起始路径下的所有非目录文件名
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))    # 返回所有匹配的文件路径列表。这里指的是所有'.png'文件
            random.seed(666)
            random.shuffle(imgs_list)       # 用于将一个列表中的元素打乱
            imgs_num = len(imgs_list)       # 所有图像的数量

            train_point = int(imgs_num * train_per)   # 属于train的图像数，占总数的0.8
            valid_point = int(imgs_num * (train_per + valid_per))  #用于验证，占总数0.9

            for i in range(imgs_num):
                if i < train_point:             # 
                    out_dir = os.path.join(train_dir, sDir)     # 将打乱后的数据的80%放在训练集中 ，sDir是文件名
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sDir)     # 将90%放在验证集中
                else:
                    out_dir = os.path.join(test_dir, sDir)      # 将10%放在测试集中

                makedir(out_dir)
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])       # 按路径将文件名和路径分开
                shutil.copy(imgs_list[i], out_path)         # 将imgs_list[i]文件，复制到out_path

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point-train_point, imgs_num-valid_point))
