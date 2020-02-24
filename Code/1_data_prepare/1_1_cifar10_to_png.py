# coding:utf-8
"""
    将cifar10的data_batch_12345 转换成 png格式的图片
    每个类别单独存放在一个文件夹，文件夹名称为0-9
"""
from scipy.misc import imsave
import numpy as np
import os
import pickle


data_dir = os.path.join("..", "..", "Data", "cifar-10-batches-py")          //构建文件相对地址:os.path.join("","")
train_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_train")
test_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")

Train = False   # 不解压训练集，仅解压测试集

# 解压缩，返回解压后的字典
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')   # 使用pickle.load 打开数据文件，返回字典格式
    return dict_

def my_mkdir(my_dir):                       
    if not os.path.isdir(my_dir):                   # os.path.isdir(A) 文件目录A是否存在
        os.makedirs(my_dir)                         # os.makedirs(A) 新建名为A的目录


# 生成训练集图片，
if __name__ == '__main__':
    if Train:
        for j in range(1, 6):
            data_path = os.path.join(data_dir, "data_batch_" + str(j))  # 文件名依次为data_batch_1，data_batch_2，data_batch_3，data_batch_4，data_batch_5...
            train_data = unpickle(data_path)  # 解包文件
            print(data_path + " is loading...")

            for i in range(0, 10000):      #每个文件种总共有10000个图片，
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))   # train_data[b'data'][0]为第一张图片的数据，将每个图片数据变为CxHxW的形式
                img = img.transpose(1, 2, 0)        #将图片数据转置为HxWxC的形式（高度x宽度x通道数）

                label_num = str(train_data[b'labels'][i])       # 当前图片对应的分类标签
                o_dir = os.path.join(train_o_dir, label_num)         
                my_mkdir(o_dir)                                  # 建立一个分类物体的图片文件夹，里面将存放属于本类的图片  

                img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'       #为每张图片取名
                img_path = os.path.join(o_dir, img_name)                    #每张图片的地址加上他自己的名字
                imsave(img_path, img)                               # 保存图片 imsave(name, img)
            print(data_path + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    test_data_path = os.path.join(data_dir, "test_batch")
    test_data = unpickle(test_data_path)
    for i in range(0, 10000):
        img = np.reshape(test_data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)

        label_num = str(test_data[b'labels'][i])        # 
        o_dir = os.path.join(test_o_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + '_' + str(i) + '.png'        
        img_path = os.path.join(o_dir, img_name)
        imsave(img_path, img)            # imgeio.imwrite(path,img)作用相同，分别引入(from scipy.misc import imsave)和（import imageio)

    print("test_batch loaded.")
