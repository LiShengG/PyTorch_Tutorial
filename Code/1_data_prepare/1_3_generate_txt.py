# coding:utf-8
import os
'''
    为数据集生成对应的txt文件
'''

train_txt_path = os.path.join("..", "..", "Data", "train.txt")          # 训练集图片路径及文件名组成的txt文件，依据此可以找到对应图片
train_dir = os.path.join("..", "..", "Data", "train")                   # 训练集图片数据保存地址

valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")
valid_dir = os.path.join("..", "..", "Data", "valid")


def gen_txt(txt_path, img_dir):             # 制作图片地址索引、标签的txt文件
    f = open(txt_path, 'w')
    
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径，os.listdir（）返回回指定的文件夹包含的文件或文件夹的名字的列表。
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):         # 若不是png文件，跳过.str.endswith("suffix", start, end),判断字符串是否以指定字符或子字符串结尾
                    continue
                label = img_list[i].split('_')[0]           # 以‘_’分割图片文件名，分割后的第一个元素就是类别标签
                img_path = os.path.join(i_dir, img_list[i]) # 每张图片的绝对路径
                line = img_path + ' ' + label + '\n'        # txt文件里存储的信息：图片地址+‘ ’+标签，然后换行。（每个图片占txt一行）
                f.write(line)                               # 写入文件
    f.close()                   # 关闭文件


if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(valid_txt_path, valid_dir)

