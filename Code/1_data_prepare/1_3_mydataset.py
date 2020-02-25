# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset                # 数据类基类


class MyDataset(Dataset):           # 从硬盘中读取图像数据，然后做相应的transform.
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')    # 只读方式打开txt_path文件
        imgs = []               # 创建图像列表
        for line in fh:         # 按行读取  
            line = line.rstrip()    # 删除 string 字符串末尾的指定字符（默认为空格）.
            words = line.split()    # 按特定字符分割字符串，默认所有空字符，包括 空格，换行，和制表符。我们在制作数据集txt的时候,每行是以图片地址+空格+分类的格式
            imgs.append((words[0], int(words[1])))  

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):               # 类dataset里面的自带函数
        fn, label = self.imgs[index]            # 用于迭代时获取每个图片的地址信息和标签信息。根据地址用PIL库里面的Image类的open函数获取图片像素数据
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label               # 返回图片数据，标签

    def __len__(self):
        return len(self.imgs)
