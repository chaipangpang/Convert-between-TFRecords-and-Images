# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:19:09 2017

@author: zhangxu
"""

import os 
import tensorflow as tf 
from PIL import Image  

#图片路径
cwd = 'F:\\flowersdata\\trainimages\\'
#文件路径
filepath = 'F:\\flowersdata\\tfrecord\\'
#存放图片个数
bestnum = 1000
#第几个图片
num = 0
#第几个TFRecord文件
recordfilenum = 0
#类别
classes=['daisy',
         'dandelion',
         'roses',
         'sunflowers',
         'tulips']
#tfrecords格式文件名
ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
writer= tf.python_io.TFRecordWriter(filepath+ftrecordfilename)
#类别和路径
for index,name in enumerate(classes):
    print(index)
    print(name)
    class_path=cwd+name+'\\'
    for img_name in os.listdir(class_path): 
        num=num+1
        if num>bestnum:
          num = 1
          recordfilenum = recordfilenum + 1
          #tfrecords格式文件名
          ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
          writer= tf.python_io.TFRecordWriter(filepath+ftrecordfilename)
        #print('路径',class_path)
        #print('第几个图片：',num)
        #print('文件的个数',recordfilenum)
        #print('图片名：',img_name)
        
        img_path = class_path+img_name #每一个图片的地址
        img=Image.open(img_path,'r')
        size = img.size
        print(size[1],size[0])
        print(size)
        #print(img.mode)
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(
             features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        })) 
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

