#!/usr/bin/python
# -*- coding:utf-8 -*-
from PIL import Image
import pytesseract
def imgCov(path):
    img = Image.open(path)
    size = 64,64
    img.resize(size,Image.ANTIALIAS)
    # Lim = img.conv                                   er("L")
    # Lim.save("test.jpg")

    #  convert to grey level image
    Lim  =  img.convert("L" )
    Lim.resize(size,Image.ANTIALIAS).save(path)

    #  setup a converting table with constant threshold
    threshold  =70
    table  =  []
    for  i  in  range( 256 ):
      if  i  <  threshold:
         table.append(0)
      else:
         table.append( 1 )

    #  convert to binary image by the table
    bim  =  Lim.point(table,"1" )

    size = 64,64
    bim.resize(size, Image.ANTIALIAS).save(path)
import os
if __name__ == '__main__':
    dir = '../datasets/train'
    list = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isfile(path):
            imgCov(path)

#text = pytesseract.image_to_string(Image.open('fun_binary.jpg'),lang='chi_sim')
