import cv2
import numpy as np
import os
import shutil

def Laplace(inpath,outpath):
    
    gray = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
    
    kernel = np.asarray([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    # kernel2 = np.asarray([[1, 1, 1],
    #                       [1, -8, 1],
    #                       [1, 1, 1]])

    des_16S = cv2.filter2D(gray, ddepth=cv2.CV_16SC1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    e = gray - des_16S
    e[e<0] = 0
    e[e>255] = 255
    cv2.imwrite(outpath, e)
    
def cifar10():
    origin = 'data/cifar10'
    enhance = 'data/cifar10_enhance_10%'
    for i in range(0,10):
        path = f'{origin}/train/{i}'
        all_item = os.listdir(path)
        files = all_item[:500]
        if not os.path.exists(f'{enhance}/train/{i}'):
            os.makedirs(f'{enhance}/train/{i}')
        for f in files:
            Laplace(f'{path}/{f}',f'{enhance}/train/{i}/{f}')
    
    shutil.copytree(origin+'/test', enhance+'/test')
    
    
def imagenet():
    origin = 'data/sub-imagenet-20'
    enhance = 'data/sub-imagenet-20_enhance_10%'
    for i in range(0,20):
        path = f'{origin}/train/{i}'
        all_item = os.listdir(path)
        files = all_item[:500]
        if not os.path.exists(f'{enhance}/train/{i}'):
            os.makedirs(f'{enhance}/train/{i}')
        for f in files:
            Laplace(f'{path}/{f}',f'{enhance}/train/{i}/{f}')
    
    shutil.copytree(origin+'/test', enhance+'/test')
    shutil.copytree(origin+'/val', enhance+'/val')
    
    
#cifar10()
imagenet()

        

        
    



