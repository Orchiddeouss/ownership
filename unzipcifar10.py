import os
import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_from_batch(batch, directory):
    data = batch[b'data']
    labels = batch[b'labels']
    for i, img_array in enumerate(data):
        img_array = img_array.reshape((3, 32, 32)).transpose(1, 2, 0)
        img = Image.fromarray(img_array)
        label = labels[i]
        label_dir = os.path.join(directory, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        img.save(os.path.join(label_dir, f'{str(label)}_{i}.png'))

def main():
    # 解压训练集
    for i in range(1, 6):
        batch = unpickle(f'data/cifar-10-batches-py/data_batch_{i}')
        save_images_from_batch(batch, 'data/cifar10/train')

    # 解压测试集
    batch = unpickle('data/cifar-10-batches-py/test_batch')
    save_images_from_batch(batch, 'data/cifar10/test')

if __name__ == '__main__':
    main()
