from keras.datasets import mnist
from PIL import Image
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

imgs = []
for i in range(10):
    imgs.append(x_train[y_train == i][0])
img = np.concatenate(imgs, axis=-1)
img = Image.fromarray(img)
img.save('images/mnist.bmp')
