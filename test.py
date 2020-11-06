import os
import numpy as np
from utilities import datareader
from model.unet import unet_org
import matplotlib.pyplot as plt

ckpt_path = r'ckpt/unet.h5'
image_list = r'dataset/val.txt'

dtreader = datareader(image_list)
model = unet_org()

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

images, masks = datareader.get_Images(image_list)

for i in range(5):
    result = model.predict(images[i])

    mask = np.argmax(masks[i], axis=2)
    result = np.squeeze(result)
    result = np.argmax(result, axis=2)

    plt.imshow(result)
    plt.show()

    plt.imshow(mask)
    plt.show()