import os
from keras.optimizers import *
from utilities import datareader
from model.unet import unet_org

ckpt_path = r'ckpt/unet.h5'
image_list = r'dataset/train.txt'
model = unet_org()
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss=['categorical_crossentropy'], metrics=['accuracy'])

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

images, masks = datareader.get_Images(image_list)
model.fit(x=images,y=masks,epochs=30,verbose=1,batch_size=1,shuffle=True)
model.save(ckpt_path, overwrite=True)
print('the checkpoint is saved successfully. Path : ' + ckpt_path)