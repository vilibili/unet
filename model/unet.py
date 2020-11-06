from keras.layers import *
from keras.models import *

def unet_org(height = 256, width = 256, num_class = 2):
    img_input = Input(shape=(height, width, 3))
    conv1 = Conv2D(64, (3, 3), padding='same')(img_input)
    conv1 = Activation(activation='relu')(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv2 = Activation(activation='relu')(conv2)
    pool1 = MaxPooling2D([2, 2],padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv3 = Activation(activation='relu')(conv3)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv4 = Activation(activation='relu')(conv4)
    pool2 = MaxPooling2D([2, 2], padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv5 = Activation(activation='relu')(conv5)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv5)
    conv6 = Activation(activation='relu')(conv6)
    pool3 = MaxPooling2D([2, 2], padding='same')(conv6)

    conv7 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv7 = Activation(activation='relu')(conv7)
    conv8 = Conv2D(512, (3, 3), padding='same')(conv7)
    conv8 = Activation(activation='relu')(conv8)
    pool4 = MaxPooling2D([2, 2], padding='same')(conv8)

    conv9 = Conv2D(1024, (3, 3), padding='same')(pool4)
    conv9 = Activation(activation='relu')(conv9)
    conv10 = Conv2D(1024, (3, 3), padding='same')(conv9)
    conv10 = Activation(activation='relu')(conv10)

    up1 = UpSampling2D()(conv10)
    concat1 = Concatenate()([up1, conv8])
    conv11 = Conv2D(512, (3, 3), padding='same')(concat1)
    conv11 = Activation(activation='relu')(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same')(conv11)
    conv12 = Activation(activation='relu')(conv12)

    up2 = UpSampling2D()(conv12)
    concat2 = Concatenate()([up2, conv6])
    conv13 = Conv2D(256, (3, 3), padding='same')(concat2)
    conv13 = Activation(activation='relu')(conv13)
    conv14 = Conv2D(256, (3, 3), padding='same')(conv13)
    conv14 = Activation(activation='relu')(conv14)

    up3 = UpSampling2D()(conv14)
    concat3 = Concatenate()([up3, conv4])
    conv15 = Conv2D(128, (3, 3), padding='same')(concat3)
    conv15 = Activation(activation='relu')(conv15)
    conv16= Conv2D(128, (3, 3), padding='same')(conv15)
    conv16 = Activation(activation='relu')(conv16)

    up4 = UpSampling2D()(conv16)
    concat4 = Concatenate()([up4, conv2])
    conv17 = Conv2D(64, (3, 3), padding='same')(concat4)
    conv17 = Activation(activation='relu')(conv17)
    conv18 = Conv2D(64, (3, 3), padding='same')(conv17)
    conv18 = Activation(activation='relu')(conv18)

    output = Conv2D(num_class, (1, 1), activation='sigmoid', padding='same')(conv18)

    model = Model(inputs=img_input, outputs=output)

    return model