from tensorflow.keras.layers import (
        Conv3D,
        Conv2D,
        Input, 
        MaxPooling3D,
        MaxPooling2D,
        Dropout, 
        concatenate,          
        Activation, 
        BatchNormalization,        
        Conv3DTranspose,
        Conv2DTranspose,
        LeakyReLU
)

def three_layer_depth(x):
    # ENCODER
    conv1 = Conv3D(8, 3, padding = 'same',data_format="channels_last")(x)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    conv1 = Conv3D(8, 3, padding = 'same')(conv1)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)

    conv2 = Conv3D(16, 3, padding = 'same')(pool1)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    conv2 = Conv3D(16, 3, padding = 'same')(conv2)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)

    conv3 = Conv3D(32, 3, padding = 'same')(pool2)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    conv3 = Conv3D(32, 3, padding = 'same')(conv3)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Conv3D(64, 3, padding = 'same')(pool3)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    conv4 = Conv3D(64, 3, padding = 'same')(conv4)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    
    # DECODER
    up5    = Conv3DTranspose(32, 3, strides = (2, 2, 2), padding = 'same')(conv4)
    up5    = Activation('relu')(BatchNormalization()(up5))
    merge5 = concatenate([conv3,up5],axis=-1)
    drop5  = Dropout(0.2)(merge5)
    conv5  = Conv3D(32, 3, padding = 'same')(drop5)
    conv5  = Activation('relu')(BatchNormalization()(conv5))    

    up6 = Conv3DTranspose(16, 3, strides = (2, 2, 2), padding = 'same')(conv5)
    up6 = Activation('relu')(BatchNormalization()(up6))
    merge6 = concatenate([conv2,up6],axis=-1)
    drop6  = Dropout(0.2)(merge6)
    conv6 = Conv3D(16, 3, padding = 'same')(drop6)
    conv6 = Activation('relu')(BatchNormalization()(conv6))

    up7 = Conv3DTranspose(8, 3, strides = (2, 2, 2), padding = 'same')(conv6)
    up7 = Activation('relu')(BatchNormalization()(up7))
    merge7 = concatenate([conv1,up7],axis=-1)
    drop7  = Dropout(0.2)(merge7)
    conv7 = Conv3D(16, 3, padding = 'same')(drop7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))
    
    conv8 = Conv3D(1, 1, activation = 'sigmoid', dtype='float32')(conv7)

    return conv8

def four_layer_depth(x):
    # DECODER
    conv1 = Conv3D(8, 3, padding = 'same',data_format="channels_last")(x)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    conv1 = Conv3D(8, 3, padding = 'same')(conv1)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16, 3, padding = 'same')(pool1)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    conv2 = Conv3D(16, 3, padding = 'same')(conv2)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(32, 3, padding = 'same')(pool2)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    conv3 = Conv3D(32, 3, padding = 'same')(conv3)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(64, 3, padding = 'same')(pool3)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    conv4 = Conv3D(64, 3, padding = 'same')(conv4)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(128, 3, padding = 'same')(pool4)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    conv5 = Conv3D(128, 3, padding = 'same')(conv5)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    
    #ENCODER
    up6 = Conv3D(32, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv5))
    up6 = Activation('relu')(BatchNormalization()(up6))
    merge6 = concatenate([conv4,up6],axis=-1)
    conv6 = Conv3D(32, 3, padding = 'same')(merge6)
    conv6 = Activation('relu')(BatchNormalization()(conv6))
    conv6 = Conv3D(32, 3, padding = 'same')(conv6)
    conv6 = Activation('relu')(BatchNormalization()(conv6))

    up7 = Conv3D(16, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    up7 = Activation('relu')(BatchNormalization()(up7))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(16, 3, padding = 'same')(merge7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))
    conv7 = Conv3D(16, 3, padding = 'same')(conv7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))

    up8 = Conv3D(8, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    up8 = Activation('relu')(BatchNormalization()(up8))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(8, 3, padding = 'same')(merge8)
    conv8 = Activation('relu')(BatchNormalization()(conv8))
    conv8 = Conv3D(8, 3, padding = 'same')(conv8)
    conv8 = Activation('relu')(BatchNormalization()(conv8))

    up9 = Conv3D(8, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    up9 = Activation('relu')(BatchNormalization()(up9))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(8, 3, padding = 'same')(merge9)
    conv9 = Activation('relu')(BatchNormalization()(conv9))
    conv9 = Conv3D(8, 3, padding = 'same')(conv9)
    conv9 = Activation('relu')(BatchNormalization()(conv9))

    conv10 = Conv3D(1, 1, activation = 'sigmoid', dtype='float32')(conv9)

    return conv10

def four_layer_depthDrop(x, dropout=0.1):
    # ENCODER
    conv1 = Conv3D(8, 3, padding = 'same',data_format="channels_last")(x)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    conv1 = Conv3D(8, 3, padding = 'same')(conv1)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)

    conv2 = Conv3D(16, 3, padding = 'same')(pool1)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    conv2 = Conv3D(16, 3, padding = 'same')(conv2)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)

    conv3 = Conv3D(32, 3, padding = 'same')(pool2)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    conv3 = Conv3D(32, 3, padding = 'same')(conv3)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Conv3D(64, 3, padding = 'same')(pool3)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    conv4 = Conv3D(64, 3, padding = 'same')(conv4)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(128, 3, padding = 'same')(pool4)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    conv5 = Conv3D(128, 3, padding = 'same')(conv5)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    
    # DECODER
    up6    = Conv3DTranspose(64, 3, strides = (2, 2, 2), padding = 'same')(conv5)
    up6    = Activation('relu')(BatchNormalization()(up6))
    merge6 = concatenate([conv4,up6],axis=-1)
    drop6  = Dropout(dropout)(merge6)
    conv6  = Conv3D(64, 3, padding = 'same')(drop6)
    conv6  = Activation('relu')(BatchNormalization()(conv6))    

    up7 = Conv3DTranspose(32, 3, strides = (2, 2, 2), padding = 'same')(conv6)
    up7 = Activation('relu')(BatchNormalization()(up7))
    merge7 = concatenate([conv3,up7],axis=-1)
    drop7  = Dropout(dropout)(merge7)
    conv7 = Conv3D(32, 3, padding = 'same')(drop7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))

    up8 = Conv3DTranspose(16, 3, strides = (2, 2, 2), padding = 'same')(conv7)
    up8 = Activation('relu')(BatchNormalization()(up8))
    merge8 = concatenate([conv2,up8],axis=-1)
    drop8  = Dropout(dropout)(merge8)
    conv8 = Conv3D(16, 3, padding = 'same')(drop8)
    conv8 = Activation('relu')(BatchNormalization()(conv8))

    up9 = Conv3DTranspose(16, 3, strides = (2, 2, 2), padding = 'same')(conv8)
    up9 = Activation('relu')(BatchNormalization()(up9))
    merge9 = concatenate([conv1,up9],axis=-1)
    drop9  = Dropout(dropout)(merge9)
    conv9 = Conv3D(16, 3, padding = 'same')(drop9)
    conv9 = Activation('relu')(BatchNormalization()(conv9))
    
    conv10 = Conv3D(1, 1, activation = 'sigmoid', dtype='float32')(conv9)

    return conv10


def four_layer_2Ddrop(x, dropout=0.1):
    # ENCODER
    conv1 = Conv2D(8, 3, padding = 'same',data_format="channels_last")(x)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    conv1 = Conv2D(8, 3, padding = 'same')(conv1)
    conv1 = Activation('relu')(BatchNormalization()(conv1))
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(drop1)

    conv2 = Conv2D(16, 3, padding = 'same')(pool1)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    conv2 = Conv2D(16, 3, padding = 'same')(conv2)
    conv2 = Activation('relu')(BatchNormalization()(conv2))
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(drop2)

    conv3 = Conv2D(32, 3, padding = 'same')(pool2)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    conv3 = Conv2D(32, 3, padding = 'same')(conv3)
    conv3 = Activation('relu')(BatchNormalization()(conv3))
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(drop3)

    conv4 = Conv2D(64, 3, padding = 'same')(pool3)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    conv4 = Conv2D(64, 3, padding = 'same')(conv4)
    conv4 = Activation('relu')(BatchNormalization()(conv4))
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)

    conv5 = Conv2D(128, 3, padding = 'same')(pool4)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    conv5 = Conv2D(128, 3, padding = 'same')(conv5)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    
    # DECODER
    up6    = Conv2DTranspose(64, 3, strides = (2,2), padding = 'same')(conv5)
    up6    = Activation('relu')(BatchNormalization()(up6))
    merge6 = concatenate([conv4,up6],axis=-1)
    drop6  = Dropout(dropout)(merge6)
    conv6  = Conv2D(64, 3, padding = 'same')(drop6)
    conv6  = Activation('relu')(BatchNormalization()(conv6))    

    up7 = Conv2DTranspose(32, 3, strides = (2,2), padding = 'same')(conv6)
    up7 = Activation('relu')(BatchNormalization()(up7))
    merge7 = concatenate([conv3,up7],axis=-1)
    drop7  = Dropout(dropout)(merge7)
    conv7 = Conv2D(32, 3, padding = 'same')(drop7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))

    up8 = Conv2DTranspose(16, 3, strides = (2,2), padding = 'same')(conv7)
    up8 = Activation('relu')(BatchNormalization()(up8))
    merge8 = concatenate([conv2,up8],axis=-1)
    drop8  = Dropout(dropout)(merge8)
    conv8 = Conv2D(16, 3, padding = 'same')(drop8)
    conv8 = Activation('relu')(BatchNormalization()(conv8))

    up9 = Conv2DTranspose(16, 3, strides = (2,2), padding = 'same')(conv8)
    up9 = Activation('relu')(BatchNormalization()(up9))
    merge9 = concatenate([conv1,up9],axis=-1)
    drop9  = Dropout(dropout)(merge9)
    conv9 = Conv2D(16, 3, padding = 'same')(drop9)
    conv9 = Activation('relu')(BatchNormalization()(conv9))
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid', dtype='float32')(conv9)

    return conv10


# =============================================================================
# Bloques para UNet 3D
# =============================================================================
def Encoding_Block_Unet3D(subsample_ba, N_mc, N_conv, a):
    if N_conv>1:
        for n in range(N_conv-1):
            if n==0:
                x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
            else:
                x = Conv3D(N_mc, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=a)(x)
            x = Conv3D(N_mc, 3, padding='same')(x)
    else:
        x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
    # x = Add()([subsample_ba, x])#No se puede en Unet
    x = BatchNormalization()(x)
    concat = LeakyReLU(alpha=a)(x)
    subsample = MaxPooling3D(2)(concat)
    return subsample, concat

def BlotlleNeck_Block_Unet3D(subsample_ba, N_mc, N_conv, a):
    for n in range(N_conv-1):
        if n==0:
            x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
        else:
            x = Conv3D(N_mc, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=a)(x)
    x = Conv3D(N_mc, 3, padding='same')(x)
    # x = Add()([subsample_ba, x])#No se puede en Unet
    x = BatchNormalization()(x)
    concat = LeakyReLU(alpha=a)(x)
    x = Conv3DTranspose(N_mc/2,2,2, padding='same')(concat)
    x = BatchNormalization()(x)
    upsample = LeakyReLU(alpha=a)(x)
    return upsample

def Decoding_Block_Unet3D(concat,upsample_ba, N_mc, N_conv, a, Bloque_1=False):
    x = concatenate([concat,upsample_ba])
    
    for n in range(N_conv-1):
        x = Conv3D(N_mc, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=a)(x)
    x = Conv3D(N_mc, 3, padding='same')(x)
    # x = Add()([upsample_ba, x])
    x = BatchNormalization()(x)
    if not Bloque_1:
        concat = LeakyReLU(alpha=a)(x)
        x = Conv3DTranspose(N_mc/2,2,2, padding='same')(concat)
        x = BatchNormalization()(x)
    upsample = LeakyReLU(alpha=a)(x)
    return upsample

# =============================================================================
# Modelos UNet3D
# =============================================================================
def UNet3D_A(input_dim):
    
    a=0.0        
    subsample_B1, concatenate_B1 = Encoding_Block_Unet3D(input_dim, 16, 2, a)
    subsample_B2, concatenate_B2 = Encoding_Block_Unet3D(subsample_B1, 32, 2, a)
    subsample_B3, concatenate_B3 = Encoding_Block_Unet3D(subsample_B2, 64, 2, a)
    subsample_B4, concatenate_B4 = Encoding_Block_Unet3D(subsample_B3, 128, 2, a)
    
    upsample_BN = BlotlleNeck_Block_Unet3D(subsample_B4, 256, 2, a)
    
    upsample_B4 = Decoding_Block_Unet3D(concatenate_B4,upsample_BN, 128, 2, a)
    upsample_B3 = Decoding_Block_Unet3D(concatenate_B3,upsample_B4, 64, 2, a)
    upsample_B2 = Decoding_Block_Unet3D(concatenate_B2,upsample_B3, 32, 2, a)
    upsample_B1 = Decoding_Block_Unet3D(concatenate_B1,upsample_B2, 16, 2, a, 'B1')
    
    output = Conv3D(1, 1)(upsample_B1)
    outputs = Activation('sigmoid', dtype='float32', name='predictions')(output)
        
    return outputs
    
def UNet3D_B(input_dim):
    
    a=0.2
    inputs = Input(shape=input_dim,name='Inputs')
    
    subsample_B1, concatenate_B1 = Encoding_Block_Unet3D(inputs, 16, 1, a)
    subsample_B2, concatenate_B2 = Encoding_Block_Unet3D(subsample_B1, 32, 2, a)
    subsample_B3, concatenate_B3 = Encoding_Block_Unet3D(subsample_B2, 64, 2, a)
    subsample_B4, concatenate_B4 = Encoding_Block_Unet3D(subsample_B3, 128, 2, a)
    
    upsample_BN = BlotlleNeck_Block_Unet3D(subsample_B4, 256, 2, a)
    
    upsample_B4 = Decoding_Block_Unet3D(concatenate_B4,upsample_BN, 128, 2, a)
    upsample_B3 = Decoding_Block_Unet3D(concatenate_B3,upsample_B4, 64, 2, a)
    upsample_B2 = Decoding_Block_Unet3D(concatenate_B2,upsample_B3, 32, 2, a)
    upsample_B1 = Decoding_Block_Unet3D(concatenate_B1,upsample_B2, 16, 2, a, 'B1')
    
    output = Conv3D(1, 1)(upsample_B1)
    outputs = Activation('sigmoid', dtype='float32', name='predictions')(output)
    
    model = Model(inputs=inputs, outputs=outputs, name='UNet3D_B')
    return model


# # =============================================================================
# # Bloques para VNet
# # =============================================================================
# def Encoding_Block_Vnet(subsample_ba, N_mc, N_conv, a, resnet=True):
#     if N_conv>1:
#         for n in range(N_conv-1):
#             if n==0:
#                 x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
#             else:
#                 x = Conv3D(N_mc, 3, padding='same')(x)
#             x = BatchNormalization()(x)
#             x = LeakyReLU(alpha=a)(x)
#         x = Conv3D(N_mc, 3, padding='same')(x)
#     else:
#         x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
#     if resnet:
#         x = Add()([subsample_ba, x])
#     x = BatchNormalization()(x)
#     concatenate = LeakyReLU(alpha=a)(x)
#     x = Conv3D(2*N_mc,2,2,padding='same')(concatenate)
#     x = BatchNormalization()(x)
#     subsample = LeakyReLU(alpha=a)(x)
#     return subsample, concatenate

# def BlottleNeck_Vnet(subsample_ba, N_mc, N_conv, a, resnet=True):
#     for n in range(N_conv-1):
#         if n==0:
#             x = Conv3D(N_mc, 3, padding='same')(subsample_ba)
#         else:
#             x = Conv3D(N_mc, 3, padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=a)(x)
#     x = Conv3D(N_mc, 3, padding='same')(x)
#     if resnet:
#         x = Add()([subsample_ba, x])
#     x = BatchNormalization()(x)
#     concatenate = LeakyReLU(alpha=a)(x)
#     x = Conv3DTranspose(N_mc/2,2,2, padding='same')(concatenate)
#     x = BatchNormalization()(x)
#     upsample = LeakyReLU(alpha=a)(x)
#     return upsample

# def Decoding_Block_Vnet(concat,upsample_ba, N_mc, N_conv, a, Bloque_1=False, resnet=True):
#     x = concatenate([concat,upsample_ba])
    
#     for n in range(N_conv-1):
#         x = Conv3D(N_mc, 3, padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=a)(x)
#     x = Conv3D(N_mc, 3, padding='same')(x)
#     if resnet:
#         x = Add()([upsample_ba, x])
#     x = BatchNormalization()(x)
#     if not Bloque_1:
#         concat = LeakyReLU(alpha=a)(x)
#         x = Conv3DTranspose(N_mc/2,2,2, padding='same')(concat)
#         x = BatchNormalization()(x)
#     upsample = LeakyReLU(alpha=a)(x)
#     return upsample


# def VNet(input_dim, a):
#     inputs = keras.Input(shape=input_dim,name='Inputs')
#     subsample_B1, concatenate_B1 = Encoding_Block_Vnet(inputs, 16, 1, a, resnet=True)
#     subsample_B2, concatenate_B2 = Encoding_Block_Vnet(subsample_B1, 32, 2, a)
#     subsample_B3, concatenate_B3 = Encoding_Block_Vnet(subsample_B2, 64, 3, a)
#     subsample_B4, concatenate_B4 = Encoding_Block_Vnet(subsample_B3, 128, 3, a)
#     upsample_BN = BlottleNeck_Vnet(subsample_B4, 256, 3, a)
#     upsample_B4 = Decoding_Block_Vnet(concatenate_B4,upsample_BN, 128, 3, a)
#     upsample_B3 = Decoding_Block_Vnet(concatenate_B3,upsample_B4, 64, 3, a)
#     upsample_B2 = Decoding_Block_Vnet(concatenate_B2,upsample_B3, 32, 2, a)
#     upsample_B1 = Decoding_Block_Vnet(concatenate_B1,upsample_B2, 16, 1, a, Bloque_1=True)
#     outputs = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(upsample_B1)
#     model = keras.Model(inputs=inputs, outputs=outputs, name='VNet')
#     return model

# def VNet_B(input_dim, a):
#     inputs = keras.Input(shape=input_dim,name='Inputs')
#     subsample_B1, concatenate_B1 = Encoding_Block_Vnet(inputs, 16, 1, a, resnet=False)
#     subsample_B2, concatenate_B2 = Encoding_Block_Vnet(subsample_B1, 32, 2, a)
#     # subsample_B3, concatenate_B3 = Encoding_Block_Vnet(subsample_B2, 64, 3, a)
#     # subsample_B4, concatenate_B4 = Encoding_Block_Vnet(subsample_B3, 128, 3, a)
#     upsample_BN = BlottleNeck_Vnet(subsample_B2, 64, 3, a)
#     # upsample_B4 = Decoding_Block_Vnet(concatenate_B4,upsample_BN, 128, 3, a)
#     # upsample_B3 = Decoding_Block_Vnet(concatenate_B3,upsample_B4, 64, 3, a)
#     upsample_B2 = Decoding_Block_Vnet(concatenate_B2,upsample_BN, 32, 2, a)
#     upsample_B1 = Decoding_Block_Vnet(concatenate_B1,upsample_B2, 16, 1, a, Bloque_1=True)
#     outputs = layers.Conv3D(1, 1, activation='sigmoid', dtype='float32')(upsample_B1)
#     model = keras.Model(inputs=inputs, outputs=outputs, name='VNet_B')