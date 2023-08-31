import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import pickle
import sklearn
import datetime
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import np_utils
import segmentation_models_3D as sm
from sklearn.pipeline import Pipeline
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedGroupKFold  
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D

########################################################### CARGA DE DATOS #################################################################################
############################################################################################################################################################
############################################################################################################################################################

root_dir = 'MRIs/'
dataset  = ['', '_ori']
# Modificar indice para cambiar entre conjunto A=0 o B=1
data=dataset[1]
# Carga de parches registrados
with open(root_dir+'X'+data+'_patch.npy', 'rb') as X:
    vols = np.load(X)
with open(root_dir+'Y'+data+'_patch.npy', 'rb') as Y:
    masks = np.load(Y)


########################################################### RED BASELINE UNET ##############################################################################
############################################################################################################################################################
############################################################################################################################################################

def Unet3D(inputs,num_classes):
    x=inputs    
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same',data_format="channels_last")(x)    
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3D(32, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3D(16, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv3D(8, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs=inputs, outputs = conv10)        
    
    return model


########################################################### INDICES CROSS VALIDATION #######################################################################
############################################################################################################################################################
############################################################################################################################################################
dout,hout,wout = 13,13,13
pats = [1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5]

pat_idx        = np.array([dout*hout*wout*str(i).split(' ') for i in pats]).astype(np.uint8).flatten()
train_val_idx  = pat_idx[:-4*(13**3)]

# TEST SET
X_test, y_test = np.expand_dims(vols[-4*(13**3):], axis=-1), np.expand_dims(masks[-4*(13**3):], axis=-1)

del pat_idx # train and validate with train_val_idx, then

########################################################### METRIC CALLBACKS ###############################################################################
############################################################################################################################################################
############################################################################################################################################################
# class F1History(tf.keras.callbacks.Callback):

#     def __init__(self, train, validation=None):
#         super(F1History, self).__init__()
#         self.validation = list(validation)
#         self.train = list(train)

#     def on_epoch_end(self, epoch, logs={}):

#         logs['F1_score_train'] = float('-inf')
#         X_train, y_train = self.train[0], self.train[1]
#         y_pred = (self.model.predict(X_train).ravel()>0.5)+0
#         score = f1_score(y_train, y_pred)       

#         if (self.validation):
#             logs['F1_score_val'] = float('-inf')
#             X_valid, y_valid = self.validation[0], self.validation[1]
#             y_val_pred = (self.model.predict(X_valid).ravel()>0.5)+0
#             val_score = f1_score(y_valid, y_val_pred)
#             logs['F1_score_train'] = np.round(score, 5)
#             logs['F1_score_val'] = np.round(val_score, 5)
#         else:
#             logs['F1_score_train'] = np.round(score, 5)

class Metrics(keras.callbacks.Callback):

	def __init__(self, val_data, batch_size = 8):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
	

    def on_train_begin(self, logs={}):

        self.precision = []
        self.recall = []
        self.f1scores = []
        self.prc=0
        self.rcl=0
        self.f1s=0        

    def on_epoch_end(self, epoch, logs={}):

    	batches = len(self.validation_data)
        total = batches * self.batch_size

    	predict = np.zeros((total,1))
        targ = np.zeros((total))

        for batch in range(batches):
            xVal, yVal = next(self.validation_data)
            predict[batch * self.batch_size : (batch+1) * self.batch_size] = np.asarray(self.model.predict(xVal)).round()
            targ[batch * self.batch_size : (batch+1) * self.batch_size] = yVal

        
        predict = np.round(np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]])))
        targ = self.validation_data[1]

        predict = (predict < 0.5).astype(bool)

        self.prc = precision_score(targ, predict)
        self.rcl = recall_score(targ, predict)
        self.f1s = f1_score(targ, predict)

        self.precision.append(self.prc)        
        self.recall.append(self.rcl)
        self.f1scores.append(self.f1s)

        #Here is where I update the logs dictionary:
        logs["prc"]=self.prc
        logs["rcl"]=self.rcl
        logs["F1_score_val"]=self.f1s

        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(self.f1s, self.prc, self.rcl))
	
            

def get_model_name(fold):
    save_dir = '/saved_models/'
    return save_dir+'model1_val_patient_'+str(fold)+'Fold'+'.h5'            
        
        
# def dice(y_true, y_pred):
#     def recall(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))


########################################################### DATA AUGMENTATION ##############################################################################
############################################################################################################################################################
############################################################################################################################################################
img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,                  
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,                     
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 


########################################################### ENTRENAMIENTO DEL MODELO #######################################################################
############################################################################################################################################################
############################################################################################################################################################

seed = 24

VALIDATION_RESULTS = []

# Cross-validation loop GroupKFold K=4 + test_set
for fold_var in range(4,0,-1):
    # i: current patient for test fold
    train_index = np.argwhere(train_val_idx!=fold_var).flatten()
    val_index  = np.argwhere(train_val_idx==fold_var).flatten()
    
    
    X_train, X_val = vols[train_index], vols[val_index]
    y_train, y_val = masks[train_index], masks[val_index]
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(X_train, augment=True, seed=seed)
    image_generator = image_data_generator.flow(X_train, batch_size=8, seed=seed)
    valid_img_generator = image_data_generator.flow(X_val, batch_size=8, seed=seed)

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train, batch_size=8, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_val, batch_size=8, seed=seed)
                

    print("\ntrain_generator")

    train_generator = my_image_mask_generator()

    print("\nvalidation_generator")

    validation_generator = my_image_mask_generator()    

    print("\nBatch SET UP")

    # Batch set up
    batch_size = 8
    train_steps_per_epoch = len(train_index)//batch_size
    valid_steps_per_epoch = len(val_index)//batch_size

    print("\nCallbacks")

    # Metrics
    #f1_callback=F1History(train=train_generator,validation=validation_generator)
    metricas=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    # CREATE NEW MODEL
    print("\n Create Model")
    inputs = Input(shape=(32,32,32,1))
    num_classes = 1
    model = Unet3D(inputs,num_classes)

    print("\nCompile and Callbacks")
    # COMPILE NEW MODEL
    opt = Adam(lr=1e-4,epsilon=1e-5)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=opt, metrics=metricas)

    # CREATE CALLBACKS
    model_metrics = Metrics()
    checkpoint = ModelCheckpoint(get_model_name(6-fold_var), monitor='F1_score_val', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(patience=3, monitor='F1_score_val', min_delta=0.01)
    callbacks_list = [checkpoint, early_stopping, model_metrics]

    print("\nmodel_fit")    
    # FIT THE MODEL
    history = model.fit((np.expand_dims(image_generator, axis=-1), np.expand_dims(mask_generator, axis=-1), [None]), 
		validation_data=(np.expand_dims(valid_img_generator, axis=-1), np.expand_dims(valid_mask_generator, axis=-1), [None]), 
        steps_per_epoch=train_steps_per_epoch, callbacks=callbacks_list,
        validation_steps=valid_steps_per_epoch, epochs=1)#Set in 25 app (?)

    print("\nLoad Weights\n")


    model.load_weights(get_model_name(6-fold_var))

    results = model.evaluate((X_test, y_test))
    results = dict(zip(model.metrics_names,results))

    VALIDATION_RESULTS.append([results[metricas[0]], results[metricas[1]], results[metricas[2]]])    

    tf.keras.backend.clear_session()
    break#remove

############################################################################################################################################################
############################################################################################################################################################