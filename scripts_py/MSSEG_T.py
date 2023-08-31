########################################################### CARGA DE LIBRERIAS #############################################################################
############################################################################################################################################################
############################################################################################################################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D, Activation, BatchNormalization


########################################################### CARGA DE DATOS #################################################################################
############################################################################################################################################################
############################################################################################################################################################

root_dir = 'MRIs/'
data  = ['', '_ori']
# Modificar indice para cambiar entre conjunto A=0 o B=1
MNI_ORI = int(input('Enter 0 for MNI or 1 for ORI: '))
dataset=data[MNI_ORI]
# Carga de parches registrados
with open(root_dir+'X'+dataset+'_patch.npy', 'rb') as X:
    vols = np.load(X)
with open(root_dir+'Y'+dataset+'_patch.npy', 'rb') as Y:
    masks = np.load(Y)


############################################################### BASELINE UNET ##############################################################################
############################################################################################################################################################
############################################################################################################################################################
#@tf.function#(experimental_relax_shapes=True)
def Unet3D(inputs,num_classes, opt):
    x=inputs
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
    
    #ENCODER
    up5 = Conv3D(32, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv4))
    up5 = Activation('relu')(BatchNormalization()(up5))
    merge5 = concatenate([conv3,up5],axis=-1)
    conv5 = Conv3D(32, 3, padding = 'same')(merge5)
    conv5 = Activation('relu')(BatchNormalization()(conv5))
    conv5 = Conv3D(32, 3, padding = 'same')(conv5)
    conv5 = Activation('relu')(BatchNormalization()(conv5))

    up6 = Conv3D(16, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv5))
    up6 = Activation('relu')(BatchNormalization()(up6))
    merge6 = concatenate([conv2,up6],axis=-1)
    conv6 = Conv3D(16, 3, padding = 'same')(merge6)
    conv6 = Activation('relu')(BatchNormalization()(conv6))
    conv6 = Conv3D(16, 3, padding = 'same')(conv6)
    conv6 = Activation('relu')(BatchNormalization()(conv6))

    up7 = Conv3D(8, 2, padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    up7 = Activation('relu')(BatchNormalization()(up7))
    merge7 = concatenate([conv1,up7],axis=-1)
    conv7 = Conv3D(8, 3, padding = 'same')(merge7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))
    conv7 = Conv3D(8, 3, padding = 'same')(conv7)
    conv7 = Activation('relu')(BatchNormalization()(conv7))

    conv8 = Conv3D(1, 1, activation = 'sigmoid')(conv7)
    model = Model(inputs=inputs, outputs = conv8)        

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt)
    return model


########################################################### INDICES CROSS VALIDATION #######################################################################
############################################################################################################################################################
############################################################################################################################################################
dout,hout,wout = 13,13,13
pats = [1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4]

train_val_idx = np.array([dout*hout*wout*str(i).split(' ') for i in pats]).astype(np.uint8).flatten()

# TEST SET
X_test, y_test = vols[-4*(dout*hout*wout):], masks[-4*(dout*hout*wout):]

test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
########################################################### METRIC CALLBACKS ###############################################################################
############################################################################################################################################################
############################################################################################################################################################

# Data augmentation setup
def my_image_mask_generator(generator):    
    for (img, mask) in generator:        
        yield img, mask, [None]

# Save the Net parameters in the save_model dir
def get_model_name(dataset, fold, opt):
    save_dir = 'saved_models/'    
    return save_dir+'UNet_set_'+dataset+'_val_patient_'+str(fold)+'Fold_'+opt+'Opt'+'.h5' 

# Save the Net report in the save_model dir
def get_metric_report(pset, results, dataset, fold, opt):            
    save_dir = 'saved_models/'+pset+'_set'+dataset+'_val_patient_'+str(fold)+'Fold_'+opt+'Opt'+'.npy'
    np.save(save_dir,results)

# Metrics function definitions
def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Metrics(Callback):

    def __init__(self, dataset, fold_var, opt, train_data, val_data, batches, val_batches, test_set):
        super().__init__()
        self._supports_tf_logs = True
        
        self.train_data = my_image_mask_generator(train_data)
        self.val_data = my_image_mask_generator(val_data)

        self.batches = batches
        self.val_batches = val_batches 

        self.test = test_set        
    

    def on_train_begin(self, logs=None):
        self.f1scores = []
        self.val_f1scores = []


    def on_epoch_begin(self, epoch, logs=None):        
        logs["f1s"]=0
        logs["val_f1s"]=0
        logs["test_prc"]=0
        logs["test_rcl"]=0
        logs["test_f1s"]=0


    def on_epoch_end(self, epoch, logs={}):
        ###########################################################
        ############### TRAINING SET F1S Monitoring ###############
        ###########################################################

        pred = []
        targ = []

        for batch in range(self.batches):
            xTrn, yTrn, _ = next(self.train_data)
            pred.append(self.model(xTrn))
            targ.append(yTrn)        

        pred = tf.stack(pred) > 0.5
        targ = tf.stack(targ)
        
        f1s = f1_score(targ, pred)
        
        self.f1scores.append(f1s)

        #Here's where the logs dictionary is updated:        
        logs["f1s"]=f1s

        
        ###########################################################
        ################ VALIDATION SET Monitoring ################
        ###########################################################
        
        val_pred = []
        val_targ = []

        for batch in range(self.val_batches):
            xVal, yVal, _ = next(self.validation_data)
            val_pred.append(self.model(xVal))
            val_targ.append(yVal)        

        val_pred = tf.stack(val_pred) > 0.5
        val_targ = tf.stack(val_targ)
        
        val_f1s = f1_score(val_targ, val_pred)
        
        self.val_f1scores.append(val_f1s)

        #Here's where the logs dictionary is updated:        
        logs["val_f1s"]=val_f1s
        

        print(" — f1s: %f — val_f1s: %f" %(f1s, val_f1s))       
        return        

    def on_train_end(self, logs=None):

        X_test, targ = self.test
        X_test = tf.expand_dims(X_test, axis=-1)
        targ = tf.expand_dims(targ, axis=-1)
        predict = np.asarray(self.model.predict(X_test))
        
        prc = precision_score(targ, predict)
        rcl = recall_score(targ, predict)
        f1s = f1_score(targ, predict)
        

        self.f1scores = np.array(self.f1scores)
        self.val_f1scores = np.array(self.val_f1scores)

        logs["test_prc"]=prc
        logs["test_rcl"]=rcl
        logs["test_f1s"]=f1s

        # Save Train/Validation report        
        get_metric_report('Train', self.f1scores, dataset, fold, opt)
        get_metric_report('Val', self.val_f1scores, dataset, fold, opt)

        # Save Test report
        get_metric_report('Test', logs, dataset, fold, opt)        
        
        # Save unpatchified volumes, solo para los mejores modelos
        #unpatched = np.zeros((4, 224, 224, 224))
        #for i in range(len(masks_patches)):
        #    unpatched[i, :, :, :] = unpatchify(np.reshape(masks_patches[i], (13, 13, 13, 32, 32, 32)), tuple(unpatched.shape[1:]))

        # Guardar y posiblemente covertir algunas a formato .nii
        


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

rlop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_f1s",
                                            factor=0.2,#tf.math.exp(-0.1)??
                                            patience=5,
                                            verbose=0,
                                            mode="max",
                                            min_delta=0.0001,
                                            cooldown=10,
                                            min_lr=0.00001)

optimizers = [SGD(learning_rate=0.1, momentum=0.0, nesterov=False, name="SGD0"), SGD(learning_rate=0.1, momentum=0.4, nesterov=False, name="SGD04"), SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD09"), SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name="SGDNTV09")]

for opt in optimizers:
    # Cross-validation loop GroupKFold K=4 + test_set
    for fold_var in range(4,0,-1):
        # fold_var: current patient for test fold
        train_index = np.argwhere(train_val_idx!=fold_var).flatten()
        val_index  = np.argwhere(train_val_idx==fold_var).flatten()        
        
        X_train, X_val = vols[train_index], vols[val_index]
        y_train, y_val = masks[train_index], masks[val_index]

        
        # Volume
        image_data_generator = ImageDataGenerator(**img_data_gen_args)
        image_data_generator.fit(X_train, augment=True, seed=seed)

        image_generator = image_data_generator.flow(X_train, batch_size=32, seed=seed)
        valid_img_generator = image_data_generator.flow(X_val, batch_size=32, seed=seed)


        # Masks
        mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
        mask_data_generator.fit(y_train, augment=True, seed=seed)

        mask_generator = mask_data_generator.flow(y_train, batch_size=32, seed=seed)
        valid_mask_generator = mask_data_generator.flow(y_val, batch_size=32, seed=seed)

        
        print("\ntf.data.Dataset.from_generator") 

        def fun(a,b):
            return tf.expand_dims(a, axis=-1), tf.expand_dims(b, axis=-1)      
                            
        
        train_generator = tf.data.Dataset.from_generator(
            lambda: zip(image_generator, mask_generator),
            output_types=(tf.float32, tf.float32), 
            output_shapes = ([None,32,32,32],[None,32,32,32])
        ).map(lambda a,b: fun(a,b))                     

        validation_generator = tf.data.Dataset.from_generator(
            lambda: zip(valid_img_generator, valid_mask_generator),
            output_types=(tf.float32, tf.float32), 
            output_shapes = ([None,32,32,32],[None,32,32,32])
        ).map(lambda a,b: fun(a,b))                   

        print("\nBatch SET UP")
        # Batch set up
        batch_size = 32
        train_steps_per_epoch = mask_generator.__len__()//batch_size    
        

        # CREATE NEW MODEL
        print("\n Create Model")

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        inputs = Input(shape=(32,32,32,1))
        num_classes = 1

        print("TrainBatch: ", mask_generator.__len__(), "ValBatch: ", valid_mask_generator.__len__())
        # with strategy.scope():
        #     model = Unet3D(inputs,num_classes, opt)
        #     model_metrics = Metrics(dataset,fold_var, opt.get_config()['name'], train_generator, validation_generator, mask_generator.__len__(), valid_mask_generator.__len__(), test_data)            

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        # train_generator = train_generator.with_options(options)
        # validation_generator = validation_generator.with_options(options)
        
        # checkpoint = ModelCheckpoint(get_model_name(dataset,fold_var, opt.get_config()['name']), monitor="val_f1s", verbose=1, save_best_only=True, mode='max')
        
        # early_stopping = EarlyStopping(patience=3, monitor="val_f1s", min_delta=0.01, restore_best_weights=True)

        # # LearningScheduer (RLoP) is outside

        # callbacks_list = [model_metrics, checkpoint, rlop, early_stopping]

        # print("\nmodel_fit")    
        # # FIT THE MODEL
        # history = model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, callbacks=callbacks_list, epochs=1)#Set in 80
           

        tf.keras.backend.clear_session()
        break#remove
    break#remove

############################################################################################################################################################
############################################################################################################################################################

