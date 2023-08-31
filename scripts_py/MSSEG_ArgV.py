############################################################################################################################################################
########################################################### LOAD LIBRARIES #################################################################################
############################################################################################################################################################
import sys    
import numpy as np    
import tensorflow as tf
import SimpleITK as sitk 
from patchify import unpatchify   
from typing import Tuple, Union, cast
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from staple_patch3d import scratch_unpatchify3d, staple
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D, Activation, BatchNormalization   



####################################################################################################################
############################################### BASELINE UNET ######################################################
####################################################################################################################
def Unet3D(inputs,num_classes, loss, opt):
    with strategy.scope():
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

        model.compile(loss=DiceLoss, optimizer=opt, metrics=[recall, precision, dice])
        return model



####################################################################################################################
############################################ METRIC CALLBACKS AND UTILS ############################################
####################################################################################################################
# Dimension utility
#@tf.autograph.experimental.do_not_convert
def exp_lst(a,b):
    a, b = tf.expand_dims(a, axis=-1), tf.expand_dims(b, axis=-1)    
    return a,b

# Save the Net parameters in the save_model dir
def get_model_name(dataset, fold, opt):
    save_dir = 'saved_models/'        
    name = save_dir+'UNet_set_'+dataset+'_val_patient_'+str(fold)+'Fold_'+opt+'Opt'+'.h5'       
    return name

# Save the Net report in the save_model dir
def save_metric_report(results, dataset, fold, opt):
    save_dir = 'saved_models/history_'+dataset+'_val_patient_'+str(fold)+'Fold_'+opt+'Opt'+'.npy'
    np.save(save_dir,results)

# Metrics function definitions
def recall(y_true, y_pred):
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])
    FN = tf.math.reduce_sum(y_predc[y_true==1])    
    return K.mean( TP/(TP+FN+K.epsilon()) )

def precision(y_true, y_pred):
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])    
    FP = tf.math.reduce_sum(y_pred[y_truec==1])
    return K.mean( TP/(TP+FP+K.epsilon()) )

def dice(y_true,y_pred):
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])
    FN = tf.math.reduce_sum(y_predc[y_true==1])
    TN = tf.math.reduce_sum(y_predc[y_truec==1])
    FP = tf.math.reduce_sum(y_pred[y_truec==1])
    return K.mean( 2*TP/(2*TP+FP+FN+K.epsilon()) )

# Loss function definitions
def DiceLoss(y_true,y_pred):        
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])
    FN = tf.math.reduce_sum(y_predc[y_true==1])
    TN = tf.math.reduce_sum(y_predc[y_truec==1])
    FP = tf.math.reduce_sum(y_pred[y_truec==1])
    return K.mean( 1 - 2*TP/(2*TP+FP+FN+K.epsilon()) )          

def dim_reshape(model, a,b):

    # Original volume dimensions
    hin = 224
    win = 224
    din = 224

    # Kernel size and patch strides
    k_size = 32
    d_stride = 16
    h_w_stride = 16

    # Output dims 
    hout = int(((hin - k_size ) / h_w_stride) + 1)
    wout = int(((win - k_size ) / h_w_stride) + 1)
    dout = int(((din - k_size ) / d_stride) + 1)

    # Model Prediction --->
    a = tf.cast(tf.reshape(model(a), (hout, wout, dout, k_size, k_size, k_size)) > 0.5, dtype=tf.float32)
    b = tf.reshape(b, (hout, wout, dout, k_size, k_size, k_size))          
    
    return a,b

#@tf.autograph.experimental.do_not_convert
def evaluate(model, test_data):

    ###########################################################
    ################### TEST AND UNPATCHIFY ###################
    ###########################################################

    # Original volume dimensions
    hin = 224
    win = 224
    din = 224

    # Metrics
    test_dice = []
    test_recall = []
    test_precision = []

    test_data = test_data.map(lambda a, b: dim_reshape(model, a,b))
    for targ, pred in test_data:

        val_targ = unpatchify(np.array(list(targ)), (hin, win, din)) 
        val_pred = staple(scratch_unpatchify3d(np.array(list(pred)), (hin, win, din)))

        test_dice.append(dice(val_targ, val_pred))
        test_recall.append(recall(val_targ, val_pred))
        test_precision.append(precision(val_targ, val_pred))
         
    t_dice = K.mean(tf.cast(test_dice, dtype=tf.float32)).numpy()
    t_recall = K.mean(tf.cast(test_recall, dtype=tf.float32)).numpy()
    t_precision = K.mean(tf.cast(test_precision, dtype=tf.float32)).numpy()

    return t_dice, t_recall, t_precision
    



####################################################################################################################
################################################# MODEL VALIDATION #################################################
####################################################################################################################
class Val_Mtrc(Callback):   

    # Original volume dimensions
    hin = 224
    win = 224
    din = 224

    def __init__(self, val_data, test_data):
        super().__init__()
        self._supports_tf_logs = True
        self.val_data = val_data
        self.test_data = test_data

    def on_train_begin(self, logs=None):
        logs["test_dice"] = 0
        logs["test_recall"] = 0
        logs["test_precision"] = 0

    def on_epoch_begin(self, epoch, logs=None):               
        logs["val_loss"]=0                                



    ###########################################################
    ################ UNPATCHIFY AND VALIDATION ################
    ###########################################################
    def on_epoch_end(self, epoch, logs={}):        

        self.val_data = self.val_data.map(lambda a, b: dim_reshape(self.model, a,b))
        epoch_val_loss = []

        for val_targ,val_pred in self.val_data:

            #                     DiceLoss(val_targ, val_pred)
            epoch_val_loss.append(DiceLoss(unpatchify(np.array(list(val_targ)), (self.hin, self.win, self.din)), 3
                                            staple(scratch_unpatchify3d(np.array(list(val_pred)), (self.hin, self.win, self.din)))))

        # Here's where the logs dictionary is updated:
        logs["val_loss"]=K.mean(tf.cast(epoch_val_loss, dtype=tf.float32))  



    ###########################################################
    ################### TEST AND UNPATCHIFY ###################
    ###########################################################
    def on_train_end(self, logs=None):        
        
        # Metrics
        test_dice = []
        test_recall = []
        test_precision = []

        self.test_data = self.test_data.map(lambda a, b: dim_reshape(self.model, a,b))

        for targ, pred in test_data:
            print(self.test_data)

        #     test_targ = unpatchify(np.array(list(targ)), (self.hin, self.win, self.din)) 
        #     test_pred = staple(scratch_unpatchify3d(np.array(list(pred)), (self.hin, self.win, self.din)))

        #     test_dice.append(dice(test_targ, test_pred))
        #     test_recall.append(recall(test_targ, test_pred))
        #     test_precision.append(precision(test_targ, test_pred))
             
        # logs["test_dice"] = K.mean(tf.cast(test_dice, dtype=tf.float32)).numpy()
        # logs["test_recall"] = K.mean(tf.cast(test_recall, dtype=tf.float32)).numpy()
        # logs["test_precision"] = K.mean(tf.cast(test_precision, dtype=tf.float32)).numpy()        



if __name__ == "__main__":       

    ################################################################################################################
    ################################################ CARGA DE DATOS ################################################
    ################################################################################################################
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)    
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    root_dir = 'MRIs/'
    data  = ['', '_ori']
    # Modificar indice para cambiar entre conjunto A=0 o B=1
    MNI_ORI = 1#int(input('Enter 0 for MNI or 1 for ORI: '))
    dataset=data[MNI_ORI]
    # Carga de parches registrados
    with open(root_dir+'X'+dataset+'_patch.npy', 'rb') as X:
        vols = np.load(X)
    with open(root_dir+'Y'+dataset+'_patch.npy', 'rb') as Y:
        masks = np.load(Y)
    


    ################################################################################################################
    ########################################### INDEX AND DIMENSIONS UTILS #########################################
    ################################################################################################################
    # Prepare dataset from the generators above
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dout,hout,wout = 13,13,13

    # TEST SET
    X_test, y_test = vols[-4*(dout*hout*wout):], masks[-4*(dout*hout*wout):]

    test_batch_size = 2197

    test_steps = len(X_test) // test_batch_size

    test_data = tf.data.Dataset.from_tensor_slices(
            ( X_test.astype(np.float32), y_test.astype(np.uint8) )
        ).map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).with_options(options).batch(test_batch_size).prefetch(tf.data.AUTOTUNE)

    # Cross-Validation indexes
    pats = [1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4]
    train_val_idx = np.array([dout*hout*wout*str(i).split(' ') for i in pats]).astype(np.uint8).flatten()



    ################################################################################################################
    ################################################ DATA AUGMENTATION #############################################
    ################################################################################################################
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



    ################################################################################################################
    ############################################## CALLBACK OBJECTS ################################################
    ################################################################################################################
    seed = 24

    rlop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                factor=0.2,
                                                patience=5,
                                                verbose=0,
                                                mode="min",
                                                min_delta=0.0001,
                                                cooldown=0,
                                                min_lr=1e-06)

    early_stopping =  EarlyStopping(patience=10,
                                    monitor="val_loss",
                                    min_delta=0.001,
                                    restore_best_weights=True,
                                    mode="min",
                                    verbose=1)

    optimizers = [
                    SGD(learning_rate=0.1, momentum=0.4, nesterov=False, name="SGD04"),
                    SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD09"),
                    SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name="SGDNV09"),
                    Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="ADAM")
                 ]



    ################################################################################################################
    ############################################### DATA TENSORS ###################################################
    ################################################################################################################
    opt = optimizers[int(sys.argv[1])]

    # Cross-validation loop GroupKFold K=4 + test_set    
    fold_var = int(sys.argv[2])
    # fold_var: current patient for test fold

    train_index = np.argwhere(train_val_idx!=fold_var).flatten()
    val_index  = np.argwhere(train_val_idx==fold_var).flatten()
    
    X_train, X_val = vols[train_index], vols[val_index]
    y_train, y_val = masks[train_index], masks[val_index]

    # Keep patches with at least 0.1% of lesion tissue, for training and validation
    lesion_ratio = 0.001   
    ind_y_tr = np.count_nonzero(y_train, axis=(1,2,3), keepdims=False)>int((32**3)*lesion_ratio)
    X_train = X_train[ind_y_tr]
    y_train = y_train[ind_y_tr]                

    # Volume
    image_data_generator = ImageDataGenerator(**img_data_gen_args)          
    image_generator = image_data_generator.flow(X_train, seed=seed)

    # Masks
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)    
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    

    # Batch set up
    batch_size = 32
    eval_batch_size = 2197
    train_steps = 2*len(X_train)//batch_size  
    valid_steps = len(X_val) // eval_batch_size

    
    # Dataset from generator
    train_generator = tf.data.Dataset.from_generator(
        lambda: zip(image_generator, mask_generator),
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 32, 32), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 32, 32, 32), dtype=tf.uint8,))
    ).map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).with_options(options).prefetch(tf.data.AUTOTUNE)
      

    validation_generator = tf.data.Dataset.from_tensor_slices(
            ( X_val.astype(np.float32), y_val.astype(np.uint8) )
    ).map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).with_options(options).batch(eval_batch_size).prefetch(tf.data.AUTOTUNE)                

    # CREATE NEW MODEL
    print("\n Create Model")        
    
    inputs = Input(shape=(32,32,32,1))
    num_classes = 1

    print("TrainSteps: ", train_steps, "ValSteps: ", valid_steps)

    del X_train, X_val, y_train, y_val    

    model = Unet3D(inputs,num_classes, DiceLoss, opt) 
        
    unpatch_validation = Val_Mtrc(validation_generator, test_data)    
    #checkpoint = ModelCheckpoint(get_model_name(dataset,fold_var, opt.get_config()['name']), monitor="val_dice", verbose=1, save_best_only=True, mode='max')                
    callbacks_list = [unpatch_validation, early_stopping, rlop]
    if opt.get_config()['name']=='ADAM':
        callbacks_list = callbacks_list[:-1]
            


    ################################################################################################################
    ############################################ TRAIN AND EVALUATION ##############################################
    ################################################################################################################
    print("\nmodel_fit_Fold"+str(fold_var)+"_"+opt.get_config()['name'])
    #FIT THE MODEL
    history = model.fit(
        train_generator, 
        steps_per_epoch=train_steps, 
        callbacks=callbacks_list, 
        use_multiprocessing=True,      
        workers=10,
        epochs=1
    )#Set to 100
    
    # validation_data=validation_generator, 
    # validation_steps=valid_steps, 
    # validation_freq=1,
    # exit()      
    
    
    # print("\nEnd fit\nSaving Model")
    # model.save(get_model_name(dataset,fold_var, opt.get_config()['name']))
    
    # print("\nModel saved!")
    
    # Set evaluations    
    results = dict()
    results["train_val_test_mtrcs"] = history.history
    # results["train_val_loss"] = history.history

    # dice_t, recall_t, precision_t = evaluate(model, test_data)
    # results["test_dice"] = dice_t
    # results["test_recall"] = recall_t
    # results["test_precision"] = precision_t

    print(results)
    # Save results Train, Validation, Test Evaluation
    # save_metric_report(results, dataset, fold_var, opt.get_config()['name'])
    print("\nHistory saved")

# ############################################################################################################################################################
# ############################################################################################################################################################

