####################################################################################################################
################################################### LOAD LIBRARIES #################################################
####################################################################################################################
import sys    
import numpy as np    
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from Models import three_layer_depth, four_layer_depth
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback



####################################################################################################################
############################################### BASELINE UNET ######################################################
####################################################################################################################
def Unet3D(model_arq, inputs, loss, opt):
    with strategy.scope():
        x=inputs
        outputs = model_arq(x)        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=loss, optimizer=opt, metrics=[recall, precision, dice])
        return model



####################################################################################################################
############################################ METRIC CALLBACKS AND UTILS ############################################
####################################################################################################################
# Dimension utility
def exp_lst(a,b):
    a, b = tf.expand_dims(a, axis=-1), tf.expand_dims(b, axis=-1)    
    return a,b

# Save the Net parameters in the save_model dir
def get_model_name(dataset, arq_name, fold, opt):
    save_dir = 'saved_models/'        
    name = save_dir+'model_'+arq_name+'_'+dataset+'_'+str(fold)+'Fold_'+opt+'_Opt'+'.h5'       
    return name

# Save the Net report in the save_model dir
def save_metric_report(results, arq_name, dataset, fold, opt):
    save_dir = 'saved_models/history_'+arq_name+'_'+dataset+'_'+str(fold)+'Fold_'+opt+'_Opt'+'.npy'
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

# Data Aug Prepare
class Augment(tf.keras.layers.Layer):
    def __init__(self, SEED):
        super().__init__()
        
        flip = RandomFlip("horizontal_and_vertical", seed=SEED)
        
        rota = RandomRotation(0.2, interpolation='nearest', seed=SEED)
        
        trans = RandomTranslation(height_factor=(-0.3, 0.3),
                                    width_factor=(-0.3, 0.3),
                                    interpolation='nearest', seed=SEED)        
        layers = [flip, trans, rota]
        self.aug_model = tf.keras.Sequential(layers)
                
    def call(self, image, mask):
        
        mask = tf.cast(mask, tf.float32)

        images_mask = tf.concat([image, mask], -1)                  
        
        images_mask = self.aug_model(images_mask, training=True)
        
        dim_fus = images_mask.shape[-1]//2

        image = images_mask[..., :dim_fus]
        mask  = images_mask[..., dim_fus:]
        mask  = tf.where(mask>0, tf.ones(tf.shape(mask)), tf.zeros(tf.shape(mask)))
        
        return image, tf.cast(mask, tf.uint8)


if __name__ == "__main__":  

    param_stdin = sys.argv[1:]
    params = dict()
    for s in param_stdin:
        key, value = s.split(':')
        params[key] = value      

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA      

    

    ################################################################################################################
    ############################################## CALLBACK OBJECTS ################################################
    ################################################################################################################    
    early_stopping =  EarlyStopping(patience=10,
                                    monitor="val_loss",
                                    min_delta=0.00001,
                                    restore_best_weights=True,
                                    mode="min",
                                    verbose=1)

    optimizers = {                    
                    'SGD09': SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD09"),
                    'SGDNV09': SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name="SGDNV09"),
                    'ADAM': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="ADAM")
                 }



    ################################################################################################################
    ############################################# DATA LOAD AND RESHAPE ############################################
    ################################################################################################################
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)    
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    root_dir = 'MRIs/'    

    # MODIFICAR indice para cambiar entre conjunto A=0 o B=1
    ########################################################
    DATA = params['data'] ##################################
    ########################################################    

    # Data paths
    x_path = root_dir+'X_'+DATA+'.npy'
    y_path = root_dir+'Y_'+DATA+'.npy'

    # Carga de parches registrados
    with open(x_path, 'rb') as X:
        vols = np.load(X)
    with open(y_path, 'rb') as Y:
        masks = np.load(Y)
    
    vols = np.reshape(vols,(21*343,32,32,32))
    masks = np.reshape(masks,(21*343,32,32,32))

    ################################################################################################################
    ########################################### INDEX AND DIMENSIONS UTILS #########################################
    ################################################################################################################        
    dout,hout,wout = 7,7,7

    # TEST SET
    X_test, y_test = vols[-4*(dout*hout*wout):], masks[-4*(dout*hout*wout):]

    test_BATCH_SIZE = 32

    test_steps = len(X_test) // test_BATCH_SIZE

    # .map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    test_data = (
        tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test.astype(np.uint8)))
        .batch(test_BATCH_SIZE)
        .with_options(options)
        .map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Cross-Validation indexes
    pats = [1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4]
    train_val_idx = np.array([dout*hout*wout*str(i).split(' ') for i in pats]).astype(np.uint8).flatten()


    
    ################################################################################################################
    ############################################### DATA TENSORS ###################################################
    ################################################################################################################
    opt = optimizers[params['opti']]

    # Cross-validation loop GroupKFold K=4 + test_set    
    fold_var = int(params['fold'])
    # fold_var: current patient for test fold

    train_index = np.argwhere(train_val_idx!=fold_var).flatten()
    val_index   = np.argwhere(train_val_idx==fold_var).flatten()

    X_train, X_val = vols[train_index], vols[val_index]
    y_train, y_val = masks[train_index], masks[val_index]

    #del vols, masks

    # Keep patches with at least 0.1% of lesion tissue, for training and validation
    lesion_ratio = 0.001
    ind_y_tr = np.count_nonzero(y_train, axis=(1,2,3), keepdims=False)>int((32**3)*lesion_ratio)
    X_train = X_train[ind_y_tr]
    y_train = y_train[ind_y_tr]                


    SEED = 42 
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32    
    valid_steps = len(X_val) // EVAL_BATCH_SIZE


    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.uint8)))    
        .batch(BATCH_SIZE)
        .with_options(options)    
        .map(Augment(SEED), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)        
        .repeat(8)
        .map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    train_steps = len(train_ds)
        
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.uint8)))
        .batch(EVAL_BATCH_SIZE)
        .with_options(options)
        .map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .prefetch(buffer_size=tf.data.AUTOTUNE)                
    )

    # CREATE NEW MODEL
    print("\n Create Model")        
    
    inputs = Input(shape=(32,32,32, 1))

    print("TrainSteps/ValSteps: "+str(train_steps)+"/"+str(valid_steps))

    del X_train, X_val, y_train, y_val    


    m_arquitecture = params['model']
    model = Unet3D(eval(m_arquitecture), inputs, DiceLoss, opt) 
        
    #unpatch_validation = Val_Mtrc(validation_generator, test_data)    
    #checkpoint = ModelCheckpoint(get_model_name(dataset,fold_var, opt.get_config()['name']), monitor="val_dice", verbose=1, save_best_only=True, mode='max')                
    callbacks_list = [early_stopping]    
            


    ################################################################################################################
    ############################################ TRAIN AND EVALUATION ##############################################
    ################################################################################################################
    print("\nmodel_fit"+str(params))
    #FIT THE MODEL
    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=valid_steps,
        validation_freq=1,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=32,
        epochs=100
    )#Set to 100
    
    results = dict()
    results["train_val"] = history.history
             
    print("\nEnd fit\nSaving Model")
    model.save(get_model_name(m_arquitecture, DATA,fold_var, opt.get_config()['name']))    
    print("\nModel saved!")
    
    # Set evaluations    
    print("Evaluate")
    ev_result = model.evaluate(
        test_data,
        steps=test_steps
    )
    results['test'] = dict(zip(model.metrics_names, ev_result))
    new_test = dict()
    
    for name in results['test']:
        new_test['test_' + name] = results['test'][name]
    results['test'] = new_test    

    print(results)
    # Save results Train, Validation, Test Evaluation
    save_metric_report(results, m_arquitecture, DATA, fold_var, opt.get_config()['name'])
    print("\nHistory saved")

####################################################################################################################
####################################################################################################################
####################################################################################################################
