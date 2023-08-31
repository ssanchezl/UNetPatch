####################################################################################################################
################################################### LOAD LIBRARIES #################################################
####################################################################################################################
import sys    
import numpy as np    
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from Models import three_layer_depth, four_layer_depth, four_layer_depthDrop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation



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
        
        rota = RandomRotation(0.2, 
                            interpolation='nearest', 
                            fill_mode='nearest',
                            seed=SEED)
        
        trans = RandomTranslation(height_factor=(-0.3, 0.3),
                                    width_factor=(-0.3, 0.3),
                                    interpolation='nearest', 
                                    fill_mode='nearest',
                                    seed=SEED)        
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

# Read
def read_data(dataname):    
    X_patch_name = 'MRIs/X_'+dataname+'.npy'
    Y_patch_name = 'MRIs/Y_'+dataname+'.npy'

    with open(X_patch_name, 'rb') as X:
        vols=np.load(X)
        
    with open(Y_patch_name, 'rb') as Y:
        masks=np.load(Y)
        
    return vols, masks              



if __name__ == "__main__":  

    param_stdin = sys.argv[1:]
    params = dict()
    for s in param_stdin:
        key, value = s.split(':')
        params[key] = value     



    ################################################################################################################
    ################################################## Optimizers ##################################################
    ################################################################################################################
    seed = 24    

    optimizers = {                    
                    'SGD09': SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD09"),
                    'SGDNV09': SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name="SGDNV09"),
                    'ADAM': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="ADAM")
                 }



    ################################################################################################################
    ############################################# DATA LOAD############ ############################################
    ################################################################################################################
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')    
    strategy = tf.distribute.MirroredStrategy(logical_gpus)    
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    root_dir = 'MRIs/'    

    DATA = params['data'] 

    vols, masks = read_data(DATA)

    ################################################################################################################
    ########################################### INDEX AND DIMENSIONS UTILS #########################################
    ################################################################################################################        
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    opt = optimizers[params['opti']]

    # GroupKFold for cross-validation
    fold_var = int(params['fold'])
    

    # Cross-Validation indexes
    pats = [1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5]
    train_val_idx = np.array(pats).astype(np.uint8).flatten()


    train_index = np.argwhere(train_val_idx!=fold_var).flatten()
    val_index   = np.argwhere(train_val_idx==fold_var).flatten()

    X_train, X_val = vols[train_index], vols[val_index]
    y_train, y_val = masks[train_index], masks[val_index]

    del vols, masks


    
    ################################################################################################################
    ################################################ DATA AUGMENTATION #############################################
    ################################################################################################################
    SEED = 42 
    BATCH_SIZE = 1

    repeat = 5

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.uint8)))    
        .batch(BATCH_SIZE)
        .with_options(options)
        .map(Augment(SEED), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)    
        .repeat(repeat)
        .map(lambda a,b: exp_lst(a,b), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    train_steps = train_ds.cardinality().numpy()


    VAL_BATCH_SIZE = 1  
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.uint8)))
        .batch(VAL_BATCH_SIZE)
        .with_options(options)
        .map(lambda a,b: exp_lst(a,b), num_parallel_calls=1, deterministic=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)                
    )
    valid_steps = val_ds.cardinality().numpy()

    del X_train, y_train, X_val, y_val

    ################################################################################################################
    ############################################ TRAIN AND EVALUATION ##############################################
    ################################################################################################################
    
    
    # CREATE NEW MODEL
    print("\n Create Model")        

    inputs = Input(shape=(224,224,224, 1))

    print("TrainSteps/ValSteps: "+str(train_steps)+"/"+str(valid_steps)) 


    m_arquitecture = params['model']
    model = Unet3D(eval(m_arquitecture), inputs, DiceLoss, opt) 
    
    #checkpoint = ModelCheckpoint(get_model_name(dataset,fold_var, opt.get_config()['name']), monitor="val_dice", verbose=1, save_best_only=True, mode='max')                    
    
    print("\nmodel: "+m_arquitecture)
    #FIT THE MODEL
    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=10
    )#Set to 100
    
    results = dict()
    results["train"] = history.history
             
    print("\nEnd fit\nSaving Model")
    model.save(get_model_name(m_arquitecture, DATA,fold_var, opt.get_config()['name']))    
    print("\nModel saved!")
    
    # Set evaluations    
    print("Evaluate")
    ev_result = model.evaluate(val_ds)
    results['test'] = dict(zip(model.metrics_names, ev_result))
    
    new_test = dict()    
    for name in results['test']:
        new_test['test_' + name] = results['test'][name]
    results['test'] = new_test    

    print(results)
    # Save results Train, Test Evaluation
    save_metric_report(results, m_arquitecture, DATA, fold_var, opt.get_config()['name'])
    print("\nHistory saved")

####################################################################################################################
####################################################################################################################
####################################################################################################################
