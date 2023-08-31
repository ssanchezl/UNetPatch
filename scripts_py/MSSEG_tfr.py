####################################################################################################################
################################################### LOAD LIBRARIES #################################################
####################################################################################################################
import sys    
import numpy as np    
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from Models import three_layer_depth, four_layer_depth, four_layer_depthDrop, UNet3D_A
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from spliter import spliter


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
print("\n"+str(len(gpus)), "Physical GPUs,", len(logical_gpus), "Logical GPU")
####################################################################################################################
############################################### BASELINE UNET ######################################################
####################################################################################################################
def Unet3D(model_arq, inputs, loss, opt):    
    
    #with strategy.scope():
    with tf.device('/device:GPU:0'):
        x=inputs
        outputs = model_arq(x)        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=loss, optimizer=opt, metrics=[dice])
        return model



####################################################################################################################
############################################ METRIC CALLBACKS AND UTILS ############################################
####################################################################################################################
# Get the approximate size of ds
def counter(dataset):
    count = 0
    for c in dataset:
        count +=1
    return count

# Dimension utility
def exp_lst(a,b):
    a, b = tf.expand_dims(a, axis=-1), tf.expand_dims(b, axis=-1)    
    return a,b

# Save the Net parameters in the save_model dir
def get_model_name(arq_name, run, dataset, res, fold, opt):
    save_dir = '../saved_models/corrida_'+run+'/'
    name = save_dir+'model_'+arq_name+'_'+dataset+'_'+res+'_'+str(fold)+'Fold_'+opt+'_Opt'+'.h5'       
    return name

# Save the Net report in the save_model dir
def save_metric_report(results, arq_name, run, dataset, res, fold, opt):
    save_dir = '../saved_models/corrida_'+run+'/'
    name = save_dir+'history_'+arq_name+'_'+dataset+'_'+res+'_'+str(fold)+'Fold_'+opt+'_Opt'+'.npy'
    np.save(name,results)

# Metrics function definitions
def metrics(y_true,y_pred):
    true_positives = (tf.reduce_sum(y_true * y_pred))
    false_positives = (tf.reduce_sum((1 - y_true) * y_pred))
    false_negatives = (tf.reduce_sum(y_true * (1 - y_pred)))
    
    precision = tf.reduce_mean(true_positives / (true_positives + false_positives + K.epsilon()))
    recall = tf.reduce_mean(true_positives / (true_positives + false_negatives + K.epsilon()))
    
    metric_dice = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return recall, precision, metric_dice

def dice(y_true, y_pred):
    true_positives = (tf.reduce_sum(y_true * y_pred))
    false_positives = (tf.reduce_sum((1 - y_true) * y_pred))
    false_negatives = (tf.reduce_sum(y_true * (1 - y_pred)))
    
    precision = tf.reduce_mean(true_positives / (true_positives + false_positives + K.epsilon()))
    recall = tf.reduce_mean(true_positives / (true_positives + false_negatives + K.epsilon()))

    tr_dice = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return tr_dice


# Loss function definitions
def DiceLoss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)        

    true_positives = (tf.reduce_sum(y_true * y_pred))
    false_positives = (tf.reduce_sum((1 - y_true) * y_pred))
    false_negatives = (tf.reduce_sum(y_true * (1 - y_pred)))

    # Tversky verion
    a = 0.7
    b = 0.3

    # Focal coeficient
    y = 4/3
    
    loss = 1 - 2*true_positives / (true_positives + a*false_negatives + b*false_positives + K.epsilon())
    focal_loss = K.clip(loss, K.epsilon(), 1.0) ** y
    return tf.reduce_mean(focal_loss)
 

# Parse a serialized example
def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.cast(tf.io.parse_tensor(example['y'], out_type=tf.bool), tf.uint8)
    return x, y


# Data Augmentation
class Augment(tf.keras.layers.Layer):
    def __init__(self, SEED=42):
        super().__init__()
        
        flip = RandomFlip("horizontal", seed=SEED)                
        
        trans = RandomTranslation(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            interpolation='nearest', 
            fill_mode='nearest',
            seed=SEED
        )  

        zoom = RandomZoom(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            fill_mode='nearest',
            interpolation='nearest',
            seed=SEED
        )      
        layers = [flip, trans, zoom]
        self.aug_model = tf.keras.Sequential(layers)
                
    def call(self, image, mask):
        
        mask = tf.cast(mask, tf.float16)

        images_mask = tf.concat([image, mask], -1)  

        images_mask = tf.ensure_shape(images_mask, tf.TensorSpec(shape=(32,32,32,2), dtype=tf.float16).shape)        
        
        images_mask = self.aug_model(images_mask, training=True)
        
        dim_fus = images_mask.shape[-1]//2

        image = images_mask[..., :dim_fus]
        mask  = images_mask[..., dim_fus:]
        mask  = tf.where(mask>0, tf.ones(tf.shape(mask)), tf.zeros(tf.shape(mask)))        
        
        return image, tf.cast(mask, tf.uint8)

def filter_dataset_by_lesion_percentage(dataset, train_samples):    
    filtered_dataset = dataset.filter(lambda x, y: tf.cast(tf.math.reduce_sum(y), tf.float32) > np.float32((0.001)*32*32*32) )  
    count = counter(filtered_dataset)
    rep_lp = int((100*BATCH_SIZE) / count)+1
    filtered_dataset = filtered_dataset.map(lambda x,y: exp_lst(x,y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    filtered_dataset = (
        filtered_dataset
        .repeat(rep_lp)
        .map(Augment())
        .map(lambda x,y: (tf.cast(x, tf.float32), tf.cast(y, tf.uint8)))
    )    
    return count, rep_lp, filtered_dataset

def filter_dataset_by_non_black_blocks(dataset, rep):       
    filtered_dataset = dataset.filter(lambda x, y: tf.math.reduce_sum(x) > np.float32(0.0) )    
    filtered_dataset = filtered_dataset.map(lambda x,y: exp_lst(x,y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    filtered_dataset = filtered_dataset.repeat(rep)
    return filtered_dataset


if __name__ == "__main__":  

    param_stdin = sys.argv[1:]
    params = dict()
    for s in param_stdin:
        key, value = s.split(':')
        params[key] = value 

    RUN    = params['corrida']
    DATA   = params['dataset'] 
    RES    = params['res']
    KSplit = int(params['split'])
    OPTI   = params['opti']



    ################################################################################################################
    ############################################# DATA LOAD ########################################################
    ################################################################################################################       
    ds_all = "/home/ssanchez/Tesis2021/tfrecord/"+DATA+"_"+RES+".tfrecord"


    # Define feature description
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string)
    }   
    
    # Load the dataset for split
    split_ds = tf.data.TFRecordDataset(ds_all, num_parallel_reads=1)
    split_ds = split_ds.map(parse_example, num_parallel_calls=1, deterministic=True)

    train, test = spliter(split_ds, DATA, RES, KSplit)
    


    ################################################################################################################
    ########################################### DATA SHARDING AND GPU UTILS ########################################
    ################################################################################################################        
    #gpus = tf.config.list_logical_devices('GPU')[0]
    #strategy = tf.distribute.MirroredStrategy(gpus)    
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))    


    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Get the output of the model
    mixed_precision.set_global_policy('mixed_float16')
    


    ################################################################################################################
    ################################################## Optimizers ##################################################
    ################################################################################################################
    optimizers = {                    
                    'SGD09': SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD09"),
                    'SGDNV09': SGD(learning_rate=0.1, momentum=0.9, nesterov=True, name="SGDNV09"),
                    'ADAM': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="ADAM")
                 }

    opt = optimizers[OPTI]    


    
    ################################################################################################################
    ################################################ DATA AUGMENTATION #############################################
    ################################################################################################################    
    # Datasets batch size
    BATCH_SIZE = 64
    VAL_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 32

    train_samples = counter(train)*0.9
    val_samples = (train_samples/0.9)*0.1

    val = train.skip(int(train_samples)).map(lambda a,b: exp_lst(a,b))
    train = train.take(int(train_samples))    

    # Augmented lesion tissue (Lesion Patches)        
    filtered, rep_lp, train_lt = filter_dataset_by_lesion_percentage(train, train_samples)
       

    # Augmented general tissue (Non-Black patches)
    rep_nb = int((filtered * rep_lp) / (train_samples))+1
    rep_nb = rep_nb if rep_nb > 1 else 2
    train_gt = filter_dataset_by_non_black_blocks(train, rep_nb)    

    # Merge augmentations
    train = train_lt.concatenate(train_gt)

    total_samples = counter(train)

    print(f'train_samples: {train_samples}, filtered: {filtered}, rep_lp: {rep_lp}, rep_nb: {rep_nb}, total_samples: {total_samples}')    
        
    train = train.repeat()    

    train_steps = (total_samples//BATCH_SIZE)+1
    val_steps = (val_samples//VAL_BATCH_SIZE)+1
    print(f'\nRaw ds length: {total_samples}')
    print(f"Train_Steps:  {train_steps} \t Val_Steps {val_steps}")  
    
    num_classes = 1

    train = (
        train
        .shuffle(1000)
        .batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)        
        .with_options(options)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )   

    val = (
        val        
        .batch(VAL_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)        
        .with_options(options)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )



    ################################################################################################################
    ############################################ TRAIN AND EVALUATION ##############################################
    ################################################################################################################    
    print("\n Create Model")    

    inputs = Input(shape=(None, None, None, 1))
    
    WEIGHT=None
    m_arquitecture = params['model']
    model = Unet3D(eval(m_arquitecture), inputs, DiceLoss, opt)
        
    print(f"\nmodel: {m_arquitecture}")

    early_stopping =  EarlyStopping(patience=40,
                                    monitor="val_loss",
                                    min_delta=1e-4,
                                    restore_best_weights=True,
                                    mode="min",
                                    verbose=1)

    callback_list = [early_stopping]
    # FIT THE MODEL
    history = model.fit(
        train,         
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callback_list,
        epochs=100
    )#Set to 100
    
    
    results = dict()
    results["train"] = history.history
             
    print("\nEnd fit\nSaving Model")
    model.save(get_model_name(m_arquitecture, RUN, DATA, RES, KSplit, opt.get_config()['name']))    
    print("\nModel saved!")
    
    # Set evaluations    
    print("Evaluate")


    test = test.map(lambda a,b: exp_lst(a,b))

    x = test.map(lambda a,b: a)
    y_true = tf.convert_to_tensor(list(test.map(lambda a,b: b).as_numpy_iterator()), dtype=tf.float32)
        
    predictions = tf.convert_to_tensor(model.predict(x.with_options(options).batch(TEST_BATCH_SIZE)))
                

    thresholds = np.arange(0, 1, 0.1)
    dice_scores = []
    precision_scores = []
    recall_scores = []
    for threshold in thresholds:
        binary_predictions = np.where(predictions >= threshold, 1, 0)        

        _recall, _precision, _dice = metrics(y_true, binary_predictions)

        dice_scores.append(_dice)
        precision_scores.append(_precision)
        recall_scores.append(_recall)

    ix = np.argmax(dice_scores)
    best_dice         = dice_scores[ix]
    best_precision    = precision_scores[ix]
    best_recall       = recall_scores[ix]
    best_threshold    = thresholds[ix]
    
    

    results['test'] = {"dice":best_dice.numpy(), "precision":best_precision.numpy(), "recall":best_recall.numpy(), "threshold":best_threshold}

    print(results['test'])
    # Save results Train, Test Evaluation
    save_metric_report(results, m_arquitecture, RUN, DATA, RES, KSplit, opt.get_config()['name'])
    print("\nHistory saved")    

####################################################################################################################
####################################################################################################################
####################################################################################################################