import tensorflow as tf
import numpy as np
from patchify import unpatchify

# # Parse a serialized example
# def parse_example(serialized_example):
#     example = tf.io.parse_single_example(serialized_example, feature_description)
#     x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
#     y = tf.cast(tf.io.parse_tensor(example['y'], out_type=tf.bool), tf.uint8)#era tf.bool
#     return x, y


# Train Test Split
def spliter(ds, DATA, RES, KSplit):
    
    Np = None
    dims = None

    if DATA == 'ISBI':
        r = [0,4,8,13,17,21]

        Np = 7*7*7

        start = r[KSplit]*Np
        end = r[KSplit+1]*Np

        train = ds.take(start).concatenate(ds.skip(end))
        test = ds.skip(start).take(end-start)
        # test = list(test.as_numpy_iterator())
        # x, y = zip(*test)
        # x = np.array(x, dtype=np.float32)
        # y = np.array(y, dtype=np.uint8)

        # test_x = []
        # test_y = []        
        # for i in range(1, r[KSplit+1]-r[KSplit] + 1):
        #     z1 = unpatchify(np.reshape(x[Np*(i-1):Np*i], (7,7,7,32,32,32)), (224,224,224))
        #     z2 = unpatchify(np.reshape(y[Np*(i-1):Np*i], (7,7,7,32,32,32)), (224,224,224))
        #     test_x.append(z1)
        #     test_y.append(z2)
        # test_x = np.array(test_x, dtype=np.float32)
        # test_y = np.array(test_y, dtype=np.uint8)
        # test = zip(test_x,test_y)

    elif DATA == 'MICCAI2016':
        # Define a function to pad each element individually
        def pad_element(element):
            paddings = tf.stack([[0, max_shape[0] - tf.shape(element)[0]], [0, max_shape[1] - tf.shape(element)[1]], [0, max_shape[2] - tf.shape(element)[2]]])
            padded_element = tf.pad(element, paddings)
            return padded_element

        dims = [
            (224,256,256),
            (224,256,256),
            (160,192,128),
            (160,192,128),
            (160,192,128),
            (224,256,256),
            (224,256,256),
            (288,320,128),
            (288,320,128),
            (288,352,128),
            (160,192,128),
            (224,256,256),
            (288,352,128),
            (288,384,128),
            (160,192,128)
        ]

        s = int((KSplit/5)*len(dims))
        e = int(((KSplit+1)/5)*len(dims))+1

        #patches per vol
        patvol = [[a//32,b//32,c//32] for a,b,c in dims]                

        cum = [0] + list(np.cumsum([np.prod((a//32,b//32,c//32)) for a,b,c in dims]))
        r = cum[::3]        

        start = r[KSplit]
        end = r[KSplit+1]                
                 
        train = ds.take(start).concatenate(ds.skip(end))
        test = ds.skip(start).take(end-start)                        
        # test = list(test.as_numpy_iterator())
        # x, y = zip(*test)
        # x=np.array(x, dtype=np.float32)
        # y=np.array(y, dtype=np.uint8)

        # ts_cum = [0] + list(np.cumsum([np.prod(i) for i in patvol[s:e-1]]))        
                          
        # test_x = []
        # test_y = []
        # for i in range(3):        	        	
        #     z1 = unpatchify(np.reshape(x[ts_cum[i]:ts_cum[i+1]], patvol[s:e-1][i]+[32, 32, 32]), dims[s:e-1][i])
        #     z2 = unpatchify(np.reshape(y[ts_cum[i]:ts_cum[i+1]], patvol[s:e-1][i]+[32, 32, 32]), dims[s:e-1][i])            
        #     test_x.append(z1)
        #     test_y.append(z2)
            
        # max_shape = tf.reduce_max([tf.shape(x) for x in test_x], axis=0)
        # test_x = np.array([pad_element(x) for x in test_x], dtype=np.float32)
        # test_y = np.array([pad_element(y) for y in test_y], dtype=np.float32)
        # test = zip(test_x,test_y)

    elif DATA == 'MICCAI2008':

        if RES=='MNI':
            Np = 5*6*5
            dims = [5,6,5]

        elif RES=='ORI':        
            Np = 9*12*9
            dims = [9,12,9]

        r = [0,4,8,12,16,20]  

        final_dim = [x * y for x, y in zip(dims, [32,32,32])]      

        start = r[KSplit]*Np
        end = r[KSplit+1]*Np
    
        #Test[:]-Train[:]U[:]
        train = ds.take(start).concatenate(ds.skip(end))
        test = ds.skip(start).take(end-start)
        # test = list(test.as_numpy_iterator())
        # x, y = zip(*test)
        # x=np.array(x, dtype=np.float32)
        # y=np.array(y, dtype=np.uint8)
        
        # test_x = []
        # test_y = []
        # for i in range(1, r[KSplit+1]-r[KSplit] + 1):                        
        #     z1 = unpatchify(np.reshape(x[Np*(i-1):Np*i], dims+[32,32,32]), final_dim)            
        #     z2 = unpatchify(np.reshape(y[Np*(i-1):Np*i], dims+[32,32,32]), final_dim)
        #     test_x.append(z1)
        #     test_y.append(z2)
        # test_x = np.array(test_x, dtype=np.float32)
        # test_y = np.array(test_y, dtype=np.uint8)
        # test = zip(test_x,test_y)

    return train, test

# def filter_dataset_by_lesion_percentage(dataset, p):    
#     filtered_dataset = dataset.filter(lambda x, y: tf.cast(tf.math.reduce_sum(y), tf.float32) > np.float32((p/100)*32*32*32) )
#     return filtered_dataset

# def filter_dataset_by_non_black_blocks(dataset):    
#     filtered_dataset = dataset.filter(lambda x, y: tf.math.reduce_sum(x) > np.float32(0.0) )
#     return filtered_dataset

# def counter(dataset):
# 	count = 0
# 	for c in dataset:
# 		count +=1
# 	return count



# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)    
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA



# DATAs = ["MICCAI2008", "MICCAI2016", "ISBI"]
# RES = "ORI"
# K = 4

# for DATA in DATAs:
#     ds_all = "/home/ssanchez/Tesis2021/tfrecord/"+DATA+"_"+RES+".tfrecord"

#     # Define feature description
#     feature_description = {
#         'x': tf.io.FixedLenFeature([], tf.string),
#         'y': tf.io.FixedLenFeature([], tf.string)
#     }

#     # Load the dataset for split
#     split_ds = tf.data.TFRecordDataset(ds_all, num_parallel_reads=1)
#     split_ds = split_ds.map(parse_example, num_parallel_calls=1, deterministic=True)


#     train_1, test = spliter(split_ds, DATA, RES, K)
#     x, y = zip(*test)
#     x = np.array(x, dtype=np.float32)
#     y = np.array(y, dtype=np.uint8)
#     print(x.shape)    




# print(test.element_spec)
# for x,y in test:
#     print(x.shape, y.shape)

# print(f'\nRaw ds length: {counter(train_1)}')

# train_1 = filter_dataset_by_non_black_blocks(train_1)

# print(f'\nNon black ds length: {counter(train_1)}')

# p = 0.5
# train_2 = filter_dataset_by_lesion_percentage(train_1, p)

# print(f'\nFilter by lesion ds length: {counter(train_2)}')


# SEED = 42 
# BATCH_SIZE = 64    

# # tissue
# repeat_1 = 2

# # Augmentation of the normal tissue
# train_1 = prep_dataset(
#     train_1, 
#     BATCH_SIZE,
#     options,
#     SEED,
#     repeat_1,
#     True, #augmentation
#     False #determinated addressing
# )  
# print(f'\nAug non black ds length: {counter(train_1)}')



# # lession
# repeat_2 = 10

# # Augmentation of the lession tissues
# train_2 = prep_dataset(
#     train_2, 
#     BATCH_SIZE,
#     options,
#     SEED,
#     repeat_2,
#     True, #augmentation
#     False #determinated addressing
# )
# print(f'\nAug lesion ds length: {counter(train_2)}')

# train = train_1.concatenate(train_2).prefetch(buffer_size=tf.data.AUTOTUNE)
# print(f'\nFinal ds length: {counter(train)}')

