import tensorflow as tf

# Define feature description
feature_description = {
    'x': tf.io.FixedLenFeature([], tf.string),
    'y': tf.io.FixedLenFeature([], tf.string)
}

# Parse a serialized example
def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.bool)
    return x, y

# Load and concatenate datasets
filenames = ['test.tfrecord', 'test1.tfrecord', 'test2.tfrecord', 'test3.tfrecord']
dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=len(filenames))
																				# TF Performance tips
dataset = dataset.map(parse_example)

#N_de_parches_por_volumen
Np = 7*7*7-ISBI / 5*6*5-2008MNI / 9*12*9 2008ORI
ISBI=           Test[:4*np]-Train[4*Np:], 
        		Test[4*np:8*np]-Train[:4*Np]U[8*Np:],
                Test[8*np:12*np]-Train[:8*np]U[12*np:]
        		Test[12:17*np]-Train[:12*Np]U[17*Np:],		
        		Test[17*np:]-Train[:17*Np]

MICCAI2016=     Test[:1016]-Train[1016:], 
                Test[1016:1704]-Train[:1016]U[1704:],
                Test[1704:2872]-Train[:1704]U[2872:],      
                Test[2872:3836]-Train[:2872]U[3836:],
                Test[3836:]-Train[:3836]

MICCAI2008MNI=  Test[:4*np]-Train[4*Np:], 
                Test[4*np:8*np]-Train[:4*Np]U[8*Np:],
                Test[8*np:12*np]-Train[:8*np]U[12*np:]
                Test[12:16*np]-Train[:12*Np]U[16*Np:],      
                Test[16*np:]-Train[:16*Np]

MICCAI2008ORI=  Test[:4*np]-Train[4*Np:], 
                Test[4*np:8*np]-Train[:4*Np]U[8*Np:],
                Test[8*np:12*np]-Train[:8*np]U[12*np:]
                Test[12:16*np]-Train[:12*Np]U[16*Np:],      
                Test[16*np:]-Train[:16*Np]


