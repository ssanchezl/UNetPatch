#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:05:34 2022

@author: gustavo
"""

import tensorflow.keras.backend as K
import tensorflow as tf

# =============================================================================
# Funciones de pérdidas
# =============================================================================
def BCE(y_true,y_pred):
    # tf.print('Before',y_true.shape,y_pred.shape)
    # y_true = K.reshape(y_true, [-1])
    # y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1,256,256])
    y_pred = K.reshape(y_pred, [-1,256,256])
    # tf.print('After',y_true.shape,y_pred.shape)
    Y0 = (1-y_true)*K.log(K.clip(1-y_pred,K.epsilon(),1)) # FP
    Y1 = y_true*K.log(K.clip(y_pred,K.epsilon(),1)) # FN
    return K.mean(-Y0 - Y1)

def Dice(y_true,y_pred):
    # y_true = K.reshape(y_true, [-1])
    # y_pred = K.reshape(y_pred, [-1])
    
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])
    FN = tf.math.reduce_sum(y_predc[y_true==1])
    TN = tf.math.reduce_sum(y_predc[y_truec==1])
    FP = tf.math.reduce_sum(y_pred[y_truec==1])
    return K.mean( 1 - 2*TP/(2*TP+FP+FN) )

def Dice_Milletari(y_true,y_pred):
    return K.mean( 1 - 2*tf.math.reduce_sum(y_true*y_pred)/(tf.math.reduce_sum(y_true)+tf.math.reduce_sum(y_pred)) )

# =============================================================================
# Métricas
# =============================================================================
def Metric_Accuracy(y_true,y_pred):
    Y0 = (1-y_true)*(1-y_pred)
    Y1 = y_true*y_pred
    return K.mean(Y0 + Y1)

# def Metric_Accuracy(y_true,y_pred): # Trabaja igual que el 'accuracy' de tensorflow
#     return K.mean(K.equal(y_true, K.round(y_pred)))

def Metric_Dice_Milletari(y_true,y_pred):
    return K.mean( 2*tf.math.reduce_sum(y_true*y_pred)/(tf.math.reduce_sum(y_true)+tf.math.reduce_sum(y_pred)) )

def Metric_Dice_Milletari_threshold05(y_true,y_pred):
    y_pred = tf.round(y_pred)
    return K.mean( 2*tf.math.reduce_sum(y_true*y_pred)/(tf.math.reduce_sum(y_true)+tf.math.reduce_sum(y_pred)) )

def Metric_Dice(y_true,y_pred):
    # y_true = K.reshape(y_true, [-1])
    # y_pred = K.reshape(y_pred, [-1])
    
    y_truec = 1-y_true
    y_predc = 1-y_pred
    TP = tf.math.reduce_sum(y_pred[y_true==1])
    FN = tf.math.reduce_sum(y_predc[y_true==1])
    TN = tf.math.reduce_sum(y_predc[y_truec==1])
    FP = tf.math.reduce_sum(y_pred[y_truec==1])
    return K.mean( 2*TP/(2*TP+FP+FN) )