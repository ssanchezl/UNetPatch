import tensorflow as tf
import numpy as np

def dice(y_true, y_pred):# y_pred must be binarized!!!
	return score

def umbral_search(y_true, y_pred):
	max_s = -float('inf')
	for umbral in np.arange(0, 1, 0.01, dtype=np.float32):
		y_pred = y_pred > umbral # binarize predictions
		score = dice(y_true, y_pred)
		if score > max:
			max = score

	return score, y_pred # maybe the umbral?

# Prediction callback
class PredictCallback(Callback):
    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
