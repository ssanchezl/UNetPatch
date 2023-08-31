import numpy as np

root_dir = 'saved_models/'    

# MODIFICAR indice para cambiar entre conjunto A=0 o B=1
########################################################
DATA = 'history_ISBI_MNI_val_patient_4Fold_SGD04Opt.npy'
########################################################    

res_path = root_dir+DATA

# Carga de parches registrados
with open(res_path, 'rb') as X:
    res = np.load(X, allow_pickle=True)

print(res)