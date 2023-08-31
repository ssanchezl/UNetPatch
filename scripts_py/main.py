import sys
import os
import numpy as np

programa = "MSSEG_tfr.py"
aug = "True"

split = range(5)
corrida = range(8,10)
model = 'UNet3D_A'

resolution = ['ORI', 'MNI']
dataset = ["ISBI", "MICCAI2016"]# agregar "MICCAI2008" si hac falta (pero no pq es horrible)
opt = 'ADAM'
for R in corrida:
    for res in resolution:
        for data in dataset:        
            for K in split:
                command = "python3 "+programa+" corrida:"+str(R)+" model:"+model+" dataset:"+data+" res:"+res+" opti:"+opt+" split:"+str(K)
                sys.stdout.flush()
                exitCode = os.system(str(command))
print("\nFIN\n")        
exit()




