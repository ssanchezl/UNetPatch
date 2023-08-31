from patchify import patchify
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, cast
import SimpleITK as sitk # instalar con conda

def _unpatchify2d(  # cambiar nombre para evitar ambigüedad
    patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    
    image1 = np.zeros(imsize, dtype=patches.dtype)
    image2 = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape    

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)
    
    s_w = int(s_w)
    s_h = int(s_h)

    i, j = 0, 0

    while True:
        i_o, j_o = i * s_h, j * s_w
        if i%2==0 and j%2==0:
            image1[i_o : i_o + p_h, j_o : j_o + p_w] = patches[i, j]
            
        elif i%2!=0 and j%2!=0:
            image2[i_o : i_o + p_h, j_o : j_o + p_w] = patches[i, j]            

        if j < n_w - 1:
            j = min((j_o + s_w) // s_w, n_w - 1)
        elif i < n_h - 1 and j >= n_w - 1:
            # Go to next row
            i = min((i_o + s_h) // s_h, n_h - 1)
            j = 0
        elif i >= n_h - 1 and j >= n_w - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")
    
    return image1, image2


#mask = np.array(ImageOps.grayscale(Image.open(r"./fotos_utiles/staples/224.jpeg")))
#mask = np.array(ImageOps.grayscale(Image.open(r"./fotos_utiles/EjemploAxial.png"))) # cambiar a mascara!!!!
mask = np.array(ImageOps.grayscale(Image.open(r"./fotos_utiles/volumeSlices/maskslice0101consenspp.png")), dtype=bool)

patches = patchify(mask, (32,32), step=16)
img1, img2 = _unpatchify2d(patches, mask.shape)


def staple(img1, img2):
    seg1_sitk = sitk.GetImageFromArray(img1.astype(np.int16))
    seg2_sitk = sitk.GetImageFromArray(img2.astype(np.int16))
        
    seg_stack = [seg1_sitk, seg2_sitk]
    # Run STAPLE algorithm
    STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 ) # 1.0 specifies the foreground value

    # convert back to numpy array
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)

    return STAPLE_seg

stpl = staple(img1, img2)

plt.imshow(stpl, cmap='gray')
plt.show()
# f, axarr = plt.subplots(1,2) 
# axarr[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
# axarr[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
# plt.show()