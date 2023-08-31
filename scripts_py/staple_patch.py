import numpy as np
from patchify import patchify
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
class NonUniformStepSizeError(RuntimeError):
    def __init__(
        self, imsize: int, n_patches: int, patch_size: int, step_size: float
    ) -> None:
        super().__init__(imsize, n_patches, patch_size, step_size)
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.imsize = imsize
        self.step_size = step_size

    def __repr__(self) -> str:
        return f"Unpatchify only supports reconstructing image with a uniform step size for all patches. \
However, reconstructing {self.n_patches} x {self.patch_size}px patches to an {self.imsize} image requires {self.step_size} as step size, which is not an integer."

    def __str__(self) -> str:
        return self.__repr__()

from typing import Tuple, Union, cast
def _unpatchify2d(  # pylint: disable=too-many-locals
    patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:

    assert len(patches.shape) == 4

    i_h, i_w = imsize    

    n_h, n_w, p_h, p_w = patches.shape        

    # stride 16x16
    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1) #16
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1) #16    

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
       raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
       raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)    

    image1 = np.zeros((i_h-p_h, i_w-p_w), dtype=patches.dtype)
    image2 = np.zeros((i_h-p_h, i_w-p_w), dtype=patches.dtype)

    patches = np.reshape(patches, (n_h*2,n_w*2, p_h//2,p_w//2))    
    
    i, j = 1, 1    
    lista1 = list()
    lista2 = list()
    while True:        
              
        i_o, j_o = i * s_h, j * s_w        
        
        if i%((p_h+p_h)//s_h) == 1 and j%((p_w+p_w)//s_w) == 1:
            #image1[i_o : i_o + p_h, j_o : j_o + p_w] = patches[i, j]            

            # print((i_o-s_h)//2, ":",(i_o-s_h)//2+p_h, (j_o-s_w)//2, ":",(j_o-s_w)//2+p_w)
                
            l_u = patches[i, j]
            r_u = patches[i, j+3]
            l_d = patches[i+3,j]
            r_d = patches[i+3,j+3]

            #u = np.concatenate((l_u, r_u), axis=0)
            #d = np.concatenate((l_d, r_d), axis=0)

            #image1[(i_o-s_h)//2:(i_o-s_h)//2+p_h, (j_o-s_w)//2:(j_o-s_w)//2+p_w] = np.concatenate((u, d), axis=1)
            print(i,j)
            #print(lx, " U ", rx, " | ", ly, " U ", ry)
            #print(len(str(lx))//2*" ", midx,(len(str(lx))+10)*" ", midy)
            #image1[i_o : i_o + p_h, j_o : j_o + p_w] = patches[ind_x[:,None], ind_y]
            # image1[(i_o-s_h)//2:(i_o-s_h)//2+p_h, (j_o-s_h)//2:(j_o-s_w)//2+p_w] = patches[i//((p_h+p_h)//s_h), j//((p_w+p_w)//s_w), ind_x[:,None], ind_y]
            # image2[(i_o-s_h)//2:(i_o-s_h)//2+p_h, (j_o-s_h)//2:(j_o-s_w)//2+p_w] = patches[i//((p_h+p_h)//s_h), j//((p_w+p_w)//s_w), midx, midy]
        
            
        if j < n_w + n_w - 2:
            j = min((j_o + s_w) // s_w, n_w + n_w - 2)
        elif i < n_h + n_h - 2 and j >= n_w + n_w - 2:
            # Go to next row
            i = min((i_o + s_h) // s_h, n_h + n_h - 2)
            j = 1
        elif i >= n_h + n_h - 2 and j >= n_w + n_w - 2:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")

    # plt.imshow(image1, cmap='gray', vmin=0, vmax=255) 
    # plt.show()
    #print(image1.shape)
    # f, axarr = plt.subplots(2,1) 
    # axarr[0].imshow(image1, cmap='gray', vmin=0, vmax=255)
    # axarr[1].imshow(image2, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    #image = staple(image1, image2)
    return 0,0#image1, image2

mask = np.array(ImageOps.grayscale(Image.open(r"./fotos_utiles/staples/224.jpeg")))
#plt.imshow(mask, cmap='gray', vmin=0, vmax=255) 
#plt.show()

patches = patchify(mask, (32,32), step=16)
img1, img2 = _unpatchify2d(patches, mask.shape)

# f, axarr = plt.subplots(2,1) 
# axarr[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
# axarr[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
# plt.show()

# print(img1.shape)
# print(img2.shape)

# simg1 = Image.fromarray(img1).convert("L")
# simg2 = Image.fromarray(img2).convert("L")

# simg1.save("./fotos_utiles/staples/pre_stpl_1.jpeg")
# simg1.save("./fotos_utiles/staples/pre_stpl_2.jpeg")
