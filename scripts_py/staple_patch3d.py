import numpy as np
from typing import Tuple, Union, cast
import SimpleITK as sitk


def scratch_unpatchify3d(   
    patches: np.ndarray, imsize: Tuple[int, int, int]
) -> np.ndarray:

    assert len(patches.shape) == 6

    i_h, i_w, i_c = imsize
    image1 = np.zeros(imsize, dtype=patches.dtype)
    image2 = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, n_c, p_h, p_w, p_c = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)
    s_c = 0 if n_c <= 1 else (i_c - p_c) / (n_c - 1)

    s_w = int(s_w)
    s_h = int(s_h)
    s_c = int(s_c)

    i, j, k = 0, 0, 0

    while True:

        i_o, j_o, k_o = i * s_h, j * s_w, k * s_c

        if i%2==0 and j%2==0 and k%2==0:
            image1[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = patches[i, j, k]
            
        elif i%2!=0 and j%2!=0 and k%2!=0:
            image2[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = patches[i, j, k]

        if k < n_c - 1:
            k = min((k_o + p_c) // s_c, n_c - 1)
        elif j < n_w - 1 and k >= n_c - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
            k = 0
        elif i < n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
            k = 0
        elif i >= n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")

    return image1, image2


def staple(img):

    img1, img2 = img
    seg1_sitk = sitk.GetImageFromArray(img1.astype(np.uint8))
    seg2_sitk = sitk.GetImageFromArray(img2.astype(np.uint8))
        
    seg_stack = [seg1_sitk, seg2_sitk]
    # Run STAPLE algorithm
    STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 ) # 1.0 specifies the foreground value

    # convert back to numpy array
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk).astype(np.float32)

    return STAPLE_seg