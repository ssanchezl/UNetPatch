import argparse
import numpy as np
from skimage import exposure
import nibabel as nib

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="path to the input NIfTI file")
parser.add_argument("output_path", help="path to save the output NIfTI file")
args = parser.parse_args()

# Load the input image
input_img = nib.load(args.input_path)
img_data = input_img.get_fdata().astype(np.float32)

# Normalize intensity values to be between 0 and 1
img_data = exposure.rescale_intensity(img_data, out_range=(0, 1))

# Calculate clip limit based on mean and standard deviation of the image
mean_val = np.mean(img_data)
std_val = np.std(img_data)
clip_limit = 0.015 * (mean_val / std_val)

# Apply CLAHE with calculated clip limit
clahe = exposure.equalize_adapthist(img_data, clip_limit=clip_limit, kernel_size=(8,8,8), nbins=256)

# Set affine matrix of output image to be same as input image
nib.save(nib.Nifti1Image(clahe, input_img.affine), args.output_path)
