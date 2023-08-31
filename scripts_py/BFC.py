import SimpleITK as sitk
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="path to the input NIfTI file")
parser.add_argument("output_path", help="path to save the output NIfTI file")
args = parser.parse_args()

# Read the input image
input_image = sitk.ReadImage(args.input_path, sitk.sitkFloat32)

# Create a bias field correction filter object with default settings
corrector = sitk.N4BiasFieldCorrectionImageFilter()

maskImage = sitk.OtsuThreshold(input_image, 0, 1, 200)

corrected = corrector.Execute(input_image, maskImage)

# Save the output image
log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
corrected_image_full_resolution = input_image / sitk.Exp(log_bias_field)
sitk.WriteImage(corrected_image_full_resolution, args.output_path)