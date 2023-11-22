"""
This script processes all *.jpg files in the 'original_data/' directory,
converts them to grayscale using the skimage library,
and saves the resulting images in the 'original_data_grayscale/' directory.
"""

import os
from pathlib import Path
from skimage import color
import imageio

input_dir = 'original_data/'
output_dir = 'original_data_grayscale/'

for file in os.listdir(input_dir):
   print('File: ' + file)
   # Process images with *.jpg
   if Path(input_dir + file).suffix == '.jpg':
       # Load the image
       image = imageio.imread(input_dir + file)
       imageGray = color.rgb2gray(image)
       print('Converted to grayscale')
       # Save the image
       imageio.imwrite(output_dir + Path(file).stem + '_grayscale.jpg', imageGray)
       print(str(file) + ' saved to ' + str(output_dir))
