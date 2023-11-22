"""
This script prepares image data for a digit recognition model training by organizing images into
train and test directories. It creates subfolders for each digit label (0 to 9) within 'train/' and 'test/' directories
and copies images from the 'original_data_grayscale/' directory based on filenames.
"""

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed, random

# Name and create the subfolders
dataset_home = 'images/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # Create directories with appropriate names
    labeldirs = ['0/', '1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(name=newdir, exist_ok=True)

# Create directories for models and plots
makedirs(name='models/', exist_ok=True)
makedirs(name='plots/', exist_ok=True)

# Intialise the random number generator with a seed
seed(1)

# Define how many of the pictures are test images
test_share = 0.2

# Copy the original images based on the beginning part of the file name
# Decide the copying to train and test directories according to the random number
src_directory = 'original_data_grayscale/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < test_share:
        dst_dir = 'test/'
    if file.startswith('Zero'):
        destination = dataset_home + dst_dir + '0/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('One'):
        destination = dataset_home + dst_dir + '1/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Two'):
        destination = dataset_home + dst_dir + '2/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Three'):
        destination = dataset_home + dst_dir + '3/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Four'):
        destination = dataset_home + dst_dir + '4/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Five'):
        destination = dataset_home + dst_dir + '5/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Six'):
        destination = dataset_home + dst_dir + '6/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Seven'):
        destination = dataset_home + dst_dir + '7/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Eight'):
        destination = dataset_home + dst_dir + '8/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('Nine'):
        destination = dataset_home + dst_dir + '9/' + file
        copyfile(src, destination)
        print("Created: " + destination)

        