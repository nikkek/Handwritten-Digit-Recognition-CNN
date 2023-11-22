"""
Perform digit recognition on images using a pre-trained model.

This script utilizes a pre-trained model ('models/basemodel_acc_95.h5') for digit recognition.
It reads unidentified images from the 'unidentified_images/' directory,
predicts the digits they represent, and saves the recognized images with predictions in the 'identified_images/' directory.
"""

import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from pathlib import Path

# Define paths
unidentified_images = 'unidentified_images/'
identified_images = 'identified_images/'

# Create directory for the identified/recognised images
os.makedirs(name=identified_images, exist_ok=True)

# Load the trained model
model = load_model('models/basemodel_acc_95.h5')

#classification = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
classification = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Counter for the recognitions that were right
correct_count = 0;
# Count of files in the directory
file_count = 0;
# Variables for each of the numbers. These are used to see which numbers this model recognizes the best and which numbers the worst
zero = 0;
one = 0;
two = 0;
three = 0;
four = 0;
five = 0;
six = 0;
seven = 0;
eight = 0;
nine = 0;

for file in os.listdir(unidentified_images):
   print('File: ' + file)
   # Add 1 to file_count
   file_count += 1
   # Make predictions for image if it is '*.jpg'
   if Path(unidentified_images + file).suffix == '.jpg':
       # Load the image for prediction
       new_image = plt.imread(unidentified_images + file)
       # Resize the image, it is also scaled as 1/255.0 just like with training
       resized_image = resize(new_image, (140, 90, 3))
       # Predict image class as array of propabilities
       prediction = model.predict(np.array([resized_image])) #[0][0]
       
       list_index = [0,1,2,3,4,5,6,7,8,9]
       x = prediction
       
       for i in range(10):
           for j in range(10):
               if x[0][list_index[i]] > x[0][list_index[j]]:
                   temp = list_index[i]
                   list_index[i] = list_index[j]
                   list_index[j] = temp

       print(list_index)
       
       
       
       for i in range(1):
           print('Prediction: ' + classification[list_index[i]])
           predicted_class = classification[list_index[i]]        
       if predicted_class == file[0]:
           correct_count += 1
           # How many times each number is predicted right
           if predicted_class == '0':
               zero += 1
           elif predicted_class == '1':
               one += 1
           elif predicted_class == '2':
               two += 1
           elif predicted_class == '3':
               three += 1
           elif predicted_class == '4':
               four += 1
           elif predicted_class == '5':
               five += 1
           elif predicted_class == '6':
               six += 1
           elif predicted_class == '7':
               seven += 1
           elif predicted_class == '8':
               eight += 1
           elif predicted_class == '9':
               nine += 1
    
       # Save the original (size unchanged) as image to be saved
       img = plt.imshow(new_image)
       # Add the predicted class as the title with prediction
       plt.title('Prediction: ' + predicted_class)
       # Clean the plot and save it
       plt.axis('off')
       plt.savefig(identified_images + file.split('.')[0] + '-identified.jpg', bbox_inches='tight', pad_inches=0)
       # Close the plot before next image
       plt.close()


print('')
print('Accuracy:')
# Correct count out of count of unidentified pictures in the directory
print(correct_count/file_count)
print('')

print('zero: ' + str(zero/10))
print('one: ' + str(one/10))
print('two: ' + str(two/10))
print('three: ' + str(three/10))
print('four: ' + str(four/10))
print('five: ' + str(five/10))
print('six: ' + str(six/10))
print('seven: ' + str(seven/10))
print('eight: ' + str(eight/10))
print('nine: ' + str(nine/10))
