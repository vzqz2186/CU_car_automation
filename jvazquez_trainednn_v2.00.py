"""
        Title: jvazquez_trainednn_test_v2.00.py

       Author: Daniel Vazquez
     
         Date: 03/11/2020 (v1.00)  * DOES NOT WORK *
               03/18/2020 (v2.00)  *     WORKS     *

  Description: Testing of trained convolutional neural network with test 
               imgs. Training of the neural network was done using the
               CIFAR-10 dataset. This program aims to predict the content of
               test imgs downloaded from pexels.com.

# -------------------------------------------------------------------------- 80

    Algorithm:
    
        1. Load trained model.
        2. Define class names.
        3. Prepare test image.
        4. Predict class.

# -------------------------------------------------------------------------- 80

   References:
   
     - Save and load models
    
       https://jovianlin.io/saving-loading-keras-models/
   
     - How to test cnn

       https://pythonistaplanet.com/cifar-10-img-classification-using-keras/
       
     - img.img_to_array

       https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/img/img_to_array
       
     - np.expand_dims
     
       https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
       
     - Predict method
     
       https://keras.io/models/sequential/#predict
       
     - Test imgs were obtained from

       https://www.pexels.com/

# -------------------------------------------------------------------------- 80

       System:
       
            CPU: Intel Core i5-7600k @ 3.80Ghz
            RAM: 16 GB
            GPU: NVIDIA GeForce GTX 1070
             OS: Winndows 10 Pro

# -------------------------------------------------------------------------- 80

Sample Output:

    CNN architecture loaded
    CNN weights loaded
    Input filename to test:
    - test_img_car.jpeg
    Image to test printed with matplotlib.
    Image contains a car

# -------------------------------------------------------------------------- 80
     Known bugs: 
      
        - None.

# -------------------------------------------------------------------------- 80

   Requirements: The following packages need to be installed for this program
                 to run correctly.
                 
        - pillow (install with pip)
        - tensorflow (install with pip)
        - numpy (install with pip)
        - matplotlib (install with pip)

# -------------------------------------------------------------------------- 80
     
          Notes: 
                
    - v1.00: Model that produced 82% accuracy  was loaded and tested with
             imgs from the validation batch. It was not as accurate as
             desired.
    
    - v2.00: Program is able to identify items related to the class names from
             .jpeg files. One image was tested per class and the cnn was able
             to classify 9 out of 10 correctly, thinking that a deer is a frog.
    
# -------------------------------------------------------------------------- 80
"""

# Step 1: Import libraries.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function definition
def main():

    # Step 2: Load trained model.

    # Load trained neural network architecture.
    with open('jvazquez_cnn_architecture_v3.02.2.h5', 'r') as f:
        trained_model = model_from_json(f.read())
    
    print('CNN architecture loaded')
        
    # Load weights.
    trained_model.load_weights('jvazquez_cnn_weights_v3.02.2.h5')
    
    print('CNN weights loaded')

    # Show trained neural network structure.
    # trained_model.summary()

# -------------------------------------------------------------------------- 80
   
    # Step 3: Dafine class names.    
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck']
    
# -------------------------------------------------------------------------- 80

    # Step 4: Load test image.
    
    print('Input filename to test: ')
    test_img = input('- ')
    test_img = load_img(test_img, target_size = (32,32))

    # Show selected image.
    print('Image to test printed with matplotlib.')
    plt.imshow(test_img)
    plt.show()
    
# -------------------------------------------------------------------------- 80
    
    # Step 5: Prepare test imgs.
    
    # Converting imgs to numpy array.
    test_img = img_to_array(test_img)
    test_img = test_img / 255.0
    test_img = test_img.reshape(1, 32, 32, 3)
    
    # Expands shape of array.
    # test_img = np.expand_dims(test_img, axis = 0)
    
# -------------------------------------------------------------------------- 80
    
    # Step 6: Predict classes.
    
    # Predict test_img class.
    result = trained_model.predict(test_img)
    
    # Check for index of highest value in result array.
    result_max = np.argmax(result)
    
    # Print predicted classes.
    print('Image contains a', class_names[result_max])

# -------------------------------------------------------------------------- 80

# Function call.
main()
