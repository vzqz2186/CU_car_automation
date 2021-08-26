"""
        Title: jvazquez_nncode_v3.02.2.py

       Author: Daniel Vazquez
     
         Date: 02/19/2020 (v1.00)    * DOES NOT WORK *
               02/20/2020 (v1.01)    * DOES NOT WORK *
               02/20/2020 (v2.00)    *   INCOMPLETE  *
               03/04/2020 (v3.00)    *     WORKS     *
               03/08/2020 (v3.01)    *     WORKS     *
               03/08/2020 (V3.02)    *     WORKS     *
               03/09/2020 (v3.03)    *     WORKS     *
               03/11/2020 (v3.02.1)  *     WORKS     *
               03/17/2020 (v3.02.2)  *     WORKS     *

  Description: Convolutional neural network using the CIFAR-10 dataset.

# -------------------------------------------------------------------------- 80

    Algorithm:
    
      1. Define commonly modified parameters.
      2. Import dataset.
      3. Prepare dataset.
      4. Define model.
      5. Compile model.
      6. Train model.
      7. Evaluate model.
      8. Save model.
      
# -------------------------------------------------------------------------- 80

   References:
   
     - CNN example with CIFAR-10 at tensorflow.org
       
       https://tensorflow.org/tutorials/images/cnn
       
     - CNN example on CIFAR-10 and how to save and test it.
       
       https://pythonistaplanet.com/cifar-10-image-classification-using-keras/

     - CNN example with MNIST digits dataset.
       
       https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f
     
     - keras.io resources:

       -- Dense layers
          https://keras.io/layers/core/#dense
       -- Convolutionnal layers
          https://keras.io/layers/convolutional/#conv2d
       -- Pooling layers
          https://keras.io/layers/pooling/#maxpooling2d
       -- Normalization
          https://keras.io/layers/normalization/
       -- Flatten layer
          https://keras.io/layers/core/#flatten
       -- Dropout layer
          https://keras.io/layers/core/#dropout
       -- Compiling, and training model
          https://keras.io/models/sequential/#compile
          https://keras.io/models/sequential/#fit
          
    - Padding
      
      https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
      
    - One hot encoding
    
      https://keras.io/utils/#to_categorical
      https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

# -------------------------------------------------------------------------- 80

       System:
       
            CPU: Intel Core i5-7600k @ 3.80Ghz
            RAM: 16 GB
            GPU: NVIDIA GeForce GTX 1070
             OS: Windows 10 Pro

# -------------------------------------------------------------------------- 80

Sample Output:

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 30, 30, 32)        896       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 30, 30, 32)        128       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        9248      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 12, 12, 64)        18496     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 12, 12, 64)        256       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 10, 10, 64)        36928     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 10, 10, 64)        256       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 3, 3, 128)         73856     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 3, 3, 128)         512       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 1, 1, 128)         147584    
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 1, 1, 128)         512       
    _________________________________________________________________
    flatten (Flatten)            (None, 128)               0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                1290      
    =================================================================
    Total params: 290,090
    Trainable params: 289,194
    Non-trainable params: 896
    _________________________________________________________________

    Train on 50000 samples, validate on 10000 samples
 
    Epoch 1/30
    50000/50000 [==============================] - 177s 4ms/sample - loss: 1.3271 - acc: 0.5297 - val_loss: 0.9894 - val_acc: 0.6522

    Epoch 2/30
    50000/50000 [==============================] - 176s 4ms/sample - loss: 0.9427 - acc: 0.6681 - val_loss: 1.0516 - val_acc: 0.6433

    Epoch 3/30
    50000/50000 [==============================] - 183s 4ms/sample - loss: 0.7920 - acc: 0.7215 - val_loss: 0.8697 - val_acc: 0.7087

    .
    .
    .

    Epoch 28/30
    50000/50000 [==============================] - 183s 4ms/sample - loss: 0.1917 - acc: 0.9321 - val_loss: 0.7132 - val_acc: 0.8070

    Epoch 29/30
    50000/50000 [==============================] - 187s 4ms/sample - loss: 0.1850 - acc: 0.9346 - val_loss: 0.7142 - val_acc: 0.8083

    Epoch 30/30
    50000/50000 [==============================] - 182s 4ms/sample - loss: 0.1880 - acc: 0.9336 - val_loss: 0.7076 - val_acc: 0.8089

    Test Loss:  0.7075541350841522
    Test accuracy: % 80.8899998664856
    Model saved

# -------------------------------------------------------------------------- 80
     Known bugs: 
      
         -- /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
  
        Possibly caused by a newer version of NumPy. Need to investigate.
        
        - v3.02.1: Message was indeed caused by a newer version of numpy.
                   Reverted to older version of numpy and the message dissapeared.
        
        -- Program keeps mentioning XLA:CPU is deactivated.
        
        - v3.02.1: Message is just saying XLA compiler is not being used.
                   XLA is just a compiler faster that Keras compiler.

# -------------------------------------------------------------------------- 80
     
          Notes: 
                
    - v3.00: Program based on the basic example in tensorflow's website
             (tensorflow.org/tutorials/images/cnn). It achieves 70% accuracy
             after around 10 epochs.
             
    - v3.01: 

    - v3.02: Program runs but accuracy is still lower than desired at 80%.  
             Need to fix overfitting issues to get a better accuracy.
             Normalization was added in the form of BatchNormalization layers.
             
    - v3.03: Introduced image preprocessing to combat overfitting issues.
             Accuracy has decreased significantly from 80% to aroud 25%. This
             couls be because the code got overcomplicated. Will return to
             previous version.
    
    - v3.02.1: Went back to the previous version since v3.03's accuracy became
             too low. Learning rate for the Adam optimizer was modified to
             control the amount of weights updated while the network is being
             trained. Included code to save the trained neural network and its
             weights to separate files.
             
    - v3.02.02: Added paddinng and an extra dense layer. The format the trained
             layer is saved got changed to only save a .h5 file with model and
             weights save in one single file. Moved the most changed variables
             to the top of the code for easy access (i.e. optimizer, learning
             rate, etc.). Added more dropouts to combat overfitting.
    
# -------------------------------------------------------------------------- 80
"""

# -------------------------------------------------------------------------- 80

# Step 1: Import necessary ibraries.
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# Function definition.
def main():

    # Step 2: Define commonly modified parameters.
    
    act = 'elu'              # Activation function.
    lr = 0.001               # Optimizer learning rate.
    epochs = 75              # Number of epochs the model will be put through.
    regularizer = l2(0.001)  # Regularizing function.
    opt = Adam(learning_rate = lr)
                             # Optimizer function. In order to cnange it,
                             # the new optimizer needs to be imported at
                             # line 219.
    loss = SparseCategoricalCrossentropy(from_logits = True)
                             # Loss function. In order to change it, the new
                             # function needs to be imported at line 217.
    padding = 'same'         # Padding is added to prevent pixel loss.

# -------------------------------------------------------------------------- 80

    # Step 3: Import CIFAR-10 dataset.
    (train_imgs, train_lbls), (val_imgs, val_lbls) = cifar10.load_data()
    
# -------------------------------------------------------------------------- 80

    # Step 4: Prepare dataset
    
    # Normalize dataset to values between 0 and 1.
    train_imgs, val_imgs = train_imgs / 255.0, val_imgs / 255.0
 
 # ------------------------------------------------------------------------- 80
 
    """
    Step 5: Define model.
    
    Structure of the neural network.
    
    > First Conv2D layer acts as the input layer, defining the input shape for
      the rest of the network.
    > Following layers act as the hidden layers. More Conv2D layers and
      MaxPooling layers compose a total of 5 hidden layers in this section.
    > Flatten layer turns the tensor obtained by the hidden layer into a flat
      vector that the final layer can use.
    > Final Dense layer with 10 nodes acts as the output layer of the network.
      One node for each class label of the images in the dataset.
      
    > Conv2D layers turns the layer's input into a tensor of outputs
      (keras.io/layers/convolutional/).
    > BatchNormalization layers maintain mean activation close to 0 and the
      activation standard deviation close to 1
      (keras.io/layers/normalization/).
    > Dropout layers set a fraction of neurons to 0 at each update to prevent
      overfitting issues (keras.io/layers/core/).
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = act, input_shape = (32,32,3)))
    model.add(BatchNormalization()) 
    
    model.add(Conv2D(32, (3, 3), activation = act,
                     kernel_regularizer = regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation = act))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation = act,
                     kernel_regularizer = regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation = act))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation = act,
                     kernel_regularizer = regularizer))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(10))
    
    # Display model's architecture.
    model.summary()
    
    # Following comment symbols are here to prevent the network form
    # compiling when the structure is the only thing wished to be seen.
    
    # """

# -------------------------------------------------------------------------- 80

    # Step 6: Compile model.
    model.compile(optimizer = opt,
                  loss = loss,
                  metrics = ['accuracy'])

# -------------------------------------------------------------------------- 80

    # Step 7: Train model.
    model.fit(train_imgs, train_lbls,
              epochs = epochs,
              verbose = 1,
              validation_data = (val_imgs, val_lbls))

# -------------------------------------------------------------------------- 80

    # Step 8: Evaluate model.
    (val_loss, val_acc) = model.evaluate(val_imgs, val_lbls, verbose = 1)
    
    # Display Test Accuracy and test loss.
    print('\nTest Loss: ',val_loss)
    print('Test accuracy: %',val_acc*100)

# -------------------------------------------------------------------------- 80

    # Step 9: Save model.
    
    # Saves arquitecture to .json file.
    with open('jvazquez_cnn_architecture_v3.02.2.json', 'w') as f:
        f.write(model.to_json())
        
    print("Architecture saved.")
        
    # Saves weights to .h5 file.
    model.save_weights('jvazquez_cnn_weights_v3.02.2.h5')
    
    print("Weights saved.")
    
    # Following comment symbols are here to prevent the network form
    # compiling when the structure is the only thing wished to be seen.
    
    # """
    
# -------------------------------------------------------------------------- 80

# Function call.
main()
