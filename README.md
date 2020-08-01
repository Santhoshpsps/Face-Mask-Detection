# Face-Mask-Detection
COVID-19 face mask detector with OpenCV and Keras/TensorFlow.

In this COVID-19 Pandemic, almost every one of us tend to wear a face mask. It becomes increasingly necessary to check if the people in the crowd wear face masks in most public gatherings such as Malls, Theatres, Parks. The development of an AI solution to detect if the person is wearing a face mask and allow their entry would be of great help to the society. In this paper, a simple Face Mask detection system is built using the Deep Learning technique called as Convolution Neural Networks (CNN). This CNN Model is built using the TensorFlow framework and the OpenCV library which is highly used for real-time applications. This Model doesnot use Haar Feature-based Cascade Classifier for face detection.

Using this model, an accuracy of over 98.79% is obtained. This can also be used further to achieve even higher levels of accuracy.

Data -
I have used this face mask dataset :https://github.com/alokproc/face-mask-detector/tree/master/resources/dataset

CNN Architecture -
In this proposed method, the Face Mask detection model is built using the Sequential API of the keras library. This allows us to create the new layers for our model step by step. The various layers used for our CNN model is described below.

The first layer is the Conv2D layer  and the filter size or the kernel size of 3X3. In this first step, the activation function used is the ‘ReLu’. This ReLu function stands for Rectified Linear Unit which will output the input directly if is positive, otherwise, it will output zero. T

In the second layer, the MaxPooling2D is used with the pool size of 2X2.

The next layer is again a Conv2D layer  of the same filter size 3X3 and the activation function used is the ‘ReLu’. This Conv2D layer is followed by a MaxPooling=2D layer with pool size 2X2.

Similarly 3 more such layers are added.

In the next step, we use the Flatten() layer to flatten all the layers into a single 1D layer.

After the Flatten layer, we use the Dropout (0.5) layer to prevent the model from overfitting.

Finally, towards the end, we use the Dense layer with 50 units and the activation function as ‘ReLu’.

The last layer of our model will be another Dense Layer, with only two units and the activation function used will be the ‘sigmoid’ function. T

After building the model, we compile the model and define the loss function and optimizer function. In this model, we use the ‘binary_crossentropy’ as the Loss function for training purpose.

Finally, the CNN model is trained for 15 epochs with two classes, one denoting the class of images with the face masks and the other without face masks.

Loss Accuracy Graph:
<p align="center">
<img src="https://github.com/Santhoshpsps/Face-Mask-Detection/blob/master/loss.PNG" height="400" width="500">
 </p>
 
 'with mask'is denoted as 0 and 'without mask'is denoted as  1
 
 Results:
 
 Detected (with mask):
 <p align="center">
<img src="https://github.com/Santhoshpsps/Face-Mask-Detection/blob/master/results_yes.PNG" height="400" width="500">
 </p>
 
 Detected(without mask):
 <p align="center">
<img src="https://github.com/Santhoshpsps/Face-Mask-Detection/blob/master/results_no.PNG" height="400" width="500">
 </p>
 
 

