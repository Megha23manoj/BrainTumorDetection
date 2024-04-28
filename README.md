# **Brain Tumor Detection using Machine Learning**
 
 ### INTRODUCTION
 
 In this modern world it is prevalent that growth in medical techniques is influential in the saving
 of human lives. Due to this, researchers all over the world are scaled up to work in various
 aspects of the medical field for the greater benefit of mankind, in this technology and its
 advancement plays a major role. Among many disorders and abnormalities caused nowadays,
 tumor is believed to have an adverse effect. A Brain tumor is considered as one of the aggressive
 diseases, among children and adults. 
 
The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge
 amount of image data is generated through the scans. These images are examined by the
 radiologist. A manual examination can be error-prone due to the level of complexities involved
 in brain tumors and their properties. Application of automated classification techniques using
 Machine Learning (ML) and Artificial Intelligence (AI) has consistently shown higher accuracy
 than manual classification. 
 
We aim to detect brain tumors from a given brain MRI image using appropriate image
 processing techniques and machine learning algorithms. We train the model with positive and
 negative test MRI images of the brain, evaluate the model in order to ready itself to rightly
 classify a new MRI image.
 
 ### PROBLEM STATEMENT 
 
 While it is possible for an experienced doctor to correctly identify brain tumor tissues from MRI
 images, a system that classifies brain MRI images to ‘tumor detected’ and ‘tumor not detected’
 labels would significantly enhance the operations of most hospitals. After training a machine
 learning model with labelled datasets, we can deploy this model to perform a binary
 classification of new images to the predefined classes confirming the presence and absence of
 brain tumor.
 
 ### DATASET DESCRIPTION
 
 The dataset we have chosen here comprises of magnetic resonance imaging (MRI) images of the
 brain; the images are rightly labelled into tumor affected and not tumor affected. The dataset
 consists of 98 images of data labeled as ’NO’ and 155 images of data labelled as ’YES’. The
images used are in .JPG/.JPEG format. 80 percent of the images are taken as training set, while
 the 20 percent forms the testing set.
 
### IMAGE CLEANING AND PREPROCESSING

 The steps taken to format images before they are used by the model training and inference
 together forms Image Preprocessing. This includes, but is not limited to, resizing, orienting, and
 color corrections. Additionally, model preprocessing may shorten model training time and
 quicken model inference.

 During this preprocessing we used the technique of image contouring. The line connecting all the
 points along an image's edge that have the same intensity is referred to as the contour. Contours
 are useful for object detection, determining the size of an object of interest, and shape analysis. 
Here, the original is taken in which the outermost boundary is found and outlined to form the
 biggest contour. After the biggest contour is found, then the extreme points along the four axes
 are marked. This marking is used for the appropriate cropping of the image.
 The preprocessing steps are applied to all the images.

 ### DATA AUGMENTATION
 
 Data augmentation comprises of techniques used to increase the amount of data by adding
 slightly modified copies of already existing data or newly created synthetic data from existing
 data. It acts as a regularizer and helps reduce overfitting when training a machine learning
 model.  For example, for images we can use: Geometric transformations – you can randomly
 flip, crop, rotate or translate images, and that is just the tip of the iceberg. Color space
 transformations – change RGB color channels, intensify any color.

 ### CONVOLUTIONAL NEURAL NETWORK MODELS
 
 The CNN models we have used in this project are VGG-16 and ResNet50V2.
 A convolutional neural network, also known as a CNN, is a kind of artificial neural network. An
 input layer, an output layer, and many hidden layers make up a convolutional neural network.
 One of the top computer vision models to date is the CNN variant known as VGG16. This
 model's developers analyzed the networks and enhanced the depth using an architecture with
 incredibly tiny convolution filters, which demonstrated a notable advancement over the state-of
the-art setups. VGG-16 is very appealing because of its very uniform Architecture.

The other CNN model is ResNet50V2 which is 50 layers deep which comprises of 48
 convolution layers along with 1 Max Pool and 1 Average Pool layer. A residual neural network
 is an artificial neural network of a kind that stacks residual blocks on top of each other to form a
 network. We can load a pretrained version of the network trained on more than a million images
 from the ImageNet database. The pretrained network can classify images into 1000 object
 categories, such as keyboard, mouse, pencil, and many animals. The network has an image input
 size of 224-by-224.

### MODEL PERFORMANCE AND EVALUATION

 In the case of VGG16 model, we can see that the accuracy increases as the number of epoch
 increases, and finally attains a validation accuracy of 88% and test accuracy of 100%
 
![image](https://github.com/Megha23manoj/BrainTumorDetection/assets/7195913/316be6ff-4897-46f5-ad3b-0387f0d4ca32)
 
 Whereas when it comes to ResNet50, the performance fluctuates in order to reach a validation
 accuracy of 62% and test accuracy of 50%
 
 ![image](https://github.com/Megha23manoj/BrainTumorDetection/assets/7195913/dcf6a540-19a4-4391-a825-5c0a28987b00)


### CONCLUSION

 We used Convolutional Neural Network to predict whether the subject has Brain Tumor or not
 from MRI Images. Using large no. of images i.e., a larger dataset, Hyperparameter Tuning and
 Using a different Convolutional Neural Network Model may result in higher accuracy. Here, VGG16 model performs better.
