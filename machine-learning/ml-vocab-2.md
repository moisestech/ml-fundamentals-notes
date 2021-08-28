# ML Vocab 2

- [Google Doc](https://docs.google.com/document/d/1ElkOp8sU7_NlBJgQ6UQmlMnJNVr2RcgYL_fhVjOQCTY/edit#)
- [Deep Learnng for Vision Systems, Glossary](https://livebook.manning.com/book/deep-learning-for-vision-systems), **Manning**,

---

## $

<b>1 × 1 convolutional layer</b>

- The idea of the 1 × 1 convolutional layer is that it preserves the spatial dimensions (height and width) of the input volume but changes the number of channels of the volume (depth)

---

## A

<b>AAVER (adaptive attention for vehicle re-identification)</b>

- Adaptive attention for vehicle re-identification (AAVER ) by Khorramshahi et al. [25] is a recent work wherein the authors construct a dual-path network for extracting global and local features. These are then concatenated to form a final embedding. The proposed embedding loss is minimized using identity and keypoint orientation annotations.

<b>acc value</b>

- **loss** and **acc** are the error and accuracy values for the training data. val_loss and val_acc are the error and accuracy values for the validation data.
- Look at the `val_loss` and `val_acc` values after each epoch. Ideally, we want val_loss to be decreasing and val_acc to be increasing, indicating that the network is actually learning after each epoch.

<b>accuracy</b>

- A recent work in vehicle re-identification, AAVER [25] boosts mAP accuracy by 5% by post-processing using re-ranking. This means the accuracy metric is not suitable to measure the “goodness” of this model. We need other evaluation metrics that measure different aspects of the model’s prediction ability.

### <b>as metric for evaluating models</b>

- A recent work in vehicle re-identification, AAVER [25] boosts mAP accuracy by 5% by post-processing using re-ranking.

- <b>improvements to</b>

  - More training epochs
    - Notice that the network was improving until epoch 123. You can increase the number of epochs to 150 or 200 and let the network train longer.
  - Deeper network
    - Try adding more layers to increase the model complexity, which increases the learning capacity.
  - Lower learning rate
    - Decrease the lr (you should train longer if you do so).
  - Different CNN architecture
    - Try something like Inception or ResNet (explained in detail in the next chapter). You can get up to 95% accuracy with the ResNet neural network after 200 epochs of training.
  - Transfer learning
    - In chapter 6, we will explore the technique of using a pretrained network on your dataset to get higher results with a fraction of the learning time.

- <b>of image classification</b>

  - To set the stage for other metrics, we will use a confusion matrix : a table that describes the performance of a classification model.

- <b>building model architecture</b>

  - The optimal learning rate will be dependent on the topology of your loss landscape, which in turn is dependent on both your model architecture and your dataset.
  - Build the model architecture. In addition to regular convolutional and pooling layers, as in chapter 3, we add the following layers to our architecture:
    - Deeper neural network to increase learning capacity
    - Dropout layers
    - L2 regularization to our convolutional layers
    - Batch normalization layers

- <b>evaluating models</b>

  - When evaluating model performance, the goal is to categorize the high-level problem. If it’s a data problem, spend more time on data preprocessing or collecting more data. If it’s a learning algorithm problem, try to tune the network.

- <b>importing dependencies</b>

  1. Keras library to download the datasets, preprocess images, and network components
  2. Imports numpy for math operations
  3. Imports the matplotlib library to visualize results

- <b>preparing data for training</b>

  1. Downloads and splits the data
  2. Breaks the training set into training and validation sets
  3. Let’s print the shape of `x_train`, `x_valid`, and `x_test`
  4. Normalizing the pixel values of our images is done by subtracting the mean from each pixel and then dividing the result by the standard deviation.
  5. To one-hot encode the labels in the train, valid, and test datasets, we use the `to_` `categorical` function in Keras.
  6. For augmentation techniques, we will arbitrarily go with the following transformations: rotation, width and height shift, and horizontal flip.

- <b>training models</b>

  - Now that you have selected the metrics you will use to evaluate your system, it is time to establish a reasonable end-to-end system for training your model. Depending on the problem you are solving, you need to design the baseline to suit your network type and architecture. In this step, you will want to answer questions like these:

    - Should I use an MLP or CNN network (or RNN, explained later in the book)?
    - Should I use other object detection techniques like YOLO or SSD (explained in later chapters)?
    - How deep should my network be?
    - Which activation type will I use?
    - What kind of optimizer do I use?
    - Do I need to add any other regularization layers like dropout or batch normalization to avoid overfitting?

### <b>activation functions</b>

- Certain activation functions, like the sigmoid function, squish a large input space into a small input space between 0 and 1 (-1 to 1 for tanh activations). Therefore, a large change in the input of the sigmoid function causes a small change in the output. As a result, the derivative becomes very small.

- <b>binary classifier</b>
- <b>heaviside step function</b>
- <b>leaky ReLU</b>
- <b>linear transfer function</b>
- <b>logistic function</b>
- <b>ReLU</b>
- <b>sigmoid function</b>
- <b>softmax function</b>
- <b>tanh</b>

<b>activation maps</b>
<b>activation type</b>
<b>Adam (adaptive moment estimation)</b>
<b>Adam optimizer</b>
<b>adaptive learning</b>
<b>adversarial training</b>
<b>AGI (artificial general intelligence)</b>
<b>AI vision systems</b>
<b>AlexNet</b>

### <b>architecture of</b>

- <b>data augmentation</b>
- <b>dropout layers</b>
- <b>features of</b>
- <b>in Keras</b>
- <b>learning hyperparameters in</b>
- <b>local response normalization</b>
- <b>performance</b>
- <b>ReLu activation function</b>
- <b>training on multiple GPUs</b>
- <b>weight regularization</b>
- <b>classifier learning algorithms</b>
- <b>in DeepDream</b>

<b>alpha</b>
<b>AMI (Amazon Machine Images)</b>
<b>Anaconda</b>
<b>anchor boxes</b>
<b>AP (average precision)</b>
<b>artificial neural networks (ANNs)</b>
<b>atrous convolutions</b>
<b>attention network</b>

### <b>AUC (area under the curve)<b>

- <b>data</b>
- <b>for image classification</b>
- <b>in AlexNet</b>
- <b>images</b>

### <b>average pooling<b>

- <b>creating AWS account</b>
- <b>Jupyter notebooks</b>
- <b>remotely connect to instance</b>
- <b>setting up</b>

---

## B

<b>background region</b>
<b>backpropagation</b>
<b>backward pass</b>

### <b>base networks<b>

- <b>predicting with</b>
- <b>to extract features</b>

<b>baseline models</b>
<b>base_model summary</b>

### <b>batch gradient descent (BGD)</b>

- <b>derivative</b>
- <b>direction</b>
- <b>gradient</b>
- <b>learning rate</b>
- <b>pitfalls of</b>
- <b>step size</b>

<b>batch hard (BH)</b>

### <b>batch normalization</b>

- <b>defined</b>
- <b>in neural networks</b>
- <b>in Keras</b>
- <b>overview</b>

<b>batch normalization (BN)</b>
<b>batch sample (BS)</b>
<b>batch weighted (BW)</b>
<b>batch_size hyperparameter</b>
<b>Bayes error rate</b>
<b>biases</b>
<b>BIER (boosting independent embeddings robustly)</b>
<b>binary classifier</b>
<b>binary_crossentropy function</b>
<b>block1_conv1 layer</b>
<b>block3_conv2 layer</b>
<b>block5_conv2 layer</b>
<b>block5_conv3 layer</b>
<b>blocks.</b>
<b>bottleneck layers</b>
<b>bottleneck residual block</b>
<b>bottleneck_residual_block function</b>
<b>bounding box coordinates</b>

### <b>bounding box prediction</b>

- <b>in YOLOv3</b>
- <b>predicting with regressors</b>

<b>bounding-box regressors</b>
<b>build_discriminator function</b>
<b>build_model() function</b>

---

## C

<b>Cars Dataset, Stanford</b>
<b>categories</b>
<b>CCTV monitoring</b>
<b>cGAN (conditional GAN)</b>
<b>chain rule</b>
<b>channels value</b>

### <b>CIFAR dataset</b>

- <b>Inception performance on</b>
- <b>ResNet performance on</b>

<b>CIFAR-10 dataset</b>
<b>class predictions</b>
<b>classes</b>
<b>classes argument</b>
<b>Class_id label</b>
<b>classification</b>
<b>classification loss</b>
<b>classification module</b>
<b>classifier learning algorithms</b>

### <b>classifiers</b>

- <b>binary</b>
- <b>in Keras</b>
- <b>pretrained networks as</b>

### <b>CLVR (cross-level vehicle re-identification)</b>

- <b>adding dropout layers to avoid overfitting</b>
- <b>advantages of</b>
- <b>in CNN architecture</b>
- <b>overview of dropout layers</b>
- <b>overview of overfitting</b>
- <b>architecture of</b>
- <b>AlexNet</b>

### <b>classification</b>

- <b>feature extraction</b>
- <b>GoogLeNet</b>
- <b>Inception</b>
- <b>LeNet-5</b>
- <b>ResNet</b>
- <b>VGGNet</b>

<b>convolutional layers</b>

### <b>convolutional operations</b>

- <b>kernel size</b>
- <b>number of filters in</b>
- <b>overview of convolution</b>
- <b>padding</b>
- <b>strides</b>
- <b>design patterns</b>
- <b>fully connected layers</b>
- <b>image classification</b>
- <b>building model architecture</b>
- <b>number of parameters</b>
- <b>weights</b>
- <b>with color images</b>
- <b>with MLPs</b>
- <b>implementing feature visualizer</b>
- <b>overview</b>
- <b>pooling layers</b>

### <b>convolutional layers</b>

- <b>max pooling vs. average pooling</b>
- <b>subsampling</b>
- <b>visualizing features</b>

<b>coarse label</b>
<b>COCO datasets</b>
<b>collecting data</b>
<b>color channel</b>

### <b>converting to grayscale images</b>

- <b>image classification for</b>

### <b>compiling models</b>

- <b>defining model architecture</b>
- <b>evaluating models</b>
- <b>image preprocessing</b>
- <b>loading datasets</b>
- <b>loading models with val_acc</b>
- <b>training models</b>

<b>combined models</b>
<b>combined-image</b>
<b>compiling models</b>
<b>computation problem</b>
<b>computer vision.</b>
<b>conda list command</b>
<b>confidence threshold</b>
<b>confusion matrix</b>
<b>connection weights</b>
<b>content image</b>
<b>content loss</b>
<b>content_image</b>
<b>content_loss function</b>
<b>content_weight parameter</b>
<b>contrastive loss</b>
<b>CONV_1 layer</b>
<b>CONV1 layer</b>
<b>CONV_2 layer</b>
<b>CONV2 layer</b>
<b>CONV3 layer</b>
<b>CONV4 layer</b>
<b>CONV5 layer</b>

### <b>ConvNet weights</b>

- <b>overview</b>

<b>convolutional layers</b>

### <b>convolutional operations</b>

- <b>kernel size</b>
- <b>number of filters in</b>
- <b>padding</b>
- <b>strides</b>

<b>convolutional neural network</b>
<b>convolutional neural networks</b>
<b>convolutional operations</b>
<b>correct prediction</b>

### <b>cost functions</b>

- <b>defined</b>
- <b>in neural networks</b>

<b>cross-entropy</b>
<b>cross-entropy loss</b>
<b>cuDNN</b>

### <b>CV (computer vision)</b>

- <b>applications of</b>

### <b>creating images</b>

- <b>face recognition</b>
- <b>image classification</b>
- <b>image recommendation systems</b>
- <b>localization</b>
- <b>neural style transfer</b>
- <b>object detection</b>

### <b>classifier learning algorithms</b>

- <b>extracting features</b>
- <b>automatically extracted features</b>
- <b>handcrafted features</b>
- <b>advantages of</b>
- <b>overview</b>
- <b>image input</b>

<b>color images</b>

### <b>computer processing of images</b>

- <b>images as functions</b>
- <b>image preprocessing</b>
- <b>interpreting devices</b>
- <b>pipeline</b>
- <b>sensing devices</b>
- <b>vision systems</b>
- <b>AI vision systems</b>
- <b>human vision systems</b>
- <b>visual perception</b>

---

## D

### <b>Darknet-53</b>

- <b>augmenting</b>
- <b>for image classification</b>
- <b>in AlexNet</b>
- <b>collecting</b>
- <b>loading</b>
- <b>mining</b>
- <b>BH</b>
- <b>BS</b>
- <b>BW</b>

### <b>dataloader</b>

- <b>finding useful triplets</b>
- <b>normalizing</b>
- <b>preparing for training</b>
- <b>preprocessing</b>
- <b>augmenting images</b>
- <b>grayscaling images</b>
- <b>resizing images</b>
- <b>splitting</b>

<b>data distillation</b>
<b>DataGenerator objects</b>
<b>dataloader</b>

### <b>downloading to GANs</b>

<b>Kaggle</b>
<b>loading</b>
<b>MNIST</b>
<b>splitting for training</b>
<b>splitting for validation</b>
<b>validation datasets</b>

<b>DCGANs (deep convolutional generative adversarial networks)</b>
<b>deep neural network</b>

### <b>DeepDream</b>

<b>algorithms in</b>
<b>in Keras</b>

<b>deltas<b>

### <b>dendrites<b>

<b>See also<b>

<b>Dense_1 layer<b>
<b>Dense_2 layer<b>
<b>dependencies, importing<b>
<b>deprocess_image(x)<b>

### <b>design patterns<b>

- <b>measuring speed of<b>
- <b>multi-stage vs. single-stage<b>
- <b>overfitting<b>
- <b>underfitting<b>

<b>dilated convolutions<b>
<b>dilation rate<b>

### <b>dimensionality reduction with Inception<b>

- <b>1 × 1 convolutional layer<b>
- <b>impact on network performance<b>

<b>direction<b>
<b>discriminator<b>

### <b>discriminator_model method</b>

- <b>in GANs</b>
- <b>training</b>
- <b>conda environment</b>
- <b>loading environments</b>
- <b>anual development environments</b>
- <b>saving environments</b>
- <b>setting up</b>

<b>dropout hyperparameter</b>

### <b>dropout layers</b>

- <b>adding to avoid overfitting</b>
- <b>advantages of</b>
- <b>in AlexNet</b>
- <b>in CNN architecture</b>
- <b>overview</b>

<b>dropout rate</b>
<b>dropout regularization</b>

---

## E

<b>early stopping</b>
<b>EC2 Management Console</b>
<b>EC2 On-Demand Pricing page</b>
<b>edges</b>
<b>embedding networks, training</b>
<b>finding similar items</b>
<b>implementation</b>
<b>object re-identification</b>
<b>testing trained models</b>
<b>object re-identification</b>
<b>retrievals</b>
<b>embedding space</b>
<b>endAnaconda</b>
<b>conda</b>
<b>developing manually</b>
<b>loading</b>
<b>saving</b>
<b>epochs</b>
<b>number of</b>
<b>training</b>
<b>error functions</b>
<b>advantages of</b>
<b>cross-entropy</b>
<b>errors</b>
<b>mean squared error</b>
<b>overview</b>
<b>weights</b>
<b>errors</b>
<b>evaluate() method</b>
<b>evaluation schemes</b>
<b>Evaluator class</b>
<b>exhaustive search algorithm</b>
<b>exploding gradients</b>
<b>exponential decay</b>

---

## F

<b>f argument</b>
<b>face identification</b>
<b>face recognition (FR)</b>
<b>face verification</b>
<b>false negatives (FN)</b>
<b>false positives (FP)</b>
<b>False setting</b>
<b>Fashion-MNIST</b>
<b>fashion_mnist.load_data() method</b>
<b>Fast R-CNNs (region-based convolutional neural networks)</b>
<b>architecture of</b>
<b>disadvantages of</b>
<b>multi-task loss function in</b>
<b>architecture of</b>
<b>base network to extract features</b>
<b>fully connected layers</b>
<b>multi-task loss function</b>
<b>object detection with</b>
<b>RPNs</b>
<b>anchor boxes</b>
<b>predicting bounding box with regressor</b>
<b>training</b>
<b>FC layer</b>
<b>FCNs (fully convolutional networks)</b>
<b>feature extraction</b>
<b>automatically</b>
<b>handcrafted features</b>
<b>feature extractors</b>
<b>feature maps</b>
<b>feature vector</b>
<b>feature visualizer</b>
<b>feature_layers</b>
<b>advantages of</b>
<b>handcrafted</b>
<b>learning</b>
<b>overview</b>
<b>transferring</b>
<b>visualizing</b>
<b>feedforward process</b>
<b>learning features</b>
<b>FID (Fréchet inception distance)</b>
<b>filter hyperparameter<b>
<b>filter_index</b>
<b>filters</b>
<b>filters argument</b>
<b>fine label</b>
<b>fine-tuning</b>
<b>advantages of</b>
<b>learning rates when</b>
<b>transfer learning</b>
<b>.fit() method</b>
<b>fit_generator() function</b>
<b>Flatten layer</b>
<b>flattened vector</b>
<b>FLOPs (floating-point operations per second)</b>
<b>flow_from_directory() method</b>
<b>foreground region</b>
<b>FPS (frames per second)</b>
<b>freezing layers</b>
<b>F-score</b>
<b>fully connected layers</b>
<b>images as</b>
<b>training</b>

---

## G

<b>gallery set</b>
<b>GANs (generative adversarial networks)</b>
<b>applications for</b>
<b>image-to-image translation</b>
<b>Pix2Pix GAN</b>
<b>SRGAN</b>
<b>architecture of</b>
<b>DCGANs</b>
<b>generator models</b>
<b>minimax function</b>
<b>building</b>
<b>combined models</b>
<b>discriminators</b>
<b>downloading datasets</b>
<b>evaluating models of</b>
<b>choosing evaluation scheme</b>
<b>FID</b>
<b>inception score</b>
<b>generators</b>
<b>importing libraries</b>
<b>training</b>
<b>discriminators</b>
<b>epochs</b>
<b>generators</b>
<b>training functions</b>
<b>visualizing datasets</b>
<b>generative models</b>
<b>generator models</b>
<b>generator_model function</b>
<b>generators</b>
<b>in GANs</b>
<b>training</b>
<b>global average pooling</b>
<b>global minima</b>
<b>Google Open Images</b>
<b>GoogLeNet</b>
<b>architecture of</b>
<b>in Keras</b>
<b>building classifiers</b>
<b>building inception modules</b>
<b>building max-pooling layers</b>
<b>building network</b>
<b>learning hyperparameters in</b>
<b>GPUs (graphics processing units)</b>
<b>gradient ascent</b>
<b>gradient descent (GD)</b>
<b>overview</b>
<b>with momentum</b>
<b>gradients function</b>
<b>gram matrix</b>
<b>graph transformer network</b>
<b>converting color images</b>
<b>images</b>
<b>ground truth bounding box</b>
<b>GSTE (group-sensitive triplet embedding)</b>

---

## H

<b>hard data mining</b>
<b>hard negative sample</b>
<b>hard positive sample</b>
<b>heaviside step function</b>
<b>height value</b>
<b>hidden layers</b>
<b>hidden units</b>
<b>high-recall model</b>
<b>human in the loop</b>
<b>human vision systems</b>
<b>hyperbolic tangent function</b>
<b>in AlexNet</b>
<b>in GoogLeNet</b>
<b>in Inception</b>
<b>in LeNet-5</b>
<b>in ResNet</b>
<b>in VGGNet</b>
<b>neural network hyperparameters</b>
<b>parameters vs.</b>
<b>tuning</b>
<b>collecting data vs.</b>
<b>neural network hyperparameters</b>

<b>parameters vs. hyperparameters</b>

- **hyperparameters** are variables that are not learned by the network.
  - They are set by the ML engineer before training the model and then tuned.
  - These are variables that define the network structure and determine how the network is trained.
  - Hyperparameter examples include **learning rate**, **batch size**, **number of epochs**, n**umber of hidden layers**

---

## I

identity function
if-else statements
image classification
for color images
compiling models
defining model architecture
evaluating models
image preprocessing
loading datasets
loading models with val_acc
training models
with CNNs
building model architecture
number of parameters
weights
with high accuracy
building model architecture
evaluating models
importing dependencies
preparing data for training
training models
with MLPs
drawbacks of
hidden layers
input layers
output layers
image classifier
image flattening
image preprocessing
image recommendation systems
ImageDataGenerator class
ImageNet
ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
images
as functions
augmenting
color images
computer processing of
creating
grayscaling
preprocessing
converting color to grayscale
one-hot encoding
preparing labels
splitting datasets for training
splitting datasets for validation
rescaling
resizing
image-to-image translation
Inception
architecture of
features of
learning hyperparameters in
modules
naive version
performance on CIFAR dataset
with dimensionality reduction
1 × 1 convolutional layer
impact on network performance
inception scores
inception_module function
include_top argument
input image
input layers
input vector
input_shape argument
instances
interpreting devices
IoU (intersection over union)

---

## J

Jaccard distance
joint training
Jupyter notebooks

---

## K

K object classes
Kaggle datasets
AlexNet in
batch normalization in
DeepDream in
GoogLeNet in
building classifiers
building inception modules
building max-pooling layers
building network
LeNet-5 in
ResNet in
keras.datasets
keras_ssd7.py file
kernel
kernel size
kernel_size hyperparameter

---

## L

L2 regularization
label smoothing
labeled data
labeled images
LabelImg application
labels
lambda parameter
lambda value
layer_name
layers
1 × 1 convolutional
dropout
adding to avoid overfitting
advantages of
in AlexNet
in CNN architecture
overview
fully connected
hidden
representing style features
Leaky ReLU
learning
adaptive
embedding
features
finding optimal learning rate
in AlexNet
in GoogLeNet
in Inception
in LeNet-5
in ResNet
in VGGNet
mini-batch size
See also
learning curves, plotting
batch gradient descent
decay
derivative and
optimal, finding
when fine-tuning
LeNet-5
architecture of
in Keras
learning hyperparameters in
on MNIST dataset
libraries in GANs
linear combination
linear datasets
linear decay
linear transfer function
load_data() method
load_dataset() method
data
datasets
environments
models
local minima
local response normalization
localization
localization module
locally connected layers
LocalResponseNorm layer
location loss
logistic function
content loss
runtime analysis of
total variance
visualizing
loss functions
contrastive loss
cross-entropy loss
naive implementation
loss value
lr variable
lr_schedule function

---

## M

MAC (multiplier-accumulator)
MAC operation
human brain vs.
with handcrafted features
main path
make_blobs
matrices
matrix multiplication
max pooling
max-pooling layers
mean absolute error (MAE)
mean average precision (mAP)
mean squared error (MSE)
Mechanical Turk crowdsourcing tool, Amazon
metrics
min_delta argument
mini-batch gradient descent (MB-GD)
mini-batch size
minimax function
mining data
BH
BS
BW
dataloader
finding useful triplets
mixed2 layer
mixed3 layer
mixed4 layer
mixed5 layer
MLPs (multilayer perceptrons)
architecture of
hidden layers
image classification with
drawbacks of
hidden layers
input layers
output layers
layers
nodes
MNIST (Modified National Institute of Standards and Technology) dataset
architecture of
building
compiling
configuring
designing
evaluating
building networks
diagnosing overfitting
diagnosing underfitting
evaluating networks
plotting learning curves
training networks
loading
choosing evaluation scheme
evaluating
FID
inception score
testing
object re-identification
retrievals
training
momentum, gradient descent with
monitor argument
MS COCO (Microsoft Common Objects in Context)
multi-scale detections
multi-scale feature layers
architecture of
multi-scale detections
multi-scale vehicle representation (MSVR)
multi-stage detectors
multi-task learning (MTL)
multi-task loss function

---

## N

naive implementation
naive representation
n-dimensional array
neg_pos_ratio
networks
architecture of
activation type
depth of neural networks
improving
width of neural networks
building
evaluating
improving
in Keras
measuring precision of
predictions
as classifiers
as feature extractors
to extract features
training
neural networks
activation functions
binary classifier
heaviside step function
leaky ReLU
linear transfer function
logistic function
ReLU
sigmoid function
softmax function
tanh
backpropagation
covariate shift in
depth of
error functions
advantages of
cross-entropy
errors
MSE
overview
weights
feedforward process
learning features
hyperparameters in
learning features
multilayer perceptrons
architecture of
hidden layers
layers
nodes
optimization
optimization algorithms
batch gradient descent
gradient descent
MB-GD
stochastic gradient descent
overview
perceptrons
learning logic of
neurons
overview
width of
neural style transfer
content loss
network training
style loss
gram matrix for measuring jointly activated feature maps
multiple layers for representing style features
total variance loss
neurons
new_model
NMS (non-maximum suppression)
no free lunch theorem
node values
nodes
noise loss
nonlinear datasets
nonlinearities
non-trainable params
normalizing data
nstaller/application.yaml file

---

## O

oad_weights() method
object detection
framework
network predictions
NMS
object-detector evaluation metrics
region proposals
with Fast R-CNNs
architecture of
disadvantages of
multi-task loss function in
with Faster R-CNNs
architecture of
base network to extract features
fully connected layers
multi-task loss function
RPNs
with R-CNNs
disadvantages of
limitations of
multi-stage detectors vs. single-stage detectors
training
with SSD
architecture of
base networks
multi-scale feature layers
NMS
training SSD networks
with YOLOv3
architecture of
overview
object re-identification
object-detector evaluation metrics
FPS to measure detection speed
IoU
mAP to measure network precision
PR CURVE
objectness score
octaves
offline training
one-hot encoding
online learning
Open Images Challenge
open source datasets
CIFAR
Fashion-MNIST
Google Open Images
ImageNet
Kaggle
MNIST
MS COCO
optimal weights
optimization
optimization algorithms
Adam (adaptive moment estimation)
batch gradient descent
derivative
direction
gradient
learning rate
pitfalls of
step size
early stopping
gradient descent
overview
with momentum
MB-GD
number of epochs
stochastic gradient descent
optimization value
optimized weights
optimizer
output layer
Output Shape columns
adding dropout layers to avoid
diagnosing
overview
regularization techniques to avoid
augmenting data
dropout layers
L2 regularization

---

## P

<b>padding</b>
<b>PAMTRI (pose aware multi-task learning)</b>
<b>calculating</b>
<b>hyperparameters vs.</b>
<b>non-trainable params</b>
<b>number of</b>
<b>overview</b>
<b>trainable params</b>
<b>non-trainable</b>
<b>trainable</b>
<b>PASCAL VOC-2012 dataset</b>
<b>Path-LSTM</b>
<b>patience variable</b>
<b>.pem file</b>
<b>perceptrons</b>
<b>learning logic of</b>
<b>neurons</b>
<b>overview</b>
<b>step activation function</b>
<b>weighted sum function</b>
<b>performance metrics</b>
<b>accuracy</b>
<b>confusion matrix</b>
<b>F-score</b>
<b>person re-identification</b>
<b>pip install</b>
<b>Pix2Pix GAN (generative adversarial network)</b>
<b>plot_generated_images() function</b>
<b>plotting learning curves</b>
<b>POOL layer</b>
<b>POOL_1 layer</b>
<b>POOL_2 layer</b>
<b>pooling layers</b>
<b>convolutional layers</b>
<b>max pooling vs. average pooling</b>
<b>PR CURVE (precision-recall curve)</b>
<b>precision</b>
<b>predictions</b>
<b>across different scales</b>
<b>bounding box with regressors</b>
<b>for networks</b>
<b>with base network</b>
<b>data</b>
<b>augmenting images</b>
<b>grayscaling images</b>
<b>normalizing data</b>
<b>resizing images</b>
<b>images</b>
<b>converting color images to grayscale images</b>
<b>one-hot encoding</b>
<b>preparing labels</b>
<b>splitting datasets for training</b>
<b>splitting datasets for validation</b>
<b>pretrained model</b>
<b>as classifiers</b>
<b>as feature extractors</b>
<b>priors</b>

---

## Q

<b>query sets</b>
<b>Quick, Draw! dataset, Google</b>

---

## R

<b>disadvantages of</b>
<b>limitations of</b>
<b>multi-stage detectors vs. single-stage detectors</b>
<b>object detection with</b>
<b>training</b>
<b>receptive field</b>
<b>reduce argument</b>
<b>reduce layer</b>
<b>reduce shortcut</b>
<b>region proposals</b>
<b>regions of interest (RoIs)</b>
<b>regression layer</b>
<b>regressors</b>
<b>regular shortcut</b>
<b>regularization techniques to avoid overfitting</b>
<b>augmenting data</b>
<b>dropout layers</b>
<b>L2 regularization</b>
<b>ReLU (rectified linear unit)</b>
<b>activation functions</b>
<b>leaky</b>
<b>rescaling images</b>
<b>residual blocks</b>
<b>residual module architecture</b>
<b>residual notation</b>
<b>resizing images</b>
<b>ResNet (Residual Neural Network)</b>
<b>features of</b>
<b>in Keras</b>
<b>learning hyperparameters in</b>
<b>performance on CIFAR dataset</b>
<b>residual blocks</b>
<b>results, observing</b>
<b>retrievals</b>
<b>RGB (Red Green Blue)</b>
<b>RoI extractor</b>
<b>RoI pooling layer</b>
<b>RoIs (regions of interest)</b>
<b>RPNs (region proposal networks)</b>
<b>anchor boxes</b>
<b>predicting bounding box with regressors</b>
<b>training</b>
<b>runtime analysis of losses</b>

---

## S

<b>s argument</g>
<b>save_interval</g>
<b>scalar</g>
<b>scalar multiplication</g>
<b>scales, predictions across</g>
<b>scipy.optimize.fmin_l_bfgs_b method</g>
<b>sensing devices</g>
<b>shortcut path</g>
<b>Siamese loss</g>
<b>sigmoid function</g>
<b>single class</g>
<b>single-stage detectors</g>
<b>skip connections</g>
<b>Softmax layer</g>
<b>source domain</g>
<b>spatial features</g>
<b>specificity</g>
<b>data</g>
<b>for training</g>
<b>for validation</g>
<b>SRGAN (super-resolution generative adversarial networks)</g>
<b>architecture of</g>
<b>base network</g>
<b>multi-scale feature layers</g>
<b>architecture of multi-scale layers</g>
<b>multi-scale detections</g>
<b>non-maximum suppression</g>
<b>object detection with</g>
<b>training networks</g>
<b>building models</g>
<b>configuring models</g>
<b>creating models</b>
<b>loading data</b>
<b>making predictions</b>
<b>training models</b>
<b>visualizing loss</b>
<b>SSDLoss function</b>
<b>ssh command</b>
<b>StackGAN (stacked generative adversarial network)</b>
<b>step activation function</b>
<b>step function</b>
<b>step functions.</b>
<b>step size</b>
<b>stochastic gradient descent (SGD)</b>
<b>strides</b>
<b>style loss</b>
<b>gram matrix for measuring jointly activated feature maps</b>
<b>multiple layers for representing style features</b>
<b>style_loss function</b>
<b>style_weight parameter</b>
<b>subsampling</b>
<b>supervised learning</b>
<b>suppression.</b>
<b>synapses</b>
<b>synset (synonym set)</b>

---

## T

<b>tanh (hyperbolic tangent function)</b>
<b>tanh activation function</b>
<b>Tensorflow playground</b>
<b>tensors</b>
<b>testing trained model</b>
<b>object re-identification</b>
<b>retrievals</b>
<b>test_path variable</b>
<b>test_targets</b>
<b>test_tensors</b>
<b>TN (true negatives)</b>
<b>to_categorical function</b>
<b>top-1 error rate</b>
<b>top-5 error rate</b>
<b>top-k accuracy</b>
<b>total variance loss</b>
<b>total variation loss</b>
<b>total_loss function</b>
<b>total_variation_loss function</b>
<b>total_variation_weight parameter</b>
<b>TP (true positives)</b>
<b>train() function</b>
<b>trainable params</b>
<b>train_acc value</b>
<b>AlexNet</b>
<b>by trial and error</b>
<b>discriminators</b>
<b>embedding networks</b>
<b>finding similar items</b>
<b>implementation</b>
<b>object re-identification</b>
<b>testing trained models</b>
<b>epochs</b>
<b>functions</b>
<b>GANs</b>
<b>generators</b>
<b>models</b>
<b>networks</b>
<b>preparing data for</b>
<b>augmenting data</b>
<b>normalizing data</b>
<b>one-hot encode labels</b>
<b>preprocessing data</b>
<b>splitting data</b>
<b>R-CNNs</b>
<b>RPNs</b>
<b>splitting datasets for</b>
<b>SSD networks</b>
<b>building models</b>
<b>configuring models</b>
<b>creating models</b>
<b>loading data</b>
<b>making predictions</b>
<b>training models</b>
<b>visualizing loss</b>
<b>train_loss value</b>
<b>train_on_batch method</b>
<b>train_path variable</b>
<b>transfer functions</b>
<b>in GANs</b>
<b>linear</b>
<b>transfer learning</b>
<b>approaches to</b>
<b>using pretrained network as classifier</b>
<b>using pretrained network as feature extractor</b>
<b>choosing level of</b>
<b>when target dataset is large and different from source dataset</b>
<b>when target dataset is large and similar to source dataset</b>
<b>when target dataset is small and different from source</b>
<b>when target dataset is small and similar to source dataset</b>
<b>fine-tuning</b>
<b>open source datasets</b>
<b>CIFAR</b>
<b>Fashion-MNIST</b>
<b>Google Open Images</b>
<b>ImageNet</b>
<b>Kaggle</b>
<b>MNIST</b>
<b>MS COCO</b>
<b>overview</b>
<b>neural networks learning features</b>
<b>transferring features</b>
<b>pretrained networks as feature extractors</b>
<b>when to use</b>
<b>transferring features</b>
<b>transposition</b>
<b>triplets, finding</b>
<b>tuning hyperparameters</b>
<b>collecting data vs.</b>
<b>neural network hyperparameters</b>
<b>parameters vs. hyperparameters</b>

---

## U

<b>underfitting</b>
<b>untrained layers</b>
<b>Upsampling layer</b>
<b>Upsampling2D layer</b>

---

## V

<b>val_acc</b>
<b>val_acc value</b>
<b>val_error value</b>
<b>overview</b>
<b>splitting</b>
<b>valid_path variable</b>
<b>val_loss value</b>
<b>VAMI (viewpoint attentive multi-view inference)</b>
<b>vanishing gradients</b>
<b>vector space</b>
<b>VeRi dataset</b>
<b>VGG16 configuration</b>
<b>VGG19 configuration</b>
<b>VGGNet (Visual Geometry Group at Oxford University)</b>
<b>configurations</b>
<b>features of</b>
<b>learning hyperparameters in</b>
<b>performance</b>
<b>vision systems</b>
<b>AI</b>
<b>human</b>
<b>visual embedding layer</b>
<b>visual embeddings</b>
<b>face recognition</b>
<b>image recommendation systems</b>
<b>learning embedding</b>
<b>loss functions</b>
<b>contrastive loss</b>
<b>cross-entropy loss</b>
<b>naive implementation</b>
<b>runtime analysis of losses</b>
<b>mining informative data</b>
<b>BH</b>
<b>BS</b>
<b>BW</b>
<b>dataloader</b>
<b>finding useful triplets</b>
<b>training embedding networks</b>
<b>finding similar items</b>
<b>implementation</b>
<b>object re-identification</b>
<b>testing trained models</b>
<b>visual perception</b>
<b>datasets</b>
<b>features</b>
<b>loss</b>
<b>VUIs (voice user interfaces)</b>

---

## W

<b>warm-up learning rate</b>
<b>weight connections</b>
<b>weight decay</b>
<b>weight layers</b>
<b>weight regularization</b>
<b>weighted sum</b>
<b>weighted sum function</b>
<b>weights</b>
<b>calculating parameters</b>
<b>non-trainable params</b>
<b>trainable params</b>
<b>weights vector</b>
<b>width value</b>

---

## X

<b>X argument</b>
<b>x_test</b>
<b>x_train</b>
<b>x_valid</b>
<b>Y</b>
<b>architecture of</b>
<b>object detection with</b>
<b>overview</b>
<b>output bounding boxes</b>
<b>predictions across different scales</b>

---

## Z

<b>zero-padding</b>

---

<link rel="stylesheet" type="text/css" media="all" href="../assets/css/custom.css" />

## Foam Related Links

- [[_ml]]
