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

- <b>binary classifier<b>
- <b>heaviside step function<b>
- <b>leaky ReLU<b>
- <b>linear transfer function<b>
- <b>logistic function<b>
- <b>ReLU<b>
- <b>sigmoid function<b>
- <b>softmax function<b>
- <b>tanh<b>

<b>activation maps<b>
<b>activation type<b>
<b>Adam (adaptive moment estimation)<b>
<b>Adam optimizer<b>
<b>adaptive learning<b>
<b>adversarial training<b>
<b>AGI (artificial general intelligence)<b>
<b>AI vision systems<b>
<b>AlexNet<b>

### <b>architecture of<b>

- <b>data augmentation<b>
- <b>dropout layers<b>
- <b>features of<b>
- <b>in Keras<b>
- <b>learning hyperparameters in<b>
- <b>local response normalization<b>
- <b>performance<b>
- <b>ReLu activation function<b>
- <b>training on multiple GPUs<b>
- <b>weight regularization<b>
- <b>classifier learning algorithms<b>
- <b>in DeepDream<b>

<b>alpha<b>
<b>AMI (Amazon Machine Images)<b>
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

### **base networks**

- **predicting with**
- **to extract features**

**baseline models**
**base_model summary**

### **batch gradient descent (BGD)**

- **derivative**
- **direction**
- **gradient**
- **learning rate**
- **pitfalls of**
- **step size**

**batch hard (BH)**

### **batch normalization**

- **defined**
- **in neural networks**
- **in Keras**
- **overview**

**batch normalization (BN)**
**batch sample (BS)**
**batch weighted (BW)**
**batch_size hyperparameter**
**Bayes error rate**
**biases**
**BIER (boosting independent embeddings robustly)**
**binary classifier**
**binary_crossentropy function**
**block1_conv1 layer**
**block3_conv2 layer**
**block5_conv2 layer**
**block5_conv3 layer**
**blocks.**
**bottleneck layers**
**bottleneck residual block**
**bottleneck_residual_block function**
**bounding box coordinates**

### **bounding box prediction**

- **in YOLOv3**
- **predicting with regressors**

**bounding-box regressors**
**build_discriminator function**
**build_model() function**

---

## C

**Cars Dataset, Stanford**
**categories**
**CCTV monitoring**
**cGAN (conditional GAN)**
**chain rule**
**channels value**

### **CIFAR dataset**

- **Inception performance on**
- **ResNet performance on**

**CIFAR-10 dataset**
**class predictions**
**classes**
**classes argument**
**Class_id label**
**classification**
**classification loss**
**classification module**
**classifier learning algorithms**

### **classifiers**

- **binary**
- **in Keras**
- **pretrained networks as**

### **CLVR (cross-level vehicle re-identification)**

- **adding dropout layers to avoid overfitting**
- **advantages of**
- **in CNN architecture**
- **overview of dropout layers**
- **overview of overfitting**
- **architecture of**
- **AlexNet**

### **classification**

- **feature extraction**
- **GoogLeNet**
- **Inception**
- **LeNet-5**
- **ResNet**
- **VGGNet**

**convolutional layers**

### **convolutional operations**

- **kernel size**
- **number of filters in**
- **overview of convolution**
- **padding**
- **strides**
- **design patterns**
- **fully connected layers**
- **image classification**
- **building model architecture**
- **number of parameters**
- **weights**
- **with color images**
- **with MLPs**
- **implementing feature visualizer**
- **overview**
- **pooling layers**

### **convolutional layers**

- **max pooling vs. average pooling**
- **subsampling**
- **visualizing features**

**coarse label**
**COCO datasets**
**collecting data**
**color channel**

### **converting to grayscale images**

- **image classification for**

### **compiling models**

- **defining model architecture**
- **evaluating models**
- **image preprocessing**
- **loading datasets**
- **loading models with val_acc**
- **training models**

**combined models**
**combined-image**
**compiling models**
**computation problem**
**computer vision.**
**conda list command**
**confidence threshold**
**confusion matrix**
**connection weights**
**content image**
**content loss**
**content_image**
**content_loss function**
**content_weight parameter**
**contrastive loss**
**CONV_1 layer**
**CONV1 layer**
**CONV_2 layer**
**CONV2 layer**
**CONV3 layer**
**CONV4 layer**
**CONV5 layer**

### **ConvNet weights**

- **overview**

**convolutional layers**

### **convolutional operations**

- **kernel size**
- **number of filters in**
- **padding**
- **strides**

**convolutional neural network**
**convolutional neural networks**
**convolutional operations**
**correct prediction**

### **cost functions**

- **defined**
- **in neural networks**

**cross-entropy**
**cross-entropy loss**
**cuDNN**

### **CV (computer vision)**

- **applications of**

### **creating images**

- **face recognition**
- **image classification**
- **image recommendation systems**
- **localization**
- **neural style transfer**
- **object detection**

### **classifier learning algorithms**

- **extracting features**
- **automatically extracted features**
- **handcrafted features**
- **advantages of**
- **overview**
- **image input**

**color images**

### **computer processing of images**

- **images as functions**
- **image preprocessing**
- **interpreting devices**
- **pipeline**
- **sensing devices**
- **vision systems**
- **AI vision systems**
- **human vision systems**
- **visual perception**

---

## D

### **Darknet-53**

- **augmenting**
- **for image classification**
- **in AlexNet**
- **collecting**
- **loading**
- **mining**
- **BH**
- **BS**
- **BW**

### **dataloader**

- **finding useful triplets**
- **normalizing**
- **preparing for training**
- **preprocessing**
- **augmenting images**
- **grayscaling images**
- **resizing images**
- **splitting**

**data distillation**
**DataGenerator objects**
**dataloader**

### **downloading to GANs**

**Kaggle**
**loading**
**MNIST**
**splitting for training**
**splitting for validation**
**validation datasets**

**DCGANs (deep convolutional generative adversarial networks)**
**deep neural network**

### **DeepDream**

**algorithms in**
**in Keras**

**deltas**

### **dendrites**

**See also**

**Dense_1 layer**
**Dense_2 layer**
**dependencies, importing**
**deprocess_image(x)**

### **design patterns**

- **measuring speed of**
- **multi-stage vs. single-stage**
- **overfitting**
- **underfitting**

**dilated convolutions**
**dilation rate**

### **dimensionality reduction with Inception**

- **1 × 1 convolutional layer**
- **impact on network performance**

**direction**
**discriminator**

### **discriminator_model method**

- **in GANs**
- **training**
- **conda environment**
- **loading environments**
- **anual development environments**
- **saving environments**
- **setting up**

**dropout hyperparameter**

### **dropout layers**

- **adding to avoid overfitting**
- **advantages of**
- **in AlexNet**
- **in CNN architecture**
- **overview**

**dropout rate**
**dropout regularization**

---

## E

**early stopping**
**EC2 Management Console**
**EC2 On-Demand Pricing page**
**edges**
**embedding networks, training**
**finding similar items**
**implementation**
**object re-identification**
**testing trained models**
**object re-identification**
**retrievals**
**embedding space**
**endAnaconda**
**conda**
**developing manually**
**loading**
**saving**
**epochs**
**number of**
**training**
**error functions**
**advantages of**
**cross-entropy**
**errors**
**mean squared error**
**overview**
**weights**
**errors**
**evaluate() method**
**evaluation schemes**
**Evaluator class**
**exhaustive search algorithm**
**exploding gradients**
**exponential decay**

---

## F

**f argument**
**face identification**
**face recognition (FR)**
**face verification**
**false negatives (FN)**
**false positives (FP)**
**False setting**
**Fashion-MNIST**
**fashion_mnist.load_data() method**
**Fast R-CNNs (region-based convolutional neural networks)**
**architecture of**
**disadvantages of**
**multi-task loss function in**
**architecture of**
**base network to extract features**
**fully connected layers**
**multi-task loss function**
**object detection with**
**RPNs**
**anchor boxes**
**predicting bounding box with regressor**
**training**
**FC layer**
**FCNs (fully convolutional networks)**
**feature extraction**
**automatically**
**handcrafted features**
**feature extractors**
**feature maps**
**feature vector**
**feature visualizer**
**feature_layers**
**advantages of**
**handcrafted**
**learning**
**overview**
**transferring**
**visualizing**
**feedforward process**
**learning features**
**FID (Fréchet inception distance)**
**filter hyperparameter**
**filter_index**
**filters**
**filters argument**
**fine label**
**fine-tuning**
**advantages of**
**learning rates when**
**transfer learning**
**.fit() method**
**fit_generator() function**
**Flatten layer**
**flattened vector**
**FLOPs (floating-point operations per second)**
**flow_from_directory() method**
**foreground region**
**FPS (frames per second)**
**freezing layers**
**F-score**
**fully connected layers**
**images as**
**training**

---

## G

**gallery set**
**GANs (generative adversarial networks)**
**applications for**
**image-to-image translation**
**Pix2Pix GAN**
**SRGAN**
**architecture of**
**DCGANs**
**generator models**
**minimax function**
**building**
**combined models**
**discriminators**
**downloading datasets**
**evaluating models of**
**choosing evaluation scheme**
**FID**
**inception score**
**generators**
**importing libraries**
**training**
**discriminators**
**epochs**
**generators**
**training functions**
**visualizing datasets**
**generative models**
**generator models**
**generator_model function**
generators
in GANs
training
global average pooling
global minima
Google Open Images
GoogLeNet
architecture of
in Keras
building classifiers
building inception modules
building max-pooling layers
building network
learning hyperparameters in
GPUs (graphics processing units)
gradient ascent
gradient descent (GD)
overview
with momentum
gradients function
gram matrix
graph transformer network
converting color images
images
ground truth bounding box
GSTE (group-sensitive triplet embedding)

---

## H

hard data mining
hard negative sample
hard positive sample
heaviside step function
height value
hidden layers
hidden units
high-recall model
human in the loop
human vision systems
<b></b>hyperbolic tangent function
<b></b>in AlexNet
<b></b>in GoogLeNet
<b></b>in Inception
<b></b>in LeNet-5
<b></b>in ResNet
<b></b>in VGGNet
<b></b>neural network hyperparameters
<b></b>parameters vs.
<b></b>tuning
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

padding
PAMTRI (pose aware multi-task learning)
calculating
hyperparameters vs.
non-trainable params
number of
overview
trainable params
non-trainable
trainable
PASCAL VOC-2012 dataset
Path-LSTM
patience variable
.pem file
perceptrons
learning logic of
neurons
overview
step activation function
weighted sum function
performance metrics
accuracy
confusion matrix
F-score
person re-identification
pip install
Pix2Pix GAN (generative adversarial network)
plot_generated_images() function
plotting learning curves
POOL layer
POOL_1 layer
POOL_2 layer
pooling layers
convolutional layers
max pooling vs. average pooling
PR CURVE (precision-recall curve)
precision
predictions
across different scales
bounding box with regressors
for networks
with base network
data
augmenting images
grayscaling images
normalizing data
resizing images
images
converting color images to grayscale images
one-hot encoding
preparing labels
splitting datasets for training
splitting datasets for validation
pretrained model
as classifiers
as feature extractors
priors

---

## Q

query sets
Quick, Draw! dataset, Google

---

## R

disadvantages of
limitations of
multi-stage detectors vs. single-stage detectors
object detection with
training
receptive field
reduce argument
reduce layer
reduce shortcut
region proposals
regions of interest (RoIs)
regression layer
regressors
regular shortcut
regularization techniques to avoid overfitting
augmenting data
dropout layers
L2 regularization
ReLU (rectified linear unit)
activation functions
leaky
rescaling images
residual blocks
residual module architecture
residual notation
resizing images
ResNet (Residual Neural Network)
features of
in Keras
learning hyperparameters in
performance on CIFAR dataset
residual blocks
results, observing
retrievals
RGB (Red Green Blue)
RoI extractor
RoI pooling layer
RoIs (regions of interest)
RPNs (region proposal networks)
anchor boxes
predicting bounding box with regressors
training
runtime analysis of losses

---

## S

s argument
save_interval
scalar
scalar multiplication
scales, predictions across
scipy.optimize.fmin_l_bfgs_b method
sensing devices
shortcut path
Siamese loss
sigmoid function
single class
single-stage detectors
skip connections
Softmax layer
source domain
spatial features
specificity
data
for training
for validation
SRGAN (super-resolution generative adversarial networks)
architecture of
base network
multi-scale feature layers
architecture of multi-scale layers
multi-scale detections
non-maximum suppression
object detection with
training networks
building models
configuring models
creating models
loading data
making predictions
training models
visualizing loss
SSDLoss function
ssh command
StackGAN (stacked generative adversarial network)
step activation function
step function
step functions.
step size
stochastic gradient descent (SGD)
strides
style loss
gram matrix for measuring jointly activated feature maps
multiple layers for representing style features
style_loss function
style_weight parameter
subsampling
supervised learning
suppression.
synapses
synset (synonym set)

---

## T

tanh (hyperbolic tangent function)
tanh activation function
Tensorflow playground
tensors
testing trained model
object re-identification
retrievals
test_path variable
test_targets
test_tensors
TN (true negatives)
to_categorical function
top-1 error rate
top-5 error rate
top-k accuracy
total variance loss
total variation loss
total_loss function
total_variation_loss function
total_variation_weight parameter
TP (true positives)
train() function
trainable params
train_acc value
AlexNet
by trial and error
discriminators
embedding networks
finding similar items
implementation
object re-identification
testing trained models
epochs
functions
GANs
generators
models
networks
preparing data for
augmenting data
normalizing data
one-hot encode labels
preprocessing data
splitting data
R-CNNs
RPNs
splitting datasets for
SSD networks
building models
configuring models
creating models
loading data
making predictions
training models
visualizing loss
train_loss value
train_on_batch method
train_path variable
transfer functions
in GANs
linear
transfer learning
approaches to
using pretrained network as classifier
using pretrained network as feature extractor
choosing level of
when target dataset is large and different from source dataset
when target dataset is large and similar to source dataset
when target dataset is small and different from source
when target dataset is small and similar to source dataset
fine-tuning
open source datasets
CIFAR
Fashion-MNIST
Google Open Images
ImageNet
Kaggle
MNIST
MS COCO
overview
neural networks learning features
transferring features
pretrained networks as feature extractors
when to use
transferring features
transposition
triplets, finding
tuning hyperparameters
collecting data vs.
neural network hyperparameters
parameters vs. hyperparameters

---

## U

underfitting
untrained layers
Upsampling layer
Upsampling2D layer

---

## V

val_acc
val_acc value
val_error value
overview
splitting
valid_path variable
val_loss value
VAMI (viewpoint attentive multi-view inference)
vanishing gradients
vector space
VeRi dataset
VGG16 configuration
VGG19 configuration
VGGNet (Visual Geometry Group at Oxford University)
configurations
features of
learning hyperparameters in
performance
vision systems
AI
human
visual embedding layer
visual embeddings
face recognition
image recommendation systems
learning embedding
loss functions
contrastive loss
cross-entropy loss
naive implementation
runtime analysis of losses
mining informative data
BH
BS
BW
dataloader
finding useful triplets
training embedding networks
finding similar items
implementation
object re-identification
testing trained models
visual perception
datasets
features
loss
VUIs (voice user interfaces)

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
