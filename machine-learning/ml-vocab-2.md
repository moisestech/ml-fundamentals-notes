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

- <b>Kaggle</b>
- <b>loading</b>
- <b>MNIST</b>
- <b>splitting for training</b>
- <b>splitting for validation</b>
- <b>validation datasets</b>

<b>DCGANs (deep convolutional generative adversarial networks)</b>
<b>deep neural network</b>

### <b>DeepDream</b>

- <b>algorithms in</b>
- <b>in Keras</b>

<b>deltas<b>

### <b>dendrites<b>

- <b>See also<b>

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

### <b>embedding networks, training</b>

- <b>finding similar items</b>
- <b>implementation</b>
- <b>object re-identification</b>
- <b>testing trained models</b>
- <b>object re-identification</b>
- <b>retrievals</b>

<b>embedding space</b>

### <b>endAnaconda</b>

- <b>conda</b>
- <b>developing manually</b>
- <b>loading</b>
- <b>saving</b>

### <b>epochs</b>

- <b>number of</b>
- <b>training</b>

### <b>error functions</b>

- <b>advantages of</b>
- <b>cross-entropy</b>

### <b>errors</b>

- <b>mean squared error</b>
- <b>overview</b>
- <b>weights</b>

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

### <b>Fast R-CNNs (region-based convolutional neural networks)</b>

- <b>architecture of</b>
- <b>disadvantages of</b>
- <b>multi-task loss function in</b>
- <b>architecture of</b>
- <b>base network to extract features</b>

### <b>fully connected layers</b>

- <b>multi-task loss function</b>
- <b>object detection with</b>
- <b>RPNs</b>
- <b>anchor boxes</b>
- <b>predicting bounding box with regressor</b>
- <b>training</b>

<b>FC layer</b>
<b>FCNs (fully convolutional networks)</b>

### <b>feature extraction</b>

- <b>automatically</b>
- <b>handcrafted features</b>

<b>feature extractors</b>
<b>feature maps</b>
<b>feature vector</b>
<b>feature visualizer</b>

### <b>feature_layers</b>

- <b>advantages of</b>
- <b>handcrafted</b>
- <b>learning</b>
- <b>overview</b>
- <b>transferring</b>
- <b>visualizing</b>

### <b>feedforward process</b>

- <b>learning features</b>

<b>FID (Fréchet inception distance)</b>
<b>filter hyperparameter<b>
<b>filter_index</b>
<b>filters</b>
<b>filters argument</b>
<b>fine label</b>

### <b>fine-tuning</b>

- <b>advantages of</b>
- <b>learning rates when</b>
- <b>transfer learning</b>

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

### <b>fully connected layers</b>

- <b>images as</b>
- <b>training</b>

---

## G

<b>gallery set</b>

### <b>GANs (generative adversarial networks)</b>

- <b>applications for</b>
- <b>image-to-image translation</b>
- <b>Pix2Pix GAN</b>
- <b>SRGAN</b>
- <b>architecture of</b>
- <b>DCGANs</b>

### <b>generator models</b>

- <b>minimax function</b>
- <b>building</b>
- <b>combined models</b>
- <b>discriminators</b>
- <b>downloading datasets</b>
- <b>evaluating models of</b>
- <b>choosing evaluation scheme</b>
- <b>FID</b>
- <b>inception score</b>

### <b>generators</b>

- <b>importing libraries</b>
- <b>training</b>
- <b>discriminators</b>
- <b>epochs</b>

### <b>generators</b>

- <b>training functions</b>
- <b>visualizing datasets</b>

<b>generative models</b>
<b>generator models</b>
<b>generator model function</b>

### <b>generators</b>

- <b>in GANs</b>
- <b>training</b>

<b>global average pooling</b>
<b>global minima</b>
<b>Google Open Images</b>

### <b>GoogLeNet</b>

- <b>architecture of</b>
- <b>in Keras</b>
- <b>building classifiers</b>
- <b>building inception modules</b>
- <b>building max-pooling layers</b>
- <b>building network</b>
- <b>learning hyperparameters in</b>

<b>GPUs (graphics processing units)</b>
<b>gradient ascent</b>

### <b>gradient descent (GD)</b>

- <b>overview</b>
- <b>with momentum</b>

<b>gradients function</b>
<b>gram matrix</b>

### <b>graph transformer network</b>

- <b>converting color images</b>
- <b>images</b>

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

### <b>hyperbolic tangent function</b>

- <b>in AlexNet</b>
- <b>in GoogLeNet</b>
- <b>in Inception</b>
- <b>in LeNet-5</b>
- <b>in ResNet</b>
- <b>in VGGNet</b>
- <b>neural network hyperparameters</b>
- <b>parameters vs.</b>
- <b>tuning</b>
- <b>collecting data vs.</b>
- <b>neural network hyperparameters</b>

- <b>parameters vs. hyperparameters</b>

- **hyperparameters** are variables that are not learned by the network.
  - They are set by the ML engineer before training the model and then tuned.
  - These are variables that define the network structure and determine how the network is trained.
  - Hyperparameter examples include **learning rate**, **batch size**, **number of epochs**, n**umber of hidden layers**

---

## I

<b>identity function</b>
<b>if-else statements</b>

### <b>image classification</b>

- <b>for color images</b>
- <b>compiling models</b>
- <b>defining model architecture</b>
- <b>evaluating models</b>

### <b>image preprocessing</b>

- <b>loading datasets</b>
- <b>loading models with val_acc</b>
- <b>training models</b>
- <b>with CNNs</b>
- <b>building model architecture</b>
- <b>number of parameters</b>
- <b>weights</b>
- <b>with high accuracy</b>
- <b>building model architecture</b>
- <b>evaluating models</b>

### <b>importing dependencies</b>

- <b>preparing data for training</b>
- <b>training models</b>
- <b>with MLPs</b>
- <b>drawbacks of</b>
- <b>hidden layers</b>

### <b>input layers</b>

- <b>output layers</b>

<b>image classifier</b>
<b>image flattening</b>
<b>image preprocessing</b>
<b>image recommendation systems</b>
<b>ImageDataGenerator class</b>
<b>ImageNet</b>
<b>ImageNet Large Scale Visual Recognition Challenge (ILSVRC)</b>

### <b>images</b>

- <b>as functions</b>
- <b>augmenting</b>
- <b>color images</b>
- <b>computer processing of</b>
- <b>creating</b>
- <b>grayscaling</b>
- <b>preprocessing</b>
- <b>converting color to grayscale</b>
- <b>one-hot encoding</b>
- <b>preparing labels</b>
- <b>splitting datasets for training</b>
- <b>splitting datasets for validation</b>
- <b>rescaling</b>
- <b>resizing</b>
- <b>image-to-image translation</b>

### <b>Inception</b>

- <b>architecture of</b>
- <b>features of</b>
- <b>learning hyperparameters in</b>
- <b>modules</b>
- <b>naive version</b>
- <b>performance on CIFAR dataset</b>
- <b>with dimensionality reduction</b>
- <b>1 × 1 convolutional layer</b>

<b>impact on network performance</b>
<b>inception scores</b>
<b>inception_module function</b>
<b>include_top argument</b>
<b>input image</b>
<b>input layers</b>
<b>input vector</b>
<b>input_shape argument</b>
<b>instances</b>
<b>interpreting devices</b>
<b>IoU (intersection over union)</b>

---

## J

<b>Jaccard distance</b>
<b>joint training</b>
<b>Jupyter notebooks</b>

---

## K

<b>K object classes</b>

### <b>Kaggle datasets</b>

- <b>AlexNet in</b>
- <b>batch normalization in</b>
- <b>DeepDream in</b>
- <b>GoogLeNet in</b>
- <b>building classifiers</b>
- <b>building inception modules</b>
- <b>building max-pooling layers</b>
- <b>building network</b>
- <b>LeNet-5 in</b>
- <b>ResNet in</b>

<b>keras.datasets</b>
<b>keras_ssd7.py file</b>
<b>kernel</b>
<b>kernel size</b>
<b>kernel_size hyperparameter</b>

---

## L

<b>L2 regularization</b>
<b>label smoothing</b>
<b>labeled data</b>
<b>labeled images</b>
<b>LabelImg application</b>
<b>labels</b>
<b>lambda parameter</b>
<b>lambda value</b>
<b>layer_name</b>

### <b>layers</b>

- <b>1 × 1 convolutional</b>
- <b>dropout</b>
- <b>adding to avoid overfitting</b>
- <b>advantages of</b>
- <b>in AlexNet</b>
- <b>in CNN architecture</b>
- <b>overview</b>
- <b>fully connected</b>
- <b>hidden<b>
- <b>representing style features</b>

<b>Leaky ReLU</b>

### <b>learning</b>

- <b>adaptive</b>
- <b>embedding</b>
- <b>features</b>
- <b>finding optimal learning rate</b>
- <b>in AlexNet</b>
- <b>in GoogLeNet</b>
- <b>in Inception</b>
- <b>in LeNet-5</b>
- <b>in ResNet</b>
- <b>in VGGNet</b>
- <b>mini-batch size</b>
- See also

### <b>learning curves, plotting</b>

- <b>batch gradient descent</b>
- <b>decay</b>
- <b>derivative and</b>
- <b>optimal, finding</b>
- <b>when fine-tuning</b>

### <b>LeNet-5</b>

- <b>architecture of</b>
- <b>in Keras</b>

### <b>learning hyperparameters in</b>

- <b>on MNIST dataset</b>

<b>libraries in GANs</b>
<b>linear combination</b>
<b>linear datasets</b>
<b>linear decay</b>
<b>linear transfer function</b>
<b>load_data() method</b>

### <b>load_dataset() method</b>

- <b>data</b>
- <b>datasets</b>
- <b>environments</b>
- <b>models</b>

<b>local minima</b>
<b>local response normalization</b>
<b>localization</b>
<b>localization module</b>
<b>locally connected layers</b>
<b>LocalResponseNorm layer</b>
<b>location loss</b>

### <b>logistic function</b>

- <b>content loss</b>
  <b>runtime analysis of</b>
  <b>total variance</b>
  <b>visualizing</b>

### <b>loss functions</b>

<b>contrastive loss</b>
<b>cross-entropy loss</b>
<b>naive implementation</b>
<b>loss value</b>
<b>lr variable</b>
<b>lr_schedule function</b>

---

## M

<b>MAC (multiplier-accumulator)</b>

### <b>MAC operation</b>

<b>human brain vs.</b>
<b>with handcrafted features</b>
<b>main path</b>
<b>make_blobs</b>
<b>matrices</b>
<b>matrix multiplication</b>
<b>max pooling</b>
<b>max-pooling layers</b>
<b>mean absolute error (MAE)</b>
<b>mean average precision (mAP)</b>
<b>mean squared error (MSE)</b>
<b>Mechanical Turk crowdsourcing tool, Amazon</b>
<b>metrics</b>
<b>min_delta argument</b>
<b>mini-batch gradient descent (MB-GD)</b>
<b>mini-batch size</b>
<b>minimax function</b>

### <b>mining data</b>

<b>BH</b>
<b>BS</b>
<b>BW</b>
<b>dataloader</b>
<b>finding useful triplets</b>
<b>mixed2 layer</b>
<b>mixed3 layer</b>
<b>mixed4 layer</b>
<b>mixed5 layer</b>

### <b>MLPs (multilayer perceptrons)</b>

<b>architecture of</b>
<b>hidden layers</b>
<b>image classification with</b>
<b>drawbacks of</b>
<b>hidden layers</b>
<b>input layers</b>
<b>output layers</b>
<b>layers</b>
<b>nodes</b>
<b>MNIST (Modified National Institute of Standards and Technology) dataset</b>
<b>architecture of</b>
<b>building</b>
<b>compiling</b>
<b>configuring</b>
<b>designing</b>
<b>evaluating</b>
<b>building networks</b>
<b>diagnosing overfitting</b>
<b>diagnosing underfitting</b>
<b>evaluating networks</b>
<b>plotting learning curves</b>
<b>training networks</b>
<b>loading</b>
<b>choosing evaluation scheme</b>
<b>evaluating</b>
<b>FID</b>
<b>inception score</b>
<b>testing</b>
<b>object re-identification</b>
<b>retrievals</b>
<b>training</b>
<b>momentum, gradient descent with</b>
<b>monitor argument</b>
<b>MS COCO (Microsoft Common Objects in Context)</b>
<b>multi-scale detections</b>

### <b>multi-scale feature layers</b>

<b>architecture of</b>
<b>multi-scale detections</b>
<b>multi-scale vehicle representation (MSVR)</b>
<b>multi-stage detectors</b>
<b>multi-task learning (MTL)</b>
<b>multi-task loss function</b>

---

## N

<b>naive implementation</b>
<b>naive representation</b>
<b>n-dimensional array</b>
<b>neg_pos_ratio</b>

### <b>networks</b>

<b>architecture of</b>
<b>activation type</b>
<b>depth of neural networks</b>
<b>improving</b>
<b>width of neural networks</b>
<b>building</b>
<b>evaluating</b>
<b>improving</b>
<b>in Keras</b>
<b>measuring precision of</b>
<b>predictions</b>
<b>as classifiers</b>
<b>as feature extractors</b>
<b>to extract features</b>
<b>training</b>

### <b>neural networks</b>

<b>activation functions</b>
<b>binary classifier</b>
<b>heaviside step function</b>
<b>leaky ReLU</b>
<b>linear transfer function</b>
<b>logistic function</b>
<b>ReLU</b>
<b>sigmoid function</b>
<b>softmax function</b>
<b>tanh</b>
<b>backpropagation</b>
<b>covariate shift in</b>
<b>depth of</b>
<b>error functions</b>
<b>advantages of</b>
<b>cross-entropy</b>
<b>errors</b>
<b>MSE</b>
<b>overview</b>
<b>weights</b>
<b>feedforward process</b>
<b>learning features</b>
<b>hyperparameters in</b>
<b>learning features</b>
<b>multilayer perceptrons</b>
<b>architecture of</b>
<b>hidden layers</b>
<b>layers</b>

### <b>nodes</b>

<b>optimization</b>
<b>optimization algorithms</b>
<b>batch gradient descent</b>
<b>gradient descent</b>
<b>MB-GD</b>
<b>stochastic gradient descent</b>
<b>overview</b>
<b>perceptrons</b>
<b>learning logic of</b>

### <b>neurons</b>

<b>overview</b>
<b>width of</b>

### <b>neural style transfer</b>

<b>content loss</b>
<b>network training</b>
<b>style loss</b>
<b>gram matrix for measuring jointly activated feature maps</b>
<b>multiple layers for representing style features</b>
<b>total variance loss</b>
<b>neurons</b>
<b>new_model</b>
<b>NMS (non-maximum suppression)</b>
<b>no free lunch theorem</b>
<b>node values</b>
<b>nodes</b>
<b>noise loss</b>
<b>nonlinear datasets</b>
<b>nonlinearities</b>
<b>non-trainable params</b>
<b>normalizing data</b>
<b>installer/application.yaml file</b>

---

## O

<b>load_weights() method</b>

### <b>object detection</b>

<b>framework</b>
<b>network predictions</b>
<b>NMS</b>
<b>object-detector evaluation metrics</b>
<b>region proposals</b>
<b>with Fast R-CNNs</b>
<b>architecture of</b>
<b>disadvantages of</b>
<b>multi-task loss function in</b>
<b>with Faster R-CNNs</b>
<b>architecture of</b>
<b>base network to extract features</b>
<b>fully connected layers</b><b>
</b><b>multi-task loss function</b>
<b>RPNs</b>
<b>with R-CNNs</b>
<b>disadvantages of</b>
<b>limitations of</b>
<b>multi-stage detectors vs. single-stage detectors</b>
<b>training</b>
<b>with SSD</b>
<b>architecture of</b>
<b>base networks</b>
<b>multi-scale feature layers</b>
<b>NMS</b>
<b>training SSD networks</b>
<b>with YOLOv3</b>
<b>architecture of</b>
<b>overview</b>
<b>object re-identification</b>

### <b>object-detector evaluation metrics</b>

<b>FPS to measure detection speed</b>
<b>IoU</b>
<b>mAP to measure network precision</b>
<b>PR CURVE</b>
<b>objectness score</b>
<b>octaves</b>
<b>offline training</b>
<b>one-hot encoding</b>
<b>online learning</b>
<b>Open Images Challenge</b>

### <b>open source datasets</b>

<b>CIFAR</b>
<b>Fashion-MNIST</b>
<b>Google Open Images</b>
<b>ImageNet</b>
<b>Kaggle</b>
<b>MNIST</b>
<b>MS COCO</b>
<b>optimal weights</b>
<b>optimization</b>

### <b>optimization algorithms</b>

<b>Adam (adaptive moment estimation)</b>
<b>batch gradient descent</b>
<b>derivative</b>
<b>direction</b>
<b>gradient</b>
<b>learning rate</b>
<b>pitfalls of</b>
<b>step size</b>
<b>early stopping</b>
<b>gradient descent</b>

### <b>overview</b>

<b>with momentum</b>
<b>MB-GD</b>
<b>number of epochs</b>
<b>stochastic gradient descent</b>
<b>optimization value</b>
<b>optimized weights</b>
<b>optimizer</b>
<b>output layer</b>

### <b>Output Shape columns</b>

<b>adding dropout layers to avoid</b>
<b>diagnosing</b>

### <b>overview</b>

<b>regularization techniques to avoid</b>
<b>augmenting data</b>
<b>dropout layers</b>
<b>L2 regularization</b>

---

## P

<b>padding</b>

### <b>PAMTRI (pose aware multi-task learning)</b>

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

### <b>perceptrons</b>

<b>learning logic of</b>
<b>neurons</b>
<b>overview</b>
<b>step activation function</b>
<b>weighted sum function</b>

### <b>performance metrics</b>

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

### <b>pooling layers</b>

<b>convolutional layers</b>
<b>max pooling vs. average pooling</b>
<b>PR CURVE (precision-recall curve)</b>
<b>precision</b>

### <b>predictions</b>

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

### <b>preparing labels</b>

<b>splitting datasets for training</b>
<b>splitting datasets for validation</b>

### <b>pretrained model</b>

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

### <b>regularization techniques to avoid overfitting</b>

<b>augmenting data</b>
<b>dropout layers</b>
<b>L2 regularization</b>

### <b>ReLU (rectified linear unit)</b>

<b>activation functions</b>
<b>leaky</b>
<b>rescaling images</b>
<b>residual blocks</b>
<b>residual module architecture</b>
<b>residual notation</b>
<b>resizing images</b>

### <b>ResNet (Residual Neural Network)</b>

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

### <b>RPNs (region proposal networks)</b>

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

### <b>specificity</g>

<b>data</g>
<b>for training</g>
<b>for validation</g>

### <b>SRGAN (super-resolution generative adversarial networks)</g>

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

### <b>testing trained model</b>

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

### <b>train_acc value</b>

<b>AlexNet</b>
<b>by trial and error</b>
<b>discriminators</b>
<b>embedding networks</b>
<b>finding similar items</b>
<b>implementation</b>
<b>object re-identification</b>

### <b>testing trained models</b>

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

### <b>training models</b>

<b>visualizing loss</b>
<b>train_loss value</b>
<b>train_on_batch method</b>
<b>train_path variable</b>

### <b>transfer functions</b>

<b>in GANs</b>
<b>linear</b>

### <b>transfer learning</b>

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

### <b>transferring features</b>

<b>pretrained networks as feature extractors</b>
<b>when to use</b>
<b>transferring features</b>
<b>transposition</b>
<b>triplets, finding</b>

### <b>tuning hyperparameters</b>

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

### <b>val_error value</b>

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

### <b>VGGNet (Visual Geometry Group at Oxford University)</b>

<b>configurations</b>
<b>features of</b>
<b>learning hyperparameters in</b>
<b>performance</b>

### <b>vision systems</b>

<b>AI</b>
<b>human</b>
<b>visual embedding layer</b>

### <b>visual embeddings</b>

- <b>face recognition</b>
- <b>image recommendation systems</b>
- <b>learning embedding</b>
- <b>loss functions</b>
- <b>contrastive loss</b>
- <b>cross-entropy loss</b>
- <b>naive implementation</b>
- <b>runtime analysis of losses</b>
- <b>mining informative data</b>
- <b>BH</b>
- <b>BS</b>
- <b>BW</b>
- <b>dataloader</b>
- <b>finding useful triplets</b>
- <b>training embedding networks</b>
- <b>finding similar items</b>
- <b>implementation</b>
- <b>object re-identification</b>
- <b>testing trained models</b>

### <b>visual perception</b>

- <b>datasets</b>
- <b>features</b>
- <b>loss</b>

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

### <b>weights</b>

- <b>calculating parameters</b>
- <b>non-trainable params</b>
- <b>trainable params</b>

<b>weights vector</b>
<b>width value</b>

---

## X

<b>X argument</b>
<b>x_test</b>
<b>x_train</b>
<b>x_valid</b>

---

## Y

<b>architecture of</b>
<b>object detection with</b>
<b>overview</b>
<b>output bounding boxes</b>
<b>predictions across different scales</b>

## Z

<b>zero-padding</b>

---

<link rel="stylesheet" type="text/css" media="all" href="../assets/css/custom.css" />

## Foam Related Links

- [[_ml]]
