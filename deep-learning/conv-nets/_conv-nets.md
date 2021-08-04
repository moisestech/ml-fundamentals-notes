# Conv Nets

[Convolutional Neural Networks, Lesson 5. Udacity, UD188](https://classroom.udacity.com/courses/ud188/lessons/b1e148af-0beb-464e-a389-9ae293cb1dcd/concepts/e7190f8c-c824-4936-89ff-db6230fd3d12)

---

## **1. Introducing Alexis**

🎥 [Udacity, Video Link](https://youtu.be/38ExGpdyvJI)

Find me on Twitter! @alexis_b_cook

---

## **2. Applications of CNNs**

🎥 [Udacity, Video Link](https://youtu.be/HrYNL_1SV2Y)

Optional Resources
Read about the WaveNet model.

Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found here.
Learn about CNNs for text classification.

You might like to sign up for the author's Deep Learning Newsletter!
Read about Facebook's novel CNN approach for language translation that achieves state-of-the-art accuracy at nine times the speed of RNN models.

Play Atari games with a CNN and reinforcement learning. You can download the code that comes with this paper.

If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's post.
Play pictionary with a CNN!

Also check out all of the other cool implementations on the A.I. Experiments website. Be sure not to miss AutoDraw!
Read more about AlphaGo.

Check out this article, which asks the question: If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?
Check out these really cool videos with drones that are powered by CNNs.

Here's an interview with a startup - Intelligent Flying Machines (IFM).
Outdoor autonomous navigation is typically accomplished through the use of the global positioning system (GPS), but here's a demo with a CNN-powered autonomous drone.
If you're excited about using CNNs in self-driving cars, you're encouraged to check out:

our Self-Driving Car Engineer Nanodegree, where we classify signs in the German Traffic Sign dataset in this project.
our Machine Learning Engineer Nanodegree, where we classify house numbers from the Street View House Numbers dataset in this project.
this series of blog posts that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.
Check out some additional applications not mentioned in the video.

Some of the world's most famous paintings have been turned into 3D for the visually impaired. Although the article does not mention how this was done, we note that it is possible to use a CNN to predict depth from a single image.
Check out this research that uses CNNs to localize breast cancer.
CNNs are used to save endangered species!
An app called FaceApp uses a CNN to make you smile in a picture or change genders.

---

## **3. Lesson Outline**

🎥 [Udacity, Video Link](https://youtu.be/77LzWE1qQrc)

Hi, I'm Cezanne Camacho, I'll be teaching this lesson in tandem with Alexis, and later on, show you how to implement CNNs in PyTorchand use them in a variety of ways.

What is a feature?
I’ve found that a helpful way to think about what a feature is, is to think about what we are visually drawn to when we first see an object and when we identify different objects. For example, what do we look at to distinguish a cat and a dog? The shape of the eyes, the size, and how they move are just a couple of examples of visual features.

As another example, say we see a person walking toward us and we want to see if it’s someone we know; we may look at their face, and even further their general shape, eyes (and even color of their eyes). The distinct shape of a person and their eye color a great examples of distinguishing features!

Next, we’ll see that features like these can be measured, and represented as numerical data, by a machine.

---

## **4. MNIST Dataset**

🎥 [Udacity, Video Link](https://youtu.be/a7bvIGZpcnk)

MNIST Data
The MNIST database is arguably the most famous database in the field of deep learning! Check out this figure that shows datasets referenced over time in NIPS papers.

---

## **5. How Computers Interpret Images**

🎥 [Udacity, Video Link](https://youtu.be/mEPfoM68Fx4)

In the case of our 28x28 images, how many entries will the corresponding, image vector have when this matrix is flattened?

Enter your response here
Normalizing image inputs
Data normalization is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. That is, the range of pixel values in one input image are the same as the range in another image. This standardization makes our model train and reach a minimum error, faster!

Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. Read more about the Normalize transformation in PyTorch.

The distribution of such data should resemble a Gaussian function centered at zero. For image inputs we need the pixel numbers to be positive, so we often choose to scale the data in a normalized range [0,1].

---

## **6. MLP Structure & Class Scores**

🎥 [Udacity, Video Link](https://youtu.be/fP0Odiai8sk)

After looking at existing work, how many hidden layers will you use in your MLP for image classification?

---

## **7. Do Your Research**

🎥 [Udacity, Video Link](https://youtu.be/CR4JeAn1fgk)

---

## **8. Loss & Optimization**

🎥 [Udacity, Video Link](https://youtu.be/BmPDtSXv18w)

---

## **9. Defining a Network in PyTorch**

🎥 [Udacity, Video Link](https://youtu.be/9gvaQvyfLfY)

ReLU Activation Function
The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values, this step ensures that our model trains efficiently!

A ReLU activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the positive part of the input, x. So, for an input image with any negative pixel values, this would turn all those values to 0, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.

ReLU function

---

## **10. Training the Network**

🎥 [Udacity, Video Link](https://youtu.be/904bfqibcCw)

Cross-Entropy Loss
In the PyTorch documentation, you can see that the cross entropy loss function actually involves two steps:

It first applies a softmax function to any output is sees
Then applies NLLLoss; negative log likelihood loss
Then it returns the average loss over a batch of data. Since it applies a softmax function, we do not have to specify that in the forward function of our model definition, but we could do this another way.

Another approach
We could separate the softmax and NLLLoss steps.

In the forward function of our model, we would explicitly apply a softmax activation function to the output, x.
...
...

## a softmax layer to convert 10 outputs into a distribution of class probabilities

x = F.log_softmax(x, dim=1)

return x
Then, when defining our loss criterion, we would apply NLLLoss

## cross entropy loss combines softmax and nn.NLLLoss() in one single class

## here, we've separated them

criterion = nn.NLLLoss()
This separates the usual criterion = nn.CrossEntropy() into two steps: softmax and NLLLoss, and is a useful approach should you want the output of a model to be class probabilities rather than class scores.

---

## **11. Pre-Notebook: MLP Classification, Exercise**

Notebook: MLP Classification
Now, you're ready to define and train an MLP in PyTorch. As you follow along this lesson, you are encouraged to open the referenced Jupyter notebooks. We will present a solution to you, but please try creating your own deep learning models! Much of the value in this experience will come from experimenting with the code, in your own way.

To open this notebook, you have two options:

Go to the next page in the classroom (recommended).
Clone the repo from Github and open the notebook mnist_mlp_exercise.ipynb in the convolutional-neural-networks > mnist-mlp folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.
Instructions
Define an MLP model for classifying MNIST images
Train it for some number of epochs and test your model to see how well it generalizes and measure its accuracy.
This is a self-assessed lab. If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

---

## **12. Notebook: MLP Classification, MNIST**

🎥 [Udacity, Jupyter Notebook Link](https://classroom.udacity.com/courses/ud188/lessons/b1e148af-0beb-464e-a389-9ae293cb1dcd/concepts/2da6f20f-5162-47b3-b1b9-f3f6c0191a0b)

---

## **13. One Solution**

🎥 [Udacity, Video Link](https://youtu.be/7q37WPjQhDA)

model.eval()
There is an omission in the above code: including model.eval() !

model.eval() will set all the layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation. So, you should set your model to evaluation mode before testing or validating your model and set it to model.train() (training mode) only during the training loop.

This is reflected in the previous notebook code and in our Github repository.

Optional Resources
Check out the first research paper to propose dropout as a technique for overfitting.
If you'd like more information on activation functions, check out this website.

---

## **14. Model Validation**

🎥 [Udacity, Video Link](https://youtu.be/b5934VsV3SA)

---

## **15. Validation Loss**

🎥 [Udacity, Video Link](https://youtu.be/uGPP_-pbBsc)

---

## **16. Image Classification Steps**

🎥 [Udacity, Video Link](https://youtu.be/UHFBnitKraA)

---

## **17. MLPs vs CNNs**

🎥 [Udacity, Video Link](https://youtu.be/Q7CR3cCOtJQ)

---

## **18. Local Connectivity**

🎥 [Udacity, Video Link](https://youtu.be/z9wiDg0w-Dc)

---

## **19. Filters and the Convolutional Layer**

🎥 [Udacity, Video Link](https://youtu.be/x_dhnhUzFNo)

---

## **20. Filters & Edges**

🎥 [Udacity, Video Link](https://youtu.be/hfqNqcEU6uI)

Filters
To detect changes in intensity in an image, you’ll be using and creating specific image filters that look at groups of pixels and react to alternating patterns of dark/light pixels. These filters produce an output that shows edges of objects and differing textures.

So, let’s take a closer look at these filters and see when they’re useful in processing images and identifying traits of interest.

---

## **21. Frequency in Images**

Frequency in images
We have an intuition of what frequency means when it comes to sound. High-frequency is a high pitched noise, like a bird chirp or violin. And low frequency sounds are low pitch, like a deep voice or a bass drum. For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s (Hz), and high pitches and made by high-frequency waves. Examples of low and high-frequency sound waves are pictured below. On the y-axis is amplitude, which is a measure of sound pressure that corresponds to the perceived loudness of a sound, and on the x-axis is time.

(Top image) a low frequency sound wave (bottom) a high frequency sound wave.

High and low frequency
Similarly, frequency in images is a rate of change. But, what does it means for an image to change? Well, images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly. This is easiest to see in an example.

High and low frequency image patterns.

Most images have both high-frequency and low-frequency components. In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another. Higher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.

High-frequency components also correspond to the edges of objects in images, which can help us classify those objects.

---

## **22. High-pass Filters**

🎥 [Udacity, Video Link](https://youtu.be/OpcFn_H2V-Q)

Edge Handling
Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

Extend The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

Padding The image is padded with a border of 0's, black pixels.

Crop Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

---

## **23. Quiz: Kernels**

Kernel convolution
Now that you know the basics of high-pass filters, let's see if you can choose the best one for a given task.

Four different kernels

QUIZ QUESTION
Of the four kernels pictured above, which would be best for finding and enhancing horizontal edges and lines in an image?

d

---

## **24. OpenCV & Creating Custom Filters**

OpenCV
Before we jump into coding our own convolutional kernels/filters, I'll introduce you to a new library that will be useful to use when dealing with computer vision tasks, such as image classification: OpenCV!

OpenCV logo

OpenCV is a computer vision and machine learning software library that includes many common image analysis algorithms that will help us build custom, intelligent computer vision applications. To start with, this includes tools that help us process images and select areas of interest! The library is widely used in academic and industrial applications; from their site, OpenCV includes an impressive list of users: “Along with well-established companies like Google, Yahoo, Microsoft, Intel, IBM, Sony, Honda, Toyota that employ the library, there are many startups such as Applied Minds, VideoSurf, and Zeitera, that make extensive use of OpenCV.”

So, note, how we import cv2 in the next notebook and use it to create and apply image filters!

Notebook: Custom Filters
The next notebook is called custom_filters.ipynb.

To open the notebook, you have two options:

Go to the next page in the classroom (recommended).
Clone the repo from Github and open the notebook custom_filters.ipynb in the convolutional-neural-networks > conv-visualization folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.
Instructions
Define your own convolutional filters and apply them to an image of a road
See if you can define filters that detect horizontal or vertical edges
This notebook is meant to be a playground where you can try out different filter sizes and weights and see the resulting, filtered output image!

---

## **25. Notebook: Finding Edges**

🎥 [Udacity, Jupyter Notebook Link](https://classroom.udacity.com/courses/ud188/lessons/b1e148af-0beb-464e-a389-9ae293cb1dcd/concepts/7fa63120-523f-46fb-ab49-b5c8481196a5)

---

## **26. Convolutional Layer**

The Importance of Filters
What you've just learned about different types of filters will be really important as you progress through this course, especially when you get to Convolutional Neural Networks (CNNs). CNNs are a kind of deep learning model that can learn to do things like image classification and object recognition. They keep track of spatial information and learn to extract features like the edges of objects in something called a convolutional layer. Below you'll see an simple CNN structure, made of multiple layers, below, including this "convolutional layer".

Layers in a CNN.

Convolutional Layer
The convolutional layer is produced by applying a series of many different image filters, also known as convolutional kernels, to an input image.

4 kernels = 4 filtered images.

In the example shown, 4 different filters produce 4 differently filtered output images. When we stack these images, we form a complete convolutional layer with a depth of 4!

A convolutional layer.

Learning
In the code you've been working with, you've been setting the values of filter weights explicitly, but neural networks will actually learn the best filter weights as they train on a set of image data. You'll learn all about this type of neural network later in this section, but know that high-pass and low-pass filters are what define the behavior of a network like this, and you know how to code those from scratch!

In practice, you'll also find that many neural networks learn to detect the edges of images because the edges of object contain valuable information about the shape of an object.

---

## **27. Convolutional Layers (Part 2)**

🎥 [Udacity, Video Link](https://youtu.be/RnM1D-XI--8)

The Jupyter notebook described in the video can be accessed from the deep-learning-v2-pytorch GitHub respository linked here. Navigate to the conv-visualization/ folder and open conv_visualization.ipynb.

Optional Resource
Check out this website, which allows you to create your own filter. You can then use your webcam as input to a convolutional layer and visualize the corresponding activation map!

---

## **28. Stride and Padding**

🎥 [Udacity, Video Link](https://youtu.be/GmStpNi8jBI)

---

## **29. CNNs in PyTorch**

🎥 [Udacity, Video Link](https://youtu.be/GNxzWfiz3do)

Check out the CIFAR-10 Competition's winning architecture!

---

## **30. Pooling Layers**

🎥 [Udacity, Video Link](https://youtu.be/_Ok5xZwOtrk)

Other kinds of pooling
Alexis mentioned one other type of pooling, and it is worth noting that some architectures choose to use average pooling, which chooses to average pixel values in a given window size. So in a 2x2 window, this operation will see 4 pixel values, and return a single, average of those four values, as output!

This kind of pooling is typically not used for image classification problems because maxpooling is better at noticing the most important details about edges and other features in an image, but you may see this used in applications for which smoothing an image is preferable.

Notebook: Layer Visualizations
The next notebooks are about visualizing the output of convolutional and pooling layers.

To open the notebook, you have two options:

Go to the next page in the classroom (recommended).
Clone the repo from Github and open the notebook conv_visualization.ipynb & maxpooling_visualization.ipynb in the convolutional-neural-networks > conv-visualization folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.
Instructions
This notebook is meant to give you a chance to explore the effect of convolutional layers, activations, and pooling layers!

---

## **31. Capsule Networks**

Alternatives to Pooling
It's important to note that pooling operations do throw away some image information. That is, they discard pixel information in order to get a smaller, feature-level representation of an image. This works quite well in tasks like image classification, but it can cause some issues.

Consider the case of facial recognition. When you think of how you identify a face, you might think about noticing features; two eyes, a nose, and a mouth, for example. And those pieces, together, form a complete face! A typical CNN that is trained to do facial recognition, should also learn to identify these features. Only, by distilling an image into a feature-level representation, you might get a weird result:

Given an image of a face that has been photoshopped to include three eyes or a nose placed above the eyes, a feature-level representation will identify these features and still recognize a face! Even though that face is fake/contains too many features in an atypical orientation.
So, there has been research into classification methods that do not discard spatial information (as in the pooling layers), and instead learn to spatial relationships between parts (like between eyes, nose, and mouth).

One such method, for learning spatial relationships between parts, is the capsule network.

Capsule Networks
Capsule Networks provide a way to detect parts of objects in an image and represent spatial relationships between those parts. This means that capsule networks are able to recognize the same object, like a face, in a variety of different poses and with the typical number of features (eyes, nose , mouth) even if they have not seen that pose in training data.

Capsule networks are made of parent and child nodes that build up a complete picture of an object.

Parts of a face, making up a whole image.
In the example above, you can see how the parts of a face (eyes, nose, mouth, etc.) might be recognized in leaf nodes and then combined to form a more complete face part in parent nodes.

What are Capsules?
Capsules are essentially a collection of nodes, each of which contains information about a specific part; part properties like width, orientation, color, and so on. The important thing to note is that each capsule outputs a vector with some magnitude and orientation.

Magnitude (m) = the probability that a part exists; a value between 0 and 1.
Orientation (theta) = the state of the part properties.
These output vectors allow us to do some powerful routing math to build up a parse tree that recognizes whole objects as comprised of several, smaller parts!

The magnitude is a special part property that should stay very high even when an object is in a different orientation, as shown below.

Cat face, recognized in a multiple orientations, co: this blog post.

Resources
You can learn more about capsules, in this blog post.
And experiment with an implementation of a capsule network in PyTorch, at this github repo.
Supporting Materials
Dynamic routing between capsules, hinton et al

---

## **32. Notebook: Layer Visualization**

🎥 [Udacity, Jupyter Notebook Link](https://classroom.udacity.com/courses/ud188/lessons/b1e148af-0beb-464e-a389-9ae293cb1dcd/concepts/c57115c6-d1d8-4055-b487-dfd6bf186b92)

---

## **33. Increasing Depth**

🎥 [Udacity, Video Link](https://youtu.be/YKif1KNpWeE)

---

## **34. CNNs for Image Classification**

🎥 [Udacity, Video Link](https://youtu.be/smaw5GqRaoY)

Padding
Padding is just adding a border of pixels around an image. In PyTorch, you specify the size of this border.

Why do we need padding?

When we create a convolutional layer, we move a square filter around an image, using a center-pixel as an anchor. So, this kernel cannot perfectly overlay the edges/corners of images. The nice feature of padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

The most common methods of padding are padding an image with all 0-pixels (zero padding) or padding them with the nearest pixel value. You can read more about calculating the amount of padding, given a kernel_size, here.

QUESTION 1 OF 2
How might you define a Maxpooling layer, such that it down-samples an input by a factor of 4? (A checkbox indicates that you should select ALL answers that apply.)

QUESTION 2 OF 2
If you want to define a convolutional layer that is the same x-y size as an input array, what padding should you have for a kernel_size of 7? (You may assume that other parameters are left as their default values.)

PyTorch Layer Documentation
Convolutional Layers
We typically define a convolutional layer in PyTorch using nn.Conv2d, with the following parameters, specified:

nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
in_channels refers to the depth of an input. For a grayscale image, this depth = 1
out_channels refers to the desired depth of the output, or the number of filtered images you want to get as output
kernel_size is the size of your convolutional kernel (most commonly 3 for a 3x3 kernel)
stride and padding have default values, but should be set depending on how large you want your output to be in the spatial dimensions x, y
Read more about Conv2d in the documentation.

Pooling Layers
Maxpooling layers commonly come after convolutional layers to shrink the x-y dimensions of an input, read more about pooling layers in PyTorch, here.

---

## **35. Convolutional Layers in PyTorch**

Convolutional Layers in PyTorch
To create a convolutional layer in PyTorch, you must first import the necessary module:

import torch.nn as nn
Then, there is a two part process to defining a convolutional layer and defining the feedforward behavior of a model (how an input moves through the layers of a network). First, you must define a Model class and fill in two functions.

init

You can define a convolutional layer in the **init** function of by using the following format:

self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
forward

Then, you refer to that layer in the forward function! Here, I am passing in an input image x and applying a ReLU function to the output of this layer.

x = F.relu(self.conv1(x))
Arguments
You must pass the following arguments:

in_channels - The number of inputs (in depth), 3 for an RGB image, for example.
out_channels - The number of output channels, i.e. the number of filtered "images" a convolutional layer is made of or the number of unique, convolutional kernels that will be applied to an input.
kernel_size - Number specifying both the height and width of the (square) convolutional kernel.
There are some additional, optional arguments that you might like to tune:

stride - The stride of the convolution. If you don't specify anything, stride is set to 1.
padding - The border of 0's around an input array. If you don't specify anything, padding is set to 0.
NOTE: It is possible to represent both kernel_size and stride as either a number or a tuple.

There are many other tunable arguments that you can set to change the behavior of your convolutional layers. To read more about these, we recommend perusing the official documentation.

Pooling Layers
Pooling layers take in a kernel_size and a stride. Typically the same value as is the down-sampling factor. For example, the following code will down-sample an input's x-y dimensions, by a factor of 2:

self.pool = nn.MaxPool2d(2,2)
forward

Here, we see that poling layer being applied in the forward function.

x = F.relu(self.conv1(x))
x = self.pool(x)
Convolutional Example #1
Say I'm constructing a CNN, and my input layer accepts grayscale images that are 200 by 200 pixels (corresponding to a 3D array with height 200, width 200, and depth 1). Then, say I'd like the next layer to be a convolutional layer with 16 filters, each filter having a width and height of 2. When performing the convolution, I'd like the filter to jump two pixels at a time. I also don't want the filter to extend outside of the image boundaries; in other words, I don't want to pad the image with zeros. Then, to construct this convolutional layer, I would use the following line of code:

self.conv1 = nn.Conv2d(1, 16, 2, stride=2)
Convolutional Example #2
Say I'd like the next layer in my CNN to be a convolutional layer that takes the layer constructed in Example 1 as input. Say I'd like my new layer to have 32 filters, each with a height and width of 3. When performing the convolution, I'd like the filter to jump 1 pixel at a time. I want this layer to have the same width and height as the input layer, and so I will pad accordingly. Then, to construct this convolutional layer, I would use the following line of code:

self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

Convolution with 3x3 window and stride 1

Image source: http://iamaaditya.github.io/2016/03/one-by-one-convolution/

Sequential Models
We can also create a CNN in PyTorch by using a Sequential wrapper in the **init** function. Sequential allows us to stack different types of layers, specifying activation functions in between!

def **init**(self):
super(ModelName, self).**init**()
self.features = nn.Sequential(
nn.Conv2d(1, 16, 2, stride=2),
nn.MaxPool2d(2, 2),
nn.ReLU(True),

              nn.Conv2d(16, 32, 3, padding=1),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True)
         )

Formula: Number of Parameters in a Convolutional Layer
The number of parameters in a convolutional layer depends on the supplied values of filters/out_channels, kernel_size, and input_shape. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
D_in - the depth of the previous layer
Notice that K = out_channels, and F = kernel_size. Likewise, D_in is the last value in the input_shape tuple, typically 1 or 3 (RGB and grayscale, respectively).

Since there are F*F*D_in weights per filter, and the convolutional layer is composed of K filters, the total number of weights in the convolutional layer is K*F*F*D_in. Since there is one bias term per filter, the convolutional layer has K biases. Thus, the number of parameters in the convolutional layer is given by K*F*F*D_in + K.

Formula: Shape of a Convolutional Layer
The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
S - the stride of the convolution
P - the padding
W_in - the width/height (square) of the previous layer
Notice that K = out_channels, F = kernel_size, and S = stride. Likewise, W_in is the first and second value of the input_shape tuple.

The depth of the convolutional layer will always equal the number of filters K.

The spatial dimensions of a convolutional layer can be calculated as: (W_in−F+2P)/S+1

Flattening
Part of completing a CNN architecture, is to flatten the eventual output of a series of convolutional and pooling layers, so that all parameters can be seen (as a vector) by a linear classification layer. At this step, it is imperative that you know exactly how many parameters are output by a layer.

For the following quiz questions, consider an input image that is 130x130 (x, y) and 3 in depth (RGB). Say, this image goes through the following layers in order:

nn.Conv2d(3, 10, 3)
nn.MaxPool2d(4, 4)
nn.Conv2d(10, 20, 5, padding=2)
nn.MaxPool2d(2, 2)
QUESTION 1 OF 3
After going through all four of these layers in sequence, what is the depth of the final output?

20

QUESTION 2 OF 3
What is the x-y size of the output of the final maxpooling layer? Careful to look at how the 130x130 image passes through (and shrinks) as it moved through each convolutional and pooling layer.

16

QUESTION 3 OF 3
How many parameters, total, will be left after an image passes through all four of the above layers in sequence?

16*16*20

---

## **36. Feature Vector**

🎥 [Udacity, Video Link](https://youtu.be/g6QuiVno8zI)

---

## **37. CIFAR Classification Example**

🎥 [Udacity, Video Link](https://youtu.be/FF_EmZ2sf2w)

model.eval()
There is an omission in the above code: including model.eval() !

model.eval() will set all the layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation. So, you should set your model to evaluation mode before testing or validating your model and set it to model.train() (training mode) only during the training loop.

This is reflected in the following notebook code and in our Github repository.

---

## **38. Notebook: CNN Classification**

Notebook: CNNs in PyTorch
Now, you're ready to define and train an CNN in PyTorch.

To open this notebook, go to your local repo (found here on Github) and open the notebook cifar10_cnn_exercise.ipynb in the convolutional-neural-networks > cifar-cnn folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.

Instructions
Define a CNN model for classifying CIFAR10 images
Train it for some number of epochs and test your model to see how well it generalizes and measure its accuracy.
This is a self-assessed lab. If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

Note about GPUs
In this notebook, you'll find training the network is much faster if you use a GPU. However, you can still complete the exercises without a GPU. If you can't use a local GPU, we suggest you use cloud platforms such as AWS, GCP, and FloydHub to train your networks on a GPU.

---

## **39. Image Augmentation**

🎥 [Udacity, Video Link](https://youtu.be/zQnx2jZmjTA)

---

## **40. Augmentation Using Transformations**

🎥 [Udacity, Video Link](https://youtu.be/J_gjHVt9pVw)

Augmentation Code
You can take a look at the complete augmentation code in the previous notebook directory, or, directly in the Github repository.

---

## **41. Groundbreaking CNN Architectures**

🎥 [Udacity, Video Link](https://youtu.be/GdYOqihgb2k)

Optional Resources
Check out the AlexNet paper!
Read more about VGGNet here.
The ResNet paper can be found here.
Here's the Keras documentation for accessing some famous CNN architectures.
Read this detailed treatment of the vanishing gradients problem.
Here's a GitHub repository containing benchmarks for different CNN architectures.
Visit the ImageNet Large Scale Visual Recognition Competition (ILSVRC) website.

---

## **42. Visualizing CNNs (Part 1)**

🎥 [Udacity, Video Link](https://youtu.be/mnqS_EhEZVg)

(REALLY COOL) Optional Resources
If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:

Here's a section from the Stanford's CS231n course on visualizing what CNNs learn.
Check out this demonstration of a cool OpenFrameworks app that visualizes CNNs in real-time, from user-supplied video!
Here's a demonstration of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this video.
Read this Keras blog post on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:

Also check out this music video that makes use of Deep Dreams (look at 3:15-3:40)!
Create your own Deep Dreams (without writing any code!) using this website.
If you'd like to read more about interpretability of CNNs:

Here's an article that details some dangers from using deep learning models (that are not yet interpretable) in real-world applications.
There's a lot of active research in this area. These authors recently made a step in the right direction.

---

## **43. Visualizing CNNs (Part 2)**

Visualizing CNNs
Let’s look at an example CNN to see how it works in action.

The CNN we will look at is trained on ImageNet as described in this paper by Zeiler and Fergus. In the images below (from the same paper), we’ll see what each layer in this network detects and see how each layer detects more and more complex ideas.

Example patterns that cause activations in the first layer of the network. These range from simple diagonal lines (top left) to green blobs (bottom middle).

The images above are from Matthew Zeiler and Rob Fergus' deep visualization toolbox, which lets us visualize what each layer in a CNN focuses on.

Each image in the above grid represents a pattern that causes the neurons in the first layer to activate - in other words, they are patterns that the first layer recognizes. The top left image shows a -45 degree line, while the middle top square shows a +45 degree line. These squares are shown below again for reference.

As visualized here, the first layer of the CNN can recognize -45 degree lines.

The first layer of the CNN is also able to recognize +45 degree lines, like the one above.

Let's now see some example images that cause such activations. The below grid of images all activated the -45 degree line. Notice how they are all selected despite the fact that they have different colors, gradients, and patterns.

Example patches that activate the -45 degree line detector in the first layer.

So, the first layer of our CNN clearly picks out very simple shapes and patterns like lines and blobs.

Layer 2

A visualization of the second layer in the CNN. Notice how we are picking up more complex ideas like circles and stripes. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.

The second layer of the CNN captures complex ideas.

As you see in the image above, the second layer of the CNN recognizes circles (second row, second column), stripes (first row, second column), and rectangles (bottom right).

The CNN learns to do this on its own. There is no special instruction for the CNN to focus on more complex objects in deeper layers. That's just how it normally works out when you feed training data into a CNN.

Layer 3

A visualization of the third layer in the CNN. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.

The third layer picks out complex combinations of features from the second layer. These include things like grids, and honeycombs (top left), wheels (second row, second column), and even faces (third row, third column).

We'll skip layer 4, which continues this progression, and jump right to the fifth and final layer of this CNN.

Layer 5

A visualization of the fifth and final layer of the CNN. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.

The last layer picks out the highest order ideas that we care about for classification, like dog faces, bird faces, and bicycles.

---

## **44. Summary of CNNs**

🎥 [Udacity, Video Link](https://youtu.be/Te9QCvhx6N8)

External Resource
Deep learning eBook (2016) authored by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; published by Cambridge: MIT Press.

---

- [[_style-transfer]]
