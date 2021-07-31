# Conv Nets

---

## **1. Introducing Alexis**

[Udacity, Video Link]()

---

## **2. Applications of CNNs**

[Udacity, Video Link]()

---

## **3. Lesson Outline**

[Udacity, Video Link]()

---

## **4. MNIST Dataset**

[Udacity, Video Link]()

---

## **5. How Computers Interpret Images**

[Udacity, Video Link]()

---

## **6. MLP Structure & Class Scores**

[Udacity, Video Link]()

---

## **7. Do Your Research**

[Udacity, Video Link]()

---

## **8. Loss & Optimization**

[Udacity, Video Link]()

---

## **9. Defining a Network in PyTorch**

[Udacity, Video Link]()

---

## **10. Training the Network**

[Udacity, Video Link]()

---

## **11. Pre-Notebook: MLP Classification, Exercise**

[Udacity, Video Link]()

---

## **12. Notebook: MLP Classification, MNIST**

[Udacity, Video Link]()

---

## **13. One Solution**

[Udacity, Video Link]()

---

## **14. Model Validation**

[Udacity, Video Link]()

---

## **15. Validation Loss**

[Udacity, Video Link]()

---

## **16. Image Classification Steps**

[Udacity, Video Link]()

---

## **17. MLPs vs CNNs**

[Udacity, Video Link]()

---

## **18. Local Connectivity**

[Udacity, Video Link]()

---

## **19. Filters and the Convolutional Layer**

[Udacity, Video Link]()

---

## **20. Filters & Edges**

[Udacity, Video Link]()

---

## **21. Frequency in Images**

[Udacity, Video Link]()

---

## **22. High-pass Filters**

[Udacity, Video Link]()

---

## **23. Quiz: Kernels**

[Udacity, Video Link]()

---

## **24. OpenCV & Creating Custom Filters**

[Udacity, Video Link]()

---

## **25. Notebook: Finding Edges**

[Udacity, Video Link]()

---

## **26. Convolutional Layer**

[Udacity, Video Link]()

---

## **27. Convolutional Layers (Part 2)**

[Udacity, Video Link]()

---

## **28. Stride and Padding**

[Udacity, Video Link]()

---

## **29. CNNs in PyTorch**

[Udacity, Video Link]()

---

## **30. Pooling Layers**

[Udacity, Video Link]()

---

## **31. Capsule Networks**

[Udacity, Video Link]()

---

## **32. Notebook: Layer Visualization**

[Udacity, Video Link]()

---

## **33. Increasing Depth**

[Udacity, Video Link]()

---

## **34. CNNs for Image Classification**

[Udacity, Video Link]()

---

## **35. Convolutional Layers in PyTorch**

[Udacity, Video Link]()

---

## **36. Feature Vector**

[Udacity, Video Link]()

---

## **37. CIFAR Classification Example**

[Udacity, Video Link]()

---

## **38. Notebook: CNN Classification**

[Udacity, Video Link]()

---

## **39. Image Augmentation**

[Udacity, Video Link]()

---

## **40. Augmentation Using Transformations**

[Udacity, Video Link]()

---

## **41. Groundbreaking CNN Architectures**

[Udacity, Video Link](https://youtu.be/GdYOqihgb2k)

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

[Udacity, Video Link](https://youtu.be/mnqS_EhEZVg)

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

[Udacity, Video Link](https://youtu.be/Te9QCvhx6N8)

External Resource
Deep learning eBook (2016) authored by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; published by Cambridge: MIT Press.
