# CNNs

## [🎓 Intro to CNNs, Lesson 4, Udacity, UD187](https://classroom.udacity.com/courses/ud187/lessons/f00868fe-5974-48c4-bf36-41c0372bed64/concepts/e22ccc36-783f-4912-9ac2-aca58787e05d)

---

## [**1. Interview with Sebastian**]

🎥 [Udacity, Video Link](https://youtu.be/4wYUnb-YjUc)

---

## **2. Introduction**

🎥 [Udacity, Video Link](https://youtu.be/m7m9XWWcQOM)

---

## **3. Convolutions**

🎥 [Udacity, Video Link](https://youtu.be/sAPg-qaT0b4)

---

## **4. Max Pooling**

🎥 [Udacity, Video Link](https://youtu.be/o_DJ-FO6dw0)

---

## **5. Recap**

- We just learned about convolutions and max pooling.

- A convolution is the process of applying a filter (“kernel”) to an image. Max pooling is the process of reducing the size of the image through downsampling.

- As you will see in the following Colab notebook, convolutional layers can be added to the neural network model using the `Conv2D` layer type in Keras. This layer is similar to the Dense layer, and has weights and biases that need to be tuned to the right values. The `Conv2D` layer also has kernels (filters) whose values need to be tuned as well. So, in a `Conv2D` layer the values inside the filter matrix are the variables that get tuned in order to produce the right output.

Here are some of terms that were introduced in this lesson:

- **CNNs**: Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers.
- **Convolution**: The process of applying a kernel (filter) to an image
- **Kernel / filter**: A matrix which is smaller than the input, used to transform the input into chunks
- Padding: Adding pixels of some value, usually 0, around the input image
- **Pooling** The process of reducing the size of an image through downsampling.There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.
- **Maxpooling**: A pooling process in which many values are converted into a single value by taking the maximum value from among them.
- **Stride**: the number of pixels to slide the kernel (filter) across the image.
- **Downsampling**: The act of reducing the size of an image
  ;

---

## **6. Colab: Fashion MNIST with CNNs**

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Fashion MNIST with CNNs](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l04c01_image_classification_with_cnns.ipynb)

🎥 [Udacity, Video Link](https://youtu.be/niylIkhErZo)

---

## **7. Summary**

### Summary

- In this lesson we learned about Convolutional Neural Networks. We learned how convolutions and max pooling works. You created and trained a Convolutional Neural Network from scratch. We saw that our CNN was able to perform better than the neural network we created in the previous lesson.

- If you want to know more details about how CNNs works make sure to check out this [Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).

---

## Foam Related Link

- [[cnns]]
- [[cnns2]]
- [[_conv-nets]]
