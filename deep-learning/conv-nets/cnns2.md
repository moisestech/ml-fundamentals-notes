# CNNs 2

## [Going Further With CNNs, Lesson 5, Udacity, UD187](https://classroom.udacity.com/courses/ud187/lessons/1771027d-8685-496f-8891-d7786efb71e1/concepts/db0b93a6-402d-4f13-8869-cf5fc1fe89ad)

---

## 1. **Interview with Sebastian**

🎥 [Udacity, Video Link](https://youtu.be/cwR1UVLSTEM)

---

## 2. **Introduction**

🎥 [Udacity, Video Link](https://youtu.be/iNyFPh62czY)

---

## 3. **Dogs and Cats Dataset**

🎥 [Udacity, Video Link](https://youtu.be/Ls1rr3GfGXw)

---

## 4. **Images of Different Sizes**

🎥 [Udacity, Video Link](https://youtu.be/g3MD39QiRys)

---

## 5. **Color Images Part 1**

🎥 [Udacity, Video Link](https://youtu.be/pJNusvS8pho)

---

## 6. **Color Images Part 2**

🎥 [Udacity, Video Link](https://youtu.be/-3vRxZq9pBQ)

---

## 7. **Convolutions with Color Images**

🎥 [Udacity, Video Link](https://youtu.be/iDH3LCZwL5M)

---

## 8. **Max Pooling with Color Images**

🎥 [Udacity, Video Link](https://youtu.be/0Un1PwdSREQ)

---

## 9. **Colab: Cats and Dogs**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Cats and Dogs](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb)

🎥 [Udacity, Video Link](https://youtu.be/bBDN6-WmRO8)

- **Correction**: In the video, at the 4:04 mark, the instructor says "feed that to a **150** neuron Dense layer" . It should be "feed that to a **512** neuron Dense layer" as shown in the Colab notebook.

---

## 10. **Softmax and Sigmoid**

- In the previous Colab, we used the following the CNN architecture:

```python
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')

])
```

- Notice that our last layer (our classifier) consists of a `Dense` layer with `2` output units and a `softmax` activation function, as seen below:

```python
  tf.keras.layers.Dense(2, activation='softmax')
```

Another popular approach when working with binary classification problems, is to use a classifier that consists of a Dense layer with 1 output unit and a sigmoid activation function, as seen below:

```python
  tf.keras.layers.Dense(1, activation='sigmoid')
```

- Either of these two options will work well in a binary classification problem. However, you should keep in mind, that if you decide to use a `sigmoid` activation function in your classifier, you will also have to change the loss parameter in the `model.compile()` method, from `'sparse_categorical_crossentropy'` to `'binary_crossentropy'`, as shown below:

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
;
```

---

## 11. **Validation**

🎥 [Udacity, Video Link](https://youtu.be/efH2YKBeTm0)

---

## 12. **Image Augmentation**

🎥 [Udacity, Video Link](https://youtu.be/Qgd7maIVytI)

---

## 13. **Dropout**

🎥 [Udacity, Video Link](https://youtu.be/9k05QQeKto0)

---

## 14. **Colab: Cats and Dogs Revisited**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Cats and Dogs with Image Augmentation](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c02_dogs_vs_cats_with_augmentation.ipynb)

---

## 15. **Other Techniques to Prevent Overfitting**

### Other Techniques to Prevent Overfitting

In this lesson we saw three different techniques to prevent overfitting:

- **Early Stopping**: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
- **Image Augmentation**: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
- **Dropout**: Removing a random selection of a fixed number of neurons in a neural network during training.
- **However**, these are not the only techniques available to prevent overfitting. You can read more about these and other techniques in the link below:

- [Memorizing is not learning! — 6 tricks to prevent overfitting in machine learning](https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42)

---

## 16. **Exercise: Classify Images of Flowers**

### Exercise: Classify Images of Flowers

- Now is your turn to apply everything you learned in this lesson to create your own CNN to classify images of flowers. We will provide you with a Colab to guide you through the creation and training of your CNN. In the Colab we will download the Flowers dataset for you and separate it into training and validation sets. We will also provide a separate Colab with our solution, so that you you can compare your answers to ours.

- We strongly suggest that you write all the code yourself and not to copy and paste the code from other Colabs. At the end of the Colab we encourage you to have fun and experiment with different model architectures and parameters to see if you can increase your accuracy.

- Have fun!

### Colab 1

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Exercise: Classify Images of Flowers](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb)

---

## 17. **Solution: Classify Images of Flowers**

### Solution: Classify Images of Flowers

- We hope you had fun creating your own CNN to classify images of flowers! Feel free to check your solution with ours by taking a look at the Colab below.

### Colab 2

- To access the Colab Notebook, login to your Google account and click on the link below:

- Solution: [Classify Images of Flowers](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb)

---

## 18. **Summary**

### Summary

- In this lesson we learned how Convolutional Neural Networks work with color images and saw various techniques that we can use to avoid overfitting . The main key points of this lesson are:

### CNNs with RGB Images of Different Sizes:

- **Resizing**: When working with images of different sizes, you must resize all the images to the same size so that they can be fed into a CNN.
- **Color Images**: Computers interpret color images as 3D arrays.
  RGB Image: Color image composed of 3 color channels: Red, Green, and Blue.
- **Convolutions**: When working with RGB images we convolve each color channel with its own convolutional filter. Convolutions on each color channel are performed in the same way as with grayscale images, i.e. by performing element-wise multiplication of the convolutional filter (kernel) and a section of the input array. The result of each convolution is added up together with a bias value to get the convoluted output.
- **Max Pooling**: When working with RGB images we perform max pooling on each color channel using the same window size and stride. Max pooling on each color channel is performed in the same way as with grayscale images, i.e. by selecting the max value in each window.
- **Validation Set**: We use a validation set to check how the model is doing during the training phase. Validation sets can be used to perform Early Stopping to prevent overfitting and can also be used to help us compare different models and choose the best one.

### Methods to Prevent Overfitting:

- **Early Stopping**: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
- **Image Augmentation**: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
- **Dropout**: Removing a random selection of a fixed number of neurons in a neural network during training.

- You also created and trained a Convolutional Neural Network to classify images of Dogs and Cats with and without Image Augmentation and Dropout. You were able to see that Image Augmentation and Dropout greatly reduces overfitting and improves accuracy. As an exercise, you were able to apply everything you learned in this lesson to create your own CNN to classify images of flowers.

---

## Foam Related Links

- [[_conv-nets]]
- [[cnns]]
