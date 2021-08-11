# Transfer Learning

## [🎓 Transfer Learning, Lesson 6, Udacity, UD187](https://classroom.udacity.com/courses/ud187/lessons/f00868fe-5974-48c4-bf36-41c0372bed64/concepts/e22ccc36-783f-4912-9ac2-aca58787e05d)

---

## 1. **Interview with Sebastian**

🎥 [Udacity, Video Link](https://youtu.be/KPBDQMhFccc)

---

## 2. **Introduction**

🎥 [Udacity, Video Link](https://youtu.be/26DUtmjG6ms)

---

## 3. **Transfer Learning**

🎥 [Udacity, Video Link](https://youtu.be/JhR1_WZCj54)

---

## 4. **MobileNet**

🎥 [Udacity, Video Link](https://youtu.be/l4-NTkEB8sk)

---

## 5. **Colab: Cats and Dogs with Transfer Learning**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Cats and Dogs with Transfer Learning](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c01_tensorflow_hub_and_transfer_learning.ipynb)

- Code Walkthrough Part 1
  🎥 [Udacity, Video Link](https://youtu.be/oMQPOvHUqwM)

- Code Walkthrough Part 2
  🎥 [Udacity, Video Link](https://youtu.be/z9q5nffdcS0)

---

## 6. **Understanding Convolutional Neural Networks**

- So far we've seen that CNN's perform really well at classifying images.

  - However, at this point, we don't really know how these CNN's actually work.
  - If we could understand what a CNN is actually learning, then in principle, we should be able to improve it even further.
  - One way to try to understand CNNs is by visualizing the convolutional layers.
  - We encourage you to look at the link below to learn more about how to visualize convolutional layers:

- [Understanding your Convolution network with Visualizations](https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b)

---

## 7. **Exercise: Flowers with Transfer Learning**

- Now is your turn to apply transfer learning to create your own CNN to classify images of flowers.

  - We will provide you with a Colab to guide you through the creation and training of your CNN.
  - This time you will have to download the Flowers dataset yourself using **TensorFlow Datasets** and split it into a training and validation set.
  - We will provide a separate Colab with our solution, so that you you can compare your answers to ours.

- We strongly suggest that you write all the code yourself and not to copy and paste the code from other Colabs.

  - At the end of the Colab we encourage you to perform transfer learning using the Inception model and compare the accuracy of this model with MobileNet.

- Have fun!

### Colab

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Exercise: Flowers with Transfer Learning](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c02_exercise_flowers_with_transfer_learning.ipynb)

---

## 8. **Solution: Flowers with Transfer Learning**

- We hope you had fun using transfer learning to create your own CNN to classify images of flowers! Feel free to check your solution with ours by taking a look at the Colab below.

### Colab

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Solution: Flowers with Transfer Learning](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb)

---

## 9. **Summary**

- In this lesson we learned how we can use Transfer Learning to create very powerful Convolutional Neural Networks with very little effort. The main key points of this lesson are:

- **Transfer Learning**: A technique that reuses a model that was created by machine learning experts and that has already been trained on a large dataset. When performing transfer learning we must always change the last layer of the pre-trained model so that it has the same number of classes that we have in the dataset we are working with.
- **Freezing Parameters**: Setting the variables of a pre-trained model to non-trainable. By freezing the parameters, we will ensure that only the variables of the last classification layer get trained, while the variables from the other layers of the pre-trained model are kept the same.
- **MobileNet**: A state-of-the-art convolutional neural network developed by Google that uses a very efficient neural network architecture that minimizes the amount of memory and computational resources needed, while maintaining a high level of accuracy. MobileNet is ideal for mobile devices that have limited memory and computational resources.

- You also used transfer learning to create a Convolutional Neural Network that uses MobileNet to classify images of Dogs and Cats. You were able to see that transfer learning greatly improves the accuracy achieved in the Dogs and Cats dataset. As an exercise, you were able to apply everything you learned in this lesson to create your own CNN using MobileNet to classify images of flowers.

;

---

## Foam Related Links

- [[_nns]]
