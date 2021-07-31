# RNNs

## **1. Intro to RNNs**

Recurrent Neural Networks
Hi! It's Luis again!

Now that you have some experience with PyTorch and deep learning, I'll be teaching you about recurrent neural networks (RNNs) and long short-term memory (LSTM) . RNNs are designed specifically to learn from sequences of data by passing the hidden state from one step in the sequence to the next step in the sequence, combined with the input. LSTMs are an improvement the RNNs, and are quite useful when our neural network needs to switch between remembering recent things, and things from long time ago. But first, I want to give you some great references to study this further. There are many posts out there about LSTMs, here are a few of my favorites:

Chris Olah's LSTM post
Edwin Chen's LSTM post
Andrej Karpathy's blog post on RNNs
Andrej Karpathy's lecture on RNNs and LSTMs from CS231n
So, let's dig in!

---

## **2. RNN vs LSTM**

🎥 [Udacity, Video Link](https://youtu.be/70MgF-IwAr8)

---

## **3. Basics of LSTM**

🎥 [Udacity, Video Link](https://youtu.be/gjb68a4XsqE)

---

## **4. Architecture of LSTM**

🎥 [Udacity, Video Link](https://youtu.be/ycwthhdx8ws)

---

## **5. The Learn Gate**

🎥 [Udacity, Video Link](https://youtu.be/aVHVI7ovbHY)

---

## **6. The Forget Gate**

🎥 [Udacity, Video Link](https://youtu.be/iWxpfxLUPSU)

---

## **7. The Remember Gate**

🎥 [Udacity, Video Link](https://youtu.be/0qlm86HaXuU)

---

## **8. The Use Gate**

🎥 [Udacity, Video Link](https://youtu.be/5Ifolm1jTdY)

---

## **9. Putting it All Together**

🎥 [Udacity, Video Link](https://youtu.be/IF8FlKW-Zo0)

---

## **10. Other architectures**

🎥 [Udacity, Video Link](https://youtu.be/MsxFDuYlTuQ)

---

## **11. Implementing RNNs**

Image of Cezanne
Hi, it's Cezanne!

Implementing Recurrent Neural Networks
Now that you've learned about RNNs and LSTMs from Luis, it's time to see how we implement them in PyTorch. With a bit of an assist from Mat, I'll be leading you through a couple notebooks showing how to build RNNs with PyTorch. First, I'll show you how to learn from time-series data. Then, you'll implement a character-level RNN. That is, it will learn from some text one character at a time, then generate new text one character at a time.

---

## **12. Time-Series Prediction**

🎥 [Udacity, Video Link](https://youtu.be/xV5jHLFfJbQ)

Code Walkthrough & Repository
The below video is a walkthrough of code that you can find in our public Github repository, if you navigate to recurrent-neural-networks > time-series and the Simple_RNN.ipynb notebook. Feel free to go through this code on your own, locally.

This example is meant to give you an idea of how PyTorch represents RNNs and how you might represent memory in code. Later, you'll be given more complex exercise and solution notebooks, in-classroom.

---

## **13. Training & Memory**

🎥 [Udacity, Video Link](https://youtu.be/sx7T_KP5v9I)

Recurrent Layers
Here is the documentation for the main types of recurrent layers in PyTorch. Take a look and read about the three main types: RNN, LSTM, and GRU.

Hidden State Dimensions
QUIZ QUESTION
Say you've defined a GRU layer with input_size = 100, hidden_size = 20, and num_layers=1. What will the dimensions of the hidden state be if you're passing in data, batch first, in batches of 3 sequences at a time?

---

## **14. Character-wise RNNs**

🎥 [Udacity, Video Link](https://youtu.be/dXl3eWCGLdU)

---

## **15. Sequence Batching**

🎥 [Udacity, Video Link](https://youtu.be/Z4OiyU0Cldg)

---

## **16. Notebook: Character-Level RNN**

Notebook: Character-Level RNN
Now you have all the information you need to implement an RNN of our own. The next few videos will be all about character-level text prediction with an LSTM!

It's suggested that you open the notebook in a new, working tab and continue working on it as you go through the instructional videos in this tab. This way you can toggle between learning new skills and coding/applying new skills.

To open this notebook, go to our notebook repo (available from here on Github) and open the notebook Character_Level_RNN_Exercise.ipynb in the recurrent-neural-networks > char-rnn folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.

Instructions
Load in text data
Pre-process that data, encoding characters as integers and creating one-hot input vectors
Define an RNN that predicts the next character when given an input sequence
Train the RNN and use it to generate new text
This is a self-assessed lab. If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

Note about GPUs
In this notebook, you'll find training these networks is much faster if you use a GPU. However, you can still complete the exercises without a GPU. If you can't use a local GPU, we suggest you use cloud platforms such as AWS, GCP, and FloydHub to train your networks on a GPU.

---

## **17. Implementing a Char-RNN**

🎥 [Udacity, Video Link]()

---

## **18. Batching Data, Solution**

🎥 [Udacity, Video Link]()

---

## **19. Defining the Model**

🎥 [Udacity, Video Link]()

---

## **20. Char-RNN, Solution**

🎥 [Udacity, Video Link]()

---

## **21. Making Predictions**

🎥 [Udacity, Video Link]()
