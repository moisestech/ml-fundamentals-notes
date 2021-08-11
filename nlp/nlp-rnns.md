# NLP RNNs

## [🎓 NLP: Recurrent Neural Networks, Lesson 10, Udacity, UD187, Intro to TensorFlow for Deep Learning](https://classroom.udacity.com/courses/ud187/lessons/52b95146-ddb7-471c-9237-20fb24d25237/concepts/0435d8ae-25af-4482-80f2-5e580976b4ab)

---

## **1. Recurrent Neural Networks Intro**

🎥 [Udacity, Video Link](https://youtu.be/S9-EPs5LO18)
In the previous lesson on Natural Language Processing with TensorFlow, we focused on Tokenization and Embeddings, which helped convert text into useful data for input into neural networks. However, these networks were not yet able to consider the actual sequence of the words in the input.

In this second lesson, we’ll dive into Recurrent Neural Networks (such as the LSTMs you saw in the Time Series Analysis lesson) as well as Text Generation, which allows for the creation of new text.

---

## **2. Basics of RNNs**

Recurrent Neural Networks (RNNs) still take in some input x and output some y, but they also feed some of the output of the network back into itself. This may be done over and over, so that with text input, the network has some memory of words that came much earlier in a sequence.

The basic RNN flow, with an input X, node F, and output Y, where information from one F is fed back into F for the next timestep.
The basic RNN flow

QUIZ QUESTION
How do recurrent neural networks (RNNs) differ from the network architectures you’ve been working with previously?

We’ve included some additional videos below that are optional for you to view and include a deeper dive into RNNs. They come from Udacity’s Deep Learning Nanodegree program, and concern the history of RNNs, and two parts on more RNN basics.

(Optional) History of RNNs

(Optional) RNN Basics Part 1

(Optional) RNN Basics Part 2

🎥 [Udacity, Video Link](https://youtu.be/uAHepbhMJh4)

---

## **3. Sentence Context and LSTMs**

🎥 [Udacity, Video Link](https://youtu.be/s9adl_Qehx8)
Simple RNNs are not always enough when working with text data. Longer sequences, such as a paragraph, often are difficult to handle, as the simple RNN structure loses information about previous inputs fairly quickly.

Long Short-Term Memory models, or LSTMs, help resolve this by keeping a “cell state” across time. These include a “forget gate”, where the cell can choose whether to keep or forget certain words to carry forward in the sequence.

Another interesting aspect of LSTMs is that they can be bidirectional, meaning that information can be passed both forward (later in the text sequence) and backward (earlier in the text sequence).

LSTMs either pass forward or forget certain information in each node.
LSTMs either pass forward or forget certain information in each node.

QUIZ QUESTION
For which of the following sentences would an LSTM likely perform better on than a vanilla RNN, if it needed to figure out both the sentiment, and the subject of that sentiment?

While watching the movie, I could only think of how terrible it was.

Once again, we’ve included some additional videos below that are optional for you to view and include a deeper dive into LSTMs. They come from Udacity’s Deep Learning Nanodegree program, and concern both LSTM basics and architectures.

(Optional) LSTM Basics

(Optional) LSTM Architectures

---

## **4. Constructing LSTMs in Code**

🎥 [Udacity, Video Link](https://youtu.be/A7mMGreNu6Q)
The code for an LSTM layer itself is just the LSTM layer from tf.keras.layers, with the number of LSTM cells to use. However, this is typically wrapped within a Bidirectional layer to make use of passing information both forward and backward in the network, as we noted on the previous page.

# A bidirectional LSTM layer with 64 nodes

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
One thing to note when using a Bidirectional layer is when you look at the model summary, if you put in 64 LSTM nodes, you will actually see a layer shape with 128 nodes (64x2).

No Need to Flatten
Unlike our more vanilla neural networks in the last lesson, you no longer need to use Flatten or GlobalAveragePooling1D after the LSTM layer - the LSTM can take the output of an Embedding layer and directly hook up to a fully-connected Dense layer with its own output.

Doubling Up
You can also feed an LSTM layer into another LSTM layer. To do so, on top of just stacking them in order when you create the model, you also need to set return_sequences to True for the earlier LSTM layer - otherwise, as noted above, the output will be ready for fully-connected layers and not be in the sequence format the LSTM layer expects.

# Two bidirectional LSTM layers with 64 nodes each

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True)
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
QUIZ QUESTION
Which of the below is valid TensorFlow code for a LSTM layer?

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

Additional Research
You can check out the TensorFlow 2.0 documentation for LSTM layers here.

## **5. Colab: Constructing LSTMs in Code**

🎥 [Udacity, Video Link](https://youtu.be/u4yuL-YRUe0)

Colab: Constructing LSTMs in Code
Colab Notebook
To access the Colab Notebook, login to your Google account and click on the link below:

Using LSTMs with Subwords

Walkthrough

So far we’ve looked at the performance of a single LSTM layer on the subwords dataset, so now let’s look at using multiple LSTM layers.

We’ll leave it for you to check how the different models do on these new reviews in the Colab.

One last thing we’ll look at is actually back at the dataset itself. This is similar to what we’ve done in looking at the embeddings visualizations in the earlier lesson from a conceptual standpoint, but might be easier to contextualize when you can see the full sentences.

---

## **6. LSTMs vs. Convolutions vs. GRUs**

🎥 [Udacity, Video Link](https://youtu.be/w_SCL4GeZoY)

Convolutional Layers for Text
Just like you did with images, you can also use convolutional layers on text, where the convolution occurs across a sequence of words instead of across an image.

To use a convolutional layer on text inputs, you can place a Conv1D layer directly after the Embedding layer:

# A 1D Convolutional layer with 128 filters and 5 words per filter

tf.keras.layers.Conv1D(128, 5, activation=’relu’)
Note that you will need to use Flatten or GlobalAveragePooling1D on the output of this layer to connect to any fully-connected layers from there.

GRUs
Gated Recurrent Units, or GRUs, have “update” and “reset” gates. These gates decide what to keep and what to throw away. They do not have a “cell state” like LSTMs do.

The code for these is very similar to an LSTM, where the GRU layer is wrapped in a Bidirectional layer.

# A bidirectional GRU layer with 32 nodes

tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))
Comparing Training Amongst Layers
Next up, let’s take a quick look at how each of these models performed during training, and the total duration of training.

An important item to note here is these performance differences (with perhaps the exception of training duration) will vary depending on the dataset and other changes to the model - you shouldn’t always assume one type of model will work better than another.

TensorFlow Documentation
Let’s take a quick look at the documentation for both GRU and LSTM layers in TensorFlow (also linked at the bottom of the page) before we move on to the Colab.

Further Research: TensorFlow Documentation
Documentation for GRU layers in TensorFlow
Documentation for LSTM layers in TensorFlow
Documentation for 1D Convolutional layers in TensorFlow
QUIZ QUESTION
Match each network type to its related functionality.

Submit to check your answer choices!
FUNCTIONALITY

NETWORK TYPE

Utilizes “update” and “reset” gates, where the “update” gate determines updates to the existing stored knowledge, and the reset gate determines how much to forget in the existing stored knowledge.

Utilizes “forget” and “learn” gates that feed to “remember” and “use” gates, where remembering is for further storage for the next input, and using is for generating the current output.

Utilizes “filters” that can slide over multiple words at a time and extract features from those sequences of words. Can be used for purposes other than a recurrent neural network.

Further Research: GRUs
Check out this blog post if you want to learn more on GRUs!

---

## **7. Colab: LSTMs vs. Convolutions vs. GRUs**

🎥 [Udacity, Video Link](https://youtu.be/BM4FUFfF9xQ)
Colab Notebook
To access the Colab Notebook, login to your Google account and click on the link below:

Using CNNs, GRUs and LSTMs on a Larger Dataset

Walkthrough

We’ve trained all the different models, so let’s look at how they perform on some new reviews next. You can skip this video if you just want to compare these on your own.

---

## **8. Text Generation**

🎥 [Udacity, Video Link](https://youtu.be/RGDiKhFcs0M)
Text generation can be done through simply predicting the next most likely word, given an input sequence. This can be done over and over by feeding the original input sequence, plus the newly predicted end word, as the next input sequence to the model. As such, the full output generated from a very short original input can effectively go on however long you want it to be.

The only real change to the network here is the output layer will now be equivalent to a node per each possible new word to generate - so, if you have 1,000 possible words in your corpus, you’d have an output array of length 1,000. You’ll also need to change the loss function from binary cross-entropy to categorical cross entropy - before, we had only a 0 or 1 as output, now there are potentially thousands of output “classes” (each possible word).

The example from the video with “Welcome to the…” as the input, with different probabilities per word as the output. Text Generation takes an input and outputs probabilities for the next most probable word.
Text Generation takes an input and outputs probabilities for the next most probable word.

QUIZ QUESTION
Match the below sentence fragments with their most likely next word.

SENTENCE FRAGMENT

LIKELIEST NEXT WORD

The dog ran across the \_\_.

The student opened their \_\_.

In front of my house, I got into the \_\_.

An Example of Text Generation
Botnik Studios used a text generation model to write a "new" chapter of a certain boy wizard's story - check it out!

---

## **9. Constructing a Text Generation Model**

🎥 [Udacity, Video Link](https://youtu.be/V5Mn2WL_a50)
As noted before, there are hardly any differences in the model itself, other than changing the number of nodes in the output layer and changing the loss function.

The more obvious changes come in working with the input and output data. The input data takes chunks of sequences and just splits off the final word as its label. So, if we had the sentence “I went to the beach with my dog”, and we had a max input length of five words, we’d get:

Input: I went to the beach

Label: with

Now, that’s not the only sequence that will come from the sentence! We would also get:

Input: went to the beach with

Label: my

And:

Input: to the beach with my

Label: dog

That’s how the N-Grams used in the pre-processing work - a single input sequence might actually become a series of sequences and labels.

With the output of the network, I’ll let you mostly investigate that code in the Colab on the next page, but the important thing there is that you can keep looping and creating more text by just appending the next word onto the previous input sequence.

---

## **10. Colab: Constructing a Text Generation Model**

🎥 [Udacity, Video Link]()
Colab Notebook
To access the Colab Notebook, login to your Google account and click on the link below:

Constructing a Text Generation Model

---

## **11. Optimizing the Text Generation Model**

🎥 [Udacity, Video Link](https://youtu.be/beOiP__yLXs)
There are a number of ways you might improve your text generation models:

Using more data?
You’ll need to consider memory and output size constraints
Also consider using only top-k most common words
Know your data - songs have many more words than a Tweet
Keep tuning your model
Add/subtract from layer sizes or embedding dimensions
Use np.random.choice with the probabilities for more variance in predicted outputs
Optimizing the Text Generation Model Quiz
What ideas do you have for utilizing text generation networks?

Enter your response here, there's no right or wrong answer

---

## **12. Colab: Optimizing the Text Generation Model**

🎥 [Udacity, Video Link](https://youtu.be/-ZT-Kq1Vwrg)

---

## **13. Lesson Conclusion**

🎥 [Udacity, Video Link](https://youtu.be/M0Mz93DDHjo)

- You’ve reached the end of your journey with us into the basics of Natural Language Processing with TensorFlow!

- You’ve learned an astounding amount in a short period of time, going from Tokenization and Embeddings in the last lesson, to working with Recurrent Neural Networks and Text Generation in this one.

- We’re excited to see what you build - good luck and have fun!

---

## Resources

---

## Foam Related Links

- [[_nlp]]
