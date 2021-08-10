# NLP

## [NLP: Tokenization and Embeddings, Lesson 9, Udacity, UD187, Intro to TensorFlow for Deep Learning](https://classroom.udacity.com/courses/ud187/lessons/a5e9e6cc-e286-430f-aaa9-735c014ee950/concepts/947fcc1e-ee5c-46f6-a98c-88846880c975)

---

## **1. Meet Your Instructors**

🎥 [Udacity, Video Link](https://youtu.be/vSFJ3mDpgrI)

- We’re happy to have you here to learn about Natural Language Processing with TensorFlow!

- Jocelyn is a Developer Advocate for the TensorFlow team at Google, and started her tech career in AI.

- Michael is a Curriculum Manager at Udacity, and first began using TensorFlow way back before v1 was released.

---

## **2. Introduction to Natural Language Processing**

🎥 [Udacity, Video Link](https://youtu.be/n8HmBdOOvhY)

- Natural Language Processing, or NLP for short, focuses on analyzing text and speech data. This can range from simple recognition (what words are in the given text/speech), to sentiment analysis (was a review positive or negative), and all the way to areas like text generation (creating novel song lyrics from scratch).

- We’ll focus only on text in these lessons and not speech, but many of the same principles apply.

- NLP got its start mostly on machine translation, where users often had to create strict, manual rules to go from one language to another. It has since morphed to be more machine learning-based, reliant on much larger datasets than the early methods were.

---

## **3. Lesson Outline**

🎥 [Udacity, Video Link](https://youtu.be/uOB5jKoO1No)

- In the first lesson on Natural Language Processing with TensorFlow, we’ll focus on Tokenization and Embeddings, which will help convert input text into useful data for input into the neural network layers you’ve seen before.

- In the second lesson, we’ll dive into Recurrent Neural Networks (such as the LSTMs you saw in the Time Series Analysis lesson) as well as Text Generation, which allows for the creation of new text.

---

## **4. Tokenizing Text**

🎥 [Udacity, Video Link](https://youtu.be/7u_ZUlh4gu0)

- Neural networks utilize numbers as their inputs, so we need to convert our input text into numbers. Tokenization is the process of assigning numbers to our inputs, but there is more than one way to do this - should each letter have its own numerical token, each word, phrase, etc.

- As you saw in the video, tokenizing based on letters with our current neural networks doesn’t always work so well - anagrams, for instance, may be made up of the same letters but have vastly different meanings. So, in our case, we’ll start by tokenizing each individual word.

### Tokenizer

- With TensorFlow, this is done easily through use of a `Tokenizer`, found within `tf.keras.preprocessing.text`. If you wanted only the first 10 most common words, you could initialize it like so:

```python
tokenizer = Tokenizer(num_words=10)
```

### Fit on Texts

- Then, to fit the tokenizer to your inputs (in the below case a list of strings called sentences), you use .fit_on_texts():

```python
tokenizer.fit_on_texts(sentences)
```

### Text to Sequences

- From there, you can use the tokenizer to convert sentences into tokenized sequences:

```python
tokenizer.texts_to_sequences(sentences)
```

### Out of Vocabulary Words

- However, new sentences may have new words that the tokenizer was not fit on. By default, the tokenizer will just ignore these words and not include them in the tokenized sequences. However, you can also add an “out of vocabulary”, or OOV, token to represent these words. This has to be specified when originally creating the `Tokenizer` object.

```python
tokenizer = Tokenizer(num_words=20, oov_token=’OOV’)
```

### Viewing the Word Index

- Lastly, if you want to see how the tokenizer has mapped numbers to words, use the tokenizer.word_index property to see this mapping.

### QUESTION 1 OF 2

- Match the different words below that would have the exact same letter-based embeddings. As a hint, look for words that are anagrams.

WORD 1

WORD 2

silent

burned

infests

looped

### QUESTION 2 OF 2

- Which of the following would be an appropriate tokenization of the below two sentences?

- The person ran quickly.

- The car drove quickly to the bank.

### Further Research

- Many NLP models get trained on very large text corpuses to avoid having too many OOV words. Below are a couple of great resources for finding datasets that you might find useful on your NLP journey:

- A [popular Github repo](https://github.com/niderhoff/nlp-datasets) for NLP datasets
- Google's newly public [dataset search](https://datasetsearch.research.google.com/)

---

## **5. Colab: Tokenizing Text**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Tokenizing Text](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c01_nlp_turn_words_into_tokens.ipynb)

---

## **6. Text to Sequences**

🎥 [Udacity, Video Link](https://youtu.be/bn_ou4GPkB4)

- Even after converting sentences to numerical values, there’s still an issue of providing equal length inputs to our neural networks - not every sentence will be the same length!

- There’s two main ways you can process the input sentences to achieve this - padding the shorter sentences with zeroes, and truncating some of the longer sequences to be shorter. In fact, you’ll likely use some combination of these.

- With TensorFlow, the `pad_sequences` function from `tf.keras.preprocessing.sequence` can be used for both of these tasks. Given a list of sequences, you can specify a `maxlen` (where any sequences longer than that will be cut shorter), as well as whether to pad and truncate from either the beginning or ending, depending on `pre` or `post` settings for the `padding` and `truncating` arguments. By default, padding and truncation will happen from the beginning of the sequence, so set these to `post` if you want it to occur at the end of the sequence.

- If you wanted to pad and truncate from the beginning, you could use the following:

```python
padded = pad_sequences(sequences, maxlen=10)
```

### Further Research

- Head here if you’d like to check out the full TensorFlow documentation for `pad_sequences`.

---

## **7. Colab: Preparing Text to Use with TensorFlow Models**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Preparing Text to Use with TensorFlow Models](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c02_nlp_padding.ipynb)

🎥 [Udacity, Video Link](https://youtu.be/kkdoQx8S_sQ)

---

## **8. Tokenization of Large Datasets**

🎥 [Udacity, Video Link](https://youtu.be/hPLWdUmtuwM)

- Everything you have learned previously applies fairly similarly to larger datasets. In many cases, you’ll want to be even more focused on the total number of words used with the Tokenizer, as well as understanding the right sequence length to create from `pad_sequences`.

- In the upcoming Colab, we’ll use portions of a [Sentiment Analysis Dataset on Kaggle](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set) that contains both Amazon product and Yelp restaurant reviews.

### QUIZ QUESTION

- When specifying num_words in the Tokenizer, if set at 1,000:

- Only the most common 1,000 words are kept

### Further Research

- Once again, it can be useful to check out Google’s Dataset Search to find large datasets to work with, and Kaggle also has a wide variety of datasets available for use.

;

---

## **9. Colab: Tokenization of Large Datasets**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Tokenize and Sequence a Larger Corpus of Text](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c03_nlp_prepare_larger_text_corpus.ipynb)

🎥 [Udacity, Video Link](https://youtu.be/zcT84FBe8Rc)

---

## **10. Word Embeddings**

🎥 [Udacity, Video Link](https://youtu.be/xQzKJgNQRK0)

- Embeddings are clusters of vectors in multi-dimensional space, where each vector represents a given word in those dimensions. While it’s difficult for us humans to think in many dimensions, luckily the TensorFlow Projector makes it fairly easy for us to view these clusters in a 3D projection (later Colabs will generate the necessary files for use with the projection tool).

- This can be very useful for sentiment analysis models, where you’d expect to see clusters around either more positive or more negative sentiment associated with each word.

- An example embedding projection, post-training. Negative sentiment words are separated quite distinctly from positive sentiment words, such as “incredible”.
- An example of a post-training embedding projection, with clear distinctions between positive and negative sentiments.

### QUIZ QUESTION

Match the below words to their most likely sentiments.

WORD
Amazing
Dog
Terrible

SENTIMENT

### Further Research

- If you want to learn more on why word embeddings are used in NLP, check out [this useful post](https://towardsdatascience.com/why-do-we-use-embeddings-in-nlp-2f20e1b632d2).

---

## **11. Building a Basic Sentiment Model**

🎥 [Udacity, Video Link](https://youtu.be/-g5Tqsna8yE)

- To create our embeddings, we’ll first use an embeddings layer, called [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding). This takes three arguments: the size of the tokenized vocabulary, the number of embedding dimensions to use, as well as the input length (from when you standardized sequence length with padding and truncation).

- The output of this layer needs to be reshaped to work with any fully-connected layers. You can do this with a pure `Flatten` layer, or use `GlobalAveragePooling1D` for a little additional computation that sometimes creates better results.

- In our case, we’re only looking at positive vs. negative sentiment, so only a single output node is needed (0 for negative, 1 for positive). You’ll be able to use a binary cross entropy loss function since the result is only binary classification.

### QUIZ QUESTION

- Given a vocabulary size of 500, maximum sequence length of 50, and embedding dimension of 16, what is the output shape of the `Embedding` layer?

### A Note on Embedding Networks

- The TensorFlow team has two additional suggestions for you, on top of what we show in the video and what we included in the Colab coming up.

- They suggest that the final network does not use a sigmoid activation layer when working with embeddings, especially when using just the two classes like we are for sentiment analysis:

```python
tf.keras.layers.Dense(1)
```

- Additionally, they suggest instead of using the string `”binary_crossentropy”` as the loss function, you use `tf.keras.losses.BinaryCrossentropy(from_logits=True)`.

---

## **12. Visualizing Embeddings**

🎥 [Udacity, Video Link](https://youtu.be/PgWGqesxH5U)

- We’ve given you the code to create the files for input into the projector. This will download two files: 1) the vectors, and 2) the metadata.

- The projector will already come with a pre-loaded visualization, so you’ll need to use the “Load” button on the left and upload each of the two files. In some cases, there may be a small difference in the number of tensors present in the vector file and the metadata file (usually with a message appearing after uploading the metadata); if this appears, wait for a few seconds for the error message to disappear, and then click outside the window. Typically, the visualization will still load just fine.

- Make sure to click the checkbox for “Sphereize data”, which will better show whether there is separation between positive and negative sentiment (or not).

### Visualizing Embeddings Quiz

- In your own words, why might being able to visualize word embeddings be useful?

- Enter your response here, there's no right or wrong answer

---

## **13. Colab: Word Embeddings and Sentiment**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Word Embeddings and Sentiment](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c04_nlp_embeddings_and_sentiment.ipynb)

---

## **14. Tweaking the Model**

🎥 [Udacity, Video Link](https://youtu.be/pW8db9sLRbM)

There are a number of ways in which you might improve the sentiment analysis model we’ve built already:

- Data and preprocessing-based approaches

  - More data
  - Adjusting vocabulary size (make sure to consider the overall size of the corpus!)
  - Adjusting sequence length (more or less padding or truncation)
  - Whether to pad or truncate pre or post (usually less of an effect than the others)

- Model-based approaches

  - Adjust the number of embedding dimensions
  - Changing use of Flatten vs. GlobalAveragePooling1D
  - Considering other layers like Dropout
  - Adjusting the number of nodes in intermediate fully-connected layers

- These are just some of the potential things you might tweak to better predict sentiment from text.

### Tweaking the Model Quiz

- Even with these potential tweaks, do you think there is anything the models still are missing that could improve performance?

- Enter your response here, there's no right or wrong answer

---

## **15. Colab: Tweaking the Model**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Tweaking the Model](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c05_nlp_tweaking_the_model.ipynb)

---

## **16. What's in a (sub)word?**

🎥 [Udacity, Video Link](https://youtu.be/A_F5ZcQzid0)

- We’ve worked with full words before for our sentiment models, and Jocelyn had shown us some issues right at the start of the lesson when using character-based tokenization. Subwords are another approach, where individual words are broken up into the more commonly appearing pieces of themselves. This helps avoid marking very rare words as OOV when you use only the most common words in a corpus.

- As shown in the video, this can further expose an issue affecting all of our models up to this point, in that they don’t understand the full context of the sequence of words in an input. The next lesson on recurrent neural networks will help address this issue.

- The example subwords from the video breaking out Decent, Decadent and Decay.
- Our example subwords using Decent, Decadent and Decay.

### Subword Datasets

- There are a number of already created subwords datasets available online. If you check out the IMDB dataset on TFDS, for instance, by scrolling down you can see datasets with both 8,000 subwords as well as 32,000 subwords in a corpus (along with regular full-word datasets).

- However, I want you to know how to create these yourself as well, so we’ll use TensorFlow’s SubwordTextEncoder and its build_from_corpus function to create one from the reviews dataset we used previously.

### QUIZ QUESTION

- Subwords are:
- Pieces of words, often made up of smaller words, that make words with similar roots be tokenized more similarly.

### Further Research

- If you’re interested in more work with subwords, there’s an unofficial Google repository called SentencePiece that contains some interesting techniques for improved creation of the subwords dataset from an original text dataset.

---

## **17. Colab: What's in a (sub)word?**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [What’s in a subword?](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c06_nlp_subwords.ipynb)

🎥 [Udacity, Video Link](https://youtu.be/8ARUNInoDrk)

---

## **18. Lesson Conclusion**

🎥 [Udacity, Video Link](https://youtu.be/rwIF6t4CFFg)

- You’ve already learned an amazing amount of material on Natural Language Processing with TensorFlow in this lesson.

- You started with Tokenization by:

  - Tokenizing input text
  - Creating and padding sequences
  - Incorporating out of vocabulary words
  - Generalizing tokenization and sequence methods to real world datasets

- From there, you moved onto Embeddings, where you:

  - transformed tokenized sequences into embeddings
  - developed a basic sentiment analysis model
  - visualized the embeddings vector
  - tweaked hyperparameters of the model to improve it
  - and diagnosed potential issues with using pre-trained subword tokenizers when the network doesn’t have sequence context

- In the next lesson, you’ll dive into Recurrent Neural Networks, which will be able to understand the sequence of inputs, and you'll learn how to generate new text.

---

## Foam Related Links

-[[_nns]]
