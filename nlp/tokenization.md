# Tokenization

## [NLP: Tokenization and Embeddings, Lesson 9, Udacity, UD187, Intro to TensorFlow for Deep Learning](https://classroom.udacity.com/courses/ud187/lessons/a5e9e6cc-e286-430f-aaa9-735c014ee950/concepts/b829bd12-65dc-4f9f-9688-6b860e4f98aa)

---

## **1. Meet Your Instructors**

🎥 [Udacity, Video Link](https://youtu.be/vSFJ3mDpgrI)

---

## **2. Introduction to Natural Language Processing**

🎥 [Udacity, Video Link](https://youtu.be/n8HmBdOOvhY)

---

## **3. Lesson Outline**

🎥 [Udacity, Video Link](https://youtu.be/uOB5jKoO1No)

---

## **4. Tokenizing Text**

🎥 [Udacity, Video Link](https://youtu.be/7u_ZUlh4gu0)

---

## **5. Colab: Tokenizing Text**

Colab Notebook
To access the Colab Notebook, login to your Google account and click on the link below:

Tokenizing Text

---

## **6. Text to Sequences**

🎥 [Udacity, Video Link](https://youtu.be/bn_ou4GPkB4)

---

## **7. Colab: Preparing Text to Use with TensorFlow Models**

🎥 [Udacity, Video Link](https://youtu.be/kkdoQx8S_sQ)

---

## **8. Tokenization of Large Datasets**

🎥 [Udacity, Video Link](https://youtu.be/hPLWdUmtuwM)

---

## **9. Colab: Tokenization of Large Datasets**

🎥 [Udacity, Video Link](https://youtu.be/zcT84FBe8Rc)

---

## **10. Word Embeddings**

🎥 [Udacity, Video Link](https://youtu.be/xQzKJgNQRK0)

---

## **11. Building a Basic Sentiment Model**

🎥 [Udacity, Video Link](https://youtu.be/-g5Tqsna8yE)

---

## **12. Visualizing Embeddings**

🎥 [Udacity, Video Link](https://youtu.be/PgWGqesxH5U)

---

## **13. Colab: Word Embeddings and Sentiment**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Word Embeddings and Sentiment](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c04_nlp_embeddings_and_sentiment.ipynb)

---

## **14. Tweaking the Model**

🎥 [Udacity, Video Link](https://youtu.be/pW8db9sLRbM)

There are a number of ways in which you might improve the sentiment analysis model we’ve built already:

Data and preprocessing-based approaches
More data
Adjusting vocabulary size (make sure to consider the overall size of the corpus!)
Adjusting sequence length (more or less padding or truncation)
Whether to pad or truncate pre or post (usually less of an effect than the others)
Model-based approaches
Adjust the number of embedding dimensions
Changing use of Flatten vs. GlobalAveragePooling1D
Considering other layers like Dropout
Adjusting the number of nodes in intermediate fully-connected layers
These are just some of the potential things you might tweak to better predict sentiment from text.

Tweaking the Model Quiz
Even with these potential tweaks, do you think there is anything the models still are missing that could improve performance?

Enter your response here, there's no right or wrong answer

---

## **15. Colab: Tweaking the Model**

### Colab Notebook

- To access the Colab Notebook, login to your Google account and click on the link below:

- [Tweaking the Model](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c05_nlp_tweaking_the_model.ipynb)

---

## **16. What's in a (sub)word?**

🎥 [Udacity, Video Link](https://youtu.be/A_F5ZcQzid0)
We’ve worked with full words before for our sentiment models, and Jocelyn had shown us some issues right at the start of the lesson when using character-based tokenization. Subwords are another approach, where individual words are broken up into the more commonly appearing pieces of themselves. This helps avoid marking very rare words as OOV when you use only the most common words in a corpus.

As shown in the video, this can further expose an issue affecting all of our models up to this point, in that they don’t understand the full context of the sequence of words in an input. The next lesson on recurrent neural networks will help address this issue.

The example subwords from the video breaking out Decent, Decadent and Decay.
Our example subwords using Decent, Decadent and Decay.

Subword Datasets
There are a number of already created subwords datasets available online. If you check out the IMDB dataset on TFDS, for instance, by scrolling down you can see datasets with both 8,000 subwords as well as 32,000 subwords in a corpus (along with regular full-word datasets).

However, I want you to know how to create these yourself as well, so we’ll use TensorFlow’s SubwordTextEncoder and its build_from_corpus function to create one from the reviews dataset we used previously.

QUIZ QUESTION
Subwords are:

Pieces of words, often made up of smaller words, that make words with similar roots be tokenized more similarly.

Further Research
If you’re interested in more work with subwords, there’s an unofficial Google repository called SentencePiece that contains some interesting techniques for improved creation of the subwords dataset from an original text dataset.

---

## **17. Colab: What's in a (sub)word?**

🎥 [Udacity, Video Link](https://youtu.be/8ARUNInoDrk)

Colab Notebook
To access the Colab Notebook, login to your Google account and click on the link below:

[What’s in a subword?](https://colab.sandbox.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c06_nlp_subwords.ipynb)

---

## **18. Lesson Conclusion**

🎥 [Udacity, Video Link](https://youtu.be/rwIF6t4CFFg)
You’ve already learned an amazing amount of material on Natural Language Processing with TensorFlow in this lesson.

You started with Tokenization by:

Tokenizing input text
Creating and padding sequences
Incorporating out of vocabulary words
Generalizing tokenization and sequence methods to real world datasets
From there, you moved onto Embeddings, where you:

transformed tokenized sequences into embeddings
developed a basic sentiment analysis model
visualized the embeddings vector
tweaked hyperparameters of the model to improve it
and diagnosed potential issues with using pre-trained subword tokenizers when the network doesn’t have sequence context
In the next lesson, you’ll dive into Recurrent Neural Networks, which will be able to understand the sequence of inputs, and you'll learn how to generate new text.

---

## Foam Related Links

- [[_nlp]]
