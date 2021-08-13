# SVM

## [Support Vector Machine, Lesson 3, Udacity, UD120](https://classroom.udacity.com/courses/ud120/lessons/2252188570/concepts/23799685490923)

## **1. Welcome to SVM**

🎥 [Udacity, Video Link](https://youtu.be/gnAmmyQ_ZcQ)

---

## **2. Separating Line**

🎥 [Udacity, Video Link](https://youtu.be/mzKPXz-Yhwk)

---

## **3. Choosing Between Separating Lines**

🎥 [Udacity, Video Link](https://youtu.be/swoZxkrxIB0)

---

## **4. What Makes A Good Separating Line**

🎥 [Udacity, Video Link](https://youtu.be/cwjvMYPB1Fk)

---

## **5. Practice with Margins**

🎥 [Udacity, Video Link](https://youtu.be/l3zXhTxQiTs)

---

## **6. SVMs and Tricky Data Distributions**

🎥 [Udacity, Video Link](https://youtu.be/wbCq7wm81BU)

---

## **7. SVM Response to Outliers**

🎥 [Udacity, Video Link](https://youtu.be/w-czJptEyBk)

---

## **8. SVM Outlier Practice**

🎥 [Udacity, Video Link](https://youtu.be/WxAO6ByCvew)

---

## **9. Handoff to Katie**

🎥 [Udacity, Video Link](https://youtu.be/GkqOdgZnkig)

---

## **10. SVM in SKlearn**

🎥 [Udacity, Video Link](https://youtu.be/R7xQtQzkvTk)

---

## **11. SVM Decision Boundary**

🎥 [Udacity, Video Link](https://youtu.be/R7xQtQzkvTk)

---

## **12. Coding Up the SVM**

🎥 [Udacity, Video Link](https://youtu.be/CvTXyvw7QLc)

---

## **13. Nonlinear SVMs**

🎥 [Udacity, Video Link](https://youtu.be/6UgInp_gf1w)

---

## **14. Nonlinear Data**

🎥 [Udacity, Video Link](https://youtu.be/EllzeBecnkU)

---

## **15. A New Feature**

🎥 [Udacity, Video Link](https://youtu.be/8xFV-I4VqZ0)

---

## **16. Visualizing the New Feature**

🎥 [Udacity, Video Link](https://youtu.be/sAdM20gFi2M)

---

## **17. Separating with the New Feature**

🎥 [Udacity, Video Link](https://youtu.be/9KAHkienFWk)

---

## **18. Practice Making a New Feature**

🎥 [Udacity, Video Link](https://youtu.be/ygveMIhCtDg)

---

## **19. Kernel Trick**

🎥 [Udacity, Video Link](https://youtu.be/3Xw6FKYP7e4)

---

## **20. Playing Around with Kernel Choices**

🎥 [Udacity, Video Link](https://youtu.be/krV6r7HxmZU)

---

## **21. Kernel and Gamma**

🎥 [Udacity, Video Link](https://youtu.be/pH51jLfGXe0)

---

## **22. SVM C Parameter**

🎥 [Udacity, Video Link](https://youtu.be/joTa_FeMZ2s)

---

## **23. SVM Gamma Parameter**

🎥 [Udacity, Video Link](https://youtu.be/m2a2K4lprQw)

---

## **24. Overfitting**

🎥 [Udacity, Video Link](https://youtu.be/CxAxRCv9WoA)

---

## **25. SVM Strengths and Weaknesses**

🎥 [Udacity, Video Link](https://youtu.be/U9-ZsbaaGAs)

---

## **26. SVM Mini-Project Video**

🎥 [Udacity, Video Link](https://youtu.be/mENzEtsiOmI)

---

## **27. SVM Mini-Project**

In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. This project also gives us a chance to play around with parameters a lot more than Naive Bayes did, so we will do that too.

You can find the code here.

---

## **28. SVM Author ID Accuracy**

Go to the svm directory to find the starter code (svm/svm_author_id.py).

Import, create, train and make predictions with the sklearn SVC classifier. When creating the classifier, use a linear kernel (if you forget this step, you will be unpleasantly surprised by how long the classifier takes to train). What is the accuracy of the classifier?

Start Quiz

---

## **29. SVM Author ID Timing**

Place timing code around the fit and predict functions, like you did in the Naive Bayes mini-project. How do the training and prediction times compare to Naive Bayes?

Start Quiz

---

## **30. A Smaller Training Set**

One way to speed up an algorithm is to train it on a smaller training dataset. The tradeoff is that the accuracy almost always goes down when you do this. Let’s explore this more concretely: add in the following two lines immediately before training your classifier.

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data. You can leave all other code unchanged. What’s the accuracy now?

Start Quiz

---

## **31. Speed-Accuracy Tradeoff**

If speed is a major consideration (and for many real-time machine learning applications, it certainly is) then you may want to sacrifice a bit of accuracy if it means you can train/predict faster. Which of these are applications where you can imagine a very quick-running algorithm is especially important?

predicting the author of an email
flagging credit card fraud, and blocking a transaction before it goes through
voice recognition, like Siri
Start Quiz

---

## **32. Deploy an RBF Kernel**

Keep the training set slice code from the last quiz, so that you are still training on only 1% of the full training set. Change the kernel of your SVM to “rbf”. What’s the accuracy now, with this more complex kernel?

Start Quiz

---

## **33. Optimize C Parameter**

Keep the training set size and rbf kernel from the last quiz, but try several values of C (say, 10.0, 100., 1000., and 10000.). Which one gives the best accuracy?

Start Quiz

---

## **34. Accuracy after Optimizing C**

Once you've optimized the C value for your RBF kernel, what accuracy does it give? Does this C value correspond to a simpler or more complex decision boundary?

(If you're not sure about the complexity, go back a few videos to the "SVM C Parameter" part of the lesson. The result that you found there is also applicable here, even though it's now much harder or even impossible to draw the decision boundary in a simple scatterplot.)

Start Quiz

---

## **35. Optimized RBF vs Linear SVM: Accuracy**

Now that you’ve optimized C for the RBF kernel, go back to using the full training set. In general, having a larger training set will improve the performance of your algorithm, so (by tuning C and training on a large dataset) we should get a fairly optimized result. What is the accuracy of the optimized SVM?

Start Quiz

---

## **36. Extracting Predictions from an SVM**

What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) predict for element 10 of the test set? The 26th? The 50th? (Use the RBF kernel, C=10000, and 1% of the training set. Normally you'd get the best results using the full training set, but we found that using 1% sped up the computation considerably and did not change our results--so feel free to use that shortcut here.)

And just to be clear, the data point numbers that we give here (10, 26, 50) assume a zero-indexed list. So the correct answer for element #100 would be found using something like answer=predictions[100]

Start Quiz
Just to be clear, the data point numbers that we give here (10, 26, 50) assume a zero-indexed list. So the correct answer for element #100 would be found using something like answer=predictions[100]

---

## **37. How Many Chris Emails Predicted?**

There are over 1700 test events--how many are predicted to be in the “Chris” (1) class? (Use the RBF kernel, C=10000., and the full training set.)

Start Quiz

---

## **38. Final Thoughts on Deploying SVMs**

Hopefully it’s becoming clearer what Sebastian meant when he said Naive Bayes is great for text--it’s faster and generally gives better performance than an SVM for this particular problem. Of course, there are plenty of other problems where an SVM might work better. Knowing which one to try when you’re tackling a problem for the first time is part of the art and science of machine learning. In addition to picking your algorithm, depending on which one you try, there are parameter tunes to worry about as well, and the possibility of overfitting (especially if you don’t have lots of training data).

Our general suggestion is to try a few different algorithms for each problem. Tuning the parameters can be a lot of work, but just sit tight for now--toward the end of the class we will introduce you to GridCV, a great sklearn tool that can find an optimal parameter tune almost automatically.

---

## Foam Related Links

- **[[supervised]]**
