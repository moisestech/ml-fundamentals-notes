# Decision Trees

## [Decision Trees, Lesson 4, Udacity, UD120](https://classroom.udacity.com/courses/ud120/lessons/2258728540/concepts/29875885970923)

## **1. Welcome To Decision Trees**

🎥 [Udacity, Video Link](https://youtu.be/5eAHVk1-Hz0)

---

## **2. Linearly Separable Data**

🎥 [Udacity, Video Link](https://youtu.be/lCWGV6ZuXt0)

---

## **3. Multiple Linear Questions**

🎥 [Udacity, Video Link](https://youtu.be/t1Y-nzgI1L4)

---

## **4. Constructing a Decision Tree First Split**

🎥 [Udacity, Video Link](https://youtu.be/iZYv1WdWwQo)

---

## **5. Constructing a Decision Tree 2nd Split**

🎥 [Udacity, Video Link](https://youtu.be/U2yZxIeG2t0)

---

## **6. Class Labels After Second Split**

🎥 [Udacity, Video Link](https://youtu.be/-3VPMBIwTtE)

---

## **7. Constructing A Decision Tree/Third Split**

🎥 [Udacity, Video Link](https://youtu.be/-3VPMBIwTtE)

---

## **8. Coding A Decision Tree**

🎥 [Udacity, Video Link](https://youtu.be/YaZu4waSryo)

---

## **9. Decision Tree Accuracy**

🎥 [Udacity, Video Link](https://youtu.be/i7pRvuVoWg0)

---

## **10. Decision Tree Parameters**

🎥 [Udacity, Video Link](https://youtu.be/jkJ4dbbpVCQ)

---

## **11. Min Samples Split**

🎥 [Udacity, Video Link](https://youtu.be/Mt5TWGYacJs)

---

## **12. Decision Tree Accuracy**

🎥 [Udacity, Video Link](https://youtu.be/1z5mVNdF1KA)

---

## **13. Data Impurity and Entropy**

🎥 [Udacity, Video Link](https://youtu.be/Bd15qhUrKCI)

---

## **14. Minimizing Impurity in Splitting**

🎥 [Udacity, Video Link](https://youtu.be/L6J6BRFgDiIs)

---

## **15. Formula of Entropy**

🎥 [Udacity, Video Link](https://youtu.be/NHAatuG0T3Q)

---

## **16. Entropy Calculation Part 1**

🎥 [Udacity, Video Link](https://youtu.be/K-rQ8KnmmH8)

---

## **17. Entropy Calculation Part 2**

🎥 [Udacity, Video Link](https://youtu.be/GtiLFC7EgFE)

---

## **18. Entropy Calculation Part 3**

🎥 [Udacity, Video Link](https://youtu.be/M2Sp-Y2a71c)

---

## **19. Entropy Calculation Part 4**

🎥 [Udacity, Video Link](https://youtu.be/V0FNwMKhIVM)

---

## **20. Entropy Calculation Part 5**

🎥 [Udacity, Video Link](https://youtu.be/ZSkYbBsFuOQ)

---

## **21. Information Gain**

🎥 [Udacity, Video Link](https://youtu.be/KYieR9y-ue4)

---

## **22. Information Gain Calculation Part 1**

🎥 [Udacity, Video Link](https://youtu.be/daVA3PI2E6o)

---

## **23. Information Gain Calculation Part 2**

🎥 [Udacity, Video Link](https://youtu.be/t4qaavAslSw)

---

## **24. Information Gain Calculation Part 3**

🎥 [Udacity, Video Link](https://youtu.be/s_-I8mbrfw0)

---

## **25. Information Gain Calculation Part 4**

🎥 [Udacity, Video Link](https://youtu.be/j0uDMc3Yrlo)

---

## **26. Information Gain Calculation Part 5**

🎥 [Udacity, Video Link](https://youtu.be/3jfQlMLyH2o)

---

## **27. Information Gain Calculation Part 6**

🎥 [Udacity, Video Link](https://youtu.be/qnfVoUChRlQ)

---

## **28. Information Gain Calculation Part 7**

🎥 [Udacity, Video Link](https://youtu.be/EDFp4wU5BMo)

---

## **29. Information Gain Calculation Part 8**

🎥 [Udacity, Video Link](https://youtu.be/F-xSYJ3y_pA)

---

## **30. Information Gain Calculation Part 9**

🎥 [Udacity, Video Link](https://youtu.be/PDqyWzZCVBY)

---

## **31. Information Gain Calculation Part 10**

🎥 [Udacity, Video Link](https://youtu.be/XYHTuv2FpWQ)

---

## **32. Tuning Criterion Parameter**

🎥 [Udacity, Video Link](https://youtu.be/V80QLNK5fFQ)

---

## **33. Bias-Variance Dilemma**

🎥 [Udacity, Video Link](https://youtu.be/W5uUYnSHDhM)

---

## **34. DT Strengths and Weaknesses**

🎥 [Udacity, Video Link](https://youtu.be/KGnhg76iRfI)

---

## **35. Decision Tree Mini-Project Video**

🎥 [Udacity, Video Link](https://youtu.be/eFjeXiC0KHA)

---

## **36. Decision Tree Mini-Project**

In this project, we will again try to identify the authors in a body of emails, this time using a decision tree. The starter code is in decision_tree/dt_author_id.py.

Get the data for this mini project from here.

Once again, you'll do the mini-project on your own computer and enter your answers in the web browser. You can find the instructions for the decision tree mini-project here.

---

## **37. Your First Email DT: Accuracy**

Using the starter code in decision_tree/dt_author_id.py, get a decision tree up and running as a classifier, setting min_samples_split=40. It will probably take a while to train. What’s the accuracy?

Start Quiz

---

## **38. Speeding Up Via Feature Selection 1**

You found in the SVM mini-project that the parameter tune can significantly speed up the training time of a machine learning algorithm. A general rule is that the parameters can tune the complexity of the algorithm, with more complex algorithms generally running more slowly.

Another way to control the complexity of an algorithm is via the number of features that you use in training/testing. The more features the algorithm has available, the more potential there is for a complex fit. We will explore this in detail in the “Feature Selection” lesson, but you’ll get a sneak preview now.

What's the number of features in your data? (Hint: the data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features; so to extract this number, use a line of code like len(features_train[0]).)

Start Quiz

---

## **39. Changing the Number of Features**

go into ../tools/email_preprocess.py, and find the line of code that looks like this:

selector = SelectPercentile(f_classif, percentile=10)

Change percentile from 10 to 1, and rerun dt_author_id.py. What’s the number of features now?

Start Quiz

---

## **40. SelectPercentile and Complexity**

What do you think SelectPercentile is doing? Would a large value for percentile lead to a more complex or less complex decision tree, all other things being equal? Note the difference in training time depending on the number of features.

Start Quiz

---

## **41. Accuracy Using 1% of Features**

What's the accuracy of your decision tree when you use only 1% of your available features (i.e. percentile=1)?

Start Quiz

---

## Foam Related Links

- **[[supervised]]**
