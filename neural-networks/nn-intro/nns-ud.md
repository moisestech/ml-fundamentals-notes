# Intro NN UD188

## [Neural Networks, Lesson, Udacity, UD, Intro To Machine Learning]()

---

## **1. Introduction**

🎥 [Udacity, Video Link](https://youtu.be/tn-CrUTkCUc)

---

## **2. Classification Problems 1**

- We'll start by defining what we mean by classification problems, and applying it to a simple example.

🎥 [Udacity, Video Link](https://youtu.be/Dh625piH7Z0)

![]()

### 📝 QUIZ QUESTION 1

### _❓ Does the student get accepted?_

- ✅ **ANSWER: Yes**

---

## **3. Classification Problems 2**

🎥 [Udacity, Video Link](https://youtu.be/46PywnGa_cQ)

---

## **4. Linear Boundaries**

🎥 [Udacity, Video Link](https://youtu.be/X-uMlsBi07k)

### 📝 QUIZ QUESTION 2

### _❓ Now that you know the equation for the line `(2x1 + x2 - 18=0)`, and similarly the “score” `(2x1 + x2 - 18)`, what is the score of the student who got `7` in the test and 6 for grades?_

- **✅ ANSWER:** `2`

---

## **5. Higher Dimensions**

🎥 [Udacity, Video Link](https://youtu.be/eBHunImDmWw)

### 📝 QUIZ QUESTION 3

### ❓ _Given the table in the video above, what would the dimensions be for input **features** `(x)`, the **weights** `(W)`, and the **bias** `(b)` to **satisfy** `(Wx + b)`?_

- ✅ **ANSWER: `(1xn), x: (nx1), b: (1x1)`**

---

## **6. Perceptrons**

🎥 [Udacity, Video Link](https://youtu.be/hImSxZyRiOw)

### Corrections

- At `3:07` in the video, the title should read "Step Function", not "Set Function".
- At `3:07` in the video, the definition of the Step function should be:

`y=1 if x >= 0; y=0 if x<0`

### 📝 QUIZ QUESTION 4

### Given Score = `2*Test + 1*Grade - 18`, suppose `w1` was `1.5` instead of `2`.

### _❓ Would the student who got `7` on the test and `6` on the grades be accepted or rejected?_

- ✅ **ANSWER Rejected**

---

## **7. Why "Neural Networks"?**

🎥 [Udacity, Video Link](https://youtu.be/zAkzOZntK6Y)

---

## **8. Perceptrons as Logical Operators**

- In this lesson, we'll see one of the many great applications of **perceptrons**.
  - As **logical operators**! You'll have the chance to create the **perceptrons** for the most common of these, the `AND`, `OR`, and `NOT` operators.
  - And then, we'll see what to do about the elusive `XOR` operator. Let's dive in!

### `AND` **Perceptron**

🎥 [Udacity, Video Link](https://youtu.be/Y-ImuxNpS40)

- **Note:** The second and third rows of the third column from `1:50-onward` should be blue in color (they have the correct value of 1) for the `OR` **Perceptron**.

![]()

### 📝 QUESTION 5: 1 OF 4

### _What are the weights and bias for the AND perceptron?_

- Set the weights (`weight1`, `weight2`) and bias (`bias`) to values that will correctly determine the `AND` operation as shown above.
- More than one set of values will work!

```python
import pandas as pd

### TODO: Set weight1, weight2, and bias

weight1 = 0.75
weight2 = 1.0
bias = 0

### DON'T CHANGE ANYTHING BELOW
### Inputs and outputs

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

### Generate and check output
for test*input, correct_output in zip(test_inputs, correct_outputs):
linear_combination = weight1 * test*input[0] + weight2 * test_input[1] + bias
output = int(linear_combination >= 0)
is_correct_string = 'Yes' if output == correct_output else 'No'
outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

### Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct'])
if not num_wrong:
print('Nice! You got it all correct.\n')
else:
print('You got {} wrong. Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
```

### `OR` **Perceptron**

![]()

- The `OR` **perceptron** is very similar to an `AND` **perceptron**.
  - In the image below, the `OR` **perceptron** has the same line as the `AND` **perceptron**, except the line is shifted down.
  - What can you do to the weights and/or bias to achieve this?
  - Use the following `AND` **perceptron** to create an `OR` **Perceptron**.

![]()

### 📝 QUESTION 4: 2 OF 4

### ❓ _What are two ways to go from an AND perceptron to an OR perceptron?_

- ✅ Increase the weights
- ✅ Decrease the magnitude of the bias

### `NOT` **Perceptron**

- Unlike the other perceptrons we looked at, the `NOT` operation only cares about one input.

  - The operation returns a `0` if the input is `1` and a `1` if it's a `0`.
  - The other inputs to the perceptron are ignored.

- In this quiz, you'll set the weights (`weight1`, `weight2`) and bias `bias` to the values that calculate the `NOT` operation on the second input and ignores the first input.

```python
import pandas as pd

### TODO: Set weight1, weight2, and bias
weight1 = 2.0
weight2 = 0.0
bias = 0.0

### DON'T CHANGE ANYTHING BELOW
### Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

### Generate and check output
for test*input, correct_output in zip(test_inputs, correct_outputs):
  linear_combination = weight1 * test*input[0] + weight2 * test_input[1] + bias
  output = int(linear_combination >= 0)
  is_correct_string = 'Yes' if output == correct_output else 'No'
  outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

### Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct'])
if not num_wrong:
print('Nice! You got it all correct.\n')
else:
print('You got {} wrong. Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
```

🎥 [Udacity, Video Link](https://youtu.be/-z9K49fdE3g)

### `XOR` **Perceptron**

![]()

### ❓Quiz: Build an XOR Multi-Layer Perceptron

- Now, let's build a multi-layer perceptron from the `AND`, `NOT`, and `OR` **perceptrons** to create `XOR` logic!

- The neural network below contains `3` **perceptrons**, A, B, and C.

  - The last one (`AND`) has been given for you.
  - The input to the neural network is from the first node.
  - The output comes out of the last node.

- The **multi-layer perceptron** below calculates `XOR`.
  - Each perceptron is a logic operation of `AND`, `OR`, and `NOT`.
  - However, the perceptrons A, B, and C don't indicate their operation.
  - In the following quiz, set the correct operations for the perceptrons to calculate `XOR`.

![]()

### 📝 QUESTION 4 OF 4

### ❓ _Set the operations for the perceptrons in the `XOR` neural network._

| ✅PERCEPTRON | OPERATORS |
| ------------ | --------- |
| A            | `AND`     |
| B            | `OR`      |
| C            | `NOT`     |

- ✅ And if we introduce the `NAND` operator as the combination of `AND` and `NOT`, then we get the following two-layer perceptron that will model `XOR`.
- That's our first neural network!

![]

---

## **9. Perceptron Trick**

- In the last section you used your logic and your mathematical knowledge to create perceptrons for some of the most common logical operators.
  - In real life, though, we can't be building these perceptrons ourselves.
  - The idea is that we give them the result, and they build themselves.
  - For this, here's a pretty neat trick that will help us.

🎥 [Udacity, Video Link](https://youtu.be/-zhTROHtscQ)

![]()

### 📝 QUESTION 1 OF 2

### ❓ _Does the misclassified point want the line to be closer or farther?_

- ✅ Closer

🎥 [Udacity, Video Link](https://youtu.be/fATmrG2hQzI)

### Time for some math

- Now that we've learned that the points that are misclassified, want the line to move closer to them, let's do some math.

🎥 [Udacity, Video Link](https://youtu.be/lif_qPmXvWA)

- The following video shows a mathematical trick that modifies the equation of the line, so that it comes closer to a particular point.

### ❓ _For the second example, where the line is described by `3x1+ 4x2 - 10 = 0`, if the learning rate was set to `0.1`_

### _how many times would you have to apply the perceptron trick to move the line to a position where the blue point, at `(1, 1)`, is correctly classified?_

- ✅ `10`

---

## **10. Perceptron Algorithm**

- And now, with the perceptron trick in our hands, we can fully develop the perceptron algorithm!
  - The following video will show you the pseudocode, and in the quiz below, you'll have the chance to code it in Python.

🎥 [Udacity, Video Link](https://youtu.be/p8Q3yu9YqYk)

- There's a small error in the above video in that `W_i` should be updated to W_i = W_i + +αx_i (plus or minus depending on the situation).

### Coding the Perceptron Algorithm

- Time to code! In this quiz, you'll have the chance to implement the perceptron algorithm to separate the following data (given in the file `data.csv`).

![]()

- Recall that the perceptron step works as follows.

  - For a point with coordinates `(p,q)`, label `y`, and prediction given by the equation `y^ = step(w_1x_1 + w_2x_2 + b)`

  - If the point is correctly classified, do nothing.
  - If the point is classified positive, but it has a negative label, subtract `αp,αq`, and `α` from `w_1`, `w_2`, and `b` respectively.
  - If the point is classified negative, but it has a positive label, add `αp`, `αp`, and `α` to `w_1`, `w_2`, and `b` respectively.

- Then click on `test run` to graph the solution that the perceptron algorithm gives you.

  - It'll actually draw a set of dotted lines, that show how the algorithm approaches to the best solution, given by the black solid line.

- Feel free to play with the parameters of the algorithm (number of epochs, learning rate, and even the randomizing of the initial parameters) to see how your initial conditions can affect the solution!

`perceptron.py`
`data.csv`
`solution.py`

```python
import numpy as np

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
  if t >= 0:
  return 1
  return 0

def prediction(X, W, b):
return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.

def perceptronStep(X, y, W, b, learn_rate = 0.01): # Fill in code
for i in range(len(X)):
y_hat = prediction(X[i],W,b)
if y[i]-y_hat == 1:
W[0] += X[i][0]*learn_rate
W[1] += X[i][1]*learn_rate
b += learn_rate
elif y[i]-y_hat == -1:
W[0] -= X[i][0]*learn_rate
W[1] -= X[i][1]*learn_rate
b -= learn_rate
return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
x_min, x_max = min(X.T[0]), max(X.T[0])
y_min, y_max = min(X.T[1]), max(X.T[1])
W = np.array(np.random.rand(2,1))
b = np.random.rand(1)[0] + x_max # These are the solution lines that get plotted below.
boundary_lines = []
for i in range(num_epochs): # In each epoch, we apply the perceptron step.
W, b = perceptronStep(X, y, W, b, learn_rate)
boundary_lines.append((-W[0]/W[1], -b/W[1]))
return boundary_lines
```

---

## **11. Non-Linear Regions**

🎥 [Udacity, Video Link](https://youtu.be/B8UrWnHh1Wc)

---

## **12. Error Functions**

🎥 [Udacity, Video Link](https://youtu.be/YfUUunxWIJw)

---

## **13. Log-loss Error Function**

🎥 [Udacity, Video Link](https://youtu.be/jfKShxGAbok)

We pick back up on log-loss error with the gradient descent concept.

### 📝 QUIZ QUESTION

Which of the following conditions should be met in order to apply gradient descent? (Check all that apply.)

The error function should be differentiable

The error function should be continuous

---

## **14. Discrete vs Continuous**

🎥 [Udacity, Video Link](https://youtu.be/rdP-RPDFkl0)
🎥 [Udacity, Video Link](https://youtu.be/Rm2KxFaPiJg)

- In the last few videos, we learned that continuous error functions are better than discrete error functions, when it comes to optimizing.
  - For this, we need to switch from discrete to continuous predictions.
  - The next two videos will guide us in doing that.

### 📝 QUIZ QUESTION

### The sigmoid function is defined as `sigmoid(x) = 1/(1+e-x)`.

### ❓ _If the score is defined by `4x1 + 5x2 - 9 = score`, then which of the following points has exactly a `50%` probability of being blue or red? (Choose all that are correct.)_

- `(1, 1)`
- `(-4, 5)`

---

## **15. Softmax**

🎥 [Udacity, Video Link](https://youtu.be/NNoezNnAMTY)

### Multi-Class Classification and Softmax

- The Softmax Function
- In the next video, we'll learn about the softmax function, which is the equivalent of the sigmoid activation function, but when the problem has 3 or more classes.

### 📝 QUESTION 1 OF 2

### _What function turns every number into a positive number?_

exp

Quiz: Coding Softmax
And now, your time to shine! Let's code the formula for the Softmax function in Python.

`softmax.py`
`solution.py`

```python
import numpy as np

### Write a function that takes as input a list of numbers, and returns
### the list of values given by the softmax function.

def softmax(L):
  expL = np.exp(L)
  sumExpL = sum(expL)
  result = []

for i in expL:
  result.append(i\*1.0/sumExpL)
  return result
```

---

## **16. One-Hot Encoding**

🎥 [Udacity, Video Link](https://youtu.be/AePvjhyvsBo)

---

## **17. Maximum Likelihood**

🎥 [Udacity, Video Link](https://youtu.be/1yJx-QtlvNI)
🎥 [Udacity, Video Link](https://youtu.be/6nUUeQ9AeUA)

- Probability will be one of our best friends as we go through Deep Learning.
  - In this lesson, we'll see how we can use probability to evaluate (and improve!) our models.

### 📝 QUIZ QUESTION

### _Based on the above video, which of the following is true for a very high value for P(all)?_

- The model classifies most points correctly with P(all) indicating how accurate the model is.

- The next video will show a more formal treatment of Maximum Likelihood.

---

## **18. Maximizing Probabilities**

🎥 [Udacity, Video Link](https://youtu.be/-xxrisIvD0E)
🎥 [Udacity, Video Link](https://youtu.be/njq6bYrPqSU)

- In this lesson and quiz, we will learn how to maximize a probability, using some math. Nothing more than high school math, so get ready for a trip down memory lane!

### 📝 QUIZ QUESTION

### _What function turns products into sums?_

- **log**

---

## **19. Cross-Entropy 1**

🎥 [Udacity, Video Link](https://youtu.be/iREoPUrpXvE)

### Correction: At 2:18, the top right point should be labelled -log(0.7) instead of -log(0.2)

---

## **20. Cross-Entropy 2**

🎥 [Udacity, Video Link](https://youtu.be/qvr_ego_d6w)
🎥 [Udacity, Video Link](https://youtu.be/1BnhC6e0TFw)

- So we're getting somewhere, there's definitely a connection between probabilities and error functions, and it's called Cross-Entropy.
  - This concept is tremendously popular in many fields, including Machine Learning. Let's dive more into the formula, and actually code it!

### 📝 QUIZ: Coding Cross-entropy

- Now, time to shine! Let's code the formula for cross-entropy in Python.
  - As in the video, Y in the quiz is for the category, and P is the probability.

`cross_entropy.py`
`solution.py`

```python
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

def cross*entropy(Y, P):
  Y = np.float*(Y)
  P = np.float\_(P)
  return -np.sum(Y _ np.log(P) + (1 - Y) _ np.log(1 - P))
```

---

## **21. Multi-Class Cross Entropy**

🎥 [Udacity, Video Link](https://youtu.be/keDswcqkees)

---

## **22. Logistic Regression**

🎥 [Udacity, Video Link](https://youtu.be/V5kkHldUlVU)
🎥 [Udacity, Video Link](https://youtu.be/KayqiYijlzc)

---

## **23. Gradient Descent**

🎥 [Udacity, Video Link](https://youtu.be/rhVIF-nigrY)

---

## **24. Logistic Regression Algorithm**

🎥 [Udacity, Video Link](https://youtu.be/snxmBgi_GeU)

---

## **25. 📙 Pre-Notebook: Gradient Descent**

### Implementing Gradient Descent

- In the following notebook, you'll be able to implement the gradient descent algorithm on the following sample dataset with two classes.

- Red and blue data points with some overlap.

### Workspace

- To open this notebook, you have two options:

- Go to the next page in the classroom (recommended)

- Clone the repo from Github and open the notebook `GradientDescent.ipynb` in the intro-neural-networks > gradient-descent folder.
  - You can either download the repository via the command line with git clone [https://github.com/udacity/deep-learning-v2-pytorch.git](https://github.com/udacity/deep-learning-v2-pytorch.git), or download it as an archive file from this link.

### Instructions

- In this notebook, you'll be implementing the functions that build the gradient descent algorithm, namely:

- sigmoid: The sigmoid activation function.
- output_formula: The formula for the prediction.
- error_formula: The formula for the error at a point.
- update_weights: The function that updates the parameters with one gradient descent step.
- When you implement them, run the train function and this will graph the several of the lines that are drawn in successive gradient descent steps.

  - It will also graph the error function, and you can see it decreasing as the number of epochs grows.

- This is a self-assessed lab.
  - If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

---

## **26. 📙 Notebook: Gradient Descent**

🎥 [Udacity, Jupyter Notebook Link](https://classroom.udacity.com/courses/ud188/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/64f025bd-1d7b-42fb-9f13-8559242c1ec9)

---

## **27. Perceptron vs Gradient Descent**

🎥 [Udacity, Video Link](https://youtu.be/uL5LuRPivTA)

- In the video at `0:12` mark, the instructor said `y^ - y`.
- It should be `y - y^` instead as stated on the slide.

---

## **28. Continuous Perceptrons**

🎥 [Udacity, Video Link](https://youtu.be/07-JJ-aGEfM)

---

## **29. Non-linear Data**

🎥 [Udacity, Video Link](https://youtu.be/F7ZiE8PQiSc)

---

## **30. Non-Linear Models**

🎥 [Udacity, Video Link](https://youtu.be/HWuBKCZsCo8)

---

## **31. Neural Network Architecture**

🎥 [Udacity, Video Link](https://youtu.be/Boy3zHVrWB4)

- Ok, so we're ready to put these building blocks together, and build great Neural Networks! (Or Multi-Layer Perceptrons, however you prefer to call them.)

- This first two videos will show us how to combine two perceptrons into a third, more complicated one.

### 📝 QUESTION 10: 1 OF 2

- Based on the above video, let's define the combination of two new perceptrons as w1*0.4 + w2*0.6 + b.

  - Which of the following values for the weights and the bias would result in the final probability of the point to be 0.88?

- w1: 3, w2: 5, b: -2.2

### Multiple layers

- Now, not all neural networks look like the one above.

  - They can be way more complicated! In particular, we can do the following things:

- Add more nodes to the input, hidden, and output layers.
- Add more layers.
- We'll see the effects of these changes in the next video.

### Multi-Class Classification

- And here we elaborate a bit more into what can be done if our neural network needs to model data with more than one output.
- How many nodes in the output layer would you require if you were trying to classify all the letters in the English alphabet?
- 26

---

## **32. Feedforward**

🎥 [Udacity, Video Link](https://youtu.be/hVCuvMGOfyY)

- **Feedforward**

  - **Feedforward** is the process neural networks use to turn the input into an output. Let's study it more carefully, before we dive into how to train the networks.

- **Error Function**
  - Just as before, neural networks will produce an error function, which at the end, is what we'll be minimizing. The following video shows the error function for a neural network.

---

## **33. Backpropagation**

🎥 [Udacity, Video Link](https://youtu.be/1SmY3TZTyUk)

### Backpropagation

- Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as backpropagation. In a nutshell, backpropagation will consist of:

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good.
- Sounds more complicated than what it actually is. Let's take a look in the next few videos. The first video will show us a conceptual interpretation of what backpropagation is.

### Backpropagation Math

- And the next few videos will go deeper into the math.

  - Feel free to tune out, since this part gets handled by Keras pretty well.
  - If you'd like to go start training networks right away, go to the next section.
  - But if you enjoy calculating lots of derivatives, let's dive in!

- In the video below at `1:24`, the edges should be directed to the sigmoid function and not the bias at that last layer; the edges of the last layer point to the bias currently which is incorrect.

- Chain Rule
- We'll need to recall the chain rule to help us calculate derivatives.

- Calculation of the derivative of the sigmoid function
- Recall that the sigmoid function has a beautiful derivative, which we can see in the following calculation. This will make our backpropagation step much cleaner.

---

## **34. 📙 Pre-Notebook: Analyzing Student Data**

- Now, we're ready to put neural networks in practice. We'll analyze a dataset of student admissions at UCLA.

- To open this notebook, you have two options:

- Go to the next page in the classroom (recommended).
- Clone the repo from Github and open the notebook `StudentAdmissions.ipynb` in the intro-neural-networks > student_admissions folder.
  - You can either download the repository with git clone [https://github.com/udacity/deep-learning-v2-pytorch.git](https://github.com/udacity/deep-learning-v2-pytorch.git), or download it as an archive file from this link.

### Instructions

- In this notebook, you'll be implementing some of the steps in the training of the neural network, namely:

- One-hot encoding the data
- Scaling the data
- Writing the backpropagation step

- This is a self-assessed lab.
  - If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

---

## **35. 📙 Notebook: Analyzing Student Data**

🎥 [Udacity, Jupyter Notebook Link](https://classroom.udacity.com/courses/ud188/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/dab588a2-51cc-4c4e-ba24-410a009943c7)

---

## **36. Training Optimization**

🎥 [Udacity, Video Link](https://youtu.be/UiGKhx9pUYc)

---

## **37. Testing**

🎥 [Udacity, Video Link](https://youtu.be/EeBZpb-PSac)

---

## **38. Overfitting and Underfitting**

🎥 [Udacity, Video Link](https://youtu.be/xj4PlXMsN-Y)

---

## **39. Early Stopping**

🎥 [Udacity, Video Link](https://youtu.be/NnS0FJyVcDQ)

---

## **40. Regularization**

🎥 [Udacity, Video Link](https://youtu.be/KxROxcRsHL8)

---

## **41. Regularization 2**

🎥 [Udacity, Video Link](https://youtu.be/ndYnUrx8xvs)

---

## **42. Dropout**

🎥 [Udacity, Video Link](https://youtu.be/Ty6K6YiGdBs)

---

## **43. Local Minima**

🎥 [Udacity, Video Link](https://youtu.be/gF_sW_nY-xw)

---

## **44. Random Restart**

🎥 [Udacity, Video Link](https://youtu.be/idyBBCzXiqg)

---

## **45. Vanishing Gradient**

🎥 [Udacity, Video Link](https://youtu.be/W_JJm_5syFw)

---

## **46. Other Activation Functions**

🎥 [Udacity, Video Link](https://youtu.be/kA-1vUt6cvQ)

---

## **47. Batch vs Stochastic Gradient Descent**

🎥 [Udacity, Video Link](https://youtu.be/2p58rVgqsgo)

---

## **48. Learning Rate Decay**

🎥 [Udacity, Video Link](https://youtu.be/TwJ8aSZoh2U)

---

## **49. Momentum**

🎥 [Udacity, Video Link](https://youtu.be/r-rYz_PEWC8)

---

## **50. Error Functions Around the World**

🎥 [Udacity, Video Link](https://youtu.be/34AAcTECu2A)

---

## Foam Related Links

- [[nn-intro-ud089]]
