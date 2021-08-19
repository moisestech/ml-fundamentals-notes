# ML UD Quiz

## Machine Learning Udacity Quiz

---

### **1. True or False: All problems where you're trying to predict some value can be solved with machine learning.**

- [ ] True. You might have to get pretty clever with how you use machine learning in some cases, but all predictions can be solved with machine learning.

- [ ] False. In some situations, you fundamentally do not have enough information to make an accurate prediciton with any technique, including machine learning.

---

### **2. What are hyperparameters?**

- [ ] Model parameters that change faster than most other model paramters during model training.
- [ ] Model paramters that have more of an impact on the final result than most other model parameters.
- [ ] Parameters within a model inference algorithm.
- [ ] Parameters which affect model training but typically cannot be incrementally adjusted like other paramters as part of model training.

---

### **3. Which of the following is the best explanation for how a model trainig algorithm works?**

- [ ] Model training algorithms incrementally adjust model parameters to minimize some loss function.
- [ ] Model training algorithms slowly add or remove model parameters until the model fits the dataset it was trained on.
- [ ] Model trainign algorithms train a separate model for each data point and then picks the one with the lowest function.
- [ ] Model training algorithms are used on a prepared machine learning model to make predictions on n ewn unseen data.

---

### **4. Which of the following would not be a good reason to us a linear model?**

- [ ] I am not yet sure what model would work well with my data, and I want to establish a quick baseline.
- [ ] The input-output relationships I'm trying to model look like they're pretty straightforward.. When plotting my label as a function of each of my input variables. I can see that fitting a line would do better than randomly guessing in many cases.
- [ ] Linear models worked for most other problems I've tacked before, so it should definately work well for my current challenge.
- [ ] I'm still early on in building my understanding of machine learning and want to start with something I can easily understand and visualize.

---

### **5. What is a machine learning model?**

- [ ] Generic program, made specific by data.
- [ ] A set of high-level statistics and visualizations that help you understand the shape of your data.
- [ ] A linear function.

---

### **6. Which of the following is not a model evaluation metric?**

- [ ] Root Mean Square (RMS)
- [ ] Model Inference Algorithm
- [ ] Silhouette Coefficient
- [ ] Accuracy

---

### **7. Your friend suggests you should ranfomize the order of your data before splitting into a training and test dataset is your friend right?**

- [ ] No, your friend is not right. There is no benefit to doing this, and the detail just makes the process too complex.
- [ ] No, your friend is not right. You should make sure your work is always repeatable, and there is no way to gt repeatability if you randomly reorder the dataset.
- [ ] Yes, your friend is right. It is possible your data was originally organized in such a way that splitting will give the model a biased view of the dataset (imagine for example if all the data was ordered by some feature)

---

### **8. Let's say you're trying to cluster anonymized foot traffic locatin data to determine common "downtown" hubs in a city. You're confident hubs will show up as tight clusters, so you're surprised when you first approach misses certain clusters you know should exist. What should you do?**

- [ ] Check the dataset to test your assumptions, is the data well-formatted? Does the data effectively represent the real world? For example, do you have data in the missing clusters?
- [ ] Inspect the model to test your assumptions. Does your model contain assumptions about how the data is organized? For example, does the model assume all clusters are the same size? How does the model handle clusters that are very close to each other?
- [ ] Inspect your model evaluation mechanism to ensure you're asking the right questions. Is your evaluation mechnism's definition of a "better" model aligned with the problem you're trying to solve?
- [ ] All of these answers.

---

### **9. Let's say you're trying to predict a child's show size from their age and parent's show sizes. You relax the problem to allow for predictions between shoe sizes. What kind of machine learning task would you use to represent this problem?**

- [ ] Regression
- [ ] Classification
- [ ] Reinforcement Learning
- [ ] Clustering

---

### **10. Is model accuracy sufficient to use in all cases?**

- [ ] Yes. By definition, it's measuring how good your model is at making predictions.
- [ ] No, You might care more about some predictions than others, and model accuracy treats all predictions as the same.

---

### **11. What is the definition of model accuracy?**

- [ ] How of your model makes a correct prediction.
- [ ] How often your model makes similar predictions.

---

### **12. In Python when objects are no longer needed**

- [ ] They exist until the program ends.
- [ ] They are removed by the developer.
- [ ] They are eligible for garbage collection.
- [ ] They are moved into another memory bucket to reduce the memory footprint of the application.

---

### **13. In an object-oriented paradigm objects can be. Select all that apply.**

- [ ] Nothing
- [ ] Places
- [ ] Things
- [ ] People

---

### **14. Which of the following are good code review questions? Select all that apply.**

- [ ] Is the code clean and modular.
- [ ] Is the code efficient.
- [ ] Is the documentation effective.
- [ ] Is the code well tested.

---

### **15. Why do we want to use object-oriented programming? Select all that apply.**

- [ ] Object oriented programming allows you to create large, modular programs that can easily expand over time.
- [ ] Object-oriented makes programs more difficult to navigate through.
- [ ] Object-oriented programs hide the implementation from the end user.
- [ ] Object-oriented programs expose the implementation to the developer.

---

### **16. The goals of object-oriented paradigm are: Select all that apply.**

- [ ] To promote code reuse.
- [ ] To increase the speed of computation operations.
- [ ] To reduce memory footprint of the application.
- [ ] To model software as real world examples.

---

### **17. Which programming paradigm uses classes as a way to mel software to promote better understanding and reusability of code? Choose one.**

- [ ] Object-oriented
- [ ] Procedural
- [ ] Both object-oriented and procedural

---

### **18. Which of the following are considered actions of a student object?**

- [ ] Name
- [ ] Age
- [ ] Schedule a class
- [ ] Student ID

---

### **19. Which of the following are consired characteristics of a student object?**

- [ ] Name
- [ ] Age
- [ ] Schedule a class
- [ ] Student ID

---

### **20. Objects and instance variable are created in**

- [ ] Stack memory
- [ ] Set memory
- [ ] Heap memory
- [ ] Hash memory

---

### **21. A class is considered the blueprint for**

- [ ] Creating objects
- [ ] Creating variables
- [ ] Creating functions
- [ ] Creating memory

---

### **22. True or False. Classes, object, attributes, methods, and inheritance are common to all object-oriented programming laguages.**

- [ ] True
- [ ] False

---

### **23. What is pip?**

- [ ] A Python class that collects unused data files.
- [ ] A virtual memory management that frees up memory by destroying unsed objects.
- [ ] A Python package manager that helps with installing and uninstalling Python packages.
- [ ] A Python library that supports adding and removing objects.

---

### **24. In Python what practice do we use to provide another class with all of the functions and properties from another class?**

- [ ] Object-oriented
- [ ] Inheritance
- [ ] Procedure
- [ ] Cross compatability

---

### **25. True or False. Conda does two things: manages pacages and manages environments.**

- [ ] True
- [ ] False

---

### **26. What is a package?**

- [ ] A collection of Python modules
- [ ] A type of class that wraps code in a container
- [ ] A virtual environment used for testing
- [ ] An output file that describes unit test failures

---

### **27. What is the correct syntax for installing numpy packages using conda?**

- [ ] numpy install
- [ ] install numpy
- [ ] C install numpy
- [ ] conda install numpy

---

### **28. True or False. Magic methods are special methods that allow users to define specific behavior of an object during lifecycle**

- [ ] True
- [ ] False

---

### **29. What is the purpose of the magic method `_init_` ?**

- [ ] It creates an init object in Python.
- [ ] It runs unit tests in Python.
- [ ] It allows users to define the deconstruction behavior of an object.
- [ ] It allows users to define the intialization behavior of an objects.

---

### **30. Select the Python package managers:**

- [ ] file_manager
- [ ] Coonda
- [ ] Conda
- [ ] Venv
- [ ] AirManager

---

### **31.True or False. In Python we can have more than one class with the same name in a different package.**

- [ ] True
- [ ] False

---

### **32. How do you install pytest?**

- [ ] Type "dos install -U pytest" in your terminal
- [ ] Type "pip install -U pytest" in your terminal
- [ ] Type "pip install -I pytest" in your terminal
- [ ] Type "pip install -J pytest" in your terminal

---

### **33. In a test output file how are test failures indentified?**

- [ ] Fs
- [ ] Comma
- [ ] Periods
- [ ] Question marks

---

### **34. How do we define unit tests within a test file?**

- [ ] @unit tests
- [ ] test\_
- [ ] \_tests
- [ ] @Test

---

### **35. What type of testing tests more than one component, including interactions between different components?**

- [ ] Unit Testing
- [ ] Integration Testing
- [ ] Test-driven development (TDD)
- [ ] Formal Testing

---

### **36. In a test output file how are test successes indentified? Choose One.**

- [ ] Fs
- [ ] Comma
- [ ] Periods
- [ ] Question marks

---

### **37. Why do we want to conduct code reviews? Select all that apply.**

- [ ] Catch errors
- [ ] Ensure Readability
- [ ] Check Standards are met
- [ ] Share knowledge among teams

---

### **38. True or False: Logging is valuable for understanding the events that occur while running your program. Choose one.**

- [ ] True
- [ ] False

---

### **39. Which item in Python will stop a unit test abruptly?**

- [ ] Assertion failures
- [ ] Syntax errors
- [ ] Dynamic linking error
- [ ] Input errors

---

### **40. Why are assertions used in unit testing?**

- [ ] Assertions are statments that are believed to be true.
- [ ] Assertions are used to document test cases.
- [ ] Assertions valudate the syntax of functions and variables.
- [ ] Assertions are used to skip test runs.

---

### **41. Why is it recommended to have only one assert statement per test?**

- [ ] In a test output file we see every assertions that fails.
- [ ] We do not see failures in the output file.
- [ ] In a test output file we only see which tests functions fail.
- [ ] In a test output we see both functions and assertions that fail.

---

### **42. In addition to classifying the True data from the Fake data, the discriminator also provides feedback to the generator.**

- [ ] True
- [ ] False

---

### **43.**

---

### **44.**

---

### **45.**

---

### **46.**

---

### **47.**

---

### **48.**

---

### **49.**

---

### **50.**

---

## Foam Related Links

- [[_ml]]
