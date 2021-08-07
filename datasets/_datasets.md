# Datasets

## [🎓 Datasets, Lesson 6, Udacity, UD120](https://classroom.udacity.com/courses/ud120/lessons/2291728537/concepts/28772285450923)

---

## **1. Introduction**

🎥 [Udacity, Video Link](https://youtu.be/wDQhif-MWuY)

---

## **2. What Is A POI**

🎥 [Udacity, Video Link](https://youtu.be/wDQhif-MWuY)

---

## **3. Accuracy vs. Training Set Size**

🎥 [Udacity, Video Link](https://youtu.be/9w1Yi5nMNgw)

---

## **4. Downloading Enron Data**

🎥 [Udacity, Video Link](https://youtu.be/TgkBAtaTqJk)

---

## **5. Types of Data Quiz 1**

🎥 [Udacity, Video Link](https://youtu.be/YZb-Uam-Ics)

---

## **6. Types of Data Quiz 2**

🎥 [Udacity, Video Link](https://youtu.be/k63Why0c1KU)

---

## **7. Types of Data Quiz 3**

🎥 [Udacity, Video Link](https://youtu.be/8TeKzSUGAJQ)

---

## **8. Types of Data Quiz 4**

🎥 [Udacity, Video Link](https://youtu.be/DzyOcsBIncA)

---

## **9. Types of Data Quiz 5**

🎥 [Udacity, Video Link](https://youtu.be/bRhdim9PTFI)

---

## **10. Types of Data Quiz 6**

🎥 [Udacity, Video Link](https://youtu.be/-LtbhZvwwM8)

---

## **11. Enron Dataset Mini-Project Video**

🎥 [Udacity, Video Link](https://youtu.be/0zGp5er3fy4)

---

## **12. Datasets and Questions Mini-Project**

Datasets and Questions Mini-Project
The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset, which you’ll explore in this mini-project.

Getting started:

Clone this git repository: https://github.com/udacity/ud120-projects
Open the starter code: datasets_questions/explore_enron_data.py
Tip: If you're not familiar with git, take this short course to get started.

---

## **13. Size of the Enron Dataset**

The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person.
The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. Use datasets_questions/explore_enron_data.py to load the dataset.

How many data points (people) are in the dataset?

Start Quiz

---

## **14. Features in the Enron Dataset**

For each person, how many features are available?

Start Quiz

---

## **15. Finding POIs in the Enron Data**

The “poi” feature records whether the person is a person of interest, according to our definition. How many POIs are there in the E+F dataset?

Start Quiz
In other words, count the number of entries in the dictionary where
data[person_name]["poi"]==1

---

## **16. How Many POIs Exist?**

We compiled a list of all POI names (in ../final_project/poi_names.txt) and associated email addresses (in ../final_project/poi_email_addresses.py).

How many POI’s were there total? (Use the names file, not the email addresses, since many folks have more than one address and a few didn’t work for Enron, so we don’t have their emails.)

Start Quiz

---

## **17. Problems with Incomplete Data**

As you can see, we have many of the POIs in our E+F dataset, but not all of them. Why is that a potential problem?

We will return to this later to explain how a POI could end up not being in the Enron E+F dataset, so you fully understand the issue before moving on.

Start Quiz

---

## **18. Query the Dataset 1**

Like any dict of dicts, individual people/features can be accessed like so:

enron_data["LASTNAME FIRSTNAME"]["feature_name"]
or, sometimes
enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]

What is the total value of the stock belonging to James Prentice?

Start Quiz
Lastname, Firstname and Middle Initial all in CAPS.

---

## **19. Query the Dataset 2**

Like any dict of dicts, individual people/features can be accessed like so:

enron_data["LASTNAME FIRSTNAME"]["feature_name"]

How many email messages do we have from Wesley Colwell to persons of interest?

Start Quiz

---

## **20. Query the Dataset 3**

Like any dict of dicts, individual people/features can be accessed like so:

enron_data["LASTNAME FIRSTNAME"]["feature_name"]

or

enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]

What’s the value of stock options exercised by Jeffrey K Skilling?

Start Quiz

---

## **21. Research the Enron Fraud**

In the coming lessons, we’ll talk about how the best features are often motivated by our human understanding of the problem at hand. In this case, that means knowing a little about the story of the Enron fraud.

If you have an hour and a half to spare, “Enron: The Smartest Guys in the Room” is a documentary that gives an amazing overview of the story. Alternatively, there are plenty of archival newspaper stories that chronicle the rise and fall of Enron.

Which of these schemes was Enron not involved in?

selling assets to shell companies at the end of each month, and buying them back at the beginning of the next month to hide accounting losses
causing electrical grid failures in California
illegally obtained a government report that enabled them to corner the market on frozen concentrated orange juice futures
conspiring to give a Saudi prince expedited American citizenship
a plan in collaboration with Blockbuster movies to stream movies over the internet
Start Quiz

---

## **22. Enron CEO**

Who was the CEO of Enron during most of the time that fraud was being perpetrated?

Start Quiz

---

## **23. Enron Chairman**

Who was chairman of the Enron board of directors?

Start Quiz

---

## **24. Enron CFO**

Who was CFO (chief financial officer) of Enron during most of the time that fraud was going on?

Start Quiz

---

## **25. Follow the Money**

Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of “total_payments” feature)?

How much money did that person get?

Start Quiz

---

## **26. Unfilled Features**

For nearly every person in the dataset, not every feature has a value. How is it denoted when a feature doesn’t have a well-defined value?

Start Quiz

---

## **27. Dealing with Unfilled Features**

How many folks in this dataset have a quantified salary? What about a known email address?

Start Quiz

---

## **28. Dict-to-array conversion**

A python dictionary can’t be read directly into an sklearn classification or regression algorithm; instead, it needs a numpy array or a list of lists (each element of the list (itself a list) is a data point, and the elements of the smaller list are the features of that point).

We’ve written some helper functions (featureFormat() and targetFeatureSplit() in tools/feature_format.py) that can take a list of feature names and the data dictionary, and return a numpy array.

In the case when a feature does not have a value for a particular person, this function will also replace the feature value with 0 (zero).

---

## **29. Missing POIs 1 (optional)**

As you saw a little while ago, not every POI has an entry in the dataset (e.g. Michael Krautz). That’s because the dataset was created using the financial data you can find in final_project/enron61702insiderpay.pdf, which is missing some POI’s (those absences propagated through to the final dataset). On the other hand, for many of these “missing” POI’s, we do have emails.

While it would be straightforward to add these POI’s and their email information to the E+F dataset, and just put “NaN” for their financial information, this could introduce a subtle problem. You will walk through that here.

How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? What percentage of people in the dataset as a whole is this?

Start Quiz

---

## **30. Missing POIs 2 (optional)**

How many POIs in the E+F dataset have “NaN” for their total payments? What percentage of POI’s as a whole is this?

Start Quiz

---

## **31. Missing POIs 3 (optional)**

If a machine learning algorithm were to use total_payments as a feature, would you expect it to associate a “NaN” value with POIs or non-POIs?

Start Quiz

---

## **32. Missing POIs 4 (optional)**

If you added in, say, 10 more data points which were all POI’s, and put “NaN” for the total payments for those folks, the numbers you just calculated would change.
What is the new number of people of the dataset? What is the new number of folks with “NaN” for total payments?

Start Quiz

---

## **33. Missing POIs 5 (optional)**

What is the new number of POI’s in the dataset? What is the new number of POI’s with NaN for total_payments?

Start Quiz

---

## **34. Missing POIs 6 (optional)**

Once the new data points are added, do you think a supervised classification algorithm might interpret “NaN” for total_payments as a clue that someone is a POI?

Start Quiz

---

## **35. Mixing Data Sources (optional)**

Adding in the new POI’s in this example, none of whom we have financial information for, has introduced a subtle problem, that our lack of financial information about them can be picked up by an algorithm as a clue that they’re POIs. Another way to think about this is that there’s now a difference in how we generated the data for our two classes--non-POIs all come from the financial spreadsheet, while many POIs get added in by hand afterwards. That difference can trick us into thinking we have better performance than we do--suppose you use your POI detector to decide whether a new, unseen person is a POI, and that person isn’t on the spreadsheet. Then all their financial data would contain “NaN” but the person is very likely not a POI (there are many more non-POIs than POIs in the world, and even at Enron)--you’d be likely to accidentally identify them as a POI, though!

This goes to say that, when generating or augmenting a dataset, you should be exceptionally careful if your data are coming from different sources for different classes. It can easily lead to the type of bias or mistake that we showed here. There are ways to deal with this, for example, you wouldn’t have to worry about this problem if you used only email data--in that case, discrepancies in the financial data wouldn’t matter because financial features aren’t being used. There are also more sophisticated ways of estimating how much of an effect these biases can have on your final answer; those are beyond the scope of this course.

For now, the takeaway message is to be very careful about introducing features that come from different sources depending on the class! It’s a classic way to accidentally introduce biases and mistakes.

---

## Foam Related Links

- [[_datasets]]
