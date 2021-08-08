# Deep Composer

## [🎓 ML w/ AWS Deep Composer, Lesson 4, Udacity, UD090, AWS ML Foundations Course](https://classroom.udacity.com/courses/ud090/lessons/099925a2-4f01-41c7-a4d4-8ce246f7b801/concepts/cf74ba9d-6964-4146-a499-fe865f57180e)

---

## **1. Introduction**

🎥 [Udacity, Video Link](https://youtu.be/0qmBjcBB38U)

Meet Your Instructor

Jyothi Nookula is a Senior Product Manager for AI Devices at Amazon Web Services.

What You Will Learn In This Lesson

In this lesson you'll get an introduction to Machine Learning. You will learn about Generative AI and AWS DeepComposer. You'll also learn how to build a custom Generative Adversarial Network.

Lesson Outline
What You Will Learn

We hope this will be an interesting and exciting learning experience for you.

Let's get started!

---

## **2. AWS Account Requirements**

AWS Account is Required
To complete the exercises in this course you will need an AWS Account ID.

To set up a new AWS Account ID, follow the directions here: How do I create and activate a new Amazon Web Services account?

You will be required to provide a payment method when you create the account. You can learn about which services are available at no cost in the the AWS Free Tier documentation

Will The Exercises Cost Anything?
Generate an Inference Exercise (Required)
Your AWS account includes free access for up to 500 inference jobs in the 12 months after you first use the AWS DeepComposer service. You can use one of these free instances to complete the exercise at no cost.

You can learn more about DeepComposer costs in the AWS DeepComposer pricing documentation

Build a Custom GAN Demo (Optional)
Amazon SageMaker is a separate service and has its own service pricing and billing tier. To create the custom GAN, our instructor uses an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the instructor and build a custom GAN on your own you may incur a cost.

Please note that creating your own custom GAN is completely optional. You are not required to do this exercise to complete the course.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation

Preparing for the Lesson
Please confirm that you have taken the necessary steps to complete this lesson.

Task List

---

## **3. Why Machine Learning on AWS?**

🎥 [Udacity, Video Link](https://youtu.be/ClDpCJ5YYE4)

AWS Mission
Put machine learning in the hands of every developer.

Why AWS?
AWS offers the broadest and deepest set of AI and ML services with unmatched flexibility.
You can accelerate your adoption of machine learning with AWS SageMaker. Models that previously took months and required specialized expertise can now be built in weeks or even days.
AWS offers the most comprehensive cloud offering optimized for machine learning.
More machine learning happens at AWS than anywhere else.
AWS ML Stack
AWS Machine Learning Stack

More Relevant Enterprise Search With Amazon Kendra
Natural language search with contextual search results
ML-optimized index to find more precise answers
20+ Native Connectors to simplify and accelerate integration
Simple API to integrate search and easily develop search applications
Incremental learning through feedback to deliver up-to-date relevant answers
Online Fraud Detection with Amazon Fraud Detector
Pre-built fraud detection model templates
Automatic creation of custom fraud detection models
One interface to review past evaluations and detection logic
Models learn from past attempts to defraud Amazon
Amazon SageMaker integration
Amazon Code Guru
Amazon CodeGuru For High-performing Software
Amazon CodeGuru For High-performing Software

Better Insights And Customer Service With Contact Lens
Identify common call types
Identify recurring themes based on customer call feedback
Alert supervisors when customers are having a poor experience
Assist agents with a knowledge base to answer questions as they are being asked

How to Get Started?
AWS DeepLens: deep learning and computer vision
AWS DeepRacer and the AWS DeepRacer League: reinforcement learning
AWS DeepComposer: Generative AI.
AWS ML Training and Certification: Curriculum used to train Amazon developers
Partnerships with Online Learning Providers: Including this course and the Udacity AWS DeepRacer course!
AWS Educational Devices
AWS Educational Devices

QUESTION 1 OF 2
How can Amazon SageMaker accelerate adoption of machine learning?

QUESTION 2 OF 2
Match the parts of the AWS ML Stack

TOOL

CATEGORY

Amazon Personalize

Amazon SageMaker

TensorFlow

SageMaker Ground Truth

PyTorch

Amazon Rekognition Image

Keras

---

## **4. ML Techniques and Generative AI**

🎥 [Udacity, Video Link](https://youtu.be/nBhS6X98NnY)

Machine Learning Techniques
Supervised Learning: Models are presented wit input data and the desired results. The model will then attempt to learn rules that map the input data to the desired results.
Unsupervised Learning: Models are presented with datasets that have no labels or predefined patterns, and the model will attempt to infer the underlying structures from the dataset. Generative AI is a type of unsupervised learning.
Reinforcement learning: The model or agent will interact with a dynamic world to achieve a certain goal. The dynamic world will reward or punish the agent based on its actions. Overtime, the agent will learn to navigate the dynamic world and accomplish its goal(s) based on the rewards and punishments that it has received.
Types of Machine Learning
Types of Machine Learning

Generative AI
Generative AI is one of the biggest recent advancements in artificial intelligence technology because of its ability to create something new. It opens the door to an entire world of possibilities for human and computer creativity, with practical applications emerging across industries, from turning sketches into images for accelerated product development, to improving computer-aided design of complex objects. It takes two neural networks against each other to produce new and original digital works based on sample inputs.

Generative AI Opens the Door to Possibilities
Generative AI Opens the Door to Possibilities

QUESTION 1 OF 2
Generative AI is usually an example of which type of Machine Learning?

Unsupervised Learning

Think Like a Machine Learning Engineer
We learned about how Airbus, NASA JPL and Glidewell Dental are using Generative AI. Can you think of other projects where Generative AI would be helpful?

Enter your response here, there's no right or wrong answer
Additional Resources
AWS DeepLens

---

## **5. AWS DeepComposer**

🎥 [Udacity, Video Link](https://youtu.be/pXDmhnKstR8)

AWS Composer and Generative AI

AWS Deep Composer uses Generative AI, or specifically Generative Adversarial Networks (GANs), to generate music. GANs pit 2 networks, a generator and a discriminator, against each other to generate new content.

The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. In this context, the generator is like the orchestra and the discriminator is like the conductor. The orchestra plays and generates the music. The conductor judges the music created by the orchestra and coaches the orchestra to improve for future iterations. So an orchestra, trains, practices, and tries to generate music, and then the conductor coaches them to produced more polished music.

Orchestra and Conductor as a metaphor for GANS
GANS is Similar to an Orchestra and a Conductor: The More They Work Together, the Better They Can Perform!

AWS DeepComposer Workflow
Use the AWS DeepComposer keyboard or play the virtual keyboard in the AWS DeepComposer console to input a melody.

Use a model in the AWS DeepComposer console to generate an original musical composition. You can choose from jazz, rock, pop, symphony or Jonathan Coulton pre-trained models or you can also build your own custom genre model in Amazon SageMaker.

Publish your tracks to SoundCloud or export MIDI files to your favorite Digital Audio Workstation (like Garage Band) and get even more creative.

QUESTION 1 OF 2
Which of these are available as pre-trained genres in AWS DeepComposer?

Jazz

Pop

Rock

Symphony

QUESTION 2 OF 2
Which model is responsible for each of these roles in Generative AI?

ROLE

MODEL

Evaluates the quality of output

Creating new output

Providing feedback

;

---

## **6. Demo: Compose Music with DeepComposer**

🎥 [Udacity, Video Link](https://youtu.be/3kBxV56vIco)

Compose Music with AWS DeepComposer Models
Now that you know a little more about AWS DeepComposer including its workflow and what GANs are, let’s compose some music with AWS DeepComposer models. We’ll begin this demonstration by listening to a sample input and a sample output, then we’ll explore DeepComposer’s music studio, and we’ll end by generating a composition with a 4 part accompaniment.

To get to the main AWS DeepComposer console, navigate to AWS DeepComposer. Make sure you are in the US East-1 region.
Once there, click on Get started
In the left hand menu, select Music studio to navigate to the DeepComposer music studio
To generate music you can use a virtual keyboard or the physical AWS DeepComposer keyboard. For this lab, we’ll use the virtual keyboard.
To view sample melody options, select the drop down arrow next to Input
Select Twinkle, Twinkle, Little Star
Next, choose a model to apply to the melody by clicking Select model
From the sample models, choose Rock and then click Select model
Next, select Generate composition. The model will take the 1 track melody and create a multitrack composition (in this case, it created 4 tracks)
Click play to hear the output
Now that you understand a little about the DeepComposer music studio and created some AI generated music, let’s move on to an exercise for generating an interface. There, you’ll have an opportunity to clone a pre-trained model to create your AI generated music!

;

---

## **7. Exercise: Generate an Interface**

Prerequisites/Requirements
Chrome Browser: You will need to use the Chrome Browser for this exercise. If you do not have Chrome you can download it here: www.google.com/chrome
AWS Account ID: You will need an AWS Account ID to sign into the console for this project. To set up a new AWS Account ID, follow the directions here: How do I create and activate a new Amazon Web Services account?
Your AWS account includes free access for up to 500 inference jobs in the 12 months after you first use the AWS DeepComposer service. You can use one of these free instances to complete the exercise at no cost.

You can learn more about DeepComposer costs in the AWS DeepComposer pricing documentation

Access AWS DeepComposer console:
Click on DeepComposer link to get started: https://us-east-1.console.aws.amazon.com/deepcomposer

Enter AWS account ID, IAM Username and Password provided

Click Sign In

AWS sign in screen
Note: You must access the console in N.Virginia (us-east-1) AWS region You can use the dropdown to select the correct region.

Console Dropdown
Get Started:
Click Music Studio from the left navigation menu

AWS DeepComposer Music Studio
Choose an Input Melody

Choose Input Melody
Click play to play the default input melody

Click Play
Select Generative Adversarial Networks as the Generative AI technique.

Select Generative Adversarial Networks
Select MuseGAN as the Generative Algorithm.

Select a Model.

Select A Model
Click Generate composition to generate a composition and an AI generated composition will be created.

Click Generate Composition
Click play to play the new AI generated musical composition.

Input melody:
To create a custom melody, click record to start recording

Click record to start recording
and play the notes on the keyboard.

Use the keyboard
Mary Had a Little Lamb music
Click the stop button to stop recording when you are done.

Stop Recording
Play the recorded music to verify the input. In case you don’t like recorded music, you may start recording again by clicking record

Select Generative Adversarial Networks as the Generative AI technique.
Select MuseGAN as the Generative Algorithm.
Select a Model.
Click Generate composition to generate a composition and an AI generated composition will be created.

Select Jazz model from Model
Click play to play the composition and enjoy the AI generated music.

Try experimenting with different genres or sample input melody.

Congratulations! You have learned how to use pre-trained models to generate new music!

;

---

## **8. How DeepComposer Works**

🎥 [Udacity, Video Link](https://youtu.be/3Z3rG9pYd0w)

Model Training with AWS DeepComposer

AWS DeepComposer uses a GAN
DeepComposer GAN Model
Each iteration of the training cycle is called an epoch. The model is trained for thousands of epochs.

Loss Functions
In machine learning, the goal of iterating and completing epochs is to improve the output or prediction of the model. Any output that deviates from the ground truth is referred to as an error. The measure of an error, given a set of weights, is called a loss function. Weights represent how important an associated feature is to determining the accuracy of a prediction, and loss functions are used to update the weights after every iteration. Ideally, as the weights update, the model improves making less and less errors. Convergence happens once the loss functions stabilize.

We use loss functions to measure how closely the output from the GAN models match the desired outcome. Or, in the case of DeepComposer, how well does DeepComposer's output music match the training music. Once the loss functions from the Generator and Discriminator converges, this indicates the GAN model is no longer learning, and we can stop its training.

We also measures the quality of the music generated by DeepComposer via additional quantitative metrics, such as drum pattern and polyphonic rate.

GAN loss functions have many fluctuations early on due to the “adversarial” nature of the generator and discriminator.

Over time, the loss functions stabilizes to a point, we call this convergence. This convergence can be zero, but doesn’t have to be.

QUESTION 1 OF 2
True or False: Loss functions help us determine when to stop training a model

True

QUESTION 2 OF 2
What does it mean when a loss function reaches convergence?

The value of the function is stable over many epochs

AWS DeepComposer diagram
AWS DeepComposer Under The Hood

How It Works
Input melody captured on the AWS DeepComposer console
Console makes a backend call to AWS DeepComposer APIs that triggers an execution Lambda.
Book-keeping is recorded in Dynamo DB.
The execution Lambda performs an inference query to SageMaker which hosts the model and the training inference container.
The query is run on the Generative AI model.
The model generates a composition.
The generated composition is returned.
The user can hear the composition in the console.
The user can share the composition to SoundCloud.
;

---

## **9. Training Architecture**

🎥 [Udacity, Video Link](https://youtu.be/cUYYsxNHYbQ)

How to measure the quality of the music we’re generating:
We can monitor the loss function to make sure the model is converging
We can check the similarity index to see how close is the model to mimicking the style of the data. When the graph of the similarity index smoothes out and becomes less spikey, we can be confident that the model is converging
We can listen to the music created by the generated model to see if it's doing a good job. The musical quality of the model should improve as the number of training epochs increases.
Training architecture
User launch a training job from the AWS DeepComposer console by selecting hyperparameters and data set filtering tags
The backend consists of an API Layer (API gateway and lambda) write request to DynamoDB
Triggers a lambda function that starts the training workflow
It then uses AWS Step Funcitons to launch the training job on Amazon SageMaker
Status is continually monitored and updated to DynamoDB
The console continues to poll the backend for the status of the training job and update the results live so users can see how the model is learning
AWS SageMaker Training Architecture
Training Architecture

Challenges with GANs
Clean datasets are hard to obtain
Not all melodies sound good in all genres
Convergence in GAN is tricky – it can be fleeting rather than being a stable state
Complexity in defining meaningful quantitive metrics to measure the quality of music created
QUESTION 1 OF 2
True or False: We expect the Similarity Index to reach zero

Reflect on how training works
Why might we want to use more than one method to evaluate the quality of our model output?

Enter your response here, there's no right or wrong answer
;

---

## **10. Generative AI Overview**

🎥 [Udacity, Video Link]()

Generative AI
Generative AI has been described as one of the most promising advances in AI in the past decade by the MIT Technology Review.

Generative AI opens the door to an entire world of creative possibilities with practical applications emerging across industries, from turning sketches into images for accelerated product development, to improving computer-aided design of complex objects.

For example, Glidewell Dental is training a generative adversarial network adept at constructing detailed 3D models from images. One network generates images and the second inspects those images. This results in an image that has even more anatomical detail than the original teeth they are replacing.

Glidewell Dental is training GPU powered GANs to create dental crown models
Glidewell Dental is training GPU powered GANs to create dental crown models

Generative AI enables computers to learn the underlying pattern associated with a provided input (image, music, or text), and then they can use that input to generate new content. Examples of Generative AI techniques include Generative Adversarial Networks (GANs), Variational Autoencoders, and Transformers.

What are GANs?
GANs, a generative AI technique, pit 2 networks against each other to generate new content. The algorithm consists of two competing networks: a generator and a discriminator.

A generator is a convolutional neural network (CNN) that learns to create new data resembling the source data it was trained on.

The discriminator is another convolutional neural network (CNN) that is trained to differentiate between real and synthetic data.

The generator and the discriminator are trained in alternating cycles such that the generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

AWS DeepComposer GAN
A schema representing a AWS DeepComposer GAN

Like the collaboration between an orchestra and its conductor
The best way we’ve found to explain this is to use the metaphor of an orchestra and conductor. An orchestra doesn’t create amazing music the first time they get together. They have a conductor who both judges their output, and coaches them to improve. So an orchestra, trains, practices, and tries to generate polished music, and then the conductor works with them, as both judge and coach.

The conductor is both judging the quality of the output (were the right notes played with the right tempo) and at the same time providing feedback and coaching to the orchestra (“strings, more volume! Horns, softer in this part! Everyone, with feeling!”). Specifically to achieve a style that the conductor knows about. So, the more they work together the better the orchestra can perform.

The Generative AI that AWS DeepComposer teaches developers about uses a similar concept. We have two machine learning models that work together in order to learn how to generate musical compositions in distinctive styles.

Conductor and Orchestra
As a conductor provides feedback to make an orchestra sound better, a GAN's discriminator gives the generator feedback on how to make its data more realistic

QUIZ QUESTION
Please identify which of the following statements are TRUE about a Generative Adverserial Network (GAN).

The generator and discriminator are occurring asynchronously.

The generator is learning to produce more realistic data and the discriminator is learning to differentiate real data from the newly created data.

;

---

## **11. Introduction to U-Net Architecture**

🎥 [Udacity, Video Link]()

Training a machine learning model using a dataset of Bach compositions
AWS DeepComposer uses GANs to create realistic accompaniment tracks. When you provide an input melody, such as twinkle-twinkle little star, using the keyboard U-Net will add three additional piano accompaniment tracks to create a new musical composition.

The U-Net architecture uses a publicly available dataset of Bach’s compositions for training the GAN. In AWS DeepComposer, the generator network learns to produce realistic Bach-syle music while the discriminator uses real Bach music to differentiate between real music compositions and newly created ones

The U-Net architecture learns from symphonies to create music
Listen to sample of Bach's music from the training dataset
Bach training sample 1

Bach training sample 2

Symphony-inspired composition created by U-Net architecture
Input melody

Generated composition

Apply your learning in AWS DeepComposer
Try generating a musical composition in Music studio

How U-Net based model interprets music
Music is written out as a sequence of human readable notes. Experts have not yet discovered a way to translate the human readable format in such a way that computers can understand it. Modern GAN-based models instead treat music as a series of images, and can therefore leverage existing techniques within the computer vision domain.

In AWS DeepComposer, we represent music as a two-dimensional matrix (also referred to as a piano roll) with “time” on the horizontal axis and “pitch” on the vertical axis. You might notice this representation looks similar to an image. A one or zero in any particular cell in this grid indicates if a note was played or not at that time for that pitch.

Piano Rolls
The piano roll format discretizes music into small buckets of time and pitch

QUIZ QUESTION
Which of the following statements about the application with Bach-style model is incorrect?

The generator is trained on realistic Bach music to output new music piece

;

---

## **12. Model Architecture**

🎥 [Udacity, Video Link]()

As described in previous sections, a GAN consists of 2 networks: a generator and a discriminator. Let’s discuss the generator and discriminator networks used in AWS DeepComposer.

Generator
The generator network used in AWS DeepComposer is adapted from the U-Net architecture, a popular convolutional neural network that is used extensively in the computer vision domain. The network consists of an “encoder” that maps the single track music data (represented as piano roll images) to a relatively lower dimensional “latent space“ and a ”decoder“ that maps the latent space back to multi-track music data.

Here are the inputs provided to the generator:

Single-track piano roll: A single melody track is provided as the input to the generator.
Noise vector: A latent noise vector is also passed in as an input and this is responsible for ensuring that there is a flavor to each output generated by the generator, even when the same input is provided.
U-Net Architecture diagram
Notice that the encoding layers of the generator on the left side and decoder layer on on the right side are connected to create a U-shape, thereby giving the name U-Net to this architecture

Discriminator
The goal of the discriminator is to provide feedback to the generator about how realistic the generated piano rolls are, so that the generator can learn to produce more realistic data. The discriminator provides this feedback by outputting a scalar value that represents how “real” or “fake” a piano roll is.

Since the discriminator tries to classify data as “real” or “fake”, it is not very different from commonly used binary classifiers. We use a simple architecture for the critic, composed of four convolutional layers and a dense layer at the end.

Discriminator network architecture consisting of four convolutional layers and a dense layer
Discriminator network architecture consisting of four convolutional layers and a dense layer

Once you complete building your model architecture, the next step is training.

QUESTION 1 OF 4
What is another term for the “discriminator” in a GAN?

QUESTION 2 OF 4
Which attribute(s) does a discriminator judge?

QUESTION 3 OF 4
What would happen without “input noise”?

QUESTION 4 OF 4
Order the steps in U-Net architecture

ORDER

STEP

1

2

3

4

5

;

---

## **13. Training Methodology**

During training, the generator and discriminator work in a tight loop as following:

Generator
The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
The discriminator then takes these generated music tracks and predicts how far it deviates from the real data present in your training dataset.
Discriminator
This feedback from the discriminator is used by the generator to update its weights. As the generator gets better at creating music accompaniments, it begins fooling the discriminator. So, the discriminator needs to be retrained as well.
Beginning with the discriminator on the first iteration, we alternate between training these two networks until we reach some stop condition (ex: the algorithm has seen the entire dataset a certain number of times).
Finer control of AWS DeepComposer with hyperparameters
As you explore training your own custom model in the AWS DeepComposer console, you will notice you have access to several hyperparameters for finer control of this process. Here are a few details on each to help guide your exploration.

Number of epochs
When the training loop has passed through the entire training dataset once, we call that one epoch. Training for a higher number of epochs will mean your model will take longer to complete its training task, but it may produce better output if it has not yet converged. You will learn how to determine when a model has completed most of its training in the next section.

Training over more epochs will take longer but can lead to a better sounding musical output
Model training is a trade-off between the number of epochs (i.e. time) and the quality of sample output.

Sample training input

Sample output at 100 epochs

Sample output at 400 epochs

Apply your learning in AWS DeepComposer
Try training a custom model using less than 250 epochs

Learning Rate
The learning rate controls how rapidly the weights and biases of each network are updated during training. A higher learning rate might allow the network to explore a wider set of model weights, but might pass over more optimal weights.

Update ratio
A ratio of the number of times the discriminator is updated per generator training epoch. Updating the discriminator multiple times per generator training epoch is useful because it can improve the discriminators accuracy. Changing this ratio might allow the generator to learn more quickly early-on, but will increase the overall training time.

While we provide sensible defaults for these hyperparameters in the AWS DeepComposer console, you are encouraged to explore other settings to see how each changes your model’s performance and time required to finish training your model.

QUIZ QUESTION
What is a hyperparameter?

A parameter whose value is set before the training process begins

---

## **14. Evaluation**

Typically when training any sort of model, it is a standard practice to monitor the value of the loss function throughout the duration of the training. The discriminator loss has been found to correlate well with sample quality. You should expect the discriminator loss to converge to zero and the generator loss to converge to some number which need not be zero. When the loss function plateaus, it is an indicator that the model is no longer learning. At this point, you can stop training the model. You can view these loss function graphs in the AWS DeepComposer console.

Sample output quality improves with more training
After 400 epochs of training, discriminator loss approaches near zero and the generator converges to a steady-state value. Loss is useful as an evaluation metric since the model will not improve as much or stop improving entirely when the loss plateaus.

Loss Function Graph
Sample output at 400 epochs features elements of the training dataset

Related audio
Sample training input

Sample output at 100 epochs

Sample output at 400 epochs

Apply your learning in AWS DeepComposer
Listen to the training sample output for the Pop sample model

While standard mechanisms exist for evaluating the accuracy of more traditional models like classification or regression, evaluating generative models is an active area of research. Within the domain of music generation, this hard problem is even less well-understood.

To address this, we take high-level measurements of our data and show how well our model produces music that aligns with those measurements. If our model produces music which is close to the mean value of these measurements for our training dataset, our music should match the general “shape”. You’ll see graphs of these measurements within the AWS DeepComposer console

Here are a few such measurements:

Empty bar rate: The ratio of empty bars to total number of bars.
Number of pitches used: A metric that captures the distribution and position of pitches.
In Scale Ratio: Ratio of the number of notes that are in the key of C, which is a common key found in music, to the total number of notes.
Music to your ears
Of course, music is much more complex than a few measurements. It is often important to listen directly to the generated music to better understand changes in model performance. You’ll find this final mechanism available as well, allowing you to listen to the model outputs as it learns.

Once training has completed, you may use the model created by the generator network to create new musical compositions.

---

## **15. Inference**

Once this model is trained, the generator network alone can be run to generate new accompaniments for a given input melody. If you recall, the model took as input a single-track piano roll representing melody and a noise vector to help generate varied output.

The final process for music generation then is as follows:

Transform single-track music input into piano roll format.
Create a series of random numbers to represent the random noise vector.
Pass these as input to our trained generator model, producing a series of output piano rolls. Each output piano roll represents some instrument in the composition.
Transform the series of piano rolls back into a common music format (MIDI), assigning an instrument for each track.
Inference performed on the input melody
Input sound

Output sound

Apply your learning in AWS DeepComposer
Try generating a musical composition in the symphony genre

To explore this process firsthand, try loading a model in the music studio, using a sample model if you have not trained your own. After selecting from a prepared list of input melodies or recording your own, you may choose “Generate a composition” to generate accompaniments.

Explore Generative AI Further
Create compositions using sample models in music studio
Inspect the training of existing sample models
Train your own model within the AWS DeepComposer console
Build your own GAN model

---

## **16. Build a Custom GAN Part 1: Notebooks and Data Preparation**

🎥 [Udacity, Video Link](https://youtu.be/8YpQiiVBwqE)

In this demonstration we’re going to synchronize what you’ve learned about software development practices and machine learning, using AWS DeepComposer to explore those best practices against a real life use case.

Coding Along With The Instructor (Optional)
To create the custom GAN, you will need to use an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the demo and build a custom GAN, you may incur a cost.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation

Getting Started

Setting Up the DeepComposer Notebook
To get to the main Amazon SageMaker service screen, navigate to the AWS SageMaker console. You can also get there from within the AWS Management Console by searching for Amazon SageMaker.
Once inside the SageMaker console, look to the left hand menu and select Notebook Instances.
Next, click on Create notebook instance.
In the Notebook instance setting section, give the notebook a name, for example, DeepComposerUdacity.
Based on the kind of CPU, GPU and memory you need the next step is to select an instance type. For our purposes, we’ll configure a ml.c5.4xlarge
Leave the Elastic Inference defaulted to none.
In the Permissions and encryption section, create a new IAM role using all of the defaults.
When you see that the role was created successfully, navigate down a little ways to the Git repositories section
Select Clone a public Git repository to this notebook instance only
Copy and paste the public URL into the Git repository URL section: https://github.com/aws-samples/aws-deepcomposer-samples
Select Create notebook instance
Give SageMaker a few minutes to provision the instance and clone the Git repository
Exploring the Notebook
Now that it’s configured and ready to use, let’s take a moment to investigate what’s inside the notebook.

When the status reads "InService" you can open the Jupyter notebook.

Status is InService
Open the Notebook
Click Open Jupyter.
When the notebook opens, click on Lab 2.
When the lab opens click on GAN.ipynb.
Review: Generative Adversarial Networks (GANs).
GANs consist of two networks constantly competing with each other:

Generator network that tries to generate data based on the data it was trained on.
Discriminator network that is trained to differentiate between real data and data which is created by the generator.
Note: The demo often refers to the discriminator as the critic. The two terms can be used interchangeably.

Set Up the Project
Run the first Dependencies cell to install the required packages
Run the second Dependencies cell to import the dependencies
Run the Configuration cell to define the configuration variables
Note: While executing the cell that installs dependency packages, you may see warning messages indicating that later versions of conda are available for certain packages. It is completely OK to ignore this message. It should not affect the execution of this notebook.

Click run
Click Run or Shift-Enter in the cell

Good Coding Practices
Do not hard-code configuration variables
Move configuration variables to a separate config file
Use code comments to allow for easy code collaboration
Data Preparation
The next section of the notebook is where we’ll prepare the data so it can train the generator network.

Why Do We Need to Prepare Data?
Data often comes from many places (like a website, IoT sensors, a hard drive, or physical paper) and it’s usually not clean or in the same format. Before you can better understand your data, you need to make sure it’s in the right format to be analyzed. Thankfully, there are library packages that can help! One such library is called NumPy, which was imported into our notebook.

Piano Roll Format
The data we are preparing today is music and it comes formatted in what’s called a “piano roll”. Think of a piano roll as a 2D image where the X-axis represents time and the Y-axis represents the pitch value. Using music as images allows us to leverage existing techniques within the computer vision domain.

Our data is stored as a NumPy Array, or grid of values. Our dataset comprises 229 samples of 4 tracks (all tracks are piano). Each sample is a 32 time-step snippet of a song, so our dataset has a shape of:

(num_samples, time_steps, pitch_range, tracks)
or

(229, 32, 128, 4)
Piano Roll visualization
Each Piano Roll Represents A Separate Piano Track in the Song

Load and View the Dataset
Run the next cell to play a song from the dataset.
Run the next cell to load the dataset as a nympy array and output the shape of the data to confirm that it matches the (229, 32, 128, 4) shape we are expecting
Run the next cell to see a graphical representation of the data.
Graphical representation of model data
Graphical Representation of Model Data

Create a Tensorflow Dataset
Much like there are different libraries to help with cleaning and formatting data, there are also different frameworks. Some frameworks are better suited for particular kinds of machine learning workloads and for this deep learning use case, we’re going to use a Tensorflow framework with a Keras library.

We'll use the dataset object to feed batches of data into our model.

Run the first Load Data cell to set parameters.
Run the second Load Data cell to prepare the data.

---

## **17. Build a Custom GAN Part 2: Training and Evaluation**

🎥 [Udacity, Video Link](https://youtu.be/ZLQkkxmSxho)

🎥 [Udacity, Video Link](https://youtu.be/cSzqlqZXrSg)

Model Architecture
Before we can train our model, let’s take a closer look at model architecture including how GAN networks interact with the batches of data we feed it, and how they communicate with each other.

How the Model Works
The model consists of two networks, a generator and a critic. These two networks work in a tight loop:

The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
The discriminator evaluates the generated music tracks and predicts how far they deviate from the real data in the training dataset.
The feedback from the discriminator is used by the generator to help it produce more realistic music the next time.
As the generator gets better at creating better music and fooling the discriminator, the discriminator needs to be retrained by using music tracks just generated by the generator as fake inputs and an equivalent number of songs from the original dataset as the real input.
We alternate between training these two networks until the model converges and produces realistic music.
The discriminator is a binary classifier which means that it classifies inputs into two groups, e.g. “real” or “fake” data.

Defining and Building Our Model
Run the cell that defines the generator
Run the cell that builds the generator
Run the cell that defines the discriminator
Run the cell that builds the discriminator
Model Training and Loss Functions

As the model tries to identify data as “real” or “fake”, it’s going to make errors. Any prediction different than the ground truth is referred to as an error.

The measure of the error in the prediction, given a set of weights, is called a loss function. Weights represent how important an associated feature is to determining the accuracy of a prediction.

Loss functions are an important element of training a machine learning model because they are used to update the weights after every iteration of your model. Updating weights after iterations optimizes the model making the errors smaller and smaller.

Setting Up and Running the Model Training
Run the cell that defines the loss functions
Run the cell to set up the optimizer
Run the cell to define the generator step function
Run the cell to define the discriminator step function
Run the cell to load the melody samples
Run the cell to set the parameters for the training
Run the cell to train the model!!!!
Training and tuning models can take a very long time – weeks or even months sometimes. Our model will take around an hour to train.

Model Evaluation
Now that the model has finished training it’s time to evaluate its results.

There are several evaluation metrics you can calculate for classification problems and typically these are decided in the beginning phases as you organize your workflow.

In our example we:

Checked to see if the losses for the networks are converging
Looked at commonly used musical metrics of the generated sample and compared them to the training dataset.
Evaluating Our Training Results
Run the cell to restore the saved checkpoint. If you don't want to wait to complete the training you can use data from a pre-trained model by setting TRAIN = False in the cell.
Run the cell to plot the losses.
Run the cell to plot the metrics.
Results and Inference
Finally, we are ready to hear what the model produced and visualize the piano roll output!

Once the model is trained and producing acceptable quality, it’s time to see how it does on data it hasn’t seen. We can test the model on these unknown inputs, using the results as a proxy for performance on future data.

Evaluate the Generated Music
In the first cell, enter 0 as the iteration number:

iteration = 0
run the cell and play the music snippet.
Or listen to this example snippet from iteration 0:

In the second cell, enter 0 as the iteration number:

iteration = 0
run the cell and display the piano roll.

Piano roll at iteration 0
Example Piano Roll at Iteration 0

In the first cell, enter 500 as the iteration number:
iteration = 500
run the cell and play the music snippet.
Or listen to the example snippet at iteration 500.
In the second cell, enter 500 as the iteration number:
iteration = 500
run the cell and display the piano roll.
Example Piano Roll at Iteration 500
Example Piano Roll at Iteration 500

Play around with the iteration number and see how the output changes over time!

Here is an example snippet at iteration 950

And here is the piano roll:

Example Piano Roll at Iteration 950
Example Piano Roll at Iteration 950

Do you see or hear a quality difference between iteration 500 and iteration 950?

Watch the Evolution of the Model!
Run the next cell to create a video to see how the generated piano rolls change over time.
Or watch the example video here:

Inference
Now that the GAN has been trained we can run it on a custom input to generate music.

Run the cell to generate a new song based on "Twinkle Twinkle Little Star".
Or listen to the example of the generated music here:

Run the next cell and play the generated music.
Or listen to the example of the generated music here:

Stop and Delete the Jupyter Notebook When You Are Finished!
This project is not covered by the AWS Free Tier so your project will continue to accrue costs as long as it is running.

Follow these steps to stop and delete the notebook.

Go back to the Amazon SageMaker console.
Select the notebook and click Actions.
Select Actions
Select Stop and wait for the instance to stop.
Select Stop
Select Delete

Recap
In this demo we learned how to setup a Jupyter notebook in Amazon SageMaker, about machine learning code in the real world, and what data preparation, model training, and model evaluation can look in a notebook instance. While this was a fun use case for us to explore, the concepts and techniques can be applied to other machine learning projects like an object detector or a sentiment analysis on text.

;

---

## **18. Lesson Recap**

🎥 [Udacity, Video Link](https://youtu.be/FdxIAs7vMW8)

Course Outline
What You Have Learned

Thanks for joining us for this lesson. We hope you continue to learn more about Machine Learning!

;

---

---

## Foam Related Link

- [[gan]]
