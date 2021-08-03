# ML w/ AWS

## 🎓 [Udacity, ND065, Lesson 3, Machine Learning with AWS](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/8b79bd0c-6a77-40bc-8f96-b669c36d6103/concepts/f5654891-eb15-4bcc-ac94-aaf812d0ac7d)

---

## **1. ML with AWS**

### Why AWS?

🎥 [Udacity, Video Link](https://youtu.be/ZBao7oT6BGs)

The AWS achine learning mission is to put machine learning in the hands of every developer.

AWS offers the broadest and deepest set of artificial intelligence (AI) and machine learning (ML) services with unmatched flexibility.
You can accelerate your adoption of machine learning with AWS SageMaker. Models that previously took months to build and required specialized expertise can now be built in weeks or even days.
AWS offers the most comprehensive cloud offering optimized for machine learning.
More machine learning happens at AWS than anywhere else.
AWS Machine Learning offerings

AWS AI services
By using AWS pre-trained AI services, you can apply ready-made intelligence to a wide range of applications such as personalized recommendations, modernizing your contact center, improving safety and security, and increasing customer engagement.

Industry-specific solutions
With no knowledge in machine learning needed, add intelligence to a wide range of applications in different industries including healthcare and manufacturing.

AWS Machine Learning services
With AWS, you can build, train, and deploy your models fast. Amazon SageMaker is a fully managed service that removes complexity from ML workflows so every developer and data scientist can deploy machine learning for a wide range of use cases.

ML infrastructure and frameworks
AWS Workflow services make it easier for you to manage and scale your underlying ML infrastructure.

ML infrastructure and frameworks
ML infrastructure and frameworks

Getting started
In addition to educational resources such as AWS Training and Certification, AWS has created a portfolio of educational devices to help put new machine learning techniques into the hands of developers in unique and fun ways, with AWS DeepLens, AWS DeepRacer, and AWS DeepComposer.

AWS DeepLens: A deep learning–enabled video camera
AWS DeepRacer: An autonomous race car designed to test reinforcement learning models by racing on a physical track
AWS DeepComposer: A composing device powered by generative AI that creates a melody that transforms into a completely original song
AWS ML Training and Certification: Curriculum used to train Amazon developers
AWS educational devices
AWS educational devices

Additional Reading
To learn more about AWS AI Services, see Explore AWS AI services.
To learn more about AWS ML Training and Certification offerings, see Training and Certification.

---

## **2. Lesson Overview**

🎥 [Udacity, Video Link](https://youtu.be/Hx7y7JKNE2I)

In this lesson, you'll get an introduction to machine learning (ML) with AWS and AWS AI devices: AWS DeepLens, AWS DeepComposer, and AWS DeepRacer. Learn the basics of computer vision with AWS DeepLens, race around a track and get familiar with reinforcement learning with AWS DeepRacer, and discover the power of generative AI by creating music using AWS DeepComposer.

The outline of the lesson
The lesson outline

By the end of the lesson, you will be able to:

Identify AWS machine learning offerings and understand how different services are used for different applications.
Explain the fundamentals of computer vision and provide examples of popular tasks.
Describe how reinforcement learning works in the context of AWS DeepRacer.
Explain the fundamentals of generative AI and its applications, and describe three famous generative AI models in the context of music and AWS DeepComposer.
We hope this will be an interesting and exciting learning experience for you.

Let's get started!

---

## **3. AWS Account Requirements**

An AWS account is required
To complete the exercises in this course, you need an AWS Account ID.

To set up a new AWS Account ID, follow the directions in How do I create and activate a new Amazon Web Services account?

You are required to provide a payment method when you create the account. To learn about which services are available at no cost, see the AWS Free Tier documentation.

Will these exercises cost anything?
This lesson contains many demos and exercises. You do not need to purchase any AWS devices to complete the lesson. However, please carefully read the following list of AWS services you may need in order to follow the demos and complete the exercises.

Train your computer vision model with AWS DeepLens (optional)
To train and deploy custom models to AWS DeepLens, you use Amazon SageMaker. Amazon SageMaker is a separate service and has its own service pricing and billing tier. It's not required to train a model for this course. If you're interested in training a custom model, please note that it incurs a cost. To learn more about SageMaker costs, see the Amazon SageMaker Pricing.
Train your reinforcement learning model with AWS DeepRacer
To get started with AWS DeepRacer, you receive 10 free hours to train or evaluate models and 5GB of free storage during your first month. This is enough to train your first time-trial model, evaluate it, tune it, and then enter it into the AWS DeepRacer League. This offer is valid for 30 days after you have used the service for the first time.
Beyond 10 hours of training and evaluation, you pay for training, evaluating, and storing your machine learning models. Charges are based on the amount of time you train and evaluate a new model and the size of the model stored. To learn more about AWS DeepRacer pricing, see the AWS DeepRacer Pricing
Generate music using AWS DeepComposer
To get started, AWS DeepComposer provides a 12-month Free Tier for first-time users. With the Free Tier, you can perform up to 500 inference jobs translating to 500 pieces of music using the AWS DeepComposer Music studio. You can use one of these instances to complete the exercise at no cost. To learn more about AWS DeepComposer costs, see the AWS DeepComposer Pricing.
Build a custom generative AI model (GAN) using Amazon SageMaker (optional)
Amazon SageMaker is a separate service and has its own service pricing and billing tier. To train the custom generative AI model, the instructor uses an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the instructor and train your own custom model, you may incur a cost. Please note, that creating your own custom model is completely optional. You are not required to do this exercise to complete the course. To learn more about SageMaker costs, see the Amazon SageMaker Pricing.
Please confirm that you have taken the necessary steps to complete this lesson.

Task List

---

## **4. Computer Vision and Its Applications**

🎥 [Udacity, Video Link](https://youtu.be/LKV97Js0QFI)

This section introduces you to common concepts in computer vision (CV), and explains how you can use AWS DeepLens to start learning with computer vision projects. By the end of this section, you will be able to explain how to create, train, deploy, and evaluate a trash-sorting project that uses AWS DeepLens.

Introduction to Computer Vision

Summary
Computer vision got its start in the 1960s in academia. Since its inception, it has been an interdisciplinary field. Machine learning practitioners use computers to understand and automate tasks associated with the visual word.

Modern-day applications of computer vision use neural networks. These networks can quickly be trained on millions of images and produce highly accurate predictions.

Since 2010, there has been exponential growth in the field of computer vision. You can start with simple tasks like image classification and objection detection and then scale all the way up to the nearly real-time video analysis required for self-driving cars to work at scale.

In the video, you have learned:

How computer vision got started
Early applications of computer vision needed hand-annotated images to successfully train a model.
These early applications had limited applications because of the human labor required to annotate images.
Three main components of neural networks
Input Layer: This layer receives data during training and when inference is performed after the model has been trained.
Hidden Layer: This layer finds important features in the input data that have predictive power based on the labels provided during training.
Output Layer: This layer generates the output or prediction of your model.
Modern computer vision
Modern-day applications of computer vision use neural networks call convolutional neural networks or CNNs.
In these neural networks, the hidden layers are used to extract different information about images. We call this process feature extraction.
These models can be trained much faster on millions of images and generate a better prediction than earlier models.
How this growth occured
Since 2010, we have seen a rapid decrease in the computational costs required to train the complex neural networks used in computer vision.
Larger and larger pre-labeled datasets have become generally available. This has decreased the time required to collect the data needed to train many models.
Computer Vision Applications

Summary
Computer vision (CV) has many real-world applications. In this video, we cover examples of image classification, object detection, semantic segmentation, and activity recognition. Here's a brief summary of what you learn about each topic in the video:

Image classification is the most common application of computer vision in use today. Image classification can be used to answer questions like What's in this image? This type of task has applications in text detection or optical character recognition (OCR) and content moderation.
Object detection is closely related to image classification, but it allows users to gather more granular detail about an image. For example, rather than just knowing whether an object is present in an image, a user might want to know if there are multiple instances of the same object present in an image, or if objects from different classes appear in the same image.
Semantic segmentation is another common application of computer vision that takes a pixel-by-pixel approach. Instead of just identifying whether an object is present or not, it tries to identify down the pixel level which part of the image is part of the object.
Activity recognition is an application of computer vision that is based around videos rather than just images. Video has the added dimension of time and, therefore, models are able to detect changes that occur over time.
New Terms
Input Layer: The first layer in a neural network. This layer receives all data that passes through the neural network.
Hidden Layer: A layer that occurs between the output and input layers. Hidden layers are tailored to a specific task.
Output Layer: The last layer in a neural network. This layer is where the predictions are generated based on the information captured in the hidden layers.
Additional Reading
You can use the AWS DeepLens Recipes website to find different learning paths based on your level of expertise. For example, you can choose either a student or teacher path. Additionally, you can choose between beginner, intermediate, and advanced projects which have been created and vetted by the AWS DeepLens team.
You can check out the AWS machine learning blog to learn about recent advancements in machine learning. Additionally, you can use the AWS DeepLens tag to see projects which have been created by the AWS DeepLens team.
Ready to get started? Check out the Getting started guide in the AWS DeepLens Developer Guide.

---

## **5. Computer Vision with AWS DeepLens**

🎥 [Udacity, Video Link](https://youtu.be/XIrF8m7xEuI)

AWS DeepLens
AWS DeepLens allows you to create and deploy end-to-end computer vision–based applications. The following video provides a brief introduction to how AWS DeepLens works and how it uses other AWS services.

Summary
AWS DeepLens is a deep learning–enabled camera that allows you to deploy trained models directly to the device. You can either use sample templates and recipes or train your own model.

AWS DeepLens is integrated with several AWS machine learning services and can perform local inference against deployed models provisioned from the AWS Cloud. It enables you to learn and explore the latest artificial intelligence (AI) tools and techniques for developing computer vision applications based on a deep learning model.

The AWS DeepLens device
The AWS DeepLens camera is powered by an Intel® Atom processor, which can process 100 billion floating-point operations per second (GFLOPS). This gives you all the computing power you need to perform inference on your device. The micro HDMI display port, audio out, and USB ports allow you to attach peripherals, so you can get creative with your computer vision applications.

You can use AWS DeepLens as soon as you register it.

An AWS DeepLens Device
An AWS DeepLens Device

An image showing how AWS DeepLens works
How AWS DeepLens works

How AWS DeepLens works
AWS DeepLens is integrated with multiple AWS services. You use these services to create, train, and launch your AWS DeepLens project. You can think of an AWS DeepLens project as being divided into two different streams as the image shown above.

First, you use the AWS console to create your project, store your data, and train your model.
Then, you use your trained model on the AWS DeepLens device. On the device, the video stream from the camera is processed, inference is performed, and the output from inference is passed into two output streams:
Device stream – The video stream passed through without processing.
Project stream – The results of the model's processing of the video frames.
Additional Reading
To learn more about the specifics of the AWS DeepLens device, see the AWS DeepLens Hardware Specifications in the AWS DeepLens Developer Guide.
You can buy an AWS DeepLens device on Amazon.com.

---

## **6. A Sample Project with AWS DeepLens**

🎥 [Udacity, Video Link](https://youtu.be/CZAtmy69_50)

This section provides a hands-on demonstration of a project created as part of an AWS DeepLens sponsored hack-a-thon. In this project, we use an AWS DeepLens device to do an image classification–based task. We train a model to detect if a piece of trash is from three potential classes: landfill, compost, or recycling.

Summary
AWS DeepLens is integrated with multiple AWS services. You use these services to create, train, and launch your AWS DeepLens project. To create any AWS DeepLens–based project you will need an AWS account.

Four key components are required for an AWS DeepLens–based project.

Collect your data: Collect data and store it in an Amazon S3 bucket.
Train your model: Use a Jupyter Notebook in Amazon SageMaker to train your model.
Deploy your model: Use AWS Lambda to deploy the trained model to your AWS DeepLens device.
View model output: Use Amazon IoT Greenrass to view your model's output after the model is deployed.
Machine Learning workflow review
The machine learning workflow contains several steps first introduced in Lesson 2. Let's briefly review the different steps and how they relate to the AWS DeepLens project.

Define the problem.
Using machine learning, we want to improve how trash is sorted. We're going to identify objects using a video stream, so we identify this as a computer vision–based problem.
We have access to data that already contains the labels, so we classify this as a supervised learning task.
Build the dataset.
Data is essential to any machine learning or computer vision–based project. Before going out and collecting lots of data, we investigate what kinds of data already exist and if they can be used for our application.
In this case, we have the data already collected and labeled.
Train the model.
Now that we have our data secured for this project, we use Amazon SageMaker to train our model. We cover specifics about this process in the demonstration video.
Evaluate the model.
Model training algorithms use loss functions to bring the model closer to its goals. The exact loss function and related details are outside the scope of this class, but the process is the same.
The loss function improves how well the model detects the different class images (compost, recycling, and landfill) while the model is being trained.
Use the model.
We deploy our trained model to our AWS DeepLens device, where inference is performed locally.
Demo: Using the AWS console to set up and deploy an AWS DeepLens project
The following video demonstrates the end-to-end application (trash-sorting project) discussed in the previous video. This video shows you how to complete this project using the AWS console.

Important

Storing data, training a model, and using AWS Lambda to deploy your model incur costs on your AWS account. For more information, see the AWS account requirements page.
You are not required to follow this demo on the AWS console. However, we recommend you watch it and understand the flow of completing a computer vision project with AWS DeepLens.
Demo Part 1: Getting Setup and Running the Code

Click here to download the Jupyer notebook the instructor used in the demo.

Summary: demo part 1
In this demo, you first saw how you can use Amazon S3 to store the image data needed for training your computer vision model. Then, you saw how to use Amazon SageMaker to train your model using a Jupyter Notebook

Demo Part 2: Deployment and Testing

Summary: demo part 2
Next, you used AWS Lambda to deploy your model onto an AWS DeepLens device. Finally, once your model has been deployed to your device, you can use AWS IoT Greengrass to view the inference output from your model actively running on your AWS DeepLens device.

More Projects on AWS DeepLens and Other AWS Services
In this blog post on the AWS Machine Learning blog, you learn about how computer vision–based applications can be used to protect workers in workplaces with autonomous robots. The post demonstrates you how you can create a virtual boundary using a computer and AWS DeepLens.
Using Amazon Rekognition and AWS DeepLens, you can create an application that uses OCR or optical character recognition to recognize a car's license plate, and open a garage door.
You can use Amazon Alexa and AWS DeepLens to create a Pictionary style game. First, you deploy a trained model to your AWS DeepLens which can recognize sketches drawn on a whiteboard and pair it with an Alexa skill that serves as the official scorekeeper.
Supporting Materials
Aws-Deeplens-Custom-Trash-Detector

---

## **7. Use the following image for quiz questions 1–3.**

Image for a quiz
Image for questions 1–3

QUESTION 1 OF 7
To detect both the cat and the dog present in this image, what kind of computer vision model would you use?

QUESTION 2 OF 7
Which computer vision–based task would you use to detect that the dog in the image is sleeping?

QUESTION 3 OF 7
Which computer vision–based task would you use to detect the exact location of the cat and dog in the image?

Image for Question 4
Image for question 4

QUESTION 4 OF 7
In the preceding image, which computer vision–based task would you use to identify where the people and the dog are?

Image for the quiz
Image for question 5

QUESTION 5 OF 7
In the preceding image, what kind of computer vision model would you use to count the number of cars?

QUESTION 6 OF 7
Which of the following are computer vision tasks? Select all the answers that apply.

QUESTION 7 OF 7
What type of computer vision task is the trash-sorting project?

---

## **8. Reinforcement Learning and Its Applications**

🎥 [Udacity, Video Link](https://youtu.be/UVnjiIYLUsQ)

This section introduces you to a type of machine learning (ML) called reinforcement learning (RL). You'll hear about its real-world applications and learn basic concepts using AWS DeepRacer as an example. By the end of the section, you will be able to create, train, and evaluate a reinforcement learning model in the AWS DeepRacer console.

Introduction to Reinforcement Learning

Summary
In reinforcement learning (RL), an agent is trained to achieve a goal based on the feedback it receives as it interacts with an environment. It collects a number as a reward for each action it takes. Actions that help the agent achieve its goal are incentivized with higher numbers. Unhelpful actions result in a low reward or no reward.

With a learning objective of maximizing total cumulative reward, over time, the agent learns, through trial and error, to map gainful actions to situations. The better trained the agent, the more efficiently it chooses actions that accomplish its goal.

Reinforcement Learning Applications

Summary
Reinforcement learning is used in a variety of fields to solve real-world problems. It’s particularly useful for addressing sequential problems with long-term goals. Let’s take a look at some examples.

RL is great at playing games:
Go (board game) was mastered by the AlphaGo Zero software.
Atari classic video games are commonly used as a learning tool for creating and testing RL software.
StarCraft II, the real-time strategy video game, was mastered by the AlphaStar software.
RL is used in video game level design:
Video game level design determines how complex each stage of a game is and directly affects how boring, frustrating, or fun it is to play that game.
Video game companies create an agent that plays the game over and over again to collect data that can be visualized on graphs.
This visual data gives designers a quick way to assess how easy or difficult it is for a player to make progress, which enables them to find that “just right” balance between boredom and frustration faster.
RL is used in wind energy optimization:
RL models can also be used to power robotics in physical devices.
When multiple turbines work together in a wind farm, the turbines in the front, which receive the wind first, can cause poor wind conditions for the turbines behind them. This is called wake turbulence and it reduces the amount of energy that is captured and converted into electrical power.
Wind energy organizations around the world use reinforcement learning to test solutions. Their models respond to changing wind conditions by changing the angle of the turbine blades. When the upstream turbines slow down it helps the downstream turbines capture more energy.
Other examples of real-world RL include:
Industrial robotics
Fraud detection
Stock trading
Autonomous driving
Some examples of real-world RL include: Industrial robotics, Fraud detection, Stock trading, and Autonomous driving
Some examples of real-world RL include: Industrial robotics, fraud detection, stock trading, and autonomous driving

New Terms
Agent: The piece of software you are training is called an agent. It makes decisions in an environment to reach a goal.
Environment: The environment is the surrounding area with which the agent interacts.
Reward: Feedback is given to an agent for each action it takes in a given state. This feedback is a numerical reward.
Action: For every state, an agent needs to take an action toward achieving its goal.

---

## **9. Reinforcement Learning with AWS DeepRacer**

🎥 [Udacity, Video Link](https://youtu.be/li-lJe3QWds)

Reinforcement Learning Concepts
In this section, we’ll learn some basic reinforcement learning terms and concepts using AWS DeepRacer as an example.

Summary
This section introduces six basic reinforcement learning terms and provides an example for each in the context of AWS DeepRacer.

Image contains icons representing the basic RL terms: Agent, Environment, State, Action, Reward, and Episode.
Basic RL terms: Agent, environment, state, action, reward, and episode

Agent

The piece of software you are training is called an agent.
It makes decisions in an environment to reach a goal.
In AWS DeepRacer, the agent is the AWS DeepRacer car and its goal is to finish \* laps around the track as fast as it can while, in some cases, avoiding obstacles.
Environment

The environment is the surrounding area within which our agent interacts.
For AWS DeepRacer, this is a track in our simulator or in real life.
State

The state is defined by the current position within the environment that is visible, or known, to an agent.
In AWS DeepRacer’s case, each state is an image captured by its camera.
The car’s initial state is the starting line of the track and its terminal state is when the car finishes a lap, bumps into an obstacle, or drives off the track.
Action

For every state, an agent needs to take an action toward achieving its goal.
An AWS DeepRacer car approaching a turn can choose to accelerate or brake and turn left, right, or go straight.
Reward

Feedback is given to an agent for each action it takes in a given state.
This feedback is a numerical reward.
A reward function is an incentive plan that assigns scores as rewards to different zones on the track.
Episode

An episode represents a period of trial and error when an agent makes decisions and gets feedback from its environment.
For AWS DeepRacer, an episode begins at the initial state, when the car leaves the starting position, and ends at the terminal state, when it finishes a lap, bumps into an obstacle, or drives off the track.
In a reinforcement learning model, an agent learns in an interactive real-time environment by trial and error using feedback from its own actions. Feedback is given in the form of rewards.

In a reinforcement learning model, an agent learns in an interactive real-time environment by trial and error using feedback from its own actions. Feedback is given in the form of rewards.
In a reinforcement learning model, an agent learns in an interactive real-time environment by trial and error using feedback from its own actions. Feedback is given in the form of rewards.

Putting Your Spin on AWS DeepRacer: The Practitioner's Role in RL

Summary
AWS DeepRacer may be autonomous, but you still have an important role to play in the success of your model. In this section, we introduce the training algorithm, action space, hyperparameters, and reward function and discuss how your ideas make a difference.

An algorithm is a set of instructions that tells a computer what to do. ML is special because it enables computers to learn without being explicitly programmed to do so.
The training algorithm defines your model’s learning objective, which is to maximize total cumulative reward. Different algorithms have different strategies for going about this.
A soft actor critic (SAC) embraces exploration and is data-efficient, but can lack stability.
A proximal policy optimization (PPO) is stable but data-hungry.
An action space is the set of all valid actions, or choices, available to an agent as it interacts with an environment.
Discrete action space represents all of an agent's possible actions for each state in a finite set of steering angle and throttle value combinations.
Continuous action space allows the agent to select an action from a range of values that you define for each sta te.
Hyperparameters are variables that control the performance of your agent during training. There is a variety of different categories with which to experiment. Change the values to increase or decrease the influence of different parts of your model.
For example, the learning rate is a hyperparameter that controls how many new experiences are counted in learning at each step. A higher learning rate results in faster training but may reduce the model’s quality.
The reward function's purpose is to encourage the agent to reach its goal. Figuring out how to reward which actions is one of your most important jobs.
Putting Reinforcement Learning into Action with AWS DeepRacer

Summary
This video put the concepts we've learned into action by imagining the reward function as a grid mapped over the race track in AWS DeepRacer’s training environment, and visualizing it as metrics plotted on a graph. It also introduced the trade-off between exploration and exploitation, an important challenge unique to this type of machine learning.

Each square is a state. The green square is the starting position, or initial state, and the finish line is the goal, or terminal state.
Each square is a state. The green square is the starting position, or initial state, and the finish line is the goal, or terminal state.

Key points to remember about reward functions:

Each state on the grid is assigned a score by your reward function. You incentivize behavior that supports your car’s goal of completing fast laps by giving the highest numbers to the parts of the track on which you want it to drive.
The reward function is the actual code you'll write to help your agent determine if the action it just took was good or bad, and how good or bad it was.
The squares containing exes are the track edges and defined as terminal states, which tell your car it has gone off track.
The squares containing exes are the track edges and defined as terminal states, which tell your car it has gone off track.

Key points to remember about exploration versus exploitation:

When a car first starts out, it explores by wandering in random directions. However, the more training an agent gets, the more it learns about an environment. This experience helps it become more confident about the actions it chooses.
Exploitation means the car begins to exploit or use information from previous experiences to help it reach its goal. Different training algorithms utilize exploration and exploitation differently.
Key points to remember about the reward graph:

While training your car in the AWS DeepRacer console, your training metrics are displayed on a reward graph.
Plotting the total reward from each episode allows you to see how the model performs over time. The more reward your car gets, the better your model performs.
Key points to remember about AWS DeepRacer:

AWS DeepRacer is a combination of a physical car and a virtual simulator in the AWS Console, the AWS DeepRacer League, and community races.
An AWS DeepRacer device is not required to start learning: you can start now in the AWS console. The 3D simulator in the AWS console is where training and evaluation take place.
New Terms
Exploration versus exploitation: An agent should exploit known information from previous experiences to achieve higher cumulative rewards, but it also needs to explore to gain new experiences that can be used in choosing the best actions in the future.
Additional Reading
If you are interested in more tips, workshops, classes, and other resources for improving your model, you'll find a wealth of resources on the AWS DeepRacer Pit Stop page.
For detailed step-by-step instructions and troubleshooting support, see the AWS DeepRacer Developer Documentation.
If you're interested in reading more posts on a range of DeepRacer topics as well as staying up to date on the newest releases, check out the AWS Discussion Forums.
If you're interested in connecting with a thriving global community of reinforcement learning racing enthusiasts, join the AWS DeepRacer Slack community.
If you're interested in tinkering with DeepRacer's open-source device software and collaborating with robotics innovators, check out our AWS DeepRacer GitHub Organization.

---

## **10. Demo: Reinforcement Learning with AWS DeepRacer**

🎥 [Udacity, Video Link](https://youtu.be/90VJxfnfR6c)

Important

To get you started with AWS DeepRacer, you receive 10 free hours to train or evaluate models and 5GB of free storage during your first month. This offer is valid for 30 days after you have used the service for the first time. Beyond 10 hours of training and evaluation, you pay for training, evaluating, and storing your machine learning models. Please read the AWS account requirements page for more information.
Demo Part 1: Create your car
Click here to go to the AWS DeepRacer console.

Summary
This demonstration introduces you to the AWS DeepRacer console and walks you through how to use it to build your first reinforcement learning model. You'll use your knowledge of basic reinforcement learning concepts and terminology to make choices about your model. In addition, you'll learn about the following features of the AWS DeepRacer service:

Pro and Open Leagues
Digital rewards
Racer profile
Garage
Sensor configuration
Race types
Time trial
Object avoidance
Head-to-head
Demo Part 2: Train your car

This demonstration walks you through the training process in the AWS DeepRacer console. You've learned about:

The reward graph
The training video
Demo Part 3: Testing your car

Summary
This demonstration walks the evaluation process in the AWS DeepRacer console.

Once you've created a successful model, you'll learn how to enter it into a race for the chance to win awards, prizes, and the opportunity to compete in the worldwide AWS DeepRacer Championship.

---

## **11. Quiz: Reinforcement Learning**

QUESTION 1 OF 5
In which type of machine learning are models trained using labeled data?

QUESTION 2 OF 5
In reinforcement learning, what is an "agent"?

QUESTION 3 OF 5
TRUE or FALSE: In reinforcement learning, "Exploration" is using experience to decide.

QUESTION 4 OF 5
How does a balance of "Exploration" and "Exploitation" help a reinforcement learning model?

QUESTION 5 OF 5

TERM

DEFINITION

State

Action

Episode

Reward

Environment

---

## **12. AWS DeepRacer Reinforcement Learning Exercise**

### Exercise: Interpret the reward graph of your first AWS DeepRacer model

### Instructions

Train a model in the AWS DeepRacer console and interpret its reward graph.

Part 1: Train a reinforcement learning model using the AWS DeepRacer console
Practice the knowledge you've learned by training your first reinforcement learning model using the AWS DeepRacer console.

If this is your first time using AWS DeepRacer, choose Get started from the service landing page, or choose Get started with reinforcement learning from the main navigation pane.
On the Get started with reinforcement learning page, under Step 2: Create a model and race, choose Create model. Alternatively, on the AWS DeepRacer home page, choose Your models from the main navigation pane to open the Your models page. On the Your models page, choose Create model.
On the Create model page, under Environment simulation, choose a track as a virtual environment to train your AWS DeepRacer agent. Then, choose Next. For your first run, choose a track with a simple shape and smooth turns. In later iterations, you can choose more complex tracks to progressively improve your models. To train a model for a particular racing event, choose the track most similar to the event track.
On the Create model page, choose Next.
On the Create Model page, under Race type, choose a training type. For your first run, choose Time trial. The agent with the default sensor configuration with a single-lens camera is suitable for this type of racing without modifications.
On the Create model page, under Training algorithm and hyperparameters, choose the Soft Actor Critic (SAC) or Proximal Policy Optimization (PPO) algorithm. In the AWS DeepRacer console, SAC models must be trained in continuous action spaces. PPO models can be trained in either continuous or discrete action spaces.
On the Create model page, under Training algorithm and hyperparameters, use the default hyperparameter values as is. Later on, to improve training performance, expand the hyperparameters and experiment with modifying the default hyperparameter values.
On the Create model page, under Agent, choose The Original DeepRacer or The Original DeepRacer (continuous action space) for your first model. If you use Soft Actor Critic (SAC) as your training algorithm, we filter your cars so that you can conveniently choose from a selection of compatible continuous action space agents.
On the Create model page, choose Next.
On the Create model page, under Reward function, use the default reward function example as is for your first model. Later on, you can choose Reward function examples to select another example function and then choose Use code to accept the selected reward function.
On the Create model page, under Stop conditions, leave the default Maximum time value as is or set a new value to terminate the training job to help prevent long-running (and possible run-away) training jobs. When experimenting in the early phase of training, you should start with a small value for this parameter and then progressively train for longer amounts of time.
On the Create model page, choose Create model to start creating the model and provisioning the training job instance.
After the submission, watch your training job being initialized and then run. The initialization process takes about 6 minutes to change status from Initializing to In progress.
Watch the Reward graph and Simulation video stream to observe the progress of your training job. You can choose the refresh button next to Reward graph periodically to refresh the Reward graph until the training job is complete.
Note: The training job is running on the AWS Cloud, so you don't need to keep the AWS DeepRacer console open during training. However, you can come back to the console to check on your model at any point while the job is in progress.

Part 2: Inspect your reward graph to assess your training progress
As you train and evaluate your first model, you'll want to get a sense of its quality. To do this, inspect your reward graph.

Find the following on your reward graph:

Average reward
Average percentage completion (training)
Average percentage completion (evaluation)
Best model line
Reward primary y-axis
Percentage track completion secondary y-axis
Iteration x-axis
Review the solution to this exercise for ideas on how to interpret it.

As you train and evaluate your first model, you'll want to get a sense of its quality. To do this, inspect your reward graph.
As you train and evaluate your first model, you'll want to get a sense of its quality. To do this, inspect your reward graph.

---

## **13.Exercise Solution: AWS DeepRacer**

To get a sense of how well your training is going, watch the reward graph. Here is a list of its parts and what they do:

Average reward
This graph represents the average reward the agent earns during a training iteration. The average is calculated by averaging the reward earned across all episodes in the training iteration. An episode begins at the starting line and ends when the agent completes one loop around the track or at the place the vehicle left the track or collided with an object. Toggle the switch to hide this data.
Average percentage completion (training)
The training graph represents the average percentage of the track completed by the agent in all training episodes in the current training. It shows the performance of the vehicle while experience is being gathered.
Average percentage completion (evaluation)
While the model is being updated, the performance of the existing model is evaluated. The evaluation graph line is the average percentage of the track completed by the agent in all episodes run during the evaluation period.
Best model line
This line allows you to see which of your model iterations had the highest average progress during the evaluation. The checkpoint for this iteration will be stored. A checkpoint is a snapshot of a model that is captured after each training (policy-updating) iteration.
Reward primary y-axis
This shows the reward earned during a training iteration. To read the exact value of a reward, hover your mouse over the data point on the graph.
Percentage track completion secondary y-axis
This shows you the percentage of the track the agent completed during a training iteration.
Iteration x-axis
This shows the number of iterations completed during your training job.

Graphic shows elements of a reward graph
List of reward graph parts and what they do

Reward Graph Interpretation
The following four examples give you a sense of how to interpret the success of your model based on the reward graph. Learning to read these graphs is as much of an art as it is a science and takes time, but reviewing the following four examples will give you a start.

Needs more training
In the following example, we see there have only been 600 iterations, and the graphs are still going up. We see the evaluation completion percentage has just reached 100%, which is a good sign but isn’t fully consistent yet, and the training completion graph still has a ways to go. This reward function and model are showing promise, but need more training time.

Graph of model that needs more training
Needs more training

No improvement
In the next example, we can see that the percentage of track completions haven’t gone above around 15 percent and it's been training for quite some time—probably around 6000 iterations or so. This is not a good sign! Consider throwing this model and reward function away and trying a different strategy.

The reward graph of a model that is not worth keeping.
No improvement

A well-trained model
In the following example graph, we see the evaluation percentage completion reached 100% a while ago, and the training percentage reached 100% roughly 100 or so iterations ago. At this point, the model is well trained. Training it further might lead to the model becoming overfit to this track.

Avoid overfitting
Overfitting or overtraining is a really important concept in machine learning. With AWS DeepRacer, this can become an issue when a model is trained on a specific track for too long. A good model should be able to make decisions based on the features of the road, such as the sidelines and centerlines, and be able to drive on just about any track.

An overtrained model, on the other hand, learns to navigate using landmarks specific to an individual track. For example, the agent turns a certain direction when it sees uniquely shaped grass in the background or a specific angle the corner of the wall makes. The resulting model will run beautifully on that specific track, but perform badly on a different virtual track, or even on the same track in a physical environment due to slight variations in angles, textures, and lighting.

This model had been overfit to a specific track.
Well-trained - Avoid overfitting

Adjust hyperparameters
The AWS DeepRacer console's default hyperparameters are quite effective, but occasionally you may consider adjusting the training hyperparameters. The hyperparameters are variables that essentially act as settings for the training algorithm that control the performance of your agent during training. We learned, for example, that the learning rate controls how many new experiences are counted in learning at each step.

In this reward graph example, the training completion graph and the reward graph are swinging high and low. This might suggest an inability to converge, which may be helped by adjusting the learning rate. Imagine if the current weight for a given node is .03, and the optimal weight should be .035, but your learning rate was set to .01. The next training iteration would then swing past optimal to .04, and the following iteration would swing under it to .03 again. If you suspect this, you can reduce the learning rate to .001. A lower learning rate makes learning take longer but can help increase the quality of your model.

This model's hyperparameters need to be adjusted.
Adjust hyperparameters

Good Job and Good Luck!
Remember: training experience helps both model and reinforcement learning practitioners become a better team. Enter your model in the monthly AWS DeepRacer League races for chances to win prizes and glory while improving your machine learning development skills!

---

## **14. Introduction to Generative AI**

🎥 [Udacity, Video Link](https://youtu.be/YziMYb9xA-g)

Generative AI and Its Applications
Generative AI is one of the biggest recent advancements in artificial intelligence because of its ability to create new things.

Until recently, the majority of machine learning applications were powered by discriminative models. A discriminative model aims to answer the question, "If I'm looking at some data, how can I best classify this data or predict a value?" For example, we could use discriminative models to detect if a camera was pointed at a cat.

As we train this model over a collection of images (some of which contain cats and others which do not), we expect the model to find patterns in images which help make this prediction.

A generative model aims to answer the question,"Have I seen data like this before?" In our image classification example, we might still use a generative model by framing the problem in terms of whether an image with the label "cat" is more similar to data you’ve seen before than an image with the label "no cat."

However, generative models can be used to support a second use case. The patterns learned in generative models can be used to create brand new examples of data which look similar to the data it seen before.

An image showing discriminative versus generative algorithms
Discriminative versus Generative algorithms

Generative AI Models
In this lesson, you will learn how to create three popular types of generative models: generative adversarial networks (GANs), general autoregressive models, and transformer-based models. Each of these is accessible through AWS DeepComposer to give you hands-on experience with using these techniques to generate new examples of music.

Autoregressive models
Autoregressive convolutional neural networks (AR-CNNs) are used to study systems that evolve over time and assume that the likelihood of some data depends only on what has happened in the past. It’s a useful way of looking at many systems, from weather prediction to stock prediction.

Generative adversarial networks (GANs)
Generative adversarial networks (GANs), are a machine learning model format that involves pitting two networks against each other to generate new content. The training algorithm swaps back and forth between training a generator network (responsible for producing new data) and a discriminator network (responsible for measuring how closely the generator network’s data represents the training dataset).

Transformer-based models
Transformer-based models are most often used to study data with some sequential structure (such as the sequence of words in a sentence). Transformer-based methods are now a common modern tool for modeling natural language.

We won't cover this approach in this course but you can learn more about transformers and how AWS DeepComposer uses transformers in AWS DeepComposer learning capsules.

---

## **15. Generative AI with AWS DeepComposer**

🎥 [Udacity, Video Link](https://youtu.be/5bMSoN-m9Po)

What is AWS DeepComposer?
AWS DeepComposer gives you a creative and easy way to get started with machine learning (ML), specifically generative AI. It consists of a USB keyboard that connects to your computer to input melody and the AWS DeepComposer console, which includes AWS DeepComposer Music studio to generate music, learning capsules to dive deep into generative AI models, and AWS DeepComposer Chartbusters challenges to showcase your ML skills.

AWS DeepComposer
AWS DeepComposer

Summary
AWS DeepComposer keyboard
You don't need an AWS DeepComposer keyboard to finish this course. You can import your own MIDI file, use one of the provided sample melodies, or use the virtual keyboard in the AWS DeepComposer Music studio.

AWS DeepComposer music studio
To generate, create, and edit compositions with AWS DeepComposer, you use the AWS DeepComposer Music studio. To get started, you need an input track and a trained model.

For the input track, you can use a sample track, record a custom track, or import a track.

Input track
Input track

For the ML technique, you can use either a sample model or a custom model.

Each AWS DeepComposer Music studio experience supports three different generative AI techniques: generative adversarial networks (GANs), autoregressive convolutional neural network (AR-CNNs), and transformers.

Use the GAN technique to create accompaniment tracks.
Use the AR-CNN technique to modify notes in your input track.
Use the transformers technique to extend your input track by up to 30 seconds.
ML models
ML models

Demo: AWS DeepComposer

Summary
In this demo, you went through the AWS DeepComposer console where you can learn about deep learning, input your music, and train deep learning models to create new music.

AWS DeepComposer learning capsules
To learn the details behind generative AI and ML techniques used in AWS DeepComposer you can use easy-to-consume, bite-sized learning capsules in the AWS DeepComposer console.

AWS DeepComposer Learning Capsules
AWS DeepComposer learning capsules

AWS DeepComposer Chartbusters challenges
Chartbusters is a global challenge where you can use AWS DeepComposer to create original compositions and compete in monthly challenges to showcase your machine learning and generative AI skills.

You don't need to participate in this challenge to finish this course, but the course teaches everything you need to win in both challenges we launched this season. Regardless of your background in music or ML, you can find a competition just right for you.

You can choose between two different challenges this season:

In the Basic challenge, “Melody-Go-Round”, you can use any machine learning technique in the AWS DeepComposer Music studio to create new compositions.
In the Advanced challenge, “Melody Harvest”, you train a custom generative AI model using Amazon SageMaker.

---

## **16. GANs with AWS DeepComposer**

🎥 [Udacity, Video Link](https://youtu.be/urpSUKYMcn8)

Summary
We’ll begin our journey of popular generative models in AWS DeepComposer with generative adversarial networks or GANs. Within an AWS DeepComposer GAN, models are used to solve a creative task: adding accompaniments that match the style of an input track you provide. Listen to the input melody and the output composition created by the AWS DeepComposer GAN model:

Input melody
Output melody
What are GANs?
A GAN is a type of generative machine learning model which pits two neural networks against each other to generate new content: a generator and a discriminator.

A generator is a neural network that learns to create new data resembling the source data on which it was trained.
A discriminator is another neural network trained to differentiate between real and synthetic data.
The generator and the discriminator are trained in alternating cycles. The generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

Collaboration between an orchestra and its conductor
A simple metaphor of an orchestra and its conductor can be used to understand a GAN. The orchestra trains, practices, and tries to generate polished music, and then the conductor works with them, as both judge and coach. The conductor judges the quality of the output and at the same time provides feedback to achieve a specific style. The more they work together, the better the orchestra can perform.

The GAN models that AWS DeepComposer uses work in a similar fashion. There are two competing networks working together to learn how to generate musical compositions in distinctive styles.

A GAN's generator produces new music as the orchestra does. And the discriminator judges whether the music generator creates is realistic and provides feedback on how to make its data more realistic, just as a conductor provides feedback to make an orchestra sound better.

Orchestra and its conductor metaphor
An orchestra and its conductor

Training Methodology
Let’s dig one level deeper by looking at how GANs are trained and used within AWS DeepComposer. During training, the generator and discriminator work in a tight loop as depicted in the following image.

A schema representing a GAN model used within AWS DeepComposer
A schema representing a GAN model used within AWS DeepComposer

Note: While this figure shows the generator taking input on the left, GANs in general can also generate new data without any input.

Generator
The generator takes in a batch of single-track piano rolls (melody) as the input and generates a batch of multi-track piano rolls as the output by adding accompaniments to each of the input music tracks.
The discriminator then takes these generated music tracks and predicts how far they deviate from the real data present in the training dataset. This deviation is called the generator loss. This feedback from the discriminator is used by the generator to incrementally get better at creating realistic output.
Discriminator
As the generator gets better at creating music accompaniments, it begins fooling the discriminator. So, the discriminator needs to be retrained as well. The discriminator measures the discriminator loss to evaluate how well it is differentiating between real and fake data.
Beginning with the discriminator on the first iteration, we alternate training these two networks until we reach some stop condition; for example, the algorithm has seen the entire dataset a certain number of times or the generator and discriminator loss reach some plateau (as shown in the following image).

Discriminator loss and generator loss reach a plateau
Discriminator loss and generator loss reach a plateau

New Terms
Generator: A neural network that learns to create new data resembling the source data on which it was trained.
Discriminator: A neural network trained to differentiate between real and synthetic data.
Generator loss: Measures how far the output data deviates from the real data present in the training dataset.
Discriminator loss: Evaluates how well the discriminator differentiates between real and fake data.
Supporting Materials
Input Twinkle Twinkle Input
Output Twinkle Twinkle Rock

---

## **17. AR-CNN with AWS DeepComposer**

🎥 [Udacity, Video Link](https://youtu.be/NkxWTTXM9pI)

Summary
Our next popular generative model is the autoregressive convolutional neural network (AR-CNN). Autoregressive convolutional neural networks make iterative changes over time to create new data.

To better understand how the AR-CNN model works, let’s first discuss how music is represented so it is machine-readable.

Image-based representation
Nearly all machine learning algorithms operate on data as numbers or sequences of numbers. In AWS DeepComposer, the input tracks are represented as a piano roll\**. *In each two-dimensional piano roll, time is on the horizontal axis and pitch\* is on the vertical axis. You might notice this representation looks similar to an image.

The AR-CNN model uses a piano roll image to represent the audio files from the dataset. You can see an example in the following image where on top is a musical score and below is a piano roll image of that same score.

Musical score and piano roll
Musical score and piano roll

How the AR-CNN Model Works
When a note is either added or removed from your input track during inference, we call it an edit event. To train the AR-CNN model to predict when notes need to be added or removed from your input track (edit event), the model iteratively updates the input track to sounds more like the training dataset. During training, the model is also challenged to detect differences between an original piano roll and a newly modified piano roll.

New Terms
Piano roll: A two-dimensional piano roll matrix that represents input tracks. Time is on the horizontal axis and pitch is on the vertical axis.
Edit event: When a note is either added or removed from your input track during inference.

---

## **18. Quiz: Generative AI**

QUESTION 1 OF 4
Which is the following statements is false in the context of AR-CNNs?

QUESTION 2 OF 4
Please identify which of the following statements are true about a generative adversarial network (GAN). There may be more than one correct answer.

QUESTION 3 OF 4
Which model is responsible for each of these roles in generative AI?

ROLES

NAME

Evaluating the output quality

Creating new output

Providing feedback

QUESTION 4 OF 4
True or false: Loss functions help us determine when to stop training a model.

---

## **19. Demo: Create Music with AWS DeepComposer**

🎥 [Udacity, Video Link, Demo Part 1:](https://youtu.be/9ofSAkC_104)
🎥 [Udacity, Video Link, Demo Part 2:](https://youtu.be/0-MSE1tCZ74)

Below you find a video demonstrating how you can use AWS DeepComposer to experiment with GANs and AR-CNN models.

Important

To get you started, AWS DeepComposer provides a 12-month Free Tier for first-time users. With the Free Tier, you can perform up to 500 inference jobs, translating to 500 pieces of music, using the AWS DeepComposer Music studio. You can use one of these instances to complete the exercise at no cost. For more information, please read the AWS account requirements page.
Demo Part 1:

Demo Part 2:

Summary
In the demo, you have learned how to create music using AWS Deepcomposer.

You will need a music track to get started. There are several ways to do it. You can record your own using the AWS keyboard device or the virtual keyboard provided in the console. Or you can input a MIDI file or choose a provided music track.

Once the music track is inputted, choose "Continue" to create a model. The models you can choose are AR-CNN, GAN, and transformers. Each of them has a slightly different function. After choosing a model, you can then adjust the parameters used to train the model.

Once you are done with model creation, you can select "Continue" to listen and improve your output melody. To edit the melody, you can either drag or extend notes directly on the piano roll or adjust the model parameters and train it again. Keep tuning your melody until you are happy with it then click "Continue" to finish the composition.

If you want to enhance your music further with another generative model, you can do it too. Simply choose a model under the "Next step" section and create a new model to enhance your music.

Congratulations on creating your first piece of music using AWS DeepComposer! Now you can download the melody or submit it to a competition. Hope you enjoy the journey of creating music with AWS DeepComposer.

---

## **20. Exercise: Generate music with AWS DeepComposer**

You have seen how the instructor generates a piece of music in AWS DeepComposer. Now, it's your turn to create your very own piece of music. To finish this exercise, you should complete the following steps.

Open the AWS DeepComposer console.
In the navigation pane, choose Music studio, then choose Start composing.
On the Input track page, record a melody using the virtual keyboard, import a MIDI file, or choose an input track. On the ML technique page, choose AR-CNN. On the Inference output page, you can do the following:
Change the AR-CNN parameters, choose Enhance again, and then choose Play to hear how your track has changed. Repeat until you like the outcome.
Choose Edit melody to modify and change the notes that were added during inference.
Choose Continue to finish creating your composition.

You can then choose Share composition, Register, or Sign in to Soundcloud and submit to the "Melody-Go-Round" competition. Participation in the competition is optional.

If you get stuck, you can check out the demo videos on Demo: Create Music with AWS DeepComposer page.

---

## **21. Build a Custom GAN Model (Optional): Part 1**

🎥 [Udacity, Video Link]()

To create the custom GAN, you will need to use an instance type that is not covered in the Amazon SageMaker free tier. You may incur a cost if you want to build a custom GAN.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation.

Important: This is an optional exercise.

Setting Up the AWS DeepComposer Notebook

Go to the AWS Management Console and search for Amazon SageMaker.
Once inside the SageMaker console, look to the left-hand menu and select Notebook Instances.
Next, choose Create notebook instance.
In the Notebook instance setting section, give the notebook a name, for example, DeepComposerUdacity.
Based on the kind of CPU, GPU and memory you need the next step is to select an instance type. For our purposes, we’ll configure a ml.c5.4xlarge
Leave the Elastic Inference defaulted to none.
In the Permissions and encryption section, create a new IAM role using all of the defaults.
When you see that the role was created successfully, navigate down a little way to the Git repositories section
Select Clone a public Git repository to this notebook instance only
Copy and paste the public URL into the Git repository URL section: https://github.com/aws-samples/aws-deepcomposer-samples
Select Create notebook instance
Give SageMaker a few minutes to provision the instance and clone the Git repository
When the status reads "InService" you can open the Jupyter notebook.
Status is InService
Exploring the Notebook
Now that it’s configured and ready to use, let’s take a moment to investigate what’s inside the notebook.

Open the Notebook
Click Open Jupyter.
When the notebook opens, click on "gan".
When the lab opens click on GAN.ipynb.
Review: Generative Adversarial Networks (GANs).
GANs consist of two networks constantly competing with each other:

Generator network that tries to generate data based on the data it was trained on.
Discriminator network that is trained to differentiate between real data and data which is created by the generator.
A diagram of generator and discriminator.
A diagram of generator and discriminator.

Set Up the Project
Run the first Dependencies cell to install the required packages
Run the second Dependencies cell to import the dependencies
Run the Configuration cell to define the configuration variables
Note: While executing the cell that installs dependency packages, you may see warning messages indicating that later versions of conda are available for certain packages. It is completely OK to ignore this message. It should not affect the execution of this notebook.

Click **Run** or `Shift-Enter` in the cell
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
Piano Roll visualization. Each Piano Roll Represents A Separate Piano Track in the Song
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

## **22. Build a Custom GAN Model (Optional): Part 2**

To create the custom GAN, you will need to use an instance type that is not covered in the Amazon SageMaker free tier. You may incur a cost if you want to build a custom GAN.

You can learn more about SageMaker costs in the Amazon SageMaker pricing documentation.

Important: This is an optional exercise.

Model Architecture
Before we can train our model, let’s take a closer look at model architecture including how GAN networks interact with the batches of data we feed into the model, and how the networks communicate with each other.

How the Model Works
The model consists of two networks, a generator and a discriminator (critic). These two networks work in a tight loop:

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

You can:

Check to see if the losses for the networks are converging
Look at commonly used musical metrics of the generated sample and compared them to the training dataset.
Evaluating Our Training Results
Run the cell to restore the saved checkpoint. If you don't want to wait to complete the training you can use data from a pre-trained model by setting TRAIN = False in the cell.
Run the cell to plot the losses.
Run the cell to plot the metrics.
Results and Inference
Finally, we are ready to hear what the model produced and visualize the piano roll output!

Once the model is trained and producing acceptable quality, it’s time to see how it does on data it hasn’t seen. We can test the model on these unknown inputs, using the results as a proxy for performance on future data.

Evaluate the Generated Music
In the first cell, enter 0 as the iteration number.
run the cell and play the music snippet.

Or listen to this example snippet from iteration 0:

Piano roll at iteration 0
Example Piano Roll at Iteration 0

In the first cell, enter 500 as the iteration number:run the cell and play the music snippet. Or listen to the example snippet at iteration 500.

In the second cell, enter 500 as the iteration number:run the cell and display the piano roll.

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
Inference
Now that the GAN has been trained we can run it on a custom input to generate music.

Run the cell to generate a new song based on "Twinkle Twinkle Little Star". Or listen to the example of the generated music here:

Run the next cell and play the generated music. Or listen to the example of the generated music here:

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
In this demo we learned how to setup a Jupyter notebook in Amazon SageMaker, reviewed a machine learning code, and what data preparation, model training, and model evaluation can look like in a notebook instance. While this was a fun use case for us to explore, the concepts and techniques can be applied to other machine learning projects like an object detector or a sentiment analysis on text.

---

## **23. Lesson Review**

🎥 [Udacity, Video Link]()

The outline of the lesson
The outline of the lesson

In this lesson, we learned many advanced machine learning techniques. Specifically, we learned:

Computer vision and its application
How to train a computer vision project with AWS DeepLens
Reinforcement learning and its application
How to train a reinforcement learning model with AWS DeepRacer
Generative AI
How to train a GAN and AR-CNN model with AWS DeepComposer
Now, you should be able to:

Identify AWS machine learning offerings and how different services are used for different applications
Explain the fundamentals of computer vision and a couple of popular tasks
Describe how reinforcement learning works in the context of AWS DeepRacer
Explain the fundamentals of Generative AI, its applications, and three famous generative AI model in the context of music and AWS DeepComposer

---

## **24. Glossary**

- **Action**: For every state, an agent needs to take an action toward achieving its goal.
- **Agent**: The piece of software you are training is called an agent. It makes decisions in an environment to reach a goal.
- **Discriminator**: A neural network trained to differentiate between real and synthetic data.
- **Discriminator loss**: Evaluates how well the discriminator differentiates between real and fake data.
- **Edit event**: When a note is either added or removed from your input track during inference.
- **Environment**: The environment is the surrounding area within which the agent interacts.
- **Exploration versus exploitation**: An agent should exploit known information from previous experiences to achieve higher cumulative rewards, but it also needs to explore to gain new experiences that can be used in choosing the best actions in the future.
- **Generator**: A neural network that learns to create new data resembling the source data on which it was trained.
- **Generator loss**: Measures how far the output data deviates from the real data present in the training dataset.
- **Hidden layer**: A layer that occurs between the output and input layers. Hidden layers are tailored to a specific task.
- **Input layer**: The first layer in a neural network. This layer receives all data that passes through the neural network.
- **Output layer**: The last layer in a neural network. This layer is where the predictions are generated based on the information captured in the hidden layers.
- **Piano roll**: A two-dimensional piano roll matrix that represents input tracks. Time is on the horizontal axis and pitch is on the vertical axis.
- **Reward**: Feedback is given to an agent for each action it takes in a given state. This feedback is a numerical reward.
