# Style Transfer

---

## **1. Style Transfer**

[Udacity, Video Link](https://youtu.be/_urN9BQ7RHM)

---

## **2. Separating Style & Content**

[Udacity, Video Link](https://youtu.be/PNFFAhymuHc)

---

## **3. VGG19 & Content Loss**

[Udacity, Video Link](https://youtu.be/PQ1UuzOIjCM)

---

## **4. Gram Matrix**

[Udacity, Video Link](https://youtu.be/e718uVAW3KU)

QUESTION 1 OF 2
Given a convolutional layer with dimensions d x h x w = (20*8*8), what length will one row of the vectorized convolutional layer have? (Vectorized means that the spatial dimensions are flattened.)

64

QUESTION 2 OF 2
Given a convolutional layer with dimensions d x h x w = (20*8*8), what dimensions (h x w) will the resultant Gram matrix have?

(20 x 20)

---

## **5. Style Loss**

[Udacity, Video Link](https://youtu.be/VazrQ7u-OHo)

---

## **6. Loss Weights**

[Udacity, Video Link](https://youtu.be/qO8oiZBtG1I)

---

## **7. VGG Features**

[Udacity, Video Link](https://youtu.be/Q5N2NEv7ADc)

---

## **8. Notebook: Style Transfer**

Notebook: Style Transfer
Now, you're ready to implement style transfer and apply it using your own images!

It's suggested that you open the notebook in a new, working tab and continue working on it as you go through the instructional videos in this tab. This way you can toggle between learning new skills and coding/applying new skills.

To open this notebook, go your local repo (from here on Github) and open the notebook Style_Transfer_Exercise.ipynb in the style-transfer folder. You can either download the repository with git clone https://github.com/udacity/deep-learning-v2-pytorch.git, or download it as an archive file from this link.

Instructions
Load in a pre-trained VGG Net
Freeze the weights in selected layers, so that the model can be used as a fixed feature extractor
Load in content and style images
Extract features from different layers of our model
Complete a function to calculate the gram matrix of a given convolutional layer
Define the content, style, and total loss for iteratively updating a target image
This is a self-assessed lab. If you need any help or want to check your answers, feel free to check out the solutions notebook in the same folder, or by clicking here.

Note about GPUs
In this notebook, you'll find optimizing the image is much faster if you use a GPU. However, you can still complete the exercises without a GPU. If you can't use a local GPU, we suggest you use cloud platforms such as AWS, GCP, and FloydHub to train your networks on a GPU.

---

## **9. Features & Gram Matrix**

[Udacity, Video Link](https://youtu.be/f89x9oAh6X0)

---

## **10. Gram Matrix Solution**

[Udacity, Video Link](https://youtu.be/uncCKMI5Yns)

---

## **11. Defining the Loss**

[Udacity, Video Link](https://youtu.be/lix8d3B2QcE)

---

## **12. Total Loss & Complete Solution**

[Udacity, Video Link](https://youtu.be/DzaQm9awcwY)
