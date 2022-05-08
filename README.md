# H&M Personalized Fashion Recommendations

# 1. The Overview
The objective of this notebook is to setup a recommender model with <a href="https://www.tensorflow.org/recommenders?hl=pt-br">Tensor Flow Recommenders</a> and submit the results to the H&M Personalized Fashion Recommendations Competition on Kaggle.

You can find the complete overview of the competition and the datasets by clicking <a href='https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations'>HERE</a>.

<b>Lets make a short recap of the H&M Personalized Fashion Recommendations Competition.</b>

H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Their online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.

In this competition, H&M Group invites competitors to develop product recommendations based on data from previous transactions, as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.

The challenge is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends.

# 2. TensorFlow Recommenders

TensorFlow Recommenders (TFRS) is a library for building recommender system models.
It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.
It's built on Keras and aims to have a gentle learning curve while still giving you the flexibility to build complex models.

TFRS makes it possible to:
* Build and evaluate flexible recommendation retrieval models.
* Freely incorporate item, user, and context information into recommendation models.
* Train multi-task models that jointly optimize multiple recommendation objectives.
* TFRS is open source and available on <a href="https://github.com/tensorflow/recommenders">Github</a>.

To learn more, see the <a href = "https://www.tensorflow.org/recommenders/examples/basic_retrieval'tutorial"> on how to build a movie recommender system</a>, or check the API docs for the <a href= "https://www.tensorflow.org/recommenders/api_docs/python/tfrs">API reference.</a>

# 3. The Plan of Attack
A retail recommender system will normally have 2 phases:
* Retrieving - Which is responsible for selecting an initial set of candidates from all possible candidates.
* Ranking - Which takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations.

In this notebook we are gonna create a only the retrieving model, which are simplier but yet powerfull.

The basic idea is that we have two models, one consists in the query model (customer model) and the other, the candidate model (article model).
Then, we combine these two models in a new model, which will perform the loss calculation and optimize the weights using ADAM optimzer.

<img src="https://lh4.googleusercontent.com/CmzEJysJCdipgp1ntCzgZ5pdowgieyB43ep6cU_0WRpsOhXQy4aDWTZxd2IhV0A210TWIP41BpvSXKrGh5sv5Mya30lh1vKHQDiZK3wobWgIt23hx7pCZ3Em8TXMGt1mukuiu-2b" width = "700">