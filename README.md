# Recommender Systems

Recommender Systems are software tools and techniques providing suggestions for **items** to be consumed by a **user**.
The suggestions provided are aimed at supporting their users in various decision-making processes, such as what items to
buy, what movie to watch, or what news to read. Recommender systems have proven to be valuable means for online users to
cope with the information overload and have become one of the most powerful and popular tools in electronic commerce.
Correspondingly, various techniques for recommendation generation have been proposed and during the last decade(s), many
of them have also been successfully deployed in commercial environments.

**Authors:** [Renqing Cuomao](mailto:renqing.cuomao@epfl.ch), [Rami Atassi](mailto:rami.atassi@epfl.ch), 
[Paulo Ribeiro](mailto:paulo.ribeirodecarvalho@epfl.ch)

## Description

In this project, you are expected to develop a recommender system that is able to predict the rating of a user for a
given book. The metadata of the books can be used as you wish and you are allowed to use external book metadata sources.
Check out the [project description](https://docs.google.com/document/d/1gHoGLcWpGv2QOMUo2tudYxAGyn4quy4upHLJhiozH4Q) for
more details.

## Requirements

The ratings you can use are the ones in the training set only. You can use standard data manipulation and analysis 
libraries and machine learning / deep learning frameworks with the following exception: **you are not allowed to use 
existing implementations of recommender systems (addressing rating prediction or next-item prediction tasks), even if 
found in the previously mentioned resources, except for the purpose of comparing your own models.**

If you implement a User-based or Item-based Collaborative Filtering system as exposed in the course, the system should 
reach a score better than the score reached by the simple baseline mentioned in the Evaluation section.

## Resources Allowed

- Functions from Numpy, Scipy, Scikit-learn, Pandas, Keras, Gensim, NLTK, or PyTorch that are not implementations of 
recommender systems (addressing rating prediction or next-item prediction tasks).


- General-purpose text pre-processing and pretrained embedding libraries for the metadata, as long as they are not 
fine-tuned for recommender system tasks (such as rating prediction or next-item prediction).


- The functions from LIGHTFM or surprise libraries you may use are limited to data preprocessing and cross-validation.


- Kaggle GPU