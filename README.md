# DataMiningProject
## Project Description
This project was aimed at creating at competitive solution to the stance detection problem described in the [Fake News Challenge]('http://www.fakenewschallenge.org/'). We attempt to build a learner that detects the stance of given training data ('train_bodies.csv','train_stances.csv') and predicts stances of new test data. Multiple models have been trained, and various data representations have been implemented for experimental purposes. 

## Methodology
A Bag-Of-Words representation has been used to transform both the headlines and bodies of each article, and an n-gram (1 to 7 n-gram long) vector is extracted which takes the top 1000 important n-grams from the BOW representation. CountVectorisers and TfidfTransformers from scikit-learn have been implemented as part of this process. 
Once the vectors are extracted for each headline and body, we then calculate various distances to determine the best distance to use for learning a model. The distances used include:
- Euclidean Distance
- Minkowski Distance
- Absolute Distance
- Mean Absolute Distance
- Sum Squared Distance
- Manhattan Distance
- Canberra Distance
- KL Divergence
- Cosine Similarity

An array of these distances are then fed into 2 models - a Linear Regression Model, and a Logistic Regression Model with Gradient Descent (found in files: 'logistic.py', 'linear.py'). Comparison of the predicting power of these models is done to see which is best suited for detecting stances.

## Running
There are several prerequisites before running:
- Python 2.7
- Scikit-learn
- Scipy
- Matplotlib
- Numpy

Run the following line of code
```python
python main.py
```
