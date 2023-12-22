---
layout: single
classes: wide
title:  "Linear Classifier for Text"
categories: 
  - NLP
tag: [Perceptron, Freedman's Paradox]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---


# Linear Classifier for Text

Here, Yelp restaurant reviews dataset is comprised of user-contributed reviews posted to Yelp for restaurants in Pittsburgh. Each review is accompanied by a binary indicator of whether or not the reviewer-assigned rating is at least four (on 5-point scale). The text of the reviews has been processed to replace all non-alphanumeric symbols with whitespace, and all letters have been changed to lowercase. The prediction problem we consider here is predicting the binary indicator from the review text.
The training data is contained in the file reviews **tr.csv**, and the test data is contained in the file reviews **te.csv**.

## Imports


```python
import sys
import numpy as np
from csv import DictReader
```

## Load Training Data


```python
vocab = {}
vocab_size = 0
train_examples = []
with open('reviews_tr.csv', 'r') as f:
    reader = DictReader(f)
    for row in reader:
        label = row['rating'] == '1'
        words = row['text'].split(' ')
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1
        train_examples.append((label, [vocab[word] for word in words]))
```

## Load Test Data


```python
test_examples = []
with open('reviews_te.csv', 'r') as f:
    reader = DictReader(f)
    for row in reader:
        label = row['rating'] == '1'
        words = row['text'].split(' ')
        test_examples.append((label, [vocab.get(word, -1) for word in words if word in vocab]))
```

## Transformation to "bag-of-words" Vector
We need to note that transforming and storing whole "bag-of-words" vectors can be expansive. So we will use "bag-of-words" vectors directly as input and avoid explicitly forming the “bag-of-words” vectors for the train-examples.



```python
def bag_of_words_rep(word_ids, dim):
    bow_vector = np.zeros(dim) # creates a numpy.ndarray of shape (dim,) 
    for word_id in word_ids:
        bow_vector[word_id] += 1
    return bow_vector

first_bow_vector = bag_of_words_rep(train_examples[0][1], vocab_size)
print(first_bow_vector)
```

    [1. 1. 1. ... 0. 0. 0.]



```python
print('Memory required for train_examples:', sys.getsizeof(train_examples))
print('Estimated required memory for bag-of-words representation:', len(train_examples) * sys.getsizeof(first_bow_vector))
```

    Memory required for train_examples: 8448728
    Estimated required memory for bag-of-words representation: 1659544000000


## Online Perceptron

The Online Perceptron algorithm is a variant of the standard Perceptron algorithm. The standard Perceptron algorithm maintains a weight vector **w**, and repeatedly updates the weight vector with any training example that is misclassified by the (homogeneous) linear classifier with parameter **w**. This continues until there are no misclassified training examples. Of course, if the training data is not linearly separable, this will go on forever, some examples will be used to update the weight vector infinitely often. 

Now implementing the Online Perceptron algorithm
In this implementation, we'll directly use the word IDs from the train_examples to update the weight vector. Instead of forming the full "bag-of-words" vector representation, we can calculate and update only those weights which correspond to the word IDs in a given example. This way, we will use less memory and avoid forming the "bag-of-words" vector, but achieves the same outcome.


```python
def averaged_perceptron(train_examples, dimension):
    w = np.zeros(dimension)  # Initializing weights
    w_sum = np.zeros(dimension)
    for example in train_examples:
        label, word_ids = example
        y_pred = 1 if np.sum(w[word_ids]) > 0 else 0
        if y_pred != label:  # Update weights
            if label:  # if y(i) = true
                for i in word_ids: w[i] += 1
            else:      # if y(i) = false
                for i in word_ids: w[i] -= 1
        w_sum += w    
    return w_sum / len(train_examples)
```


```python
def online_perceptron(train_examples, dimension):
    
    # Initializing weights
    w = np.zeros(dimension)
    # For each training examples
    for example in train_examples:
        label, word_ids = example
        
        # Our prediction
        y_pred = 1 if np.sum(w[word_ids]) > 0 else 0
        
        # Update weights if misclassified
        if y_pred != label:
            if label:  # if y(i) = true
                for i in word_ids:
                    w[i] += 1
            else:      # if y(i) = false
                for i in word_ids:
                    w[i] -= 1
    
    return w

# Training the perceptron
w = online_perceptron(train_examples, vocab_size)
```

### Compute Error Rate:
Let's write functions to calculate the training and test error rates


```python
def error_rate(w, data):
    errors = 0
    
    for example in data:
        label, word_ids = example
        pred = 1 if np.sum(w[word_ids]) > 0 else 0
        if pred != label:
            errors += 1
            
    return errors / len(data)


# Calculate Training Error Rate
training_error = error_rate(w, train_examples)

# Calculate Test Error Rate
test_error = error_rate(w, test_examples)

# Reporting upto 3 signiticant digits
print(f"Training Error Rate: {training_error:.3g}")
print(f"Test Error Rate: {test_error:.3g}")
```

    Training Error Rate: 0.179
    Test Error Rate: 0.18


## Top 10 Words
Identifying the top 10 positive and negative weights and their respective words:


```python
# Invert the vocabulary mapping
inv_vocab = {v: k for k, v in vocab.items()}

# Get the top and bottom 10 weights
top_10_indices = w.argsort()[-10:][::-1]
bottom_10_indices = w.argsort()[:10]

top_10_words = [inv_vocab[i] for i in top_10_indices]
bottom_10_words = [inv_vocab[i] for i in bottom_10_indices]

print("10 words that have the highest weights:", top_10_words)
print()
print("10 words that have the lowest weights:", bottom_10_words)
```

    10 words that have the highest weights: ['perfection', 'disappoint', 'gem', 'phenomenal', 'heaven', 'perfectly', 'incredible', 'perfect', 'superb', 'fantastic']
    
    10 words that have the lowest weights: ['mediocre', 'worst', 'poisoning', 'inedible', 'awful', 'worse', 'flavorless', 'bland', 'tasteless', 'horrible']


## Upgrade Online Perceptron
For better performance, we will try implementing the "Averaged Perceptron" variant.


```python
vocab_size
```




    207429




```python
len(w_avg)
```




    207429




```python
def averaged_perceptron(train_examples, dimension):
    # Initializing weights
    w = np.zeros(dimension)
    w_sum = np.zeros(dimension)
    
    # For each training examples
    for example in train_examples:
        label, word_ids = example
        
        # Our prediction
        y_pred = 1 if np.sum(w[word_ids]) > 0 else 0
        
        # Update weights if misclassified
        if y_pred != label:
            if label:  # if y(i) = true
                for i in word_ids:
                    w[i] += 1
            else:      # if y(i) = false
                for i in word_ids:
                    w[i] -= 1
        w_sum += w
        
    return w_sum / len(train_examples)

# Train Averaged Perceptron variant
w_avg = averaged_perceptron(train_examples, vocab_size)

# Calculate Training and Test Error Rate for Averaged Perceptron
training_error_avg = error_rate(w_avg, train_examples)
test_error_avg = error_rate(w_avg, test_examples)

print(f"Averaged Perceptron Training Error Rate: {training_error_avg:.3g}")
print(f"Averaged Perceptron Test Error Rate: {test_error_avg:.3g}")
```

    Averaged Perceptron Training Error Rate: 0.104
    Averaged Perceptron Test Error Rate: 0.107


We did better! 

And here are the 10 highest weighted words and 10 lowest weighted words for the Averaged Perceptron


```python
# Get the top and bottom 10 weights
top_10_indices = w_avg.argsort()[-10:][::-1]
bottom_10_indices = w_avg.argsort()[:10]

top_10_words = [inv_vocab[i] for i in top_10_indices]
bottom_10_words = [inv_vocab[i] for i in bottom_10_indices]

print("10 words that have the highest weights:", top_10_words)
print()
print("10 words that have the lowest weights:", bottom_10_words)
```

    10 words that have the highest weights: ['perfection', 'perfect', 'incredible', 'perfectly', 'fantastic', 'delicious', 'gem', 'amazing', 'disappoint', 'excellent']
    
    10 words that have the lowest weights: ['worst', 'mediocre', 'bland', 'disappointing', 'horrible', 'lacked', 'awful', 'terrible', 'meh', 'disappointment']


# Observing Freedman's Paradox
What is freedman's paradox?
https://en.wikipedia.org/wiki/Freedman%27s_paradox


```python
import pickle
import numpy as np
from numpy.linalg import inv
with open('freedman.pkl', 'rb') as f:
    freedman = pickle.load(f)
```


```python
    
# Extract the relevant arrays
X = freedman['data']
Y = freedman['labels']
n, d = X.shape
```


```python
rho_hat = np.sum(data * labels[:, np.newaxis], axis=0) / n
```


```python
threshold = 2 /10
J_hat = np.where(abs(rho_hat) > threshold)[0]
num_features = len(J_hat)
```


```python
num_features
```




    42




```python
# Step 3: OLS estimation
X_J = X[:, J_hat]
w_hat = np.dot(inv(np.dot(X_J.T, X_J)), np.dot(X_J.T, Y))
empirical_risk_train = (1/n) * np.sum((np.dot(X_J, w_hat) - Y)**2)
X_test = freedman['testdata']
Y_test = freedman['testlabels']
X_test_J = X_test[:, J_hat]
empirical_risk_test = (1/len(Y_test)) * np.sum((np.dot(X_test_J, w_hat) - Y_test)**2)
X2 = freedman['data2']
Y2 = freedman['labels2']
X2_J = X2[:, J_hat]
w_hat_2 = np.dot(inv(np.dot(X2_J.T, X2_J)), np.dot(X2_J.T, Y2))
empirical_risk_train_2 = (1/n) * np.sum((np.dot(X2_J, w_hat_2) - Y2)**2)
```


```python
print(f"Number of features in J_hat: {num_features}")
print(f"Empirical risk on training set: {empirical_risk_train:.3f}")
print(f"Empirical risk on test set: {empirical_risk_test:.3f}")
print(f"Empirical risk on different training set: {empirical_risk_train_2:.3f}")
```

    Number of features in J_hat: 42
    Empirical risk on training set: 0.207
    Empirical risk on test set: 1.447
    Empirical risk on different training set: 0.601

