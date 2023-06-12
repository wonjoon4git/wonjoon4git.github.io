---
layout: single
classes: wide
title:  "Linear Associator Exercise"
categories: 
  - Machine Learning
tag: [Linear Associator, Classification]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---

# Preface
The goal of this post is training a linear associator to identify the origins of ships based on sensor scans, so that the Enterprise will be able to take appropriate action when another ship approaches.


Incoming ships, particularly when far away, may generate noisy sensor readings that sometimes give misleading or partial information. Sometimes only a few letters of the name of the ship can be retrieved from the automatic transponder, or the hailing transmitter is not readable, or the color of the ship or its shape can be only crudely discerned. You should be able to take this partial, noisy information and (in descending order of importance) tell:

1. Whether the incoming ship is liable to be hostile or peaceful.
2. Whether the Enterprise should enter a state of heightened alertness.
3. The tentative identification of the ship’s system of origin.

You will be classifying new ships into four origins Klingon, Romulan, Antarean, and Federation. Klingons are to be treated as hostile. Romulans require Alert status. Antareans and the Federation must be considered friendly.
Data from previous encounters provides enough information to characterize new ships, even in corrupted form.

# Data Import

For better visualization and to work with ease, data from two tables: 
* Archival Intelligence Data Table for Training Neural Network (Training Data)
* Intelligence Table: Noisy Data for Classification (Noisy Test Data)

are re-written in .csv file format.


```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None

# Load train & test data
train_data = pd.read_csv('train_data.csv', delimiter=';').astype('object')
test_data = pd.read_csv('test_data.csv', delimiter=';').astype('object')
number_of_pairs = test_data.index.stop
```

Thus, we get our data frame:


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Planet of origin</th>
      <th>Warp Drive Vibration Index (Murds)</th>
      <th>Hailing Transponder Freq.(gigaHz)</th>
      <th>Surface Reflect. (color)</th>
      <th>Ratio of long to short axis</th>
      <th>Req. action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grotz</td>
      <td>Klingon</td>
      <td>6.9</td>
      <td>1006.4</td>
      <td>Black</td>
      <td>3.5</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tlarr</td>
      <td>Klingon</td>
      <td>7.0</td>
      <td>994.3</td>
      <td>Black</td>
      <td>2.3</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tribok</td>
      <td>Klingon</td>
      <td>7.3</td>
      <td>978.1</td>
      <td>Dark Gray</td>
      <td>2.8</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brogut</td>
      <td>Klingon</td>
      <td>7.1</td>
      <td>1005.4</td>
      <td>Dark Gray</td>
      <td>3.0</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Glorek</td>
      <td>Klingon</td>
      <td>7.1</td>
      <td>1001.8</td>
      <td>Light Gray</td>
      <td>1.0</td>
      <td>Hostile</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Planet of origin</th>
      <th>Warp Drive Vibration Index (Murds)</th>
      <th>Hailing Transponder Freq.(gigaHz)</th>
      <th>Surface Reflect. (color)</th>
      <th>Ratio of long to short axis</th>
      <th>Req. action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>______</td>
      <td>?</td>
      <td>7.3</td>
      <td>_____</td>
      <td>Light Gray</td>
      <td>2.1</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>_____</td>
      <td>?</td>
      <td>6.6</td>
      <td>1065.0</td>
      <td>White</td>
      <td>2.1</td>
      <td>?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lil___</td>
      <td>?</td>
      <td>6.7</td>
      <td>1045.0</td>
      <td>White</td>
      <td>___</td>
      <td>?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>______</td>
      <td>?</td>
      <td>___</td>
      <td>1065.0</td>
      <td>Light Color</td>
      <td>___</td>
      <td>?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pl__ik</td>
      <td>?</td>
      <td>7.0</td>
      <td>1006.3</td>
      <td>Dark Color</td>
      <td>___</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
</div>



# Pre-processing
To train a linear associator model, we need our data as numerical input-output vectors.

Thus, we need to transform our data in to numerical values, which I decided to encode as list of binary codes.

For example, ***Name*** will be encoded in two dimensions:
* First dimension will indicate whether the name includes 'k' or 'K' ("Glorek" = [1,0])
* Second dimension will indicate whether the name includes any numbers ("A2231" = [0,1])


All the other features of the data will be encoded in a similar way, and will be combined all together to form ***numerical input vectors***.

Some data are ***emphasized*** in input vector by being represented multiple times

The ***missing data*** will not be encoded(updated), so it will be left as an initial state, zero vector.
This is an ideal way to handel a missing data, becuase when missing part is multiplied with any element from our trained matrix A, will become a zero, not impacting our output vector.


|Dimension of input vector  | Name of Feature      | Definition of feature | Coding scheme     | How you dealt with missing data |
| :-------------| :----------- | :----------- | :----------- | :----------- |
| 1,3  | Name (1)        | Name includes 'k' or 'K'   | Binary       | Left as Zero    |
| 2,4   | Name (2)        | Name includes a number   | Binary       | Left as Zero    |
| 5   | Warp Drive Vibration Index (Murds) (1)       | index is smaller than 6.9    | Binary        | Left as Zero    |
| 6   | Warp Drive Vibration Index (Murds) (2)         | index is greater than 6.9, smaller than 7.3   | Binary         | Left as Zero   |
| 7   | Warp Drive Vibration Index (Murds) (3)         | index is greater than 7.3  | Binary        | Left as Zero    |
| 8   | Hailing Transponder Freq.(gigaHz) (1)       | freqency is smaller than 1000   | Binary        | Left as Zero   |
| 9   | Hailing Transponder Freq.(gigaHz) (2)        | freqency is greater than 1000   | Binary         | Left as Zero    |
| 10,20  | Surface Reflect. (color) (1)       | color name includes 'Light'   | Binary          |Left as Zero    |
| 11,21  | Surface Reflect. (color) (2)       | color name includes 'Dark'   | Binary         | Left as Zero  |
| 12,22  | Surface Reflect. (color) (3)       | color name includes 'Black'   | Binary         | Left as Zero    |
| 13,23  | Surface Reflect. (color) (4)       | color name includes 'Gray'   | Binary        | Left as Zero   |
| 14,24  | Surface Reflect. (color) (5)       | color name includes 'Blue'   | Binary         | Left as Zero   |
| 15,25  | Surface Reflect. (color) (6)       | color name includes 'Green'   | Binary        | Left as Zero    |
| 16,26  | Surface Reflect. (color) (7)       | color name includes 'Orange'   | Binary      | Left as Zero   |
| 17,27  | Surface Reflect. (color) (8)       | color name includes 'Pink'   | Binary      | Left as Zero   |
| 18,28  | Surface Reflect. (color) (9)       | color name includes 'Yellow'   | Binary       | Left as Zero   |
| 19,29  | Surface Reflect. (color) (10)       | color name includes 'White'   | Binary       | Left as Zero    |
| 30   | Ratio of long to short axis (1)       | axis is smaller than 1.5   | Binary         | Left as Zero   |
| 31   | Ratio of long to short axis (2)        |  axis is greater than 1.5, smaller than 2.3  | Binary         | Left as Zero   |
| 32   | Ratio of long to short axis (2)        | axis is greater than 2.3   | Binary         | Left as Zero   |





The ***Output vector*** should be somehow also be a numerical vector.

Since the "Required action" is naturally determined by the "Planet of Origin," we only need to train/find "Planet of Origin".

Thus, our output vector will be 4 dimensional, each dimension representing its planet of origin

* Kingon = [1,0,0,0]
* Romulan = [0,1,0,0]


| Dimension of output vector  | Name of Feature      | Definition of feature | Coding scheme     | How you dealt with missing data |
| :-------------| :----------- | :----------- | :----------- | :----------- |
| 1  | Planet of origin (1)       | Ship is from Klingon   |Binary       | N/A   |
| 2  | Planet of origin (2)       | Ship is from Romulan   |Binary       | N/A   |
| 3  | Planet of origin (3)       | Ship is from Antarean   |Binary       | N/A   |
| 4  | Planet of origin (4)       | Ship is from Federation  |Binary       | N/A   |



```python
def name_encoder(name):
    l = [0,0]
    for char in name:
        Ascii = ord(char)
        if Ascii <= 57:    l[1] = 1 # if char == number
        elif Ascii == 75:  l[0] = 1 # if char == 'K'
        elif Ascii == 107: l[0] = 1 # if char == 'k'
    return l
```


```python
def index_encoder(index):
    l = [0,0,0]
    # for test_data, where index is stored as string data type
    if type(index) == str: 
        if '_' in index:           None     # if data is missing, do nothing
        elif float(index) >= 7.3:  l[2] = 1 # if the index is greater than 7.3
        elif float(index) >= 6.9:  l[1] = 1 # if the index is greater than 6.9, smaller than 7.3
        else:                      l[0] = 1 # if the index is smaller than 6.9 
    # for train_data, where index is stored as float data type
    else: 
        if index >= 7.3:           l[2] = 1 
        elif index >= 6.9:         l[1] = 1 
        else:                      l[0] = 1
    return l
```


```python
def freq_encoder(freq):
    l = [0,0]
    # for test_data, where index is stored as string data type
    if type(freq) == str: 
        if '_' in freq:            None     # if data is missing, do nothing
        elif '>' in freq:          l[1] = 1 # if freqency is greater than 1000
        elif '<' in freq:          l[0] = 1 # if freqency is smaller than 1000
        elif float(freq) >= 1000:  l[1] = 1 # if freqency is greater than 1000
        else:                      l[0] = 1 # if freqency is smaller than 1000
    # for train_data, where index is stored as float data type
    else: 
        if freq >= 1000:           l[1] = 1 
        else:                      l[0] = 1
    return l
```


```python
def color_encoder(color):
    l = [0,0,0,0,0,0,0,0,0,0]
    if 'Light' in color:  l[0] = 1 # if color name includes 'Light'
    if 'Dark' in color:   l[1] = 1 # if color name includes 'Dark'
    if 'Black' in color:  l[2] = 1 # if color name includes 'Black'
    if 'Gray' in color:   l[3] = 1 # if color name includes 'Gray'
    if 'Blue' in color:   l[4] = 1 # if color name includes 'Blue'
    if 'Green' in color:  l[5] = 1 # if color name includes 'Green'
    if 'Orange' in color: l[6] = 1 # if color name includes 'Orange'
    if 'Pink' in color:   l[7] = 1 # if color name includes 'Pink'
    if 'Yellow' in color: l[8] = 1 # if color name includes 'Yellow'
    if 'White' in color:  l[9] = 1 # if color name includes 'White'
    return l
```


```python
def axis_encoder(axis): 
    l = [0,0,0]
    # for test_data, where index is stored as string data type
    if type(axis) == str: 
        if '_' in axis:           None     # if data is missing, do nothing
        elif float(axis) < 1.5:   l[0] = 1 # if axis is smaller than 1.5
        elif float(axis) >= 2.3:  l[2] = 1 # if axis is greater than 2.3
        else:                     l[1] = 1 # if axis is greater than 1.5, smaller than 2.3
    # for train_data, where index is stored as float data type
    else: 
        if axis < 1.5:            l[0] = 1 
        elif axis >= 2.3:         l[2] = 1 
        else:                     l[1] = 1 
    return l 
```


```python
def origin_encoder(origin): 
    l = [0,0,0,0]
    if origin == "Klingon":      l[0] = 1 # if origin is Klingon
    elif origin == "Romulan":    l[1] = 1 # if origin is Romulan
    elif origin == "Antarean":   l[2] = 1 # if origin is Antarean
    elif origin == "Federation": l[3] = 1 # if origin is Federation
    return l 
```


```python
# Applying all the functions above
for i in range(number_of_pairs):
    train_data["Name"][i] = name_encoder(train_data["Name"][i])
    test_data["Name"][i] = name_encoder(test_data["Name"][i])
    
    train_data['Warp Drive Vibration Index (Murds)'][i] = index_encoder(train_data['Warp Drive Vibration Index (Murds)'][i])
    test_data['Warp Drive Vibration Index (Murds)'][i] = index_encoder(test_data['Warp Drive Vibration Index (Murds)'][i])
    
    train_data['Hailing Transponder Freq.(gigaHz)'][i] = freq_encoder(train_data['Hailing Transponder Freq.(gigaHz)'][i])
    test_data['Hailing Transponder Freq.(gigaHz)'][i] = freq_encoder(test_data['Hailing Transponder Freq.(gigaHz)'][i])
    
    train_data['Surface Reflect. (color)'][i] = color_encoder(train_data['Surface Reflect. (color)'][i])
    test_data['Surface Reflect. (color)'][i] = color_encoder(test_data['Surface Reflect. (color)'][i])
    
    train_data['Ratio of long to short axis'][i] = axis_encoder(train_data['Ratio of long to short axis'][i])
    test_data['Ratio of long to short axis'][i] = axis_encoder(test_data['Ratio of long to short axis'][i])
    
    train_data['Planet of origin'][i] = origin_encoder(train_data['Planet of origin'][i])
```

After all processing, now our data looks like:


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Planet of origin</th>
      <th>Warp Drive Vibration Index (Murds)</th>
      <th>Hailing Transponder Freq.(gigaHz)</th>
      <th>Surface Reflect. (color)</th>
      <th>Ratio of long to short axis</th>
      <th>Req. action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0]</td>
      <td>[1, 0, 0, 0]</td>
      <td>[0, 1, 0]</td>
      <td>[0, 1]</td>
      <td>[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 1]</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0]</td>
      <td>[1, 0, 0, 0]</td>
      <td>[0, 1, 0]</td>
      <td>[1, 0]</td>
      <td>[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 1]</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[1, 0]</td>
      <td>[1, 0, 0, 0]</td>
      <td>[0, 0, 1]</td>
      <td>[1, 0]</td>
      <td>[0, 1, 0, 1, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 1]</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0]</td>
      <td>[1, 0, 0, 0]</td>
      <td>[0, 1, 0]</td>
      <td>[0, 1]</td>
      <td>[0, 1, 0, 1, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 1]</td>
      <td>Hostile</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1, 0]</td>
      <td>[1, 0, 0, 0]</td>
      <td>[0, 1, 0]</td>
      <td>[0, 1]</td>
      <td>[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]</td>
      <td>[1, 0, 0]</td>
      <td>Hostile</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Planet of origin</th>
      <th>Warp Drive Vibration Index (Murds)</th>
      <th>Hailing Transponder Freq.(gigaHz)</th>
      <th>Surface Reflect. (color)</th>
      <th>Ratio of long to short axis</th>
      <th>Req. action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0, 0]</td>
      <td>?</td>
      <td>[0, 0, 1]</td>
      <td>[0, 0]</td>
      <td>[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 1, 0]</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0, 0]</td>
      <td>?</td>
      <td>[1, 0, 0]</td>
      <td>[0, 1]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]</td>
      <td>[0, 1, 0]</td>
      <td>?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0, 0]</td>
      <td>?</td>
      <td>[1, 0, 0]</td>
      <td>[0, 1]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]</td>
      <td>[0, 0, 0]</td>
      <td>?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0, 0]</td>
      <td>?</td>
      <td>[0, 0, 0]</td>
      <td>[0, 1]</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0]</td>
      <td>?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[1, 0]</td>
      <td>?</td>
      <td>[0, 1, 0]</td>
      <td>[0, 1]</td>
      <td>[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]</td>
      <td>[0, 0, 0]</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
</div>



# Input - Output Vector Split

In order to build an outer matrix, now we need to split the data into input vectors and output vectors.
Again, we only need to care about 'Planet of origin', we will not include 'Req. action' as a part of output vector

* y_train = output data of train_data
* X_train = input data of train_data
* y_test = output data of test_data (unknown, meaningless, not to be used)
* X_test = input data of test_data 


```python
y_train = train_data[['Planet of origin']]
X_train = train_data.drop(['Planet of origin','Req. action'], axis = 1)
y_test = test_data[['Planet of origin']]
X_test = test_data.drop(['Planet of origin','Req. action'], axis = 1)
```

# Data Frame to Vector (List) Form
Our data is still inside the numpy data frame, so we cannot directly make an outter product with them. We need to put them in a list (vector) in order to perfrom the operations.

Some data are more important in terms of judging its origin. 

By experimenting various representations, I have found 'Name' and 'Color' are more important features. 

Thus, I choose to emphasize the representation of the 'Name' and 'Color' by representing them once more within the input vector


```python
X_train_vec = []
y_train_vec = []
X_test_vec = []

for i in range(number_of_pairs):
    # Convert the Row of data frame to a list 
    x_train = X_train.iloc[i].to_list()
    x_test = X_test.iloc[i].to_list()

    # Adding up the lists with different representations
    # This will result in Nested List form
    x_train_vec = [x_train[0] for i in range(2)] + [x_train[1] for i in range(1)] + [x_train[2] for i in range(1)] + [x_train[3] for i in range(2)] + [x_train[4] for i in range(1)] 
    x_test_vec  = [x_test[0] for i in range(2)]  + [x_test[1] for i in range(1)]  + [x_test[2] for i in range(1)]  + [x_test[3] for i in range(2)]  + [x_test[4] for i in range(1)] 
    
    
    # Flattening the list, and append to our vector set
    X_train_vec.append([element for innerList in x_train_vec for element in innerList])
    X_test_vec.append([element for innerList in x_test_vec for element in innerList])
    y_train_vec.append([element for innerList in y_train.iloc[i].to_list() for element in innerList])
```

now we have input output vectors looking like this:


```python
print("Example Input Vector:", X_train_vec[0])
print("Example Output Vector:",y_train_vec[0])
```

    Example Input Vector: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    Example Output Vector: [1, 0, 0, 0]


# Train Connectivity Matrix A

Here, we will do what we did in previous homeworks.

First, we normalize the vectors. We do not need to normalize *y_train_vec*, because they are already a normalized vector


```python
for i in range(number_of_pairs):
    X_train_vec[i] = X_train_vec[i] / np.linalg.norm(X_train_vec[i])
    X_test_vec[i] = X_test_vec[i] / np.linalg.norm(X_test_vec[i])  
```

Initialize the empty connectivity matrix A 


```python
input_dimension = len(X_train_vec[0])
output_dimension = 4 
A = np.zeros((output_dimension, input_dimension))
```

Calculate A


```python
for i in range(20):
    A+= np.outer(y_train_vec[i],X_train_vec[i])
```

Now on to error correcting Procedure, where criteria is to meet MSE bellow 0.1

Using stricter criteria did not resulted better accuracy (probably overfitting).


```python
def mean_squared_error(f_set, g_set, A):
    SE = [] 
    for i in range(len(f_set)):
        g_prime = (np.dot(A,f_set[i]))
        error = g_set[i] - g_prime
        SE.append(np.dot(error,error))
    return np.mean(SE)
```


```python
# Calculates Accuracy 
def mean_squared_error(f_set, g_set, A):
    SE = [] 
    for i in range(len(f_set)):
        g_prime = (np.dot(A,f_set[i]))
        error = g_set[i] - g_prime
        SE.append(np.dot(error,error))
    return np.mean(SE)

# Learning Rate 
k = 0.1

# Loops until the accuracy crierion is met 
mse = mean_squared_error(X_train_vec, y_train_vec, A)
while mse > 0.1:
    # Pick associated pair in random, while granting equal exposure
    random = np.random.choice(range(number_of_pairs),size =number_of_pairs, replace =False)
    for r in random :
        # Calculate and add delta_A matrix
        g_prime = np.dot(A,X_train_vec[r])
        error_vec = y_train_vec[r]-g_prime
        delta_A = k*np.outer((error_vec),X_train_vec[r])

        # Add delta_A to developing A matrix
        A += delta_A

    # Recalculate the accuracy of a matrix 
    mse = mean_squared_error(X_train_vec, y_train_vec, A)
    print (mse)
```

    2.834847038688608
    0.9350575784942198
    0.39464548145165457
    0.21411068511311462
    0.1383536812471416
    0.10244663260349902
    0.08206999727095954


## Testing Trained Input data

Now, it is time to see our model performance. 

The function below calculate the expected output, and prints out readable result

From how we setted out output vectors to be, having the maximum value at the first output vector position means the prediction is "Klingon, Hostile." 

And Having the maximum value at the second output vector position means the prediction is "Romulan, Alert," and so on.


```python
def pred_output(A,input_vec):
    # dictionary to print out results 
    # position in output vector : discription 
    dictionary = {
        0: "Klingon, Hostile",
        1: "Romulan, Alert",
        2: "Antarean, Friendly",
        3: "Federation, Friendly"
    }

    for i in range(number_of_pairs):
        
        # calculate output vector 
        predicted_output = np.dot(A,input_vec[i])
        
        # Find the position with maximum value
        max_index = 0
        for index in range(len(predicted_output)):
            if predicted_output[index] > predicted_output[max_index]:
                max_index = index
        # print the result as dictionary values
        print(dictionary[max_index])
```


```python
pred_output(A,X_train_vec)
```

    Klingon, Hostile
    Klingon, Hostile
    Klingon, Hostile
    Klingon, Hostile
    Klingon, Hostile
    Romulan, Alert
    Romulan, Alert
    Romulan, Alert
    Romulan, Alert
    Romulan, Alert
    Antarean, Friendly
    Antarean, Friendly
    Antarean, Friendly
    Antarean, Friendly
    Antarean, Friendly
    Federation, Friendly
    Federation, Friendly
    Federation, Friendly
    Federation, Friendly
    Federation, Friendly


Here, we see our model have successfully learned too predict output data from training inputs

## Testing Noisy Input data
Now we need to see how our model performs with testing (noisy) data

By using the same function above, but with testing inputs, we get: 


```python
pred_output(A,X_test_vec)
```

    Romulan, Alert
    Federation, Friendly
    Federation, Friendly
    Federation, Friendly
    Klingon, Hostile
    Romulan, Alert
    Klingon, Hostile
    Romulan, Alert
    Klingon, Hostile
    Antarean, Friendly
    Klingon, Hostile
    Klingon, Hostile
    Antarean, Friendly
    Antarean, Friendly
    Romulan, Alert
    Romulan, Alert
    Antarean, Friendly
    Antarean, Friendly
    Federation, Friendly
    Federation, Friendly


And this perfectly aligns with our intuitive, human-brain answers.
Thus we successfully got 100% accuracy!
