---
layout: single
classes: wide
title:  "Fake News Classification"
categories: 
  - Machine Learning
tag: [Tensorflow]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---

“Fake News”—is one of the defining features of contemporary democratic life. 

In this Blog Post, we will develop and assess a fake news classifier using Tensorflow.

*Note: Working on this Blog Post in Google Colab is highly recommended.*

# Import


```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import numpy as np
import pandas as pd
import re
import string
from matplotlib import pyplot as plt
import plotly.express as px 
import plotly.io as pio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.decomposition import PCA 
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.


# Acquire Training Data


```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
raw_df = pd.read_csv(train_url)
raw_df.head()
```





  <div id="df-9b50b56c-0412-4c69-bb93-8194c0fd8d21">
    <div class="colab-df-container">
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
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9b50b56c-0412-4c69-bb93-8194c0fd8d21')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9b50b56c-0412-4c69-bb93-8194c0fd8d21 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9b50b56c-0412-4c69-bb93-8194c0fd8d21');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Make a Dataset

## make_dataset Funciton
This function does two things:

1. Remove stopwords (such as “the,” “and,” or “but”) from the article text and title. 
2. Constructs and returns tf.data.Dataset with two inputs and one output. The input is in the form of (title, text), and the output consist only of the fake column.



```python
def make_dataset(df):
  stop_words = stopwords.words('english')

  # remove stop words from titles and texts
  df["title"] = df["title"].apply(lambda title: ' '.join([word for word in title.split() if word not in (stop_words)])) 
  df['text'] = df['text'].apply(lambda text: ' '.join([word for word in text.split() if word not in (stop_words)])) 
  
  dataset = tf.data.Dataset.from_tensor_slices((
      # dictionary for input data
       {"title": df[["title"]], "text": df[["text"]]},
       # dictionary for output data
        { "fake": df["fake"]}   
        ))
  
  return dataset.batch(100) # batch the dataset
```

## Split Dataset for Validation
spliting 20% of dataset to use for validation.


```python
# Process data
df = make_dataset(raw_df)
df = df.shuffle(buffer_size = len(df))

# Split the dataset 
train_size = int(0.8*len(df)) 
val_size   = int(0.2*len(df)) 

train = df.take(train_size)
val = df.skip(train_size).take(val_size)

# Print Results 
print("Train Size: ", len(train))
print("Validation Size: ", len(val))
```

    Train Size:  180
    Validation Size:  45


##Base Rate


```python
labels_iterator= train.unbatch().map(lambda dict_title_text, label: label).as_numpy_iterator()

real = 0 
fake = 0 

for label in labels_iterator:
    if label["fake"]==0: #if label is not fake, increase the count of the real
        real +=1 
    else: #if label is fake, increase the count of the fake
        fake +=1

print("Real: ", real)
print("Fake: ", fake)
```

    Real:  8521
    Fake:  9479


Our base rate (accuracy when a model makes only the same guess) prediction is somewhere around 50% 

## Text Vectorization



```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

title_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

title_vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
```

# Create Models

Using Functional API (rather than Sequential API)

## First Model 
Using titles to detect fake news


```python
# Input Layer
titles_input = keras.Input(
    shape = (1,), 
    name = "title",
    dtype = "string"
)

# Hidden Layers
titles_features = title_vectorize_layer(titles_input) 
titles_features = layers.Embedding(size_vocabulary, output_dim = 2, name = "embedding")(titles_features) 
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.GlobalAveragePooling1D()(titles_features)
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.Dense(32)(titles_features)

# Output Layer
output = layers.Dense(2, name = "fake")(titles_features) 

```


```python
model1 = keras.Model(
    inputs = titles_input,
    outputs = output
) 
```


```python
keras.utils.plot_model(model1)
```




    
![png](/assets/images/Fake News Classification/output_19_0.png)
    




```python
# compile model1
model1.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
# fit model1
history1 = model1.fit(train, 
                      validation_data = val,
                      epochs = 20, 
                      verbose = False)
```

    /usr/local/lib/python3.10/dist-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)



```python
# Visualize Accuracy
plt.plot(history1.history["accuracy"], label = "training")
plt.plot(history1.history["val_accuracy"], label = "validation")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8a200e80a0>




    
![png](/assets/images/Fake News Classification/output_22_1.png)
    



```python
print("<Model 1 Final Accuracy: ", history1.history["val_accuracy"][-1])
```

    <Model 1 Final Accuracy:  0.9857777953147888


## Second Model 
Using article text to detect fake news


```python
# Text Vectorization Layer
text_vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

text_vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```


```python
# Input Layer
text_input = keras.Input(
    shape = (1,), 
    name = "text",
    dtype = "string"
)

# Hidden Layer
text_features = text_vectorize_layer(text_input) 
text_features = layers.Embedding(size_vocabulary, output_dim = 2, name = "embedding2")(text_features) 
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)

# Output layer
output = layers.Dense(2, name = "fake")(text_features) 

```


```python
# Create model2
model2 = keras.Model(
    inputs = text_input,
    outputs = output
) 
```


```python
# Visualize model2
keras.utils.plot_model(model2)
```




    
![png](/assets/images/Fake News Classification/output_28_0.png)
    




```python
# Compile model2
model2.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
# Fit model2
history2 = model2.fit(train, 
                      validation_data = val,
                      epochs = 20, 
                      verbose = False)
```

    /usr/local/lib/python3.10/dist-packages/keras/engine/functional.py:639: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)



```python
# Visualize Accuracy
plt.plot(history2.history["accuracy"], label = "training")
plt.plot(history2.history["val_accuracy"], label = "validation")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f89a74064a0>




    
![png](/assets/images/Fake News Classification/output_31_1.png)
    



```python
print("<Model 2 Final Accuracy: ", history2.history["val_accuracy"][-1])
```

    <Model 2 Final Accuracy:  0.9913333058357239


## Third Model
Using article titles & text to detect fake news

Two pipeline are same exact code from first two models


```python
# First Pipeline
titles_features = title_vectorize_layer(titles_input) 
titles_features = layers.Embedding(size_vocabulary, output_dim = 2, name = "embedding_title")(titles_features) 
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.GlobalAveragePooling1D()(titles_features)
titles_features = layers.Dropout(0.2)(titles_features)
titles_features = layers.Dense(32)(titles_features)

# Second Pipeline
text_features = text_vectorize_layer(text_input) 
text_features = layers.Embedding(size_vocabulary, output_dim = 2, name = "embedding_text")(text_features) 
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)

# Concatonate Two Pipelines
main = layers.concatenate([titles_features, text_features], axis = 1)

# Output Layer
output=layers.Dense(2,name='fake')(main)
```


```python
# Create model3
model3 = keras.Model(
    inputs = [titles_input, text_input],
    outputs = output
)
```


```python
# Visualize model3
keras.utils.plot_model(model3)
```




    
![png](/assets/images/Fake News Classification/output_36_0.png)
    




```python
# Compile model3
model3.compile(optimizer="adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
# Fit model3
history3 = model3.fit(train, 
                      validation_data = val,
                      epochs = 20, 
                      verbose = False)
```


```python
plt.plot(history3.history["accuracy"], label = "training")
plt.plot(history3.history["val_accuracy"], label = "validation")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f89a512add0>




    
![png](/assets/images/Fake News Classification/output_39_1.png)
    



```python
print("<Model 3 Final Accuracy: ", history3.history["val_accuracy"][-1])
```

    <Model 3 Final Accuracy:  0.998651385307312


1. Model 1 Final Accuracy:  0.987333357334137
2. Model 2 Final Accuracy:  0.9877777695655823
3. Model 3 Final Accuracy:  0.9973333477973938


Based on three models’ performances, it is best to use both title and text upon detecting fake news.

# Model Evaluation


```python
# Download & Process Test Data
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true" #test data
test_data = pd.read_csv(test_url)
test_data = make_dataset(test_data)

# Print Model Performance
print(model3.evaluate(test_data))
```

    225/225 [==============================] - 2s 9ms/step - loss: 0.0191 - accuracy: 0.9947
    [0.019058916717767715, 0.9946991205215454]


The final model got 99.5% accuracy. 


Meaning, that the model will detect the fake news 99.5% of the time

# Embedding Visualization 

looking at the embedding learned by our model

Using 2D embedding

## embedding_title Visualization


```python
weights = model3.get_layer("embedding_title").get_weights()[0] # get the weights from the embedding layer
vocab = title_vectorize_layer.get_vocabulary()                 # get the vocabulary from our data prep for later

#Reducing to 2D dimension
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word': vocab,
    'x0':weights[:, 0],
    'x1':weights[:, 1]
})

# Plot Embedding Layer
fig = px.scatter(embedding_df,
                x = "x0",
                y = "x1",
                size=[2]*len(embedding_df),
                hover_name = "word")

fig.show()
```

![png](/assets/images/Fake News Classification/newplot_1.png)




## embedding_text Visualization


```python
weights = model3.get_layer("embedding_text").get_weights()[0] # get the weights from the embedding layer
vocab = title_vectorize_layer.get_vocabulary()                 # get the vocabulary from our data prep for later

#Reducing to 2D dimension
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word': vocab,
    'x0':weights[:, 0],
    'x1':weights[:, 1]
})

# Plot Embedding Layer
fig = px.scatter(embedding_df,
                x = "x0",
                y = "x1",
                size=[2]*len(embedding_df),
                hover_name = "word")

fig.show()
```

![png](/assets/images/Fake News Classification/newplot_2.png)



Embedding layer exposes minor detials of fake news that are not identifiable with normal human's understanding. For example, "trumps" is highly correlated with fake news, but "trump's" is highly correlated with real news. These minor differences are hard to notice but we can see from embedding visualization that our model is taking advantage of it.
