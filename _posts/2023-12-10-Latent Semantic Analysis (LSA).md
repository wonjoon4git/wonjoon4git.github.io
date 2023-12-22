---
layout: single
classes: wide
title:  "Latent Semantic Analysis (LSA)"
categories: 
  - NLP
tag: [LSA]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---



# Latent Semantic Analysis
**Latent Semantic Analysis (LSA)** is a method that uses **Singular Value Decomposition (SVD)** to explore relationships between documents and their constituent words. Here, we will apply a simplified form of LSA to a subset of Yelp restaurant reviews.

The dataset, contained in "reviews limited vocab.txt", includes 100,000 reviews from a training set, each review on a separate line. This dataset has been filtered to remove less common words, leaving a vocabulary of 1,731 words. From this data, we construct an $n × d$ "document-term" matrix *A*, where "n" is the number of reviews (100,000) and "d" is the number of words in the vocabulary (1,731). Each element $A_{i,j}$ in this matrix indicates how many times word "j" appears in review "i".

In this matrix:
- Each row represents a different review, showing the frequency of each word in that review.
- Each column corresponds to a specific word from the vocabulary, indicating its frequency across all reviews.

However, in LSA, we don't use the matrix A directly. Instead, we derive representations of reviews and words using a truncated SVD of A. This involves projecting columns of A onto a subspace formed by the left singular vectors of A that correspond to the "k" largest singular values, where "k" is a chosen parameter in LSA. This projection helps to reduce "noise" in the data, which might result from random occurrences of words in reviews.

An important application of LSA is to find words that are similar to a given word. This similarity is usually measured by the cosine similarity between their vector representations. Words are considered similar if their cosine similarity is high (close to 1) and dissimilar if it's low (close to 0 or negative).


## Import Libriries


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## Code Sinippet Provided in HW description


```python
vocab = {}
vocab_size = 0
reviews = []

with open('reviews_limited_vocab.txt', 'r') as f:
    for line in f.readlines():
        words = line.strip().split(' ')
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1
        reviews.append([vocab[word] for word in words])

invert_vocab = [''] * vocab_size
for (word, word_id) in vocab.items():
    invert_vocab[word_id] = word
invert_vocab = np.array(invert_vocab)

words_to_compare = ['excellent', 'amazing', 'delicious', 'fantastic', 'gem', 'perfectly', 'incredible', 'worst', 'mediocre', 'bland', 'meh', 'awful', 'horrible', 'terrible']

k_to_try = [ 2, 4, 8 ]
```

## Function Declarations
These are necessary functions to perform LSA


```python
def create_document_term_matrix(reviews, vocab):
    """Function to create cocument term matrix A"""
    # Initialize the document-term matrix A with zeros
    A = np.zeros((len(reviews), len(vocab)))

    # Construct the document-term matrix A
    for i, review in enumerate(reviews):
        for word_id in review:
            A[i, word_id] += 1
    return A

def cosine_similarity(v1, v2):
    """ Function to compute cosine similarity between two vectors """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_similarity_matrix(vectors):
    """ Function to calculate cosine similarity matrix """
    n = vectors.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    for i in range(vectors.shape[0]):
        for j in range(i, vectors.shape[0]):
            sim = cosine_similarity(vectors[i, :], vectors[j, :]) 
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # fill in the symmetric element
    
    return similarity_matrix


def lsa_cosine_similarity_matrix(A, k_to_try, words_to_compare, vocab, invert_vocab, ignore_first_singular_value=False):
    """Function to apply LSA and compute the cosine similarity matrix for a given k"""
    
    # Perform SVD on document-term matrix A
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # To store all results
    similarity_matrices = {}
    
    # Trying different k
    for k in k_to_try:
        if ignore_first_singular_value:
            # Ignore the first singular value by starting from the second one
            S_k = S[1:k+1]
            U_k = U[:, 1:k+1]
            Vt_k = Vt[1:k+1, :]
        else:
            # Use the first k singular values
            S_k = S[:k]
            U_k = U[:, :k]
            Vt_k = Vt[:k, :]
            
            
        # Interesting Visualization: Visualize Singular values 
        sns.barplot(x=list(range(len(S_k))), y = S_k)
        plt.xlabel('Latent Component')    # Set the x-axis label
        plt.ylabel('Relative Importance of each component')    # Set the y-axis label
        plt.title(f'Magnitude of Singular Values When k = {k}')    # Set the title of the plot
        plt.show()    # Display the plot

        
        # Get the word vectors from Vt
        word_vectors = Vt_k.T

        # Select the vectors corresponding to the words we want to compare
        words_indices = [vocab[word] for word in words_to_compare]
        selected_vectors = word_vectors[words_indices, :]

        # Calculate the cosine similarity matrix for the selected word vectors
        k_sim_matrix = cosine_similarity_matrix(selected_vectors)

        similarity_matrices[k] = k_sim_matrix
    
    return similarity_matrices


def display_similarity_matrix(similarity_matrix, words_to_compare):
    """ Function to display a similarity matrix using pandas DataFrame """
    
    # Create a DataFrame from the similarity matrix
    df = pd.DataFrame(similarity_matrix, index=words_to_compare, columns=words_to_compare)

    # Displaying df will make more prettier output, but only works on Jupyternotebook
    display(df)
    
    # For terminal, and more generally, 
    #print(df)
```

# Constructing Document-term Matrix
We first construct the document term matrix A from the Yelp review data. For each value of $k$ in {2, 4, 8}, apply LSA with the given $k$, and then compute the cosine similarity between all pairs of words in the "words_to_compare" we declared above.


```python
# Initialize document-term matrix A
A = create_document_term_matrix(reviews, vocab)
```


```python
# Perform LSA for each k in k_to_try 
# And show the magnitudes of singular values for each k
similarity_matrices = lsa_cosine_similarity_matrix(A, k_to_try, words_to_compare, vocab, invert_vocab)
```


    
![png](/assets/images/LSA/output_8_0.png)
    



    
![png](/assets/images/LSA/output_8_1.png)
    



    
![png](/assets/images/LSA/output_8_2.png)
    



```python
# Print out the tables
for k, matrix in similarity_matrices.items():
    print()
    print(f"Cosine Similarity Matrix for k={k}")
    display_similarity_matrix(matrix, words_to_compare)
```

    
    Cosine Similarity Matrix for k=2



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.852972</td>
      <td>0.849728</td>
      <td>0.957973</td>
      <td>0.966526</td>
      <td>0.994959</td>
      <td>0.982914</td>
      <td>0.396789</td>
      <td>0.587761</td>
      <td>0.794862</td>
      <td>0.129143</td>
      <td>0.328731</td>
      <td>0.243410</td>
      <td>0.552588</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.852972</td>
      <td>1.000000</td>
      <td>0.999981</td>
      <td>0.966852</td>
      <td>0.958337</td>
      <td>0.901014</td>
      <td>0.934473</td>
      <td>0.817559</td>
      <td>0.923624</td>
      <td>0.994713</td>
      <td>0.627740</td>
      <td>0.773346</td>
      <td>0.713879</td>
      <td>0.906369</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.849728</td>
      <td>0.999981</td>
      <td>1.000000</td>
      <td>0.965254</td>
      <td>0.956553</td>
      <td>0.898314</td>
      <td>0.932253</td>
      <td>0.821104</td>
      <td>0.925977</td>
      <td>0.995329</td>
      <td>0.632542</td>
      <td>0.777252</td>
      <td>0.718196</td>
      <td>0.908964</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.957973</td>
      <td>0.966852</td>
      <td>0.965254</td>
      <td>1.000000</td>
      <td>0.999505</td>
      <td>0.981910</td>
      <td>0.994406</td>
      <td>0.643423</td>
      <td>0.795137</td>
      <td>0.935519</td>
      <td>0.408171</td>
      <td>0.585831</td>
      <td>0.511410</td>
      <td>0.768447</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.966526</td>
      <td>0.958337</td>
      <td>0.956553</td>
      <td>0.999505</td>
      <td>1.000000</td>
      <td>0.987383</td>
      <td>0.997237</td>
      <td>0.619013</td>
      <td>0.775659</td>
      <td>0.923938</td>
      <td>0.379240</td>
      <td>0.560036</td>
      <td>0.484113</td>
      <td>0.747929</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.994959</td>
      <td>0.901014</td>
      <td>0.898314</td>
      <td>0.981910</td>
      <td>0.987383</td>
      <td>1.000000</td>
      <td>0.996417</td>
      <td>0.486836</td>
      <td>0.665928</td>
      <td>0.851704</td>
      <td>0.227931</td>
      <td>0.421780</td>
      <td>0.339446</td>
      <td>0.633381</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.982914</td>
      <td>0.934473</td>
      <td>0.932253</td>
      <td>0.994406</td>
      <td>0.997237</td>
      <td>0.996417</td>
      <td>1.000000</td>
      <td>0.558965</td>
      <td>0.726634</td>
      <td>0.892970</td>
      <td>0.309460</td>
      <td>0.496950</td>
      <td>0.417780</td>
      <td>0.696556</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>0.396789</td>
      <td>0.817559</td>
      <td>0.821104</td>
      <td>0.643423</td>
      <td>0.619013</td>
      <td>0.486836</td>
      <td>0.558965</td>
      <td>1.000000</td>
      <td>0.975838</td>
      <td>0.872371</td>
      <td>0.961466</td>
      <td>0.997333</td>
      <td>0.986885</td>
      <td>0.984297</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>0.587761</td>
      <td>0.923624</td>
      <td>0.925977</td>
      <td>0.795137</td>
      <td>0.775659</td>
      <td>0.665928</td>
      <td>0.726634</td>
      <td>0.975838</td>
      <td>1.000000</td>
      <td>0.958103</td>
      <td>0.878165</td>
      <td>0.957287</td>
      <td>0.927768</td>
      <td>0.999083</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.794862</td>
      <td>0.994713</td>
      <td>0.995329</td>
      <td>0.935519</td>
      <td>0.923938</td>
      <td>0.851704</td>
      <td>0.892970</td>
      <td>0.872371</td>
      <td>0.958103</td>
      <td>1.000000</td>
      <td>0.704360</td>
      <td>0.834363</td>
      <td>0.782018</td>
      <td>0.944963</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>0.129143</td>
      <td>0.627740</td>
      <td>0.632542</td>
      <td>0.408171</td>
      <td>0.379240</td>
      <td>0.227931</td>
      <td>0.309460</td>
      <td>0.961466</td>
      <td>0.878165</td>
      <td>0.704360</td>
      <td>1.000000</td>
      <td>0.978968</td>
      <td>0.993236</td>
      <td>0.897838</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>0.328731</td>
      <td>0.773346</td>
      <td>0.777252</td>
      <td>0.585831</td>
      <td>0.560036</td>
      <td>0.421780</td>
      <td>0.496950</td>
      <td>0.997333</td>
      <td>0.957287</td>
      <td>0.834363</td>
      <td>0.978968</td>
      <td>1.000000</td>
      <td>0.996035</td>
      <td>0.968787</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>0.243410</td>
      <td>0.713879</td>
      <td>0.718196</td>
      <td>0.511410</td>
      <td>0.484113</td>
      <td>0.339446</td>
      <td>0.417780</td>
      <td>0.986885</td>
      <td>0.927768</td>
      <td>0.782018</td>
      <td>0.993236</td>
      <td>0.996035</td>
      <td>1.000000</td>
      <td>0.942893</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>0.552588</td>
      <td>0.906369</td>
      <td>0.908964</td>
      <td>0.768447</td>
      <td>0.747929</td>
      <td>0.633381</td>
      <td>0.696556</td>
      <td>0.984297</td>
      <td>0.999083</td>
      <td>0.944963</td>
      <td>0.897838</td>
      <td>0.968787</td>
      <td>0.942893</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Cosine Similarity Matrix for k=4



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.838417</td>
      <td>0.832617</td>
      <td>0.877425</td>
      <td>0.367586</td>
      <td>0.899915</td>
      <td>0.905511</td>
      <td>0.238379</td>
      <td>0.395601</td>
      <td>0.339647</td>
      <td>0.040934</td>
      <td>0.218489</td>
      <td>0.223238</td>
      <td>0.388603</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.838417</td>
      <td>1.000000</td>
      <td>0.925742</td>
      <td>0.920904</td>
      <td>0.499862</td>
      <td>0.753482</td>
      <td>0.917296</td>
      <td>0.342296</td>
      <td>0.545550</td>
      <td>0.333673</td>
      <td>0.346251</td>
      <td>0.296571</td>
      <td>0.234892</td>
      <td>0.400729</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.832617</td>
      <td>0.925742</td>
      <td>1.000000</td>
      <td>0.888833</td>
      <td>0.384616</td>
      <td>0.739697</td>
      <td>0.790877</td>
      <td>0.420572</td>
      <td>0.522929</td>
      <td>0.307485</td>
      <td>0.263357</td>
      <td>0.444029</td>
      <td>0.534012</td>
      <td>0.659214</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.877425</td>
      <td>0.920904</td>
      <td>0.888833</td>
      <td>1.000000</td>
      <td>0.715555</td>
      <td>0.641796</td>
      <td>0.951718</td>
      <td>0.015855</td>
      <td>0.206862</td>
      <td>0.013009</td>
      <td>-0.045417</td>
      <td>0.007949</td>
      <td>0.110768</td>
      <td>0.255955</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.367586</td>
      <td>0.499862</td>
      <td>0.384616</td>
      <td>0.715555</td>
      <td>1.000000</td>
      <td>-0.031121</td>
      <td>0.651108</td>
      <td>-0.620109</td>
      <td>-0.426191</td>
      <td>-0.623057</td>
      <td>-0.466367</td>
      <td>-0.628290</td>
      <td>-0.431861</td>
      <td>-0.368266</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.899915</td>
      <td>0.753482</td>
      <td>0.739697</td>
      <td>0.641796</td>
      <td>-0.031121</td>
      <td>1.000000</td>
      <td>0.730438</td>
      <td>0.590659</td>
      <td>0.713089</td>
      <td>0.713688</td>
      <td>0.392810</td>
      <td>0.542778</td>
      <td>0.378855</td>
      <td>0.546352</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.905511</td>
      <td>0.917296</td>
      <td>0.790877</td>
      <td>0.951718</td>
      <td>0.651108</td>
      <td>0.730438</td>
      <td>1.000000</td>
      <td>0.038360</td>
      <td>0.283291</td>
      <td>0.149389</td>
      <td>0.041035</td>
      <td>-0.022526</td>
      <td>-0.064762</td>
      <td>0.112853</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>0.238379</td>
      <td>0.342296</td>
      <td>0.420572</td>
      <td>0.015855</td>
      <td>-0.620109</td>
      <td>0.590659</td>
      <td>0.038360</td>
      <td>1.000000</td>
      <td>0.951259</td>
      <td>0.918977</td>
      <td>0.873191</td>
      <td>0.981945</td>
      <td>0.745096</td>
      <td>0.802657</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>0.395601</td>
      <td>0.545550</td>
      <td>0.522929</td>
      <td>0.206862</td>
      <td>-0.426191</td>
      <td>0.713089</td>
      <td>0.283291</td>
      <td>0.951259</td>
      <td>1.000000</td>
      <td>0.939858</td>
      <td>0.915015</td>
      <td>0.887203</td>
      <td>0.569918</td>
      <td>0.678875</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.339647</td>
      <td>0.333673</td>
      <td>0.307485</td>
      <td>0.013009</td>
      <td>-0.623057</td>
      <td>0.713688</td>
      <td>0.149389</td>
      <td>0.918977</td>
      <td>0.939858</td>
      <td>1.000000</td>
      <td>0.832056</td>
      <td>0.845680</td>
      <td>0.480877</td>
      <td>0.580324</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>0.040934</td>
      <td>0.346251</td>
      <td>0.263357</td>
      <td>-0.045417</td>
      <td>-0.466367</td>
      <td>0.392810</td>
      <td>0.041035</td>
      <td>0.873191</td>
      <td>0.915015</td>
      <td>0.832056</td>
      <td>1.000000</td>
      <td>0.788501</td>
      <td>0.410072</td>
      <td>0.478992</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>0.218489</td>
      <td>0.296571</td>
      <td>0.444029</td>
      <td>0.007949</td>
      <td>-0.628290</td>
      <td>0.542778</td>
      <td>-0.022526</td>
      <td>0.981945</td>
      <td>0.887203</td>
      <td>0.845680</td>
      <td>0.788501</td>
      <td>1.000000</td>
      <td>0.855481</td>
      <td>0.889138</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>0.223238</td>
      <td>0.234892</td>
      <td>0.534012</td>
      <td>0.110768</td>
      <td>-0.431861</td>
      <td>0.378855</td>
      <td>-0.064762</td>
      <td>0.745096</td>
      <td>0.569918</td>
      <td>0.480877</td>
      <td>0.410072</td>
      <td>0.855481</td>
      <td>1.000000</td>
      <td>0.980162</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>0.388603</td>
      <td>0.400729</td>
      <td>0.659214</td>
      <td>0.255955</td>
      <td>-0.368266</td>
      <td>0.546352</td>
      <td>0.112853</td>
      <td>0.802657</td>
      <td>0.678875</td>
      <td>0.580324</td>
      <td>0.478992</td>
      <td>0.889138</td>
      <td>0.980162</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Cosine Similarity Matrix for k=8



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.757212</td>
      <td>0.913244</td>
      <td>0.946878</td>
      <td>0.642860</td>
      <td>0.728462</td>
      <td>0.886134</td>
      <td>-0.446355</td>
      <td>-0.245513</td>
      <td>0.202035</td>
      <td>-0.057355</td>
      <td>-0.410635</td>
      <td>-0.285739</td>
      <td>-0.316515</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.757212</td>
      <td>1.000000</td>
      <td>0.885628</td>
      <td>0.866461</td>
      <td>0.461421</td>
      <td>0.307658</td>
      <td>0.801804</td>
      <td>-0.100694</td>
      <td>0.008962</td>
      <td>0.285970</td>
      <td>0.116652</td>
      <td>-0.011437</td>
      <td>0.145668</td>
      <td>0.058541</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.913244</td>
      <td>0.885628</td>
      <td>1.000000</td>
      <td>0.962863</td>
      <td>0.595044</td>
      <td>0.649186</td>
      <td>0.867623</td>
      <td>-0.439328</td>
      <td>-0.174808</td>
      <td>0.314393</td>
      <td>0.103448</td>
      <td>-0.272696</td>
      <td>-0.178628</td>
      <td>-0.249375</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.946878</td>
      <td>0.866461</td>
      <td>0.962863</td>
      <td>1.000000</td>
      <td>0.742615</td>
      <td>0.580032</td>
      <td>0.880537</td>
      <td>-0.456739</td>
      <td>-0.244197</td>
      <td>0.155815</td>
      <td>0.007043</td>
      <td>-0.401286</td>
      <td>-0.236362</td>
      <td>-0.284120</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.642860</td>
      <td>0.461421</td>
      <td>0.595044</td>
      <td>0.742615</td>
      <td>1.000000</td>
      <td>0.373033</td>
      <td>0.686922</td>
      <td>-0.623148</td>
      <td>-0.535959</td>
      <td>-0.319720</td>
      <td>-0.265467</td>
      <td>-0.696574</td>
      <td>-0.508087</td>
      <td>-0.522608</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.728462</td>
      <td>0.307658</td>
      <td>0.649186</td>
      <td>0.580032</td>
      <td>0.373033</td>
      <td>1.000000</td>
      <td>0.687379</td>
      <td>-0.648347</td>
      <td>-0.244231</td>
      <td>0.418162</td>
      <td>0.069499</td>
      <td>-0.388572</td>
      <td>-0.592331</td>
      <td>-0.600019</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.886134</td>
      <td>0.801804</td>
      <td>0.867623</td>
      <td>0.880537</td>
      <td>0.686922</td>
      <td>0.687379</td>
      <td>1.000000</td>
      <td>-0.404604</td>
      <td>-0.280858</td>
      <td>0.167819</td>
      <td>-0.132857</td>
      <td>-0.317890</td>
      <td>-0.254532</td>
      <td>-0.325077</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>-0.446355</td>
      <td>-0.100694</td>
      <td>-0.439328</td>
      <td>-0.456739</td>
      <td>-0.623148</td>
      <td>-0.648347</td>
      <td>-0.404604</td>
      <td>1.000000</td>
      <td>0.640314</td>
      <td>0.013827</td>
      <td>0.075029</td>
      <td>0.801343</td>
      <td>0.898425</td>
      <td>0.917839</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>-0.245513</td>
      <td>0.008962</td>
      <td>-0.174808</td>
      <td>-0.244197</td>
      <td>-0.535959</td>
      <td>-0.244231</td>
      <td>-0.280858</td>
      <td>0.640314</td>
      <td>1.000000</td>
      <td>0.679409</td>
      <td>0.792349</td>
      <td>0.855243</td>
      <td>0.596824</td>
      <td>0.715415</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.202035</td>
      <td>0.285970</td>
      <td>0.314393</td>
      <td>0.155815</td>
      <td>-0.319720</td>
      <td>0.418162</td>
      <td>0.167819</td>
      <td>0.013827</td>
      <td>0.679409</td>
      <td>1.000000</td>
      <td>0.851241</td>
      <td>0.499588</td>
      <td>0.078887</td>
      <td>0.140180</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>-0.057355</td>
      <td>0.116652</td>
      <td>0.103448</td>
      <td>0.007043</td>
      <td>-0.265467</td>
      <td>0.069499</td>
      <td>-0.132857</td>
      <td>0.075029</td>
      <td>0.792349</td>
      <td>0.851241</td>
      <td>1.000000</td>
      <td>0.504808</td>
      <td>0.119485</td>
      <td>0.235764</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>-0.410635</td>
      <td>-0.011437</td>
      <td>-0.272696</td>
      <td>-0.401286</td>
      <td>-0.696574</td>
      <td>-0.388572</td>
      <td>-0.317890</td>
      <td>0.801343</td>
      <td>0.855243</td>
      <td>0.499588</td>
      <td>0.504808</td>
      <td>1.000000</td>
      <td>0.834196</td>
      <td>0.863486</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>-0.285739</td>
      <td>0.145668</td>
      <td>-0.178628</td>
      <td>-0.236362</td>
      <td>-0.508087</td>
      <td>-0.592331</td>
      <td>-0.254532</td>
      <td>0.898425</td>
      <td>0.596824</td>
      <td>0.078887</td>
      <td>0.119485</td>
      <td>0.834196</td>
      <td>1.000000</td>
      <td>0.976510</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>-0.316515</td>
      <td>0.058541</td>
      <td>-0.249375</td>
      <td>-0.284120</td>
      <td>-0.522608</td>
      <td>-0.600019</td>
      <td>-0.325077</td>
      <td>0.917839</td>
      <td>0.715415</td>
      <td>0.140180</td>
      <td>0.235764</td>
      <td>0.863486</td>
      <td>0.976510</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


# Improving by Ignoring the Largest Singular Value 
Sometimes LSA can be improved by ignoring the direction corresponding to the largest singular value. Otherwise, we repeat the previous step


```python
# Now for Problem 6, repeat the process from Problem 5, but ignoring the first singular value
similarity_matrices_ignore_first = lsa_cosine_similarity_matrix(A, k_to_try, words_to_compare, vocab, invert_vocab, ignore_first_singular_value=True)
```


    
![png](/assets/images/LSA/output_11_0.png)
    



    
![png](/assets/images/LSA/output_11_1.png)
    



    
![png](/assets/images/LSA/output_11_2.png)
    



```python
# Repeat the process from Problem 5, for the matrices where the first singular value is ignored
for k, matrix in similarity_matrices_ignore_first.items():
    print(f"Cosine Similarity Matrix ignoring first singular value for k={k}")
    display_similarity_matrix(matrix, words_to_compare)
```

    Cosine Similarity Matrix ignoring first singular value for k=2



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.940112</td>
      <td>0.978348</td>
      <td>0.853661</td>
      <td>0.210705</td>
      <td>0.911815</td>
      <td>0.871880</td>
      <td>-0.037405</td>
      <td>0.141824</td>
      <td>0.297500</td>
      <td>-0.501481</td>
      <td>-0.064247</td>
      <td>-0.077675</td>
      <td>0.087554</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.940112</td>
      <td>1.000000</td>
      <td>0.849209</td>
      <td>0.980070</td>
      <td>0.531301</td>
      <td>0.717247</td>
      <td>0.986594</td>
      <td>-0.375793</td>
      <td>-0.204091</td>
      <td>-0.045750</td>
      <td>-0.766356</td>
      <td>-0.400562</td>
      <td>-0.412861</td>
      <td>-0.257247</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.978348</td>
      <td>0.849209</td>
      <td>1.000000</td>
      <td>0.727385</td>
      <td>0.003825</td>
      <td>0.977053</td>
      <td>0.751647</td>
      <td>0.170225</td>
      <td>0.343626</td>
      <td>0.488652</td>
      <td>-0.311564</td>
      <td>0.143681</td>
      <td>0.130346</td>
      <td>0.291829</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.853661</td>
      <td>0.980070</td>
      <td>0.727385</td>
      <td>1.000000</td>
      <td>0.689007</td>
      <td>0.564528</td>
      <td>0.999350</td>
      <td>-0.552396</td>
      <td>-0.394495</td>
      <td>-0.243283</td>
      <td>-0.878700</td>
      <td>-0.574598</td>
      <td>-0.585564</td>
      <td>-0.444087</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.210705</td>
      <td>0.531301</td>
      <td>0.003825</td>
      <td>0.689007</td>
      <td>1.000000</td>
      <td>-0.209259</td>
      <td>0.662436</td>
      <td>-0.984747</td>
      <td>-0.937785</td>
      <td>-0.870603</td>
      <td>-0.951410</td>
      <td>-0.989067</td>
      <td>-0.990963</td>
      <td>-0.955347</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.911815</td>
      <td>0.717247</td>
      <td>0.977053</td>
      <td>0.564528</td>
      <td>-0.209259</td>
      <td>1.000000</td>
      <td>0.593913</td>
      <td>0.376208</td>
      <td>0.535768</td>
      <td>0.663275</td>
      <td>-0.102018</td>
      <td>0.351172</td>
      <td>0.338536</td>
      <td>0.488858</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.871880</td>
      <td>0.986594</td>
      <td>0.751647</td>
      <td>0.999350</td>
      <td>0.662436</td>
      <td>0.593913</td>
      <td>1.000000</td>
      <td>-0.521990</td>
      <td>-0.361117</td>
      <td>-0.208163</td>
      <td>-0.860922</td>
      <td>-0.544724</td>
      <td>-0.555964</td>
      <td>-0.411502</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>-0.037405</td>
      <td>-0.375793</td>
      <td>0.170225</td>
      <td>-0.552396</td>
      <td>-0.984747</td>
      <td>0.376208</td>
      <td>-0.521990</td>
      <td>1.000000</td>
      <td>0.983894</td>
      <td>0.942926</td>
      <td>0.883321</td>
      <td>0.999639</td>
      <td>0.999186</td>
      <td>0.992188</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>0.141824</td>
      <td>-0.204091</td>
      <td>0.343626</td>
      <td>-0.394495</td>
      <td>-0.937785</td>
      <td>0.535768</td>
      <td>-0.361117</td>
      <td>0.983894</td>
      <td>1.000000</td>
      <td>0.987264</td>
      <td>0.785301</td>
      <td>0.978735</td>
      <td>0.975885</td>
      <td>0.998508</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.297500</td>
      <td>-0.045750</td>
      <td>0.488652</td>
      <td>-0.243283</td>
      <td>-0.870603</td>
      <td>0.663275</td>
      <td>-0.208163</td>
      <td>0.942926</td>
      <td>0.987264</td>
      <td>1.000000</td>
      <td>0.676805</td>
      <td>0.933636</td>
      <td>0.928729</td>
      <td>0.977103</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>-0.501481</td>
      <td>-0.766356</td>
      <td>-0.311564</td>
      <td>-0.878700</td>
      <td>-0.951410</td>
      <td>-0.102018</td>
      <td>-0.860922</td>
      <td>0.883321</td>
      <td>0.785301</td>
      <td>0.676805</td>
      <td>1.000000</td>
      <td>0.895600</td>
      <td>0.901507</td>
      <td>0.817939</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>-0.064247</td>
      <td>-0.400562</td>
      <td>0.143681</td>
      <td>-0.574598</td>
      <td>-0.989067</td>
      <td>0.351172</td>
      <td>-0.544724</td>
      <td>0.999639</td>
      <td>0.978735</td>
      <td>0.933636</td>
      <td>0.895600</td>
      <td>1.000000</td>
      <td>0.999909</td>
      <td>0.988477</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>-0.077675</td>
      <td>-0.412861</td>
      <td>0.130346</td>
      <td>-0.585564</td>
      <td>-0.990963</td>
      <td>0.338536</td>
      <td>-0.555964</td>
      <td>0.999186</td>
      <td>0.975885</td>
      <td>0.928729</td>
      <td>0.901507</td>
      <td>0.999909</td>
      <td>1.000000</td>
      <td>0.986349</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>0.087554</td>
      <td>-0.257247</td>
      <td>0.291829</td>
      <td>-0.444087</td>
      <td>-0.955347</td>
      <td>0.488858</td>
      <td>-0.411502</td>
      <td>0.992188</td>
      <td>0.998508</td>
      <td>0.977103</td>
      <td>0.817939</td>
      <td>0.988477</td>
      <td>0.986349</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    Cosine Similarity Matrix ignoring first singular value for k=4



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.952511</td>
      <td>0.927426</td>
      <td>0.949991</td>
      <td>0.600398</td>
      <td>0.957977</td>
      <td>0.958607</td>
      <td>-0.792342</td>
      <td>-0.619188</td>
      <td>0.401247</td>
      <td>-0.152142</td>
      <td>-0.699292</td>
      <td>-0.689470</td>
      <td>-0.687167</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.952511</td>
      <td>1.000000</td>
      <td>0.979118</td>
      <td>0.985011</td>
      <td>0.686898</td>
      <td>0.943633</td>
      <td>0.949524</td>
      <td>-0.929325</td>
      <td>-0.770642</td>
      <td>0.338148</td>
      <td>-0.076859</td>
      <td>-0.840224</td>
      <td>-0.834480</td>
      <td>-0.850561</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.927426</td>
      <td>0.979118</td>
      <td>1.000000</td>
      <td>0.964047</td>
      <td>0.615095</td>
      <td>0.933615</td>
      <td>0.881725</td>
      <td>-0.896924</td>
      <td>-0.748275</td>
      <td>0.355308</td>
      <td>-0.044969</td>
      <td>-0.766005</td>
      <td>-0.736131</td>
      <td>-0.771511</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.949991</td>
      <td>0.985011</td>
      <td>0.964047</td>
      <td>1.000000</td>
      <td>0.776727</td>
      <td>0.893334</td>
      <td>0.964140</td>
      <td>-0.935094</td>
      <td>-0.830149</td>
      <td>0.193120</td>
      <td>-0.245969</td>
      <td>-0.864644</td>
      <td>-0.805914</td>
      <td>-0.820088</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.600398</td>
      <td>0.686898</td>
      <td>0.615095</td>
      <td>0.776727</td>
      <td>1.000000</td>
      <td>0.428507</td>
      <td>0.771553</td>
      <td>-0.839107</td>
      <td>-0.940355</td>
      <td>-0.426393</td>
      <td>-0.654373</td>
      <td>-0.918491</td>
      <td>-0.754277</td>
      <td>-0.749785</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.957977</td>
      <td>0.943633</td>
      <td>0.933615</td>
      <td>0.893334</td>
      <td>0.428507</td>
      <td>1.000000</td>
      <td>0.881865</td>
      <td>-0.763050</td>
      <td>-0.517256</td>
      <td>0.611104</td>
      <td>0.133591</td>
      <td>-0.630926</td>
      <td>-0.696013</td>
      <td>-0.705159</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.958607</td>
      <td>0.949524</td>
      <td>0.881725</td>
      <td>0.964140</td>
      <td>0.771553</td>
      <td>0.881865</td>
      <td>1.000000</td>
      <td>-0.872983</td>
      <td>-0.746083</td>
      <td>0.217615</td>
      <td>-0.273820</td>
      <td>-0.846616</td>
      <td>-0.824180</td>
      <td>-0.809089</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>-0.792342</td>
      <td>-0.929325</td>
      <td>-0.896924</td>
      <td>-0.935094</td>
      <td>-0.839107</td>
      <td>-0.763050</td>
      <td>-0.872983</td>
      <td>1.000000</td>
      <td>0.932621</td>
      <td>-0.054413</td>
      <td>0.183163</td>
      <td>0.965789</td>
      <td>0.917141</td>
      <td>0.941963</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>-0.619188</td>
      <td>-0.770642</td>
      <td>-0.748275</td>
      <td>-0.830149</td>
      <td>-0.940355</td>
      <td>-0.517256</td>
      <td>-0.746083</td>
      <td>0.932621</td>
      <td>1.000000</td>
      <td>0.301769</td>
      <td>0.455950</td>
      <td>0.944274</td>
      <td>0.794027</td>
      <td>0.823833</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.401247</td>
      <td>0.338148</td>
      <td>0.355308</td>
      <td>0.193120</td>
      <td>-0.426393</td>
      <td>0.611104</td>
      <td>0.217615</td>
      <td>-0.054413</td>
      <td>0.301769</td>
      <td>1.000000</td>
      <td>0.777166</td>
      <td>0.112555</td>
      <td>-0.142947</td>
      <td>-0.143978</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>-0.152142</td>
      <td>-0.076859</td>
      <td>-0.044969</td>
      <td>-0.245969</td>
      <td>-0.654373</td>
      <td>0.133591</td>
      <td>-0.273820</td>
      <td>0.183163</td>
      <td>0.455950</td>
      <td>0.777166</td>
      <td>1.000000</td>
      <td>0.302955</td>
      <td>0.017012</td>
      <td>-0.005956</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>-0.699292</td>
      <td>-0.840224</td>
      <td>-0.766005</td>
      <td>-0.864644</td>
      <td>-0.918491</td>
      <td>-0.630926</td>
      <td>-0.846616</td>
      <td>0.965789</td>
      <td>0.944274</td>
      <td>0.112555</td>
      <td>0.302955</td>
      <td>1.000000</td>
      <td>0.944061</td>
      <td>0.949244</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>-0.689470</td>
      <td>-0.834480</td>
      <td>-0.736131</td>
      <td>-0.805914</td>
      <td>-0.754277</td>
      <td>-0.696013</td>
      <td>-0.824180</td>
      <td>0.917141</td>
      <td>0.794027</td>
      <td>-0.142947</td>
      <td>0.017012</td>
      <td>0.944061</td>
      <td>1.000000</td>
      <td>0.993956</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>-0.687167</td>
      <td>-0.850561</td>
      <td>-0.771511</td>
      <td>-0.820088</td>
      <td>-0.749785</td>
      <td>-0.705159</td>
      <td>-0.809089</td>
      <td>0.941963</td>
      <td>0.823833</td>
      <td>-0.143978</td>
      <td>-0.005956</td>
      <td>0.949244</td>
      <td>0.993956</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    Cosine Similarity Matrix ignoring first singular value for k=8



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
      <th>excellent</th>
      <th>amazing</th>
      <th>delicious</th>
      <th>fantastic</th>
      <th>gem</th>
      <th>perfectly</th>
      <th>incredible</th>
      <th>worst</th>
      <th>mediocre</th>
      <th>bland</th>
      <th>meh</th>
      <th>awful</th>
      <th>horrible</th>
      <th>terrible</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>1.000000</td>
      <td>0.825902</td>
      <td>0.804167</td>
      <td>0.959289</td>
      <td>0.764380</td>
      <td>0.426687</td>
      <td>0.830271</td>
      <td>-0.348296</td>
      <td>0.161682</td>
      <td>0.238291</td>
      <td>-0.032080</td>
      <td>-0.222790</td>
      <td>0.026182</td>
      <td>0.005882</td>
    </tr>
    <tr>
      <th>amazing</th>
      <td>0.825902</td>
      <td>1.000000</td>
      <td>0.823278</td>
      <td>0.900467</td>
      <td>0.614963</td>
      <td>0.115254</td>
      <td>0.786415</td>
      <td>-0.088094</td>
      <td>0.270989</td>
      <td>0.302473</td>
      <td>0.103273</td>
      <td>0.067038</td>
      <td>0.310632</td>
      <td>0.240466</td>
    </tr>
    <tr>
      <th>delicious</th>
      <td>0.804167</td>
      <td>0.823278</td>
      <td>1.000000</td>
      <td>0.884065</td>
      <td>0.566796</td>
      <td>0.563780</td>
      <td>0.855699</td>
      <td>-0.522318</td>
      <td>-0.145022</td>
      <td>0.279013</td>
      <td>0.036923</td>
      <td>-0.320334</td>
      <td>-0.140033</td>
      <td>-0.220939</td>
    </tr>
    <tr>
      <th>fantastic</th>
      <td>0.959289</td>
      <td>0.900467</td>
      <td>0.884065</td>
      <td>1.000000</td>
      <td>0.819723</td>
      <td>0.346943</td>
      <td>0.850093</td>
      <td>-0.401065</td>
      <td>0.092409</td>
      <td>0.186666</td>
      <td>0.003272</td>
      <td>-0.267488</td>
      <td>0.012609</td>
      <td>-0.026147</td>
    </tr>
    <tr>
      <th>gem</th>
      <td>0.764380</td>
      <td>0.614963</td>
      <td>0.566796</td>
      <td>0.819723</td>
      <td>1.000000</td>
      <td>0.171534</td>
      <td>0.691531</td>
      <td>-0.505794</td>
      <td>-0.067471</td>
      <td>-0.191903</td>
      <td>-0.204276</td>
      <td>-0.473570</td>
      <td>-0.167153</td>
      <td>-0.174568</td>
    </tr>
    <tr>
      <th>perfectly</th>
      <td>0.426687</td>
      <td>0.115254</td>
      <td>0.563780</td>
      <td>0.346943</td>
      <td>0.171534</td>
      <td>1.000000</td>
      <td>0.569491</td>
      <td>-0.767211</td>
      <td>-0.415155</td>
      <td>0.320437</td>
      <td>-0.033403</td>
      <td>-0.518119</td>
      <td>-0.674398</td>
      <td>-0.701069</td>
    </tr>
    <tr>
      <th>incredible</th>
      <td>0.830271</td>
      <td>0.786415</td>
      <td>0.855699</td>
      <td>0.850093</td>
      <td>0.691531</td>
      <td>0.569491</td>
      <td>1.000000</td>
      <td>-0.490503</td>
      <td>-0.199292</td>
      <td>0.127180</td>
      <td>-0.223340</td>
      <td>-0.359359</td>
      <td>-0.181406</td>
      <td>-0.263427</td>
    </tr>
    <tr>
      <th>worst</th>
      <td>-0.348296</td>
      <td>-0.088094</td>
      <td>-0.522318</td>
      <td>-0.401065</td>
      <td>-0.505794</td>
      <td>-0.767211</td>
      <td>-0.490503</td>
      <td>1.000000</td>
      <td>0.541594</td>
      <td>-0.030733</td>
      <td>0.021576</td>
      <td>0.785444</td>
      <td>0.853108</td>
      <td>0.861272</td>
    </tr>
    <tr>
      <th>mediocre</th>
      <td>0.161682</td>
      <td>0.270989</td>
      <td>-0.145022</td>
      <td>0.092409</td>
      <td>-0.067471</td>
      <td>-0.415155</td>
      <td>-0.199292</td>
      <td>0.541594</td>
      <td>1.000000</td>
      <td>0.624990</td>
      <td>0.675774</td>
      <td>0.798432</td>
      <td>0.677179</td>
      <td>0.765852</td>
    </tr>
    <tr>
      <th>bland</th>
      <td>0.238291</td>
      <td>0.302473</td>
      <td>0.279013</td>
      <td>0.186666</td>
      <td>-0.191903</td>
      <td>0.320437</td>
      <td>0.127180</td>
      <td>-0.030733</td>
      <td>0.624990</td>
      <td>1.000000</td>
      <td>0.839319</td>
      <td>0.488788</td>
      <td>0.119103</td>
      <td>0.166041</td>
    </tr>
    <tr>
      <th>meh</th>
      <td>-0.032080</td>
      <td>0.103273</td>
      <td>0.036923</td>
      <td>0.003272</td>
      <td>-0.204276</td>
      <td>-0.033403</td>
      <td>-0.223340</td>
      <td>0.021576</td>
      <td>0.675774</td>
      <td>0.839319</td>
      <td>1.000000</td>
      <td>0.477533</td>
      <td>0.120404</td>
      <td>0.218177</td>
    </tr>
    <tr>
      <th>awful</th>
      <td>-0.222790</td>
      <td>0.067038</td>
      <td>-0.320334</td>
      <td>-0.267488</td>
      <td>-0.473570</td>
      <td>-0.518119</td>
      <td>-0.359359</td>
      <td>0.785444</td>
      <td>0.798432</td>
      <td>0.488788</td>
      <td>0.477533</td>
      <td>1.000000</td>
      <td>0.835591</td>
      <td>0.854364</td>
    </tr>
    <tr>
      <th>horrible</th>
      <td>0.026182</td>
      <td>0.310632</td>
      <td>-0.140033</td>
      <td>0.012609</td>
      <td>-0.167153</td>
      <td>-0.674398</td>
      <td>-0.181406</td>
      <td>0.853108</td>
      <td>0.677179</td>
      <td>0.119103</td>
      <td>0.120404</td>
      <td>0.835591</td>
      <td>1.000000</td>
      <td>0.981254</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>0.005882</td>
      <td>0.240466</td>
      <td>-0.220939</td>
      <td>-0.026147</td>
      <td>-0.174568</td>
      <td>-0.701069</td>
      <td>-0.263427</td>
      <td>0.861272</td>
      <td>0.765852</td>
      <td>0.166041</td>
      <td>0.218177</td>
      <td>0.854364</td>
      <td>0.981254</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Observation
By ignoring the direction corresponding to the largest singular value, we do see some improvement. Difference between cosine similarities seem more distinict with smaller $k$. 
