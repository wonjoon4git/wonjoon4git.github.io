---
layout: single
classes: wide
title:  "Semantic Role Labelling with BERT"
categories: 
  - NLP
tag: [BERT, Semantic Role Labelling]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---

# Semantic Role Labelling with BERT

In this post, we are going to train and evaluate a PropBank-style semantic role labeling (SRL) system. Following (Collobert et al. 2011) and others, we will treat this problem as a sequence-labeling task. For each input token, the system will predict a B-I-O tag, as illustrated in the following example:

|The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
|---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|             
|B-ARG1|I-ARG1|B-V|B-ARG2|I-ARG2|I-ARG2|I-ARG2|I-ARG2|O|O|O|O|O|O|O|
|||schedule.01|||||||||||||

Note that the same sentence may have multiple annotations for different predicates

|The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
|---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|             
|B-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|I-ARG1|O|B-V|B-ARG2|I-ARG2|I-ARG2|B-ARGM-TMP|O|
||||||||||remove.01||||||

and not all predicates need to be verbs

|The|judge|scheduled|to|preside|over|his|trial|was|removed|from|the|case|today|.|             
|---|-----|---------|--|-------|----|---|-----|---|-------|----|---|----|-----|-|    
|O|O|O|O|O|O|B-ARG1|B-V|O|O|O|O|O|O|O|
||||||||try.02||||||||

The SRL system will be implemented in [PyTorch](https://pytorch.org/). We will use BERT (in the implementation provided by the [Huggingface transformers](https://huggingface.co/docs/transformers/index) library) to compute contextualized token representations and a custom classification head to predict semantic roles. We will fine-tune the pretrained BERT model on the SRL task.


### Overview of the Approach

The model we will train is straightforward. Essentially, we  just encode the sentence with BERT, then take the contextualized embedding for each token and feed it into a classifier to predict the corresponding tag.

Because we are only working on argument identification and labeling (not predicate identification), it is essentially that we tell the model where the predicate is. This can be accomplished in various ways. The approach we will choose here repurposes Bert's *segment embeddings*.

Recall that BERT is trained on two input sentences, seperated by [SEP], and on a next-sentence-prediction objective (in addition to the masked LM objective). To help BERT comprehend which sentence a given token belongs to, the original BERT uses a segment embedding, using A for the first sentene, and B for the second sentence 2.
Because we are labeling only a single sentence at a time, we can use the segment embeddings to indicate the predicate position instead: The predicate is labeled as segment B (1) and all other tokens will be labeled as segment A (0).

<img src="https://github.com/daniel-bauer/4705-f23-hw5/blob/main/bert_srl_model.png?raw=true" width=400px>

## Dataset: Ontonotes 5.0 English SRL annotations

We will work with the English part of the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) data. This is an extension of PropBank, using the same type of annotation. Ontonotes contains annotations other than predicate/argument structures, but we will use the PropBank style SRL annotations only.


```python
# ontonotes_srl.zip Download link is private. Please look for the data elsewhere
```


```python
! unzip ontonotes_srl.zip
```

    Archive:  ontonotes_srl.zip
      inflating: propbank_dev.tsv        
      inflating: propbank_test.tsv       
      inflating: propbank_train.tsv      
      inflating: role_list.txt           


The data has been pre-processed in the following format. There are three files:

`propbank_dev.tsv`	`propbank_test.tsv`	`propbank_train.tsv`

Each of these files is in a tab-separated value format. A single predicate/argument structure annotation consists of four rows. For example

```
ontonotes/bc/cnn/00/cnn_0000.152.1
The     judge   scheduled       to      preside over    his     trial   was     removed from    the     case    today   /.
                schedule.01
B-ARG1  I-ARG1  B-V     B-ARG2  I-ARG2  I-ARG2  I-ARG2  I-ARG2  O       O       O       O       O       O       O
```

* The first row is a unique identifier (1st annotation of the 152nd sentence in the file ontonotes/bc/cnn/00/cnn_0000).
* The second row contains the tokens of the sentence (tab-separated).
* The third row contains the probank frame name for the predicate (empty field for all other tokens).
* The fourth row contains the B-I-O tag for each token.

The file `rolelist.txt` contains a list of propbank BIO labels in the dataset (i.e. possible output tokens). This list has been filtered to contain only roles that appeared more than 1000 times in the training data.
We will load this list and create mappings from numeric ids to BIO tags and back.


```python
role_to_id = {}
with open("role_list.txt",'r') as f:
    role_list = [x.strip() for x in f.readlines()]
    role_to_id = dict((role, index) for (index, role) in enumerate(role_list))
    role_to_id['[PAD]'] = -100
    id_to_role = dict((index, role) for (role, index) in role_to_id.items())
```

Note that we are also mapping the '[PAD]' token to the value -100. This allows the loss function to ignore these tokens during training.

## Part 1 - Data Preparation

Before we build the SRL model, we first need to preprocess the data.


### 1.1 - Tokenization

One challenge is that the pre-trained BERT model uses subword ("WordPiece") tokenization, but the Ontonotes data does not. Fortunately Huggingface transformers provides a tokenizer.


```python
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer.tokenize("This is an unbelievably boring test sentence.")
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]





    ['this',
     'is',
     'an',
     'un',
     '##bel',
     '##ie',
     '##va',
     '##bly',
     'boring',
     'test',
     'sentence',
     '.']



**TODO**:
We need to be able to maintain the correct labels (B-I-O tags) for each of the subwords.
Complete the following function that takes a list of tokens and a list of B-I-O labels of the same length as parameters, and returns a new token / label pair, as illustrated in the following example.


```
>>> tokenize_with_labels("the fancyful penguin devoured yummy fish .".split(), "B-ARG0 I-ARG0 I-ARG0 B-V B-ARG1 I-ARG1 O".split(), tokenizer)
(['the',
  'fancy',
  '##ful',
  'penguin',
  'dev',
  '##oured',
  'yu',
  '##mmy',
  'fish',
  '.'],
 ['B-ARG0',
  'I-ARG0',
  'I-ARG0',
  'I-ARG0',
  'B-V',
  'I-V',
  'B-ARG1',
  'I-ARG1',
  'I-ARG1',
  'O'])

```

To approach this problem, iterate through each word/label pair in the sentence. Call the tokenizer on the word. This may result in one or more tokens. Create the correct number of labels to match the number of tokens. Take care to not generate multiple B- tokens.


This approach is a bit slower than tokenizing the entire sentence, but is necessary to produce proper input tokenization for the pre-trained BERT model, and the matching target labels.


```python
def tokenize_with_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces.
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        subwords = tokenizer.tokenize(word)

        if not subwords:
            continue
        elif label.startswith('B-'):
            # B-label for the first subword, and I-labels for any remaining subwords
            new_labels = [label] + [f'I{label[1:]}' for _ in subwords[1:]]
        else:
            # same label for all subwords
            new_labels = [label for _ in subwords]

        tokenized_sentence.extend(subwords)
        labels.extend(new_labels)

    return tokenized_sentence, labels
```


```python
tokenize_with_labels("the fancyful penguin devoured yummy fish .".split(), "B-ARG0 I-ARG0 I-ARG0 B-V B-ARG1 I-ARG1 O".split(), tokenizer)
```




    (['the',
      'fancy',
      '##ful',
      'penguin',
      'dev',
      '##oured',
      'yu',
      '##mmy',
      'fish',
      '.'],
     ['B-ARG0',
      'I-ARG0',
      'I-ARG0',
      'I-ARG0',
      'B-V',
      'I-V',
      'B-ARG1',
      'I-ARG1',
      'I-ARG1',
      'O'])



### 1.2 Loading the Dataset

Next, we are creating a PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class. This class acts as a contained for the training, development, and testing data in memory. 

1.2.1 **TODO**: Write the \_\_init\_\_(self, filename) method that reads in the data from a data file (specified by the filename).

For each annotation we start with the tokens in the sentence, and the BIO tags. Then we need to create the following

1. call the `tokenize_with_labels` function to tokenize the sentence.
2. Add the (token, label) pair to the self.items list.

1.2.2 Write the \_\_len\_\_(self) method that returns the total number of items.

1.2.3 Write the \_\_getitem\_\_(self, k) method that returns a single item in a format BERT will understand.
* We need to process the sentence by adding "\[CLS\]" as the first token and "\[SEP\]" as the last token. The need to pad the token sequence to 128 tokens using the "\[PAD\]" symbol. This needs to happen both for the inputs (sentence token sequence) and outputs (BIO tag sequence).
* We need to create an *attention mask*, which is a sequence of 128 tokens indicating the actual input symbols (as a 1) and \[PAD\] symbols (as a 0).
* We need to create a *predicate indicator* mask, which is a sequence of 128 tokens with at most one 1, in the position of the "B-V" tag. All other entries should be 0. The model will use this information to understand where the predicate is located.

* Finally, we need to convert the token and tag sequence into numeric indices. For the tokens, this can be done using the `tokenizer.convert_tokens_to_ids` method. For the tags, use the `role_to_id` dictionary.
Each sequence must be a pytorch tensor of shape (1,128). We can convert a list of integer values like this `torch.tensor(token_ids, dtype=torch.long)`.

To keep everything organized, we will return a dictionary in the following format

```
{'ids': token_tensor,
 'targets': tag_tensor,
 'mask': attention_mask_tensor,
 'pred': predicate_indicator_tensor}
```


```python
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class SrlData(Dataset):

    def __init__(self, filename):
        super(SrlData, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.items = []
        self.max_len = 128

        with open(filename, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            # Read four lines for each data entry
            sentence_id = lines[i].strip()
            sentence_tokens = lines[i+1].strip().split()
            predicate = lines[i+2].strip()
            labels = lines[i+3].strip().split()

            # Tokenize sentence and labels
            tokenized_sentence, tokenized_labels = tokenize_with_labels(sentence_tokens, labels, self.tokenizer)
            self.items.append((tokenized_sentence, tokenized_labels))

            # Move to the next set of four lines
            i += 4

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        tokens, labels = self.items[k]

        # Adjust tokens for special tokens and padding
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        pad_length = self.max_len - len(tokens)
        tokens.extend(['[PAD]'] * pad_length)

        # Convert tokens to IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Adjust labels for special tokens and padding
        tag_ids = [role_to_id['[CLS]']] + [role_to_id.get(label, role_to_id['O']) for label in labels] + [role_to_id['[SEP]']]
        tag_ids.extend([role_to_id['[PAD]']] * pad_length)

        # Predicate indicator mask
        pred_mask = [1 if label == 'B-V' else 0 for label in labels]
        pred_mask = [0] + pred_mask + [0] + [0] * pad_length

        # Create attention mask
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]

        # Convert to tensors with single examples (batch size 1)
        token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        label_tensor = torch.tensor(tag_ids, dtype=torch.long).unsqueeze(0)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        predicate_indicator_tensor = torch.tensor(pred_mask, dtype=torch.long).unsqueeze(0)

        return {
            'ids': token_tensor,
            'targets': label_tensor,
            'mask': attention_mask_tensor,
            'pred': predicate_indicator_tensor
        }
```


```python
# Reading the training data takes a while for the entire data because we preprocess all data offline
data = SrlData("propbank_train.tsv")
```

## 2. Model Definition


```python
from torch.nn import Module, Linear, CrossEntropyLoss
from transformers import BertModel
```

We will define the pyTorch model as a subclass of the [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class. The code for the model is provided. It may help to take a look at the documentation to remind of how Module works. Take a look at how the huggingface BERT model simply becomes another sub-module.


```python
class SrlModel(Module):

    def __init__(self):

        super(SrlModel, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        # The following two lines would freeze the BERT parameters and allow us to train the classifier by itself.
        # We are fine-tuning the model, so we can leave this commented out!
        # for param in self.encoder.parameters():
        #    param.requires_grad = False

        # The linear classifier head, see model figure in the introduction.
        self.classifier = Linear(768, len(role_to_id))


    def forward(self, input_ids, attn_mask, pred_indicator):

        # This defines the flow of data through the model

        # Note the use of the "token type ids" which represents the segment encoding explained in the introduction.
        # In our segment encoding, 1 indicates the predicate, and 0 indicates everything else.
        bert_output =  self.encoder(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=pred_indicator)

        enc_tokens = bert_output[0] # the result of encoding the input with BERT
        logits = self.classifier(enc_tokens) #feed into the classification layer to produce scores for each tag.

        # Note that we are only interested in the argmax for each token, so we do not have to normalize
        # to a probability distribution using softmax. The CrossEntropyLoss loss function takes this into account.
        # It essentially computes the softmax first and then computes the negative log-likelihood for the target classes.
        return logits
```


```python
model = SrlModel().to('cuda') # create new model and store weights in GPU memory
```


    model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]


Now we are ready to try running the model with just a single input example to check if it is working correctly. Clearly it has not been trained, so the output is not what we expect. But we can see what the loss looks like for an initial sanity check.

* Take a single data item from the dev set, as provided by our Dataset class defined above. Obtain the input token ids, attention mask, predicate indicator mask, and target labels.
* Run the model on the ids, attention mask, and predicate mask like this:


```python
# pick an item from the dataset. Then run
ids, targets, mask, pred = data[1].values()
ids = ids.cuda()
targets = targets.cuda()
mask = mask.cuda()
pred = pred.cuda()

outputs = model(ids, mask, pred)
outputs
```




    tensor([[[ 0.2430,  0.2473, -0.8754,  ..., -0.1716,  0.1206, -0.3380],
             [ 0.3486,  0.0150, -0.5613,  ...,  0.2095,  0.2367,  0.3086],
             [ 0.4523, -0.3314, -0.1239,  ...,  0.0796,  0.4871, -0.1380],
             ...,
             [ 0.1238, -0.0772, -0.1371,  ..., -0.0722, -0.0473,  0.2705],
             [ 0.2100, -0.1111, -0.0955,  ..., -0.0011, -0.0905,  0.1999],
             [ 0.1812, -0.1591, -0.1639,  ...,  0.0151,  0.1221,  0.3297]]],
           device='cuda:0', grad_fn=<ViewBackward0>)




Compute the loss on this one item only.
The initial loss should be close to -ln(1/num_labels)

Without training we would assume that all labels for each token (including the target label) are equally likely, so the negative log probability for the targets should be approximately $$-\ln\left(\frac{1}{\text{num\_labels}}\right)$$
This is what the loss function should return on a single example. This is a good sanity check to run for any multi-class prediction problem.


```python
import math
-math.log(1 / len(role_to_id), math.e)
```




    3.970291913552122




```python
loss_function = CrossEntropyLoss(ignore_index = -100, reduction='mean')

loss = loss_function(outputs.transpose(1, 2), targets)

# Print or retrieve the loss value
print("Loss value:", loss.item())
```

    Loss value: 4.0529255867004395


At this point we should also obtain the actual predictions by taking the argmax over each position.
The result should look something like this (values will differ).

```
tensor([[ 1,  4,  4,  4,  4,  4,  5, 29, 29, 29,  4, 28,  6, 32, 32, 32, 32, 32,
         32, 32, 30, 30, 32, 30, 32,  4, 32, 32, 30,  4, 49,  4, 49, 32, 30,  4,
         32,  4, 32, 32,  4,  2,  4,  4, 32,  4, 32, 32, 32, 32, 30, 32, 32, 30,
         32,  4,  4, 49,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  6,  6, 32, 32,
         30, 32, 32, 32, 32, 32, 30, 30, 30, 32, 30, 49, 49, 32, 32, 30,  4,  4,
          4,  4, 29,  4,  4,  4,  4,  4,  4, 32,  4,  4,  4, 32,  4, 30,  4, 32,
         30,  4, 32,  4,  4,  4,  4,  4, 32,  4,  4,  4,  4,  4,  4,  4,  4,  4,
          4,  4]], device='cuda:0')
```

Then use the id_to_role dictionary to decode to actual tokens.

```
['[CLS]', 'O', 'O', 'O', 'O', 'O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'O', 'B-V', 'B-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG1', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'O', 'I-ARGM-TMP', 'O', 'I-ARGM-TMP', 'I-ARG2', 'I-ARG1', 'O', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'O', '[SEP]', 'O', 'O', 'I-ARG2', 'O', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'O', 'O', 'I-ARGM-TMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARG1', 'B-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG2', 'I-ARG1', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARG2', 'I-ARG2', 'I-ARG1', 'O', 'O', 'O', 'O', 'I-ARG0', 'O', 'O', 'O', 'O', 'O', 'O', 'I-ARG2', 'O', 'O', 'O', 'I-ARG2', 'O', 'I-ARG1', 'O', 'I-ARG2', 'I-ARG1', 'O', 'I-ARG2', 'O', 'O', 'O', 'O', 'O', 'I-ARG2', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
```

Later, we will write a more formal function to do this once we have trained the model.


```python
predicted_classes = torch.argmax(outputs, dim=2)

predicted_classes
```




    tensor([[23, 13, 27, 51, 27, 51, 51,  7, 28,  9, 20, 14, 34, 23, 33, 33, 22, 22,
             34, 25, 25, 22,  9,  9,  3, 34, 50,  9, 34, 50, 50, 22,  3,  3, 20,  9,
              3,  3,  3, 34, 34, 34,  7,  7, 25, 22, 22, 22,  9,  9, 22, 14, 14, 22,
             22, 22, 50, 23, 22, 22, 23,  3, 20,  9,  3, 20, 14, 34, 34, 17, 34, 25,
             22, 34, 25, 22, 22, 14,  3, 17, 34, 14, 22, 17, 17, 25, 23, 22,  3, 23,
              3, 23,  9,  9,  3,  3, 34, 34,  7, 25, 34, 22, 25, 25, 25, 22, 14,  9,
             25, 34, 34, 25, 14, 17, 25, 25, 22, 20, 14,  3,  3,  3, 14, 34, 17, 34,
             34, 34]], device='cuda:0')




```python
decoded_labels = [[id_to_role[idx.item()] for idx in row] for row in predicted_classes]

decoded_labels
```




    [['B-ARGM-PRP',
      'B-ARGM-CAU',
      'B-ARGM-LVB',
      'I-ARGM-LVB',
      'B-ARGM-LVB',
      'I-ARGM-LVB',
      'I-ARGM-LVB',
      'B-ARG1-DSP',
      'B-V',
      'B-ARG3',
      'B-ARGM-MOD',
      'B-ARGM-DIR',
      'I-ARG4',
      'B-ARGM-PRP',
      'I-ARG3',
      'I-ARG3',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'I-ARG4',
      'B-ARGM-TMP',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'B-ARG3',
      'B-ARG3',
      '[prd]',
      'I-ARG4',
      'I-ARGM-CXN',
      'B-ARG3',
      'I-ARG4',
      'I-ARGM-CXN',
      'I-ARGM-CXN',
      'B-ARGM-PRD',
      '[prd]',
      '[prd]',
      'B-ARGM-MOD',
      'B-ARG3',
      '[prd]',
      '[prd]',
      '[prd]',
      'I-ARG4',
      'I-ARG4',
      'I-ARG4',
      'B-ARG1-DSP',
      'B-ARG1-DSP',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'B-ARG3',
      'B-ARG3',
      'B-ARGM-PRD',
      'B-ARGM-DIR',
      'B-ARGM-DIR',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'I-ARGM-CXN',
      'B-ARGM-PRP',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'B-ARGM-PRP',
      '[prd]',
      'B-ARGM-MOD',
      'B-ARG3',
      '[prd]',
      'B-ARGM-MOD',
      'B-ARGM-DIR',
      'I-ARG4',
      'I-ARG4',
      'B-ARGM-GOL',
      'I-ARG4',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'I-ARG4',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'B-ARGM-PRD',
      'B-ARGM-DIR',
      '[prd]',
      'B-ARGM-GOL',
      'I-ARG4',
      'B-ARGM-DIR',
      'B-ARGM-PRD',
      'B-ARGM-GOL',
      'B-ARGM-GOL',
      'B-ARGM-TMP',
      'B-ARGM-PRP',
      'B-ARGM-PRD',
      '[prd]',
      'B-ARGM-PRP',
      '[prd]',
      'B-ARGM-PRP',
      'B-ARG3',
      'B-ARG3',
      '[prd]',
      '[prd]',
      'I-ARG4',
      'I-ARG4',
      'B-ARG1-DSP',
      'B-ARGM-TMP',
      'I-ARG4',
      'B-ARGM-PRD',
      'B-ARGM-TMP',
      'B-ARGM-TMP',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'B-ARGM-DIR',
      'B-ARG3',
      'B-ARGM-TMP',
      'I-ARG4',
      'I-ARG4',
      'B-ARGM-TMP',
      'B-ARGM-DIR',
      'B-ARGM-GOL',
      'B-ARGM-TMP',
      'B-ARGM-TMP',
      'B-ARGM-PRD',
      'B-ARGM-MOD',
      'B-ARGM-DIR',
      '[prd]',
      '[prd]',
      '[prd]',
      'B-ARGM-DIR',
      'I-ARG4',
      'B-ARGM-GOL',
      'I-ARG4',
      'I-ARG4',
      'I-ARG4']]



## 3. Training loop

pytorch provides a DataLoader class that can be wrapped around a Dataset to easily use the dataset for training. The DataLoader allows us to easily adjust the batch size and shuffle the data.


```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# I was getting dimension errors, so here I define custom collate function
def custom_collate_fn(batch):
    # Separate the data into different lists
    ids_list, targets_list, mask_list, pred_list = [], [], [], []
    for item in batch:
        ids_list.append(item['ids'].squeeze(0))
        targets_list.append(item['targets'].squeeze(0))
        mask_list.append(item['mask'].squeeze(0))
        pred_list.append(item['pred'].squeeze(0))

    # Pad the sequences
    ids_padded = pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=role_to_id['[PAD]'])
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0)
    pred_padded = pad_sequence(pred_list, batch_first=True, padding_value=0)

    return {'ids': ids_padded, 'targets': targets_padded, 'mask': mask_padded, 'pred': pred_padded}

# Usig the custom collate function in DataLoader
loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
```

The following cell contains the main training loop. The code should work as written and report the loss after each batch,
cumulative average loss after each 100 batches, and print out the final average loss after the epoch.

Modify the training loop belowso that it also computes the accuracy for each batch and reports the
average accuracy after the epoch.
The accuracy is the number of correctly predicted token labels out of the number of total predictions.
Make sure to exclude [PAD] tokens, i.e. tokens for which the target label is -100. It's okay to include [CLS] and [SEP] in the accuracy calculation.


```python
loss_function = CrossEntropyLoss(ignore_index = -100, reduction='mean')

LEARNING_RATE = 1e-05
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

def train():
    # Set the model to training mode
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        # Move batch data to GPU
        ids = batch['ids'].cuda()
        targets = batch['targets'].cuda()
        mask = batch['mask'].cuda()
        pred_mask = batch['pred'].cuda()

        logits = model(input_ids=ids, attn_mask=mask, pred_indicator=pred_mask)
        loss = loss_function(logits.transpose(2, 1), targets)
        total_loss += loss.item()

        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Backpropagation, compute gradients 
        optimizer.step() # Update model parameters

        # Update metrics
        predictions = logits.argmax(-1)
        valid_tokens = (targets != -100)
        correct_predictions = (predictions == targets) & valid_tokens
        total_correct += correct_predictions.sum().item()
        total_tokens += valid_tokens.sum().item()

    # Calculate metrics
    average_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
```

Now let's train the model for one epoch. This will take a while (up to a few hours).


```python
train()
torch.save(model.state_dict(), "srl_model_fulltrain_1epoch_finetune_1e-05.pt")
```

    Average Loss: 0.3972, Accuracy: 0.8896



```python
train()
```

    Average Loss: 0.1962, Accuracy: 0.9392


At this point, it's a good idea to save the model (or rather the parameter dictionary) so we can continue evaluating the model without having to retrain.


```python
torch.save(model.state_dict(), "srl_model_fulltrain_2epoch_finetune_1e-05.pt")
```

## 4. Decoding


```python
# Optional step: If we stopped working after part 3, first load the trained model

model = SrlModel().to('cuda')
model.load_state_dict(torch.load("srl_model_fulltrain_2epoch_finetune_1e-05.pt"))
model = model.to('cuda')
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


Now that we have a trained model, let's try labeling an unseen example sentence. Complete the functions decode_output and label_sentence below. decode_output takes the logits returned by the model, extracts the argmax to obtain the label predictions for each token, and then translate the result into a list of string labels.

label_sentence takes a list of input tokens and a predicate index, prepares the model input, call the model and then call decode_output to produce a final result.

Note that we have already implemented all components necessary (preparing the input data from the token list and predicate index, decoding the model output). But now we are putting it together in one convenient function.


```python
tokens = "A U. N. team spent an hour inside the hospital , where it found evident signs of shelling and gunfire .".split()
```


```python
def decode_output(logits):
    """
    Given the model output logits, return a list of string labels for each token.
    """

    # Get the most likely labels
    label_indices = torch.argmax(logits, dim=-1)
    # Convert numeric labels to strings (assuming label_map is a dict mapping indices to string labels)
    decoded_labels = [[id_to_role[idx.item()] for idx in row] for row in label_indices]

    return decoded_labels
```


```python
def label_sentence(tokens, pred_idx):
    """
    Prepare input data, predict labels, and decode the output.
    """
    # Tokenize the sentence and add special tokens
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Prepare attention mask
    attention_mask = [1] * len(token_ids)

    # Prepare predicate mask
    predicate_mask = [0] * len(token_ids)
    predicate_mask[pred_idx] = 1

    # Convert to tensors and move to gpu
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).cuda()
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).cuda()
    predicate_mask_tensor = torch.tensor([predicate_mask], dtype=torch.long).cuda()

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids=token_ids_tensor, attn_mask=attention_mask_tensor, pred_indicator=predicate_mask_tensor)

    # Decode logits to labels
    labels = decode_output(logits)
    return labels[0]
```


```python
label_test = label_sentence(tokens, 13) # Predicate is "found"
for token, label in zip(tokens, label_test):
    print(f"('{token}', '{label}'),")
```

    ('A', 'O'),
    ('U.', 'O'),
    ('N.', 'O'),
    ('team', 'O'),
    ('spent', 'O'),
    ('an', 'O'),
    ('hour', 'O'),
    ('inside', 'O'),
    ('the', 'O'),
    ('hospital', 'O'),
    (',', 'O'),
    ('where', 'O'),
    ('it', 'O'),
    ('found', 'O'),
    ('evident', 'O'),
    ('signs', 'I-ARG1'),
    ('of', 'I-ARG1'),
    ('shelling', 'I-ARG1'),
    ('and', 'I-ARG1'),
    ('gunfire', 'I-ARG1'),
    ('.', 'O'),


The expected output is somethign like this:
```   
 ('A', 'O'),
 ('U.', 'O'),
 ('N.', 'O'),
 ('team', 'O'),
 ('spent', 'O'),
 ('an', 'O'),
 ('hour', 'O'),
 ('inside', 'O'),
 ('the', 'B-ARGM-LOC'),
 ('hospital', 'I-ARGM-LOC'),
 (',', 'O'),
 ('where', 'B-ARGM-LOC'),
 ('it', 'B-ARG0'),
 ('found', 'B-V'),
 ('evident', 'B-ARG1'),
 ('signs', 'I-ARG1'),
 ('of', 'I-ARG1'),
 ('shelling', 'I-ARG1'),
 ('and', 'I-ARG1'),
 ('gunfire', 'I-ARG1'),
 ('.', 'O'),
```


### 5. Evaluation 1: Token-Based Accuracy
We want to evaluate the model on the dev or test set.


```python
dev_data = SrlData("propbank_dev.tsv") # Takes a while because we preprocess all data offline
```


```python
from torch.utils.data import DataLoader
loader = DataLoader(dev_data, batch_size = 1, shuffle = False, collate_fn=custom_collate_fn)
```

We write the evaluate_token_accuracy function below. The function should iterate through the items in the data loader (see training loop in part 3). Run the model on each sentence/predicate pair and extract the predictions.

For each sentence, count the correct predictions and the total predictions. Finally, compute the accuracy as #correct_predictions / #total_predictions

Note: We need to filter out the padded positions ([PAD] target tokens), as well as [CLS] and [SEP]. It's okay to include [B-V] in the count though.


```python
def evaluate_token_accuracy(model, loader):
    model.eval()  # Set the model to evaluation mode

    total_correct = 0
    total_tokens = 0  # Total non-pad tokens counted

    for batch in loader:
        ids = batch['ids'].cuda()
        targets = batch['targets'].cuda()
        mask = batch['mask'].cuda()
        pred_mask = batch['pred'].cuda()

        # No gradient needed for evaluation
        with torch.no_grad():
            logits = model(input_ids=ids, attn_mask=mask, pred_indicator=pred_mask)
            predictions = logits.argmax(-1)  # Get the most probable labels

        # Calculate correct predictions
        valid_tokens = (targets != role_to_id['[PAD]'])  
        correct_predictions = (predictions == targets) & valid_tokens

        # Update totals
        total_correct += correct_predictions.sum().item()
        total_tokens += valid_tokens.sum().item()

    # Compute accuracy
    accuracy = total_correct / total_tokens
    print(f"Accuracy: {accuracy:.4f}")

```


```python
evaluate_token_accuracy(model, loader)
```

    Accuracy: 0.9360


### 6. Span-Based evaluation

While the accuracy score in part 5 is encouraging, an accuracy-based evaluation is problematic for two reasons. First, most of the target labels are actually O. Second, it only tells us that per-token prediction works, but does not directly evaluate the SRL performance.

Instead, SRL systems are typically evaluated on micro-averaged precision, recall, and F1-score for predicting labeled spans.

More specifically, for each sentence/predicate input, we run the model, decode the output, and extract a set of labeled spans (from the output and the target labels). These spans are (i,j,label) tuples.  

We then compute the true_positives, false_positives, and false_negatives based on these spans.

In the end, we can compute

* Precision:  true_positive / (true_positives + false_positives)  , that is the number of correct spans out of all predicted spans.

* Recall: true_positives / (true_positives + false_negatives) , that is the number of correct spans out of all target spans.

* F1-score:   (2 * precision * recall) / (precision + recall)


```python
def extract_spans(labels):
    spans = {} # map (start,end) ids to label
    current_span_start = 0
    current_span_type = ""
    inside = False
    for i, label in enumerate(labels):
        if label.startswith("B"):
            if inside:
                if current_span_type != "V":
                    spans[(current_span_start,i)] = current_span_type
            current_span_start = i
            current_span_type = label[2:]
            inside = True
        elif inside and label.startswith("O"):
            if current_span_type != "V":
                spans[(current_span_start,i)] = current_span_type
            inside = False
        elif inside and label.startswith("I") and label[2:] != current_span_type:
            if current_span_type != "V":
                spans[(current_span_start,i)] = current_span_type
            inside = False
    return spans

```


```python
def evaluate_spans(model, loader):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in loader:
        ids = batch['ids'].cuda()
        targets = batch['targets'].cuda()
        mask = batch['mask'].cuda()
        pred_mask = batch['pred'].cuda()

        with torch.no_grad():
            logits = model(input_ids=ids, attn_mask=mask, pred_indicator=pred_mask)
            predictions = logits.argmax(-1)  # Get the most probable labels

        # For each item in the batch
        for i in range(ids.size(0)):
            pred_labels = [id_to_role[pred] for pred in predictions[i][mask[i] == 1].tolist()]
            true_labels = [id_to_role[true] for true in targets[i][mask[i] == 1].tolist()]

            pred_spans = extract_spans(pred_labels)
            true_spans = extract_spans(true_labels)

            pred_span_set = set(pred_spans.items())
            true_span_set = set(true_spans.items())

            tp = len(pred_span_set & true_span_set)
            fp = len(pred_span_set - true_span_set)
            fn = len(true_span_set - pred_span_set)

            # Update metrics
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate metrics
    total_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    total_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    total_f = (2 * total_p * total_r) / (total_p + total_r) if total_p + total_r > 0 else 0

    print(f"Overall P: {total_p:.4f}  Overall R: {total_r:.4f}  Overall F1: {total_f:.4f}")
```


```python
evaluate_spans(model, loader)
```

    Overall P: 0.8112  Overall R: 0.8312  Overall F1: 0.8211


Note: F score of 0.82 is slightly below the state-of-the art performance (in 2018)
