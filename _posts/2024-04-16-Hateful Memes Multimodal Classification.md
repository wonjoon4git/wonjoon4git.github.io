---
layout: single
classes: wide
title:  "Hateful Memes Multimodal Classification"
categories: 
  - Multimodal
tag: [Classification, Multimodal AI]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---


# Fun with Multimodal Models

In this post, we are going to construct two different multimodal models using tools from the Huggingface library as well as pre-trained models from PyTorch's `torchvision`.

The goal of a multimodal model is to take inputs from two or more modalities, and jointly inference on them to make a prediction.  Here, we will be using the Hateful Memes (https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/) dataset, which has the goal of classifying a pair of an image with an associated caption as either Hateful or not (binary classification).

This is considered one of the harder tasks, even the original meta paper showed around 60% accuracy




# Preperation

## Import & Install


```python
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore') # ignore uncessary warnings
```


```python
# Install Huggingface locally (may not be necessary)
#!pip install transformers

# Download the Hateful memes dataset from Google Drive.  We need gdown
# for this (may not be necessary to pip install)
#!pip install gdown
```


```python
#!gdown "https://drive.google.com/uc?id=1NH3DTDuNVsLInGQaLug7w49AHOnWJaKQ"
# !unzip "hateful_memes.zip" -d .
```


```python
!ls data
```

    dev.jsonl  img	LICENSE.txt  README.md	test.jsonl  train.jsonl


## Prepare Data

### First Look of .jsonl file


```python
with open('data/train.jsonl', 'r') as file:
    for i, line in enumerate(file):
        if i >= 3:
            break
        json_obj = json.loads(line)
        print(json_obj)
```

    {'id': 42953, 'img': 'img/42953.png', 'label': 0, 'text': 'its their character not their color that matters'}
    {'id': 23058, 'img': 'img/23058.png', 'label': 0, 'text': "don't be afraid to love again everyone is not like your ex"}
    {'id': 13894, 'img': 'img/13894.png', 'label': 0, 'text': 'putting bows on your pet'}


### Creating Dataset & Data Loader


```python
class HatefulMemesDataset(Dataset):
    """
    A custom dataset class for the Hateful Memes dataset, which loads images and their
    corresponding labels and text from a given directory and JSONL file.
    """
    def __init__(self, jsonl_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # Load data from the JSONL file
        with open(jsonl_file, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Correcting the image path and loading the image
        img_subpath = item['img'].lstrip('/').replace('img/', '')  # remove duplicated 'img/' path
        img_path = os.path.join(self.img_dir, img_subpath) 
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        image = self.transform(image)

        # Extracting text and label
        text = item['text']
        label = item.get('label')

        return {'image': image, 'caption': text, 'label': torch.tensor(label)}
        
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch]) 
    captions = [item['caption'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return {'image': images, 'caption': captions, 'label': labels}

# Image Transformation definition
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize image to 224x224
    transforms.ToTensor(),          # image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) # normalize for ResNet
])
```

### Test Dataloader by Visualizing
Images appear a bit weird due to normalization for ResNet


```python
# Load Dataset
train_dataset = HatefulMemesDataset(jsonl_file='data/train.jsonl', img_dir='data/img', 
                                    transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)

# prepare Dataloader 
dataloader_iter = iter(train_dataloader)
next(dataloader_iter)
next(dataloader_iter)
next(dataloader_iter)
batch = next(dataloader_iter)

# Show iamges
images, captions, labels = batch['image'], batch['caption'], batch['label']
indices = np.random.choice(len(images), size=4, replace=False)
plt.figure(figsize=(12, 12))
for i, idx in enumerate(indices):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[idx].permute(1, 2, 0))
    plt.title(f'Label: {labels[idx].item()}')
    plt.xticks([])
    plt.yticks([])
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](/assets/images/HatefulMeMe/output_13_1.png)
    


### Baseline (Random) Accuracy
After multiple trials of the following code, our baseline(random) accuracy is around 0.5

This also means we have almost a balanced dataset


```python
# Load dataset
val_dataset = HatefulMemesDataset(jsonl_file='data/dev.jsonl', img_dir='data/img', 
                                  transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn)

# Calculate baseline (random) accuracy using np.random.choice()
total_correct, total = 0, 0

for batch in iter(val_dataloader):
    labels = batch['label'].numpy()  
    predictions = np.random.choice([0, 1], size=len(labels))
    
    correct_predictions = (predictions == labels).astype(int)
    
    total_correct += correct_predictions.sum()
    total += len(labels)

# Print Results
accuracy = total_correct / total
print(f'Baseline (Random) Accuracy: {accuracy}')
```

    Baseline (Random) Accuracy: 0.496


# Part 1: Concatenation Model

For a baseline, we will create a late-fusion concatenation-based model.  What this means is we will load in a pre-trained image encoder from `PyTorch`'s `torchvision` ImageNet-trained model (https://pytorch.org/vision/stable/models.html#classification) to encode embeddings from the images, and load in a pre-trained `BERT` transformer from `Huggingface` (https://huggingface.co/docs/transformers/model_doc/bert) to encode embeddings from the captions.  Then we will concatenate these together, and make a 1-layer classifier layer to predict on the Hateful Memes task.


```python
class FusionModel(nn.Module):
    """
    A fusion model that combines features from text and image encoders to perform classification.
    """
    def __init__(self, text_encoder, image_encoder, num_labels, num_ftrs):
        super(FusionModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.classifier = nn.Linear(num_ftrs + 768, num_labels) 

    def forward(self, images, input_ids, attention_mask):
        # Extracting image features from the image encoder
        image_features = self.image_encoder(images)
        
        # Extracting text features from the text encoder.
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        # Concatenating the image and text features 
        combined_features = torch.cat((image_features, text_features), 1)
        
        # Passing the combined features through the classifier to get the logits
        logits = self.classifier(combined_features)
        
        return logits

# Helper function to fix model's gradients
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# get_optimizer allow to test different optimizers (hyperparameters)
def get_optimizer(model, hyperparams):
    if hyperparams['optimizer_type'] == 'Adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    elif hyperparams['optimizer_type'] == 'SGD':
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=hyperparams['learning_rate'], momentum=hyperparams['momentum'])
    else:
        raise ValueError("Unsupported optimizer type")
```


```python
# Function to run the training and validation
def run_experiment(hyperparams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained models
    image_encoder = resnet18(pretrained=True)
    set_parameter_requires_grad(image_encoder, hyperparams['freeze_image_encoder'])  # Freeze if needed
    num_ftrs = image_encoder.fc.in_features
    image_encoder.fc = nn.Identity()
    
    text_encoder = BertModel.from_pretrained('bert-base-uncased')
    set_parameter_requires_grad(text_encoder, hyperparams['freeze_text_encoder'])  # Freeze if needed
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare datasets and dataloaders
    train_dataset = HatefulMemesDataset(jsonl_file='data/train.jsonl', 
                                        img_dir='data/img', transform=transform)
    val_dataset = HatefulMemesDataset(jsonl_file='data/dev.jsonl', 
                                      img_dir='data/img', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    # Initialize the fusion model, loss function, and optimizer
    model = FusionModel(text_encoder, image_encoder, 2, num_ftrs).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, hyperparams)

    # Training loop
    model.train()
    total_epoch = hyperparams['num_epochs']
    total_loss = 0
    for epoch in range(total_epoch):
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{total_epoch}')
        for batch in progress_bar:
            # Prepare batch data
            images = batch['image'].to(device)
            captions = batch['caption']
            labels = batch['label'].to(device)
            
            # Tokenize text data
            inputs = tokenizer(captions, return_tensors='pt', padding=True, 
                               truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            # Backward pass and parameter update
            loss.backward()
            optimizer.step()
            
        # Calculate and print average loss for the epoch
        avg_loss = total_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch + 1}/{total_epoch}, Loss: {avg_loss:.4f}')
        total_loss = 0

    # Validation loop
    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for batch in iter(val_dataloader):
            # Prepare batch data
            images = batch['image'].to(device)
            captions = batch['caption']
            labels = batch['label'].to(device)
            
            # Tokenize text data
            inputs = tokenizer(captions, return_tensors='pt', padding=True, 
                               truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Prediction and accuracy calculation
            outputs = model(images, input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate and print validation accuracy
    accuracy = total_correct / total
    print(f'Validation Accuracy: {accuracy}')
    return accuracy
```

## Explore Hyper Parameters
With limiting computing rescources, we cannot explore all the possible parameter combinations. Thus, here are some interestering combinations to investigate.

### Adam Optimizer, Moderate Learning Rate
A balanced starting point, with moderate learning rate and common settings.
Nothing speacial for the result. Very similar to the baseline accuracy.



```python
hyperparams = {
    'batch_size': 64,
    'num_epochs': 2,
    'learning_rate': 1e-2,
    'optimizer_type': 'Adam',
    'weight_decay': 0,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/2: 100%|██████████| 133/133 [01:14<00:00,  1.80it/s]


    Epoch 1/2, Loss: 1.4650


    Epoch 2/2: 100%|██████████| 133/133 [01:12<00:00,  1.83it/s]


    Epoch 2/2, Loss: 0.7518
    Validation Accuracy: 0.496





    0.496



### Low Learning Rate & High Regularization
To test how a smaller learning rate combined with weight decay affects generalization.


```python
hyperparams = {
    'batch_size': 128,
    'num_epochs': 5,
    'learning_rate': 1e-4,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 67/67 [01:13<00:00,  1.09s/it]


    Epoch 1/5, Loss: 0.5872


    Epoch 2/5: 100%|██████████| 67/67 [01:13<00:00,  1.09s/it]


    Epoch 2/5, Loss: 0.3848


    Epoch 3/5: 100%|██████████| 67/67 [01:12<00:00,  1.08s/it]


    Epoch 3/5, Loss: 0.1952


    Epoch 4/5: 100%|██████████| 67/67 [01:12<00:00,  1.08s/it]


    Epoch 4/5, Loss: 0.0741


    Epoch 5/5: 100%|██████████| 67/67 [01:13<00:00,  1.10s/it]


    Epoch 5/5, Loss: 0.0348
    Validation Accuracy: 0.554





    0.554



### SGD Momentum Trial
Understanding how momentum in SGD compares to the default Adam setup.


```python
hyperparams = {
    'batch_size': 128,
    'num_epochs': 5,
    'learning_rate': 1e-2,
    'optimizer_type': 'SGD',
    'weight_decay': 0,
    'momentum': 0.9,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 67/67 [01:31<00:00,  1.36s/it, Loss=0.774]
    Epoch 2/5: 100%|██████████| 67/67 [01:31<00:00,  1.36s/it, Loss=0.598]
    Epoch 3/5: 100%|██████████| 67/67 [01:31<00:00,  1.36s/it, Loss=0.494]
    Epoch 4/5: 100%|██████████| 67/67 [01:31<00:00,  1.37s/it, Loss=0.498]
    Epoch 5/5: 100%|██████████| 67/67 [01:31<00:00,  1.36s/it, Loss=0.455]


    Validation Accuracy: 0.556





    0.556



### Fast Learning, Quick Iteration
For rapid prototyping, to see if the model learns anything quickly.



```python
hyperparams = {
    'batch_size': 64,
    'num_epochs': 2,
    'learning_rate': 1e-2,
    'optimizer_type': 'SGD',
    'weight_decay': 0,
    'momentum': 0.9,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/2: 100%|██████████| 133/133 [01:19<00:00,  1.67it/s]


    Epoch 1/2, Loss: 0.8373


    Epoch 2/2: 100%|██████████| 133/133 [01:18<00:00,  1.70it/s]


    Epoch 2/2, Loss: 0.7087
    Validation Accuracy: 0.522





    0.522



### Encoder Freezing Experiment - Image
Understanding the effect of freezing the image encoder while leaving text dynamic.

No notable performance.


```python
hyperparams = {
    'batch_size': 64,
    'num_epochs': 5,
    'learning_rate': 1e-2,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'freeze_image_encoder': True,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 133/133 [01:27<00:00,  1.52it/s, Loss=0.813]
    Epoch 2/5: 100%|██████████| 133/133 [01:27<00:00,  1.52it/s, Loss=0.941]
    Epoch 3/5: 100%|██████████| 133/133 [01:26<00:00,  1.53it/s, Loss=0.983]
    Epoch 4/5: 100%|██████████| 133/133 [01:27<00:00,  1.52it/s, Loss=1.75] 
    Epoch 5/5: 100%|██████████| 133/133 [01:27<00:00,  1.53it/s, Loss=0.773]


    Validation Accuracy: 0.5





    0.5



### Encoder Freezing Experiment - Text
Investigating the impact of freezing the text encoder.

Performed slightly better than Encoder Freezing Experiment - Image. 


```python
hyperparams = {
    'batch_size': 128,
    'num_epochs': 5,
    'learning_rate': 1e-2,
    'optimizer_type': 'SGD',
    'weight_decay': 0,
    'momentum': 0.9,
    'freeze_image_encoder': False,
    'freeze_text_encoder': True
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 67/67 [01:16<00:00,  1.14s/it]


    Epoch 1/5, Loss: 0.8575


    Epoch 2/5: 100%|██████████| 67/67 [01:21<00:00,  1.22s/it]


    Epoch 2/5, Loss: 0.6182


    Epoch 3/5: 100%|██████████| 67/67 [01:56<00:00,  1.74s/it]


    Epoch 3/5, Loss: 0.5307


    Epoch 4/5: 100%|██████████| 67/67 [01:17<00:00,  1.15s/it]


    Epoch 4/5, Loss: 0.5770


    Epoch 5/5: 100%|██████████| 67/67 [01:16<00:00,  1.14s/it]


    Epoch 5/5, Loss: 0.3206
    Validation Accuracy: 0.516





    0.516



### High Regularization and Slow Learning
Testing the impact of high regularization and slow learning rate on overfitting.

So far, the most impresive result.


```python
hyperparams = {
    'batch_size': 128,
    'num_epochs': 5,
    'learning_rate': 1e-4,
    'optimizer_type': 'Adam',
    'weight_decay': 1e-4,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 67/67 [01:13<00:00,  1.09s/it]


    Epoch 1/5, Loss: 0.5810


    Epoch 2/5: 100%|██████████| 67/67 [01:13<00:00,  1.09s/it]


    Epoch 2/5, Loss: 0.3823


    Epoch 3/5: 100%|██████████| 67/67 [01:13<00:00,  1.09s/it]


    Epoch 3/5, Loss: 0.1950


    Epoch 4/5: 100%|██████████| 67/67 [01:12<00:00,  1.09s/it]


    Epoch 4/5, Loss: 0.0905


    Epoch 5/5: 100%|██████████| 67/67 [01:12<00:00,  1.08s/it]


    Epoch 5/5, Loss: 0.0362
    Validation Accuracy: 0.558





    0.558



### Exploring Batch Size Impact
Seeing how increasing the batch size affects the optimization dynamics.



```python
hyperparams = {
    'batch_size': 128,
    'num_epochs': 2,
    'learning_rate': 1e-2,
    'optimizer_type': 'SGD',
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/2: 100%|██████████| 67/67 [01:12<00:00,  1.08s/it]


    Epoch 1/2, Loss: 0.7041


    Epoch 2/2: 100%|██████████| 67/67 [01:12<00:00,  1.08s/it]


    Epoch 2/2, Loss: 0.5334
    Validation Accuracy: 0.52





    0.52



### All Encoders Frozen 
Checking baseline performance with all components frozen; used for debugging.

Not a bad performance actually. Quite interesting ...


```python
hyperparams = {
    'batch_size': 64,
    'num_epochs': 2,
    'learning_rate': 1e-2,
    'optimizer_type': 'Adam',
    'weight_decay': 0,
    'freeze_image_encoder': True,
    'freeze_text_encoder': True
}
run_experiment(hyperparams)
```

    Epoch 1/2: 100%|██████████| 133/133 [01:10<00:00,  1.88it/s]


    Epoch 1/2, Loss: 1.0023


    Epoch 2/2: 100%|██████████| 133/133 [01:11<00:00,  1.85it/s]


    Epoch 2/2, Loss: 0.7202
    Validation Accuracy: 0.538





    0.538



### Balanced SGD Setup
Evaluating a balanced approach using SGD with momentum and moderate regularization.


```python
hyperparams = {
    'batch_size': 64,
    'num_epochs': 5,
    'learning_rate': 1e-5,
    'optimizer_type': 'SGD',
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'freeze_image_encoder': False,
    'freeze_text_encoder': False
}
run_experiment(hyperparams)
```

    Epoch 1/5: 100%|██████████| 133/133 [01:13<00:00,  1.82it/s]


    Epoch 1/5, Loss: 0.6692


    Epoch 2/5: 100%|██████████| 133/133 [01:13<00:00,  1.82it/s]


    Epoch 2/5, Loss: 0.6650


    Epoch 3/5: 100%|██████████| 133/133 [01:13<00:00,  1.82it/s]


    Epoch 3/5, Loss: 0.6627


    Epoch 4/5: 100%|██████████| 133/133 [01:13<00:00,  1.82it/s]


    Epoch 4/5, Loss: 0.6598


    Epoch 5/5: 100%|██████████| 133/133 [01:13<00:00,  1.82it/s]


    Epoch 5/5, Loss: 0.6578
    Validation Accuracy: 0.504





    0.504



### Observations

**Order by Validation Accuracy:**
1. 0.558: High Regularization and Slow Learning
2. 0.556: SGD Momentum Trial
3. 0.554: Low Learning Rate & High Regularization
4. 0.538: All Encoders Frozen - Quick Check
5. 0.522: Fast Learning, Quick Iteration
6. 0.520: Exploring Batch Size Impact
7. 0.516: Encoder Freezing Experiment - Text
8. 0.504: Balanced SGD Setup
9. 0.500: Encoder Freezing Experiment - Image
10. 0.496: Adam Optimizer, Moderate Learning Rate

Overall, we did not observed any impressive results (say, over 90% accuracy), proving that Humor/Sarcasm detection is indeed a challanging task in machine learing. Still, many of our models performed slightly better than the baseline accuracy (50%), **High Regularization and Slow Learning** performing the best. 


**Couple interesting findings:**
* Adam performed better in relatively lower learning rates
* SDG performed better in relatively higher learning rates
* Freezing encoders are not a bad idea 
* Batch sizes didn't impacted the results alot


# Part 2: Multimodal Transformers

Here, what we will do is to leverage the open-source code from Huggingface to access the internals of the BERT language model (https://huggingface.co/docs/transformers/model_doc/bert) and construct a model that fuses image embeddings from a pre-trained CNN (ex. ResNet18) with the token embeddings from the language (on the given caption), and glue all of this together to produce a binary classification model on the Hateful Memes dataset.

We will the Huggingface `Trainer` class, `Adam` optimizer and built-in learning rate scheduler to use `cosine` decay with a base learning rate of 1e-6.

When we construct our model architecture, we should take care to use positional embeddings for the image embeddings that are different from those that are used for the language token embeddings.  Also, we should add in an additional, learned "modality embedding" which messages to the model which "token feature" is coming from an image versus coming from language (recall how we do this with `nn.Embedding`).  What we want to accomplish here is "Early Fusion", meaning that the image and language embeddings are fused together EARLY on in the model (e.g., before they enter the Transformer Encoder).  Therefore, our job is to "get in" to the Huggingface code to be able to manipulate all of the embeddings to pull out and combine all this information as we see fit.

## Import & Redefine Dataset


```python
import torch
from PIL import Image
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torch.utils.data import Dataset
import json
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
from transformers import BertTokenizer, Trainer, TrainingArguments
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from transformers import TrainingArguments, Trainer
import evaluate
import warnings

warnings.filterwarnings('ignore') # ignore uncessary warnings


class HatefulMemesDataset(Dataset):
    """
    A custom dataset class for the Hateful Memes dataset, which loads images and their
    corresponding labels and text from a given directory and JSONL file.
    """
    def __init__(self, jsonl_file, img_dir, tokenizer, transforms):
        self.data = [json.loads(line) for line in open(jsonl_file, 'r')]
        self.img_dir = img_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.encoding = self.tokenizer([item['text'] for item in self.data], 
                                       padding=True, truncation=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get image from path
        img_subpath = item['img'].lstrip('/').replace('img/', '')  
        img_path = os.path.join(self.img_dir, img_subpath) 
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        return {
            'input_ids': self.encoding['input_ids'][idx],
            'attention_mask': self.encoding['attention_mask'][idx],
            'image': image,
            'label': torch.tensor(item['label']),
        }

# Define Toeknizer and Transformations
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # following ResNet
])

# Define Dataset
train_dataset = HatefulMemesDataset(jsonl_file='data/train.jsonl', img_dir='data/img', 
                                    tokenizer=tokenizer, transforms=transforms)
val_dataset = HatefulMemesDataset(jsonl_file='data/dev.jsonl', img_dir='data/img', 
                                  tokenizer=tokenizer, transforms=transforms)
```

## Create Model


```python
class MultimodalTransformer(nn.Module):
    def __init__(self):
        super(MultimodalTransformer, self).__init__()
        self.num_labels = 2
        resnet18 = models.resnet18(pretrained=True)

        # Text related layers (we will call parts of transformer separately)
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.word_modality_embeddings = nn.Parameter(torch.zeros(768))

        # Image related layers
        # using 2nd to last hidden layer
        self.image_encoder = nn.Sequential(*(list(resnet18.children())[:-2]))  
        self.image_dim_augmentation = nn.Linear(512, 768)
        self.image_positional_embeddings = nn.Embedding(49, 768)
        self.image_modality_embeddings = nn.Parameter(torch.ones(768))

        # Classifier on the combined features
        self.classifier = nn.Linear(768, self.num_labels)

        # Freeze image encoder weights
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, image, labels=None):        

        # input_ids: B x 88 
        # attention_mask: B x 88 
        # image: B x 3 x 224 x224
        # labels: B

        # Text processing 
        # Dimension: [B, 88, 768]
        word_position_ids = torch.arange(input_ids.size(1), device='cuda').unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.transformer.embeddings.word_embeddings(input_ids)
        word_position_embeddings = self.transformer.embeddings.position_embeddings(word_position_ids) 
        word_modality_embedding = self.word_modality_embeddings
        text_embedding = word_embeddings + word_position_embeddings + word_modality_embedding 
        
        # Image processing (Adjust dimension to fit text embedding)
        image_embedding = self.image_encoder(image) # [B, 512, 7, 7]
        image_embedding = image_embedding.view(image_embedding.size(0), 512, -1) # [B, 512, 49]
        image_embedding = image_embedding.permute(0, 2, 1) # [B, 49, 512]
        image_embedding = self.image_dim_augmentation(image_embedding) # [B, 49, 768]

        # Get image positional and modality embeddings
        image_position_ids = torch.arange(49, device='cuda').expand(image_embedding.size(0), -1)
        image_positional_embeddings = self.image_positional_embeddings(image_position_ids)
        image_modality_embeddings = self.image_modality_embeddings

        # Add embeddings to the image features
        image_embedding += image_positional_embeddings + image_modality_embeddings

        # Early Fusion, Concatenate text and image embeddings
        # [B, 49, 768] + [B, 88, 768] = [B, 137, 768]
        combined_embedding = torch.cat((image_embedding, text_embedding), dim=1) 

        # Combined attention mask for text and image
        image_attention = torch.ones([attention_mask.size(0), image_embedding.size(1)], device='cuda' )
        combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1) # [B,137]

        # BERT Encoder with combined features
        encoder_outputs = self.transformer(inputs_embeds=combined_embedding, 
                                           attention_mask=combined_attention_mask, return_dict=True)
        sequence_output = encoder_outputs.last_hidden_state

        # Classification
        pooled_output = sequence_output[:, 0, :] # CLS Token
        logits = self.classifier(pooled_output)

        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

```

## Train and evaluate


```python
# Setup the TrainingArguments and the Trainer
model = MultimodalTransformer()

training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    num_train_epochs=30,
    per_device_train_batch_size=128, 
    per_device_eval_batch_size=128, 
    warmup_steps=200,
    logging_steps=20,
    learning_rate=5e-6, 
    lr_scheduler_type="cosine",
    optim = "adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), 
                          references=p.label_ids)

trainer = Trainer( 
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()
```



    <div>

      <progress value='2010' max='2010' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [2010/2010 55:00, Epoch 30/30]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.655300</td>
      <td>0.741707</td>
      <td>0.498000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.653000</td>
      <td>0.770195</td>
      <td>0.494000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.617500</td>
      <td>0.753360</td>
      <td>0.504000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.620500</td>
      <td>0.823132</td>
      <td>0.504000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.577300</td>
      <td>0.843439</td>
      <td>0.530000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.529700</td>
      <td>0.915889</td>
      <td>0.538000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.499700</td>
      <td>0.832750</td>
      <td>0.554000</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.487700</td>
      <td>0.827720</td>
      <td>0.554000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.485700</td>
      <td>0.935908</td>
      <td>0.548000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.435500</td>
      <td>0.960028</td>
      <td>0.560000</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.438300</td>
      <td>0.968181</td>
      <td>0.568000</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.411900</td>
      <td>1.025490</td>
      <td>0.556000</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.401300</td>
      <td>1.092870</td>
      <td>0.562000</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.399000</td>
      <td>1.000045</td>
      <td>0.572000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.357300</td>
      <td>1.012873</td>
      <td>0.572000</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.344000</td>
      <td>1.175833</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.326300</td>
      <td>1.180638</td>
      <td>0.574000</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.294700</td>
      <td>1.220287</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.306700</td>
      <td>1.154577</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.300600</td>
      <td>1.196747</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.293900</td>
      <td>1.291886</td>
      <td>0.568000</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.256500</td>
      <td>1.249656</td>
      <td>0.562000</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.271900</td>
      <td>1.267073</td>
      <td>0.558000</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.253200</td>
      <td>1.302945</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.246400</td>
      <td>1.289166</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.248600</td>
      <td>1.317408</td>
      <td>0.562000</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.250600</td>
      <td>1.321625</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.260400</td>
      <td>1.289194</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.249600</td>
      <td>1.308939</td>
      <td>0.564000</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.250400</td>
      <td>1.306262</td>
      <td>0.562000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=2010, training_loss=0.3906482454556138, metrics={'train_runtime': 3303.1312, 'train_samples_per_second': 77.199, 'train_steps_per_second': 0.609, 'total_flos': 0.0, 'train_loss': 0.3906482454556138, 'epoch': 30.0})



## Observation

Despite some variations, the general trend shows an improvement in performance (accuracy) over time. The highest recorded accuracy reached 57.4% (comparable to 60% accuracy of the original paper), surpassing the results of all models evaluated in Part 1 and showcasing the capabilities of the transformer architecture. Nevertheless, this level of performance still falls short of the state-of-the-art (STOA) expectations set for other classification tasks, underscoring the complexity of hateful meme classification as a persistently challenging issue.

