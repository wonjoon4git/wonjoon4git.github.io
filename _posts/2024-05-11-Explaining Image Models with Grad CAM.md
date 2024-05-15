---
layout: single
classes: wide
title:  "Explaining Image Models with Grad CAM"
categories: 
  - CV
tag: [Explainable AI]
toc: true 
author_profile: false 
sidebar:
    nav: "counts"
---

# Grad CAM Demonstration 

Machine Learning models are often referred as a 'Black Box', due to its lack of explanability. We do not exactly know how the model came up with a conclusion whether a picture of an animal is a dog or a cat. 

Luckily, there are couple techniques out there to grant better explainability of how the model comes up with such conclusion. In this post, we will take a brief look into one of the most popular method, Grad CAM

The Grad-CAM technique utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score. The places where this gradient is large are exactly the places where the final score depends most on the data.


```python
# Install 
ip install "grad-cam" "transformers" "einops"
```

    Collecting grad-cam
      Downloading grad-cam-1.5.0.tar.gz (7.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m15.6 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.38.2)
    Collecting einops
      Downloading einops-0.7.0-py3-none-any.whl (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.6/44.6 kB[0m [31m4.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from grad-cam) (1.25.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from grad-cam) (9.4.0)
    Requirement already satisfied: torch>=1.7.1 in /usr/local/lib/python3.10/dist-packages (from grad-cam) (2.2.1+cu121)
    Requirement already satisfied: torchvision>=0.8.2 in /usr/local/lib/python3.10/dist-packages (from grad-cam) (0.17.1+cu121)
    Collecting ttach (from grad-cam)
      Downloading ttach-0.0.3-py3-none-any.whl (9.8 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from grad-cam) (4.66.2)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (from grad-cam) (4.8.0.76)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from grad-cam) (3.7.1)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from grad-cam) (1.2.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.3)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (4.11.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.7.1->grad-cam) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.7.1->grad-cam) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.7.1->grad-cam) (3.1.3)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    Collecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    Collecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    Collecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    Collecting nvidia-nccl-cu12==2.19.3 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl (166.0 MB)
    Collecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.7.1->grad-cam)
      Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.7.1->grad-cam) (2.2.0)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.7.1->grad-cam)
      Using cached nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (1.2.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (1.4.5)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->grad-cam) (2.8.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)
    Requirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->grad-cam) (1.11.4)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->grad-cam) (1.4.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->grad-cam) (3.4.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib->grad-cam) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.7.1->grad-cam) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.7.1->grad-cam) (1.3.0)
    Building wheels for collected packages: grad-cam
      Building wheel for grad-cam (pyproject.toml) ... [?25l[?25hdone
      Created wheel for grad-cam: filename=grad_cam-1.5.0-py3-none-any.whl size=38071 sha256=921df4d2142dd81897d3f43750b80e03e4935353085d77abce3a41898cba713f
      Stored in directory: /root/.cache/pip/wheels/5b/e5/3d/8548241d5cffe53ad1476c566a61ad9bf09cc61a9430f09726
    Successfully built grad-cam
    Installing collected packages: ttach, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, einops, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, grad-cam
    Successfully installed einops-0.7.0 grad-cam-1.5.0 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.1.105 ttach-0.0.3



```python
#Import
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
import requests
import numpy as np
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from transformers import AutoImageProcessor, ViTModel, ViTForImageClassification
import torch

from torchvision import transforms
from einops import rearrange
import matplotlib.pyplot as plt
```


```python
model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
```

    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    100%|██████████| 97.8M/97.8M [00:01<00:00, 78.9MB/s]



```python
image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
image = Image.open(requests.get(image_url, stream=True).raw)
```


```python
image
```




    
![png](/assets/images/GradCAM/output_5_0.png)





```python
img = np.array(image)
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(input_tensor.shape)
```

    torch.Size([1, 3, 224, 224])



```python
# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
targets = [ClassifierOutputTarget(295)]
target_layers = [model.layer4]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
Image.fromarray(images)
```




    
![png](/assets/images/GradCAM/output_7_0.png)
    




```python
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/69.7k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]



```python
inputs = image_processor(image, return_tensors="pt")
```


```python
inputs.update({"output_attentions": True})
```


```python
with torch.no_grad():
    outputs = model(**inputs)
```


```python
outputs.attentions[0].shape
```




    torch.Size([1, 12, 197, 197])




```python
attention_maps = outputs.attentions[0].squeeze(0).cpu().detach().numpy()
print(attention_maps.shape)
```

    (12, 197, 197)



```python
# Take the representation from the CLS token (from all heads in the first layer)
# attention_overlay = attention_maps[:, 0, :]
attention_overlay = attention_maps[:, :, 0]
print(attention_overlay.shape)
```

    (12, 197)



```python
# Take the mean across the heads
attention_overlay = attention_overlay.mean(0)
print(attention_overlay.shape)
```

    (197,)



```python
# Reshape it based on the number of tokens (-1 to ignore the CLS token itself)
num_patches = len(attention_overlay) - 1
patch_size = int(np.sqrt(num_patches))
attention_overlay = attention_overlay[1:].reshape(patch_size, patch_size)


print("Num patches:", num_patches)
print("Patch size:", patch_size)
print(attention_overlay.shape)
```

    Num patches: 196
    Patch size: 14
    (14, 14)



```python
# Reshape to overlay on the image
attention_overlay = cv2.resize(attention_overlay, (image.size[0], image.size[1]))
print(attention_overlay.shape)
```

    (1200, 1600)



```python
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13, 13))

axes.imshow(image)
axes.imshow(attention_overlay, cmap="inferno", alpha=0.6)
axes.title.set_text(f"Mean Attention Map: {0}")
axes.axis("off")
```




    (-0.5, 1599.5, 1199.5, -0.5)




    
![png](/assets/images/GradCAM/output_18_1.png)
    

