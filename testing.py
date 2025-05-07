import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import PIL
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
import random
from decoder import TransformerDecoder

print(PIL.__version__)

torch.manual_seed(42)
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
])


# making images the same embedding dim as caption
class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
projection = Projection(768, 512)

# Defining the dataset class that takes in images and captions and returns the hidden state of the image and the caption embeddings
class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions, image_transform=transform, model="openai/clip-vit-base-patch32"):
        self.images = images
        self.captions = captions
        self.image_transform = image_transform
        self.image_model = CLIPVisionModel.from_pretrained(model)
        self.image_processor = AutoProcessor.from_pretrained(model)
        self.projection = projection

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        captions = self.captions[idx]
        image = self.image_transform(image)

        # getting hidden state of image from CLIP
        inputs = self.image_processor(images=image, return_tensors="pt")

        outputs = self.image_model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        last_hidden_state = self.projection(last_hidden_state)

        caption = self.captions[idx]

        caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        caption_inputs = tokenizer(caption, return_tensors="pt", padding=True)

        input_ids = caption_inputs['input_ids']
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)

        token_embeddings = caption_model.text_model.embeddings.token_embedding(input_ids)
        position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)

        input_embeddings = token_embeddings + position_embeddings

        return last_hidden_state, input_embeddings




# loading the dataset
ds = load_dataset("nlphuji/flickr30k")
test_dataset = ds['test']

# splitting the dataset as it only has one split
split_dataset = test_dataset.train_test_split(test_size=0.2, seed=42)
print(split_dataset['train'][0])


# is this right
train_dataset = DecoderDataset(split_dataset['train']['image']['caption'])
test_dataset = DecoderDataset(split_dataset['test']['image']['caption'])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)