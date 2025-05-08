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
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(PIL.__version__)

torch.manual_seed(42)
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

with torch.no_grad():
    # defining the models
    caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size


    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)




def collate_fn(batch):
    images, captions = zip(*batch)

    # Stack and process images
    images = torch.stack(images).to(device)
    with torch.no_grad():
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        outputs = image_model(**inputs)
        vision_embeds = outputs.last_hidden_state  # (B, N, 768)

    # Tokenize captions
    caption_inputs = tokenizer(
        list(captions),
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    ).to(device)

    input_ids = caption_inputs['input_ids']
    input_ids_in = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    position_ids = torch.arange(0, input_ids_in.size(1), dtype=torch.long).unsqueeze(0).repeat(input_ids_in.size(0), 1).to(device)

    token_embeddings = caption_model.text_model.embeddings.token_embedding(input_ids_in)
    position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)

    input_embeddings = token_embeddings + position_embeddings  # (B, T, 512)

    return vision_embeds, input_embeddings, labels




# Defining the dataset class that takes in images and captions and returns the hidden state of the image and the caption embeddings
class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions, image_transform=transform, caption_model=caption_model, tokenizer=tokenizer, image_model=image_model, image_processor=image_processor):
        self.images = images
        self.captions = captions
        self.image_transform = image_transform
        self.caption_model = caption_model
        self.tokenizer = tokenizer
        self.image_model = image_model
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.image_transform(image)

        # make random
        caption = self.captions[idx][random.randint(0, 4)]


        return image, caption



# loading the dataset
ds = load_dataset("nlphuji/flickr30k")
test_dataset = ds['test']


# splitting the dataset as it only has one split
split_dataset = test_dataset.train_test_split(test_size=0.2, seed=42)

train_images = split_dataset['train']['image']
train_captions = split_dataset['train']['caption']

test_images = split_dataset['test']['image']
test_captions = split_dataset['test']['caption']

train_dataset = DecoderDataset(train_images, train_captions)
test_dataset = DecoderDataset(test_images, test_captions)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)





if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    learning_rate = 0.0005
    epochs = 10

    wandb.init(project="image-captioning", config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    decoder = TransformerDecoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

 
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, captions, labels in progress_bar:
            images, captions, labels = images.to(device), captions.to(device), labels.to(device)


            optimizer.zero_grad()


            output, loss = decoder(captions, images, targets=labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        print(f"Epoch {epoch} - Train Loss: {train_loss}")

        torch.save(decoder.state_dict(), f"decoder_{epoch}.pth")






