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
    transforms.ToTensor(),
])

with torch.no_grad():
    # defining the models
    caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size


    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# making images the same embedding dim as caption
class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
projection = Projection(768, 512)



# doing preprocessing outside of dataset to make it faster
def process_dataset(images, captions, image_model, caption_model, tokenizer, image_processor, projection, device, batch_size=32):
    processed_captions = preprocess_captions(captions, tokenizer)
    
    all_data = []
    
    # Process images in batches
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing image batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        
        # Get current batch of images
        batch_images = [transform(images[i]) for i in range(start_idx, end_idx)]
        
        # Convert to format expected by processor
        with torch.no_grad():
            # Process all images in batch at once
            inputs = image_processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run model on batch
            outputs = image_model(**inputs)
            batch_features = projection(outputs.last_hidden_state)
            
            # Store results along with preprocessed captions
            for i, idx in enumerate(range(start_idx, end_idx)):
                all_data.append({
                    "image_features": batch_features[i:i+1].cpu(),
                    "captions": processed_captions[idx]
                })
    
    return all_data

# Defining the dataset class that takes in images and captions and returns the hidden state of the image and the caption embeddings
class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions, image_transform=transform, caption_model=caption_model, tokenizer=tokenizer, image_model=image_model, image_processor=image_processor, projection=projection):
        self.images = images
        self.captions = captions
        self.image_transform = image_transform
        self.projection = projection
        self.caption_model = caption_model
        self.tokenizer = tokenizer
        self.image_model = image_model
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        captions = self.captions[idx][:5]
        caption = captions[0]

        image = self.image_transform(image)

        with torch.no_grad():

            # getting hidden state of image from CLIP
            inputs = self.image_processor(images=image, return_tensors="pt")

            outputs = self.image_model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            last_hidden_state = self.projection(last_hidden_state)
            print(f"last_hidden_state shape: {last_hidden_state.shape}")

            caption_inputs = tokenizer(caption, return_tensors="pt", padding="max_length", max_length=77, truncation=True)

            input_ids = caption_inputs['input_ids'].squeeze(0)

            input_ids_in = input_ids[:-1]
            labels = input_ids[1:]


            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)

            input_ids_in = input_ids_in.to(device)
            position_ids = position_ids.to(device)

            token_embeddings = caption_model.text_model.embeddings.token_embedding(input_ids_in)
            position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)

            input_embeddings = token_embeddings + position_embeddings
            print(f"input_embeddings shape: {input_embeddings.shape}")

            decoder_input = torch.cat((last_hidden_state, input_embeddings), dim=1)

        return decoder_input.squeeze(0), labels.to(device)



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


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





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


    decoder = TransformerDecoder()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

 

    for epoch in range(epochs):
        decoder.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output, loss = decoder(inputs, targets=labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        print(f"Epoch {epoch} - Train Loss: {train_loss}")

    torch.save(decoder.state_dict(), f"decoder.pth")





