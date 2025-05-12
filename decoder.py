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
import torch.nn.functional as F

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


# computing training accuracy
def compute_training_accuracy(predictions, targets, pad_token_id=None):
    if pad_token_id is not None:
        mask = targets != pad_token_id
        correct_predictions = (predictions==targets) & mask
        correct_count = correct_predictions.sum().float()
        total_count = mask.sum().float()
    else:
        correct_predictions = (predictions==targets)
        correct_count = correct_predictions.sum().float()
        total_count = targets.numel()

    accuracy = correct_count / total_count
    return accuracy

def collate_fn(batch):
    images, captions = zip(*batch)

    # Stack and process images
    images = torch.stack(images).to(device)
    with torch.no_grad():
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        outputs = image_model(**inputs)
        vision_embeds = outputs.last_hidden_state  # (B, N, 768)

    tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.pad_token

    # Tokenize captions
    caption_inputs = tokenizer(
        list(captions),
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True, 
        padding_side="right",
        pad_to_multiple_of=77,
    ).to(device)

    input_ids = caption_inputs['input_ids']
    input_ids_in = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    position_ids = torch.arange(0, input_ids_in.size(1), dtype=torch.long).unsqueeze(0).repeat(input_ids_in.size(0), 1).to(device)

    token_embeddings = caption_model.text_model.embeddings.token_embedding(input_ids_in)
    position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)

    input_embeddings = token_embeddings + position_embeddings  # (B, T, 512)

    return vision_embeds, input_embeddings, labels


# generating captions to for inference
def generate_caption(model, image_embeds, max_length=77, start_token_id=tokenizer.bos_token_id, eos_token_id=None):
    model.eval()
    
    # Ensure image_embeds has batch size 1
    if image_embeds.dim() == 3:  # If it's (B, N, D)
        image_embeds = image_embeds[0:1]  # Take first image and keep batch dimension
    
    input_ids = torch.tensor([[start_token_id]]).to(device)  # [1, 1]
    
    generated_sequence = [start_token_id]
    
    with torch.no_grad():
        for _ in range(max_length-1):
            # Get embeddings from CLIP text model
            token_embeddings = caption_model.text_model.embeddings.token_embedding(input_ids)
            
            # Forward pass - unpack the tuple
            logits, _ = model(token_embeddings, image_embeds)  # Get logits and ignore loss
            
            # Get the prediction for the next token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature to make predictions less deterministic
            temperature = 0.7
            next_token_logits = next_token_logits / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution instead of taking argmax
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Add to the sequence
            generated_sequence.append(next_token.item())
            
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Break if end of sequence token is generated
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    
    return generated_sequence



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


def contrastive_loss(image_features, text_features):
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # Calculate similarity
    logits = torch.matmul(text_features, image_features.transpose(-2, -1))
    
    # Create labels (diagonal is positive pairs)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Calculate loss
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss



if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    learning_rate = 0.0005
    epochs = 20

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
        total_correct = 0
        total_count = 0
    

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, captions, labels in progress_bar:
            images, captions, labels = images.to(device), captions.to(device), labels.to(device)


            optimizer.zero_grad()


            output, loss = decoder(captions, images, targets=labels)

            # Get global features for contrastive loss
            '''image_feat_global = images.mean(dim=1)  # [B, 768]
            text_feat_global = captions.mean(dim=1)  # [B, 512]

            # Project image features to match text feature dimension
            image_feat_global = decoder.projection(image_feat_global)  # [B, 512]

            # Calculate contrastive loss
            contrast_loss = contrastive_loss(image_feat_global, text_feat_global)

            # Combined loss
            loss = caption_loss + 0.2 * contrast_loss'''
            loss.backward()
            optimizer.step()

            # Use output[0] to get the logits for predictions
            predictions = output.argmax(dim=-1)

            mask = labels != tokenizer.bos_token_id
            correct_predictions = (predictions==labels) & mask
            batch_correct = correct_predictions.sum().item()
            batch_total = mask.sum().item()

            total_correct += batch_correct
            total_count += batch_total

            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0


            train_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item(), "accuracy": batch_accuracy})

        for i, (images, captions, labels) in enumerate(test_loader):
            if i >= 3: 
                break
                
            images =  images.to(device)
            
            with torch.no_grad():
                # Generate caption
                caption = generate_caption(
                    decoder, 
                    images, 
                    max_length=77, 
                    start_token_id=tokenizer.bos_token_id, 
                    eos_token_id=tokenizer.eos_token_id
                )
        
                # Decode caption
                caption_text = tokenizer.decode(caption, skip_special_tokens=True)
                print(f"Caption: {caption_text}")
                    


        train_loss /= len(train_loader)
        avg_accuracy = total_correct / total_count if total_count > 0 else 0

        wandb.log({"train_loss": train_loss, "train_accuracy": avg_accuracy, "epoch": epoch})

        print(f"Epoch {epoch} - Train Loss: {train_loss} - Train Accuracy: {avg_accuracy}")

        torch.save(decoder.state_dict(), f"decoder_{epoch}.pth")