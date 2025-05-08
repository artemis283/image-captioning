import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
import torchvision.transforms as transforms
from decoder import TransformerDecoder

# Set device and seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)

def test_with_matching_process():
    print(f"Running on device: {device}")
    
    # 1. Load models - using the same setup as in your training code
    print("Loading models...")
    
    # Load the decoder model
    model_path = "decoder_1.pth"  # Update with your model path
    decoder = TransformerDecoder().to(device)
    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()
    print("Decoder loaded successfully")
    
    # Load CLIP models - exactly as in your training
    caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)
    print("CLIP models loaded successfully")
    
    # Create projection layer (same as in your training)
    class Projection(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            return self.linear(x)
    
    projection = Projection(768, 512).to(device)
    
    # 2. Load test images
    print("Loading test images from Flickr30k...")
    ds = load_dataset("nlphuji/flickr30k", split="test[:5]")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 3. Process each test image using the exact same process as your training
    for idx in range(len(ds)):
        print(f"\nTesting on sample {idx}:")
        
        # Get image and original caption
        image = ds[idx]['image']
        original_caption = ds[idx]['caption'][0]
        print(f"Original caption: {original_caption}")
        
        # Process image exactly as in your training code
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Process image through CLIP vision model
            inputs = image_processor(images=img_tensor, return_tensors="pt").to(device)
            outputs = image_model(**inputs)
            vision_embeds = outputs.last_hidden_state  # (1, N, 768)
            vision_embeds = projection(vision_embeds)  # (1, N, 512)
            
            # Now we need to create text embeddings for generation
            # Start with a beginning of sequence token
            start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
            if start_token is None:
                # Use default start token ID for CLIP
                start_token = 49406  # <|startoftext|>
            
            # Initialize with just the start token
            current_ids = torch.tensor([[start_token]], device=device)
            
            # Generate caption step by step - matching your training
            generated_caption = []
            max_length = 20  # Cap at 20 tokens for brevity
            
            for _ in range(max_length):
                # Create position IDs - exactly as in your training
                position_ids = torch.arange(0, current_ids.size(1), dtype=torch.long).unsqueeze(0).to(device)
                
                # Get token embeddings - exactly as in your training
                token_embeddings = caption_model.text_model.embeddings.token_embedding(current_ids)
                position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)
                text_embeds = token_embeddings + position_embeddings  # (1, curr_len, 512)
                
                # Concatenate with vision embeddings - exactly as in your training
                full_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # (1, N+curr_len, 512)
                
                # Critical: Remove the vision tokens as in your training
                decoder_inputs = full_embeds[:, 50:, :]  # Remove the image tokens
                
                # Forward through decoder
                logits, _ = decoder(decoder_inputs)
                
                # Get next token
                next_token_id = torch.argmax(logits[0, -1]).item()
                generated_caption.append(next_token_id)
                
                # Append token to current sequence
                current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                # Break if end token
                if next_token_id == tokenizer.eos_token_id or next_token_id == 49407:  # <|endoftext|>
                    break
            
            # Convert to text
            caption_text = tokenizer.decode(generated_caption)
            print(f"Generated caption: {caption_text}")
            
            # Display first few token predictions
            print("First 5 token predictions:")
            for i, token_id in enumerate(generated_caption[:5]):
                token_text = tokenizer.decode([token_id])
                print(f"  {i+1}: {token_id} ({token_text})")
        
        # 4. Display the image with captions
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"Original: {original_caption}\nGenerated: {caption_text}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"matching_test_{idx}.png")
        print(f"Image saved as matching_test_{idx}.png")
    
    print("\nTesting complete.")

if __name__ == "__main__":
    test_with_matching_process()