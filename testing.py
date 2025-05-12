import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
import torchvision.transforms as transforms
from decoder import TransformerDecoder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_caption_generation():
    print(f"Running on device: {device}")
    
    # Load models
    decoder_path = "decoder_8.pth"
    decoder = TransformerDecoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    
    caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)
    
    # Load test images
    ds = load_dataset("nlphuji/flickr30k", split="test[:5]")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Process each image
    for idx in range(len(ds)):
        print(f"\nImage {idx}:")
        
        # Get image and original caption
        image = ds[idx]['image']
        original_caption = ds[idx]['caption'][0]
        print(f"Original caption: {original_caption}")
        
        # Process image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get image features
            inputs = image_processor(images=img_tensor, return_tensors="pt").to(device)
            outputs = image_model(**inputs)
            vision_embeds = outputs.last_hidden_state  # (1, N, 768)
            
            # Start with beginning of sequence token
            start_token = 49406  # <|startoftext|>
            current_ids = torch.tensor([[start_token]], device=device)
            
            # Print the token being added at each step
            print("Token-by-token generation:")
            print(f"  Starting with token: {start_token} ({tokenizer.decode([start_token])})")
            
            # Generate caption token by token
            generated_caption = []
            max_length = 76
            current_text = ""
            
            for step in range(max_length):
                # Create position IDs and embeddings
                position_ids = torch.arange(0, current_ids.size(1), dtype=torch.long).unsqueeze(0).to(device)
                token_embeddings = caption_model.text_model.embeddings.token_embedding(current_ids)
                position_embeddings = caption_model.text_model.embeddings.position_embedding(position_ids)
                text_embeds = token_embeddings + position_embeddings
                
                # Get next token prediction
                logits, _ = decoder(captions=text_embeds, images=vision_embeds)
                next_token_id = torch.argmax(logits[0, -1]).item()
                
                # Add token to caption
                generated_caption.append(next_token_id)
                
                # Add token to current sequence - CRITICAL STEP
                new_token_tensor = torch.tensor([[next_token_id]], device=device)
                current_ids = torch.cat([current_ids, new_token_tensor], dim=1)
                
                # Decode for printing
                new_token_text = tokenizer.decode([next_token_id])
                current_text += new_token_text
                
                # Print progress
                print(f"  Step {step+1}: Added token {next_token_id} ({new_token_text}) → Current: {current_text}")
                
                # Break if end token
                if next_token_id == 49407:  # <|endoftext|>
                    break
            
            # Final caption
            caption_text = tokenizer.decode(generated_caption)
            print(f"Final caption: {caption_text}")
            
            # Save image with caption
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(f"Original: {original_caption}\nGenerated: {caption_text}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"simple_test_{idx}.png")
            plt.close()
    
    print("Testing complete.")

if __name__ == "__main__":
    test_caption_generation()