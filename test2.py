import torch
import torchvision.transforms as T
from decoder import TransformerDecoder
from transformers import CLIPModel, CLIPTokenizer
from datasets import load_dataset
from PIL import Image
import random
import torch.nn.functional as F

# Set up environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP & Tokenizer
CLIP = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
CLIP.eval()
for param in CLIP.parameters():
    param.requires_grad = False
    
# Load token embedding
token_embedding = CLIP.text_model.embeddings.token_embedding.to(device)
token_embedding.weight.requires_grad = False

# Load dataset
flickr = load_dataset("nlphuji/flickr30k")

# Set seeds
torch.manual_seed(42)
random.seed(42)

# Load model
model = TransformerDecoder().to(device)
checkpoint = torch.load("decoder_0.pth", map_location=device)
model.load_state_dict(checkpoint)

def get_test_image(index=5):
    test_image = flickr['test'][index]['image']
    original_caption = flickr['test'][index]['caption'][0]
    print(f"Original caption: {original_caption}")
    
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    with torch.no_grad():
        # Process through CLIP vision model
        pixel_values = preprocess(test_image).unsqueeze(0).to(device)
        vision_outputs = CLIP.vision_model(pixel_values)
        patch_embeddings = vision_outputs.last_hidden_state
        return patch_embeddings, test_image

def inference(model, image_index=5, start_token=49406, end_token=49407, max_len=20):
    model.eval()
    
    # Get image and process it
    img, original_image = get_test_image(image_index)
    
    with torch.no_grad():
        # Start with <start> token
        generated = [start_token]
        
        print("Generating caption token by token:")
        for step in range(max_len):
            # Convert current tokens to tensor
            y = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            y_emb = token_embedding(y).to(device)
            
            # Get next token prediction
            logits, _ = model(captions=y_emb, images=img)
            next_token_logits = logits[0, -1]
            
            # Apply temperature to make predictions less deterministic
            temperature = 0.7
            next_token_logits = next_token_logits / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Add repetition penalty
            for token_id in set(generated):
                probs[token_id] /= 1.2  # Reduce probability of already generated tokens
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Print progress
            token_text = tokenizer.decode([next_token])
            print(f"  Step {step+1}: Token {next_token} ({token_text})")
            
            # Add to generated caption
            generated.append(next_token)
            
            # Break if end token
            if next_token == end_token:
                break
                
        # Decode final caption
        caption = tokenizer.decode(generated[1:])
        print(f"\nGenerated caption: {caption}")
        
        return caption, original_image

if __name__ == "__main__":
    # Generate captions for a few images
    for idx in range(5):
        print(f"\n{'='*50}")
        print(f"IMAGE {idx}")
        print(f"{'='*50}")
        
        caption, image = inference(model, image_index=idx)
        
        # You can uncomment to display/save the image
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # plt.imshow(image)
        # plt.title(f"Generated: {caption}")
        # plt.axis('off')
        # plt.savefig(f"caption_result_{idx}.png")
        # plt.close()