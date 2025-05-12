import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
from decoder import TransformerDecoder
import torch.nn.functional as F

# Set up the page
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="��",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .caption-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📷 Image Captioning")
st.markdown("Upload an image or take a photo to generate a caption!")

# Initialize models (only once)
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP models
    caption_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load decoder
    decoder = TransformerDecoder().to(device)
    decoder.load_state_dict(torch.load("decoder_3.pth", map_location=device))
    decoder.eval()
    
    return decoder, caption_model, tokenizer, image_model, image_processor, device

# Load models
decoder, caption_model, tokenizer, image_model, image_processor, device = load_models()

def generate_caption(image, max_length=77):
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get image features
        inputs = image_processor(images=img_tensor, return_tensors="pt").to(device)
        outputs = image_model(**inputs)
        vision_embeds = outputs.last_hidden_state
        
        # Start with beginning of sequence token
        start_token = tokenizer.bos_token_id
        current_ids = torch.tensor([[start_token]], device=device)
        
        # Generate caption token by token
        generated_caption = []
        
        for _ in range(max_length-1):
            # Get embeddings
            token_embeddings = caption_model.text_model.embeddings.token_embedding(current_ids)
            
            # Forward pass
            logits, _ = decoder(token_embeddings, vision_embeds)
            next_token_logits = logits[0, -1]
            
            # Apply temperature
            temperature = 0.7
            next_token_logits = next_token_logits / temperature
            
            # Apply softmax
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Add repetition penalty
            for token_id in set(generated_caption):
                probs[token_id] /= 1.2
            
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to sequence
            generated_caption.append(next_token)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Break if end token
            if next_token == tokenizer.eos_token_id:
                break
        
        # Decode caption
        caption = tokenizer.decode(generated_caption)
        return caption

# Create two columns for upload and camera
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("Take Photo")
    camera_input = st.camera_input("Take a photo")

# Process the image (either uploaded or from camera)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
            st.markdown(f"""
            <div class="caption-box">
                <h3>Generated Caption:</h3>
                <p>{caption}</p>
            </div>
            """, unsafe_allow_html=True)

elif camera_input is not None:
    image = Image.open(camera_input)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
            st.markdown(f"""
            <div class="caption-box">
                <h3>Generated Caption:</h3>
                <p>{caption}</p>
            </div>
            """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")