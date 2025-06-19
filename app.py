import streamlit as st
import os
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms, models
from pycocotools.coco import COCO
import cv2
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# === Model Classes (same as your original code) ===
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}

    def build_vocab(self, sentence_list):
        freq = {}
        idx = 4
        for sentence in sentence_list:
            words = sentence.lower().split()
            for word in words:
                freq[word] = freq.get(word, 0) + 1
                if freq[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.lower().split()]

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

# === Helper Functions ===
@st.cache_resource
def load_models():
    """Load and cache models to avoid reloading on every run"""
    try:
        # Initialize vocabulary (you might want to load this from a saved file)
        vocab = Vocabulary(freq_threshold=5)
        # For demo purposes, add some basic words
        basic_words = ["a", "an", "the", "person", "car", "dog", "cat", "house", "tree", "sky"]
        for i, word in enumerate(basic_words):
            vocab.word2idx[word] = i + 4
            vocab.idx2word[i + 4] = word
        
        # Initialize models
        embed_size = 256
        hidden_size = 512
        encoder = EncoderCNN(embed_size)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab.word2idx))
        
        # Load segmentation model
        seg_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        seg_model.eval()
        
        encoder.eval()
        decoder.eval()
        
        return encoder, decoder, vocab, seg_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def generate_caption(encoder, decoder, image_tensor, vocab, max_len=20):
    """Generate caption for an image"""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        feature = encoder(image_tensor)
        caption = [vocab.word2idx['<START>']]
        for _ in range(max_len):
            cap_tensor = torch.tensor([caption])
            output = decoder(feature, cap_tensor)
            _, pred = output[:, -1, :].max(1)
            caption.append(pred.item())
            if pred.item() == vocab.word2idx['<END>']:
                break
    return ' '.join([vocab.idx2word.get(idx, '<unk>') for idx in caption[1:-1]])

def get_coco_labels():
    """Get COCO class labels"""
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
        'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

def process_image(image, encoder, decoder, vocab, seg_model, threshold=0.5):
    """Process image for both captioning and segmentation"""
    # Prepare transforms
    transform_caption = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    transform_seg = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Convert PIL to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Generate caption
    img_tensor_caption = transform_caption(image).unsqueeze(0)
    caption = generate_caption(encoder, decoder, img_tensor_caption, vocab)
    
    # Generate segmentation
    img_tensor_seg = transform_seg(image).unsqueeze(0)
    with torch.no_grad():
        preds = seg_model(img_tensor_seg)[0]
    
    return caption, preds

def create_segmentation_plot(image, preds, threshold=0.5):
    """Create segmentation visualization"""
    coco_labels = get_coco_labels()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(np.array(image))
    
    detection_count = 0
    detected_objects = []
    
    for i, (box, label, score) in enumerate(zip(preds['boxes'], preds['labels'], preds['scores'])):
        if score < threshold:
            continue
            
        detection_count += 1
        x1, y1, x2, y2 = map(int, box.tolist())
        
        # Get class name
        if label.item() < len(coco_labels):
            class_name = coco_labels[label.item()]
        else:
            class_name = "Unknown"
            
        detected_objects.append(f"{class_name} ({score:.2f})")
        
        display_text = f"{class_name} {score:.2f}"
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(rect)
        
        # Draw label
        ax.text(x1, y1 - 10, display_text,
                fontsize=10, color='white',
                bbox=dict(facecolor='orange', alpha=0.8))
    
    ax.axis("off")
    ax.set_title("Object Detection Results", fontsize=16, fontweight='bold')
    
    return fig, detection_count, detected_objects

# === Main Streamlit App ===
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ Image Captioning & Segmentation</h1>
        <p>Upload an image to generate captions and detect objects</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment."):
        encoder, decoder, vocab, seg_model = load_models()
    
    if encoder is None:
        st.error("Failed to load models. Please check your setup.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Minimum confidence score for object detection"
    )
    
    show_caption = st.sidebar.checkbox("Show Caption", value=True)
    show_segmentation = st.sidebar.checkbox("Show Object Detection", value=True)
    
    # File uploader
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image to analyze"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🤖 Analysis")
            
            # Process image
            with st.spinner("Analyzing image..."):
                try:
                    caption, preds = process_image(
                        image, encoder, decoder, vocab, seg_model, threshold
                    )
                    
                    # Show caption
                    if show_caption:
                        st.markdown("### 💬 Generated Caption")
                        st.markdown(f"""
                        <div class="caption-box">
                            <h4>"{caption}"</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detection stats
                    if show_segmentation:
                        valid_detections = sum(1 for score in preds['scores'] if score >= threshold)
                        st.metric("Objects Detected", valid_detections)
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.info("This might be due to model weights not being loaded. In a production setup, you would load pre-trained weights.")
        
        # Show segmentation results
        if show_segmentation and uploaded_file is not None:
            st.subheader("🎯 Object Detection Results")
            
            try:
                fig, detection_count, detected_objects = create_segmentation_plot(
                    image, preds, threshold
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("### 📊 Detection Summary")
                    st.write(f"**Total Objects:** {detection_count}")
                    
                    if detected_objects:
                        st.markdown("**Detected Objects:**")
                        for obj in detected_objects:
                            st.write(f"• {obj}")
                    else:
                        st.info("No objects detected above the threshold.")
            
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
    
    else:
        # Show example or instructions
        st.info("👆 Please upload an image to get started!")
        
        # Example images section
        st.subheader("📋 How to use:")
        st.write("""
        1. **Upload an image** using the file uploader above
        2. **Adjust settings** in the sidebar if needed
        3. **View results** including:
           - AI-generated caption describing the image
           - Object detection with bounding boxes
           - Detection statistics and object list
        """)
        
        # Technical info
        with st.expander("🔧 Technical Information"):
            st.write("""
            **Models Used:**
            - **Caption Generation**: ResNet50 + LSTM encoder-decoder architecture
            - **Object Detection**: Mask R-CNN with ResNet50 backbone
            - **Dataset**: Trained on COCO dataset
            
            **Features:**
            - Real-time image analysis
            - Adjustable detection threshold
            - Detailed object classification
            - Confidence scores for each detection
            """)

if __name__ == "__main__":
    main()