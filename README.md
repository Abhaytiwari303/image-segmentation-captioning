project:
  title: "🖼️ Image Captioning & Segmentation"
  description: >
    A real-time web app that generates intelligent captions and performs object detection
    on uploaded images using deep learning models.
  live_app: "https://image-segmentation-captioning-evntpdwjz2xdbaccnpyqz7.streamlit.app"

features:
  - "🔤 Image Captioning using ResNet-50 + LSTM"
  - "🎯 Object Detection & Segmentation using Mask R-CNN"
  - "📸 Upload or drag-and-drop images"
  - "🎛 Adjustable detection confidence threshold"
  - "📊 Detection stats and visualized results"

tech_stack:
  - area: "UI"
    tool: "Streamlit"
  - area: "Captioning"
    tool: "ResNet-50 (Encoder) + LSTM (Decoder)"
  - area: "Segmentation"
    tool: "Mask R-CNN with ResNet-50 FPN"
  - area: "Dataset"
    tool: "MS COCO 2017"
  - area: "Visualization"
    tool: "Matplotlib, OpenCV"

setup:
  steps:
    - name: "Clone Repository"
      commands:
        - "git clone https://github.com/Abhaytiwari303/image-segmentation-captioning.git"
        - "cd image-segmentation-captioning"
    - name: "Create Virtual Environment"
      commands:
        - "python -m venv venv"
        - "source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    - name: "Install Dependencies"
      commands:
        - "pip install -r requirements.txt"
      note: "Ensure a C++ compiler is available on Windows for pycocotools."
    - name: "Run the App"
      commands:
        - "streamlit run app.py"
      url: "http://localhost:8501"

project_structure:
  - "app.py                   # Streamlit frontend app"
  - "models/                  # Pre-trained models (optional)"
  - "requirements.txt         # Project dependencies"
  - "utils/                   # Utility/helper functions (if modularized)"
  - "README.md                # This file"

model_details:
  captioning:
    encoder: "ResNet-50 pretrained on ImageNet (features only)"
    decoder: "LSTM network trained on COCO vocabulary"
    vocabulary: "Custom tokenizer and numericalizer"
  segmentation:
    model: "maskrcnn_resnet50_fpn(pretrained=True)"
    output: "Bounding boxes, masks, labels, confidence scores"

use_cases:
  - "📚 Educational demos for AI/ML concepts"
  - "🧪 Research on computer vision models"
  - "🖼️ Automatic dataset annotation"
  - "🧠 Real-time image understanding systems"

deployment:
  platform: "Streamlit Cloud"
  steps:
    - "Push project to GitHub"
    - "Go to Streamlit Cloud"
    - "Click 'New App' and link your repository"
    - "Set 'app.py' as the entry point"
    - "Ensure 'requirements.txt' is correctly configured"

requirements:
  python_version: "3.13"
  packages:
    - "torch==2.7.1"
    - "torchvision==0.22.1"
    - "torchaudio==2.7.1"
    - "opencv-python-headless==4.9.0.80"
    - "numpy==1.26.4"
    - "pandas==2.2.2"
    - "matplotlib==3.8.3"
    - "nltk==3.8.1"
    - "tqdm==4.66.2"
    - "streamlit==1.33.0"
    - "pillow==10.3.0"
    - "pycocotools==2.0.7"

author:
  name: "Abhay Tiwari"
  github: "https://github.com/Abhaytiwari303"
  linkedin: "https://linkedin.com/in/abhaytiwari303"
  email: "your.email@example.com"  # Replace with your actual email

license:
  type: "MIT"
  file: "LICENSE"

acknowledgements:
  - "PyTorch"
  - "TorchVision"
  - "Streamlit"
  - "COCO Dataset"
