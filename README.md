# 🖼️ Image Captioning & Segmentation

A real-time web application that generates intelligent captions and performs object detection on uploaded images using state-of-the-art deep learning models.

🚀 **[Live Demo](https://www.linkedin.com/posts/abhaytiwari30_ai-deeplearning-computervision-activity-7341756104831287296-Rk4b?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD6u7bkBrCAv1pvhwbTv6f0OH_xLVQBEEw4)**

## ✨ Features

- 🔤 **Image Captioning** using ResNet-50 + LSTM architecture
- 🎯 **Object Detection & Segmentation** using Mask R-CNN
- 📸 **Upload or drag-and-drop** images with intuitive interface
- 🎛 **Adjustable detection confidence threshold** for fine-tuned results
- 📊 **Detection statistics** and beautifully visualized results

## 🛠 Tech Stack

| Area | Technology |
|------|------------|
| **UI** | Streamlit |
| **Captioning** | ResNet-50 (Encoder) + LSTM (Decoder) |
| **Segmentation** | Mask R-CNN with ResNet-50 FPN |
| **Dataset** | MS COCO 2017 |
| **Visualization** | Matplotlib, OpenCV |

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Abhaytiwari303/image-segmentation-captioning.git
cd image-segmentation-captioning
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** Ensure a C++ compiler is available on Windows for pycocotools installation.

### 4. Run the App
```bash
streamlit run app.py
```
The app will be available at [http://localhost:8501](http://localhost:8501)

## 📁 Project Structure

```
image-segmentation-captioning/
├── app.py                   # Streamlit frontend app
├── models/                  # Pre-trained models (optional)
├── requirements.txt         # Project dependencies
├── utils/                   # Utility/helper functions (if modularized)
└── README.md               # This file
```

## 🧠 Model Details

### Image Captioning
- **Encoder:** ResNet-50 pretrained on ImageNet (features only)
- **Decoder:** LSTM network trained on COCO vocabulary
- **Vocabulary:** Custom tokenizer and numericalizer

### Object Segmentation
- **Model:** maskrcnn_resnet50_fpn (pretrained=True)
- **Output:** Bounding boxes, instance masks, object labels, confidence scores

## 🎯 Use Cases

- 📚 **Educational demos** for AI/ML concepts and computer vision
- 🧪 **Research applications** on computer vision models
- 🖼️ **Automatic dataset annotation** for machine learning projects
- 🧠 **Real-time image understanding** systems and applications

## 🌐 Deployment

This project is deployed on **Streamlit Cloud**. To deploy your own version:

1. Push your project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New App" and link your repository
4. Set `app.py` as the entry point
5. Ensure `requirements.txt` is correctly configured

## 📋 Requirements

**Python Version:** 3.10

### Dependencies
```
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
opencv-python-headless==4.9.0.80
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.3
nltk==3.8.1
tqdm==4.66.2
streamlit==1.33.0
pillow==10.3.0
pycocotools==2.0.7
```

## 👨‍💻 Author

**Abhay Tiwari**
- 🐙 GitHub: [@Abhaytiwari303](https://github.com/Abhaytiwari303)
- 💼 LinkedIn: [abhaytiwari303](https://linkedin.com/in/abhaytiwari303)
- 📧 Email: at3032003@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TorchVision](https://pytorch.org/vision/) - Computer vision library
- [Streamlit](https://streamlit.io/) - Web app framework
- [COCO Dataset](https://cocodataset.org/) - Training data source

---

⭐ If you found this project helpful, please give it a star on GitHub!
