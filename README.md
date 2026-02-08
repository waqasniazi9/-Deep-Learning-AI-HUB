# 🧠 AI & Deep Learning Hub

A comprehensive, **free, and open-source** Streamlit-based platform for learning and experimenting with AI, Deep Learning, Computer Vision, and Natural Language Processing.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ Features

### 📚 **250+ Interactive Courses**

- PyTorch Fundamentals (100 series)
- Deep Learning & CNN (200 series)  
- Advanced Topics - RNN/LSTM/Transformers/BERT (300+ series)

### 🎯 **Comprehensive Sections**

| Section | Features |
|---------|----------|
| 🏠 Home | Overview, statistics, learning paths |
| 📚 PyTorch Basics | Tensors, operations, networks, training |
| 🖼️ Computer Vision | Classification, processing, CNN visualization |
| 📊 Data Tools | Dataset viewer, augmentation, statistics |
| 🔧 Model Tools | Architecture explorer, training monitor, export |
| ⚡ Advanced Tools | Batch processing, model comparison, hyperparameter tuning |
| 📈 Visualizations | Activation maps, feature space, learning curves |
| 📖 Resources | Course catalog, tutorials, documentation |

---

## 🚀 Quick Start

### Option 1: Run Directly (Windows)

```bash
# Double-click run.bat
run.bat
```

### Option 2: Run via Terminal

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run online_app.py
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run online_app.py
```

### ✅ The app will open automatically at: `http://localhost:8501`

---

## ☁️ Online Deployment

Your app is ready to deploy globally! Choose one:

### **Streamlit Cloud (RECOMMENDED)** ⭐

1. Push to GitHub
2. Go to <https://streamlit.io/cloud>
3. Connect & deploy (1 minute)
4. Get free public URL

See [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) for detailed instructions.

### Other Options

- 🔵 **Google Cloud Run** (free tier)
- 🟠 **AWS EC2** (free tier eligible)
- 🟣 **Heroku** (with trial)
- 🟢 **PythonAnywhere** (free + paid)

---

## 📦 Requirements

- Python 3.8 or higher
- 500MB disk space
- 1GB RAM (for full features)
- No special GPU required

### Dependencies

```
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=9.0.0
pandas>=1.5.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
scipy>=1.11.0
```

Automatically installed via: `pip install -r requirements.txt`

---

## 📚 Folder Structure

```
CV-main 2/
├── online_app.py              # Main Streamlit application
├── requirements.txt           # Dependencies
├── run.bat                    # Windows quick starter
├── run.sh                     # Linux/macOS quick starter
├── DEPLOYMENT_COMPLETE.md     # Detailed deployment guide
├── DEPLOYMENT.md              # Quick deployment info
├── README.md                  # This file
└── CV-main/                   # 250+ Jupyter notebooks
    ├── 100-122 series/        # PyTorch basics
    ├── 200-245 series/        # Deep learning & CNN
    └── 246-268+ series/       # Advanced topics
```

---

## 🎓 Learning Paths

### **Beginner (Week 1-4)**

1. PyTorch Basics (Section 1)
2. Tensors & Operations (Section 1)
3. Data Loading (Section 1)
4. Neural Networks Intro (Section 1)

### **Intermediate (Week 5-12)**

1. CNN Architecture (Section 2)
2. Image Classification (Section 2)
3. Object Detection (Section 2)
4. Optimization Techniques (Section 2)

### **Advanced (Week 13+)**

1. RNN/LSTM/GRU (Section 3)
2. Transformers & BERT (Section 3)
3. Attention Mechanisms (Section 3)
4. Custom Architectures

---

## 🛠️ Key Tools & Demos

### **Interactive ML Training**

```bash
# Train on Iris or Digits datasets
# Real-time accuracy monitoring
# Feature importance visualization
```

### **Image Processing Suite**

```bash
# Grayscale conversion
# Gaussian blur
# Edge detection
# Sharpening & inversion
# Batch processing
```

### **Model Comparison**

```bash
# Compare ResNet, VGG, MobileNet, EfficientNet
# Metrics: Parameters, Speed, Accuracy, Memory
# Visual comparison charts
```

### **Hyperparameter Tuning**

```bash
# Learning rate optimization
# Batch size selection
# Epoch scheduling
# Bayesian optimization visualizer
```

---

## 🔧 Installation Options

### **Full Installation (Recommended)**

```bash
pip install -r requirements.txt
```

### **GPU Support (NVIDIA CUDA)**

```bash
# For GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **CPU Only (Lightweight)**

```bash
pip install -r requirements.txt
```

---

## 📖 Usage Examples

### Example 1: Create a Tensor

```python
import torch
x = torch.randn(3, 4)
y = torch.zeros(3, 4)
z = x + y
```

### Example 2: Build a Neural Network

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyNet()
```

---

## 🌐 Global Access

Once deployed, your app is accessible **24/7** from anywhere:

- 📱 Mobile browsers
- 💻 Desktop browsers
- 🌍 Any device with internet
- 🔒 Secure HTTPS connection

Example URL: `https://your-app.streamlit.app`

---

## 📊 Statistics

- **250+** Course materials
- **50+** Interactive tools
- **100+** Code examples
- **0 Cost** to deploy (Streamlit Cloud)
- **∞ Scalability** (cloud-hosted)

---

## 🐛 Troubleshooting

### **App won't start?**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
streamlit run online_app.py --logger.level=debug
```

### **Port already in use?**

```bash
streamlit run online_app.py --server.port 8502
```

### **Module not found?**

```bash
# Activate virtual environment first
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

See [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) for more troubleshooting.

---

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Official Site](https://pytorch.org)
- [Papers with Code](https://paperswithcode.com)
- [Hugging Face](https://huggingface.co)
- [Stanford CS231N](http://cs231n.stanford.edu)
- [Dive into Deep Learning](https://d2l.ai)

---

## 🤝 Contributing

Contributions welcome!

1. Fork repository
2. Create feature branch
3. Make improvements
4. Submit pull request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Support

- **Issues:** GitHub Issues
- **Docs:** See [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md)
- **Community:** Streamlit Discord

---

## 🚀 Deploy Now

```bash
# 1. Test locally
streamlit run online_app.py

# 2. Push to GitHub
git push origin main

# 3. Deploy on Streamlit Cloud
# Go to: https://streamlit.io/cloud
# Connect your GitHub repo → Deploy → Done! 🎉
```

**Your app is live in 60 seconds!**

---

## 📈 Next Steps

1. ✅ Run locally: `streamlit run online_app.py`
2. ✅ Test all features in the app
3. ✅ Create GitHub repository
4. ✅ Deploy to Streamlit Cloud
5. ✅ Share the public URL
6. ✅ Monitor usage & feedback

---

Made with ❤️ for the AI/ML community

**Version:** 1.0  
**Status:** Production Ready ✅  
**Last Updated:** 2024
