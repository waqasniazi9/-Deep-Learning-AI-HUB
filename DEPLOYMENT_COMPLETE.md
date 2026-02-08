# 🚀 AI & Deep Learning Hub - Complete Setup & Deployment Guide

## 📋 Table of Contents

1. [Local Installation & Running](#local-installation--running)
2. [Online Deployment Options](#online-deployment-options)
3. [Platform-Specific Instructions](#platform-specific-instructions)
4. [Features Overview](#features-overview)
5. [Troubleshooting](#troubleshooting)

---

## 🌐 Local Installation & Running

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Windows, macOS, or Linux

### Step 1: Clone/Download the Repository

```bash
cd CV-main\ 2
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install individually:**

```bash
pip install streamlit torch torchvision numpy matplotlib pillow pandas scikit-learn opencv-python scipy plotly
```

### Step 4: Run the Application Locally

```bash
streamlit run online_app.py
```

The app will automatically open at:

- 🔗 **<http://localhost:8501>**

**Optional: Run with custom settings:**

```bash
streamlit run online_app.py --logger.level=info --client.showErrorDetails=true
```

---

## ☁️ Online Deployment Options

### ⭐ **Option 1: Streamlit Cloud (RECOMMENDED - FREE)**

#### Setup Steps

1. **Prepare GitHub Repository**

   ```bash
   git init
   git add .
   git commit -m "Initial AI Hub deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cv-deep-learning.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to <https://streamlit.io/cloud>
   - Click **"New app"**
   - Connect your GitHub account
   - Select repository: `cv-deep-learning`
   - Select main file: `online_app.py`
   - Click **"Deploy"**
   - Your app is live! Share the public URL

3. **Features:**
   - ✅ Free hosting
   - ✅ Auto-deploys from GitHub
   - ✅ SSL certificate included
   - ✅ 1GB RAM, 1 vCPU
   - ✅ Custom domain support (paid)

---

### **Option 2: Heroku (Free with Trial)**

#### Setup Steps

1. **Install Heroku CLI**

   ```bash
   # Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
   # macOS: brew tap heroku/brew && brew install heroku
   # Linux: curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Heroku App**

   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Create Procfile**

   ```
   web: streamlit run --server.port=$PORT --server.address=0.0.0.0 online_app.py
   ```

4. **Deploy**

   ```bash
   git push heroku main
   ```

5. **Access App**

   ```
   https://your-app-name.herokuapp.com
   ```

---

### **Option 3: Google Cloud Run (FREE TIER)**

#### Setup Steps

1. **Create Dockerfile**

   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD ["streamlit", "run", "--server.port=8080", "--server.address=0.0.0.0", "online_app.py"]
   ```

2. **Create .dockerignore**

   ```
   .git
   .gitignore
   README.md
   .venv
   venv
   __pycache__
   *.pyc
   .pytest_cache
   ```

3. **Deploy**

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud run deploy ai-hub --source .
   ```

---

### **Option 4: AWS EC2 (Free Tier Eligible)**

#### Setup Steps

1. **Launch EC2 Instance**
   - Go to <https://aws.amazon.com/ec2/>
   - Launch Ubuntu 22.04 LTS instance (free tier)
   - Allow HTTP/HTTPS in security group

2. **Connect via SSH**

   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies**

   ```bash
   sudo apt update
   sudo apt install python3-pip git
   git clone https://github.com/YOUR_USERNAME/cv-deep-learning.git
   cd cv-deep-learning
   pip3 install -r requirements.txt
   ```

4. **Run with Gunicorn (Production)**

   ```bash
   pip3 install gunicorn
   gunicorn --bind 0.0.0.0:8080 "streamlit.web.bootstrap:run_with_command_line online_app.py"
   ```

5. **Access at:**

   ```
   http://your-instance-ip:8080
   ```

---

### **Option 5: PythonAnywhere (Free + Paid)**

1. **Create Account:** <https://www.pythonanywhere.com/>
2. **Upload Files:** Via web interface
3. **Create Web App:** Select Streamlit
4. **Configure:** Set main file to `online_app.py`
5. **Access:** yourusername.pythonanywhere.com

---

## 🎯 Features Overview

### 📚 **PyTorch Basics**

- ✅ Tensor creation & operations
- ✅ Neural network building
- ✅ Training simulations
- ✅ GPU support

### 🖼️ **Computer Vision**

- ✅ Image classification demo
- ✅ Image processing (blur, sharpen, edge detection)
- ✅ CNN architecture visualization
- ✅ Feature visualization

### 📊 **Data Tools**

- ✅ Dataset information viewer
- ✅ Data augmentation tools
- ✅ Statistical analysis
- ✅ Distribution visualization

### 🔧 **Model Tools**

- ✅ Model architecture explorer
- ✅ Training monitor
- ✅ Model export (PyTorch, ONNX, TensorFlow, TorchScript)
- ✅ ML Playground (Iris, Digits datasets)

### ⚡ **Advanced Tools**

- ✅ Batch processing pipeline
- ✅ Model comparison tools
- ✅ Feature engineering methods
- ✅ Hyperparameter tuning visualizer

### 📈 **Visualizations**

- ✅ Activation map viewer
- ✅ t-SNE feature space
- ✅ Learning curve monitor
- ✅ Interactive plots

### 📖 **Resources**

- ✅ 250+ course materials
- ✅ Quick start tutorials
- ✅ Complete documentation links
- ✅ Learning paths

---

## 🔧 Troubleshooting

### **Issue: Streamlit command not found**

```bash
python -m pip install --upgrade streamlit
```

### **Issue: PyTorch installation failing**

```bash
# For CPU only
pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html

# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Issue: Module not found errors**

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### **Issue: Port 8501 already in use**

```bash
streamlit run online_app.py --server.port 8502
```

### **Issue: Out of memory**

```bash
streamlit run online_app.py --logger.level=error --client.maxMessageSize=200
```

---

## 📊 Performance Optimization

### Local Development

```bash
streamlit run online_app.py --logger.level=warning --client.showErrorDetails=false
```

### Production Deployment

```bash
streamlit run online_app.py \
  --server.port=8080 \
  --server.address=0.0.0.0 \
  --logger.level=error \
  --client.maxMessageSize=200 \
  --server.maxUploadSize=200
```

---

## 📱 Access Methods

| Platform | URL | Status |
|----------|-----|--------|
| Local | <http://localhost:8501> | ✅ Active |
| Streamlit Cloud | <https://your-app.streamlit.app> | ✅ Recommended |
| Heroku | <https://your-app.herokuapp.com> | ⏳ Trial |
| Google Cloud | <https://your-region-your-project.run.app> | ✅ Free Tier |
| AWS EC2 | <http://your-ip:8080> | ⚙️ Setup Required |
| PythonAnywhere | <https://yourusername.pythonanywhere.com> | ✅ Free |

---

## 🚀 Next Steps

1. **Local Testing:** Run locally first to test all features
2. **GitHub Setup:** Push code to GitHub repository
3. **Choose Platform:** Select deployment platform (Streamlit Cloud recommended)
4. **Deploy:** Follow platform-specific instructions
5. **Share:** Share public URL with users
6. **Monitor:** Check logs and usage metrics
7. **Update:** Deploy updates automatically via GitHub push

---

## 📞 Support & Resources

- **Streamlit Docs:** <https://docs.streamlit.io>
- **PyTorch Docs:** <https://pytorch.org/docs>
- **GitHub Issues:** Create issues for bugs/features
- **Community:** Streamlit Discord & Reddit

---

**Version:** 1.0
**Last Updated:** 2024
**Status:** Production Ready ✅
