# AI & Deep Learning Hub - Setup Instructions

## 🚀 Quick Start

### Local Installation

1. **Clone/Download the repository**

   ```bash
   cd CV-main 2
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app locally**

   ```bash
   streamlit run online_app.py
   ```

   The app will open at `http://localhost:8501`

---

## 🌐 Deploy Online (Global Access)

Choose one of these platforms for free global deployment:

### Option 1: Streamlit Cloud (EASIEST) ⭐

1. Push code to GitHub:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. Go to <https://streamlit.io/cloud>
3. Click "New app"
4. Connect your GitHub repo
5. Select `online_app.py` as the main file
6. Click "Deploy"
7. Your app is live! Share the URL globally

### Option 2: Hugging Face Spaces

1. Create account at <https://huggingface.co>
2. Create new Space (Streamlit)
3. Upload files
4. It auto-deploys!
5. Share the URL

### Option 3: Heroku

1. Create `Procfile`:

   ```
   web: streamlit run online_app.py
   ```

2. Create `.streamlit/config.toml`:

   ```
   [server]
   headless = true
   port = $PORT
   enableCORS = false
   ```

3. Deploy:

   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Option 4: Railway

1. Go to <https://railway.app>
2. Connect GitHub repo
3. Select `online_app.py` as start command
4. Deploy (free tier available)

---

## 📊 Features Available

✅ **PyTorch Basics** - Tensor operations, neural networks, training loops
✅ **Computer Vision** - Image classification, processing, CNN architecture viewer
✅ **Data Tools** - Dataset viewer, augmentation, statistics
✅ **Model Tools** - Training monitor, model export
✅ **Visualizations** - Activation maps, feature space, learning curves
✅ **Resources** - Complete course material links and learning paths

---

## 💡 Customization

Edit `online_app.py` to add:

- Your own models
- Real datasets
- Live training
- Model inference
- API integration

---

## 📞 Support

For issues or questions, refer to:

- [Streamlit Docs](https://docs.streamlit.io)
- [PyTorch Docs](https://pytorch.org/docs)

---

## 🎯 Next Steps

1. **Deploy to cloud** using one of the options above
2. **Share the link** with anyone globally
3. **Monitor usage** in your cloud platform dashboard
4. **Enhance features** based on user feedback

Enjoy! 🚀
