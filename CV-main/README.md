# Deep Learning Notebook Hub

A comprehensive online platform for running 130+ deep learning tutorial notebooks with multiple deployment options.

## 📊 Features

- ✅ **Streamlit Dashboard** - Interactive web interface to browse and manage all notebooks
- ✅ **Voila Integration** - Convert notebooks to standalone web apps
- ✅ **Binder Support** - Free cloud deployment via mybinder.org
- ✅ **Docker Support** - Run everything in containers with docker-compose
- ✅ **Jupyter Interface** - Classic notebook server
- ✅ **Search & Filter** - Find notebooks by category or keyword
- ✅ **Statistics** - View notebook metrics and storage usage

## 📚 Notebook Categories

- **100 Series (PyTorch Basics)**: Installation, data loading, tensor operations
- **200 Series (Deep Learning)**: Neural networks, CNNs, optimization, training
- **300 Series (Advanced Topics)**: Object detection, NLP, Transformers, Kaggle competitions

## 🚀 Quick Start

### Option 1: Streamlit Dashboard (Recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>

### Option 2: Voila Web Apps

```bash
voila notebook_name.ipynb
```

Converts any notebook into a standalone web app

### Option 3: Classic Jupyter

```bash
jupyter notebook
```

Opens at <http://localhost:8888>

### Option 4: Docker Compose

```bash
docker-compose up
```

Runs all services simultaneously:

- Streamlit: <http://localhost:8501>
- Jupyter: <http://localhost:8888>
- Voila: <http://localhost:8867>

## 📋 Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

## 🌐 Cloud Deployment

### Deploy to Binder (Free)

1. Push this repository to GitHub
2. Visit [mybinder.org](https://mybinder.org)
3. Enter your GitHub repo URL
4. Click "Launch"

### Deploy to Heroku

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Deploy to AWS/Google Cloud

- Use provided `Dockerfile` for containerized deployment
- Configure environment variables as needed

## 📁 File Structure

```
CV-main/
├── *.ipynb                    # All notebook files (130+ notebooks)
├── app.py                     # Streamlit dashboard app
├── HOME.ipynb                 # Voila home dashboard
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version for Binder
├── Dockerfile                # Docker image config
├── docker-compose.yml        # Multi-service orchestration
├── setup.sh                  # Linux/Mac setup script
├── setup.bat                 # Windows setup script
└── README.md                 # This file
```

## 🔧 Configuration

### Voila Settings

Edit the voila configuration in individual notebooks or create a `voila_config.py` file:

```python
c.VoilaConfiguration.template = 'gridstack'
c.VoilaConfiguration.resources = {'inlineJS': False}
```

### Streamlit Settings

Edit `~/.streamlit/config.toml` for custom Streamlit settings

### Docker Environment

Modify `docker-compose.yml` to adjust ports and environment variables

## 📊 Statistics

- **Total Notebooks**: 131
- **Total Size**: ~300+ MB
- **Categories**: 3 main series
- **Coverage**: PyTorch, CNN, RNN, NLP, Transformers, Object Detection, and more

## 🎓 Learning Path

1. Start with **100 Series** - Learn PyTorch basics
2. Progress to **200 Series** - Understand deep learning theory
3. Master **300 Series** - Advanced applications and competitions

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Change Streamlit port
streamlit run app.py --server.port 8502

# Change Jupyter port
jupyter notebook --port 8889
```

### Module Not Found

```bash
pip install -r requirements.txt --upgrade
```

### Docker Issues

```bash
docker-compose down
docker-compose up --build
```

## 📝 License

Educational content for deep learning learning. Based on popular deep learning courses.

## 🤝 Contributing

To add or modify notebooks:

1. Edit/create notebook files in the main directory
2. Notebooks are automatically picked up by the dashboard
3. Restart Streamlit to see changes

## 📧 Support

For issues with:

- **Notebooks**: Check individual notebook comments
- **Streamlit Dashboard**: Review `app.py`
- **Voila**: See `.voila.ipynb` configuration
- **Docker**: Check `docker-compose.yml`

---

**Happy Learning! 🧠**
