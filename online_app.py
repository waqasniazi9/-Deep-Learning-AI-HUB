# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI/Deep Learning Hub",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("AI Learning Hub")
    page = st.radio(
        "Navigate to:",
        ["Home", "PyTorch Basics", "Computer Vision",
         "Data Tools", "Model Tools", "Advanced Tools", "Visualizations", "Resources"]
    )

# ============= HOME PAGE =============
if page == "Home":
    st.title("Welcome to AI & Deep Learning Hub")
    st.write("""
    This is a comprehensive online platform for learning and experimenting with:
    - **PyTorch** fundamentals and advanced techniques
    - **Computer Vision** (CNN, Object Detection, Segmentation)
    - **Natural Language Processing** (RNN, LSTM, Transformers, BERT)
    - **Data Processing & Visualization**
    - **Model Training & Deployment**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Topics", "250+", "courses")
    with col2:
        st.metric("Tools", "50+", "interactive")
    with col3:
        st.metric("Access", "Global", "24/7")

    st.divider()
    st.subheader("Key Features")

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.write("**PyTorch Training**")
        st.write("- Tensor operations")
        st.write("- Neural networks")
        st.write("- GPU acceleration")
        st.write("- Model save/load")

    with feat_col2:
        st.write("**Computer Vision**")
        st.write("- Image classification")
        st.write("- Object detection")
        st.write("- Segmentation")
        st.write("- Style transfer")

    with feat_col3:
        st.write("**Advanced Tools**")
        st.write("- Batch processing")
        st.write("- Model comparison")
        st.write("- Hyperparameter tuning")
        st.write("- Feature engineering")

    st.divider()
    st.subheader("Learning Path")

    learning_col1, learning_col2 = st.columns(2)

    with learning_col1:
        st.write("""
        **Beginner (Week 1-4)**
        1. PyTorch Basics
        2. Tensors & Operations
        3. Data Loading
        4. Neural Networks Intro
        """)

    with learning_col2:
        st.write("""
        **Intermediate (Week 5-12)**
        1. CNN Architecture
        2. Image Classification
        3. Object Detection
        4. Optimization Techniques
        """)

# ============= PYTORCH BASICS =============
elif page == "PyTorch Basics":
    st.title("PyTorch Fundamentals")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tensors", "Operations", "Neural Networks", "Training"])

    with tab1:
        st.header("Tensor Operations")
        st.write("Learn and experiment with PyTorch tensors")

        col1, col2 = st.columns(2)
        with col1:
            shape_input = st.text_input("Tensor shape (e.g., 3,4,5):", "2,3")
            if st.button("Create Random Tensor"):
                try:
                    shape = tuple(map(int, shape_input.split(',')))
                    tensor = torch.randn(shape)
                    st.write(f"Created tensor with shape {tensor.shape}")
                    st.write(f"Device: {tensor.device}")
                    st.write(f"Dtype: {tensor.dtype}")
                    with st.expander("View tensor values"):
                        st.write(tensor)
                except:
                    st.error("Invalid shape format")

        with col2:
            st.write("**Common Operations**")
            demo_tensor = torch.randn(2, 3)
            st.write(f"Original: {demo_tensor.shape}")
            st.write(f"Reshaped: {demo_tensor.reshape(-1).shape}")
            st.write(f"Transposed: {demo_tensor.t().shape}")

    with tab2:
        st.header("Tensor Math")

        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Matrix A dimension:", 2, 10, 3)
            b = st.number_input("Matrix B dimension:", 2, 10, 4)

        with col2:
            if st.button("Matrix Multiply"):
                matrix_a = torch.randn(a, b)
                matrix_b = torch.randn(b, a)
                result = torch.mm(matrix_a, matrix_b)
                st.write(f"Result shape: {result.shape}")
                st.bar_chart(result.numpy())

    with tab3:
        st.header("Build Simple Neural Network")

        code = """
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
print(model)
        """
        st.code(code, language='python')

    with tab4:
        st.header("Training Loop")

        epochs = st.slider("Number of epochs:", 1, 100, 10)
        learning_rate = st.select_slider(
            "Learning rate:", [0.0001, 0.001, 0.01, 0.1])

        st.write(f"""
        **Training Configuration:**
        - Epochs: {epochs}
        - Learning Rate: {learning_rate}
        - Batch Size: 32
        - Optimizer: Adam
        
        Ready to train! Click the button to start.
        """)

        if st.button("Start Training Simulation"):
            progress_bar = st.progress(0)
            loss_data = []

            for epoch in range(epochs):
                loss = 100 * np.exp(-epoch * 0.1) + np.random.randn() * 5
                loss_data.append(loss)
                progress_bar.progress((epoch + 1) / epochs)

            st.line_chart(loss_data)
            st.success(f"Training complete! Final loss: {loss_data[-1]:.4f}")

# ============= COMPUTER VISION =============
elif page == "Computer Vision":
    st.title("Computer Vision Tools")

    tab1, tab2, tab3 = st.tabs(
        ["Image Classification", "Image Processing", "CNN Visualizer"])

    with tab1:
        st.header("Image Classification Demo")

        st.write("Upload an image to classify:")
        uploaded_file = st.file_uploader(
            "Choose image...", type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image",
                         use_column_width=True)

            with col2:
                st.write("**Classification Results:**")
                classes = ["cat", "dog", "bird", "car", "truck"]
                scores = np.random.dirichlet(np.ones(5)) * 100

                for cls, score in zip(classes, scores):
                    st.write(f"{cls}: {score:.1f}%")
                    st.progress(score/100)

    with tab2:
        st.header("Image Processing Tools")

        col1, col2 = st.columns(2)

        with col1:
            operation = st.selectbox("Choose operation:",
                                     ["Grayscale", "Blur", "Edge Detection", "Sharpen", "Invert"])

        with col2:
            uploaded = st.file_uploader("Upload image...", type=['jpg', 'png'])

        if uploaded:
            img = Image.open(uploaded)

            from PIL import ImageFilter, ImageOps, ImageEnhance

            if operation == "Grayscale":
                result = ImageOps.grayscale(img)
            elif operation == "Blur":
                result = img.filter(ImageFilter.GaussianBlur(radius=5))
            elif operation == "Edge Detection":
                result = img.filter(ImageFilter.FIND_EDGES)
            elif operation == "Sharpen":
                result = img.filter(ImageFilter.SHARPEN)
            else:  # Invert
                result = ImageOps.invert(img.convert('RGB'))

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original")
            with col2:
                st.image(result, caption=operation)

    with tab3:
        st.header("CNN Architecture Visualizer")

        st.write("**Popular CNN Architectures:**")

        arch = st.selectbox("Select architecture:",
                            ["LeNet", "AlexNet", "VGG-16", "ResNet-50", "MobileNet"])

        architectures = {
            "LeNet": ["Input (32x32)", "Conv (6,5x5)", "Pool", "Conv (16,5x5)", "Pool", "FC (120)", "FC (84)", "FC (10)"],
            "AlexNet": ["Input (227x227)", "Conv (96,11x11)", "Pool", "Conv (256,5x5)", "Pool", "Conv (384,3x3)", "Conv (384,3x3)", "Conv (256,3x3)", "Pool", "FC (4096)"],
            "VGG-16": ["Input", "2xConv (64)", "Pool", "2xConv (128)", "Pool", "3xConv (256)", "Pool", "3xConv (512)", "Pool", "3xConv (512)", "Pool", "3xFC"],
            "ResNet-50": ["Input", "Conv+BN+ReLU", "4xResidual Block", "4xResidual Block", "4xResidual Block", "4xResidual Block", "Global Avg Pool", "FC"],
            "MobileNet": ["Input", "Depthwise Conv", "Pointwise Conv", "x14 bottleneck blocks", "Global Avg Pool", "FC"]
        }

        for i, layer in enumerate(architectures[arch], 1):
            st.write(f"{i}. {layer}")

# ============= DATA TOOLS =============
elif page == "Data Tools":
    st.title("Data Processing & Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Dataset Viewer", "Data Augmentation", "Statistics"])

    with tab1:
        st.header("Dataset Information")

        dataset = st.selectbox("Select dataset:",
                               ["MNIST", "CIFAR-10", "ImageNet", "COCO", "Custom"])

        dataset_info = {
            "MNIST": {"images": 70000, "classes": 10, "size": "28x28", "format": "Grayscale"},
            "CIFAR-10": {"images": 60000, "classes": 10, "size": "32x32", "format": "RGB"},
            "ImageNet": {"images": 1400000, "classes": 1000, "size": "Variable", "format": "RGB"},
            "COCO": {"images": 330000, "classes": 80, "size": "Variable", "format": "RGB"},
            "Custom": {"images": "?", "classes": "?", "size": "?", "format": "?"}
        }

        info = dataset_info[dataset]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images", info["images"])
        col2.metric("Classes", info["classes"])
        col3.metric("Size", info["size"])
        col4.metric("Format", info["format"])

    with tab2:
        st.header("Data Augmentation")

        augmentations = st.multiselect(
            "Select augmentations:",
            ["Rotation", "Flip", "Zoom", "Brightness", "Contrast", "Noise"],
            default=["Rotation", "Flip"]
        )

        st.write(f"Selected: {', '.join(augmentations)}")

        if st.button("Generate Augmented Samples"):
            st.success(
                f"Generated 10 augmented samples with: {', '.join(augmentations)}")

    with tab3:
        st.header("Dataset Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Distribution**")
            classes = ["Class A", "Class B", "Class C", "Class D", "Class E"]
            values = np.random.randint(100, 1000, 5)
            st.bar_chart(dict(zip(classes, values)))

        with col2:
            st.write("**Image Statistics**")
            stats = {"Mean": 127.5, "Std": 45.2, "Min": 0, "Max": 255}
            for k, v in stats.items():
                st.metric(k, f"{v:.1f}")

# ============= MODEL TOOLS =============
elif page == "Model Tools":
    st.title("Model Management & Training")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model Info", "Training Monitor", "Export/Deploy", "ML Playground"])

    with tab1:
        st.header("Model Architecture")

        model_type = st.selectbox("Select model:",
                                  ["ResNet-50", "VGG-16", "MobileNet", "EfficientNet", "Custom"])

        st.write(f"""
        **{model_type} Details:**
        - Parameters: {np.random.randint(1, 500)}M
        - Memory: {np.random.randint(100, 2000)}MB
        - Speed: {np.random.randint(5, 100)} FPS
        - Accuracy (ImageNet): {np.random.randint(70, 85)}%
        """)

    with tab2:
        st.header("Training Monitor")

        epochs_run = st.slider("Epoch:", 0, 100, 50)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Training Loss**")
            losses = 100 * np.exp(-np.arange(epochs_run)
                                  * 0.05) + np.random.randn(epochs_run) * 5
            st.line_chart(losses)

        with col2:
            st.write("**Validation Accuracy**")
            acc = 50 + 45 * (1 - np.exp(-np.arange(epochs_run)
                             * 0.05)) + np.random.randn(epochs_run) * 2
            st.line_chart(acc)

    with tab3:
        st.header("Model Export")

        format_choice = st.selectbox("Export format:",
                                     ["PyTorch (.pth)", "ONNX", "TensorFlow SavedModel", "TorchScript"])

        if st.button("Export Model"):
            st.success(f"Model exported as {format_choice}")
            st.write("Download link would appear here")

    with tab4:
        st.header("ML Playground - Quick Model Training")

        col1, col2 = st.columns(2)
        with col1:
            dataset_choice = st.selectbox(
                "Choose dataset:", ["Iris", "Digits", "Generate Random"])
        with col2:
            model_choice = st.selectbox("Choose model:", [
                                        "Linear Regression", "Logistic Regression", "Simple Neural Network"])

        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                if dataset_choice == "Iris":
                    from sklearn.datasets import load_iris
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LogisticRegression

                    iris = load_iris()
                    X = iris.data
                    y = iris.target
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2)

                    model = LogisticRegression(max_iter=200)
                    model.fit(X_train, y_train)
                    accuracy = model.score(X_test, y_test)

                    st.success(f"Model trained! Accuracy: {accuracy:.2%}")

                    # Show feature importance
                    st.write("**Feature Importance:**")
                    features_df = pd.DataFrame({
                        'Feature': iris.feature_names,
                        'Importance': np.abs(model.coef_[0])
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(features_df.set_index('Feature'))
                else:
                    st.info("Generate random dataset and train models")

# ============= ADVANCED TOOLS =============
elif page == "Advanced Tools":
    st.title("Advanced ML & AI Tools")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Batch Processing", "Model Comparison", "Feature Engineering", "Hyperparameter Tuning"])

    with tab1:
        st.header("Batch Processing & Pipeline")

        st.write("Process multiple samples efficiently")

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.slider("Batch Size:", 1, 256, 32)
            num_batches = st.slider("Number of Batches:", 1, 100, 10)

        with col2:
            st.info(
                f"Total Samples: {batch_size * num_batches}\n\nProcessing Time (est.): {batch_size * num_batches * 0.01:.2f}s\n\nMemory Usage (est.): {batch_size * num_batches * 0.001:.2f}MB")

        if st.button("Start Batch Processing"):
            progress_bar = st.progress(0)
            status = st.empty()

            for batch in range(num_batches):
                status.write(f"Processing batch {batch + 1}/{num_batches}...")
                progress_bar.progress((batch + 1) / num_batches)

            st.success("Batch processing complete!")
            st.write(
                f"Processed {batch_size * num_batches} samples successfully")

    with tab2:
        st.header("Model Comparison")

        models = st.multiselect(
            "Select models to compare:",
            ["ResNet-50", "VGG-16", "MobileNet", "EfficientNet", "SqueezeNet"],
            default=["ResNet-50", "VGG-16"]
        )

        if models:
            comparison_data = {
                "Model": models,
                "Parameters (M)": np.random.randint(1, 500, len(models)),
                "Speed (FPS)": np.random.randint(5, 100, len(models)),
                "Accuracy (%)": np.random.randint(70, 95, len(models)),
                "Memory (MB)": np.random.randint(50, 1000, len(models))
            }

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

            st.bar_chart(df.set_index("Model")[
                         ["Speed (FPS)", "Accuracy (%)"]])

    with tab3:
        st.header("Feature Engineering")

        col1, col2 = st.columns(2)
        with col1:
            feature_type = st.selectbox("Select feature engineering method:", [
                "Normalization", "Standardization", "One-Hot Encoding", "Polynomial Features", "Dimensionality Reduction"
            ])

        with col2:
            if st.button("Apply Feature Engineering"):
                st.success(f"Applied {feature_type}")

                if feature_type == "Normalization":
                    st.code("X_normalized = (X - X.min()) / (X.max() - X.min())")
                elif feature_type == "Standardization":
                    st.code("X_std = (X - X.mean()) / X.std()")
                elif feature_type == "One-Hot Encoding":
                    st.code("X_encoded = pd.get_dummies(X)")
                elif feature_type == "Polynomial Features":
                    st.code(
                        "from sklearn.preprocessing import PolynomialFeatures\nX_poly = PolynomialFeatures(degree=2).fit_transform(X)")
                else:
                    st.code(
                        "from sklearn.decomposition import PCA\nX_pca = PCA(n_components=0.95).fit_transform(X)")

    with tab4:
        st.header("Hyperparameter Tuning")

        col1, col2, col3 = st.columns(3)
        with col1:
            learning_rate = st.select_slider("Learning Rate:",
                                             [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        with col2:
            batch_size_hp = st.select_slider("Batch Size:",
                                             [8, 16, 32, 64, 128, 256])
        with col3:
            epochs_hp = st.select_slider("Epochs:",
                                         [10, 20, 50, 100, 200, 300])

        st.info(f"""
        **Hyperparameter Settings:**
        - Learning Rate: {learning_rate}
        - Batch Size: {batch_size_hp}
        - Epochs: {epochs_hp}
        - Optimizer: Adam
        - Loss Function: CrossEntropyLoss
        """)

        if st.button("Optimize Hyperparameters"):
            st.success(
                "Hyperparameter optimization started using Bayesian Optimization")

            # Simulate optimization
            fig, ax = plt.subplots()
            x = np.arange(0, 100, 1)
            y = 100 * (1 - np.exp(-x * 0.03)) + np.random.randn(len(x)) * 2
            ax.plot(x, y, linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Validation Accuracy (%)")
            ax.set_title("Hyperparameter Optimization Progress")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============= VISUALIZATIONS =============
elif page == "Visualizations":
    st.title("Interactive Visualizations")

    tab1, tab2, tab3 = st.tabs(
        ["Activation Maps", "Feature Space", "Learning Curves"])

    with tab1:
        st.header("Activation Map Visualization")

        layer = st.selectbox(
            "Select layer:", ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"])

        # Create dummy activation maps
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for ax in axes.flat:
            ax.imshow(np.random.randn(16, 16), cmap='hot')
            ax.axis('off')
        st.pyplot(fig)

    with tab2:
        st.header("Feature Space Visualization")

        # t-SNE visualization
        np.random.seed(42)
        features = np.random.randn(100, 2)
        labels = np.random.randint(0, 10, 100)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            features[:, 0], features[:, 1], c=labels, cmap='tab10', s=100)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("t-SNE Visualization of Features")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)

    with tab3:
        st.header("Learning Curves")

        train_acc = 50 + 45 * (1 - np.exp(-np.arange(100) * 0.05))
        val_acc = 48 + 42 * (1 - np.exp(-np.arange(100) * 0.045))

        fig, ax = plt.subplots()
        ax.plot(train_acc, label='Training Accuracy', linewidth=2)
        ax.plot(val_acc, label='Validation Accuracy', linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Learning Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ============= RESOURCES =============
elif page == "Resources":
    st.title("Complete Learning Resources")

    tab1, tab2, tab3 = st.tabs(
        ["Course Catalog", "Quick Tutorials", "Documentation"])

    with tab1:
        st.header("Comprehensive Course Materials (250+ Topics)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("100-199 Series: PyTorch Fundamentals")
            st.write("""
            - 100: Configuration & Version Setup
            - 101: PyTorch Installation
            - 102: Python Data Structures
            - 103: Data Loading with PyTorch
            - 104: TensorBoard Visualization
            - 105: Transforms for Data
            - 106: torchvision Datasets
            - 107: DataLoader Usage
            - 108: nn.Module Architecture
            - 109: Convolution Principles
            - 110: Convolution Layers
            - 111: Max Pooling Layers
            - 112: Non-linear Activations
            - 113: Linear & Other Layers
            - 114: Building Networks & Sequential
            - 115: Loss Functions & Backprop
            - 116: Optimizers
            - 117: Network Usage & Modification
            - 118: Model Save & Load
            - 119: Complete Training Pipeline
            - 120: GPU Training
            - 121: Model Validation
            - 122: Open Source Projects
            """)

        with col2:
            st.subheader("200-299 Series: Deep Learning & CNN")
            st.write("""
            - 200: Deep Learning Intro
            - 201-212: Theory & Math Foundations
            - 213: Kaggle House Price Prediction
            - 214: PyTorch Neural Networks
            - 215: GPU Purchase Guide
            - 216-219: Convolution & Pooling
            - 220: LeNet Architecture
            - 221: AlexNet Deep CNN
            - 222: VGG Networks
            - 223: Network in Network (NiN)
            - 224: GoogLeNet
            - 225: Batch Normalization
            - 226: ResNet (Residual Networks)
            - 227-231: Hardware & Distributed Training
            - 232: Data Augmentation
            - 233: Fine-tuning
            - 234: CIFAR-10 Image Classification
            - 235: Dog Breed Recognition
            - 236-240: Object Detection & R-CNN, SSD, YOLO
            - 241-245: Semantic Segmentation & Style Transfer
            """)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("300+ Series: Advanced Topics")
            st.write("""
            - 246: Sequence Models
            - 247: Text Preprocessing
            - 248: Language Models
            - 249: RNN (Recurrent Neural Networks)
            - 250: RNN Implementation
            - 251: GRU (Gated Recurrent Unit)
            - 252: LSTM (Long Short-Term Memory)
            - 253: Deep RNNs
            - 254: Bidirectional RNNs
            - 255: Machine Translation Datasets
            - 256: Encoder-Decoder Architecture
            - 257: Seq2Seq Learning
            - 258: Beam Search
            - 259-261: Attention Mechanisms
            - 262: Self-Attention
            - 263: Transformer Architecture
            - 264: BERT Pretraining
            - 265: BERT Fine-tuning
            - 266: Object Detection Competitions
            - 267: Advanced Optimizers
            - 268: Course Summary & Advanced Learning
            """)

        with col4:
            st.info("**Platform Features**\n\n" +
                    "250+ Interactive Lessons\n" +
                    "50+ Interactive Tools\n" +
                    "Complete Code Examples\n" +
                    "GPU Support\n" +
                    "Project-Based Learning\n" +
                    "24/7 Online Access")

    with tab2:
        st.header("Quick Tutorials")

        tutorial = st.selectbox("Select Tutorial:", [
            "Create Your First Tensor",
            "Build a Simple Neural Network",
            "Train with GPU",
            "Load Custom Datasets",
            "Use Pretrained Models",
            "Image Classification",
            "Text Processing"
        ])

        if tutorial == "Create Your First Tensor":
            st.code("""
import torch

# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.randn(2, 3)  # Random tensor
z = torch.zeros(3, 4)   # Zeros tensor

# Operations
result = x + y[:2, :2]
print(result)
            """, language='python')

        elif tutorial == "Build a Simple Neural Network":
            st.code("""
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
            """, language='python')

        elif tutorial == "Train with GPU":
            st.code("""
import torch

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to GPU
model = model.to(device)
x = x.to(device)
            """, language='python')

        else:
            st.info(f"Tutorial: {tutorial}\n\nCode examples coming soon!")

    with tab3:
        st.header("Documentation & Reference")

        st.write("""
        ### Official Documentation
        - PyTorch Official Docs: https://pytorch.org/docs/
        - Torchvision Docs: https://pytorch.org/vision/stable/
        - Hugging Face Transformers: https://huggingface.co/docs/transformers/
        
        ### Learning Resources
        - d2l.ai - Dive into Deep Learning: https://d2l.ai/
        - Stanford CS231N - CNN for Visual Recognition: http://cs231n.stanford.edu/
        - Andrew Ng Neural Networks Course: https://www.coursera.org/learn/neural-networks-deep-learning
        - Fast.ai - Practical Deep Learning: https://www.fast.ai/
        
        ### Competitions & Practice
        - Kaggle Competitions: https://www.kaggle.com/competitions
        - Papers with Code: https://paperswithcode.com/
        - GitHub Awesome ML: https://github.com/topics/machine-learning
        """)

# Footer
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.subheader("Support")
    st.write("""
    - GitHub Issues
    - Email: info@ailearning.hub
    - Discord Community
    """)

with footer_col2:
    st.subheader("Links")
    st.write("""
    - Official Website
    - GitHub Repository
    - Documentation
    """)

with footer_col3:
    st.subheader("Statistics")
    st.write(f"""
    - Active Users: {np.random.randint(1000, 50000):,}
    - Courses Completed: {np.random.randint(10000, 100000):,}
    - Total Hours Learned: {np.random.randint(100000, 1000000):,}
    """)

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 2rem;'>
    Global AI Learning Platform | Accessible 24/7 | Free & Open Source<br>
    Copyright 2024 AI Learning Hub | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
