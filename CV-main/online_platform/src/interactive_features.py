"""
Interactive Features for Online Platform
"""
import numpy as np
import io
from typing import Any, List, Dict
import streamlit as st


class InteractiveFeatures:
    """Provides interactive demos and visualizations"""

    @staticmethod
    def pytorch_demo():
        """Interactive PyTorch demonstration"""
        st.subheader("🔬 PyTorch Interactive Demo")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Tensor Operations")
            tensor_size = st.slider("Tensor Size", 2, 100, 10)
            operation = st.selectbox(
                "Operation", ["Sum", "Mean", "Std", "Max"])

        try:
            import torch

            # Create random tensor
            tensor = torch.randn(tensor_size, tensor_size)

            if operation == "Sum":
                result = tensor.sum().item()
            elif operation == "Mean":
                result = tensor.mean().item()
            elif operation == "Std":
                result = tensor.std().item()
            else:
                result = tensor.max().item()

            with col2:
                st.write(f"### Result")
                st.metric("Output", f"{result:.4f}")
                st.write(f"Tensor shape: {tensor.shape}")
                st.write(f"Device: {tensor.device}")

        except ImportError:
            st.error("PyTorch not installed")

    @staticmethod
    def image_processing_demo():
        """Image processing demonstrations"""
        st.subheader("🖼️ Image Processing Demo")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Image Transforms")
            transform_type = st.selectbox(
                "Transform Type",
                ["Rotation", "Blur", "Brightness", "Contrast"]
            )

        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import numpy as np

            # Create a sample image
            img_array = np.random.randint(
                0, 256, (200, 200, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            with col2:
                if transform_type == "Rotation":
                    angle = st.slider("Rotation Angle", 0, 360, 45)
                    result_img = img.rotate(angle)

                elif transform_type == "Blur":
                    blur_radius = st.slider("Blur Radius", 1, 20, 5)
                    result_img = img.filter(
                        ImageFilter.GaussianBlur(radius=blur_radius))

                elif transform_type == "Brightness":
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
                    enhancer = ImageEnhance.Brightness(img)
                    result_img = enhancer.enhance(brightness)

                else:  # Contrast
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
                    enhancer = ImageEnhance.Contrast(img)
                    result_img = enhancer.enhance(contrast)

                st.image(
                    result_img, caption=f"{transform_type} Applied", use_column_width=True)

        except ImportError as e:
            st.error(f"Required library not installed: {e}")

    @staticmethod
    def model_visualization_demo():
        """Neural network visualization"""
        st.subheader("🧠 Neural Network Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Network Configuration")
            input_size = st.slider("Input Size", 1, 128, 10)
            hidden_size = st.slider("Hidden Size", 1, 256, 64)
            output_size = st.slider("Output Size", 1, 128, 10)
            layers = st.slider("Number of Layers", 1, 10, 3)

        with col2:
            st.write("### Network Summary")
            total_params = (input_size * hidden_size +
                            hidden_size * hidden_size * (layers - 2) +
                            hidden_size * output_size)

            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Layers", layers)
            st.metric("Input Size", input_size)
            st.metric("Output Size", output_size)

    @staticmethod
    def training_simulator():
        """Simulate model training"""
        st.subheader("📈 Training Simulator")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### Training Config")
            epochs = st.slider("Epochs", 1, 100, 20)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                value=0.001
            )
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256])

        with col2:
            st.write("### Loss Progression")

            # Simulate training curve
            import numpy as np
            import matplotlib.pyplot as plt

            epoch_range = np.arange(1, epochs + 1)
            # Simulate loss decay
            loss = 2.0 * np.exp(-epoch_range / (epochs / 4)) + \
                np.random.normal(0, 0.05, epochs)
            loss = np.maximum(loss, 0.1)  # Ensure positive loss

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epoch_range, loss, linewidth=2, color='#1f77b4')
            ax.fill_between(epoch_range, loss, alpha=0.3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Progression')
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.write(f"**Final Loss:** {loss[-1]:.4f}")
            st.write(f"**Learning Rate:** {learning_rate}")
            st.write(f"**Batch Size:** {batch_size}")

    @staticmethod
    def data_exploration_demo():
        """Data exploration tools"""
        st.subheader("📊 Data Exploration Tools")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Dataset Statistics")
            dataset_size = st.slider("Dataset Size", 100, 10000, 1000)
            num_features = st.slider("Number of Features", 2, 100, 10)

        with col2:
            st.write("### Statistics")
            st.metric("Total Samples", dataset_size)
            st.metric("Features", num_features)
            st.metric("Training Set (80%)", int(dataset_size * 0.8))
            st.metric("Test Set (20%)", int(dataset_size * 0.2))

        # Generate sample data
        import numpy as np
        import matplotlib.pyplot as plt

        data = np.random.randn(dataset_size, num_features)

        # Visualize feature distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Feature Distributions", fontsize=14, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < num_features:
                ax.hist(data[:, i], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'Feature {i+1}')
                ax.set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

    @staticmethod
    def gpu_info_demo():
        """GPU availability information"""
        st.subheader("⚡ GPU Information")

        try:
            import torch

            col1, col2 = st.columns(2)

            with col1:
                st.metric("PyTorch Version", torch.__version__)
                st.metric("CUDA Available", torch.cuda.is_available())

            with col2:
                if torch.cuda.is_available():
                    st.metric("GPU Count", torch.cuda.device_count())
                    st.metric("Current Device", torch.cuda.current_device())
                    st.metric("GPU Name", torch.cuda.get_device_name(0))
                else:
                    st.info("No GPU detected. Running on CPU.")

        except ImportError:
            st.warning("PyTorch not installed")
