import streamlit as st

st.set_page_config(page_title="AI Hub", layout="wide")

st.title("AI & Deep Learning Hub")
st.write("This is a test page")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Topics", "250+")
with col2:
    st.metric("Tools", "50+")
with col3:
    st.metric("Access", "24/7")

st.divider()

with st.sidebar:
    st.title("Navigation")
    section = st.radio(
        "Choose:", ["Home", "PyTorch", "Vision", "Data", "Models"])

if section == "Home":
    st.header("Welcome")
    st.write("Select a section from the sidebar")
elif section == "PyTorch":
    st.header("PyTorch Basics")
    st.code("import torch\nx = torch.tensor([1, 2, 3])")
elif section == "Vision":
    st.header("Computer Vision")
    st.write("Image processing tools coming soon")
elif section == "Data":
    st.header("Data Tools")
    st.write("Data analysis features coming soon")
else:
    st.header("Model Tools")
    st.write("Model training features coming soon")
