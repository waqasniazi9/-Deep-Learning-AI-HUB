#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os

# Add the directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("Starting Streamlit app...", file=sys.stderr)


print("Streamlit imported successfully", file=sys.stderr)

st.set_page_config(page_title="Test", layout="wide")

st.title("AI Learning Hub - Test")
st.success("App is working!")
st.write("If you see this message, the app is running correctly.")

with st.sidebar:
    st.write("Sidebar is working")
    option = st.selectbox("Choose section:", ["Home", "PyTorch", "Vision"])

if option == "Home":
    st.header("Home Section")
    st.write("Welcome to the AI Learning Hub")
elif option == "PyTorch":
    st.header("PyTorch")
    st.code("import torch")
else:
    st.header("Computer Vision")
    st.write("CV tools coming soon")
