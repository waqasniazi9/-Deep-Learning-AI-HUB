import streamlit as st
import os
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

st.set_page_config(
    page_title="Deep Learning Notebook Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { padding: 2rem; }
        .notebook-card { 
            border: 1px solid #ddd; 
            padding: 1.5rem; 
            border-radius: 8px; 
            margin: 1rem 0;
            background-color: #f9f9f9;
            transition: all 0.3s ease;
        }
        .notebook-card:hover { 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background-color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Deep Learning Notebook Hub")
st.markdown("Interactive platform for running all deep learning tutorials online")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio(
        "Select View Mode:",
        ["📚 All Notebooks", "🔍 Search & Filter", "⚙️ Settings", "📊 Statistics"]
    )
    
    st.divider()
    
    category_filter = st.multiselect(
        "Filter by Category:",
        ["100 - PyTorch Basics", "200 - Deep Learning", "300 - Advanced Topics"],
        default=["100 - PyTorch Basics", "200 - Deep Learning", "300 - Advanced Topics"]
    )

# Get all notebook files
notebook_dir = Path(__file__).parent
notebooks = sorted(notebook_dir.glob("*.ipynb"))
notebooks = [nb for nb in notebooks if not nb.name.startswith(".")]

def categorize_notebook(filename):
    """Categorize notebook by its prefix number"""
    prefix = filename.split("_")[0]
    if prefix.startswith("1"):
        return "100 - PyTorch Basics"
    elif prefix.startswith("2"):
        return "200 - Deep Learning"
    elif prefix.startswith("3"):
        return "300 - Advanced Topics"
    return "Other"

# Filter notebooks by category
filtered_notebooks = [
    nb for nb in notebooks 
    if categorize_notebook(nb.name) in category_filter
]

if view_mode == "📚 All Notebooks":
    st.header("All Available Notebooks")
    st.markdown(f"**Total Notebooks: {len(filtered_notebooks)}**")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, notebook in enumerate(filtered_notebooks):
        notebook_name = notebook.stem
        category = categorize_notebook(notebook.name)
        
        col = [col1, col2, col3][idx % 3]
        
        with col:
            with st.container():
                st.markdown(f"### 📓 {notebook_name}")
                st.markdown(f"*Category: {category}*")
                
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if st.button("▶️ Run", key=f"run_{notebook.name}"):
                        st.info(f"Opening {notebook_name}...")
                        st.markdown(f"To run this notebook with Voila:\n```bash\nvoila {notebook.name}\n```")
                
                with col_btn2:
                    if st.button("📖 View", key=f"view_{notebook.name}"):
                        st.info(f"Notebook: {notebook.name}")
                
                with col_btn3:
                    if st.button("⬇️ Info", key=f"info_{notebook.name}"):
                        file_size = notebook.stat().st_size / 1024  # KB
                        st.info(f"Size: {file_size:.2f} KB")

elif view_mode == "🔍 Search & Filter":
    st.header("Search Notebooks")
    
    search_term = st.text_input("🔍 Search by name:", placeholder="e.g., CNN, LSTM, ResNet...")
    
    if search_term:
        results = [
            nb for nb in filtered_notebooks 
            if search_term.lower() in nb.name.lower()
        ]
        st.markdown(f"**Found {len(results)} notebooks matching '{search_term}'**")
        
        for notebook in results:
            st.markdown(f"- **{notebook.stem}**")
    else:
        st.info("Enter a search term to find notebooks")

elif view_mode == "⚙️ Settings":
    st.header("Settings & Configuration")
    
    st.subheader("1. Voila Configuration")
    st.markdown("""
    To run any notebook as a web app using Voila:
    ```bash
    pip install voila
    voila notebook_name.ipynb
    ```
    """)
    
    st.subheader("2. Streamlit Dashboard")
    st.markdown("""
    To run this dashboard locally:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
    
    st.subheader("3. Binder Configuration")
    st.markdown("""
    To deploy on Binder (free hosting):
    1. Push this repository to GitHub
    2. Visit [mybinder.org](https://mybinder.org)
    3. Enter your GitHub repo URL
    4. Click "Launch"
    """)
    
    st.subheader("4. System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")
    with col2:
        st.metric("Total Notebooks", len(notebooks))
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

elif view_mode == "📊 Statistics":
    st.header("Notebook Statistics")
    
    # Count by category
    categories = {}
    for nb in filtered_notebooks:
        cat = categorize_notebook(nb.name)
        categories[cat] = categories.get(cat, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Notebooks", len(filtered_notebooks))
    with col2:
        st.metric("Categories", len(set(categorize_notebook(nb.name) for nb in filtered_notebooks)))
    with col3:
        total_size = sum(nb.stat().st_size for nb in filtered_notebooks) / (1024 * 1024)
        st.metric("Total Size", f"{total_size:.2f} MB")
    
    st.subheader("Notebooks by Category")
    category_data = json.loads(json.dumps({k: v for k, v in sorted(categories.items())}))
    st.bar_chart(category_data)
    
    st.subheader("Top 10 Largest Notebooks")
    top_notebooks = sorted(
        filtered_notebooks, 
        key=lambda x: x.stat().st_size, 
        reverse=True
    )[:10]
    
    size_data = {nb.stem: nb.stat().st_size / 1024 for nb in top_notebooks}
    st.bar_chart(size_data)

# Footer
st.divider()
st.markdown("""
---
**Quick Start Guide:**
1. **Local Voila**: `voila notebook_name.ipynb` → Opens notebook as web app
2. **Streamlit Dashboard**: `streamlit run app.py` → This interface
3. **Jupyter Server**: `jupyter notebook` → Classic Jupyter interface
4. **Binder**: Push to GitHub and use [mybinder.org](https://mybinder.org) for free hosting

**Requirements:**
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
""")
