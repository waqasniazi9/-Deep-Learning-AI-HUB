#!/usr/bin/env python
"""Simple web app using Flask instead of Streamlit"""
from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Learning Hub</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .sidebar { float: left; width: 20%; padding-right: 20px; }
        .main { float: left; width: 75%; }
        .sidebar h3 { margin-top: 0; }
        .sidebar a { display: block; padding: 8px; margin: 5px 0; background: #007bff; color: white; text-decoration: none; border-radius: 3px; }
        .sidebar a:hover { background: #0056b3; }
        .metrics { display: flex; gap: 20px; margin: 20px 0; }
        .metric { flex: 1; background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI & Deep Learning Hub</h1>
        <p>Welcome! This is a comprehensive platform for learning AI and deep learning.</p>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">250+</div>
                <div class="metric-label">Topics</div>
            </div>
            <div class="metric">
                <div class="metric-value">50+</div>
                <div class="metric-label">Tools</div>
            </div>
            <div class="metric">
                <div class="metric-value">24/7</div>
                <div class="metric-label">Access</div>
            </div>
        </div>

        <div class="sidebar">
            <h3>Navigation</h3>
            <a href="#home">Home</a>
            <a href="#pytorch">PyTorch Basics</a>
            <a href="#vision">Computer Vision</a>
            <a href="#data">Data Tools</a>
            <a href="#models">Model Tools</a>
            <a href="#advanced">Advanced Tools</a>
            <a href="#viz">Visualizations</a>
            <a href="#resources">Resources</a>
        </div>

        <div class="main">
            <h2>Welcome to AI Learning Hub</h2>
            <p>This is a comprehensive platform for learning and experimenting with:</p>
            <ul>
                <li><strong>PyTorch</strong> - Fundamentals and advanced techniques</li>
                <li><strong>Computer Vision</strong> - CNN, Object Detection, Segmentation</li>
                <li><strong>Deep Learning</strong> - Theory and practice</li>
                <li><strong>Data Processing</strong> - Analysis and visualization</li>
                <li><strong>Model Training</strong> - Building and deploying models</li>
            </ul>
            <h3>Choose a section from the navigation to get started!</h3>
        </div>

        <div style="clear: both;"></div>
        <hr style="margin-top: 40px;">
        <p style="text-align: center; color: #999; font-size: 12px;">
            Global AI Learning Platform | Accessible 24/7 | Free & Open Source
        </p>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


if __name__ == '__main__':
    print("Starting AI Learning Hub on http://localhost:8501")
    app.run(host='127.0.0.1', port=8501, debug=False)
