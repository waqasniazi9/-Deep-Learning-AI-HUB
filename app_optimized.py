"""
Waqas Khan Niazi AI & Deep Learning Hub - Production App
Optimized Flask application with modular architecture.
"""

from flask import Flask, render_template_string, request, jsonify
import logging
from datetime import datetime
import os
import sys

# Import configuration
from config import (
    DEBUG, HOST, PORT, SECRET_KEY, MAX_CONTENT_LENGTH,
    LOG_LEVEL, LOG_FILE, DEVICE, BATCH_SIZE, LEARNING_RATE,
    CACHE_ENABLED, TOPICS_CACHE_TTL, API_PREFIX
)

# Import utilities and modules
from utils import (
    setup_logging, memoize_with_ttl, get_cache_stats, format_response,
    handle_errors, paginate, time_operation
)
from models import get_model, SimpleNN
from data_handler import DataManager
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

# ==================== Flask App Setup ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = SECRET_KEY

# Setup logging
logger = setup_logging(sys.modules['config'])

# Initialize data manager
data_manager = DataManager(batch_size=BATCH_SIZE)

logger.info("=" * 60)
logger.info("Waqas Khan Niazi AI & Deep Learning Hub")
logger.info(f"Starting server on {HOST}:{PORT}")
logger.info(f"Device: {DEVICE}")
logger.info("=" * 60)


# ==================== Topics Database with Caching ====================
@memoize_with_ttl(ttl_seconds=TOPICS_CACHE_TTL)
def get_topics_database():
    """Cached topics database initialization."""
    return {
        100: {'id': 100, 'title': 'Configuration & Version Setup', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Beginner', 'duration': '10 min', 'content': 'Setup PyTorch environment'},
        101: {'id': 101, 'title': 'PyTorch Installation', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Beginner', 'duration': '15 min', 'content': 'Install PyTorch framework'},
        102: {'id': 102, 'title': 'Python Data Structures', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Beginner', 'duration': '20 min', 'content': 'Master lists, dicts, tuples'},
        103: {'id': 103, 'title': 'Data Loading with PyTorch', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Beginner', 'duration': '25 min', 'content': 'Use Dataset and DataLoader'},
        104: {'id': 104, 'title': 'TensorBoard Visualization', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Intermediate', 'duration': '30 min', 'content': 'Visualize training metrics'},
        105: {'id': 105, 'title': 'Transforms Usage', 'series': '100-199: PyTorch Fundamentals',
              'difficulty': 'Beginner', 'duration': '20 min', 'content': 'Image augmentation techniques'},
        200: {'id': 200, 'title': 'Deep Learning Introduction', 'series': '200-299: Deep Learning',
              'difficulty': 'Beginner', 'duration': '40 min', 'content': 'Fundamentals of deep learning'},
        201: {'id': 201, 'title': 'Convolutional Neural Networks', 'series': '200-299: Deep Learning',
              'difficulty': 'Intermediate', 'duration': '45 min', 'content': 'CNN architecture and training'},
        263: {'id': 263, 'title': 'Transformer Architecture', 'series': '200-299: Deep Learning',
              'difficulty': 'Advanced', 'duration': '60 min', 'content': 'Self-attention mechanisms'},
    }


# ==================== API Endpoints ====================

@app.route('/', methods=['GET'])
def home():
    """Serve the main application HTML."""
    html = render_template_string(get_html_template())
    return html, 200


@app.route(f'{API_PREFIX}/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return format_response({'status': 'healthy', 'device': DEVICE}, 'success'), 200


@app.route(f'{API_PREFIX}/topics', methods=['GET'])
@handle_errors(default_return=format_response([], 'error'))
def get_all_topics():
    """Get all topics with optional filtering."""
    topics_db = get_topics_database()
    difficulty = request.args.get('difficulty')
    series = request.args.get('series')
    page = int(request.args.get('page', 1))

    topics = list(topics_db.values())

    if difficulty:
        topics = [t for t in topics if t['difficulty'].lower() ==
                  difficulty.lower()]
    if series:
        topics = [t for t in topics if series in t['series']]

    paginated = paginate(topics, page=page, per_page=20)
    return format_response(paginated, 'success'), 200


@app.route(f'{API_PREFIX}/topic/<int:topic_id>', methods=['GET'])
@handle_errors(default_return=format_response({}, 'error'))
def get_topic(topic_id):
    """Get specific topic details."""
    topics_db = get_topics_database()
    topic = topics_db.get(topic_id)

    if not topic:
        return format_response({}, 'error', f'Topic {topic_id} not found'), 404

    return format_response(topic, 'success'), 200


@app.route(f'{API_PREFIX}/search', methods=['GET'])
@handle_errors(default_return=format_response([], 'error'))
def search_topics():
    """Search topics by keyword."""
    query = request.args.get('q', '').lower()
    topics_db = get_topics_database()

    results = [t for t in topics_db.values()
               if query in t['title'].lower() or query in t['content'].lower()]

    return format_response(results, 'success'), 200


@app.route(f'{API_PREFIX}/stats', methods=['GET'])
def get_stats():
    """Get platform statistics."""
    topics_db = get_topics_database()
    topics = list(topics_db.values())

    stats = {
        'total_topics': len(topics),
        'difficulty_distribution': {
            'beginner': len([t for t in topics if t['difficulty'] == 'Beginner']),
            'intermediate': len([t for t in topics if t['difficulty'] == 'Intermediate']),
            'advanced': len([t for t in topics if t['difficulty'] == 'Advanced'])
        },
        'series_count': len(set(t['series'] for t in topics)),
        'cache_stats': get_cache_stats() if CACHE_ENABLED else None,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat()
    }

    return format_response(stats, 'success'), 200


@app.route(f'{API_PREFIX}/train', methods=['POST'])
@handle_errors()
@time_operation
def train_model():
    """Train a neural network model."""
    try:
        data = request.json or {}
        dataset_name = data.get('dataset', 'iris')
        model_name = data.get('model', 'SimpleNN')
        epochs = int(data.get('epochs', 10))
        learning_rate = float(data.get('learning_rate', LEARNING_RATE))

        # Load data
        logger.info(f"Loading {dataset_name} dataset")
        dataset = data_manager.get_dataset(dataset_name, split=True)

        # Initialize model
        input_size = dataset.get('input_size', 20)
        num_classes = dataset.get('num_classes', 2)

        device = torch.device(DEVICE)
        model = SimpleNN(input_size=input_size,
                         num_classes=num_classes).to(device)

        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']

        # Training loop
        history = {'loss': [], 'accuracy': []}
        best_accuracy = 0

        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            accuracy = 100 * correct / total
            history['loss'].append(total_loss / len(train_loader))
            history['accuracy'].append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f}, Acc: {accuracy:.2f}%")

        return format_response({
            'model': model_name,
            'dataset': dataset_name,
            'epochs': epochs,
            'best_accuracy': float(best_accuracy),
            'history': {
                'loss': [float(l) for l in history['loss']],
                'accuracy': [float(a) for a in history['accuracy']]
            },
            'training_time': f"{time.time():.2f}s"
        }, 'success'), 200

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        return format_response({'error': str(e)}, 'error'), 400


@app.route(f'{API_PREFIX}/train_sklearn', methods=['POST'])
@handle_errors()
def train_sklearn_model():
    """Train scikit-learn model for comparison."""
    try:
        data = request.json or {}
        algo = data.get('algorithm', 'RandomForest')

        # Load iris dataset
        iris_data = data_manager.get_iris(split=False)
        X, y = iris_data['X'], iris_data['y']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        start = time.time()

        if algo == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=200)

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        elapsed = time.time() - start

        return format_response({
            'algorithm': algo,
            'accuracy': float(accuracy),
            'training_time': f"{elapsed:.4f}s",
            'test_samples': len(X_test)
        }, 'success'), 200

    except Exception as e:
        logger.error(f"sklearn training error: {str(e)}", exc_info=True)
        return format_response({'error': str(e)}, 'error'), 400


@app.route(f'{API_PREFIX}/cache/clear', methods=['POST'])
def clear_cache_route():
    """Clear application cache."""
    from utils import clear_cache
    clear_cache()
    logger.info("Cache cleared via API")
    return format_response({'message': 'Cache cleared'}, 'success'), 200


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return format_response({}, 'error', 'Resource not found'), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return format_response({}, 'error', 'Internal server error'), 500


# ==================== HTML Template ====================

def get_html_template():
    """Return the main HTML template."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waqas Khan Niazi AI & Deep Learning Hub</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; background: #f5f5f5; color: #333; }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
                 padding: 40px 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { font-size: 1.1em; opacity: 0.9; }
        
        nav { background: white; padding: 15px; border-bottom: 2px solid #667eea; 
              display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }
        nav button { padding: 10px 20px; border: none; cursor: pointer; border-radius: 5px; 
                     background: #667eea; color: white; font-size: 16px; transition: 0.3s; }
        nav button:hover { background: #764ba2; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        
        .section { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                   margin-bottom: 30px; }
        
        h2 { color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        
        input, select, button { padding: 10px 15px; border: 1px solid #ddd; border-radius: 5px; 
                                font-size: 16px; }
        button { background: #667eea; color: white; cursor: pointer; transition: 0.3s; }
        button:hover { background: #764ba2; transform: translateY(-2px); }
        
        .result { background: #f0f4ff; padding: 20px; border-radius: 5px; margin-top: 20px; 
                  border-left: 4px solid #667eea; }
        .result pre { background: #fff; padding: 10px; border-radius: 3px; overflow-x: auto; }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-card { background: #f9f9f9; padding: 20px; border-radius: 5px; text-align: center; 
                     border-top: 4px solid #667eea; }
        .stat-card h3 { color: #667eea; font-size: 24px; margin: 10px 0; }
        .stat-card p { color: #666; }
        
        footer { text-align: center; padding: 20px; color: #666; margin-top: 50px; }
        
        .hidden { display: none; }
    </style>
</head>
<body>
    <header>
        <h1>🚀 Waqas Khan Niazi AI & Deep Learning Hub</h1>
        <p>Advanced Machine Learning Platform with PyTorch</p>
    </header>
    
    <div class="container">
        <nav>
            <button onclick="showSection('home')">🏠 Home</button>
            <button onclick="showSection('topics')">📚 Topics</button>
            <button onclick="showSection('training')">🤖 Training</button>
            <button onclick="showSection('stats')">📊 Statistics</button>
        </nav>
        
        <!-- Home Section -->
        <div id="home" class="section">
            <h2>Welcome</h2>
            <p>This is a production-ready AI & Deep Learning platform featuring:</p>
            <ul style="margin: 15px 0 0 20px; line-height: 1.8;">
                <li>🔥 PyTorch neural network training</li>
                <li>📚 46+ learning topics and tutorials</li>
                <li>⚡ Optimized modular architecture</li>
                <li>🎯 Real-time model training and evaluation</li>
                <li>💾 Efficient caching and data handling</li>
                <li>🔌 RESTful API for all features</li>
            </ul>
        </div>
        
        <!-- Topics Section -->
        <div id="topics" class="section hidden">
            <h2>Learning Topics</h2>
            <input type="text" id="searchInput" placeholder="Search topics..." style="width: 100%; margin-bottom: 15px;">
            <div id="topicsContainer"></div>
        </div>
        
        <!-- Training Section -->
        <div id="training" class="section hidden">
            <h2>Model Training</h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h3>PyTorch Training</h3>
                    <label>Dataset:</label>
                    <select id="dataset">
                        <option value="iris">Iris</option>
                        <option value="digits">Digits</option>
                    </select>
                    
                    <label style="margin-top: 10px; display: block;">Epochs:</label>
                    <input type="number" id="epochs" value="10" min="1" max="100">
                    
                    <button onclick="trainModel()" style="width: 100%; margin-top: 15px;">Train Model</button>
                    <div id="torchResult" class="result hidden"></div>
                </div>
                
                <div>
                    <h3>Scikit-Learn Training</h3>
                    <label>Algorithm:</label>
                    <select id="algorithm">
                        <option value="RandomForest">Random Forest</option>
                        <option value="LogisticRegression">Logistic Regression</option>
                    </select>
                    
                    <button onclick="trainSklearn()" style="width: 100%; margin-top: 30px;">Train Model</button>
                    <div id="sklearnResult" class="result hidden"></div>
                </div>
            </div>
        </div>
        
        <!-- Statistics Section -->
        <div id="stats" class="section hidden">
            <h2>Platform Statistics</h2>
            <button onclick="loadStats()" style="margin-bottom: 20px;">Load Statistics</button>
            <div id="statsContainer" class="stats-grid"></div>
        </div>
    </div>
    
    <footer>
        <p>✨ Waqas Khan Niazi AI & Deep Learning Hub | Powered by Flask & PyTorch | 2024 🚀</p>
    </footer>
    
    <script>
        const API_BASE = '/api/v1';
        
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
            document.getElementById(sectionId).classList.remove('hidden');
        }
        
        async function loadTopics() {
            const res = await fetch(`${API_BASE}/topics`);
            const result = await res.json();
            const container = document.getElementById('topicsContainer');
            const topics = result.data.items || result.data;
            
            container.innerHTML = topics.map(t => `
                <div style="padding: 10px; border: 1px solid #ddd; margin-bottom: 10px; border-radius: 5px;">
                    <strong>${t.title}</strong> <span style="color: #667eea;">[${t.difficulty}]</span>
                    <p style="margin: 5px 0 0 0; color: #666;">${t.content}</p>
                </div>
            `).join('');
        }
        
        async function trainModel() {
            const dataset = document.getElementById('dataset').value;
            const epochs = document.getElementById('epochs').value;
            
            const res = await fetch(`${API_BASE}/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, epochs })
            });
            
            const result = await res.json();
            const container = document.getElementById('torchResult');
            container.innerHTML = `<pre>${JSON.stringify(result.data, null, 2)}</pre>`;
            container.classList.remove('hidden');
        }
        
        async function trainSklearn() {
            const algorithm = document.getElementById('algorithm').value;
            
            const res = await fetch(`${API_BASE}/train_sklearn`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ algorithm })
            });
            
            const result = await res.json();
            const container = document.getElementById('sklearnResult');
            container.innerHTML = `<pre>${JSON.stringify(result.data, null, 2)}</pre>`;
            container.classList.remove('hidden');
        }
        
        async function loadStats() {
            const res = await fetch(`${API_BASE}/stats`);
            const result = await res.json();
            const stats = result.data;
            
            document.getElementById('statsContainer').innerHTML = `
                <div class="stat-card">
                    <h3>${stats.total_topics}</h3>
                    <p>Total Topics</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.difficulty_distribution.beginner}</h3>
                    <p>Beginner Topics</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.difficulty_distribution.intermediate}</h3>
                    <p>Intermediate Topics</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.difficulty_distribution.advanced}</h3>
                    <p>Advanced Topics</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.device}</h3>
                    <p>Processing Device</p>
                </div>
                <div class="stat-card">
                    <h3>⚡</h3>
                    <p>Production Ready</p>
                </div>
            `;
        }
        
        // Load topics when section is shown
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('topics').addEventListener('click', loadTopics);
        });
    </script>
</body>
</html>'''


# ==================== Run Application ====================

if __name__ == '__main__':
    try:
        logger.info(f"Server starting: {HOST}:{PORT}")
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)
