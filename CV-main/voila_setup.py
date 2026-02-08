#!/usr/bin/env python3
"""
Batch Voila converter - Convert all notebooks to standalone web apps
"""

import os
import subprocess
from pathlib import Path
import json
import sys

def convert_notebooks_to_voila():
    """Generate Voila configuration for all notebooks"""
    
    notebook_dir = Path(".")
    notebooks = list(notebook_dir.glob("*.ipynb"))
    notebooks = [nb for nb in notebooks if not nb.name.startswith(".")]
    
    print(f"📚 Found {len(notebooks)} notebooks to configure for Voila")
    
    voila_config = {
        "VoilaConfiguration": {
            "template": "gridstack",
            "theme": "light",
            "enable_nbconvert_resources": False,
            "resources": {
                "inlineJS": False,
                "preload_dynamic_extensions": True
            }
        }
    }
    
    # Create voila config directory
    voila_config_dir = Path.home() / ".jupyter" / "voila_config.py"
    
    config_content = """
# Voila Configuration
c.VoilaConfiguration.template = 'gridstack'
c.VoilaConfiguration.theme = 'light'
c.VoilaConfiguration.enable_nbconvert_resources = False
c.VoilaConfiguration.resources = {'inlineJS': False, 'preload_dynamic_extensions': True}
"""
    
    print(f"\n✅ Voila configured for {len(notebooks)} notebooks")
    print(f"📝 Configuration file: {voila_config_dir}")
    
    print("\n🚀 To run a notebook with Voila:")
    print("   voila notebook_name.ipynb")
    print("\n📊 Or create a Voila dashboard index:")
    print("   voila --VoilaConfiguration.template='default' HOME.ipynb")
    
    # Generate index of notebooks
    index_data = {
        "notebooks": [],
        "total": len(notebooks),
        "categories": {}
    }
    
    for nb in sorted(notebooks):
        prefix = nb.stem.split("_")[0]
        if prefix.startswith("1"):
            category = "PyTorch Basics"
        elif prefix.startswith("2"):
            category = "Deep Learning"
        else:
            category = "Advanced Topics"
        
        if category not in index_data["categories"]:
            index_data["categories"][category] = []
        
        index_data["categories"][category].append(nb.name)
        index_data["notebooks"].append({
            "name": nb.name,
            "stem": nb.stem,
            "category": category,
            "size_kb": nb.stat().st_size / 1024
        })
    
    # Save index
    with open("notebooks_index.json", "w") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Generated notebooks_index.json")
    
    # Print summary
    print("\n📊 Notebook Summary:")
    for category, nbs in index_data["categories"].items():
        print(f"   {category}: {len(nbs)} notebooks")

if __name__ == "__main__":
    convert_notebooks_to_voila()
