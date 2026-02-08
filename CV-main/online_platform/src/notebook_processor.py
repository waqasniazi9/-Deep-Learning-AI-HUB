"""
Notebook Processor - Extracts and executes notebook code
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import sys

class NotebookProcessor:
    def __init__(self, notebooks_dir: str):
        self.notebooks_dir = notebooks_dir
        self.notebooks = self._discover_notebooks()
    
    def _discover_notebooks(self) -> List[Dict[str, Any]]:
        """Discover all .ipynb files in the directory"""
        notebooks = []
        notebook_dir = Path(self.notebooks_dir)
        
        for nb_file in sorted(notebook_dir.glob("*.ipynb")):
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    nb_data = json.load(f)
                    
                # Extract metadata
                title = nb_file.stem
                num_cells = len(nb_data.get('cells', []))
                
                notebooks.append({
                    'path': str(nb_file),
                    'name': nb_file.name,
                    'title': title,
                    'num_cells': num_cells,
                    'data': nb_data
                })
            except Exception as e:
                print(f"Error reading {nb_file}: {e}")
        
        return notebooks
    
    def get_notebook_list(self) -> List[Dict[str, str]]:
        """Get list of all notebooks"""
        return [{'name': nb['title'], 'path': nb['path'], 'cells': nb['num_cells']} 
                for nb in self.notebooks]
    
    def get_notebook_content(self, notebook_title: str) -> Dict[str, Any]:
        """Get full content of a specific notebook"""
        for nb in self.notebooks:
            if nb['title'] == notebook_title:
                return nb
        return None
    
    def extract_code_cells(self, notebook_data: Dict) -> List[Dict[str, str]]:
        """Extract all code cells from a notebook"""
        code_cells = []
        for i, cell in enumerate(notebook_data.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                code_cells.append({
                    'index': i,
                    'code': source,
                    'has_output': len(cell.get('outputs', [])) > 0
                })
        return code_cells
    
    def extract_markdown_cells(self, notebook_data: Dict) -> List[Dict[str, str]]:
        """Extract all markdown cells from a notebook"""
        markdown_cells = []
        for i, cell in enumerate(notebook_data.get('cells', [])):
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                markdown_cells.append({
                    'index': i,
                    'content': source
                })
        return markdown_cells
    
    def get_notebook_summary(self, notebook_title: str) -> Dict[str, Any]:
        """Get a summary of notebook structure"""
        nb = self.get_notebook_content(notebook_title)
        if not nb:
            return None
        
        code_cells = self.extract_code_cells(nb['data'])
        markdown_cells = self.extract_markdown_cells(nb['data'])
        
        return {
            'title': notebook_title,
            'total_cells': len(nb['data'].get('cells', [])),
            'code_cells': len(code_cells),
            'markdown_cells': len(markdown_cells),
            'code_samples': code_cells,
            'markdown': markdown_cells
        }
    
    def run_notebook_code(self, notebook_title: str, cell_index: int = None) -> str:
        """Run code from a notebook"""
        try:
            nb = self.get_notebook_content(notebook_title)
            if not nb:
                return "Notebook not found"
            
            code_cells = self.extract_code_cells(nb['data'])
            
            if cell_index is not None:
                if cell_index < len(code_cells):
                    code = code_cells[cell_index]['code']
                else:
                    return "Cell index out of range"
            else:
                # Run all code cells
                code = '\n\n'.join([cell['code'] for cell in code_cells])
            
            # Execute the code
            exec_globals = {}
            exec(code, exec_globals)
            return "Code executed successfully"
        
        except Exception as e:
            return f"Error executing code: {str(e)}"
