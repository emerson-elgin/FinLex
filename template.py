import os
import json

# Project directory structure
PROJECT_NAME = "FinSentimentAI"

DIR_STRUCTURE = {
    "data": ["all-data.csv", "raw_news.csv", "labeled_data.csv"],
    "models": [],
    "notebooks": ["data_preprocessing.ipynb", "model_training.ipynb", "evaluation.ipynb"],
    "scripts": ["train.py", "predict.py", "preprocess.py", "convert_data.py"],
    "app": ["main.py"],
    "app/templates": ["index.html"],
}

# Function to create an empty Jupyter notebook
def create_empty_notebook(file_path):
    notebook_content = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(file_path, "w") as f:
        json.dump(notebook_content, f, indent=4)

# Creating directories and files
for folder, files in DIR_STRUCTURE.items():
    os.makedirs(os.path.join(PROJECT_NAME, folder), exist_ok=True)
    for file in files:
        file_path = os.path.join(PROJECT_NAME, folder, file)
        if not os.path.exists(file_path):
            if file.endswith(".ipynb"):
                create_empty_notebook(file_path)  # Create an empty Jupyter notebook
            else:
                with open(file_path, "w") as f:
                    f.write(f"# {file} - Auto-generated\n")

# Creating additional config and documentation files
config_files = ["README.md", "requirements.txt", "config.yaml", "Dockerfile", ".gitignore"]
for file in config_files:
    file_path = os.path.join(PROJECT_NAME, file)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(f"# {file} - Auto-generated\n")

print(f"✅ Project '{PROJECT_NAME}' initialized successfully!")
