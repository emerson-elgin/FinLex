import os

# Project directory structure
PROJECT_NAME = "FinSentimentAI"

DIR_STRUCTURE = {
    "data": ["raw_news.csv", "labeled_data.csv"],
    "models": [],
    "notebooks": ["data_preprocessing01.ipynb", "model_training02.ipynb", "evaluation03.ipynb"],
    "scripts": ["train.py", "predict.py", "preprocess.py"],
    "app": ["main.py"],
    "app/templates": ["index.html"],
}

# Creating directories and files
for folder, files in DIR_STRUCTURE.items():
    os.makedirs(os.path.join(PROJECT_NAME, folder), exist_ok=True)
    for file in files:
        file_path = os.path.join(PROJECT_NAME, folder, file)
        if not os.path.exists(file_path):  
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
