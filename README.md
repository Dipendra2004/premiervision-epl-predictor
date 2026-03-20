# PremierVision EPL Predictor

A Streamlit app that predicts EPL match outcomes using a trained TensorFlow model.

## Project Files

- app.py: Streamlit web app
- epl_final.csv: EPL match dataset used for team statistics and feature generation
- epl_model.keras: Trained prediction model
- scaler.pkl: Saved scaler artifact from model training workflow

## Run Locally

1. Open terminal in this folder.
2. Create and activate virtual environment.
3. Install dependencies.
4. Start Streamlit app.

PowerShell commands:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py

## Notes on Predictions

- Repeated runs with the same season and same home/away teams are stable.
- Inputs are built from real season statistics (goals, shots, corners, fouls, cards, points per match) and mapped to the model's expected 36-feature input.

## Publish To GitHub

Your current global git root appears to be your Desktop. To publish only this project, initialize git inside this folder.

PowerShell commands:

cd C:\Users\dipen\OneDrive\Desktop\Dipendra\EPL_Project
git init
git add .
git commit -m "Initial commit: PremierVision EPL Predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main

If remote already exists, run:

git remote set-url origin https://github.com/<your-username>/<your-repo-name>.git
