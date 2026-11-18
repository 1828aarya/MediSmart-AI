# MediSmart AI — educational starter project

## What it does
- Classifies an entered medicine text record into a broad informational category.
- Compares Logistic Regression, Decision Tree, and Random Forest.
- Reports accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.
- Provides a simple expiry-date checker.
- Displays the included CSV dataset.

## Important limitations
The included CSV is a small, synthetic teaching dataset. It is not a validated medicine database and the model is not clinically accurate. This app does not identify pills from appearance, diagnose conditions, recommend treatment, or suggest substitutes. Do not use it for medical decisions.

## Setup (Windows)
1. Install Python 3.10 or later.
2. Extract the ZIP folder.
3. Open the folder in VS Code.
4. Open Terminal in VS Code.
5. Create a virtual environment:
   `python -m venv venv`
6. Activate it:
   PowerShell: `venv\Scripts\Activate.ps1`
   CMD: `venv\Scripts\activate`
7. Install libraries:
   `pip install -r requirements.txt`
8. Train and compare models:
   `python train_model.py`
9. Start the web app:
   `streamlit run app.py`
10. Open the local URL shown in the terminal (usually http://localhost:8501).

## Files
- `data/medicine_dataset.csv`: sample training data.
- `train_model.py`: preprocessing, train/test split, three algorithms, evaluation and model saving.
- `app.py`: Streamlit user interface.
- `requirements.txt`: required Python packages.

## How to explain the ML workflow
Dataset -> text-field combination -> TF-IDF vectorization -> train/test split -> train 3 classifiers -> evaluate -> save best model -> user enters package details -> category prediction.

## Suggested improvements
For a real-world prototype, replace synthetic records with a legally obtained, pharmacist-reviewed and up-to-date medicine catalogue. Add OCR only as a text-extraction aid and require the user to verify the extracted name against the printed package. Add a trusted source and human review before showing medicine information.
