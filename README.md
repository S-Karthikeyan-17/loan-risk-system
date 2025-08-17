# AI-Powered Loan Eligibility & Risk Scoring System

This repository implements **Project A** from the AI Engineer – Round 2 Assignment: an end-to-end AI-powered system for predicting loan default risk. The system includes data preprocessing, feature engineering, model training, evaluation, and a FastAPI backend for serving predictions and model insights. It uses a machine learning model to compute a risk score (default probability) based on borrower features such as income, loan amount, and credit score.

The system is built with Python, FastAPI, scikit-learn, and related libraries. It includes scripts for training/evaluation, a production-ready API, and visualizations for dataset insights and model performance.

![Personal Information](output_screenshots/output1.png)
![Employment Details](output_screenshots/output2.png)
![Financial Information](output_screenshots/output3.png)
![Loan Details](output_screenshots/output4.png)
![Prediction](output_screenshots/output5.png)

## 📌 Features
- **Data Preprocessing & Cleaning**: Handles missing values, outliers, and data types from the provided Excel dataset.
- **Feature Engineering**: Derives new features like income-to-loan ratios, age/DTI buckets, employment years, and high-interest flags to improve model accuracy.
- **Model Training & Selection**: Trains multiple classifiers (Logistic Regression, Random Forest, and optionally XGBoost if installed), evaluates via cross-validation and holdout set, and selects the best based on ROC-AUC.
- **Model Evaluation**: Computes metrics like accuracy, F1-score, ROC-AUC, and generates classification reports, ROC/PR curves, and feature importance plots.
- **FastAPI Backend**: Exposes endpoints for risk prediction, model health checks, and performance insights with input validation and error handling.
- **Visualizations**: Includes plots for feature distributions, correlations, class imbalance, risk segmentation, and model performance.

## 🏗️ Tech Stack
- **Backend**: FastAPI, Uvicorn
- **ML Libraries**: scikit-learn, pandas, numpy, joblib (optional: xgboost for advanced modeling)
- **Visualization**: Matplotlib
- **Other**: Pydantic for validation, argparse for CLI

## 📂 Project Structure
```
loan-risk-system/
├── .git/                     # Git repository files
├── __pycache__/              # Python cache
├── artifacts/                # Model artifacts and outputs
│   ├── model.pkl             # Serialized trained model
│   ├── eval_metrics.json     # Evaluation metrics
│   ├── eval_classification_report.txt  # Classification report
│   ├── metrics.json          # Training metrics (CV and holdout)
│   ├── classification_report.txt  # Best model classification report
│   ├── feature_names.json    # List of feature names post-preprocessing
│   └── plots/                # Generated visualizations (EDA and performance plots)
├── data/                     # Dataset directory
│   └── 6S_AI_TASK-Loan_default_Loan_default.xlsx  # Input dataset
├── frontend/                 # Static frontend files
│   ├── index.html            # Multi-step loan application form
│   ├── style.css             # UI styling
│   └── script.js             # Form logic and API integration
├── output_screenshots/       # Screenshots of the application in action
│   ├── output1.png           # Personal Information form
│   ├── output2.png           # Employment Details form
│   ├── output3.png           # Financial Information form
│   ├── output4.png           # Loan Details form
│   └── output5.png           # Prediction result with risk meter
├── venv/                     # Virtual environment (not committed)
├── .gitignore                # Git ignore file
├── app.py                    # FastAPI backend (API endpoints)
├── evaluate.py               # Evaluation script for model metrics
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
└── train.py                  # Training script (preprocessing, FE, training, evaluation)
```

## ⚙️ Setup Instructions
1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/loan-risk-system.git
   cd loan-risk-system
   ```

2. **Create and Activate Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   - Note: If you want to use XGBoost (optional for better performance), install it separately: `pip install xgboost`. The training script will detect and use it if available.

4. **Dataset**: Ensure the dataset (`data/6S_AI_TASK-Loan_default_Loan_default.xlsx`) is present. Download from [here](https://provided-link) if missing.

## 🚀 How to Run
### Training the Model
Run the training script to preprocess data, engineer features, train models, evaluate, and save artifacts:
```
python train.py --data-path data/6S_AI_TASK-Loan_default_Loan_default.xlsx --out-dir artifacts
```
- This generates `model.pkl`, metrics files, reports, and plots in `artifacts/`.
- Optional: Use `--help` for CLI options.

### Evaluating the Model
Run evaluation on the full dataset using the trained model:
```
python evaluate.py --data-path data/6S_AI_TASK-Loan_default_Loan_default.xlsx --model-path artifacts/model.pkl --out-dir artifacts
```
- Outputs metrics like accuracy, F1, ROC-AUC, and saves `eval_metrics.json` and `eval_classification_report.txt`.

### Running the FastAPI Backend
Start the API server:
```
uvicorn app:app --reload --port 8000
```
- Access the API at `http://127.0.0.1:8000`.
- The frontend is served at `http://127.0.0.1:8000/` (open in browser for interactive form).

### Retraining/Regenerating Artifacts
- To retrain: Run `python train.py` with updated dataset or parameters (e.g., change `--test-size` in code if needed).
- Artifacts are overwritten in `artifacts/`. Visualizations are regenerated in `artifacts/plots/`.
- For custom configurations: Modify `TrainConfig` in `train.py` (e.g., random_state, n_splits for CV).

## 🔗 API Usage
The FastAPI backend provides the following endpoints. Use tools like Postman or curl for testing, or interact via the frontend form.

### 1. Health Check
- **Method**: GET
- **Endpoint**: `/health`
- **Description**: Checks if the API and model are loaded.
- **Response Example**:
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 2. Predict Risk Score
- **Method**: POST
- **Endpoint**: `/predict`
- **Description**: Predicts loan default risk based on input features.
- **Request Body** (JSON, validated via Pydantic):
  ```json
  {
    "Age": 30,
    "Income": 100000,
    "LoanAmount": 20000,
    "CreditScore": 650,
    "MonthsEmployed": 24,
    "NumCreditLines": 5,
    "InterestRate": 10,
    "LoanTerm": 12,
    "DTIRatio": 0.3,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Business",
    "HasCoSigner": "No"
  }
  ```
- **Response Example**:
  ```json
  {
    "risk_score": 0.4855,
    "decision": "Low Risk",
    "threshold": 0.5
  }
  ```
- **Error Handling**: Returns 400 Bad Request with details (e.g., invalid input values).

### 3. Model Insights
- **Method**: GET
- **Endpoint**: `/model`
- **Description**: Returns model performance metrics.
- **Response Example**:
  ```json
  {
    "status": "success",
    "metrics": {
      "accuracy": 0.85,
      "f1": 0.78,
      "roc_auc": 0.92
    }
  }
  ```

## 📊 Dataset Insights
The dataset (`6S_AI_TASK-Loan_default_Loan_default.xlsx`) contains ~255,347 rows with features like Age, Income, LoanAmount, CreditScore, etc., and a binary target `Default` (0: No Default, 1: Default).

- **Key Trends**:
  - Class Imbalance: ~88% non-defaults vs. ~12% defaults – addressed via stratified splitting.
  - Feature Distributions: Age is normally distributed (~18-70), Income skewed right (many low-income applicants). LoanAmount and InterestRate show wide ranges.
  - Correlations: High negative correlation between CreditScore and Default; positive between LoanAmount/InterestRate and Default.
  - High-Impact Variables: CreditScore, Income_to_Loan_Ratio (engineered), DTIRatio, and NumCreditLines strongly influence risk.

- **Patterns**:
  - Higher DTIRatio (>0.5) and InterestRate (>=15%) correlate with defaults.
  - Younger applicants (<=25) and those with short employment (<2 years) have elevated risk.

For details, see the training script's EDA section or generated plots.

## 📈 Visualizations & Reports
Visualizations are generated during training/evaluation and saved in `artifacts/plots/`:
- `eda_class_balance.png`: Class imbalance bar chart.
- `eda_dist_{feature}.png`: Histograms for numeric features (e.g., Age, Income).
- `eda_correlation_heatmap.png`: Correlation heatmap of numeric features.
- `perf_roc_curves.png`: ROC curves for all models.
- `perf_pr_curves.png`: Precision-Recall curves.
- `perf_risk_segmentation.png`: Risk score distribution by default class.
- `perf_feature_importance.png`: Permutation feature importance for the best model.

Screenshots of the frontend/application are in `output_screenshots/` (see above for previews).

Reports:
- `metrics.json`: CV and holdout metrics.
- `classification_report.txt`: Precision, recall, F1 per class.
- `eval_metrics.json`: Full-dataset evaluation metrics.

## 🛠️ Future Enhancements
- Integrate XGBoost/LightGBM by default.
- Add SHAP/LIME for explainable predictions.
- Deploy to cloud (e.g., Render/Heroku) with Docker.
- Enhance frontend with real-time validation.

## 👨‍💻 Author
Karthikeyan S  
GitHub: S-Karthikeyan-17  

