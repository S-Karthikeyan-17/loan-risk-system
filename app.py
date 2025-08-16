from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd, numpy as np, joblib, json, logging
from pathlib import Path

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("loan-risk")

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI(title="AI Loan Risk Scoring")

# CORS configuration with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Match frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly list methods
    allow_headers=["Content-Type", "Accept"],  # Explicitly list headers
)

# -------------------------
# Serve Frontend
# -------------------------
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve index.html at root"""
    return FileResponse(FRONTEND_DIR / "index.html")

# -------------------------
# Schema
# -------------------------
YesNo = Literal["Yes", "No"]
EducationType = Literal["High School", "Bachelor's", "Master's", "PhD"]
EmploymentType = Literal["Full-time", "Part-time", "Self-employed", "Unemployed"]
MaritalStatusType = Literal["Single", "Married", "Divorced"]
LoanPurposeType = Literal["Personal", "Business", "Education", "Home", "Car"]

class LoanApplication(BaseModel):
    Age: float = Field(..., ge=18, description="Applicant's age (18 or older)")
    Income: float = Field(..., gt=0, description="Annual income in dollars")
    LoanAmount: float = Field(..., gt=0, description="Requested loan amount")
    CreditScore: float = Field(..., ge=300, le=850, description="Credit score between 300 and 850")
    MonthsEmployed: float = Field(..., ge=0, description="Months of employment")
    NumCreditLines: float = Field(..., ge=0, description="Number of active credit lines")
    InterestRate: float = Field(..., gt=0, description="Loan interest rate in percentage")
    LoanTerm: float = Field(..., gt=0, description="Loan term in months")
    DTIRatio: float = Field(..., ge=0, le=1, description="Debt-to-Income ratio (0 to 1)")

    Education: EducationType
    EmploymentType: EmploymentType
    MaritalStatus: MaritalStatusType
    HasMortgage: YesNo
    HasDependents: YesNo
    LoanPurpose: LoanPurposeType
    HasCoSigner: YesNo

# -------------------------
# Model Loading
# -------------------------
MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

if not MODEL_PATH.exists():
    logger.error("Model file not found at artifacts/model.pkl")
    raise RuntimeError("❌ Model not found. Train first and save to artifacts/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"❌ Failed to load model: {str(e)}")

# -------------------------
# Feature Engineering
# -------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    def safe_ratio(num, den, name):
        if num in X.columns and den in X.columns:
            denom = (X[den].replace(0, np.nan) + 1e-6)
            X[name] = (X[num] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    safe_ratio("Income", "LoanAmount", "Income_to_Loan_Ratio")
    safe_ratio("CreditScore", "LoanAmount", "Credit_per_LoanAmount")
    safe_ratio("LoanAmount", "NumCreditLines", "LoanAmount_per_CreditLine")

    if "MonthsEmployed" in X.columns:
        X["EmploymentYears"] = X["MonthsEmployed"] / 12.0
    if "InterestRate" in X.columns:
        X["HighInterestFlag"] = (X["InterestRate"] >= 15).astype(int)
    if "LoanTerm" in X.columns:
        X["TermYears"] = X["LoanTerm"] / 12.0
    if "Age" in X.columns:
        X["AgeBucket"] = pd.cut(
            X["Age"],
            bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
            labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
    if "DTIRatio" in X.columns:
        X["DTI_Bucket"] = pd.cut(
            X["DTIRatio"],
            bins=[-np.inf, 0.2, 0.35, 0.5, 0.65, 0.8, np.inf],
            labels=["<=0.2", "0.21-0.35", "0.36-0.5", "0.51-0.65", "0.66-0.8", "0.8+"],
        )

    # Fill missing values
    for c in X.select_dtypes(include=[np.number]).columns:
        X[c] = X[c].fillna(X[c].median())
    for c in X.select_dtypes(exclude=[np.number]).columns:
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    return X

# -------------------------
# API Endpoints
# -------------------------
@app.get("/health")
def health():
    """Check if the API is running"""
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(app_data: LoanApplication):
    try:
        data = app_data.dict()
        df = pd.DataFrame([data])
        df_fe = feature_engineer(df)

        # Ensure feature order matches model training
        expected_features = getattr(model, "feature_names_in_", df_fe.columns)  # Fallback to df_fe.columns
        df_fe = df_fe.reindex(columns=expected_features, fill_value=0)

        proba = model.predict_proba(df_fe)[0, 1]
        decision = "High Risk" if proba >= 0.5 else "Low Risk"

        # Log prediction
        logger.info(f"Prediction made | Score={proba:.4f} | Decision={decision} | Input={data}")

        return {
            "risk_score": round(float(proba), 4),
            "decision": decision,
            "threshold": 0.5
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model")
def model_info():
    """Return model metrics if available"""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                return {
                    "status": "success",
                    "metrics": metrics
                }
        except Exception as e:
            logger.error(f"Failed to load metrics: {str(e)}")
            return {"status": "error", "message": f"Failed to load metrics: {str(e)}"}
    return {"status": "error", "message": "metrics.json not found. Train the model first."}
#python -m uvicorn app:app --reload --port 8000 