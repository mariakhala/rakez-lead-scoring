"""
Lead Scoring Model Serving API
FastAPI application for serving lead score predictions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
from datetime import datetime
import numpy as np

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lead Scoring API",
    description="ML-powered lead scoring for RAKEZ CRM",
    version="1.0.0"
)

# Global model reference
model = None
model_version = None


# ============== Data Models ==============

class LeadInput(BaseModel):
    """Input schema for a single lead"""
    lead_id: str
    lead_source: str
    industry: str
    company_size: int = Field(..., ge=1, description="Number of employees")
    engagement_score: float = Field(..., ge=0, le=100)
    website_visits: int = Field(default=0, ge=0)
    email_opens: int = Field(default=0, ge=0)
    email_clicks: int = Field(default=0, ge=0)
    days_since_first_contact: int = Field(default=0, ge=0)
    region: Optional[str] = None


class LeadScoreResponse(BaseModel):
    """Response schema for lead score prediction"""
    lead_id: str
    lead_score: float
    score_bucket: str
    confidence: Optional[float] = None
    model_version: str
    scored_at: str


class BatchRequest(BaseModel):
    """Request schema for batch predictions"""
    leads: List[LeadInput]


class BatchResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[LeadScoreResponse]
    total_processed: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_version: str
    timestamp: str


# ============== Helper Functions ==============

def get_score_bucket(score: float) -> str:
    """Categorize score into buckets"""
    if score >= 0.8:
        return "Hot"
    elif score >= 0.5:
        return "Warm"
    elif score >= 0.3:
        return "Cool"
    else:
        return "Cold"


def prepare_features(lead: LeadInput) -> dict:
    """Prepare features for model prediction"""
    return {
        "lead_source": lead.lead_source,
        "industry": lead.industry,
        "company_size": lead.company_size,
        "engagement_score": lead.engagement_score,
        "website_visits": lead.website_visits,
        "email_opens": lead.email_opens,
        "email_clicks": lead.email_clicks,
        "days_since_first_contact": lead.days_since_first_contact,
        "region": lead.region or "Unknown"
    }


def log_prediction(lead_id: str, score: float, latency_ms: float):
    """Log prediction for monitoring"""
    logger.info(
        "prediction",
        extra={
            "lead_id": lead_id,
            "score": score,
            "latency_ms": latency_ms,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============== Startup Events ==============

@app.on_event("startup")
async def load_model():
    """Load model from MLflow registry on startup"""
    global model, model_version
    
    if MLFLOW_AVAILABLE:
        try:
            model_uri = "models:/lead-scoring-production/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            model_version = model.metadata.run_id if model.metadata else "unknown"
            logger.info(f"Model loaded successfully. Version: {model_version}")
            return
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
    
    # For demo purposes, create a mock model
    model = MockModel()
    model_version = "demo-v1.0"
    logger.info("Using demo model for demonstration")


class MockModel:
    """Mock model for demonstration when MLflow is not available"""
    def predict(self, features):
        # Generate realistic-looking scores based on features
        if isinstance(features, list):
            return [self._score_single(f) for f in features]
        return [self._score_single(features)]
    
    def _score_single(self, features):
        base_score = 0.3
        if features.get("engagement_score", 0) > 50:
            base_score += 0.2
        if features.get("website_visits", 0) > 5:
            base_score += 0.1
        if features.get("email_opens", 0) > 3:
            base_score += 0.1
        if features.get("company_size", 0) > 100:
            base_score += 0.1
        return min(base_score + np.random.uniform(-0.1, 0.1), 1.0)


# ============== API Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_version=model_version or "not_loaded",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=LeadScoreResponse)
async def predict_single(lead: LeadInput, background_tasks: BackgroundTasks):
    """Score a single lead"""
    start_time = time.time()
    
    try:
        # Prepare features
        features = prepare_features(lead)
        
        # Get prediction
        prediction = model.predict([features])
        score = float(prediction[0])
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log in background
        background_tasks.add_task(log_prediction, lead.lead_id, score, latency_ms)
        
        return LeadScoreResponse(
            lead_id=lead.lead_id,
            lead_score=round(score, 4),
            score_bucket=get_score_bucket(score),
            model_version=model_version,
            scored_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error for lead {lead.lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Score multiple leads in batch"""
    start_time = time.time()
    
    try:
        predictions = []
        
        for lead in request.leads:
            features = prepare_features(lead)
            prediction = model.predict([features])
            score = float(prediction[0])
            
            predictions.append(LeadScoreResponse(
                lead_id=lead.lead_id,
                lead_score=round(score, 4),
                score_bucket=get_score_bucket(score),
                model_version=model_version,
                scored_at=datetime.utcnow().isoformat()
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    return {
        "model_version": model_version,
        "status": "loaded" if model is not None else "not_loaded",
        "endpoint": "lead-scoring-production",
        "last_updated": datetime.utcnow().isoformat()
    }


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

