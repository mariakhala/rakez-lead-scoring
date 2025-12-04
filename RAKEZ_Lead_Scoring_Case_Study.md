# RAKEZ Case Study: Deploying and Monitoring a Lead Scoring Model

## ML Engineer Case Study Submission

**Prepared by:** Maria Muneeb 
**Date:** December 2025  
**Role:** Machine Learning Engineer

---

# Executive Summary

This document presents a comprehensive MLOps solution for deploying, testing, monitoring, and maintaining RAKEZ's lead scoring model in production. The solution leverages industry best practices using Python, Databricks, MLflow, and modern DevOps principles to ensure reliability, scalability, and business value.

---

# 1. Deployment Strategy

## 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │   CRM/Lead   │────▶│   API        │────▶│   Model Serving Layer    │   │
│   │   Sources    │     │   Gateway    │     │   (Databricks/Container) │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│                              │                          │                    │
│                              ▼                          ▼                    │
│                       ┌──────────────┐          ┌──────────────┐            │
│                       │   Load       │          │   MLflow     │            │
│                       │   Balancer   │          │   Registry   │            │
│                       └──────────────┘          └──────────────┘            │
│                                                         │                    │
│                                                         ▼                    │
│                                                 ┌──────────────┐            │
│                                                 │   Model      │            │
│                                                 │   Artifacts  │            │
│                                                 └──────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Deployment Approach

### Option A: Databricks Model Serving (Recommended)

**Why Databricks Model Serving:**
- Native integration with MLflow
- Auto-scaling capabilities
- Built-in monitoring
- Low operational overhead

**Implementation Steps:**

1. **Register Model in MLflow Registry**
```python
import mlflow
from mlflow.tracking import MlflowClient

# Log model with signature
with mlflow.start_run(run_name="lead_scoring_v1") as run:
    mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="lead_scoring_model",
        registered_model_name="lead-scoring-production",
        signature=infer_signature(X_train, y_pred),
        input_example=X_train.iloc[:5]
    )
```

2. **Create Serving Endpoint**
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

w = WorkspaceClient()

# Create or update serving endpoint
w.serving_endpoints.create_and_wait(
    name="lead-scoring-endpoint",
    config=EndpointCoreConfigInput(
        served_models=[{
            "model_name": "lead-scoring-production",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": False
        }]
    )
)
```

### Option B: Containerized REST API

**For more control and customization:**

```python
# app/main.py - FastAPI Application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow
import logging
from datetime import datetime

app = FastAPI(title="Lead Scoring API", version="1.0.0")

# Load model at startup
model = None
model_version = None

@app.on_event("startup")
async def load_model():
    global model, model_version
    model = mlflow.pyfunc.load_model("models:/lead-scoring-production/Production")
    model_version = model.metadata.run_id
    logging.info(f"Loaded model version: {model_version}")

class LeadInput(BaseModel):
    lead_id: str
    lead_source: str
    industry: str
    company_size: int = Field(..., ge=1)
    engagement_score: float = Field(..., ge=0, le=100)
    website_visits: int = Field(default=0, ge=0)
    email_opens: int = Field(default=0, ge=0)
    days_since_first_contact: int = Field(default=0, ge=0)

class LeadScoreResponse(BaseModel):
    lead_id: str
    lead_score: float
    score_bucket: str
    model_version: str
    scored_at: str

class BatchRequest(BaseModel):
    leads: List[LeadInput]

@app.post("/predict", response_model=LeadScoreResponse)
async def predict_single(lead: LeadInput):
    try:
        start_time = datetime.utcnow()
        
        # Prepare features
        features = prepare_features(lead)
        
        # Get prediction
        score = float(model.predict([features])[0])
        
        # Determine bucket
        bucket = get_score_bucket(score)
        
        # Log prediction
        log_prediction(lead.lead_id, score, model_version, start_time)
        
        return LeadScoreResponse(
            lead_id=lead.lead_id,
            lead_score=round(score, 4),
            score_bucket=bucket,
            model_version=model_version,
            scored_at=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    results = []
    for lead in request.leads:
        result = await predict_single(lead)
        results.append(result)
    return {"predictions": results}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": model_version,
        "timestamp": datetime.utcnow().isoformat()
    }

def get_score_bucket(score: float) -> str:
    if score >= 0.8:
        return "Hot"
    elif score >= 0.5:
        return "Warm"
    elif score >= 0.3:
        return "Cool"
    else:
        return "Cold"
```

## 1.3 Model Versioning & Rollback

### Version Control Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE STAGES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Development ──▶ Staging ──▶ Production ──▶ Archived           │
│                                                                  │
│   - Experiments    - A/B Tests   - Live Traffic  - Historical   │
│   - Validation     - Shadow Mode - Monitored     - Rollback     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Rollback Implementation

```python
class ModelVersionManager:
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "lead-scoring-production"
    
    def get_current_production(self):
        """Get current production model version"""
        versions = self.client.get_latest_versions(
            self.model_name, 
            stages=["Production"]
        )
        return versions[0] if versions else None
    
    def get_version_history(self):
        """Get all model versions with metadata"""
        versions = self.client.search_model_versions(
            f"name='{self.model_name}'"
        )
        return [{
            "version": v.version,
            "stage": v.current_stage,
            "created": v.creation_timestamp,
            "metrics": self.get_version_metrics(v.run_id)
        } for v in versions]
    
    def rollback_to_version(self, target_version: str):
        """Rollback production to a specific version"""
        # Step 1: Archive current production
        current = self.get_current_production()
        if current:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=current.version,
                stage="Archived",
                archive_existing_versions=False
            )
            logging.info(f"Archived version {current.version}")
        
        # Step 2: Promote target to production
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=target_version,
            stage="Production"
        )
        logging.info(f"Promoted version {target_version} to Production")
        
        # Step 3: Trigger endpoint update
        self.update_serving_endpoint(target_version)
        
        # Step 4: Send notification
        self.notify_rollback(current.version, target_version)
        
        return {
            "status": "success",
            "previous_version": current.version,
            "current_version": target_version
        }
    
    def update_serving_endpoint(self, version):
        """Update Databricks serving endpoint"""
        w = WorkspaceClient()
        w.serving_endpoints.update_config_and_wait(
            name="lead-scoring-endpoint",
            served_models=[{
                "model_name": self.model_name,
                "model_version": version,
                "workload_size": "Small"
            }]
        )
```

### Auditability

All model changes are tracked with:
- **Git commits** for code changes
- **MLflow runs** for experiments and metrics
- **Model Registry** for version history
- **Deployment logs** for production changes

---

# 2. Online Testing Approach

## 2.1 Testing Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ONLINE TESTING STRATEGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 1: Shadow Deployment (Week 1)                                        │
│   ├── New model runs in parallel                                            │
│   ├── Predictions logged but not served                                     │
│   └── Compare with production model                                         │
│                                                                              │
│   Phase 2: Canary Release (Week 2)                                          │
│   ├── 5% traffic to new model                                               │
│   ├── Monitor key metrics                                                   │
│   └── Gradual increase if successful                                        │
│                                                                              │
│   Phase 3: A/B Test (Week 3-4)                                              │
│   ├── 50/50 split for statistical power                                     │
│   ├── Measure business outcomes                                             │
│   └── Make promotion decision                                               │
│                                                                              │
│   Phase 4: Full Rollout                                                      │
│   ├── 100% traffic to winning model                                         │
│   └── Continue monitoring                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Shadow Deployment Implementation

```python
class ShadowDeployment:
    """Run new model in shadow mode alongside production"""
    
    def __init__(self):
        self.production_model = load_model("Production")
        self.shadow_model = load_model("Staging")
        self.comparison_metrics = []
    
    async def predict_with_shadow(self, lead_data: dict):
        # Production prediction (this gets returned)
        prod_start = time.time()
        prod_prediction = self.production_model.predict(lead_data)
        prod_latency = time.time() - prod_start
        
        # Shadow prediction (logged only, not returned)
        shadow_start = time.time()
        shadow_prediction = self.shadow_model.predict(lead_data)
        shadow_latency = time.time() - shadow_start
        
        # Log comparison
        self.log_comparison({
            "timestamp": datetime.utcnow().isoformat(),
            "lead_id": lead_data.get("lead_id"),
            "production_score": float(prod_prediction),
            "shadow_score": float(shadow_prediction),
            "score_difference": abs(float(prod_prediction) - float(shadow_prediction)),
            "production_latency_ms": prod_latency * 1000,
            "shadow_latency_ms": shadow_latency * 1000
        })
        
        # Return only production prediction
        return prod_prediction
    
    def analyze_shadow_results(self):
        """Analyze shadow deployment results"""
        df = pd.DataFrame(self.comparison_metrics)
        
        return {
            "total_predictions": len(df),
            "mean_score_difference": df["score_difference"].mean(),
            "correlation": df["production_score"].corr(df["shadow_score"]),
            "shadow_avg_latency": df["shadow_latency_ms"].mean(),
            "production_avg_latency": df["production_latency_ms"].mean(),
            "recommendation": self.get_recommendation(df)
        }
```

## 2.3 A/B Testing Framework

```python
import hashlib
from scipy import stats
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    experiment_name: str
    control_model_version: str
    treatment_model_version: str
    traffic_split: float = 0.5  # 50% to treatment
    min_sample_size: int = 1000
    significance_level: float = 0.05

class ABTestManager:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.control_model = load_model(config.control_model_version)
        self.treatment_model = load_model(config.treatment_model_version)
        self.results_table = "ab_test_results"
    
    def assign_variant(self, lead_id: str) -> str:
        """Deterministic assignment for consistency"""
        hash_value = int(hashlib.md5(lead_id.encode()).hexdigest(), 16)
        if (hash_value % 100) < (self.config.traffic_split * 100):
            return "treatment"
        return "control"
    
    def predict(self, lead_data: dict) -> dict:
        lead_id = lead_data["lead_id"]
        variant = self.assign_variant(lead_id)
        
        if variant == "control":
            score = self.control_model.predict(lead_data)
        else:
            score = self.treatment_model.predict(lead_data)
        
        # Log for analysis
        self.log_experiment_event({
            "experiment_name": self.config.experiment_name,
            "lead_id": lead_id,
            "variant": variant,
            "score": float(score),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "score": float(score),
            "variant": variant,
            "experiment": self.config.experiment_name
        }
    
    def analyze_results(self, include_conversions: bool = True):
        """Statistical analysis of A/B test results"""
        # Get results with conversion outcomes
        query = f"""
            SELECT 
                r.variant,
                r.lead_id,
                r.score,
                COALESCE(c.converted, 0) as converted
            FROM {self.results_table} r
            LEFT JOIN conversions c ON r.lead_id = c.lead_id
            WHERE r.experiment_name = '{self.config.experiment_name}'
        """
        df = spark.sql(query).toPandas()
        
        control = df[df["variant"] == "control"]
        treatment = df[df["variant"] == "treatment"]
        
        # Conversion rate comparison
        control_rate = control["converted"].mean()
        treatment_rate = treatment["converted"].mean()
        
        # Chi-square test
        contingency = [
            [control["converted"].sum(), len(control) - control["converted"].sum()],
            [treatment["converted"].sum(), len(treatment) - treatment["converted"].sum()]
        ]
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Lift calculation
        lift = (treatment_rate - control_rate) / control_rate * 100 if control_rate > 0 else 0
        
        # Confidence interval for lift
        se = np.sqrt(
            treatment_rate * (1 - treatment_rate) / len(treatment) +
            control_rate * (1 - control_rate) / len(control)
        )
        ci_lower = (treatment_rate - control_rate - 1.96 * se) / control_rate * 100
        ci_upper = (treatment_rate - control_rate + 1.96 * se) / control_rate * 100
        
        return {
            "control_size": len(control),
            "treatment_size": len(treatment),
            "control_conversion_rate": round(control_rate * 100, 2),
            "treatment_conversion_rate": round(treatment_rate * 100, 2),
            "lift_percentage": round(lift, 2),
            "lift_ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
            "p_value": round(p_value, 4),
            "is_significant": p_value < self.config.significance_level,
            "recommendation": self.get_recommendation(p_value, lift),
            "sample_size_sufficient": min(len(control), len(treatment)) >= self.config.min_sample_size
        }
    
    def get_recommendation(self, p_value, lift):
        if p_value >= self.config.significance_level:
            return "CONTINUE_TEST - Not yet statistically significant"
        elif lift > 0:
            return "PROMOTE_TREATMENT - Significant positive lift"
        else:
            return "KEEP_CONTROL - Treatment performed worse"
```

## 2.4 Success Metrics

| Metric Category | Metric | Target | Measurement |
|----------------|--------|--------|-------------|
| **Model Performance** | AUC-ROC | > 0.80 | Daily calculation |
| **Model Performance** | Precision@10% | > 0.70 | Weekly review |
| **Business Metrics** | Conversion Rate | +5% lift | A/B test |
| **Business Metrics** | Lead-to-Close Time | -10% | CRM tracking |
| **Technical** | Latency P95 | < 200ms | Real-time monitoring |
| **Technical** | Error Rate | < 0.1% | API logs |

## 2.5 Protecting Business Operations

```python
class SafetyGuardrails:
    """Ensure testing doesn't harm business"""
    
    def __init__(self):
        self.guardrail_metrics = {
            "min_conversion_rate": 0.08,  # Don't drop below 8%
            "max_latency_p95_ms": 500,
            "max_error_rate": 0.01,
            "min_daily_predictions": 100
        }
    
    def check_guardrails(self) -> dict:
        """Check all guardrails and trigger alerts/rollback if needed"""
        current_metrics = self.get_current_metrics()
        violations = []
        
        if current_metrics["conversion_rate"] < self.guardrail_metrics["min_conversion_rate"]:
            violations.append({
                "metric": "conversion_rate",
                "current": current_metrics["conversion_rate"],
                "threshold": self.guardrail_metrics["min_conversion_rate"],
                "action": "ROLLBACK"
            })
        
        if current_metrics["latency_p95"] > self.guardrail_metrics["max_latency_p95_ms"]:
            violations.append({
                "metric": "latency_p95",
                "current": current_metrics["latency_p95"],
                "threshold": self.guardrail_metrics["max_latency_p95_ms"],
                "action": "ALERT"
            })
        
        if current_metrics["error_rate"] > self.guardrail_metrics["max_error_rate"]:
            violations.append({
                "metric": "error_rate",
                "current": current_metrics["error_rate"],
                "threshold": self.guardrail_metrics["max_error_rate"],
                "action": "ROLLBACK"
            })
        
        if violations:
            self.handle_violations(violations)
        
        return {
            "status": "HEALTHY" if not violations else "VIOLATIONS_DETECTED",
            "violations": violations,
            "current_metrics": current_metrics
        }
    
    def handle_violations(self, violations):
        """Handle guardrail violations"""
        for v in violations:
            if v["action"] == "ROLLBACK":
                self.trigger_automatic_rollback()
                self.send_critical_alert(v)
            elif v["action"] == "ALERT":
                self.send_warning_alert(v)
```

---

# 3. Monitoring Plan

## 3.1 Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│   │   Model      │   │   Data       │   │   Business   │   │  Technical │  │
│   │   Metrics    │   │   Metrics    │   │   Metrics    │   │  Metrics   │  │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └─────┬──────┘  │
│          │                  │                  │                 │          │
│          └──────────────────┼──────────────────┼─────────────────┘          │
│                             ▼                                                │
│                    ┌──────────────────┐                                     │
│                    │  Metrics Store   │                                     │
│                    │  (Delta Lake)    │                                     │
│                    └────────┬─────────┘                                     │
│                             │                                                │
│              ┌──────────────┼──────────────┐                                │
│              ▼              ▼              ▼                                │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐                           │
│       │Dashboard │   │ Alerting │   │ Reports  │                           │
│       │(Grafana) │   │(PagerDuty│   │ (Email)  │                           │
│       └──────────┘   └──────────┘   └──────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Data Drift Monitoring

```python
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DataDriftMonitor:
    """Monitor for data distribution changes"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.numerical_columns = reference_data.select_dtypes(include=[np.number]).columns
        self.categorical_columns = reference_data.select_dtypes(include=['object', 'category']).columns
        
        # Thresholds
        self.psi_threshold = 0.2  # Population Stability Index
        self.ks_threshold = 0.1   # Kolmogorov-Smirnov statistic
    
    def calculate_psi(self, reference: np.array, current: np.array, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate percentages for each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Add small constant to avoid division by zero
        ref_pct = (ref_counts + 1) / (len(reference) + bins)
        cur_pct = (cur_counts + 1) / (len(current) + bins)
        
        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return psi
    
    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """Detect drift across all features"""
        drift_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "features": {},
            "overall_drift_detected": False,
            "drifted_features": []
        }
        
        # Check numerical features
        for col in self.numerical_columns:
            if col in current_data.columns:
                psi = self.calculate_psi(
                    self.reference_data[col].dropna().values,
                    current_data[col].dropna().values
                )
                ks_stat, ks_pvalue = ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_detected = psi > self.psi_threshold or ks_stat > self.ks_threshold
                
                drift_results["features"][col] = {
                    "type": "numerical",
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pvalue, 4),
                    "drift_detected": drift_detected
                }
                
                if drift_detected:
                    drift_results["drifted_features"].append(col)
        
        # Check categorical features
        for col in self.categorical_columns:
            if col in current_data.columns:
                drift_detected, chi2_stat = self.check_categorical_drift(
                    self.reference_data[col],
                    current_data[col]
                )
                
                drift_results["features"][col] = {
                    "type": "categorical",
                    "chi2_statistic": round(chi2_stat, 4),
                    "drift_detected": drift_detected
                }
                
                if drift_detected:
                    drift_results["drifted_features"].append(col)
        
        drift_results["overall_drift_detected"] = len(drift_results["drifted_features"]) > 0
        drift_results["drift_severity"] = self.calculate_severity(drift_results)
        
        return drift_results
    
    def check_categorical_drift(self, reference: pd.Series, current: pd.Series):
        """Chi-square test for categorical drift"""
        # Get value counts
        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        contingency = np.array([ref_aligned, cur_aligned])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        return p_value < 0.05, chi2
    
    def calculate_severity(self, drift_results: dict) -> str:
        """Calculate overall drift severity"""
        drifted_count = len(drift_results["drifted_features"])
        total_features = len(drift_results["features"])
        drift_ratio = drifted_count / total_features if total_features > 0 else 0
        
        if drift_ratio > 0.5:
            return "CRITICAL"
        elif drift_ratio > 0.2:
            return "HIGH"
        elif drift_ratio > 0:
            return "MEDIUM"
        return "LOW"
```

## 3.3 Prediction Drift Monitoring

```python
class PredictionDriftMonitor:
    """Monitor for changes in prediction distribution"""
    
    def __init__(self, reference_predictions: np.array):
        self.reference_predictions = reference_predictions
        self.reference_mean = np.mean(reference_predictions)
        self.reference_std = np.std(reference_predictions)
        self.reference_distribution = np.histogram(reference_predictions, bins=20)[0]
    
    def detect_prediction_drift(self, current_predictions: np.array) -> dict:
        """Detect drift in predictions"""
        current_mean = np.mean(current_predictions)
        current_std = np.std(current_predictions)
        
        # KS test
        ks_stat, ks_pvalue = ks_2samp(self.reference_predictions, current_predictions)
        
        # Mean shift detection
        mean_shift = abs(current_mean - self.reference_mean)
        mean_shift_pct = mean_shift / self.reference_mean * 100 if self.reference_mean != 0 else 0
        
        # Distribution comparison
        current_distribution = np.histogram(current_predictions, bins=20)[0]
        distribution_correlation = np.corrcoef(
            self.reference_distribution, 
            current_distribution
        )[0, 1]
        
        drift_detected = (
            ks_stat > 0.1 or 
            mean_shift_pct > 10 or 
            distribution_correlation < 0.9
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(current_mean, 4),
            "mean_shift_pct": round(mean_shift_pct, 2),
            "reference_std": round(self.reference_std, 4),
            "current_std": round(current_std, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "distribution_correlation": round(distribution_correlation, 4),
            "drift_detected": drift_detected,
            "severity": "HIGH" if ks_stat > 0.2 else ("MEDIUM" if drift_detected else "LOW")
        }
```

## 3.4 Latency & Throughput Monitoring

```python
import time
from collections import deque
from threading import Lock

class PerformanceMonitor:
    """Monitor API performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
        self.lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def record_request(self, latency_ms: float, is_error: bool = False):
        """Record a request's metrics"""
        with self.lock:
            self.latencies.append(latency_ms)
            self.request_count += 1
            if is_error:
                self.error_count += 1
    
    def get_metrics(self) -> dict:
        """Get current performance metrics"""
        with self.lock:
            if not self.latencies:
                return {"status": "no_data"}
            
            latencies_sorted = sorted(self.latencies)
            elapsed_seconds = time.time() - self.start_time
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "latency": {
                    "p50_ms": round(np.percentile(latencies_sorted, 50), 2),
                    "p95_ms": round(np.percentile(latencies_sorted, 95), 2),
                    "p99_ms": round(np.percentile(latencies_sorted, 99), 2),
                    "mean_ms": round(np.mean(latencies_sorted), 2),
                    "max_ms": round(max(latencies_sorted), 2)
                },
                "throughput": {
                    "total_requests": self.request_count,
                    "requests_per_second": round(self.request_count / elapsed_seconds, 2),
                    "requests_per_minute": round(self.request_count / elapsed_seconds * 60, 2)
                },
                "errors": {
                    "total_errors": self.error_count,
                    "error_rate": round(self.error_count / self.request_count * 100, 4) if self.request_count > 0 else 0
                },
                "sla_compliance": {
                    "p95_under_200ms": latencies_sorted[int(len(latencies_sorted) * 0.95)] < 200,
                    "error_rate_under_1pct": (self.error_count / self.request_count * 100) < 1 if self.request_count > 0 else True
                }
            }
```

## 3.5 Business Performance Monitoring

```python
class BusinessMetricsMonitor:
    """Monitor business KPIs related to model performance"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def calculate_conversion_metrics(self, start_date: str, end_date: str) -> dict:
        """Calculate conversion-related business metrics"""
        query = f"""
            WITH scored_leads AS (
                SELECT 
                    p.lead_id,
                    p.score,
                    p.score_bucket,
                    p.scored_at,
                    COALESCE(c.converted, 0) as converted,
                    c.conversion_date,
                    c.deal_value
                FROM predictions p
                LEFT JOIN conversions c ON p.lead_id = c.lead_id
                WHERE p.scored_at BETWEEN '{start_date}' AND '{end_date}'
            )
            SELECT 
                score_bucket,
                COUNT(*) as total_leads,
                SUM(converted) as conversions,
                AVG(converted) as conversion_rate,
                AVG(deal_value) as avg_deal_value,
                SUM(deal_value) as total_revenue
            FROM scored_leads
            GROUP BY score_bucket
            ORDER BY conversion_rate DESC
        """
        
        results = self.spark.sql(query).toPandas()
        
        # Calculate model lift
        overall_conversion = results["conversions"].sum() / results["total_leads"].sum()
        hot_leads = results[results["score_bucket"] == "Hot"]
        hot_conversion = hot_leads["conversion_rate"].values[0] if len(hot_leads) > 0 else 0
        
        return {
            "period": {"start": start_date, "end": end_date},
            "by_bucket": results.to_dict(orient="records"),
            "overall_metrics": {
                "total_leads_scored": int(results["total_leads"].sum()),
                "total_conversions": int(results["conversions"].sum()),
                "overall_conversion_rate": round(overall_conversion * 100, 2),
                "total_revenue": round(results["total_revenue"].sum(), 2),
                "model_lift": round((hot_conversion / overall_conversion - 1) * 100, 2) if overall_conversion > 0 else 0
            },
            "model_effectiveness": {
                "hot_lead_conversion_rate": round(hot_conversion * 100, 2),
                "prioritization_working": hot_conversion > overall_conversion
            }
        }
    
    def check_score_calibration(self) -> dict:
        """Check if scores are well-calibrated with actual conversions"""
        query = """
            SELECT 
                FLOOR(score * 10) / 10 as score_decile,
                AVG(converted) as actual_conversion_rate,
                COUNT(*) as sample_size
            FROM predictions p
            JOIN conversions c ON p.lead_id = c.lead_id
            WHERE p.scored_at >= DATE_SUB(CURRENT_DATE, 30)
            GROUP BY FLOOR(score * 10) / 10
            ORDER BY score_decile
        """
        
        results = self.spark.sql(query).toPandas()
        
        # Perfect calibration: predicted probability = actual conversion rate
        results["calibration_error"] = abs(results["score_decile"] - results["actual_conversion_rate"])
        
        return {
            "calibration_by_decile": results.to_dict(orient="records"),
            "mean_calibration_error": round(results["calibration_error"].mean(), 4),
            "is_well_calibrated": results["calibration_error"].mean() < 0.1
        }
```

## 3.6 Alerting & Logging Implementation

```python
import logging
from enum import Enum
from typing import Optional

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertManager:
    """Centralized alerting system"""
    
    def __init__(self):
        self.alert_channels = {
            AlertSeverity.INFO: ["slack_general"],
            AlertSeverity.WARNING: ["slack_ml_alerts", "email_team"],
            AlertSeverity.CRITICAL: ["slack_ml_alerts", "pagerduty", "email_oncall"]
        }
        
        self.alert_thresholds = {
            "data_drift_psi": (0.2, AlertSeverity.WARNING),
            "data_drift_psi_critical": (0.4, AlertSeverity.CRITICAL),
            "prediction_drift_ks": (0.1, AlertSeverity.WARNING),
            "latency_p95_ms": (200, AlertSeverity.WARNING),
            "latency_p95_ms_critical": (500, AlertSeverity.CRITICAL),
            "error_rate_pct": (1.0, AlertSeverity.WARNING),
            "error_rate_pct_critical": (5.0, AlertSeverity.CRITICAL),
            "conversion_rate_drop_pct": (10, AlertSeverity.WARNING),
            "conversion_rate_drop_pct_critical": (20, AlertSeverity.CRITICAL)
        }
    
    def send_alert(
        self, 
        severity: AlertSeverity, 
        title: str, 
        message: str,
        metrics: Optional[dict] = None
    ):
        """Send alert to appropriate channels"""
        alert_payload = {
            "severity": severity.value,
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "dashboard_link": "https://monitoring.rakez.com/lead-scoring"
        }
        
        channels = self.alert_channels[severity]
        
        for channel in channels:
            if channel.startswith("slack"):
                self.send_slack_alert(channel, alert_payload)
            elif channel == "pagerduty":
                self.send_pagerduty_alert(alert_payload)
            elif channel.startswith("email"):
                self.send_email_alert(channel, alert_payload)
        
        # Always log
        self.log_alert(alert_payload)
    
    def send_slack_alert(self, channel: str, payload: dict):
        """Send to Slack"""
        color = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
        
        slack_message = {
            "channel": channel.replace("slack_", "#"),
            "attachments": [{
                "color": color[payload["severity"]],
                "title": f"🚨 {payload['title']}",
                "text": payload["message"],
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in payload["metrics"].items()
                ],
                "footer": f"<{payload['dashboard_link']}|View Dashboard>",
                "ts": datetime.utcnow().timestamp()
            }]
        }
        # Send via Slack API
        requests.post(SLACK_WEBHOOK_URL, json=slack_message)
    
    def send_pagerduty_alert(self, payload: dict):
        """Send to PagerDuty for critical alerts"""
        pd_payload = {
            "routing_key": PAGERDUTY_ROUTING_KEY,
            "event_action": "trigger",
            "payload": {
                "summary": payload["title"],
                "severity": "critical",
                "source": "lead-scoring-model",
                "custom_details": payload
            }
        }
        requests.post("https://events.pagerduty.com/v2/enqueue", json=pd_payload)


# Logging Configuration
def setup_logging():
    """Configure structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add structured logging for metrics
    class MetricsLogger:
        def __init__(self):
            self.logger = logging.getLogger("metrics")
        
        def log_prediction(self, lead_id: str, score: float, latency_ms: float, model_version: str):
            self.logger.info(
                "prediction",
                extra={
                    "lead_id": lead_id,
                    "score": score,
                    "latency_ms": latency_ms,
                    "model_version": model_version,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        def log_drift_check(self, drift_results: dict):
            self.logger.info(
                "drift_check",
                extra=drift_results
            )
    
    return MetricsLogger()
```

## 3.7 Investigating Performance Issues

When the sales team reports that lead scores are no longer effective:

```python
class ModelInvestigator:
    """Systematic investigation of model performance issues"""
    
    def __init__(self):
        self.data_drift_monitor = DataDriftMonitor()
        self.prediction_drift_monitor = PredictionDriftMonitor()
        self.business_monitor = BusinessMetricsMonitor()
    
    def run_full_investigation(self) -> dict:
        """Complete investigation checklist"""
        investigation = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "investigating",
            "findings": [],
            "root_cause": None,
            "recommendations": []
        }
        
        # STEP 1: Check Data Quality & Drift
        print("Step 1: Checking data quality and drift...")
        data_issues = self.check_data_layer()
        investigation["data_layer"] = data_issues
        if data_issues["issues_found"]:
            investigation["findings"].extend(data_issues["findings"])
        
        # STEP 2: Check Model Performance
        print("Step 2: Checking model performance...")
        model_issues = self.check_model_layer()
        investigation["model_layer"] = model_issues
        if model_issues["issues_found"]:
            investigation["findings"].extend(model_issues["findings"])
        
        # STEP 3: Check Integration & Pipeline
        print("Step 3: Checking integration and pipeline...")
        integration_issues = self.check_integration_layer()
        investigation["integration_layer"] = integration_issues
        if integration_issues["issues_found"]:
            investigation["findings"].extend(integration_issues["findings"])
        
        # STEP 4: Check Business Context
        print("Step 4: Checking business context...")
        business_issues = self.check_business_layer()
        investigation["business_layer"] = business_issues
        if business_issues["issues_found"]:
            investigation["findings"].extend(business_issues["findings"])
        
        # Determine root cause
        investigation["root_cause"] = self.determine_root_cause(investigation)
        investigation["recommendations"] = self.generate_recommendations(investigation)
        investigation["status"] = "complete"
        
        return investigation
    
    def check_data_layer(self) -> dict:
        """Check for data-related issues"""
        results = {"issues_found": False, "findings": [], "checks": {}}
        
        # Check 1: Data freshness
        latest_data_timestamp = self.get_latest_data_timestamp()
        hours_since_update = (datetime.utcnow() - latest_data_timestamp).total_seconds() / 3600
        
        results["checks"]["data_freshness"] = {
            "latest_update": latest_data_timestamp.isoformat(),
            "hours_since_update": round(hours_since_update, 2),
            "is_stale": hours_since_update > 24
        }
        
        if hours_since_update > 24:
            results["issues_found"] = True
            results["findings"].append("DATA_STALE: Feature data hasn't been updated in 24+ hours")
        
        # Check 2: Missing values
        missing_rates = self.check_missing_values()
        high_missing = {k: v for k, v in missing_rates.items() if v > 0.1}
        
        results["checks"]["missing_values"] = {
            "by_feature": missing_rates,
            "high_missing_features": list(high_missing.keys())
        }
        
        if high_missing:
            results["issues_found"] = True
            results["findings"].append(f"HIGH_MISSING_VALUES: Features with >10% missing: {list(high_missing.keys())}")
        
        # Check 3: Data drift
        drift_results = self.data_drift_monitor.detect_drift(self.get_current_data())
        
        results["checks"]["data_drift"] = drift_results
        
        if drift_results["overall_drift_detected"]:
            results["issues_found"] = True
            results["findings"].append(f"DATA_DRIFT_DETECTED: Drift in features: {drift_results['drifted_features']}")
        
        # Check 4: New categories
        new_categories = self.check_new_categories()
        
        results["checks"]["new_categories"] = new_categories
        
        if new_categories:
            results["issues_found"] = True
            results["findings"].append(f"NEW_CATEGORIES: Unseen categories in: {list(new_categories.keys())}")
        
        return results
    
    def check_model_layer(self) -> dict:
        """Check for model-related issues"""
        results = {"issues_found": False, "findings": [], "checks": {}}
        
        # Check 1: Prediction distribution drift
        pred_drift = self.prediction_drift_monitor.detect_prediction_drift(
            self.get_recent_predictions()
        )
        
        results["checks"]["prediction_drift"] = pred_drift
        
        if pred_drift["drift_detected"]:
            results["issues_found"] = True
            results["findings"].append(f"PREDICTION_DRIFT: Score distribution has shifted significantly")
        
        # Check 2: Score calibration
        calibration = self.business_monitor.check_score_calibration()
        
        results["checks"]["calibration"] = calibration
        
        if not calibration["is_well_calibrated"]:
            results["issues_found"] = True
            results["findings"].append("MISCALIBRATION: Predicted scores don't match actual conversion rates")
        
        # Check 3: Feature importance shift
        importance_shift = self.check_feature_importance_shift()
        
        results["checks"]["importance_shift"] = importance_shift
        
        if importance_shift["significant_shift"]:
            results["issues_found"] = True
            results["findings"].append(f"IMPORTANCE_SHIFT: Key features have changed importance")
        
        return results
    
    def check_integration_layer(self) -> dict:
        """Check for integration/pipeline issues"""
        results = {"issues_found": False, "findings": [], "checks": {}}
        
        # Check 1: API errors
        error_rate = self.get_api_error_rate()
        
        results["checks"]["api_errors"] = {
            "error_rate_pct": error_rate,
            "is_high": error_rate > 1.0
        }
        
        if error_rate > 1.0:
            results["issues_found"] = True
            results["findings"].append(f"HIGH_ERROR_RATE: API error rate is {error_rate}%")
        
        # Check 2: Feature pipeline health
        pipeline_status = self.check_feature_pipeline()
        
        results["checks"]["feature_pipeline"] = pipeline_status
        
        if not pipeline_status["all_healthy"]:
            results["issues_found"] = True
            results["findings"].append(f"PIPELINE_ISSUES: Problems with: {pipeline_status['unhealthy_pipelines']}")
        
        # Check 3: CRM integration
        crm_sync = self.check_crm_sync()
        
        results["checks"]["crm_sync"] = crm_sync
        
        if not crm_sync["is_syncing"]:
            results["issues_found"] = True
            results["findings"].append("CRM_SYNC_BROKEN: Scores may not be reaching the sales team")
        
        return results
    
    def check_business_layer(self) -> dict:
        """Check for business context changes"""
        results = {"issues_found": False, "findings": [], "checks": {}}
        
        # Check 1: Lead source distribution change
        lead_source_shift = self.check_lead_source_distribution()
        
        results["checks"]["lead_source_shift"] = lead_source_shift
        
        if lead_source_shift["significant_change"]:
            results["issues_found"] = True
            results["findings"].append("LEAD_SOURCE_SHIFT: Lead source mix has changed significantly")
        
        # Check 2: Conversion rate trends
        conversion_trend = self.analyze_conversion_trends()
        
        results["checks"]["conversion_trend"] = conversion_trend
        
        if conversion_trend["declining"]:
            results["issues_found"] = True
            results["findings"].append(f"CONVERSION_DECLINING: Overall conversion down {conversion_trend['decline_pct']}%")
        
        # Check 3: Market/seasonal factors
        external_factors = self.check_external_factors()
        
        results["checks"]["external_factors"] = external_factors
        
        return results
    
    def determine_root_cause(self, investigation: dict) -> str:
        """Determine the most likely root cause"""
        findings = investigation["findings"]
        
        # Priority order for root cause
        if any("DATA_STALE" in f for f in findings):
            return "DATA_PIPELINE_FAILURE"
        if any("CRM_SYNC_BROKEN" in f for f in findings):
            return "INTEGRATION_FAILURE"
        if any("DATA_DRIFT" in f for f in findings):
            return "DATA_DRIFT_REQUIRES_RETRAINING"
        if any("MISCALIBRATION" in f for f in findings):
            return "MODEL_DEGRADATION"
        if any("LEAD_SOURCE_SHIFT" in f for f in findings):
            return "BUSINESS_CONTEXT_CHANGE"
        if any("CONVERSION_DECLINING" in f for f in findings):
            return "MARKET_CONDITIONS_CHANGE"
        
        return "UNKNOWN_REQUIRES_FURTHER_INVESTIGATION"
    
    def generate_recommendations(self, investigation: dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        root_cause = investigation["root_cause"]
        
        if root_cause == "DATA_PIPELINE_FAILURE":
            recommendations.append("IMMEDIATE: Investigate and fix data pipeline")
            recommendations.append("SHORT_TERM: Add pipeline monitoring and alerts")
        
        elif root_cause == "INTEGRATION_FAILURE":
            recommendations.append("IMMEDIATE: Check CRM integration and sync status")
            recommendations.append("SHORT_TERM: Implement end-to-end integration tests")
        
        elif root_cause == "DATA_DRIFT_REQUIRES_RETRAINING":
            recommendations.append("IMMEDIATE: Trigger model retraining with recent data")
            recommendations.append("SHORT_TERM: Implement automated drift-based retraining")
        
        elif root_cause == "MODEL_DEGRADATION":
            recommendations.append("IMMEDIATE: Consider rollback to previous model version")
            recommendations.append("SHORT_TERM: Investigate feature engineering improvements")
        
        elif root_cause == "BUSINESS_CONTEXT_CHANGE":
            recommendations.append("IMMEDIATE: Communicate with sales team about changes")
            recommendations.append("MEDIUM_TERM: Retrain model with new lead source data")
        
        return recommendations
```

---

# 4. Automation, Reproducibility & Retraining

## 4.1 Reproducibility Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REPRODUCIBILITY COMPONENTS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │  Data Snapshots │  │  Code Version   │  │  Environment    │            │
│   │  (Delta Lake)   │  │  (Git)          │  │  (requirements) │            │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│            │                    │                    │                      │
│            └────────────────────┼────────────────────┘                      │
│                                 ▼                                           │
│                    ┌────────────────────────┐                               │
│                    │    MLflow Tracking     │                               │
│                    │    - Parameters        │                               │
│                    │    - Metrics           │                               │
│                    │    - Artifacts         │                               │
│                    │    - Model             │                               │
│                    └────────────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Versioning

```python
from delta.tables import DeltaTable
import hashlib

class DataVersionManager:
    """Manage data snapshots for reproducibility"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.snapshots_path = f"{base_path}/snapshots"
    
    def create_training_snapshot(self, df: DataFrame, version_tag: str) -> dict:
        """Create immutable training data snapshot"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = f"{self.snapshots_path}/{version_tag}_{timestamp}"
        
        # Calculate data hash for verification
        data_hash = self.calculate_data_hash(df)
        
        # Save as Delta table
        df.write.format("delta").mode("overwrite").save(snapshot_path)
        
        # Create metadata
        metadata = {
            "version_tag": version_tag,
            "timestamp": timestamp,
            "snapshot_path": snapshot_path,
            "row_count": df.count(),
            "column_count": len(df.columns),
            "columns": df.columns,
            "data_hash": data_hash
        }
        
        # Log to MLflow
        mlflow.log_params({
            "data_snapshot_path": snapshot_path,
            "data_version": version_tag,
            "data_hash": data_hash[:16],
            "data_row_count": metadata["row_count"]
        })
        
        return metadata
    
    def calculate_data_hash(self, df: DataFrame) -> str:
        """Calculate hash of dataframe for verification"""
        # Convert to pandas and hash
        pandas_df = df.toPandas()
        return hashlib.sha256(
            pandas_df.to_json().encode()
        ).hexdigest()
    
    def load_snapshot(self, snapshot_path: str) -> DataFrame:
        """Load a specific data snapshot"""
        return spark.read.format("delta").load(snapshot_path)
    
    def verify_snapshot(self, snapshot_path: str, expected_hash: str) -> bool:
        """Verify snapshot integrity"""
        df = self.load_snapshot(snapshot_path)
        actual_hash = self.calculate_data_hash(df)
        return actual_hash == expected_hash
```

### Environment Management

```yaml
# config/training_config.yaml
experiment:
  name: "lead_scoring_v2"
  description: "Lead scoring model with improved features"
  
data:
  source_table: "leads.features_silver"
  snapshot_version: "v2.1"
  train_test_split: 0.2
  random_seed: 42
  
features:
  numerical:
    - engagement_score
    - website_visits
    - email_opens
    - days_since_first_contact
    - company_size
  categorical:
    - lead_source
    - industry
    - region
  
model:
  algorithm: "xgboost"
  hyperparameters:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 200
    min_child_weight: 3
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
  
training:
  cross_validation_folds: 5
  early_stopping_rounds: 50
  optimization_metric: "auc"
```

```python
# requirements.txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
mlflow==2.8.0
pyspark==3.4.1
evidently==0.4.8
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
scipy==1.11.3
```

```python
class ReproducibilityManager:
    """Ensure experiments are reproducible"""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def setup_experiment(self) -> str:
        """Setup MLflow experiment with full tracking"""
        experiment_name = self.config["experiment"]["name"]
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log configuration
            mlflow.log_artifact(self.config_path)
            
            # Log environment
            mlflow.log_artifact("requirements.txt")
            mlflow.log_param("python_version", sys.version)
            
            # Log git info
            mlflow.log_param("git_commit", self.get_git_commit())
            mlflow.log_param("git_branch", self.get_git_branch())
            mlflow.log_param("git_repo", self.get_git_repo())
            
            # Log all config parameters
            self.log_config_params(self.config)
            
            return run.info.run_id
    
    def get_git_commit(self) -> str:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    
    def get_git_branch(self) -> str:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
    
    def log_config_params(self, config: dict, prefix: str = ""):
        """Recursively log all config parameters"""
        for key, value in config.items():
            param_name = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                self.log_config_params(value, f"{param_name}.")
            else:
                mlflow.log_param(param_name, value)
```

## 4.2 CI/CD Pipeline

```yaml
# .github/workflows/ml-cicd.yaml
name: ML Model CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'config/**'
      - 'tests/**'
  pull_request:
    branches: [main]

env:
  DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
  DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # Stage 1: Code Quality & Unit Tests
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov black flake8
      
      - name: Code formatting check
        run: black --check src/
      
      - name: Linting
        run: flake8 src/ --max-line-length=120
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # Stage 2: Model Validation
  validate-model:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Validate model artifacts
        run: python scripts/validate_model.py
      
      - name: Run model quality checks
        run: |
          python scripts/model_quality_checks.py \
            --min-auc 0.75 \
            --max-latency-ms 200

  # Stage 3: Integration Tests
  integration-test:
    needs: validate-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run integration tests
        run: |
          docker-compose up -d
          pytest tests/integration/ -v
          docker-compose down

  # Stage 4: Deploy to Staging
  deploy-staging:
    needs: integration-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy model to staging
        run: |
          python scripts/deploy_model.py \
            --stage Staging \
            --endpoint lead-scoring-staging
      
      - name: Run smoke tests
        run: python scripts/smoke_tests.py --endpoint staging
      
      - name: Start shadow deployment
        run: python scripts/start_shadow_mode.py --duration-hours 24

  # Stage 5: Deploy to Production
  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Check shadow mode results
        run: python scripts/check_shadow_results.py --min-correlation 0.95
      
      - name: Deploy to production
        run: |
          python scripts/deploy_model.py \
            --stage Production \
            --endpoint lead-scoring-production
      
      - name: Health check
        run: python scripts/health_check.py --endpoint production
      
      - name: Notify deployment
        run: |
          python scripts/notify.py \
            --channel ml-deployments \
            --message "Model v${{ github.sha }} deployed to production"
```

### Deployment Scripts

```python
# scripts/deploy_model.py
import argparse
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

def deploy_model(model_name: str, stage: str, endpoint: str):
    """Deploy model to specified stage and endpoint"""
    client = MlflowClient()
    w = WorkspaceClient()
    
    # Get latest model version
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda x: int(x.version))
    
    # Transition to target stage
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage=stage
    )
    
    # Update serving endpoint
    w.serving_endpoints.update_config_and_wait(
        name=endpoint,
        served_models=[{
            "model_name": model_name,
            "model_version": latest.version,
            "workload_size": "Small"
        }]
    )
    
    print(f"Deployed {model_name} v{latest.version} to {stage}")
    return latest.version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="lead-scoring-production")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    
    deploy_model(args.model_name, args.stage, args.endpoint)
```

## 4.3 Automated Retraining Pipeline

```python
class RetrainingOrchestrator:
    """Orchestrate automated model retraining"""
    
    def __init__(self):
        self.config = {
            "scheduled_retrain_days": 7,
            "drift_psi_threshold": 0.2,
            "performance_auc_threshold": 0.75,
            "min_new_samples": 5000
        }
        self.alert_manager = AlertManager()
    
    def check_retrain_triggers(self) -> dict:
        """Check all retraining triggers"""
        triggers = {
            "scheduled": self.check_scheduled_trigger(),
            "drift_detected": self.check_drift_trigger(),
            "performance_degraded": self.check_performance_trigger(),
            "new_data_available": self.check_new_data_trigger()
        }
        
        return {
            "should_retrain": any(triggers.values()),
            "triggers": triggers,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def check_scheduled_trigger(self) -> bool:
        """Weekly scheduled retraining"""
        last_retrain = self.get_last_retrain_date()
        days_since = (datetime.utcnow() - last_retrain).days
        return days_since >= self.config["scheduled_retrain_days"]
    
    def check_drift_trigger(self) -> bool:
        """Check if data drift exceeds threshold"""
        drift_metrics = self.get_latest_drift_metrics()
        max_psi = max(drift_metrics.values())
        return max_psi > self.config["drift_psi_threshold"]
    
    def check_performance_trigger(self) -> bool:
        """Check if model performance has degraded"""
        current_auc = self.get_current_model_auc()
        return current_auc < self.config["performance_auc_threshold"]
    
    def check_new_data_trigger(self) -> bool:
        """Check if enough new labeled data is available"""
        new_samples = self.count_new_labeled_samples()
        return new_samples >= self.config["min_new_samples"]
    
    def execute_retraining(self) -> dict:
        """Execute the full retraining pipeline"""
        trigger_check = self.check_retrain_triggers()
        
        if not trigger_check["should_retrain"]:
            return {"status": "skipped", "reason": "No retrain triggers active"}
        
        try:
            # Step 1: Prepare data
            print("Step 1: Preparing training data...")
            train_data, test_data = self.prepare_training_data()
            
            # Step 2: Create data snapshot
            print("Step 2: Creating data snapshot...")
            snapshot_info = self.data_manager.create_training_snapshot(
                train_data, 
                version_tag=f"retrain_{datetime.utcnow().strftime('%Y%m%d')}"
            )
            
            # Step 3: Train new model
            print("Step 3: Training new model...")
            with mlflow.start_run(run_name=f"retrain_{datetime.utcnow().isoformat()}"):
                model, metrics = self.train_model(train_data, test_data)
                
                # Log everything
                mlflow.log_metrics(metrics)
                mlflow.log_params({"trigger": str(trigger_check["triggers"])})
                mlflow.sklearn.log_model(model, "model")
            
            # Step 4: Validate new model
            print("Step 4: Validating new model...")
            validation_passed = self.validate_new_model(model, metrics)
            
            if not validation_passed:
                return {
                    "status": "validation_failed",
                    "metrics": metrics,
                    "threshold_metrics": self.get_threshold_metrics()
                }
            
            # Step 5: Register model
            print("Step 5: Registering model...")
            model_version = self.register_model(model, metrics)
            
            # Step 6: Deploy to staging for shadow testing
            print("Step 6: Deploying to staging...")
            self.deploy_to_staging(model_version)
            
            # Step 7: Run shadow tests
            print("Step 7: Running shadow tests (24 hours)...")
            shadow_results = self.run_shadow_deployment(duration_hours=24)
            
            if shadow_results["passed"]:
                # Step 8: Promote to production
                print("Step 8: Promoting to production...")
                self.promote_to_production(model_version)
                
                self.alert_manager.send_alert(
                    AlertSeverity.INFO,
                    "Model Retrained Successfully",
                    f"New model v{model_version} deployed to production",
                    metrics
                )
                
                return {
                    "status": "success",
                    "model_version": model_version,
                    "metrics": metrics,
                    "triggers": trigger_check["triggers"]
                }
            else:
                return {
                    "status": "shadow_test_failed",
                    "shadow_results": shadow_results
                }
                
        except Exception as e:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "Retraining Pipeline Failed",
                str(e)
            )
            raise
    
    def validate_new_model(self, model, metrics: dict) -> bool:
        """Validate new model meets minimum requirements"""
        validations = [
            metrics["auc"] >= self.config["performance_auc_threshold"],
            metrics["auc"] >= self.get_current_model_auc() * 0.95,  # Not more than 5% worse
            metrics["precision_at_10pct"] >= 0.5
        ]
        return all(validations)
```

### Retraining Triggers Summary

| Trigger | Condition | Check Frequency | Action |
|---------|-----------|-----------------|--------|
| **Scheduled** | Every 7 days | Daily | Full retrain |
| **Data Drift** | PSI > 0.2 | Hourly | Evaluate + retrain |
| **Performance Drop** | AUC < 0.75 | Real-time | Alert + retrain |
| **New Data** | 5000+ new labeled samples | Daily | Evaluate for retrain |

---

# Bonus Section

## B.1 Monitoring Dashboard

```python
# dashboard/app.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Lead Scoring Model Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-healthy { color: #00FF00; }
    .status-warning { color: #FFCC00; }
    .status-critical { color: #FF0000; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Lead Scoring Model Dashboard")
st.markdown("Real-time monitoring for the lead scoring ML model")

# Sidebar
st.sidebar.header("Settings")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Model AUC",
        value="0.847",
        delta="+0.02",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Conversion Rate",
        value="12.3%",
        delta="+1.2%"
    )

with col3:
    st.metric(
        label="Avg Latency",
        value="45ms",
        delta="-5ms"
    )

with col4:
    st.metric(
        label="Daily Predictions",
        value="12,450",
        delta="+8%"
    )

with col5:
    st.metric(
        label="Error Rate",
        value="0.02%",
        delta="-0.01%",
        delta_color="inverse"
    )

st.markdown("---")

# Two column layout for charts
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📊 Data Drift - PSI Over Time")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    psi_data = pd.DataFrame({
        'date': dates,
        'psi': np.random.uniform(0.05, 0.25, 30)
    })
    
    fig = px.line(psi_data, x='date', y='psi', title='Population Stability Index')
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                  annotation_text="Alert Threshold")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("📈 Score Distribution Comparison")
    
    # Sample prediction distributions
    reference_scores = np.random.beta(2, 5, 1000)
    current_scores = np.random.beta(2.2, 4.8, 1000)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=reference_scores, name='Training', opacity=0.7))
    fig.add_trace(go.Histogram(x=current_scores, name='Production', opacity=0.7))
    fig.update_layout(barmode='overlay', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Conversion by Score Bucket
st.subheader("💰 Conversion Rate by Score Bucket")

bucket_data = pd.DataFrame({
    'Bucket': ['Cold (0-0.3)', 'Cool (0.3-0.5)', 'Warm (0.5-0.8)', 'Hot (0.8-1.0)'],
    'Leads': [5000, 3500, 2500, 1500],
    'Conversions': [150, 280, 500, 600],
    'Conversion Rate': [3.0, 8.0, 20.0, 40.0]
})

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(x=bucket_data['Bucket'], y=bucket_data['Leads'], name='Total Leads'),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=bucket_data['Bucket'], y=bucket_data['Conversion Rate'], 
               name='Conversion Rate %', mode='lines+markers', line=dict(color='#00FF00', width=3)),
    secondary_y=True
)

fig.update_layout(template="plotly_dark")
fig.update_yaxes(title_text="Number of Leads", secondary_y=False)
fig.update_yaxes(title_text="Conversion Rate %", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Alerts section
st.markdown("---")
st.subheader("🚨 Recent Alerts")

alerts_data = [
    {"time": "2024-12-04 10:30", "severity": "warning", "message": "PSI for 'lead_source' reached 0.18"},
    {"time": "2024-12-04 08:15", "severity": "info", "message": "Daily model performance report generated"},
    {"time": "2024-12-03 22:00", "severity": "info", "message": "Scheduled monitoring job completed"}
]

for alert in alerts_data:
    icon = "⚠️" if alert["severity"] == "warning" else "ℹ️"
    st.markdown(f"{icon} **{alert['time']}** - {alert['message']}")
```

## B.2 CRM Integration

```python
# integrations/crm_connector.py
from salesforce_api import Salesforce
from typing import List, Dict
import schedule
import time

class CRMIntegration:
    """Integrate lead scores with Salesforce CRM"""
    
    def __init__(self):
        self.sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN"),
            domain='login'  # or 'test' for sandbox
        )
    
    def push_lead_scores(self, predictions: List[Dict]):
        """Push lead scores to CRM in batch"""
        results = {"success": 0, "failed": 0, "errors": []}
        
        for pred in predictions:
            try:
                self.sf.Lead.update(pred["lead_id"], {
                    "ML_Lead_Score__c": pred["score"],
                    "ML_Score_Bucket__c": pred["bucket"],
                    "ML_Score_Confidence__c": pred.get("confidence", None),
                    "ML_Model_Version__c": pred["model_version"],
                    "ML_Scored_At__c": datetime.utcnow().isoformat()
                })
                results["success"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "lead_id": pred["lead_id"],
                    "error": str(e)
                })
        
        return results
    
    def create_priority_view(self):
        """Create or update CRM view for high-priority leads"""
        view_query = """
            SELECT Id, Name, Company, ML_Lead_Score__c, ML_Score_Bucket__c, 
                   CreatedDate, Status, LeadSource
            FROM Lead
            WHERE ML_Score_Bucket__c = 'Hot'
            AND Status != 'Converted'
            ORDER BY ML_Lead_Score__c DESC
            LIMIT 100
        """
        return self.sf.query(view_query)
    
    def setup_score_dashboard(self):
        """Create Salesforce dashboard for lead scores"""
        dashboard_config = {
            "name": "ML Lead Scoring Dashboard",
            "components": [
                {
                    "type": "chart",
                    "title": "Leads by Score Bucket",
                    "query": """
                        SELECT ML_Score_Bucket__c, COUNT(Id)
                        FROM Lead
                        WHERE ML_Lead_Score__c != null
                        GROUP BY ML_Score_Bucket__c
                    """
                },
                {
                    "type": "metric",
                    "title": "Hot Leads Today",
                    "query": """
                        SELECT COUNT(Id)
                        FROM Lead
                        WHERE ML_Score_Bucket__c = 'Hot'
                        AND ML_Scored_At__c = TODAY
                    """
                },
                {
                    "type": "table",
                    "title": "Top 10 Hot Leads",
                    "query": """
                        SELECT Name, Company, ML_Lead_Score__c, LeadSource
                        FROM Lead
                        WHERE ML_Score_Bucket__c = 'Hot'
                        ORDER BY ML_Lead_Score__c DESC
                        LIMIT 10
                    """
                }
            ]
        }
        return dashboard_config


# Scheduled sync job
def sync_scores_to_crm():
    """Scheduled job to sync scores to CRM"""
    connector = CRMIntegration()
    
    # Get recent predictions
    recent_predictions = get_predictions_since_last_sync()
    
    if recent_predictions:
        results = connector.push_lead_scores(recent_predictions)
        logging.info(f"CRM sync complete: {results}")
    
    return results

# Schedule to run every 15 minutes
schedule.every(15).minutes.do(sync_scores_to_crm)
```

## B.3 Feedback Loop Implementation

```python
# feedback/feedback_loop.py

class FeedbackCollector:
    """Collect and process feedback from sales team"""
    
    def __init__(self):
        self.feedback_table = "feedback.lead_score_feedback"
    
    def collect_feedback(self, feedback: Dict):
        """Record feedback from sales team"""
        feedback_record = {
            "feedback_id": str(uuid.uuid4()),
            "lead_id": feedback["lead_id"],
            "predicted_score": feedback["predicted_score"],
            "actual_outcome": feedback.get("outcome"),  # converted, lost, no_response
            "sales_rep_id": feedback["sales_rep_id"],
            "feedback_type": feedback["type"],  # score_too_high, score_too_low, accurate
            "comments": feedback.get("comments", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store feedback
        spark.sql(f"""
            INSERT INTO {self.feedback_table}
            VALUES (
                '{feedback_record["feedback_id"]}',
                '{feedback_record["lead_id"]}',
                {feedback_record["predicted_score"]},
                '{feedback_record["actual_outcome"]}',
                '{feedback_record["sales_rep_id"]}',
                '{feedback_record["feedback_type"]}',
                '{feedback_record["comments"]}',
                '{feedback_record["timestamp"]}'
            )
        """)
        
        return feedback_record
    
    def analyze_feedback(self, days: int = 30) -> Dict:
        """Analyze collected feedback"""
        query = f"""
            SELECT 
                feedback_type,
                COUNT(*) as count,
                AVG(predicted_score) as avg_predicted_score
            FROM {self.feedback_table}
            WHERE timestamp >= DATE_SUB(CURRENT_DATE, {days})
            GROUP BY feedback_type
        """
        
        results = spark.sql(query).toPandas()
        
        total_feedback = results["count"].sum()
        
        analysis = {
            "period_days": days,
            "total_feedback": int(total_feedback),
            "by_type": results.to_dict(orient="records"),
            "accuracy_rate": self.calculate_accuracy_rate(results),
            "recommendations": self.generate_recommendations(results)
        }
        
        return analysis
    
    def calculate_accuracy_rate(self, results: pd.DataFrame) -> float:
        """Calculate how often scores are perceived as accurate"""
        accurate = results[results["feedback_type"] == "accurate"]["count"].sum()
        total = results["count"].sum()
        return round(accurate / total * 100, 2) if total > 0 else 0
    
    def generate_recommendations(self, results: pd.DataFrame) -> List[str]:
        """Generate recommendations based on feedback"""
        recommendations = []
        
        total = results["count"].sum()
        
        for _, row in results.iterrows():
            pct = row["count"] / total * 100
            
            if row["feedback_type"] == "score_too_high" and pct > 20:
                recommendations.append(
                    f"Consider recalibrating model - {pct:.1f}% of feedback indicates scores are too high"
                )
            
            if row["feedback_type"] == "score_too_low" and pct > 20:
                recommendations.append(
                    f"Model may be undervaluing leads - {pct:.1f}% feedback says scores too low"
                )
        
        return recommendations
    
    def export_for_retraining(self) -> DataFrame:
        """Export feedback data for model retraining"""
        query = f"""
            SELECT 
                f.lead_id,
                f.predicted_score,
                f.actual_outcome,
                f.feedback_type,
                l.*
            FROM {self.feedback_table} f
            JOIN leads.features l ON f.lead_id = l.lead_id
            WHERE f.actual_outcome IS NOT NULL
        """
        
        return spark.sql(query)


# Feedback API endpoint
@app.post("/feedback")
async def submit_feedback(
    lead_id: str,
    predicted_score: float,
    feedback_type: str,
    outcome: Optional[str] = None,
    comments: Optional[str] = None,
    sales_rep_id: str = None
):
    """API endpoint for sales team to submit feedback"""
    collector = FeedbackCollector()
    
    result = collector.collect_feedback({
        "lead_id": lead_id,
        "predicted_score": predicted_score,
        "type": feedback_type,
        "outcome": outcome,
        "comments": comments,
        "sales_rep_id": sales_rep_id
    })
    
    return {"status": "success", "feedback_id": result["feedback_id"]}
```

---

# Summary

This comprehensive MLOps solution addresses all aspects of deploying and maintaining the lead scoring model:

| Component | Solution |
|-----------|----------|
| **Deployment** | MLflow + Databricks Serving with Docker containerization |
| **Versioning** | MLflow Model Registry with staging workflows |
| **Online Testing** | Shadow deployment + A/B testing framework |
| **Monitoring** | Data drift, prediction drift, latency, business metrics |
| **Alerting** | Multi-channel (Slack, PagerDuty, Email) with severity levels |
| **Reproducibility** | Data snapshots, config versioning, MLflow tracking |
| **CI/CD** | GitHub Actions with multi-stage deployment |
| **Retraining** | Automated triggers with validation and shadow testing |
| **CRM Integration** | Salesforce sync with dashboards |
| **Feedback Loop** | Sales team feedback collection and analysis |

This solution ensures the model remains accurate, reliable, and aligned with business objectives over time.

---

**Submitted by:** Maria Muneeb 
**Date:** December 2025

