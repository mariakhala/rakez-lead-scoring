# RAKEZ Lead Scoring Model - MLOps Case Study

## 📋 Overview

This repository contains a comprehensive MLOps solution for deploying, monitoring, and maintaining a lead scoring model at RAKEZ. The solution addresses all aspects of the ML lifecycle from deployment to retraining.

## 🎯 Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Main Document** | Complete case study with all answers | `RAKEZ_Lead_Scoring_Case_Study.md` |
| **Model Serving API** | FastAPI REST API for predictions | `src/model_serving.py` |
| **Monitoring** | Data drift, prediction drift, performance | `src/monitoring.py` |
| **Retraining Pipeline** | Automated model retraining | `src/retraining.py` |
| **Dashboard** | Streamlit monitoring dashboard | `dashboard/app.py` |
| **CI/CD Pipeline** | GitHub Actions workflow | `.github/workflows/ml-cicd.yaml` |
| **Configuration** | Training and deployment config | `config/training_config.yaml` |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- MLflow (optional, for experiment tracking)

### Installation

```bash
# Clone repository
git clone https://github.com/rakez/lead-scoring.git
cd lead-scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
# Start the API server
uvicorn src.model_serving:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_id": "lead-123",
    "lead_source": "Web",
    "industry": "Technology",
    "company_size": 150,
    "engagement_score": 75.0,
    "website_visits": 10,
    "email_opens": 5
  }'
```

### Run with Docker

```bash
# Build image
docker build -t lead-scoring-api .

# Run container
docker run -p 8000:8000 lead-scoring-api
```

### Run the Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │   CRM/Lead   │────▶│   API        │────▶│   Model Serving Layer    │   │
│   │   Sources    │     │   Gateway    │     │   (FastAPI/Databricks)   │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│                              │                          │                    │
│                              ▼                          ▼                    │
│                       ┌──────────────┐          ┌──────────────┐            │
│                       │   Load       │          │   MLflow     │            │
│                       │   Balancer   │          │   Registry   │            │
│                       └──────────────┘          └──────────────┘            │
│                              │                          │                    │
│                              ▼                          ▼                    │
│                    ┌──────────────────────────────────────────────┐         │
│                    │              Monitoring Layer                 │         │
│                    │  • Data Drift  • Prediction Drift  • Latency │         │
│                    │  • Business Metrics  • Alerting              │         │
│                    └──────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
lead-scoring/
├── .github/
│   └── workflows/
│       └── ml-cicd.yaml          # CI/CD pipeline
├── config/
│   └── training_config.yaml      # Model configuration
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── src/
│   ├── model_serving.py          # FastAPI application
│   ├── monitoring.py             # Monitoring utilities
│   └── retraining.py             # Retraining pipeline
├── tests/
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── RAKEZ_Lead_Scoring_Case_Study.md  # Full case study
└── README.md                     # This file
```

## 🔧 Key Features

### 1. Deployment Strategy
- MLflow Model Registry for versioning
- Databricks Model Serving (primary)
- Docker containerization (alternative)
- REST API with FastAPI
- Automatic rollback capability

### 2. Online Testing
- Shadow deployment for new models
- A/B testing framework
- Gradual canary rollouts
- Statistical significance testing

### 3. Monitoring
- **Data Drift**: PSI, KS test, chi-square
- **Prediction Drift**: Distribution comparison
- **Performance**: Latency, throughput, error rates
- **Business**: Conversion rates, model lift

### 4. Automation
- CI/CD with GitHub Actions
- Automated retraining triggers
- Data versioning and snapshots
- Reproducibility tracking

## 📈 Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Model** | AUC-ROC, Precision@K, Recall, F1 |
| **Drift** | PSI, KS Statistic, Chi-square |
| **Technical** | Latency (P50, P95, P99), Error Rate |
| **Business** | Conversion Rate, Lead-to-Close Time |

## 🚨 Alerting

| Severity | Channels | Triggers |
|----------|----------|----------|
| Info | Slack | Daily reports, deployments |
| Warning | Slack, Email | PSI > 0.2, Latency > 200ms |
| Critical | PagerDuty, Slack | Error rate > 5%, Model failure |

## 🔄 Retraining Triggers

1. **Scheduled**: Weekly (every 7 days)
2. **Drift-based**: PSI > 0.2
3. **Performance-based**: AUC < 0.75
4. **Data-based**: 5000+ new labeled samples

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📚 Documentation

- [Full Case Study](RAKEZ_Lead_Scoring_Case_Study.md) - Complete answers to all requirements
- [API Documentation](http://localhost:8000/docs) - Swagger UI (when running)
- [Configuration Guide](config/training_config.yaml) - All configurable parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📝 License

This project is part of the RAKEZ ML Engineer case study.

---

**Author:** ML Engineering Team  
**Date:** December 2024  
**Version:** 1.0.0

