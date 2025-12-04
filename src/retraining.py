"""
Lead Scoring Model Retraining Pipeline
Automated retraining with triggers, validation, and deployment
"""

# Optional MLflow import
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import logging
import yaml
import hashlib
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Configuration ==============

class RetrainingConfig:
    """Configuration for retraining pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "triggers": {
                "scheduled_retrain_days": 7,
                "drift_psi_threshold": 0.2,
                "performance_auc_threshold": 0.75,
                "min_new_samples": 5000
            },
            "training": {
                "test_size": 0.2,
                "cv_folds": 5,
                "random_seed": 42
            },
            "validation": {
                "min_auc": 0.75,
                "min_improvement": -0.05,  # Allow up to 5% degradation
                "min_precision_at_10": 0.5
            },
            "deployment": {
                "shadow_duration_hours": 24,
                "min_shadow_correlation": 0.95,
                "canary_percentage": 10
            }
        }


# ============== Data Versioning ==============

class DataVersionManager:
    """Manage data snapshots for reproducibility"""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path
        self.snapshots_path = f"{base_path}/snapshots"
        os.makedirs(self.snapshots_path, exist_ok=True)
    
    def create_snapshot(
        self, 
        df: pd.DataFrame, 
        version_tag: str
    ) -> Dict:
        """Create immutable training data snapshot"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{version_tag}_{timestamp}"
        snapshot_path = f"{self.snapshots_path}/{snapshot_name}.parquet"
        
        # Calculate data hash for verification
        data_hash = self._calculate_hash(df)
        
        # Save as parquet
        df.to_parquet(snapshot_path)
        
        metadata = {
            "version_tag": version_tag,
            "timestamp": timestamp,
            "snapshot_path": snapshot_path,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_hash": data_hash
        }
        
        # Save metadata
        metadata_path = f"{self.snapshots_path}/{snapshot_name}_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Created data snapshot: {snapshot_path}")
        return metadata
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataframe for verification"""
        return hashlib.sha256(
            df.to_json().encode()
        ).hexdigest()
    
    def load_snapshot(self, snapshot_path: str) -> pd.DataFrame:
        """Load a specific data snapshot"""
        return pd.read_parquet(snapshot_path)
    
    def verify_snapshot(
        self, 
        snapshot_path: str, 
        expected_hash: str
    ) -> bool:
        """Verify snapshot integrity"""
        df = self.load_snapshot(snapshot_path)
        actual_hash = self._calculate_hash(df)
        return actual_hash == expected_hash


# ============== Model Training ==============

class ModelTrainer:
    """Train lead scoring models"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config.config
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_column: str = "converted"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["training"]["test_size"],
            random_state=self.config["training"]["random_seed"],
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ):
        """Train the model"""
        try:
            from xgboost import XGBClassifier
            
            model = XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config["training"]["random_seed"],
                use_label_encoder=False,
                eval_metric='auc'
            )
        except ImportError:
            # Fallback to sklearn if xgboost not available
            from sklearn.ensemble import GradientBoostingClassifier
            
            model = GradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                random_state=self.config["training"]["random_seed"]
            )
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Precision at top 10%
        threshold_10 = np.percentile(y_pred_proba, 90)
        top_10_mask = y_pred_proba >= threshold_10
        precision_at_10 = y_test[top_10_mask].mean() if top_10_mask.sum() > 0 else 0
        
        metrics = {
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "precision_at_10pct": round(precision_at_10, 4),
            "test_size": len(y_test)
        }
        
        return metrics
    
    def cross_validate(
        self, 
        model, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict:
        """Perform cross-validation"""
        cv_scores = cross_val_score(
            model, X, y, 
            cv=self.config["training"]["cv_folds"],
            scoring='roc_auc'
        )
        
        return {
            "cv_mean_auc": round(float(np.mean(cv_scores)), 4),
            "cv_std_auc": round(float(np.std(cv_scores)), 4),
            "cv_scores": [round(s, 4) for s in cv_scores]
        }


# ============== Model Validation ==============

class ModelValidator:
    """Validate new models before deployment"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config.config["validation"]
    
    def validate(
        self, 
        metrics: Dict, 
        current_production_metrics: Optional[Dict] = None
    ) -> Dict:
        """Validate model meets minimum requirements"""
        validations = []
        passed = True
        
        # Check minimum AUC
        auc_check = metrics["auc"] >= self.config["min_auc"]
        validations.append({
            "check": "minimum_auc",
            "passed": auc_check,
            "value": metrics["auc"],
            "threshold": self.config["min_auc"]
        })
        passed = passed and auc_check
        
        # Check precision at 10%
        precision_check = metrics["precision_at_10pct"] >= self.config["min_precision_at_10"]
        validations.append({
            "check": "precision_at_10",
            "passed": precision_check,
            "value": metrics["precision_at_10pct"],
            "threshold": self.config["min_precision_at_10"]
        })
        passed = passed and precision_check
        
        # Compare with production if available
        if current_production_metrics:
            improvement = metrics["auc"] - current_production_metrics["auc"]
            improvement_check = improvement >= self.config["min_improvement"]
            validations.append({
                "check": "improvement_vs_production",
                "passed": improvement_check,
                "value": round(improvement, 4),
                "threshold": self.config["min_improvement"]
            })
            passed = passed and improvement_check
        
        return {
            "passed": passed,
            "validations": validations,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============== Retraining Orchestrator ==============

class RetrainingOrchestrator:
    """Orchestrate the full retraining pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = RetrainingConfig(config_path)
        self.data_manager = DataVersionManager()
        self.trainer = ModelTrainer(self.config)
        self.validator = ModelValidator(self.config)
        self.client = MlflowClient() if MLFLOW_AVAILABLE and self._mlflow_available() else None
    
    def _mlflow_available(self) -> bool:
        """Check if MLflow is configured"""
        if not MLFLOW_AVAILABLE:
            return False
        try:
            mlflow.get_tracking_uri()
            return True
        except:
            return False
    
    def check_triggers(self) -> Dict:
        """Check all retraining triggers"""
        triggers = {
            "scheduled": self._check_scheduled_trigger(),
            "drift_detected": self._check_drift_trigger(),
            "performance_degraded": self._check_performance_trigger(),
            "new_data_available": self._check_new_data_trigger()
        }
        
        return {
            "should_retrain": any(triggers.values()),
            "triggers": triggers,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_scheduled_trigger(self) -> bool:
        """Check if scheduled retraining is due"""
        # In production, would check last retrain date from database
        # For demo, return False
        return False
    
    def _check_drift_trigger(self) -> bool:
        """Check if data drift exceeds threshold"""
        # In production, would check latest drift metrics
        return False
    
    def _check_performance_trigger(self) -> bool:
        """Check if model performance has degraded"""
        # In production, would check current model AUC
        return False
    
    def _check_new_data_trigger(self) -> bool:
        """Check if enough new labeled data is available"""
        # In production, would count new samples
        return False
    
    def execute_retraining(
        self, 
        training_data: pd.DataFrame,
        force: bool = False
    ) -> Dict:
        """Execute the full retraining pipeline"""
        logger.info("Starting retraining pipeline...")
        
        result = {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
            "steps": []
        }
        
        try:
            # Step 1: Check triggers
            trigger_check = self.check_triggers()
            result["steps"].append({
                "step": "check_triggers",
                "result": trigger_check
            })
            
            if not trigger_check["should_retrain"] and not force:
                result["status"] = "skipped"
                result["reason"] = "No retrain triggers active"
                return result
            
            # Step 2: Create data snapshot
            logger.info("Step 2: Creating data snapshot...")
            snapshot = self.data_manager.create_snapshot(
                training_data,
                version_tag=f"retrain_{datetime.utcnow().strftime('%Y%m%d')}"
            )
            result["steps"].append({
                "step": "create_snapshot",
                "result": snapshot
            })
            
            # Step 3: Prepare data
            logger.info("Step 3: Preparing data...")
            X_train, X_test, y_train, y_test = self.trainer.prepare_data(
                training_data
            )
            result["steps"].append({
                "step": "prepare_data",
                "result": {
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                }
            })
            
            # Step 4: Train model
            logger.info("Step 4: Training model...")
            model = self.trainer.train_model(X_train, y_train)
            result["steps"].append({
                "step": "train_model",
                "result": {"status": "completed"}
            })
            
            # Step 5: Evaluate model
            logger.info("Step 5: Evaluating model...")
            metrics = self.trainer.evaluate_model(model, X_test, y_test)
            result["steps"].append({
                "step": "evaluate_model",
                "result": metrics
            })
            result["metrics"] = metrics
            
            # Step 6: Cross-validation
            logger.info("Step 6: Cross-validation...")
            cv_results = self.trainer.cross_validate(
                model, 
                pd.concat([X_train, X_test]), 
                pd.concat([y_train, y_test])
            )
            result["steps"].append({
                "step": "cross_validation",
                "result": cv_results
            })
            result["cv_results"] = cv_results
            
            # Step 7: Validate model
            logger.info("Step 7: Validating model...")
            validation = self.validator.validate(metrics)
            result["steps"].append({
                "step": "validate_model",
                "result": validation
            })
            
            if not validation["passed"]:
                result["status"] = "validation_failed"
                result["validation"] = validation
                logger.warning("Model validation failed")
                return result
            
            # Step 8: Log to MLflow (if available)
            if MLFLOW_AVAILABLE and self.client:
                logger.info("Step 8: Logging to MLflow...")
                with mlflow.start_run(run_name=f"retrain_{datetime.utcnow().isoformat()}"):
                    mlflow.log_params({
                        "trigger": str(trigger_check["triggers"]),
                        "snapshot_path": snapshot["snapshot_path"]
                    })
                    mlflow.log_metrics(metrics)
                    mlflow.log_metrics({
                        f"cv_{k}": v for k, v in cv_results.items() 
                        if isinstance(v, (int, float))
                    })
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        registered_model_name="lead-scoring-production"
                    )
                
                result["steps"].append({
                    "step": "log_to_mlflow",
                    "result": {"status": "completed"}
                })
            else:
                logger.info("Step 8: Skipping MLflow (not available)")
                result["steps"].append({
                    "step": "log_to_mlflow",
                    "result": {"status": "skipped", "reason": "MLflow not available"}
                })
            
            result["status"] = "success"
            result["model"] = model
            logger.info("Retraining pipeline completed successfully!")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Retraining failed: {e}")
        
        return result


# ============== Reproducibility Manager ==============

class ReproducibilityManager:
    """Ensure experiments are reproducible"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def setup_experiment(self, experiment_name: str) -> Dict:
        """Setup MLflow experiment with full tracking"""
        setup_info = {
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": sys.version,
            "config": self.config
        }
        
        try:
            import subprocess
            setup_info["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            setup_info["git_branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode().strip()
        except:
            setup_info["git_commit"] = "unavailable"
            setup_info["git_branch"] = "unavailable"
        
        return setup_info
    
    def log_to_mlflow(self, setup_info: Dict):
        """Log setup info to MLflow"""
        for key, value in setup_info.items():
            if isinstance(value, (str, int, float)):
                mlflow.log_param(key, value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (str, int, float)):
                        mlflow.log_param(f"{key}_{k}", v)


# ============== Example Usage ==============

if __name__ == "__main__":
    # Generate sample training data
    np.random.seed(42)
    n_samples = 1000
    
    training_data = pd.DataFrame({
        "engagement_score": np.random.uniform(0, 100, n_samples),
        "website_visits": np.random.poisson(5, n_samples),
        "email_opens": np.random.poisson(3, n_samples),
        "company_size": np.random.lognormal(4, 1, n_samples).astype(int),
        "days_since_contact": np.random.exponential(30, n_samples).astype(int),
        "converted": np.random.binomial(1, 0.15, n_samples)
    })
    
    # Initialize orchestrator
    orchestrator = RetrainingOrchestrator()
    
    # Check triggers
    print("\n=== Checking Retrain Triggers ===")
    triggers = orchestrator.check_triggers()
    print(f"Should retrain: {triggers['should_retrain']}")
    print(f"Triggers: {triggers['triggers']}")
    
    # Execute retraining (forced for demo)
    print("\n=== Executing Retraining Pipeline ===")
    result = orchestrator.execute_retraining(training_data, force=True)
    
    print(f"\nStatus: {result['status']}")
    if "metrics" in result:
        print(f"Metrics: {result['metrics']}")
    if "cv_results" in result:
        print(f"CV Results: {result['cv_results']}")
    
    print("\n=== Steps Completed ===")
    for step in result["steps"]:
        print(f"- {step['step']}: {step['result'].get('status', 'OK') if isinstance(step['result'], dict) else 'OK'}")

