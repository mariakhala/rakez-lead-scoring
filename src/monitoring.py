"""
Lead Scoring Model Monitoring
Comprehensive monitoring for data drift, prediction drift, and business metrics
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Alert Management ==============

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertThresholds:
    """Configurable alert thresholds"""
    psi_warning: float = 0.2
    psi_critical: float = 0.4
    ks_warning: float = 0.1
    ks_critical: float = 0.2
    latency_p95_warning_ms: float = 200
    latency_p95_critical_ms: float = 500
    error_rate_warning_pct: float = 1.0
    error_rate_critical_pct: float = 5.0
    conversion_drop_warning_pct: float = 10
    conversion_drop_critical_pct: float = 20


class AlertManager:
    """Centralized alerting system"""
    
    def __init__(self, thresholds: AlertThresholds = None):
        self.thresholds = thresholds or AlertThresholds()
        self.alert_history = []
    
    def send_alert(
        self, 
        severity: AlertSeverity, 
        title: str, 
        message: str,
        metrics: Optional[Dict] = None
    ):
        """Send alert to appropriate channels"""
        alert = {
            "severity": severity.value,
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics or {}
        }
        
        self.alert_history.append(alert)
        
        # Log the alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }
        
        logger.log(
            log_level[severity],
            f"[ALERT] {title}: {message}"
        )
        
        # In production, would send to Slack/PagerDuty/Email
        if severity == AlertSeverity.CRITICAL:
            self._send_to_pagerduty(alert)
        
        self._send_to_slack(alert)
        
        return alert
    
    def _send_to_slack(self, alert: Dict):
        """Send to Slack (placeholder)"""
        logger.info(f"Slack notification: {alert['title']}")
    
    def _send_to_pagerduty(self, alert: Dict):
        """Send to PagerDuty (placeholder)"""
        logger.info(f"PagerDuty alert: {alert['title']}")


# ============== Data Drift Monitoring ==============

class DataDriftMonitor:
    """Monitor for data distribution changes"""
    
    def __init__(
        self, 
        reference_data: pd.DataFrame,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1
    ):
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        
        self.numerical_columns = reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        self.categorical_columns = reference_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.alert_manager = AlertManager()
    
    def calculate_psi(
        self, 
        reference: np.ndarray, 
        current: np.ndarray, 
        bins: int = 10
    ) -> float:
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
        return float(psi)
    
    def check_categorical_drift(
        self, 
        reference: pd.Series, 
        current: pd.Series
    ) -> Tuple[bool, float]:
        """Chi-square test for categorical drift"""
        # Get value counts
        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        if len(all_categories) < 2:
            return False, 0.0
        
        contingency = np.array([ref_aligned, cur_aligned])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        return p_value < 0.05, float(chi2)
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift across all features"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "features": {},
            "overall_drift_detected": False,
            "drifted_features": [],
            "severity": "LOW"
        }
        
        # Check numerical features
        for col in self.numerical_columns:
            if col not in current_data.columns:
                continue
                
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            psi = self.calculate_psi(ref_values, cur_values)
            ks_stat, ks_pvalue = ks_2samp(ref_values, cur_values)
            
            drift_detected = psi > self.psi_threshold or ks_stat > self.ks_threshold
            
            results["features"][col] = {
                "type": "numerical",
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 4),
                "drift_detected": drift_detected,
                "reference_mean": round(float(np.mean(ref_values)), 4),
                "current_mean": round(float(np.mean(cur_values)), 4)
            }
            
            if drift_detected:
                results["drifted_features"].append(col)
        
        # Check categorical features
        for col in self.categorical_columns:
            if col not in current_data.columns:
                continue
            
            drift_detected, chi2_stat = self.check_categorical_drift(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            results["features"][col] = {
                "type": "categorical",
                "chi2_statistic": round(chi2_stat, 4),
                "drift_detected": drift_detected
            }
            
            if drift_detected:
                results["drifted_features"].append(col)
        
        # Calculate overall drift
        results["overall_drift_detected"] = len(results["drifted_features"]) > 0
        results["severity"] = self._calculate_severity(results)
        
        # Send alerts if needed
        if results["overall_drift_detected"]:
            self._send_drift_alert(results)
        
        return results
    
    def _calculate_severity(self, results: Dict) -> str:
        """Calculate overall drift severity"""
        if not results["features"]:
            return "LOW"
            
        drifted_count = len(results["drifted_features"])
        total_features = len(results["features"])
        drift_ratio = drifted_count / total_features
        
        if drift_ratio > 0.5:
            return "CRITICAL"
        elif drift_ratio > 0.2:
            return "HIGH"
        elif drift_ratio > 0:
            return "MEDIUM"
        return "LOW"
    
    def _send_drift_alert(self, results: Dict):
        """Send alert for detected drift"""
        severity = AlertSeverity.WARNING
        if results["severity"] == "CRITICAL":
            severity = AlertSeverity.CRITICAL
        
        self.alert_manager.send_alert(
            severity=severity,
            title="Data Drift Detected",
            message=f"Drift detected in {len(results['drifted_features'])} features: {results['drifted_features']}",
            metrics={"severity": results["severity"], "features": results["drifted_features"]}
        )


# ============== Prediction Drift Monitoring ==============

class PredictionDriftMonitor:
    """Monitor for changes in prediction distribution"""
    
    def __init__(self, reference_predictions: np.ndarray):
        self.reference_predictions = reference_predictions
        self.reference_mean = float(np.mean(reference_predictions))
        self.reference_std = float(np.std(reference_predictions))
        self.alert_manager = AlertManager()
    
    def detect_drift(self, current_predictions: np.ndarray) -> Dict:
        """Detect drift in predictions"""
        current_mean = float(np.mean(current_predictions))
        current_std = float(np.std(current_predictions))
        
        # KS test
        ks_stat, ks_pvalue = ks_2samp(
            self.reference_predictions, 
            current_predictions
        )
        
        # Mean shift detection
        mean_shift = abs(current_mean - self.reference_mean)
        mean_shift_pct = (mean_shift / self.reference_mean * 100 
                         if self.reference_mean != 0 else 0)
        
        # Distribution correlation
        ref_hist, _ = np.histogram(self.reference_predictions, bins=20)
        cur_hist, _ = np.histogram(current_predictions, bins=20)
        
        if np.std(ref_hist) > 0 and np.std(cur_hist) > 0:
            distribution_correlation = float(np.corrcoef(ref_hist, cur_hist)[0, 1])
        else:
            distribution_correlation = 1.0
        
        drift_detected = (
            ks_stat > 0.1 or 
            mean_shift_pct > 10 or 
            distribution_correlation < 0.9
        )
        
        severity = "LOW"
        if ks_stat > 0.2:
            severity = "CRITICAL"
        elif drift_detected:
            severity = "MEDIUM"
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(current_mean, 4),
            "mean_shift_pct": round(mean_shift_pct, 2),
            "reference_std": round(self.reference_std, 4),
            "current_std": round(current_std, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "distribution_correlation": round(distribution_correlation, 4),
            "drift_detected": drift_detected,
            "severity": severity
        }
        
        if drift_detected:
            self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING if severity != "CRITICAL" else AlertSeverity.CRITICAL,
                title="Prediction Drift Detected",
                message=f"Score distribution shift detected. KS={ks_stat:.3f}, Mean shift={mean_shift_pct:.1f}%",
                metrics=results
            )
        
        return results


# ============== Performance Monitoring ==============

class PerformanceMonitor:
    """Monitor API performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.latencies: List[float] = []
        self.window_size = window_size
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        self.alert_manager = AlertManager()
    
    def record_request(
        self, 
        latency_ms: float, 
        is_error: bool = False
    ):
        """Record a request's metrics"""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
        
        self.request_count += 1
        if is_error:
            self.error_count += 1
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.latencies:
            return {"status": "no_data"}
        
        latencies_sorted = sorted(self.latencies)
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        error_rate = (self.error_count / self.request_count * 100 
                     if self.request_count > 0 else 0)
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "latency": {
                "p50_ms": round(float(np.percentile(latencies_sorted, 50)), 2),
                "p95_ms": round(float(np.percentile(latencies_sorted, 95)), 2),
                "p99_ms": round(float(np.percentile(latencies_sorted, 99)), 2),
                "mean_ms": round(float(np.mean(latencies_sorted)), 2),
                "max_ms": round(float(max(latencies_sorted)), 2)
            },
            "throughput": {
                "total_requests": self.request_count,
                "requests_per_second": round(self.request_count / elapsed, 2) if elapsed > 0 else 0,
                "requests_per_minute": round(self.request_count / elapsed * 60, 2) if elapsed > 0 else 0
            },
            "errors": {
                "total_errors": self.error_count,
                "error_rate_pct": round(error_rate, 4)
            },
            "sla_compliance": {
                "p95_under_200ms": float(np.percentile(latencies_sorted, 95)) < 200,
                "error_rate_under_1pct": error_rate < 1
            }
        }
        
        # Check for SLA violations
        self._check_sla_violations(metrics)
        
        return metrics
    
    def _check_sla_violations(self, metrics: Dict):
        """Check and alert on SLA violations"""
        if not metrics["sla_compliance"]["p95_under_200ms"]:
            self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="Latency SLA Violation",
                message=f"P95 latency ({metrics['latency']['p95_ms']}ms) exceeds 200ms threshold",
                metrics=metrics["latency"]
            )
        
        if not metrics["sla_compliance"]["error_rate_under_1pct"]:
            self.alert_manager.send_alert(
                severity=AlertSeverity.CRITICAL,
                title="Error Rate SLA Violation",
                message=f"Error rate ({metrics['errors']['error_rate_pct']}%) exceeds 1% threshold",
                metrics=metrics["errors"]
            )


# ============== Business Metrics Monitoring ==============

class BusinessMetricsMonitor:
    """Monitor business KPIs related to model performance"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
    
    def calculate_conversion_metrics(
        self, 
        predictions_df: pd.DataFrame,
        conversions_df: pd.DataFrame
    ) -> Dict:
        """Calculate conversion-related business metrics"""
        # Merge predictions with outcomes
        merged = predictions_df.merge(
            conversions_df, 
            on="lead_id", 
            how="left"
        )
        merged["converted"] = merged["converted"].fillna(0)
        
        # Create score buckets
        merged["score_bucket"] = pd.cut(
            merged["score"],
            bins=[0, 0.3, 0.5, 0.8, 1.0],
            labels=["Cold", "Cool", "Warm", "Hot"]
        )
        
        # Calculate metrics by bucket
        bucket_metrics = merged.groupby("score_bucket").agg({
            "lead_id": "count",
            "converted": ["sum", "mean"]
        }).round(4)
        
        bucket_metrics.columns = ["total_leads", "conversions", "conversion_rate"]
        
        # Overall metrics
        overall_conversion = merged["converted"].mean()
        hot_leads = merged[merged["score_bucket"] == "Hot"]
        hot_conversion = hot_leads["converted"].mean() if len(hot_leads) > 0 else 0
        
        # Model lift
        model_lift = ((hot_conversion / overall_conversion - 1) * 100 
                     if overall_conversion > 0 else 0)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "by_bucket": bucket_metrics.to_dict(orient="index"),
            "overall_metrics": {
                "total_leads_scored": len(merged),
                "total_conversions": int(merged["converted"].sum()),
                "overall_conversion_rate_pct": round(overall_conversion * 100, 2),
                "model_lift_pct": round(model_lift, 2)
            },
            "model_effectiveness": {
                "hot_lead_conversion_rate_pct": round(hot_conversion * 100, 2),
                "prioritization_working": hot_conversion > overall_conversion
            }
        }
    
    def check_score_calibration(
        self, 
        predictions_df: pd.DataFrame,
        conversions_df: pd.DataFrame
    ) -> Dict:
        """Check if scores are well-calibrated with actual conversions"""
        merged = predictions_df.merge(
            conversions_df, 
            on="lead_id", 
            how="inner"
        )
        
        # Create deciles
        merged["score_decile"] = pd.cut(
            merged["score"],
            bins=10,
            labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
        )
        
        calibration = merged.groupby("score_decile")["converted"].agg(["mean", "count"])
        calibration.columns = ["actual_rate", "sample_size"]
        
        # Calculate calibration error
        calibration["expected_rate"] = [0.05, 0.15, 0.25, 0.35, 0.45, 
                                        0.55, 0.65, 0.75, 0.85, 0.95][:len(calibration)]
        calibration["calibration_error"] = abs(
            calibration["actual_rate"] - calibration["expected_rate"]
        )
        
        mean_error = calibration["calibration_error"].mean()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "calibration_by_decile": calibration.to_dict(orient="index"),
            "mean_calibration_error": round(float(mean_error), 4),
            "is_well_calibrated": mean_error < 0.1
        }


# ============== Model Investigator ==============

class ModelInvestigator:
    """Systematic investigation of model performance issues"""
    
    def __init__(
        self,
        data_drift_monitor: DataDriftMonitor,
        prediction_drift_monitor: PredictionDriftMonitor,
        business_monitor: BusinessMetricsMonitor
    ):
        self.data_drift_monitor = data_drift_monitor
        self.prediction_drift_monitor = prediction_drift_monitor
        self.business_monitor = business_monitor
    
    def run_investigation(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray,
        predictions_df: pd.DataFrame,
        conversions_df: pd.DataFrame
    ) -> Dict:
        """Complete investigation when issues are reported"""
        investigation = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "investigating",
            "findings": [],
            "checks": {}
        }
        
        # Check 1: Data Drift
        logger.info("Checking data drift...")
        data_drift = self.data_drift_monitor.detect_drift(current_data)
        investigation["checks"]["data_drift"] = data_drift
        
        if data_drift["overall_drift_detected"]:
            investigation["findings"].append({
                "issue": "DATA_DRIFT",
                "severity": data_drift["severity"],
                "details": f"Drift in features: {data_drift['drifted_features']}"
            })
        
        # Check 2: Prediction Drift
        logger.info("Checking prediction drift...")
        pred_drift = self.prediction_drift_monitor.detect_drift(current_predictions)
        investigation["checks"]["prediction_drift"] = pred_drift
        
        if pred_drift["drift_detected"]:
            investigation["findings"].append({
                "issue": "PREDICTION_DRIFT",
                "severity": pred_drift["severity"],
                "details": f"Score distribution shifted. Mean shift: {pred_drift['mean_shift_pct']}%"
            })
        
        # Check 3: Model Calibration
        logger.info("Checking model calibration...")
        calibration = self.business_monitor.check_score_calibration(
            predictions_df, conversions_df
        )
        investigation["checks"]["calibration"] = calibration
        
        if not calibration["is_well_calibrated"]:
            investigation["findings"].append({
                "issue": "MISCALIBRATION",
                "severity": "HIGH",
                "details": f"Mean calibration error: {calibration['mean_calibration_error']}"
            })
        
        # Check 4: Business Metrics
        logger.info("Checking business metrics...")
        business = self.business_monitor.calculate_conversion_metrics(
            predictions_df, conversions_df
        )
        investigation["checks"]["business_metrics"] = business
        
        if not business["model_effectiveness"]["prioritization_working"]:
            investigation["findings"].append({
                "issue": "MODEL_NOT_PRIORITIZING",
                "severity": "CRITICAL",
                "details": "Hot leads not converting better than average"
            })
        
        # Determine root cause
        investigation["root_cause"] = self._determine_root_cause(investigation)
        investigation["recommendations"] = self._generate_recommendations(investigation)
        investigation["status"] = "complete"
        
        return investigation
    
    def _determine_root_cause(self, investigation: Dict) -> str:
        """Determine most likely root cause"""
        findings = investigation["findings"]
        
        if any(f["issue"] == "MODEL_NOT_PRIORITIZING" for f in findings):
            if any(f["issue"] == "DATA_DRIFT" for f in findings):
                return "DATA_DRIFT_CAUSING_MODEL_DEGRADATION"
            return "MODEL_DEGRADATION"
        
        if any(f["issue"] == "DATA_DRIFT" and f["severity"] == "CRITICAL" for f in findings):
            return "SEVERE_DATA_DRIFT"
        
        if any(f["issue"] == "MISCALIBRATION" for f in findings):
            return "MODEL_MISCALIBRATION"
        
        if any(f["issue"] == "PREDICTION_DRIFT" for f in findings):
            return "PREDICTION_DISTRIBUTION_SHIFT"
        
        return "NO_CLEAR_ROOT_CAUSE"
    
    def _generate_recommendations(self, investigation: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        root_cause = investigation["root_cause"]
        
        if "DATA_DRIFT" in root_cause:
            recommendations.append("IMMEDIATE: Investigate data pipeline for issues")
            recommendations.append("SHORT_TERM: Trigger model retraining with recent data")
            recommendations.append("LONG_TERM: Implement automated drift-based retraining")
        
        if "MODEL_DEGRADATION" in root_cause:
            recommendations.append("IMMEDIATE: Consider rollback to previous model version")
            recommendations.append("SHORT_TERM: Analyze feature importance changes")
            recommendations.append("MEDIUM_TERM: Collect more training data and retrain")
        
        if "MISCALIBRATION" in root_cause:
            recommendations.append("IMMEDIATE: Apply calibration adjustment (Platt scaling)")
            recommendations.append("SHORT_TERM: Retrain with proper calibration")
        
        if root_cause == "NO_CLEAR_ROOT_CAUSE":
            recommendations.append("Investigate external factors (market changes, seasonality)")
            recommendations.append("Check CRM integration and data freshness")
            recommendations.append("Consult with sales team for qualitative feedback")
        
        return recommendations


# ============== Example Usage ==============

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Reference data (training time)
    reference_data = pd.DataFrame({
        "engagement_score": np.random.uniform(0, 100, 1000),
        "website_visits": np.random.poisson(5, 1000),
        "email_opens": np.random.poisson(3, 1000),
        "company_size": np.random.lognormal(4, 1, 1000).astype(int),
        "lead_source": np.random.choice(["Web", "Email", "Referral"], 1000)
    })
    
    # Current data (production)
    current_data = pd.DataFrame({
        "engagement_score": np.random.uniform(10, 90, 500),  # Shifted!
        "website_visits": np.random.poisson(7, 500),  # Shifted!
        "email_opens": np.random.poisson(3, 500),
        "company_size": np.random.lognormal(4, 1, 500).astype(int),
        "lead_source": np.random.choice(["Web", "Email", "Referral", "Social"], 500)  # New category!
    })
    
    # Initialize monitors
    data_monitor = DataDriftMonitor(reference_data)
    
    # Run drift detection
    drift_results = data_monitor.detect_drift(current_data)
    
    print("\n=== Data Drift Results ===")
    print(f"Overall drift detected: {drift_results['overall_drift_detected']}")
    print(f"Severity: {drift_results['severity']}")
    print(f"Drifted features: {drift_results['drifted_features']}")
    
    # Performance monitoring
    perf_monitor = PerformanceMonitor()
    
    # Simulate requests
    for _ in range(100):
        latency = np.random.exponential(50)
        is_error = np.random.random() < 0.01
        perf_monitor.record_request(latency, is_error)
    
    perf_metrics = perf_monitor.get_metrics()
    
    print("\n=== Performance Metrics ===")
    print(f"P95 Latency: {perf_metrics['latency']['p95_ms']}ms")
    print(f"Error Rate: {perf_metrics['errors']['error_rate_pct']}%")
    print(f"SLA Compliant: {perf_metrics['sla_compliance']}")

