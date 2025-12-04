"""
Lead Scoring Model Monitoring Dashboard
Real-time monitoring for the lead scoring ML model
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============== Page Configuration ==============

st.set_page_config(
    page_title="Lead Scoring Model Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Custom CSS ==============

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #888;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFF;
        font-size: 2rem;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #00FF00;
    }
    .status-healthy { color: #00FF00; font-weight: bold; }
    .status-warning { color: #FFCC00; font-weight: bold; }
    .status-critical { color: #FF0000; font-weight: bold; }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-info { background-color: #1E3A5F; border-left: 4px solid #4A9EFF; }
    .alert-warning { background-color: #3D3A1F; border-left: 4px solid #FFCC00; }
    .alert-critical { background-color: #3D1F1F; border-left: 4px solid #FF0000; }
    </style>
""", unsafe_allow_html=True)


# ============== Helper Functions ==============

@st.cache_data(ttl=60)
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    return {
        "drift": pd.DataFrame({
            'date': dates,
            'psi': np.random.uniform(0.05, 0.25, 30),
            'engagement_score_psi': np.random.uniform(0.02, 0.15, 30),
            'website_visits_psi': np.random.uniform(0.03, 0.18, 30),
            'company_size_psi': np.random.uniform(0.01, 0.12, 30)
        }),
        "predictions": pd.DataFrame({
            'date': dates,
            'count': np.random.poisson(500, 30) + 200,
            'mean_score': np.random.uniform(0.4, 0.5, 30),
            'hot_leads': np.random.poisson(50, 30)
        }),
        "performance": pd.DataFrame({
            'date': dates,
            'auc': np.random.uniform(0.82, 0.88, 30),
            'precision': np.random.uniform(0.75, 0.82, 30),
            'latency_p95': np.random.uniform(40, 80, 30)
        }),
        "conversions": pd.DataFrame({
            'bucket': ['Hot', 'Warm', 'Cool', 'Cold'],
            'leads': [1500, 2500, 3500, 5000],
            'conversions': [600, 500, 280, 150],
            'rate': [40.0, 20.0, 8.0, 3.0]
        })
    }


def get_status_class(value, warning_threshold, critical_threshold, higher_is_bad=True):
    """Get CSS class for status indicator"""
    if higher_is_bad:
        if value >= critical_threshold:
            return "status-critical"
        elif value >= warning_threshold:
            return "status-warning"
        return "status-healthy"
    else:
        if value <= critical_threshold:
            return "status-critical"
        elif value <= warning_threshold:
            return "status-warning"
        return "status-healthy"


# ============== Sidebar ==============

st.sidebar.title("🎯 Lead Scoring Monitor")
st.sidebar.markdown("---")

# Time range selector
time_range = st.sidebar.selectbox(
    "📅 Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=True)

st.sidebar.markdown("---")

# Model info
st.sidebar.subheader("📊 Model Info")
st.sidebar.markdown("""
- **Version:** v2.3.1
- **Last Retrained:** 2024-12-01
- **Endpoint:** production
- **Status:** 🟢 Healthy
""")

st.sidebar.markdown("---")

# Quick actions
st.sidebar.subheader("⚡ Quick Actions")
if st.sidebar.button("🔄 Trigger Retraining"):
    st.sidebar.success("Retraining job queued!")
if st.sidebar.button("📊 Generate Report"):
    st.sidebar.info("Report generation started...")
if st.sidebar.button("🔙 Rollback Model"):
    st.sidebar.warning("Are you sure? This will rollback to v2.3.0")


# ============== Main Content ==============

st.title("🎯 Lead Scoring Model Dashboard")
st.markdown("Real-time monitoring for the production lead scoring model")

# Load data
data = generate_sample_data()

# ============== Key Metrics Row ==============

st.markdown("### 📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_auc = data["performance"]["auc"].iloc[-1]
    prev_auc = data["performance"]["auc"].iloc[-2]
    st.metric(
        label="Model AUC",
        value=f"{current_auc:.3f}",
        delta=f"{(current_auc - prev_auc):.3f}"
    )

with col2:
    current_rate = data["conversions"]["rate"].mean()
    st.metric(
        label="Avg Conversion Rate",
        value=f"{current_rate:.1f}%",
        delta="+1.2%"
    )

with col3:
    current_latency = data["performance"]["latency_p95"].iloc[-1]
    prev_latency = data["performance"]["latency_p95"].iloc[-2]
    st.metric(
        label="Latency (P95)",
        value=f"{current_latency:.0f}ms",
        delta=f"{(current_latency - prev_latency):.0f}ms",
        delta_color="inverse"
    )

with col4:
    daily_predictions = data["predictions"]["count"].iloc[-1]
    prev_predictions = data["predictions"]["count"].iloc[-2]
    st.metric(
        label="Daily Predictions",
        value=f"{daily_predictions:,}",
        delta=f"{((daily_predictions - prev_predictions) / prev_predictions * 100):.1f}%"
    )

with col5:
    hot_leads = data["predictions"]["hot_leads"].iloc[-1]
    st.metric(
        label="Hot Leads Today",
        value=f"{hot_leads}",
        delta="+8"
    )

st.markdown("---")

# ============== Charts Row 1 ==============

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Data Drift Monitoring (PSI)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["drift"]["date"],
        y=data["drift"]["psi"],
        name="Overall PSI",
        line=dict(color="#4A9EFF", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data["drift"]["date"],
        y=data["drift"]["engagement_score_psi"],
        name="Engagement Score",
        line=dict(color="#00FF88", width=1, dash="dot")
    ))
    
    fig.add_hline(
        y=0.2, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Alert Threshold (0.2)",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=0.1, 
        line_dash="dash", 
        line_color="yellow",
        annotation_text="Warning (0.1)",
        annotation_position="right"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📈 Prediction Distribution")
    
    # Generate score distributions
    reference_scores = np.random.beta(2, 5, 1000)
    current_scores = np.random.beta(2.2, 4.8, 1000)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=reference_scores,
        name='Training Distribution',
        opacity=0.6,
        marker_color='#4A9EFF'
    ))
    
    fig.add_trace(go.Histogram(
        x=current_scores,
        name='Production Distribution',
        opacity=0.6,
        marker_color='#00FF88'
    ))
    
    fig.update_layout(
        barmode='overlay',
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Lead Score",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============== Charts Row 2 ==============

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("💰 Conversion Rate by Score Bucket")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=data["conversions"]["bucket"],
            y=data["conversions"]["leads"],
            name="Total Leads",
            marker_color="#4A9EFF",
            opacity=0.7
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=data["conversions"]["bucket"],
            y=data["conversions"]["rate"],
            name="Conversion Rate %",
            mode='lines+markers',
            line=dict(color='#00FF88', width=3),
            marker=dict(size=10)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_yaxes(title_text="Number of Leads", secondary_y=False)
    fig.update_yaxes(title_text="Conversion Rate %", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with col_right2:
    st.subheader("⚡ Model Performance Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["performance"]["date"],
        y=data["performance"]["auc"],
        name="AUC",
        line=dict(color="#4A9EFF", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data["performance"]["date"],
        y=data["performance"]["precision"],
        name="Precision",
        line=dict(color="#FF6B6B", width=2)
    ))
    
    fig.add_hline(
        y=0.75, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Min AUC (0.75)"
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_range=[0.6, 1.0],
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============== Alerts Section ==============

st.subheader(" Recent Alerts")

col_alerts, col_status = st.columns([2, 1])

with col_alerts:
    alerts = [
        {"time": "2024-12-04 10:30", "severity": "warning", "message": "PSI for 'lead_source' reached 0.18 - approaching threshold"},
        {"time": "2024-12-04 09:15", "severity": "info", "message": "Daily model performance report generated successfully"},
        {"time": "2024-12-04 08:00", "severity": "info", "message": "Scheduled data quality check completed - all checks passed"},
        {"time": "2024-12-03 22:45", "severity": "info", "message": "Model retraining completed - v2.3.1 deployed to staging"},
        {"time": "2024-12-03 18:30", "severity": "warning", "message": "Latency spike detected - P95 reached 180ms temporarily"},
    ]
    
    for alert in alerts:
        if alert["severity"] == "critical":
            icon = "🔴"
            css_class = "alert-critical"
        elif alert["severity"] == "warning":
            icon = "🟡"
            css_class = "alert-warning"
        else:
            icon = "🔵"
            css_class = "alert-info"
        
        st.markdown(
            f"""<div class="alert-box {css_class}">
                {icon} <strong>{alert['time']}</strong> - {alert['message']}
            </div>""",
            unsafe_allow_html=True
        )

with col_status:
    st.markdown("### System Status")
    
    status_items = [
        ("API Endpoint", "🟢 Healthy", "100%"),
        ("Model Serving", "🟢 Running", "v2.3.1"),
        ("Data Pipeline", "🟢 Active", "< 1min lag"),
        ("MLflow Registry", "🟢 Connected", ""),
        ("CRM Sync", "🟢 Active", "Last: 2min ago"),
    ]
    
    for name, status, detail in status_items:
        st.markdown(f"**{name}:** {status}")
        if detail:
            st.caption(detail)

st.markdown("---")

# ============== Feature Importance ==============

st.subheader(" Feature Importance (Current Model)")

importance_data = pd.DataFrame({
    'feature': ['engagement_score', 'website_visits', 'email_opens', 'company_size', 
                'days_since_contact', 'lead_source', 'industry', 'region'],
    'importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
})

fig = px.bar(
    importance_data,
    x='importance',
    y='feature',
    orientation='h',
    color='importance',
    color_continuous_scale='Blues'
)

fig.update_layout(
    template="plotly_dark",
    height=300,
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(fig, use_container_width=True)

# ============== Footer ==============

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Refresh interval: 60s | 
        <a href="https://mlflow.rakez.com" target="_blank">MLflow</a> | 
        <a href="https://github.com/rakez/lead-scoring" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

