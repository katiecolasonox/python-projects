import streamlit as st
import pandas as pd
import plotly.express as px
from engine import get_data, build_grade_model, run_silverton_prediction

# 1. PAGE CONFIG & STYLING
st.set_page_config(page_title="Silverton 50K Tracker", layout="wide", page_icon="🏔️")

# Custom CSS for the "Glassmorphism" look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                    url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=2070');
        background-size: cover;
    }
    div[data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; }
    .stTable { background: rgba(255, 255, 255, 0.05); border-radius: 10px; }
    </style>
    """, unsafe_content_code=True)

# 2. DATA PIPELINE (Using st.cache to avoid hitting Strava too often)
@st.cache_data(ttl=14400) # Refreshes every 4 hours
def get_dashboard_data():
    # Pull from Streamlit Secrets (Advanced Settings)
    cid = st.secrets["CLIENT_ID"]
    cs = st.secrets["CLIENT_SECRET"]
    rt = st.secrets["REFRESH_TOKEN"]
    
    # Run the Engine
    raw_streams, sync_time = get_data(cid, cs, rt)
    my_curve = build_grade_model(raw_streams)
    
    # Load Course Profile
    course_df = pd.read_csv("silverton_50K_profile.csv")
    results, predicted_time = run_silverton_prediction(course_df, my_curve)
    
    return results, predicted_time, sync_time

# Execute the pipeline
try:
    results_df, total_minutes, last_sync = get_dashboard_data()
except Exception as e:
    st.error(f"Waiting for Strava connection... (Error: {e})")
    st.stop()

# 3. DASHBOARD LAYOUT
st.title("Silverton 50K Finish Time Predicition")
st.caption(f"● Live Feed — Last synced via Strava: **{last_sync}**")

# Hero Metrics
hours = int(total_minutes // 60)
minutes = int(total_minutes % 60)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Finish", f"{hours}h {minutes}m")
with col2:
    st.metric("Elevation Gain", "12,000 ft")
with col3:
    st.metric("Course Distance", "31.0 mi")

st.divider()

# 4. THE VISUALIZATION: Elevation Profile
st.subheader("Course Strategy: Where to Run vs. Power Hike")

# Color mapping to match your "Runnable" vs "Power Hike" logic
fig = px.area(
    results_df, 
    x="Distance (miles)", 
    y="Elevation (feet)",
    color="effort_zone",
    color_discrete_map={'Runnable': '#00ffa2', 'Power Hike': '#ffcc00'},
    hover_data={'fatigued_pace': ':.2f'}
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color="white",
    yaxis_range=[9000, 13500], # Silverton is high altitude!
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 5. DATA BREAKDOWN
with st.expander("View Effort Breakdown"):
    summary = results_df.groupby('effort_zone').agg({
        'delta_dist_mi': 'sum',
        'segment_time_mins': 'sum'
    }).reset_index()
    
    summary.columns = ['Zone', 'Distance (mi)', 'Total Time (min)']
    st.table(summary)

st.markdown("---")
st.write("Current Status: **Optimizing for Altitude in Boulder.**")