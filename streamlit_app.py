import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="ClimateRisk360",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Create sample data if it doesn't exist
def create_sample_data():
    """Generate sample data for demonstration."""
    regions = [f"Region_{i}" for i in range(1, 6)]
    years = [2020, 2021, 2022, 2023]
    
    data = []
    for year in years:
        for region in regions:
            data.append({
                'region': region,
                'year': year,
                'avg_temperature': 20 + (hash(region) % 10) + (year - 2020) * 0.5,
                'total_rainfall': 1000 + (hash(region) % 500) - (year - 2020) * 50,
                'flood_events': (hash(region) % 5) + (year - 2020) * 0.5,
                'claims_count': (hash(region) % 20) + (year - 2020) * 2,
                'avg_claim_amount': 5000 + (hash(region) % 10000) + (year - 2020) * 500,
            })
    
    df = pd.DataFrame(data)
    df['climate_risk_score'] = (df['avg_temperature'] * 0.3 + 
                               df['total_rainfall'] * 0.2 + 
                               df['flood_events'] * 0.5).round(2)
    
    return df

# Load or create data
@st.cache_data
def load_data():
    data_file = DATA_DIR / "processed_data.parquet"
    if data_file.exists():
        return pd.read_parquet(data_file)
    else:
        # In a real app, this would load from your processed data
        return create_sample_data()

def main():
    st.title("🌍 ClimateRisk360 - ESG & Insurance Risk Dashboard")
    st.write("Analyzing climate risks and insurance claims correlation")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_year = st.sidebar.slider(
        "Select Year", 
        min_value=int(df['year'].min()), 
        max_value=int(df['year'].max()),
        value=int(df['year'].max())
    )
    
    # Filter data
    filtered_df = df[df['year'] == selected_year]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Temperature", f"{filtered_df['avg_temperature'].mean():.1f}°C")
    with col2:
        st.metric("Total Rainfall", f"{filtered_df['total_rainfall'].sum():,.0f} mm")
    with col3:
        st.metric("Total Claims", f"{filtered_df['claims_count'].sum():,}")
    with col4:
        st.metric("Avg Claim Amount", f"${filtered_df['avg_claim_amount'].mean():,.2f}")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Heatmap
        st.subheader("Climate Risk Heatmap")
        fig = px.choropleth(
            filtered_df,
            locations="region",
            color="climate_risk_score",
            hover_name="region",
            color_continuous_scale="RdYlGn_r",
            title="Climate Risk by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Claims vs Climate Factors
        st.subheader("Claims vs Climate Factors")
        x_axis = st.selectbox(
            "X-Axis",
            ["avg_temperature", "total_rainfall", "flood_events"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y="claims_count",
            size="avg_claim_amount",
            color="region",
            hover_name="region",
            size_max=30,
            title=f"Claims Count vs {x_axis.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Analysis
    st.subheader("Trend Analysis")
    metric = st.selectbox(
        "Select Metric",
        ["avg_temperature", "total_rainfall", "flood_events", "claims_count", "avg_claim_amount"]
    )
    
    fig = px.line(
        df,
        x="year",
        y=metric,
        color="region",
        markers=True,
        title=f"{metric.replace('_', ' ').title()} Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.subheader("Detailed Data")
    st.dataframe(
        df.sort_values(["year", "region"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
