import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Welding Performance Dashboard",
    page_icon="🔧",
    layout="wide"
)

@st.cache_data
def load_default_data():
    """Load the default welding data CSV file"""
    try:
        data = pd.read_csv('Weld Sample Data.csv')
        return process_weld_data(data)
    except FileNotFoundError:
        return None

def process_weld_data(data):
    """Process welding data with date parsing and forecast calculations"""
    # Convert date columns
    data['Plan Available to Start'] = pd.to_datetime(data['Plan Available to Start'])
    data['Actual Available to Start'] = pd.to_datetime(data['Actual Available to Start'])
    
    # Forecast Data Calculation (from notebook)
    ForecastWait = 30
    ForecastSetup = 45
    ForecastDuration = 70
    ForecastCycleTime = ForecastWait + ForecastSetup + ForecastDuration
    PlanCycleTime = 120
    CurrentDelay = 13
    
    data['Forecast Start'] = data['Plan Available to Start'] + pd.to_timedelta(
        np.floor(CurrentDelay + (ForecastCycleTime / PlanCycleTime - 1) * (data['Weld Number'] - 50)), 
        unit='D'
    )
    
    
    MitigatedWait = 5
    MitigatedSetup = 35
    MitigatedDuration = 60
    MitigatedHours = 9
    BaseHours = 8
    MitigatedCycleTime = MitigatedWait + MitigatedSetup + MitigatedDuration
    
    data['Mitigated Forecast'] = data['Plan Available to Start'] + pd.to_timedelta(
        np.floor(CurrentDelay + (MitigatedCycleTime / PlanCycleTime - 1) * (BaseHours / MitigatedHours) * (data['Weld Number'] - 50)), 
        unit='D'
    )
    
    data['Mitigated Forecast'] = data[['Plan Available to Start','Mitigated Forecast']].groupby('Plan Available to Start').transform('last')
    
    # Set forecast values to NaN for weld numbers < 51
    data.loc[data['Weld Number'] < 51, ['Mitigated Forecast', 'Forecast Start']] = np.nan
    
    return data

def create_completion_chart(data):
    """Create the main completion chart showing actual vs planned vs forecast"""
    fig = go.Figure()
    
    # Welds Completed (Actual)
    actual_data = data.dropna(subset=['Actual Available to Start'])
    fig.add_trace(go.Scatter(
        x=actual_data['Actual Available to Start'],
        y=actual_data['Weld Number'],
        mode='lines+markers',
        name='Welds Completed',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Original Plan
    fig.add_trace(go.Scatter(
        x=data['Plan Available to Start'],
        y=data['Weld Number'],
        mode='lines',
        name='Original Plan',
        line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
        hoverinfo='skip'
    ))
    
    # Forecast
    forecast_data = data.dropna(subset=['Forecast Start'])
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['Forecast Start'],
            y=forecast_data['Weld Number'],
            mode='lines',
            name='Current Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Mitigated Forecast
    mitigated_data = data.dropna(subset=['Mitigated Forecast'])
    if not mitigated_data.empty:
        fig.add_trace(go.Scatter(
            x=mitigated_data['Mitigated Forecast'],
            y=mitigated_data['Weld Number'],
            mode='lines',
            name='Mitigated Forecast',
            line=dict(color='green', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title='Welding Progress: Actual vs Planned vs Forecast',
        xaxis_title='Date',
        yaxis_title='Weld Number',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def create_cycle_time_chart(data):
    """Create cycle time analysis chart"""
    actual_data = data.dropna(subset=['Actual Cycle Time (min)'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data['Weld Number'],
        y=actual_data['Actual Cycle Time (min)'],
        mode='markers',
        name='Actual Cycle Time',
        marker=dict(
            color=actual_data['Actual Cycle Time (min)'],
            colorscale='Viridis',
            size=8,
            colorbar=dict(title="Cycle Time (min)")
        )
    ))
    
    # Add planned cycle time line
    fig.add_hline(y=120, line_dash="dash", line_color="red", 
                  annotation_text="Planned Cycle Time (120 min)")
    
    fig.update_layout(
        title='Actual Cycle Time by Weld Number',
        xaxis_title='Weld Number',
        yaxis_title='Cycle Time (minutes)',
        height=400
    )
    
    return fig

def create_weld_rate_chart(data):
    """Create weld rate analysis chart"""
    actual_data = data.dropna(subset=['Weld Rate (kg / min)'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data['Weld Number'],
        y=actual_data['Weld Rate (kg / min)'],
        mode='markers',
        name='Weld Rate',
        marker=dict(
            color=actual_data['Weld Rate (kg / min)'],
            colorscale='Plasma',
            size=8,
            colorbar=dict(title="Weld Rate (kg/min)")
        )
    ))
    
    fig.update_layout(
        title='Weld Rate Performance by Weld Number',
        xaxis_title='Weld Number',
        yaxis_title='Weld Rate (kg/min)',
        height=400
    )
    
    return fig

def calculate_key_metrics(data):
    """Calculate key performance metrics"""
    actual_data = data.dropna(subset=['Actual Available to Start'])
    
    if len(actual_data) == 0:
        return {
            'total_welds_completed': 0,
            'avg_cycle_time': 0,
            'avg_weld_rate': 0,
            'total_delays': 0,
            'avg_delay_days': 0,
            'rework_rate': 0,
            'excessive_wait_rate': 0,
            'anomaly_rate': 0
        }
    
    # Calculate delays
    actual_data = actual_data.copy()
    actual_data['delay_days'] = (actual_data['Actual Available to Start'] - actual_data['Plan Available to Start']).dt.days
    actual_data['ActualWait'] = actual_data['Actual Wait+Setup (min)'] - actual_data['Estimated Actual Setup (min)']

    
    metrics = {
        'total_welds_completed': len(actual_data),
        'avg_cycle_time': actual_data['Actual Cycle Time (min)'].mean(),
        'avg_wait': actual_data['ActualWait'].mean(),
        'avg_setup': actual_data['Estimated Actual Setup (min)'].mean(),
        'avg_duration': actual_data['Actual Duration (min)'].mean(),
        'total_delays': len(actual_data[actual_data['delay_days'] > 0]),
        'avg_delay_days': actual_data['delay_days'].mean(),
        'rework_rate': actual_data['Rework Flagged'].sum() / len(actual_data) * 100,
        'excessive_wait_rate': actual_data['Excessive Wait Flagged'].sum() / len(actual_data) * 100,
        'anomaly_rate': actual_data['Anomaly Flagged'].sum() / len(actual_data) * 100
    }
    
    return metrics

def main():
    st.title("🔧 Welding Performance Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Data Analysis"])
    
    # Data upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    # Load data
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = process_weld_data(data)
        st.sidebar.success("Custom data loaded successfully!")
    else:
        data = load_default_data()
        if data is not None:
            st.sidebar.info("Using default sample data")
        else:
            st.sidebar.error("No data available. Please upload a CSV file.")
            st.error("No data file found. Please upload a CSV file to continue.")
            return
    
    if page == "Dashboard":
        show_dashboard(data)
    else:
        show_data_analysis(data)

def apply_advanced_filters(data):
    """Apply advanced filtering options and return filtered data"""
    st.sidebar.header("🔍 Advanced Filters")
    
    # Date range filter
    actual_dates = data['Plan Available to Start']#.dropna()
    if not actual_dates.empty:
        min_date = actual_dates.min().date()
        max_date = data['Forecast Start'].max().date()
        
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Filter by Actual Start Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            data = data[
                (data['Plan Available to Start'].dt.date >= start_date) &
                (data['Plan Available to Start'].dt.date <= end_date)
            ]
    
    # Critical Path filter
    st.sidebar.subheader("Critical Path")
    critical_path_options = ['All'] + sorted(data['Critical Path'].unique().tolist())
    selected_critical_path = st.sidebar.selectbox("Critical Path", critical_path_options)
    
    if selected_critical_path != 'All':
        data = data[data['Critical Path'] == selected_critical_path]


    st.sidebar.header('Mitigating Forecast Parameters')

    # Sliders for the requested variables
    MitigatedWait = st.sidebar.slider(
        'Mitigated Wait (min)',
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )
    MitigatedSetup = st.sidebar.slider(
        'Mitigated Setup (min)',
        min_value=0,
        max_value=100,
        value=35,
        step=1
    )
    MitigatedDuration = st.sidebar.slider(
        'Mitigated Duration (min)',
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )
    MitigatedHours = st.sidebar.slider(
        'Mitigated Hours',
        min_value=7,
        max_value=12,
        value=9,
        step=1
    )


    PlanCycleTime = 120
    CurrentDelay = 13
    BaseHours = 8
    MitigatedCycleTime = MitigatedWait + MitigatedSetup + MitigatedDuration
    
    data['Mitigated Forecast'] = data['Plan Available to Start'] + pd.to_timedelta(
        np.floor(CurrentDelay + (MitigatedCycleTime / PlanCycleTime - 1 + (BaseHours / MitigatedHours - 1)/2 ) * (data['Weld Number'] - 50)), 
        unit='D'
    )
    
    data['Mitigated Forecast'] = data[['Plan Available to Start','Mitigated Forecast']].groupby('Plan Available to Start').transform('last')
    data.loc[data['Weld Number'] < 51, ['Mitigated Forecast', 'Forecast Start']] = np.nan
    
    
    
    return data

def show_dashboard(data):
    """Display the main dashboard page"""
    # Apply advanced filters
    original_data = data.copy()
    filtered_data = apply_advanced_filters(data)
    
    # Show filter summary
    if len(filtered_data) != len(original_data):
        st.info(f"📊 Showing {len(filtered_data)} of {len(original_data)} total records (filters applied)")
    
    st.header("Welding Performance Overview")
    
    # Key Metrics
    metrics = calculate_key_metrics(filtered_data)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Welds Completed",
            value=int(metrics['total_welds_completed']),
            delta=f"of {metrics['total_welds_completed']-73:.0f} vs plan", delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Avg Cycle Time",
            value=f"{metrics['avg_cycle_time']:.1f} min",
            delta=f"{metrics['avg_cycle_time'] - 120:.1f} vs plan", delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Wait Time",
            value=f"{metrics['avg_wait']:.1f} min",
            delta=f"{-metrics['avg_wait']:.1f} vs plan"
        )
    
    with col4:
        st.metric(
            label="Avg Setup",
            value=f"{metrics['avg_setup']:.1f} min",
            delta=f"{metrics['avg_setup']-60:.1f} vs plan", delta_color="inverse"
        )

    with col5:
        st.metric(
            label="Avg Welding",
            value=f"{metrics['avg_duration']:.1f} min",
            delta=f"{metrics['avg_duration']-60:.1f} vs plan", delta_color="inverse"
        )
        
    # Quality Metrics
    st.subheader("Quality & Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Work-in-Progress",
            value=f"{1:.1f} Weld"
        )
    
    with col2:
        st.metric(
            label="Rework Rate",
            value=f"{metrics['rework_rate']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Excessive Wait Rate",
            value=f"{metrics['excessive_wait_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Anomaly Rate",
            value=f"{metrics['anomaly_rate']:.1f}%"
        )
    
    # Main completion chart
    st.subheader("Completion Progress")
    completion_chart = create_completion_chart(filtered_data)
    st.plotly_chart(completion_chart, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        cycle_time_chart = create_cycle_time_chart(filtered_data)
        st.plotly_chart(cycle_time_chart, use_container_width=True)
    
    with col2:
        weld_rate_chart = create_weld_rate_chart(filtered_data)
        st.plotly_chart(weld_rate_chart, use_container_width=True)

def show_data_analysis(data):
    """Display the data analysis page"""
    st.header("Data Analysis & Exploration")
    
    # Apply advanced filters (same as dashboard)
    original_data = data.copy()
    filtered_data = apply_advanced_filters(data)
    
    # Additional local filters for data analysis
    st.subheader("📋 Additional Analysis Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Weld number range filter
        min_weld = int(filtered_data['Weld Number'].min()) if not filtered_data.empty else 1
        max_weld = int(filtered_data['Weld Number'].max()) if not filtered_data.empty else 100
        weld_range = st.slider(
            "Weld Number Range",
            min_value=min_weld,
            max_value=max_weld,
            value=(min_weld, max_weld),
            key="analysis_weld_range"
        )
        
        filtered_data = filtered_data[
            (filtered_data['Weld Number'] >= weld_range[0]) & 
            (filtered_data['Weld Number'] <= weld_range[1])
        ]
    
    with col2:
        # Inspection status filter
        inspection_options = ['All', 'Inspected Only', 'Passed Only', 'Failed Only']
        inspection_filter = st.selectbox("Inspection Status", inspection_options)
        
        if inspection_filter == 'Inspected Only':
            filtered_data = filtered_data[filtered_data['Inspection'] == 1]
        elif inspection_filter == 'Passed Only':
            filtered_data = filtered_data[filtered_data['Inspection Pass'] == 1]
        elif inspection_filter == 'Failed Only':
            filtered_data = filtered_data[filtered_data['Inspection Fail'] == 1]
    
    with col3:
        # Completion status filter
        completion_options = ['All', 'Completed Only', 'Planned Only']
        completion_filter = st.selectbox("Completion Status", completion_options)
        
        if completion_filter == 'Completed Only':
            filtered_data = filtered_data[filtered_data['Actual Available to Start'].notna()]
        elif completion_filter == 'Planned Only':
            filtered_data = filtered_data[filtered_data['Actual Available to Start'].isna()]
    
    st.write(f"📊 Showing {len(filtered_data)} of {len(original_data)} total rows")
    
    # Raw data table
    st.subheader("Raw Data Table")
    
    # Column selection for display
    all_columns = list(data.columns)
    display_columns = st.multiselect(
        "Select columns to display",
        all_columns,
        default=['Weld Number', 'Critical Path', 'Plan Available to Start', 
                'Actual Available to Start', 'Actual Cycle Time (min)', 'Weld Rate (kg / min)']
    )
    
    if display_columns:
        st.dataframe(
            filtered_data[display_columns],
            use_container_width=True,
            height=400
        )
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    # Select numeric columns for analysis
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns:
        selected_numeric_cols = st.multiselect(
            "Select numeric columns for summary",
            numeric_columns,
            default=['Actual Cycle Time (min)', 'Weld Rate (kg / min)', 'Actual Total Metal (kg)']
        )
        
        if selected_numeric_cols:
            summary_stats = filtered_data[selected_numeric_cols].describe()
            st.dataframe(summary_stats, use_container_width=True)
    
    # Histogram generator
    st.subheader("Histogram Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hist_column = st.selectbox("Select column for histogram", numeric_columns)
    
    with col2:
        bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
    
    if hist_column:
        hist_data = filtered_data[hist_column].dropna()
        
        if len(hist_data) > 0:
            fig = px.histogram(
                hist_data,
                nbins=bins,
                title=f'Distribution of {hist_column}',
                labels={'value': hist_column, 'count': 'Frequency'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic statistics for the histogram
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{hist_data.mean():.2f}")
            with col2:
                st.metric("Std Dev", f"{hist_data.std():.2f}")
            with col3:
                st.metric("Min", f"{hist_data.min():.2f}")
            with col4:
                st.metric("Max", f"{hist_data.max():.2f}")
    
    # Pivot table functionality
    st.subheader("Pivot Table Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pivot_index = st.selectbox("Index (rows)", ['Critical Path', 'Weld Number'] + 
                                  [col for col in data.columns if data[col].dtype == 'object'])
    
    with col2:
        pivot_values = st.selectbox("Values (aggregation)", numeric_columns)
    
    with col3:
        pivot_aggfunc = st.selectbox("Aggregation function", ['mean', 'sum', 'count', 'min', 'max'])
    
    if pivot_index and pivot_values:
        try:
            pivot_table = filtered_data.pivot_table(
                index=pivot_index,
                values=pivot_values,
                aggfunc=pivot_aggfunc
            )
            st.dataframe(pivot_table, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pivot table: {str(e)}")
    
    # Data export
    st.subheader("Export Data")
    
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"welding_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
