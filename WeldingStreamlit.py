#from matplotlib.ticker import scale_range
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Welding Production Model",
    page_icon="🔧",
    layout="wide"
)

@st.cache_data
def load_default_data():
    """Load the default welding data CSV file"""
    try:
        data = pd.read_csv('WeldData.csv')
        return process_weld_data(data)
    except FileNotFoundError:
        return None

@st.cache_data
def load_classification_data():
    """Load the classification graph data CSV file"""
    try:
        data = pd.read_csv('ClassificationGraph.csv', index_col='datetime', parse_dates=True)
        # Normalize column names: strip spaces and standardize case
 #       data.columns = [c.strip() for c in data.columns]
 #       # Coerce expected columns to numeric where present
 #       for col in ['PredWeldNumber', 'ActualWeldNumber', 'PredictionStart', 'WeldStart']:
 #           if col in data.columns:
 #               data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        return data
    except FileNotFoundError:
        return None


@st.cache_data
def load_quality_data():
    """Load the default welding data CSV file"""
    try:
        data = pd.read_csv('QualityGraphData.csv') #, index_col='datetime_min', parse_dates=True
        return data
    except FileNotFoundError:
        return None

def process_weld_data(data):
    """Process welding data with date parsing and forecast calculations"""
    # Convert date columns
    data['datetime_min'] = pd.to_datetime(data['datetime_min'])
    data['datetime_max'] = pd.to_datetime(data['datetime_max'])
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data['EndTime'] = pd.to_datetime(data['EndTime'])
    data['ForecastStart'] = pd.to_datetime(data['ForecastStart'])
    data['ForecastEnd'] = pd.to_datetime(data['ForecastEnd'])
    data['SimulationStart'] = pd.to_datetime(data['SimulationStart'])
    data['SimulationEnd'] = pd.to_datetime(data['SimulationEnd'])
    
    
    Start = len(data['Duration'].dropna())
    MitigatedWait = pd.Timedelta(seconds = .75 * 4*5*9.5*60*60 / data.dropna().rolling("28D", on="datetime_min")["WeldNumber"].count().iloc[-1])
    MitigatedSetup = pd.Timedelta(seconds = data['EstSetupTime'].dropna().mean())
    MitigatedDuration = pd.Timedelta(seconds = data['Duration'].dropna().mean())
    MitigatedHours = 9
    BaseHours = 9
    MitigatedCycleTime = MitigatedWait + MitigatedSetup + MitigatedDuration
    
    for i in range(Start,len(data)):
        StartTime = data.loc[i-1,'SimulationEnd'] + MitigatedWait + MitigatedSetup 
        if StartTime.hour < 8 + MitigatedHours:
            data.loc[i,'SimulationStart'] = StartTime
            data.loc[i,'SimulationEnd'] = StartTime + MitigatedDuration
        else:
            data.loc[i,'SimulationStart'] = (StartTime + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=8)
            data.loc[i,'SimulationEnd'] = data.loc[i,'SimulationStart'] + MitigatedDuration
            while data.loc[i,'SimulationEnd'].weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                data.loc[i,'SimulationEnd'] += pd.Timedelta(days=1)

    
    return data

def create_completion_chart(data):
    """Create the main completion chart showing actual vs planned vs forecast"""
    fig = go.Figure()
    
    # Welds Completed (Actual)
    actual_data = data.dropna(subset=['datetime_max'])
    marker_sizes = actual_data['CycleTime'] / 60 if 'CycleTime' in actual_data.columns else None
    fig.add_trace(go.Scatter(
        x=actual_data['datetime_max'],
        y=actual_data['WeldNumber'],
        mode='lines+markers',
        name='Welds Completed',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=(marker_sizes.fillna(0) if marker_sizes is not None else 6), color="#1f77b4", sizemode="area", sizeref=actual_data['CycleTime'].mean()/60),
        hovertemplate=("Weld Number: %{y}<br>Cycle Time (min): %{marker.size:.0f}<extra></extra>" if marker_sizes is not None else None)
    ))
    
    # Original Plan
    fig.add_trace(go.Scatter(
        x=data['EndTime'],
        y=data['WeldNumber'],
        mode='lines',
        name='Original Plan',
        line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
        hoverinfo='skip'
    ))
    
    # Forecast
    forecast_data = data.dropna(subset=['ForecastEnd'])
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['ForecastEnd'],
            y=forecast_data['WeldNumber'],
            mode='lines',
            name='Current Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Mitigated Forecast
    mitigated_data = data.dropna(subset=['SimulationEnd'])
    if not mitigated_data.empty:
        fig.add_trace(go.Scatter(
            x=mitigated_data['SimulationEnd'],
            y=mitigated_data['WeldNumber'],
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
    actual_data = data.dropna(subset=['CycleTime'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data['WeldNumber'],
        y=actual_data['CycleTime'],
        mode='markers',
        name='Actual Cycle Time',
        marker=dict(
            color=actual_data['CycleTime'],
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
    actual_data = data.dropna(subset=['Energy'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_data['WeldNumber'],
        y=actual_data['Energy'],
        mode='markers',
        name='Weld Rate',
        marker=dict(
            color=actual_data['Energy'],
            colorscale='Plasma',
            size=8,
            colorbar=dict(title="Weld Energy (KJ)")
        )
    ))
    
    fig.update_layout(
        title='Weld Rate Performance by Weld Number',
        xaxis_title='Weld Number',
        yaxis_title='Weld Energy (KJ)',
        height=400
    )
    
    return fig

def calculate_key_metrics(data):
    """Calculate key performance metrics"""
    # Choose an available completion timestamp
    date_col = None
    for candidate in ['datetime_max', 'EndTime']:
        if candidate in data.columns:
            date_col = candidate
            break
    if date_col is None:
        return {
            'total_welds_completed': 0,
            'avg_cycle_time': 0,
            'avg_weld_rate': 0,
            'total_delays': 0,
            'avg_delay_days': 0,
            'rework_rate': 0,
            'excessive_wait_rate': 0,
            'anomaly_rate': 0,
            'avg_wait': 0,
            'avg_setup': 0,
            'avg_duration': 0,
        }
    
    actual_data = data.dropna(subset=[date_col])
    plan_data = data.loc[data['EndTime']<data['datetime_max'].max(),['WeldNumber','DesignWaitTime','DesignSetupTime','DesignDuration']].copy()


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
    actual_data['delay_days'] = (actual_data['datetime_min'] - actual_data['EndTime']).dt.days
    actual_data['ActualWait'] = actual_data['WaitTimeCalc']

    
    metrics = {
        'total_welds_completed': actual_data['WeldNumber'].iloc[-1],
        'plan_welds_complete': plan_data['WeldNumber'].iloc[-1],
        'avg_cycle_time': actual_data['CycleTime'].mean()/60,
        'plan_cycle_time': (plan_data['DesignWaitTime'].mean()+plan_data['DesignSetupTime'].mean()+plan_data['DesignDuration'].mean())/60,
        'avg_wait': actual_data['WaitTimeCalc'].mean()/60,
        'plan_wait': plan_data['DesignWaitTime'].mean()/60, 
        'avg_setup': actual_data['EstSetupTime'].mean(),
        'plan_setup': plan_data['DesignSetupTime'].mean(),
        'avg_duration': actual_data['Duration'].mean(),
        'plan_duration': plan_data['DesignDuration'].mean(),
#        'total_delays': len(actual_data[actual_data['delay_days'] > 0]),
#        'avg_delay_days': actual_data['delay_days'].mean(),
        'rework_rate': 0,
        'work_in_progress': actual_data['WIP'].dropna().iloc[-1],
        'excessive_wait_rate': (actual_data['WaitTime']>actual_data['WaitTime'].median()*10).sum() / len(actual_data) * 100,
        'anomaly_rate': actual_data['QualityFlag'].sum() / len(actual_data) * 100
    }
    
    return metrics

def create_classification_graph(GraphData):
    """Create the weld classification graph"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=GraphData.index, y=GraphData['PredWeldNumber'],
                             mode="lines", name="Model", marker=dict(color="#1f77b4")))

    fig.add_trace(go.Scatter(x=GraphData.index, y=GraphData['ActualWeldNumber'],
                             mode="lines", name="Actual", marker=dict(color="rgba(0, 0, 0, 0.3)")))

    # Make it "web app" style
    fig.update_layout(
        title="Weld Classification",
        yaxis_title="Welds",
        xaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18),
        title_font=dict(size=28, color="black", family="Arial"),
        hovermode="x unified",
        height=500
    )

    # Add gridlines
    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray", 
                     tickfont=dict(size=20))
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
                     tickfont=dict(size=20), titlefont=dict(size=20))

    return fig

def show_weld_classification(data):
    """Display the weld classification page"""
    st.header("🔍 Weld Classification Analysis")
    
    # Load classification data
    classification_data = load_classification_data()
    
    if classification_data is None:
        st.error("ClassificationGraph.csv file not found. Please ensure the file is in the same directory.")
        return
    
    st.subheader("Weld Classification Model vs Actual")
    
    # Create and display the classification graph
    classification_graph = create_classification_graph(classification_data)
    st.plotly_chart(classification_graph, use_container_width=True)
    
    # Classification metrics
    st.subheader("Classification Performance Metrics")

    # Calculate metrics
    if 'PredictionStart' in classification_data.columns and 'WeldStart' in classification_data.columns:
            # Calculate precision, recall, and f1-score
            # For regression tasks, we'll use a threshold-based approach
            pred_values = classification_data['PredictionStart']
            actual_values = classification_data['WeldStart']
            
            # Calculate precision (true positives / (true positives + false positives))
            # For regression, we consider predictions within threshold as "correct"
            correct_predictions = np.abs(pred_values - actual_values)
            TP = np.sum((pred_values == 1) & (actual_values == 1))
            FP = np.sum((pred_values == 1) & (actual_values == 0))
            FN = np.sum((pred_values == 0) & (actual_values == 1))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            
            # Calculate f1-score (harmonic mean of precision and recall)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = max(0, 100 - (np.mean(np.abs(pred_values - actual_values)) / actual_values.mean() * 100))
                st.metric("Model Accuracy", f"{accuracy:.1f}%")

            with col2:
                st.metric("Precision", f"{precision:.3f}")
            
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            
            with col4:
                st.metric("F1-Score", f"{f1_score:.3f}")
            
    
    # Data table
    st.subheader("Classification Data")
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(classification_data, use_container_width=True, height=400)
    
    with col2:
        st.subheader("Data Summary")
        st.write(f"**Total Records:** {len(classification_data)}")
        
        if 'PredictionStart' in classification_data.columns:
            st.write(f"**Avg Predicted:** {classification_data['PredictionStart'].mean():.1f}")
        
        if 'WeldStart' in classification_data.columns:
            st.write(f"**Avg Actual:** {classification_data['WeldStart'].mean():.1f}")
        
        # Show available columns
        st.write("**Available Columns:**")
        for col in classification_data.columns:
            st.write(f"- {col}")
    
    # Analysis section
    st.subheader("Classification Analysis")
    
    # Create residual plot
    if 'PredictionStart' in classification_data.columns and 'WeldStart' in classification_data.columns:
        residuals = classification_data['WeldStart'] - classification_data['PredictionStart']
        
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(
            x=classification_data.index,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='red', size=6, opacity=0.7)
        ))
        
        # Add zero line
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig_residuals.update_layout(
            title="Prediction Residuals (Actual - Predicted)",
            xaxis_title="Index",
            yaxis_title="Residuals",
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    
    if st.button("Download Classification Data as CSV"):
        csv = classification_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"classification_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def create_quality_prediction_graph(data):
    """Create the quality prediction graph"""
    GraphData = data.copy()

    ## Classification Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=GraphData['datetime_min'], y=GraphData['Correctly Identified'],
                             mode="lines", name="Correctly Identified", marker=dict(color="#1f77b4")))

    fig.add_trace(go.Scatter(x=GraphData['datetime_min'], y=GraphData['Misclassfied'],
                             mode="lines", name="Misclassified", marker=dict(color="rgba(0, 0, 0, 0.3)")))

    # Make it "web app" style
    fig.update_layout(
        title = "Quality Prediction",
        yaxis_title="Welds",
        xaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        title_font=dict(size=20, color="#003366", family="Arial", weight="bold"),
        hovermode="x unified",
        height=500
    )

    # Add gridlines
    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray")
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray")

    return fig, GraphData

def show_weld_quality_prediction(data):
    """Display the weld quality prediction page"""
    st.header("🎯 Weld Quality Prediction")
    
    # Check if required columns exist
    if 'datetime_min' not in data.columns or 'QualityFlag' not in data.columns:
        st.error("Required columns 'datetime_min' and 'QualityFlag' not found in data.")
        return

    QualityData = load_quality_data()
    
    st.subheader("Quality Prediction Model vs Actual")
    
    # Create and display the quality prediction graph
    quality_graph, graph_data = create_quality_prediction_graph(QualityData)
    st.plotly_chart(quality_graph, use_container_width=True)
    
    # Quality prediction metrics
    st.subheader("Quality Prediction Performance Metrics")
    
    # Calculate metrics
    correct_predictions = (graph_data['QualityPred'] == graph_data['QualityFlag']).sum()
    total_predictions = len(graph_data)
    accuracy = correct_predictions / total_predictions * 100
    
    # Calculate precision, recall, and f1-score
    TP = np.sum((graph_data['QualityPred'] == 1) & (graph_data['QualityFlag'] == 1))
    FP = np.sum((graph_data['QualityPred'] == 1) & (graph_data['QualityFlag'] == 0))
    FN = np.sum((graph_data['QualityPred'] == 0) & (graph_data['QualityFlag'] == 1))
    TN = np.sum((graph_data['QualityPred'] == 0) & (graph_data['QualityFlag'] == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{f1_score:.3f}")
    
    # Data table
    st.subheader("Quality Prediction Data")
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(graph_data, use_container_width=True, height=400)
    
    with col2:
        st.subheader("Data Summary")
        st.write(f"**Total Records:** {len(graph_data)}")
        st.write(f"**Correctly Identified:** {graph_data['Correctly Identified'].iloc[-1]}")
        st.write(f"**Misclassified:** {graph_data['Misclassfied'].iloc[-1]}")
        st.write(f"**Actual Quality Issues:** {graph_data['QualityFlag'].sum()}")
        st.write(f"**Predicted Quality Issues:** {graph_data['QualityPred'].sum()}")
        
        # Show available columns
        st.write("**Available Columns:**")
        for col in graph_data.columns:
            st.write(f"- {col}")
    
    # Analysis section
    st.subheader("Quality Prediction Analysis")
    
    # Create confusion matrix visualization
    confusion_matrix = np.array([[TN, FP], [FN, TP]])
    
    fig_confusion = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Good', 'Predicted Bad'],
        y=['Actual Good', 'Actual Bad'],
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    
    fig_confusion.update_layout(
        title="Confusion Matrix",
        height=400
    )
    
    st.plotly_chart(fig_confusion, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    
    if st.button("Download Quality Prediction Data as CSV"):
        csv = graph_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"quality_prediction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    st.title("🔧 Welding Production Model")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Data Analysis", "Weld Classification", "Weld Quality Prediction"])
    
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
    elif page == "Data Analysis":
        show_data_analysis(data)
    elif page == "Weld Classification":
        show_weld_classification(data)
    else:  # Weld Quality Prediction
        show_weld_quality_prediction(data)

def apply_advanced_filters(data):
    """Apply advanced filtering options and return filtered data"""
    st.sidebar.header('Mitigating Forecast Parameters')

    BaseWait = int(np.rint(pd.Timedelta(seconds = .75*4*5*9.5*60*60 / data.dropna().rolling("28D", on="datetime_min")["WeldNumber"].count().iloc[-1]).total_seconds()/60))
    BaseSetup = int(np.rint(pd.Timedelta(seconds = data['EstSetupTime'].dropna().mean()).total_seconds()))
    BaseDuration = int(np.rint(pd.Timedelta(seconds = data['Duration'].dropna().mean()).total_seconds()))


    # Sliders for the requested variables
    MitigatedWait = st.sidebar.slider(
        'Mitigated Wait (min)',
        min_value=0,
        max_value=(BaseWait * 2),
        value=BaseWait,
        step=1
    )
    MitigatedSetup = st.sidebar.slider(
        'Mitigated Setup (sec)',
        min_value=0,
        max_value=BaseSetup*2,
        value=BaseSetup,
        step=1
    )
    MitigatedDuration = st.sidebar.slider(
        'Mitigated Duration (sec)',
        min_value=0,
        max_value=(BaseDuration*2),
        value=BaseDuration,
        step=1
    )
    MitigatedHours = st.sidebar.slider(
        'Mitigated Hours',
        min_value=7,
        max_value=12,
        value=9,
        step=1
    )
    MitigatedRework = st.sidebar.slider(
        'Mitigated Rework Rate (%)',
        min_value=0,
        max_value=25,
        value=0,
        step=1
    )

    MitigatedWait = pd.Timedelta(minutes = MitigatedWait / (1 - MitigatedRework/100))
    MitigatedSetup = pd.Timedelta(seconds = MitigatedSetup / (1 - MitigatedRework/100))
    MitigatedDuration = pd.Timedelta(seconds = MitigatedDuration / (1 - MitigatedRework/100))

    Start = len(data['Duration'].dropna())
    

    for i in range(Start,len(data)):
        StartTime = data.loc[i-1,'SimulationEnd'] + MitigatedWait + MitigatedSetup 
        if StartTime.hour < 8 + MitigatedHours:
            data.loc[i,'SimulationStart'] = StartTime
            data.loc[i,'SimulationEnd'] = StartTime + MitigatedDuration
        else:
            data.loc[i,'SimulationStart'] = (StartTime + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=8)
            data.loc[i,'SimulationEnd'] = data.loc[i,'SimulationStart'] + MitigatedDuration
            while data.loc[i,'SimulationEnd'].weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                data.loc[i,'SimulationEnd'] += pd.Timedelta(days=1)

    data.loc[data['WeldNumber'] < Start, ['ForecastEnd', 'SimulationEnd']] = np.nan
    
    
    st.sidebar.header("🔍 Advanced Filters")
    
    # Date range filter
    actual_dates = data['ForecastStart']#.dropna()
    if not actual_dates.empty:
        min_date = actual_dates.min().date()
        max_date = data['ForecastStart'].max().date()
        
        st.sidebar.subheader("Actual Date Range")
        date_range = st.sidebar.date_input(
            "Filter by Actual Start Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            data = data[
                (data['ForecastStart'].dt.date >= start_date) &
                (data['ForecastStart'].dt.date <= end_date)
            ]

    # Weld number range filter (supports either 'WeldNumber' or 'Weld Number')
 #   weld_col = None
 #   if 'WeldNumber' in data.columns:
 #       weld_col = 'WeldNumber'
 #   elif 'Weld Number' in data.columns:
  #      weld_col = 'Weld Number'

    weld_col = 'WeldNumber'

    if weld_col is not None and not data[weld_col].dropna().empty:
        st.sidebar.subheader("Weld Number Range")
        min_weld = int(np.floor(data[weld_col].min()))
        max_weld = int(np.ceil(data[weld_col].max()))
        weld_range = st.sidebar.slider(
            "Select weld number range",
            min_value=min_weld,
            max_value=max_weld,
            value=(min_weld, max_weld),
            key="weld_number_range"
        )
        data = data[(data[weld_col] >= weld_range[0]) & (data[weld_col] <= weld_range[1])]

   # Critical Path filter
    st.sidebar.subheader("Critical Path")
    critical_path_options = ['All'] + sorted(data['Critical Path'].unique().tolist())
    selected_critical_path = st.sidebar.selectbox("Critical Path", critical_path_options)
    
    if selected_critical_path != 'All':
        data = data[data['Critical Path'] == selected_critical_path]

 

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
            delta=f"of {metrics['total_welds_completed']-metrics['plan_welds_complete']:.0f} vs plan", delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Avg Cycle Time",
            value=f"{metrics['avg_cycle_time']:.1f} min",
            delta=f"{metrics['avg_cycle_time'] - metrics['plan_cycle_time']:.1f} vs plan", delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Avg Wait Time",
            value=f"{metrics['avg_wait']:.1f} min",
            delta=f"{metrics['avg_wait'] - metrics['plan_wait']:.1f} vs plan", delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Avg Setup",
            value=f"{metrics['avg_setup']:.1f} sec",
            delta=f"{metrics['avg_setup']-metrics['plan_setup']:.1f} vs plan", delta_color="inverse"
        )

    with col5:
        st.metric(
            label="Avg Welding",
            value=f"{metrics['avg_duration']:.1f} sec",
            delta=f"{metrics['avg_duration']-metrics['plan_duration']:.1f} vs plan", delta_color="inverse"
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
        # Weld number range filter (use 'WeldNumber')
        if 'WeldNumber' in filtered_data.columns and not filtered_data['WeldNumber'].dropna().empty:
            min_weld = int(np.floor(filtered_data['WeldNumber'].min()))
            max_weld = int(np.ceil(filtered_data['WeldNumber'].max()))
            weld_range = st.slider(
                "Weld Number Range",
                min_value=min_weld,
                max_value=max_weld,
                value=(min_weld, max_weld),
                key="analysis_weld_range"
            )
            
            filtered_data = filtered_data[
                (filtered_data['WeldNumber'] >= weld_range[0]) & 
                (filtered_data['WeldNumber'] <= weld_range[1])
            ]
    
    with col2:
        # Completion status filter based on presence of 'datetime_max'
        completion_options = ['All', 'Completed Only', 'Planned Only']
        completion_filter = st.selectbox("Completion Status", completion_options)
        
        if completion_filter == 'Completed Only' and 'datetime_max' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['datetime_max'].notna()]
        elif completion_filter == 'Planned Only' and 'datetime_max' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['datetime_max'].isna()]
    
    with col3:
        # Anomaly status filter using 'QualityFlag' if present
        anomaly_options = ['All', 'Anomalies Only', 'No Anomalies']
        anomaly_filter = st.selectbox("Anomaly Status", anomaly_options)
        if 'QualityFlag' in filtered_data.columns:
            if anomaly_filter == 'Anomalies Only':
                filtered_data = filtered_data[filtered_data['QualityFlag'] == 1]
            elif anomaly_filter == 'No Anomalies':
                filtered_data = filtered_data[filtered_data['QualityFlag'] == 0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Cycle time range filter (use 'CycleTime' in seconds, display in minutes)
        if 'CycleTime' in filtered_data.columns and not filtered_data['CycleTime'].dropna().empty:
            min_cycle_min = int(np.floor(filtered_data['CycleTime'].min() / 60))
            max_cycle_min = int(np.ceil(filtered_data['CycleTime'].max() / 60))
            cycle_time_range = st.slider(
                "Actual Cycle Time Range (min)",
                min_value=min_cycle_min,
                max_value=max_cycle_min,
                value=(min_cycle_min, max_cycle_min),
                key="analysis_cycle_time_range"
            )
            
            filtered_data = filtered_data[
                (filtered_data['CycleTime'] >= cycle_time_range[0] * 60) & 
                (filtered_data['CycleTime'] <= cycle_time_range[1] * 60)
            ]
        
    with col2:
        # Energy range filter (optional)
        if 'Energy' in filtered_data.columns and not filtered_data['Energy'].dropna().empty:
            min_energy = float(np.floor(filtered_data['Energy'].min()))
            max_energy = float(np.ceil(filtered_data['Energy'].max()))
            energy_range = st.slider(
                "Weld Energy Range (KJ)",
                min_value=min_energy,
                max_value=max_energy,
                value=(min_energy, max_energy),
                key="analysis_energy_range"
            )
            filtered_data = filtered_data[
                (filtered_data['Energy'] >= energy_range[0]) & 
                (filtered_data['Energy'] <= energy_range[1])
            ]

    with col3:
        # Date completion filter (optional)
        if 'datetime_max' in filtered_data.columns and filtered_data['datetime_max'].notna().any():
            dates = filtered_data['datetime_max'].dropna()
            min_date = dates.min().date()
            max_date = dates.max().date()
            completion_date_range = st.date_input(
                "Completion Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="analysis_completion_date_range"
            )
            if isinstance(completion_date_range, (list, tuple)) and len(completion_date_range) == 2:
                start_date, end_date = completion_date_range
                filtered_data = filtered_data[
                    (filtered_data['datetime_max'].dt.date >= start_date) &
                    (filtered_data['datetime_max'].dt.date <= end_date)
                ]
            

    st.write(f"📊 Showing {len(filtered_data)} of {len(original_data)} total rows")
    



    # Raw data table
    st.subheader("Raw Data Table")
    
    # Column selection for display
    all_columns = list(data.columns)
    display_columns = st.multiselect(
        "Select columns to display",
        all_columns,
        default=[
            'WeldNumber', 'Critical Path', 'ForecastStart', 'ForecastEnd',
            'datetime_min', 'datetime_max', 'CycleTime', 'Energy', 'EstSetupTime', 'Duration'
        ]
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
            default=['WeldNumber', 'Energy', 'Duration']
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
        pivot_index = st.selectbox("Index (rows)", [col for col in data.columns if data[col].max()==1])#.dtype == 'object'])
    
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
