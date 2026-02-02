"""
CDR Data Analyser - Professional Edition
Modern UI with Enhanced Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="CDR Data Analyser Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.8);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .metric-card.blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .metric-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    .metric-card.purple {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }

    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e0e5ec 0%, #f5f7fa 100%);
        border-left: 5px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }

    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }

    /* Metric delta styling */
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }

    /* Card container */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #f0f0f0;
    }

    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Stats row */
    .stats-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }

    /* Insight cards */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .insight-card.positive {
        border-color: #38ef7d;
        background: linear-gradient(90deg, rgba(56, 239, 125, 0.1) 0%, white 100%);
    }

    .insight-card.negative {
        border-color: #f5576c;
        background: linear-gradient(90deg, rgba(245, 87, 108, 0.1) 0%, white 100%);
    }

    .insight-card.neutral {
        border-color: #667eea;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, white 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for charts
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#38ef7d',
    'warning': '#ffc107',
    'danger': '#f5576c',
    'info': '#4facfe',
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ffecd2', '#fcb69f', '#a1c4fd', '#c2e9fb', '#d4fc79', '#96e6a1']
}

# ==================== HELPER FUNCTIONS ====================

def parse_duration_to_seconds(duration_str):
    if pd.isna(duration_str) or duration_str == '':
        return 0
    try:
        if isinstance(duration_str, (int, float)):
            return float(duration_str)
        parts = str(duration_str).split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        return float(duration_str)
    except:
        return 0

def get_duration_bucket(seconds):
    if pd.isna(seconds) or seconds == 0:
        return "0 seconds"
    elif seconds < 60:
        return "< 1 min"
    elif seconds < 120:
        return "1-2 mins"
    elif seconds < 180:
        return "2-3 mins"
    elif seconds < 300:
        return "3-5 mins"
    elif seconds < 360:
        return "5-6 mins"
    elif seconds < 420:
        return "6-7 mins"
    else:
        return "> 7 mins"

def format_seconds(seconds):
    if pd.isna(seconds) or seconds == 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(int(num))

def is_sale_disposition(disposition, disp_type):
    if pd.isna(disposition):
        return False
    sale_keywords = ['sale', 'sold', 'converted', 'done', 'booked', 'confirmed']
    disp_lower = str(disposition).lower()
    type_lower = str(disp_type).lower() if pd.notna(disp_type) else ''
    return any(kw in disp_lower for kw in sale_keywords) or 'sale' in type_lower

def is_connected(call_status, talk_seconds):
    if pd.isna(call_status):
        return talk_seconds > 0
    return str(call_status).upper() == 'CONNECTED' or talk_seconds > 0

# ==================== DATA LOADING ====================

@st.cache_data
def load_and_process_data(uploaded_file, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)

        if isinstance(df, dict):
            df = df[list(df.keys())[0]]

        if 'Talk Duration' in df.columns:
            df['Talk_Seconds'] = df['Talk Duration'].apply(parse_duration_to_seconds)
        else:
            df['Talk_Seconds'] = 0

        if 'Call Duration' in df.columns:
            df['Call_Seconds'] = df['Call Duration'].apply(parse_duration_to_seconds)
        else:
            df['Call_Seconds'] = 0

        df['Duration_Bucket'] = df['Talk_Seconds'].apply(get_duration_bucket)

        for col in ['CallStartDate', 'Call Start Date', 'datetime', 'Date']:
            if col in df.columns:
                df['CallDateTime'] = pd.to_datetime(df[col], errors='coerce')
                df['Hour'] = df['CallDateTime'].dt.hour
                df['Date'] = df['CallDateTime'].dt.date
                df['Interval'] = df['Hour'].apply(lambda x: f"{x:02d}:00-{x:02d}:59" if pd.notna(x) else "Unknown")
                df['Day'] = df['CallDateTime'].dt.day_name()
                break

        df['Is_Connected'] = df.apply(
            lambda row: is_connected(row.get('Call Status', None), row['Talk_Seconds']), axis=1
        )

        if 'Disposition' in df.columns:
            df['Is_Sale'] = df.apply(
                lambda row: is_sale_disposition(row.get('Disposition', None), row.get('DispositionType', None)), axis=1
            )
        else:
            df['Is_Sale'] = False

        for col, target in [('Username', 'Agent'), ('User Full Name', 'AgentName'),
                           ('Trunk Group Name', 'Trunk'), ('Call Type', 'CallType')]:
            if col in df.columns:
                df[target] = df[col]

        return df, None
    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}"

# ==================== ANALYSIS FUNCTIONS ====================

def get_overview_metrics(df):
    total_calls = len(df)
    connected_calls = df['Is_Connected'].sum()
    total_sales = df['Is_Sale'].sum()
    connectivity_rate = (connected_calls / total_calls * 100) if total_calls > 0 else 0
    conversion_rate = (total_sales / total_calls * 100) if total_calls > 0 else 0
    avg_talk_time = df[df['Talk_Seconds'] > 0]['Talk_Seconds'].mean()
    total_talk_time = df['Talk_Seconds'].sum()

    return {
        'total_calls': total_calls,
        'connected_calls': int(connected_calls),
        'not_connected': total_calls - int(connected_calls),
        'total_sales': int(total_sales),
        'connectivity_rate': round(connectivity_rate, 2),
        'conversion_rate': round(conversion_rate, 2),
        'avg_talk_time': round(avg_talk_time, 2) if pd.notna(avg_talk_time) else 0,
        'total_talk_time_hrs': round(total_talk_time / 3600, 2)
    }

def analyze_dispositions(df):
    if 'Disposition' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('Disposition').agg(
        Call_Count=('Disposition', 'count'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum')
    ).reset_index()
    analysis['Percentage'] = (analysis['Call_Count'] / len(df) * 100).round(2)
    analysis['Avg_Talk_Sec'] = analysis['Avg_Talk_Sec'].round(1)
    return analysis.sort_values('Call_Count', ascending=False)

def analyze_agent_disposition(df):
    if 'Agent' not in df.columns or 'Disposition' not in df.columns:
        return pd.DataFrame()
    agent_df = df[df['Agent'].notna()]
    pivot = pd.crosstab(agent_df['Agent'], agent_df['Disposition'], margins=True, margins_name='Total')
    return pivot

def analyze_campaign_disposition(df):
    if 'Campaign' not in df.columns or 'Disposition' not in df.columns:
        return pd.DataFrame()
    pivot = pd.crosstab(df['Campaign'], df['Disposition'], margins=True, margins_name='Total')
    return pivot

def analyze_connectivity_by_interval(df):
    if 'Interval' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('Interval').agg(
        Total_Calls=('Is_Connected', 'count'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Not_Connected'] = analysis['Total_Calls'] - analysis['Connected']
    analysis['Connectivity_Pct'] = (analysis['Connected'] / analysis['Total_Calls'] * 100).round(2)
    analysis['Conversion_Pct'] = (analysis['Sales'] / analysis['Total_Calls'] * 100).round(2)
    return analysis.sort_values('Interval')

def analyze_connectivity_by_trunk(df):
    if 'Trunk' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('Trunk').agg(
        Total_Calls=('Is_Connected', 'count'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Not_Connected'] = analysis['Total_Calls'] - analysis['Connected']
    analysis['Connectivity_Pct'] = (analysis['Connected'] / analysis['Total_Calls'] * 100).round(2)
    return analysis.sort_values('Total_Calls', ascending=False)

def analyze_sales_by_duration(df):
    sales_df = df[df['Is_Sale'] == True]
    if len(sales_df) == 0:
        return pd.DataFrame()
    bucket_order = ["0 seconds", "< 1 min", "1-2 mins", "2-3 mins", "3-5 mins", "5-6 mins", "6-7 mins", "> 7 mins"]
    analysis = sales_df.groupby('Duration_Bucket').agg(
        Sale_Count=('Is_Sale', 'count'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean'),
        Total_Talk_Sec=('Talk_Seconds', 'sum')
    ).reset_index()
    total_sales = analysis['Sale_Count'].sum()
    analysis['Percentage'] = (analysis['Sale_Count'] / total_sales * 100).round(2)
    analysis['sort_order'] = analysis['Duration_Bucket'].apply(lambda x: bucket_order.index(x) if x in bucket_order else 99)
    return analysis.sort_values('sort_order').drop('sort_order', axis=1)

def analyze_sales_by_agent(df):
    if 'Agent' not in df.columns:
        return pd.DataFrame()
    agent_df = df[df['Agent'].notna()]
    analysis = agent_df.groupby('Agent').agg(
        Total_Calls=('Agent', 'count'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Connectivity_Pct'] = (analysis['Connected'] / analysis['Total_Calls'] * 100).round(2)
    analysis['Conversion_Pct'] = (analysis['Sales'] / analysis['Total_Calls'] * 100).round(2)
    analysis['Sales_Per_Connected'] = (analysis['Sales'] / analysis['Connected'].replace(0, 1) * 100).round(2)
    analysis['Talk_Time_Formatted'] = analysis['Total_Talk_Sec'].apply(format_seconds)
    return analysis.sort_values('Sales', ascending=False)

def analyze_sales_by_agent_duration(df):
    if 'Agent' not in df.columns:
        return pd.DataFrame()
    sales_df = df[(df['Is_Sale'] == True) & (df['Agent'].notna())]
    if len(sales_df) == 0:
        return pd.DataFrame()
    pivot = pd.crosstab(sales_df['Agent'], sales_df['Duration_Bucket'], margins=True, margins_name='Total')
    bucket_order = ["0 seconds", "< 1 min", "1-2 mins", "2-3 mins", "3-5 mins", "5-6 mins", "6-7 mins", "> 7 mins", "Total"]
    ordered_cols = [c for c in bucket_order if c in pivot.columns]
    return pivot[ordered_cols]

def analyze_campaign_performance(df):
    if 'Campaign' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('Campaign').agg(
        Total_Calls=('Campaign', 'count'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Connectivity_Pct'] = (analysis['Connected'] / analysis['Total_Calls'] * 100).round(2)
    analysis['Conversion_Pct'] = (analysis['Sales'] / analysis['Total_Calls'] * 100).round(2)
    return analysis.sort_values('Total_Calls', ascending=False)

def analyze_hourly_trend(df):
    if 'Hour' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('Hour').agg(
        Total_Calls=('Hour', 'count'),
        Connected=('Is_Connected', 'sum'),
        Sales=('Is_Sale', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Connectivity_Pct'] = (analysis['Connected'] / analysis['Total_Calls'] * 100).round(2)
    analysis['Conversion_Pct'] = (analysis['Sales'] / analysis['Total_Calls'] * 100).round(2)
    return analysis.sort_values('Hour')

def analyze_disposition_type_summary(df):
    if 'DispositionType' not in df.columns:
        return pd.DataFrame()
    analysis = df.groupby('DispositionType').agg(
        Call_Count=('DispositionType', 'count'),
        Connected=('Is_Connected', 'sum'),
        Total_Talk_Sec=('Talk_Seconds', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Percentage'] = (analysis['Call_Count'] / len(df) * 100).round(2)
    return analysis.sort_values('Call_Count', ascending=False)

def get_duration_distribution(df):
    bucket_order = ["0 seconds", "< 1 min", "1-2 mins", "2-3 mins", "3-5 mins", "5-6 mins", "6-7 mins", "> 7 mins"]
    analysis = df.groupby('Duration_Bucket').agg(
        Call_Count=('Duration_Bucket', 'count'),
        Sales=('Is_Sale', 'sum'),
        Avg_Talk_Sec=('Talk_Seconds', 'mean')
    ).reset_index()
    analysis['Percentage'] = (analysis['Call_Count'] / len(df) * 100).round(2)
    analysis['sort_order'] = analysis['Duration_Bucket'].apply(lambda x: bucket_order.index(x) if x in bucket_order else 99)
    return analysis.sort_values('sort_order').drop('sort_order', axis=1)

def analyze_trunk_campaign(df):
    if 'Trunk' not in df.columns or 'Campaign' not in df.columns:
        return pd.DataFrame()
    pivot = pd.crosstab(df['Trunk'], df['Campaign'], margins=True, margins_name='Total')
    return pivot

# ==================== EXPORT FUNCTION ====================

def export_all_reports(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Header format
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#667eea', 'font_color': 'white',
            'border': 1, 'align': 'center', 'valign': 'vcenter'
        })

        metrics = get_overview_metrics(df)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Overview', index=False, startrow=1)

        for name, func in [
            ('Dispositions', analyze_dispositions),
            ('Disposition Types', analyze_disposition_type_summary),
            ('Connectivity Interval', analyze_connectivity_by_interval),
            ('Connectivity Trunk', analyze_connectivity_by_trunk),
            ('Sales by Duration', analyze_sales_by_duration),
            ('Sales by Agent', analyze_sales_by_agent),
            ('Campaign Performance', analyze_campaign_performance),
            ('Hourly Trend', analyze_hourly_trend),
            ('Duration Distribution', get_duration_distribution),
        ]:
            data = func(df)
            if not data.empty:
                data.to_excel(writer, sheet_name=name[:31], index=False, startrow=1)

        for name, func in [
            ('Agent Disposition', analyze_agent_disposition),
            ('Campaign Disposition', analyze_campaign_disposition),
            ('Agent Sales Duration', analyze_sales_by_agent_duration),
            ('Trunk Campaign', analyze_trunk_campaign),
        ]:
            data = func(df)
            if not data.empty:
                data.to_excel(writer, sheet_name=name[:31], startrow=1)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(2, 0)
            ws.set_column('A:Z', 15)

    output.seek(0)
    return output

# ==================== VISUALIZATION ====================

def create_chart_layout():
    return dict(
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

def plot_disposition_pie(df):
    disp_counts = df['Disposition'].value_counts().head(10)
    fig = px.pie(
        values=disp_counts.values,
        names=disp_counts.index,
        title='<b>Top 10 Dispositions</b>',
        hole=0.5,
        color_discrete_sequence=COLORS['gradient']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
    fig.update_layout(**create_chart_layout())
    return fig

def plot_connectivity_gauge(metrics):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=metrics['connectivity_rate'],
        title={'text': "<b>Connectivity Rate</b>", 'font': {'size': 18}},
        delta={'reference': 50, 'increasing': {'color': "#38ef7d"}, 'decreasing': {'color': "#f5576c"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(245, 87, 108, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(56, 239, 125, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#764ba2", 'width': 4},
                'thickness': 0.75,
                'value': metrics['connectivity_rate']
            }
        },
        number={'suffix': "%", 'font': {'size': 40, 'color': '#667eea'}}
    ))
    fig.update_layout(height=300, **create_chart_layout())
    return fig

def plot_hourly_trend(df):
    hourly = analyze_hourly_trend(df)
    if hourly.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=hourly['Hour'], y=hourly['Total_Calls'],
        name='Total Calls',
        marker=dict(color='rgba(102, 126, 234, 0.7)', line=dict(color='#667eea', width=1)),
        hovertemplate='Hour: %{x}<br>Calls: %{y}<extra></extra>'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hourly['Hour'], y=hourly['Connectivity_Pct'],
        name='Connectivity %',
        mode='lines+markers',
        line=dict(color='#38ef7d', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Connectivity: %{y:.1f}%<extra></extra>'
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=hourly['Hour'], y=hourly['Conversion_Pct'],
        name='Conversion %',
        mode='lines+markers',
        line=dict(color='#f5576c', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='Conversion: %{y:.1f}%<extra></extra>'
    ), secondary_y=True)

    fig.update_layout(
        title='<b>Hourly Performance Trend</b>',
        xaxis_title='Hour of Day',
        hovermode='x unified',
        **create_chart_layout()
    )
    fig.update_yaxes(title_text="<b>Call Volume</b>", secondary_y=False, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(title_text="<b>Percentage</b>", secondary_y=True)
    return fig

def plot_sales_by_duration(df):
    sales_dur = analyze_sales_by_duration(df)
    if sales_dur.empty:
        fig = go.Figure()
        fig.add_annotation(text="No sales data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#666"))
        fig.update_layout(**create_chart_layout())
        return fig

    fig = px.bar(
        sales_dur, x='Duration_Bucket', y='Sale_Count',
        title='<b>Sales Distribution by Call Duration</b>',
        color='Sale_Count',
        color_continuous_scale=['#c2e9fb', '#667eea', '#764ba2'],
        text='Sale_Count'
    )
    fig.update_traces(textposition='outside', textfont_size=12)
    fig.update_layout(**create_chart_layout(), showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return fig

def plot_agent_performance(df):
    agent_perf = analyze_sales_by_agent(df)
    if agent_perf.empty:
        return go.Figure()

    top_agents = agent_perf.head(12)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_agents['Agent'], y=top_agents['Total_Calls'],
        name='Total Calls',
        marker=dict(color='rgba(102, 126, 234, 0.7)', line=dict(color='#667eea', width=1)),
        yaxis='y'
    ))

    fig.add_trace(go.Bar(
        x=top_agents['Agent'], y=top_agents['Sales'],
        name='Sales',
        marker=dict(color='rgba(56, 239, 125, 0.8)', line=dict(color='#11998e', width=1)),
        yaxis='y'
    ))

    fig.update_layout(
        title='<b>Top Agents Performance</b>',
        barmode='group',
        xaxis_tickangle=-45,
        **create_chart_layout()
    )
    return fig

def plot_campaign_comparison(df):
    camp_perf = analyze_campaign_performance(df)
    if camp_perf.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=camp_perf['Campaign'], y=camp_perf['Total_Calls'],
        name='Total Calls',
        marker_color='rgba(102, 126, 234, 0.7)'
    ))

    fig.add_trace(go.Bar(
        x=camp_perf['Campaign'], y=camp_perf['Connected'],
        name='Connected',
        marker_color='rgba(56, 239, 125, 0.7)'
    ))

    fig.add_trace(go.Bar(
        x=camp_perf['Campaign'], y=camp_perf['Sales'],
        name='Sales',
        marker_color='rgba(245, 87, 108, 0.7)'
    ))

    fig.update_layout(
        title='<b>Campaign Performance Comparison</b>',
        barmode='group',
        xaxis_tickangle=-45,
        **create_chart_layout()
    )
    return fig

def plot_trunk_performance(df):
    trunk_perf = analyze_connectivity_by_trunk(df)
    if trunk_perf.empty:
        return go.Figure()

    fig = px.bar(
        trunk_perf, x='Trunk', y='Connectivity_Pct',
        title='<b>Trunk Connectivity Performance</b>',
        color='Connectivity_Pct',
        color_continuous_scale=['#f5576c', '#ffc107', '#38ef7d'],
        text='Connectivity_Pct'
    )
    fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig.update_layout(**create_chart_layout(), showlegend=False)
    fig.update_xaxes(tickangle=-45)
    return fig

def plot_disposition_type_pie(df):
    if 'DispositionType' not in df.columns:
        return go.Figure()

    disp_type = df['DispositionType'].value_counts().head(8)
    fig = px.pie(
        values=disp_type.values,
        names=disp_type.index,
        title='<b>Disposition Types</b>',
        hole=0.5,
        color_discrete_sequence=COLORS['gradient']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10)
    fig.update_layout(**create_chart_layout())
    return fig

def plot_funnel(metrics):
    fig = go.Figure(go.Funnel(
        y=['Total Calls', 'Connected', 'Sales'],
        x=[metrics['total_calls'], metrics['connected_calls'], metrics['total_sales']],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=['#667eea', '#38ef7d', '#f5576c']),
        connector=dict(line=dict(color="rgba(0,0,0,0.1)", width=2))
    ))
    fig.update_layout(title='<b>Conversion Funnel</b>', **create_chart_layout())
    return fig

# ==================== RENDER FUNCTIONS ====================

def render_metric_card(value, label, color_class=""):
    return f"""
    <div class="metric-card {color_class}">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>📊 CDR Data Analyser Pro</h1>
        <p>Advanced Call Detail Record Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

def main():
    render_header()

    with st.sidebar:
        st.markdown("### 📁 Upload Data")
        uploaded_file = st.file_uploader("Choose Excel or CSV file", type=['xlsx', 'xls', 'csv'])

        if uploaded_file:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ File Loaded</strong><br>
                📄 {uploaded_file.name}<br>
                📦 {uploaded_file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">
        Professional CDR analysis tool with:
        <ul>
            <li>📊 15+ Analysis Reports</li>
            <li>📈 Interactive Charts</li>
            <li>📥 Excel Export</li>
            <li>🔍 Deep Drill-downs</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <h2 class="gradient-text">Welcome to CDR Data Analyser Pro</h2>
            <p style="color: #666; font-size: 1.1rem; margin: 1.5rem 0;">
                Upload your CDR file to unlock powerful analytics and insights
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem;">📊</div>
                    <div style="font-weight: 600; color: #667eea;">Dispositions</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem;">📈</div>
                    <div style="font-weight: 600; color: #764ba2;">Connectivity</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem;">💰</div>
                    <div style="font-weight: 600; color: #38ef7d;">Sales</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem;">👥</div>
                    <div style="font-weight: 600; color: #f5576c;">Agents</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: #333;">📋 Supported Formats</h3>
            <p>Excel (.xlsx, .xls) • CSV (.csv)</p>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("🔄 Analyzing your data..."):
        df, error = load_and_process_data(uploaded_file, uploaded_file.name)

    if error:
        st.error(f"❌ {error}")
        return

    metrics = get_overview_metrics(df)

    # Tabs with icons
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "📋 Dispositions", "📡 Connectivity", "💰 Sales",
        "👥 Agents", "🎯 Campaigns", "📈 Trends", "📥 Export"
    ])

    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)

        # Metric cards row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_metric_card(f"{metrics['total_calls']:,}", "Total Calls", "blue"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card(f"{metrics['connected_calls']:,}", "Connected", "green"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card(f"{metrics['connectivity_rate']}%", "Connectivity", "purple"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_metric_card(f"{metrics['total_sales']:,}", "Total Sales", "orange"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric cards row 2
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown(render_metric_card(f"{metrics['conversion_rate']}%", "Conversion Rate", ""), unsafe_allow_html=True)
        with col6:
            st.markdown(render_metric_card(f"{metrics['avg_talk_time']:.0f}s", "Avg Talk Time", ""), unsafe_allow_html=True)
        with col7:
            st.markdown(render_metric_card(f"{metrics['total_talk_time_hrs']:.1f}h", "Total Talk Time", ""), unsafe_allow_html=True)
        with col8:
            st.markdown(render_metric_card(f"{metrics['not_connected']:,}", "Not Connected", ""), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_disposition_pie(df), use_container_width=True, key="ov_pie")
        with col2:
            st.plotly_chart(plot_connectivity_gauge(metrics), use_container_width=True, key="ov_gauge")

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_funnel(metrics), use_container_width=True, key="ov_funnel")
        with col4:
            st.plotly_chart(plot_disposition_type_pie(df), use_container_width=True, key="ov_dtype")

    # ========== TAB 2: DISPOSITIONS ==========
    with tab2:
        st.markdown('<h2 class="section-header">Disposition Analysis</h2>', unsafe_allow_html=True)

        disp = analyze_dispositions(df)
        if not disp.empty:
            st.dataframe(disp, use_container_width=True, height=400)

        st.markdown('<h3 class="section-header">By Disposition Type</h3>', unsafe_allow_html=True)
        disp_type = analyze_disposition_type_summary(df)
        if not disp_type.empty:
            st.dataframe(disp_type, use_container_width=True)

        st.markdown('<h3 class="section-header">Agent × Disposition</h3>', unsafe_allow_html=True)
        agent_disp = analyze_agent_disposition(df)
        if not agent_disp.empty:
            st.dataframe(agent_disp, use_container_width=True)

        st.markdown('<h3 class="section-header">Campaign × Disposition</h3>', unsafe_allow_html=True)
        camp_disp = analyze_campaign_disposition(df)
        if not camp_disp.empty:
            st.dataframe(camp_disp, use_container_width=True)

    # ========== TAB 3: CONNECTIVITY ==========
    with tab3:
        st.markdown('<h2 class="section-header">Connectivity Analysis</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(render_metric_card(f"{metrics['total_calls']:,}", "Total Calls", "blue"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card(f"{metrics['connected_calls']:,}", "Connected", "green"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card(f"{metrics['connectivity_rate']}%", "Connectivity Rate", "purple"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Hourly Connectivity</h3>', unsafe_allow_html=True)

        conn_int = analyze_connectivity_by_interval(df)
        if not conn_int.empty:
            st.dataframe(conn_int, use_container_width=True)

        st.markdown('<h3 class="section-header">Trunk Performance</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        conn_trunk = analyze_connectivity_by_trunk(df)
        if not conn_trunk.empty:
            with col1:
                st.dataframe(conn_trunk, use_container_width=True)
            with col2:
                st.plotly_chart(plot_trunk_performance(df), use_container_width=True, key="trunk_perf")

    # ========== TAB 4: SALES ==========
    with tab4:
        st.markdown('<h2 class="section-header">Sales Analysis</h2>', unsafe_allow_html=True)

        if metrics['total_sales'] == 0:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No Sales Detected</strong><br>
                Sales are identified by disposition keywords: SALE, SALE_DONE, CONVERTED, etc.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card(f"{metrics['total_sales']:,}", "Total Sales", "green"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">Sales by Duration</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            sales_dur = analyze_sales_by_duration(df)
            if not sales_dur.empty:
                with col1:
                    st.dataframe(sales_dur, use_container_width=True)
                with col2:
                    st.plotly_chart(plot_sales_by_duration(df), use_container_width=True, key="sales_dur_chart")

            st.markdown('<h3 class="section-header">Sales by Agent</h3>', unsafe_allow_html=True)
            sales_agent = analyze_sales_by_agent(df)
            if not sales_agent.empty:
                st.dataframe(sales_agent, use_container_width=True)

            st.markdown('<h3 class="section-header">Agent Sales × Duration</h3>', unsafe_allow_html=True)
            sales_agent_dur = analyze_sales_by_agent_duration(df)
            if not sales_agent_dur.empty:
                st.dataframe(sales_agent_dur, use_container_width=True)

    # ========== TAB 5: AGENTS ==========
    with tab5:
        st.markdown('<h2 class="section-header">Agent Performance</h2>', unsafe_allow_html=True)

        agent_perf = analyze_sales_by_agent(df)
        if not agent_perf.empty:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.dataframe(agent_perf, use_container_width=True, height=500)
            with col2:
                st.plotly_chart(plot_agent_performance(df), use_container_width=True, key="agent_chart")

    # ========== TAB 6: CAMPAIGNS ==========
    with tab6:
        st.markdown('<h2 class="section-header">Campaign Performance</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        camp_perf = analyze_campaign_performance(df)
        if not camp_perf.empty:
            with col1:
                st.dataframe(camp_perf, use_container_width=True)
            with col2:
                st.plotly_chart(plot_campaign_comparison(df), use_container_width=True, key="camp_chart")

        st.markdown('<h3 class="section-header">Trunk × Campaign</h3>', unsafe_allow_html=True)
        trunk_camp = analyze_trunk_campaign(df)
        if not trunk_camp.empty:
            st.dataframe(trunk_camp, use_container_width=True)

    # ========== TAB 7: TRENDS ==========
    with tab7:
        st.markdown('<h2 class="section-header">Trend Analysis</h2>', unsafe_allow_html=True)

        st.plotly_chart(plot_hourly_trend(df), use_container_width=True, key="hourly_main")

        col1, col2 = st.columns([1, 1])
        hourly = analyze_hourly_trend(df)
        if not hourly.empty:
            with col1:
                st.markdown('<h3 class="section-header">Hourly Data</h3>', unsafe_allow_html=True)
                st.dataframe(hourly, use_container_width=True)
            with col2:
                st.markdown('<h3 class="section-header">Duration Distribution</h3>', unsafe_allow_html=True)
                dur_dist = get_duration_distribution(df)
                if not dur_dist.empty:
                    st.dataframe(dur_dist, use_container_width=True)

    # ========== TAB 8: EXPORT ==========
    with tab8:
        st.markdown('<h2 class="section-header">Export Reports</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📥 Download Comprehensive Excel Report</h3>
            <p style="color: #666;">Your report will include 15+ sheets with detailed analysis:</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0;">
                <div>✅ Overview Metrics</div>
                <div>✅ Disposition Analysis</div>
                <div>✅ Disposition Types</div>
                <div>✅ Connectivity by Interval</div>
                <div>✅ Connectivity by Trunk</div>
                <div>✅ Sales by Duration</div>
                <div>✅ Sales by Agent</div>
                <div>✅ Campaign Performance</div>
                <div>✅ Hourly Trends</div>
                <div>✅ Agent × Disposition</div>
                <div>✅ Campaign × Disposition</div>
                <div>✅ Duration Distribution</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("📊 Generating your comprehensive report..."):
                    excel_data = export_all_reports(df)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    st.download_button(
                        "📥 Download Excel Report",
                        data=excel_data,
                        file_name=f"CDR_Analysis_Pro_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Report Generated Successfully!</strong><br>
                        Click the download button above to save your report.
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
