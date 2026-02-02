"""
CDR Data Analyser - Main Streamlit Application
A comprehensive tool for analyzing Call Detail Records
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import CDRDataLoader
from analyzer import CDRAnalyzer
from report_generator import CDRReportGenerator, QuickReportGenerator
from visualizations import CDRVisualizer

# Page configuration
st.set_page_config(
    page_title="CDR Data Analyser",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4472C4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #4472C4;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None


def render_sidebar():
    """Render sidebar with file upload and settings"""
    with st.sidebar:
        st.markdown("### 📁 Data Upload")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CDR File",
            type=['csv', 'xlsx', 'xls', 'xlsm'],
            help="Upload your CDR data in CSV or Excel format"
        )

        if uploaded_file is not None:
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "Size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)

            # For Excel files, allow sheet selection
            sheet_name = None
            sheet_names = []
            if uploaded_file.name.endswith(('.xlsx', '.xls', '.xlsm')):
                try:
                    uploaded_file.seek(0)
                    xl = pd.ExcelFile(uploaded_file)
                    sheet_names = xl.sheet_names
                    uploaded_file.seek(0)  # Reset file pointer

                    if len(sheet_names) > 1:
                        sheet_name = st.selectbox(
                            "Select Sheet",
                            options=sheet_names,
                            index=0
                        )
                    elif len(sheet_names) == 1:
                        sheet_name = sheet_names[0]
                        st.info(f"Sheet: {sheet_name}")
                except Exception as e:
                    st.warning(f"Could not read sheet names: {e}")

            # Load button
            if st.button("Load & Analyze", type="primary", use_container_width=True):
                uploaded_file.seek(0)  # Reset file pointer before loading
                with st.spinner("Loading and analyzing data..."):
                    load_data(uploaded_file, uploaded_file.name, sheet_name)

        st.markdown("---")

        # Column mapping section (shown after data is loaded)
        if st.session_state.data_loaded and st.session_state.loader:
            st.markdown("### Column Mapping")

            with st.expander("View/Edit Column Mappings"):
                mapping = st.session_state.loader.get_column_mapping()
                available_cols = st.session_state.loader.get_raw_data().columns.tolist()

                new_mapping = {}
                for field in ['agent', 'disposition', 'duration', 'trunk', 'campaign', 'datetime']:
                    current = mapping.get(field, '')
                    options = [''] + available_cols
                    idx = options.index(current) if current in options else 0

                    new_mapping[field] = st.selectbox(
                        field.title(),
                        options=options,
                        index=idx,
                        key=f"mapping_{field}"
                    )

                if st.button("Apply Mapping", use_container_width=True):
                    # Filter out empty mappings
                    filtered_mapping = {k: v for k, v in new_mapping.items() if v}
                    st.session_state.loader.update_column_mapping(filtered_mapping)
                    # Reinitialize analyzer
                    data = st.session_state.loader.get_data()
                    st.session_state.analyzer = CDRAnalyzer(data)
                    st.session_state.report_generator = CDRReportGenerator(st.session_state.analyzer)
                    st.session_state.visualizer = CDRVisualizer(st.session_state.analyzer)
                    st.success("Mapping updated!")
                    st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **CDR Data Analyser** v1.0

        Analyze your Call Detail Records with:
        - Disposition Analysis
        - Connectivity Metrics
        - Sales Performance
        - Agent Performance
        - Campaign Analysis
        - Duration Analysis
        """)


def load_data(uploaded_file, file_name: str, sheet_name=None):
    """Load and process the uploaded data"""
    loader = CDRDataLoader()
    success, message = loader.load_from_upload(uploaded_file, file_name, sheet_name)

    if success:
        st.session_state.loader = loader
        st.session_state.data_loaded = True

        # Initialize analyzer
        data = loader.get_data()
        st.session_state.analyzer = CDRAnalyzer(data)
        st.session_state.report_generator = CDRReportGenerator(st.session_state.analyzer)
        st.session_state.visualizer = CDRVisualizer(st.session_state.analyzer)

        st.success(message)
        st.rerun()
    else:
        st.error(message)


def render_overview_tab():
    """Render overview/dashboard tab"""
    st.markdown("## Dashboard Overview")

    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer

    # KPI Cards
    kpis = visualizer.plot_kpi_cards()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=kpis['total_calls']['label'],
            value=f"{kpis['total_calls']['value']:,}"
        )

    with col2:
        st.metric(
            label=kpis['connected_calls']['label'],
            value=f"{kpis['connected_calls']['value']:,}"
        )

    with col3:
        st.metric(
            label=kpis['connectivity_rate']['label'],
            value=kpis['connectivity_rate']['value']
        )

    with col4:
        st.metric(
            label=kpis['avg_talk_time']['label'],
            value=kpis['avg_talk_time']['value']
        )

    if 'total_sales' in kpis:
        col5, col6 = st.columns(2)
        with col5:
            st.metric(
                label=kpis['total_sales']['label'],
                value=f"{kpis['total_sales']['value']:,}"
            )
        with col6:
            st.metric(
                label=kpis['conversion_rate']['label'],
                value=kpis['conversion_rate']['value']
            )

    st.markdown("---")

    # Key Insights
    st.markdown("### Key Insights")
    insights = analyzer.get_insights()

    if insights:
        for insight in insights:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No insights available. Ensure your data has disposition, agent, and trunk information.")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(visualizer.plot_disposition_distribution(), use_container_width=True, key="overview_disp_dist")

    with col2:
        st.plotly_chart(visualizer.plot_connectivity_gauge(), use_container_width=True, key="overview_conn_gauge")

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(visualizer.plot_sales_funnel(), use_container_width=True, key="overview_sales_funnel")

    with col4:
        st.plotly_chart(visualizer.plot_duration_distribution(), use_container_width=True, key="overview_duration_dist")


def render_disposition_tab():
    """Render disposition analysis tab"""
    st.markdown("## Disposition Analysis")

    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer

    # Overall disposition analysis
    st.markdown("### Overall Disposition Distribution")
    disp_analysis = analyzer.analyze_dispositions()

    if not disp_analysis.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(disp_analysis, use_container_width=True, height=400)

        with col2:
            st.plotly_chart(visualizer.plot_disposition_distribution(), use_container_width=True, key="disp_tab_dist")

    st.markdown("---")

    # Disposition by agent
    st.markdown("### Disposition by Agent")
    disp_by_agent = analyzer.disposition_by_agent()

    if not disp_by_agent.empty:
        st.dataframe(disp_by_agent, use_container_width=True)

    st.markdown("---")

    # Disposition by campaign
    st.markdown("### Disposition by Campaign")
    disp_by_campaign = analyzer.disposition_by_campaign()

    if not disp_by_campaign.empty:
        st.dataframe(disp_by_campaign, use_container_width=True)

    st.markdown("---")

    # Disposition by duration
    st.markdown("### Disposition by Duration")
    disp_by_duration = analyzer.disposition_by_duration()

    if not disp_by_duration.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(disp_by_duration, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_disposition_by_duration_heatmap(), use_container_width=True, key="disp_duration_heatmap")


def render_connectivity_tab():
    """Render connectivity analysis tab"""
    st.markdown("## Connectivity Analysis")

    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer

    # Overall connectivity
    st.markdown("### Overall Connectivity Metrics")
    connectivity = analyzer.analyze_connectivity()

    if connectivity:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Calls", f"{connectivity['total_calls']:,}")
        with col2:
            st.metric("Connected", f"{connectivity['connected_calls']:,}")
        with col3:
            st.metric("Not Connected", f"{connectivity['not_connected_calls']:,}")
        with col4:
            st.metric("Connectivity Rate", f"{connectivity['connectivity_rate']}%")

    st.markdown("---")

    # Connectivity by interval
    st.markdown("### Connectivity by Time Interval")
    conn_interval = analyzer.connectivity_by_interval()

    if not conn_interval.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(conn_interval, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_connectivity_by_interval(), use_container_width=True, key="conn_interval_chart")

    st.markdown("---")

    # Connectivity by trunk
    st.markdown("### Connectivity by Trunk")
    conn_trunk = analyzer.connectivity_by_trunk()

    if not conn_trunk.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(conn_trunk, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_connectivity_by_trunk(), use_container_width=True, key="conn_trunk_chart")

    st.markdown("---")

    # Connectivity by agent
    st.markdown("### Connectivity by Agent")
    conn_agent = analyzer.connectivity_by_agent()

    if not conn_agent.empty:
        st.dataframe(conn_agent, use_container_width=True)

    st.markdown("---")

    # Connectivity by campaign
    st.markdown("### Connectivity by Campaign")
    conn_campaign = analyzer.connectivity_by_campaign()

    if not conn_campaign.empty:
        st.dataframe(conn_campaign, use_container_width=True)


def render_sales_tab():
    """Render sales analysis tab"""
    st.markdown("## Sales Analysis")

    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer

    # Check if sale data is available
    if 'is_sale' not in analyzer.data.columns or analyzer.data['is_sale'].sum() == 0:
        st.warning("""
        No sales data detected in your CDR.

        The analyzer looks for disposition keywords like:
        'sale', 'sold', 'converted', 'successful', 'confirmed', 'booked',
        'appointment', 'qualified', 'interested', 'payment', 'paid', 'closed', 'won'

        Please ensure your disposition column contains these keywords for sales records.
        """)
        return

    # Sales by duration
    st.markdown("### Sales by Duration Bucket")
    st.markdown("""
    This report shows sale dispositions categorized by call duration:
    - **0 seconds**: Potentially data quality issues
    - **< 1 min**: Very short sales calls
    - **1-7+ mins**: Various duration ranges
    """)

    sales_duration = analyzer.analyze_sales_by_duration()

    if not sales_duration.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(sales_duration, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_sales_by_duration(), use_container_width=True, key="sales_duration_chart")

    st.markdown("---")

    # Sales by agent
    st.markdown("### Sales by Agent")
    sales_agent = analyzer.sales_by_agent()

    if not sales_agent.empty:
        st.dataframe(sales_agent, use_container_width=True)

    st.markdown("---")

    # Sales by campaign
    st.markdown("### Sales by Campaign")
    sales_campaign = analyzer.sales_by_campaign()

    if not sales_campaign.empty:
        st.dataframe(sales_campaign, use_container_width=True)

    st.markdown("---")

    # Sales by interval
    st.markdown("### Sales by Time Interval")
    sales_interval = analyzer.sales_by_interval()

    if not sales_interval.empty:
        st.dataframe(sales_interval, use_container_width=True)

    st.markdown("---")

    # Sales by trunk
    st.markdown("### Sales by Trunk")
    sales_trunk = analyzer.sales_by_trunk()

    if not sales_trunk.empty:
        st.dataframe(sales_trunk, use_container_width=True)


def render_performance_tab():
    """Render performance analysis tab"""
    st.markdown("## Performance Analysis")

    analyzer = st.session_state.analyzer
    visualizer = st.session_state.visualizer

    # Agent performance
    st.markdown("### Agent Performance Summary")
    agent_perf = analyzer.agent_performance_summary()

    if not agent_perf.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(agent_perf, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_agent_performance(), use_container_width=True, key="agent_perf_chart")

    st.markdown("---")

    # Campaign performance
    st.markdown("### Campaign Performance Summary")
    campaign_perf = analyzer.campaign_performance_summary()

    if not campaign_perf.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(campaign_perf, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_campaign_performance(), use_container_width=True, key="campaign_perf_chart")

    st.markdown("---")

    # Duration analysis
    st.markdown("### Duration Distribution")
    duration_dist = analyzer.duration_distribution()

    if not duration_dist.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(duration_dist, use_container_width=True)

        with col2:
            st.plotly_chart(visualizer.plot_duration_distribution(), use_container_width=True, key="perf_duration_chart")

    st.markdown("---")

    # Duration by agent
    st.markdown("### Duration Distribution by Agent")
    duration_agent = analyzer.duration_by_agent()

    if not duration_agent.empty:
        st.dataframe(duration_agent, use_container_width=True)

    st.markdown("---")

    # Duration by campaign
    st.markdown("### Duration Distribution by Campaign")
    duration_campaign = analyzer.duration_by_campaign()

    if not duration_campaign.empty:
        st.dataframe(duration_campaign, use_container_width=True)


def render_reports_tab():
    """Render reports/export tab"""
    st.markdown("## Generate Reports")

    report_generator = st.session_state.report_generator

    # Generate all reports
    reports = report_generator.generate_all_reports()

    st.markdown("### Available Reports")
    st.write(f"**{len(reports)}** reports generated")

    # Report list with checkboxes
    st.markdown("---")

    # Export all reports
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Export All Reports")

        if st.button("Generate Excel Report", type="primary", use_container_width=True):
            excel_data = report_generator.export_to_excel_bytes()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CDR_Analysis_Report_{timestamp}.xlsx"

            st.download_button(
                label="Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with col2:
        st.markdown("#### Export Single Report")

        report_name = st.selectbox(
            "Select Report",
            options=list(reports.keys())
        )

        if st.button("Generate Single Report", use_container_width=True):
            single_report = report_generator.export_single_report_bytes(report_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = report_name.replace(' ', '_')[:20]
            filename = f"{safe_name}_{timestamp}.xlsx"

            st.download_button(
                label=f"Download {report_name}",
                data=single_report,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    st.markdown("---")

    # Preview reports
    st.markdown("### Report Preview")

    preview_report = st.selectbox(
        "Select report to preview",
        options=list(reports.keys()),
        key="preview_select"
    )

    if preview_report:
        df = reports[preview_report]

        # Reset index if needed for display
        if isinstance(df.index, pd.Index) and df.index.name is not None:
            df = df.reset_index()

        st.dataframe(df, use_container_width=True)


def render_data_tab():
    """Render raw data view tab"""
    st.markdown("## Data View")

    loader = st.session_state.loader
    data = loader.get_data()

    # Data summary
    st.markdown("### Data Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(data):,}")

    with col2:
        st.metric("Columns", len(data.columns))

    with col3:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")

    st.markdown("---")

    # Column info
    st.markdown("### Column Information")

    quality_report = QuickReportGenerator.generate_data_quality_report(data)
    st.dataframe(quality_report, use_container_width=True)

    st.markdown("---")

    # Data preview with filters
    st.markdown("### Data Preview")

    # Filter options
    with st.expander("Filter Options"):
        filter_cols = st.multiselect(
            "Select columns to display",
            options=data.columns.tolist(),
            default=data.columns.tolist()[:10]
        )

        max_rows = st.slider("Max rows to display", 10, 1000, 100)

    if filter_cols:
        st.dataframe(data[filter_cols].head(max_rows), use_container_width=True)


def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1 style='color: #4472C4;'>Welcome to CDR Data Analyser</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Upload your Call Detail Records to get started with comprehensive analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Supported Formats
        - CSV (.csv)
        - Excel (.xlsx, .xls, .xlsm)
        """)

    with col2:
        st.markdown("""
        ### Analysis Features
        - Disposition Analysis
        - Connectivity Metrics
        - Sales Performance
        - Agent/Campaign Analysis
        """)

    with col3:
        st.markdown("""
        ### Export Options
        - Excel Reports
        - Individual Reports
        - Interactive Charts
        """)

    st.markdown("---")

    st.markdown("""
    ### Expected Columns
    The analyzer will auto-detect columns, but for best results include:

    | Column Type | Examples |
    |-------------|----------|
    | Agent/User | agent, agent_name, user, operator |
    | Disposition | disposition, status, call_status, result |
    | Duration | duration, talk_time, call_duration, billsec |
    | Trunk | trunk, gateway, channel, did, carrier |
    | Campaign | campaign, project, program, list |
    | DateTime | datetime, call_date, timestamp, start_time |

    You can manually map columns after uploading if auto-detection doesn't work perfectly.
    """)


def main():
    """Main application entry point"""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">CDR Data Analyser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Call Detail Record Analysis Tool</p>', unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Main content
    if not st.session_state.data_loaded:
        render_welcome_screen()
    else:
        # Create tabs
        tabs = st.tabs([
            "Overview",
            "Disposition",
            "Connectivity",
            "Sales",
            "Performance",
            "Reports",
            "Data"
        ])

        with tabs[0]:
            render_overview_tab()

        with tabs[1]:
            render_disposition_tab()

        with tabs[2]:
            render_connectivity_tab()

        with tabs[3]:
            render_sales_tab()

        with tabs[4]:
            render_performance_tab()

        with tabs[5]:
            render_reports_tab()

        with tabs[6]:
            render_data_tab()


if __name__ == "__main__":
    main()
