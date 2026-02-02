# CDR Data Analyser

A comprehensive Call Detail Record (CDR) analysis tool with an interactive web interface.

## Features

### Data Input
- Support for CSV and Excel files (.xlsx, .xls, .xlsm)
- Auto-detection of common CDR column names
- Manual column mapping option
- Multi-sheet Excel support

### Analysis Capabilities

#### Disposition Analysis
- Overall disposition distribution
- Disposition by agent
- Disposition by campaign
- Disposition by call duration

#### Connectivity Analysis
- Overall connectivity metrics
- Connectivity by time interval (hourly)
- Connectivity by trunk/gateway
- Connectivity by agent
- Connectivity by campaign

#### Sales Analysis
- Sales by duration bucket:
  - 0 seconds
  - Less than 1 minute
  - 1-2 minutes
  - 2-3 minutes
  - 3-5 minutes
  - 5-6 minutes
  - 6-7 minutes
  - Greater than 7 minutes
- Sales by agent with conversion rates
- Sales by campaign
- Sales by time interval
- Sales by trunk

#### Performance Reports
- Agent performance summary
- Campaign performance summary
- Duration distribution analysis

### Dimensions Available
- Agent/User Name
- Time Intervals (hourly)
- Trunk/Gateway
- Campaign
- Duration Buckets
- Disposition

### Export Options
- Comprehensive Excel report with all analyses
- Individual report export
- Formatted Excel output with:
  - Color-coded headers
  - Auto-adjusted column widths
  - Filters enabled
  - Frozen header rows

## Installation

1. Ensure Python 3.8+ is installed

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

Or on Windows, simply double-click `run.bat`

## Usage

1. **Upload Data**: Use the sidebar to upload your CDR file (CSV or Excel)

2. **Column Mapping**: The tool auto-detects columns, but you can manually map them if needed

3. **Analyze**: Navigate through tabs to view different analyses:
   - Overview: Dashboard with KPIs and key insights
   - Disposition: Detailed disposition analysis
   - Connectivity: Connection rate analysis
   - Sales: Sales performance by various dimensions
   - Performance: Agent and campaign performance
   - Reports: Export reports to Excel
   - Data: View raw data and quality metrics

4. **Export**: Generate Excel reports from the Reports tab

## Expected Column Names

The analyzer auto-detects these column patterns:

| Field | Common Names |
|-------|--------------|
| Agent | agent, agent_name, user, username, operator, rep |
| Disposition | disposition, status, call_status, result, outcome |
| Duration | duration, talk_time, call_duration, billsec |
| Trunk | trunk, gateway, channel, did, carrier |
| Campaign | campaign, project, program, list |
| DateTime | datetime, call_date, timestamp, start_time |

## Sale Detection

Sales are identified by disposition keywords:
- sale, sold, converted, successful, confirmed
- booked, appointment, qualified, interested
- payment, paid, closed, won

## Project Structure

```
CDR Data Analyser/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── run.bat               # Windows launcher
├── README.md             # This file
└── src/
    ├── __init__.py
    ├── data_loader.py    # Data loading and parsing
    ├── analyzer.py       # Analysis engine
    ├── report_generator.py # Excel report generation
    └── visualizations.py  # Chart creation
```

## Requirements

- Python 3.8+
- pandas
- openpyxl
- xlsxwriter
- streamlit
- plotly
- numpy
