"""
CDR Report Generator Module
Generates comprehensive Excel reports from CDR analysis
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from io import BytesIO
import os


class CDRReportGenerator:
    """Generates Excel reports from CDR analysis"""

    def __init__(self, analyzer):
        """
        Initialize report generator

        Args:
            analyzer: CDRAnalyzer instance with analyzed data
        """
        self.analyzer = analyzer
        self.reports = {}

    def generate_all_reports(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all available reports

        Returns:
            Dictionary of report name to DataFrame
        """
        self.reports = {}

        # Executive Summary
        summary = self.analyzer.generate_executive_summary()
        self.reports['Executive Summary'] = self._summary_to_dataframe(summary)

        # Disposition Reports
        disp_analysis = self.analyzer.analyze_dispositions()
        if not disp_analysis.empty:
            self.reports['Disposition Analysis'] = disp_analysis

        disp_by_agent = self.analyzer.disposition_by_agent()
        if not disp_by_agent.empty:
            self.reports['Disposition by Agent'] = disp_by_agent

        disp_by_campaign = self.analyzer.disposition_by_campaign()
        if not disp_by_campaign.empty:
            self.reports['Disposition by Campaign'] = disp_by_campaign

        disp_by_duration = self.analyzer.disposition_by_duration()
        if not disp_by_duration.empty:
            self.reports['Disposition by Duration'] = disp_by_duration

        # Connectivity Reports
        conn_overall = self.analyzer.analyze_connectivity()
        if conn_overall:
            self.reports['Connectivity Overview'] = pd.DataFrame([conn_overall])

        conn_by_interval = self.analyzer.connectivity_by_interval()
        if not conn_by_interval.empty:
            self.reports['Connectivity by Interval'] = conn_by_interval

        conn_by_trunk = self.analyzer.connectivity_by_trunk()
        if not conn_by_trunk.empty:
            self.reports['Connectivity by Trunk'] = conn_by_trunk

        conn_by_agent = self.analyzer.connectivity_by_agent()
        if not conn_by_agent.empty:
            self.reports['Connectivity by Agent'] = conn_by_agent

        conn_by_campaign = self.analyzer.connectivity_by_campaign()
        if not conn_by_campaign.empty:
            self.reports['Connectivity by Campaign'] = conn_by_campaign

        # Sales Reports
        sales_by_duration = self.analyzer.analyze_sales_by_duration()
        if not sales_by_duration.empty:
            self.reports['Sales by Duration'] = sales_by_duration

        sales_by_agent = self.analyzer.sales_by_agent()
        if not sales_by_agent.empty:
            self.reports['Sales by Agent'] = sales_by_agent

        sales_by_campaign = self.analyzer.sales_by_campaign()
        if not sales_by_campaign.empty:
            self.reports['Sales by Campaign'] = sales_by_campaign

        sales_by_interval = self.analyzer.sales_by_interval()
        if not sales_by_interval.empty:
            self.reports['Sales by Interval'] = sales_by_interval

        sales_by_trunk = self.analyzer.sales_by_trunk()
        if not sales_by_trunk.empty:
            self.reports['Sales by Trunk'] = sales_by_trunk

        # Performance Reports
        agent_perf = self.analyzer.agent_performance_summary()
        if not agent_perf.empty:
            self.reports['Agent Performance'] = agent_perf

        campaign_perf = self.analyzer.campaign_performance_summary()
        if not campaign_perf.empty:
            self.reports['Campaign Performance'] = campaign_perf

        # Duration Reports
        duration_dist = self.analyzer.duration_distribution()
        if not duration_dist.empty:
            self.reports['Duration Distribution'] = duration_dist

        duration_by_agent = self.analyzer.duration_by_agent()
        if not duration_by_agent.empty:
            self.reports['Duration by Agent'] = duration_by_agent

        duration_by_campaign = self.analyzer.duration_by_campaign()
        if not duration_by_campaign.empty:
            self.reports['Duration by Campaign'] = duration_by_campaign

        # Insights
        insights = self.analyzer.get_insights()
        if insights:
            self.reports['Key Insights'] = pd.DataFrame({
                'Insight': insights
            })

        return self.reports

    def _summary_to_dataframe(self, summary: Dict) -> pd.DataFrame:
        """Convert summary dictionary to DataFrame"""
        rows = []

        for section, data in summary.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    rows.append({
                        'Section': section.replace('_', ' ').title(),
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value
                    })

        return pd.DataFrame(rows)

    def export_to_excel(self, output_path: str) -> str:
        """
        Export all reports to Excel file

        Args:
            output_path: Path for output Excel file

        Returns:
            Path to created file
        """
        if not self.reports:
            self.generate_all_reports()

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            number_format = workbook.add_format({
                'num_format': '#,##0',
                'border': 1
            })

            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'border': 1
            })

            decimal_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1
            })

            for sheet_name, df in self.reports.items():
                # Truncate sheet name if too long
                safe_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name

                # Reset index if it's a crosstab
                if isinstance(df.index, pd.Index) and df.index.name is not None:
                    df = df.reset_index()

                df.to_excel(writer, sheet_name=safe_name, index=False, startrow=1)

                worksheet = writer.sheets[safe_name]

                # Write headers with formatting
                for col_num, column in enumerate(df.columns):
                    worksheet.write(0, col_num, column, header_format)

                # Auto-adjust column widths
                for col_num, column in enumerate(df.columns):
                    max_length = max(
                        df[column].astype(str).apply(len).max(),
                        len(str(column))
                    )
                    worksheet.set_column(col_num, col_num, min(max_length + 2, 50))

                # Add autofilter
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

                # Freeze first row
                worksheet.freeze_panes(1, 0)

        return output_path

    def export_to_excel_bytes(self) -> BytesIO:
        """
        Export all reports to Excel file in memory

        Returns:
            BytesIO object containing Excel file
        """
        if not self.reports:
            self.generate_all_reports()

        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            for sheet_name, df in self.reports.items():
                # Truncate sheet name if too long
                safe_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name

                # Reset index if it's a crosstab
                if isinstance(df.index, pd.Index) and df.index.name is not None:
                    df = df.reset_index()

                df.to_excel(writer, sheet_name=safe_name, index=False, startrow=1)

                worksheet = writer.sheets[safe_name]

                # Write headers with formatting
                for col_num, column in enumerate(df.columns):
                    worksheet.write(0, col_num, column, header_format)

                # Auto-adjust column widths
                for col_num, column in enumerate(df.columns):
                    max_length = max(
                        df[column].astype(str).apply(len).max() if len(df) > 0 else 0,
                        len(str(column))
                    )
                    worksheet.set_column(col_num, col_num, min(max_length + 2, 50))

                # Add autofilter
                if len(df) > 0:
                    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

                # Freeze first row
                worksheet.freeze_panes(1, 0)

        output.seek(0)
        return output

    def get_report_names(self) -> List[str]:
        """Get list of available report names"""
        if not self.reports:
            self.generate_all_reports()
        return list(self.reports.keys())

    def get_report(self, name: str) -> Optional[pd.DataFrame]:
        """Get specific report by name"""
        if not self.reports:
            self.generate_all_reports()
        return self.reports.get(name)

    def export_single_report(self, report_name: str, output_path: str) -> str:
        """Export a single report to Excel"""
        if not self.reports:
            self.generate_all_reports()

        if report_name not in self.reports:
            raise ValueError(f"Report '{report_name}' not found")

        df = self.reports[report_name]

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=report_name[:31], index=False)

        return output_path

    def export_single_report_bytes(self, report_name: str) -> BytesIO:
        """Export a single report to Excel in memory"""
        if not self.reports:
            self.generate_all_reports()

        if report_name not in self.reports:
            raise ValueError(f"Report '{report_name}' not found")

        df = self.reports[report_name]
        output = BytesIO()

        # Reset index if needed
        if isinstance(df.index, pd.Index) and df.index.name is not None:
            df = df.reset_index()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            df.to_excel(writer, sheet_name=report_name[:31], index=False, startrow=1)

            worksheet = writer.sheets[report_name[:31]]

            # Write headers
            for col_num, column in enumerate(df.columns):
                worksheet.write(0, col_num, column, header_format)

            # Auto-adjust column widths
            for col_num, column in enumerate(df.columns):
                max_length = max(
                    df[column].astype(str).apply(len).max() if len(df) > 0 else 0,
                    len(str(column))
                )
                worksheet.set_column(col_num, col_num, min(max_length + 2, 50))

            if len(df) > 0:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

            worksheet.freeze_panes(1, 0)

        output.seek(0)
        return output


class QuickReportGenerator:
    """Generate quick summary reports without full analysis"""

    @staticmethod
    def generate_data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
        """Generate data quality report"""
        quality_data = []

        for column in data.columns:
            total = len(data)
            non_null = data[column].count()
            null_count = total - non_null
            unique_count = data[column].nunique()

            quality_data.append({
                'Column': column,
                'Total Records': total,
                'Non-Null': non_null,
                'Null Count': null_count,
                'Null %': round(null_count / total * 100, 2) if total > 0 else 0,
                'Unique Values': unique_count,
                'Data Type': str(data[column].dtype)
            })

        return pd.DataFrame(quality_data)

    @staticmethod
    def generate_quick_summary(data: pd.DataFrame) -> Dict:
        """Generate quick data summary"""
        return {
            'total_records': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
