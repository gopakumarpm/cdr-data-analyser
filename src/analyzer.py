"""
CDR Analyzer Module
Comprehensive analysis engine for CDR data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


class CDRAnalyzer:
    """Comprehensive CDR analysis engine"""

    # Duration bucket order for sorting
    DURATION_BUCKET_ORDER = [
        "0 seconds", "< 1 min", "1-2 mins", "2-3 mins",
        "3-5 mins", "5-6 mins", "6-7 mins", "> 7 mins"
    ]

    # Connected disposition keywords
    CONNECTED_KEYWORDS = ['connected', 'answered', 'talk', 'speaking', 'conversation',
                          'sale', 'sold', 'callback', 'interested', 'qualified',
                          'hot', 'warm', 'follow', 'dnc', 'not interested']

    # Not connected disposition keywords
    NOT_CONNECTED_KEYWORDS = ['no answer', 'busy', 'failed', 'invalid', 'disconnect',
                              'voicemail', 'machine', 'fax', 'dead', 'wrong number',
                              'not reachable', 'switched off', 'network', 'congestion']

    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with CDR data

        Args:
            data: Processed CDR DataFrame
        """
        self.data = data.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for analysis"""
        # Ensure duration_seconds exists
        if 'duration_seconds' not in self.data.columns:
            if 'duration' in self.data.columns:
                self.data['duration_seconds'] = pd.to_numeric(
                    self.data['duration'], errors='coerce'
                ).fillna(0)
            else:
                self.data['duration_seconds'] = 0

        # Ensure duration bucket exists
        if 'duration_bucket' not in self.data.columns:
            self.data['duration_bucket'] = self.data['duration_seconds'].apply(
                self._get_duration_bucket
            )

        # Determine connectivity
        if 'disposition' in self.data.columns:
            self.data['is_connected'] = self.data.apply(
                lambda row: self._is_connected(row), axis=1
            )
        else:
            self.data['is_connected'] = self.data['duration_seconds'] > 0

    def _get_duration_bucket(self, seconds: float) -> str:
        """Categorize duration into buckets"""
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

    def _is_connected(self, row) -> bool:
        """Determine if call was connected"""
        # If duration > 0, likely connected
        if row.get('duration_seconds', 0) > 0:
            return True

        # Check disposition
        disposition = str(row.get('disposition', '')).lower()

        # Check for connected keywords
        if any(kw in disposition for kw in self.CONNECTED_KEYWORDS):
            return True

        # Check for not connected keywords
        if any(kw in disposition for kw in self.NOT_CONNECTED_KEYWORDS):
            return False

        return row.get('duration_seconds', 0) > 0

    # ==================== DISPOSITION ANALYSIS ====================

    def analyze_dispositions(self) -> pd.DataFrame:
        """
        Analyze call dispositions

        Returns:
            DataFrame with disposition analysis
        """
        if 'disposition' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('disposition').agg({
            'disposition': 'count',
            'duration_seconds': ['sum', 'mean', 'min', 'max']
        }).reset_index()

        analysis.columns = ['Disposition', 'Call Count', 'Total Duration (sec)',
                           'Avg Duration (sec)', 'Min Duration (sec)', 'Max Duration (sec)']

        # Calculate percentage
        total_calls = analysis['Call Count'].sum()
        analysis['Percentage'] = (analysis['Call Count'] / total_calls * 100).round(2)

        # Sort by call count
        analysis = analysis.sort_values('Call Count', ascending=False)

        return analysis

    def disposition_by_agent(self) -> pd.DataFrame:
        """Analyze dispositions by agent"""
        if 'disposition' not in self.data.columns or 'agent' not in self.data.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(
            self.data['agent'],
            self.data['disposition'],
            margins=True,
            margins_name='Total'
        )

        return pivot

    def disposition_by_campaign(self) -> pd.DataFrame:
        """Analyze dispositions by campaign"""
        if 'disposition' not in self.data.columns or 'campaign' not in self.data.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(
            self.data['campaign'],
            self.data['disposition'],
            margins=True,
            margins_name='Total'
        )

        return pivot

    def disposition_by_duration(self) -> pd.DataFrame:
        """
        Analyze dispositions by talk duration buckets

        Returns:
            DataFrame with disposition counts per duration bucket
        """
        if 'disposition' not in self.data.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(
            self.data['disposition'],
            self.data['duration_bucket']
        )

        # Reorder columns
        ordered_cols = [col for col in self.DURATION_BUCKET_ORDER if col in pivot.columns]
        pivot = pivot[ordered_cols]

        # Add totals
        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)

        return pivot

    # ==================== CONNECTIVITY ANALYSIS ====================

    def analyze_connectivity(self) -> Dict:
        """
        Analyze overall connectivity metrics

        Returns:
            Dictionary with connectivity statistics
        """
        total_calls = len(self.data)
        connected_calls = self.data['is_connected'].sum()
        not_connected = total_calls - connected_calls

        connected_data = self.data[self.data['is_connected']]

        return {
            'total_calls': total_calls,
            'connected_calls': int(connected_calls),
            'not_connected_calls': int(not_connected),
            'connectivity_rate': round(connected_calls / total_calls * 100, 2) if total_calls > 0 else 0,
            'avg_talk_time': round(connected_data['duration_seconds'].mean(), 2) if len(connected_data) > 0 else 0,
            'total_talk_time': round(connected_data['duration_seconds'].sum(), 2),
            'max_talk_time': round(connected_data['duration_seconds'].max(), 2) if len(connected_data) > 0 else 0,
        }

    def connectivity_by_interval(self) -> pd.DataFrame:
        """
        Analyze connectivity by time intervals (hourly)

        Returns:
            DataFrame with hourly connectivity analysis
        """
        if 'interval' not in self.data.columns and 'hour' not in self.data.columns:
            return pd.DataFrame()

        interval_col = 'interval' if 'interval' in self.data.columns else 'hour'

        analysis = self.data.groupby(interval_col).agg({
            'is_connected': ['count', 'sum'],
            'duration_seconds': ['sum', 'mean']
        }).reset_index()

        analysis.columns = [interval_col, 'Total Calls', 'Connected Calls',
                           'Total Duration (sec)', 'Avg Duration (sec)']

        analysis['Not Connected'] = analysis['Total Calls'] - analysis['Connected Calls']
        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        # Sort by interval
        analysis = analysis.sort_values(interval_col)

        return analysis

    def connectivity_by_trunk(self) -> pd.DataFrame:
        """
        Analyze connectivity by trunk/gateway

        Returns:
            DataFrame with trunk-wise connectivity analysis
        """
        if 'trunk' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('trunk').agg({
            'is_connected': ['count', 'sum'],
            'duration_seconds': ['sum', 'mean']
        }).reset_index()

        analysis.columns = ['Trunk', 'Total Calls', 'Connected Calls',
                           'Total Duration (sec)', 'Avg Duration (sec)']

        analysis['Not Connected'] = analysis['Total Calls'] - analysis['Connected Calls']
        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        analysis = analysis.sort_values('Total Calls', ascending=False)

        return analysis

    def connectivity_by_agent(self) -> pd.DataFrame:
        """Analyze connectivity by agent"""
        if 'agent' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('agent').agg({
            'is_connected': ['count', 'sum'],
            'duration_seconds': ['sum', 'mean']
        }).reset_index()

        analysis.columns = ['Agent', 'Total Calls', 'Connected Calls',
                           'Total Duration (sec)', 'Avg Duration (sec)']

        analysis['Not Connected'] = analysis['Total Calls'] - analysis['Connected Calls']
        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        analysis = analysis.sort_values('Total Calls', ascending=False)

        return analysis

    def connectivity_by_campaign(self) -> pd.DataFrame:
        """Analyze connectivity by campaign"""
        if 'campaign' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('campaign').agg({
            'is_connected': ['count', 'sum'],
            'duration_seconds': ['sum', 'mean']
        }).reset_index()

        analysis.columns = ['Campaign', 'Total Calls', 'Connected Calls',
                           'Total Duration (sec)', 'Avg Duration (sec)']

        analysis['Not Connected'] = analysis['Total Calls'] - analysis['Connected Calls']
        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        analysis = analysis.sort_values('Total Calls', ascending=False)

        return analysis

    # ==================== SALE ANALYSIS ====================

    def analyze_sales_by_duration(self) -> pd.DataFrame:
        """
        Analyze sale dispositions by duration buckets

        Returns:
            DataFrame with sale analysis per duration bucket
        """
        if 'is_sale' not in self.data.columns:
            return pd.DataFrame()

        sales_data = self.data[self.data['is_sale']]

        if len(sales_data) == 0:
            return pd.DataFrame()

        # Create detailed duration buckets for sales
        sales_data = sales_data.copy()
        sales_data['sale_duration_bucket'] = sales_data['duration_seconds'].apply(
            self._get_sale_duration_bucket
        )

        analysis = sales_data.groupby('sale_duration_bucket').agg({
            'is_sale': 'count',
            'duration_seconds': ['mean', 'sum']
        }).reset_index()

        analysis.columns = ['Duration Bucket', 'Sale Count', 'Avg Duration (sec)', 'Total Duration (sec)']

        # Calculate percentage
        total_sales = analysis['Sale Count'].sum()
        analysis['Percentage'] = (analysis['Sale Count'] / total_sales * 100).round(2)

        # Sort by bucket order
        bucket_order = ['0 seconds', '< 1 min', '1-2 mins', '2-3 mins',
                       '3-5 mins', '5-6 mins', '6-7 mins', '> 7 mins']
        analysis['sort_order'] = analysis['Duration Bucket'].apply(
            lambda x: bucket_order.index(x) if x in bucket_order else 99
        )
        analysis = analysis.sort_values('sort_order').drop('sort_order', axis=1)

        return analysis

    def _get_sale_duration_bucket(self, seconds: float) -> str:
        """Get duration bucket for sales analysis"""
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

    def sales_by_agent(self) -> pd.DataFrame:
        """Analyze sales by agent with duration breakdown"""
        if 'is_sale' not in self.data.columns or 'agent' not in self.data.columns:
            return pd.DataFrame()

        sales_data = self.data[self.data['is_sale']]

        if len(sales_data) == 0:
            return pd.DataFrame()

        analysis = sales_data.groupby('agent').agg({
            'is_sale': 'count',
            'duration_seconds': ['mean', 'sum', 'min', 'max']
        }).reset_index()

        analysis.columns = ['Agent', 'Sale Count', 'Avg Duration (sec)',
                           'Total Duration (sec)', 'Min Duration (sec)', 'Max Duration (sec)']

        # Add total calls for conversion rate
        total_calls = self.data.groupby('agent').size().reset_index(name='Total Calls')
        analysis = analysis.merge(total_calls, on='Agent', how='left')
        analysis['Conversion %'] = (analysis['Sale Count'] / analysis['Total Calls'] * 100).round(2)

        analysis = analysis.sort_values('Sale Count', ascending=False)

        return analysis

    def sales_by_campaign(self) -> pd.DataFrame:
        """Analyze sales by campaign"""
        if 'is_sale' not in self.data.columns or 'campaign' not in self.data.columns:
            return pd.DataFrame()

        sales_data = self.data[self.data['is_sale']]

        if len(sales_data) == 0:
            return pd.DataFrame()

        analysis = sales_data.groupby('campaign').agg({
            'is_sale': 'count',
            'duration_seconds': ['mean', 'sum']
        }).reset_index()

        analysis.columns = ['Campaign', 'Sale Count', 'Avg Duration (sec)', 'Total Duration (sec)']

        # Add total calls for conversion rate
        total_calls = self.data.groupby('campaign').size().reset_index(name='Total Calls')
        analysis = analysis.merge(total_calls, on='Campaign', how='left')
        analysis['Conversion %'] = (analysis['Sale Count'] / analysis['Total Calls'] * 100).round(2)

        analysis = analysis.sort_values('Sale Count', ascending=False)

        return analysis

    def sales_by_interval(self) -> pd.DataFrame:
        """Analyze sales by time interval"""
        if 'is_sale' not in self.data.columns or 'interval' not in self.data.columns:
            return pd.DataFrame()

        sales_data = self.data[self.data['is_sale']]

        if len(sales_data) == 0:
            return pd.DataFrame()

        analysis = sales_data.groupby('interval').agg({
            'is_sale': 'count',
            'duration_seconds': ['mean', 'sum']
        }).reset_index()

        analysis.columns = ['Interval', 'Sale Count', 'Avg Duration (sec)', 'Total Duration (sec)']

        # Add total calls for conversion rate
        total_calls = self.data.groupby('interval').size().reset_index(name='Total Calls')
        analysis = analysis.merge(total_calls, on='Interval', how='left')
        analysis['Conversion %'] = (analysis['Sale Count'] / analysis['Total Calls'] * 100).round(2)

        analysis = analysis.sort_values('Interval')

        return analysis

    def sales_by_trunk(self) -> pd.DataFrame:
        """Analyze sales by trunk"""
        if 'is_sale' not in self.data.columns or 'trunk' not in self.data.columns:
            return pd.DataFrame()

        sales_data = self.data[self.data['is_sale']]

        if len(sales_data) == 0:
            return pd.DataFrame()

        analysis = sales_data.groupby('trunk').agg({
            'is_sale': 'count',
            'duration_seconds': ['mean', 'sum']
        }).reset_index()

        analysis.columns = ['Trunk', 'Sale Count', 'Avg Duration (sec)', 'Total Duration (sec)']

        # Add total calls for conversion rate
        total_calls = self.data.groupby('trunk').size().reset_index(name='Total Calls')
        analysis = analysis.merge(total_calls, on='Trunk', how='left')
        analysis['Conversion %'] = (analysis['Sale Count'] / analysis['Total Calls'] * 100).round(2)

        analysis = analysis.sort_values('Sale Count', ascending=False)

        return analysis

    # ==================== AGENT PERFORMANCE ====================

    def agent_performance_summary(self) -> pd.DataFrame:
        """
        Comprehensive agent performance summary

        Returns:
            DataFrame with detailed agent metrics
        """
        if 'agent' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('agent').agg({
            'agent': 'count',
            'is_connected': 'sum',
            'duration_seconds': ['sum', 'mean', 'max']
        }).reset_index()

        analysis.columns = ['Agent', 'Total Calls', 'Connected Calls',
                           'Total Talk Time (sec)', 'Avg Talk Time (sec)', 'Max Talk Time (sec)']

        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        # Add sales if available
        if 'is_sale' in self.data.columns:
            sales = self.data[self.data['is_sale']].groupby('agent').size().reset_index(name='Sales')
            analysis = analysis.merge(sales, left_on='Agent', right_on='agent', how='left')
            analysis['Sales'] = analysis['Sales'].fillna(0).astype(int)
            analysis['Conversion %'] = (analysis['Sales'] / analysis['Total Calls'] * 100).round(2)
            analysis = analysis.drop('agent', axis=1, errors='ignore')

        analysis = analysis.sort_values('Total Calls', ascending=False)

        return analysis

    # ==================== CAMPAIGN PERFORMANCE ====================

    def campaign_performance_summary(self) -> pd.DataFrame:
        """
        Comprehensive campaign performance summary

        Returns:
            DataFrame with detailed campaign metrics
        """
        if 'campaign' not in self.data.columns:
            return pd.DataFrame()

        analysis = self.data.groupby('campaign').agg({
            'campaign': 'count',
            'is_connected': 'sum',
            'duration_seconds': ['sum', 'mean', 'max']
        }).reset_index()

        analysis.columns = ['Campaign', 'Total Calls', 'Connected Calls',
                           'Total Talk Time (sec)', 'Avg Talk Time (sec)', 'Max Talk Time (sec)']

        analysis['Connectivity %'] = (
            analysis['Connected Calls'] / analysis['Total Calls'] * 100
        ).round(2)

        # Add sales if available
        if 'is_sale' in self.data.columns:
            sales = self.data[self.data['is_sale']].groupby('campaign').size().reset_index(name='Sales')
            analysis = analysis.merge(sales, left_on='Campaign', right_on='campaign', how='left')
            analysis['Sales'] = analysis['Sales'].fillna(0).astype(int)
            analysis['Conversion %'] = (analysis['Sales'] / analysis['Total Calls'] * 100).round(2)
            analysis = analysis.drop('campaign', axis=1, errors='ignore')

        analysis = analysis.sort_values('Total Calls', ascending=False)

        return analysis

    # ==================== DURATION ANALYSIS ====================

    def duration_distribution(self) -> pd.DataFrame:
        """
        Analyze call duration distribution

        Returns:
            DataFrame with duration bucket analysis
        """
        analysis = self.data.groupby('duration_bucket').agg({
            'duration_bucket': 'count',
            'duration_seconds': ['mean', 'sum']
        }).reset_index()

        analysis.columns = ['Duration Bucket', 'Call Count', 'Avg Duration (sec)', 'Total Duration (sec)']

        total_calls = analysis['Call Count'].sum()
        analysis['Percentage'] = (analysis['Call Count'] / total_calls * 100).round(2)

        # Sort by bucket order
        analysis['sort_order'] = analysis['Duration Bucket'].apply(
            lambda x: self.DURATION_BUCKET_ORDER.index(x) if x in self.DURATION_BUCKET_ORDER else 99
        )
        analysis = analysis.sort_values('sort_order').drop('sort_order', axis=1)

        return analysis

    def duration_by_agent(self) -> pd.DataFrame:
        """Duration distribution by agent"""
        if 'agent' not in self.data.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(
            self.data['agent'],
            self.data['duration_bucket']
        )

        # Reorder columns
        ordered_cols = [col for col in self.DURATION_BUCKET_ORDER if col in pivot.columns]
        pivot = pivot[ordered_cols]

        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)

        return pivot

    def duration_by_campaign(self) -> pd.DataFrame:
        """Duration distribution by campaign"""
        if 'campaign' not in self.data.columns:
            return pd.DataFrame()

        pivot = pd.crosstab(
            self.data['campaign'],
            self.data['duration_bucket']
        )

        # Reorder columns
        ordered_cols = [col for col in self.DURATION_BUCKET_ORDER if col in pivot.columns]
        pivot = pivot[ordered_cols]

        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)

        return pivot

    # ==================== COMPREHENSIVE REPORTS ====================

    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary of all key metrics

        Returns:
            Dictionary with comprehensive summary
        """
        summary = {
            'overview': {
                'total_calls': len(self.data),
                'date_range': self._get_date_range(),
            },
            'connectivity': self.analyze_connectivity(),
        }

        # Add sales summary if available
        if 'is_sale' in self.data.columns:
            total_sales = self.data['is_sale'].sum()
            summary['sales'] = {
                'total_sales': int(total_sales),
                'conversion_rate': round(total_sales / len(self.data) * 100, 2) if len(self.data) > 0 else 0,
            }

        # Add dimension counts
        if 'agent' in self.data.columns:
            summary['overview']['unique_agents'] = self.data['agent'].nunique()

        if 'campaign' in self.data.columns:
            summary['overview']['unique_campaigns'] = self.data['campaign'].nunique()

        if 'trunk' in self.data.columns:
            summary['overview']['unique_trunks'] = self.data['trunk'].nunique()

        if 'disposition' in self.data.columns:
            summary['overview']['unique_dispositions'] = self.data['disposition'].nunique()

        return summary

    def _get_date_range(self) -> str:
        """Get date range of data"""
        if 'call_datetime' in self.data.columns:
            min_date = self.data['call_datetime'].min()
            max_date = self.data['call_datetime'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        return "Unknown"

    def get_insights(self) -> List[str]:
        """
        Generate key insights from the data

        Returns:
            List of insight strings
        """
        insights = []

        # Connectivity insights
        connectivity = self.analyze_connectivity()
        if connectivity:
            rate = connectivity['connectivity_rate']
            if rate < 30:
                insights.append(f"⚠️ Low connectivity rate: {rate}%. Consider reviewing dialing strategy.")
            elif rate > 70:
                insights.append(f"✓ Strong connectivity rate: {rate}%")

            avg_talk = connectivity['avg_talk_time']
            if avg_talk < 60:
                insights.append(f"⚠️ Low average talk time: {avg_talk:.0f} seconds. Calls may be ending prematurely.")
            elif avg_talk > 300:
                insights.append(f"✓ Good average talk time: {avg_talk:.0f} seconds")

        # Sales insights
        if 'is_sale' in self.data.columns:
            sales_by_duration = self.analyze_sales_by_duration()
            if not sales_by_duration.empty:
                zero_duration_sales = sales_by_duration[
                    sales_by_duration['Duration Bucket'] == '0 seconds'
                ]['Sale Count'].sum()

                if zero_duration_sales > 0:
                    insights.append(
                        f"⚠️ {zero_duration_sales} sales with 0 seconds duration. Review data quality."
                    )

                # Find best performing duration bucket
                if len(sales_by_duration) > 0:
                    best_bucket = sales_by_duration.loc[
                        sales_by_duration['Sale Count'].idxmax(), 'Duration Bucket'
                    ]
                    insights.append(f"📊 Most sales occur in the '{best_bucket}' duration range")

        # Agent insights
        if 'agent' in self.data.columns:
            agent_perf = self.agent_performance_summary()
            if not agent_perf.empty:
                top_agent = agent_perf.iloc[0]['Agent']
                top_calls = agent_perf.iloc[0]['Total Calls']
                insights.append(f"🏆 Top performer by volume: {top_agent} with {top_calls} calls")

                if 'Conversion %' in agent_perf.columns:
                    best_conversion = agent_perf.loc[agent_perf['Conversion %'].idxmax()]
                    insights.append(
                        f"🎯 Best conversion rate: {best_conversion['Agent']} at {best_conversion['Conversion %']}%"
                    )

        # Trunk insights
        trunk_analysis = self.connectivity_by_trunk()
        if not trunk_analysis.empty:
            worst_trunk = trunk_analysis.loc[trunk_analysis['Connectivity %'].idxmin()]
            if worst_trunk['Connectivity %'] < 30:
                insights.append(
                    f"⚠️ Trunk '{worst_trunk['Trunk']}' has low connectivity: {worst_trunk['Connectivity %']}%"
                )

        # Interval insights
        interval_analysis = self.connectivity_by_interval()
        if not interval_analysis.empty:
            best_interval = interval_analysis.loc[interval_analysis['Connectivity %'].idxmax()]
            interval_col = 'interval' if 'interval' in interval_analysis.columns else interval_analysis.columns[0]
            insights.append(
                f"📞 Best time to call: {best_interval[interval_col]} with {best_interval['Connectivity %']}% connectivity"
            )

        return insights
