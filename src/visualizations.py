"""
CDR Visualization Module
Creates interactive charts and graphs for CDR analysis
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, List


class CDRVisualizer:
    """Creates visualizations for CDR analysis"""

    # Color scheme
    COLORS = {
        'primary': '#4472C4',
        'secondary': '#ED7D31',
        'success': '#70AD47',
        'warning': '#FFC000',
        'danger': '#C00000',
        'info': '#5B9BD5',
        'connected': '#70AD47',
        'not_connected': '#C00000',
    }

    COLOR_SEQUENCE = [
        '#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5',
        '#C00000', '#7030A0', '#00B0F0', '#92D050', '#FF6600'
    ]

    def __init__(self, analyzer):
        """
        Initialize visualizer

        Args:
            analyzer: CDRAnalyzer instance
        """
        self.analyzer = analyzer
        self.data = analyzer.data

    def plot_disposition_distribution(self) -> go.Figure:
        """Create disposition distribution pie chart"""
        disp_analysis = self.analyzer.analyze_dispositions()

        if disp_analysis.empty:
            return self._empty_chart("No disposition data available")

        # Get top 10 dispositions
        top_dispositions = disp_analysis.head(10)

        fig = px.pie(
            top_dispositions,
            values='Call Count',
            names='Disposition',
            title='Call Disposition Distribution (Top 10)',
            color_discrete_sequence=self.COLOR_SEQUENCE,
            hole=0.4
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )

        return fig

    def plot_connectivity_gauge(self) -> go.Figure:
        """Create connectivity rate gauge chart"""
        connectivity = self.analyzer.analyze_connectivity()

        if not connectivity:
            return self._empty_chart("No connectivity data available")

        rate = connectivity['connectivity_rate']

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rate,
            title={'text': "Connectivity Rate"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.COLORS['primary']},
                'steps': [
                    {'range': [0, 30], 'color': "#ffebee"},
                    {'range': [30, 60], 'color': "#fff3e0"},
                    {'range': [60, 100], 'color': "#e8f5e9"}
                ],
                'threshold': {
                    'line': {'color': self.COLORS['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            number={'suffix': "%"}
        ))

        fig.update_layout(height=300)

        return fig

    def plot_connectivity_by_interval(self) -> go.Figure:
        """Create connectivity by time interval chart"""
        interval_data = self.analyzer.connectivity_by_interval()

        if interval_data.empty:
            return self._empty_chart("No interval data available")

        interval_col = interval_data.columns[0]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for call counts
        fig.add_trace(
            go.Bar(
                x=interval_data[interval_col],
                y=interval_data['Connected Calls'],
                name='Connected',
                marker_color=self.COLORS['connected']
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=interval_data[interval_col],
                y=interval_data['Not Connected'],
                name='Not Connected',
                marker_color=self.COLORS['not_connected']
            ),
            secondary_y=False
        )

        # Line chart for connectivity %
        fig.add_trace(
            go.Scatter(
                x=interval_data[interval_col],
                y=interval_data['Connectivity %'],
                name='Connectivity %',
                mode='lines+markers',
                line=dict(color=self.COLORS['primary'], width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title='Connectivity by Time Interval',
            barmode='stack',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        fig.update_yaxes(title_text="Call Count", secondary_y=False)
        fig.update_yaxes(title_text="Connectivity %", secondary_y=True)

        return fig

    def plot_connectivity_by_trunk(self) -> go.Figure:
        """Create connectivity by trunk chart"""
        trunk_data = self.analyzer.connectivity_by_trunk()

        if trunk_data.empty:
            return self._empty_chart("No trunk data available")

        fig = px.bar(
            trunk_data.head(15),
            x='Trunk',
            y=['Connected Calls', 'Not Connected'],
            title='Connectivity by Trunk',
            barmode='stack',
            color_discrete_sequence=[self.COLORS['connected'], self.COLORS['not_connected']]
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        return fig

    def plot_duration_distribution(self) -> go.Figure:
        """Create duration distribution chart"""
        duration_data = self.analyzer.duration_distribution()

        if duration_data.empty:
            return self._empty_chart("No duration data available")

        fig = px.bar(
            duration_data,
            x='Duration Bucket',
            y='Call Count',
            title='Call Duration Distribution',
            color='Call Count',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )

        return fig

    def plot_sales_by_duration(self) -> go.Figure:
        """Create sales by duration bucket chart"""
        sales_data = self.analyzer.analyze_sales_by_duration()

        if sales_data.empty:
            return self._empty_chart("No sales data available")

        fig = px.bar(
            sales_data,
            x='Duration Bucket',
            y='Sale Count',
            title='Sales Distribution by Call Duration',
            color='Percentage',
            color_continuous_scale='Greens',
            text='Sale Count'
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)

        return fig

    def plot_agent_performance(self) -> go.Figure:
        """Create agent performance comparison chart"""
        agent_data = self.analyzer.agent_performance_summary()

        if agent_data.empty:
            return self._empty_chart("No agent data available")

        # Get top 15 agents by call volume
        top_agents = agent_data.head(15)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=top_agents['Agent'],
                y=top_agents['Total Calls'],
                name='Total Calls',
                marker_color=self.COLORS['primary']
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=top_agents['Agent'],
                y=top_agents['Connectivity %'],
                name='Connectivity %',
                mode='lines+markers',
                line=dict(color=self.COLORS['success'], width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        if 'Conversion %' in top_agents.columns:
            fig.add_trace(
                go.Scatter(
                    x=top_agents['Agent'],
                    y=top_agents['Conversion %'],
                    name='Conversion %',
                    mode='lines+markers',
                    line=dict(color=self.COLORS['secondary'], width=2),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )

        fig.update_layout(
            title='Agent Performance Overview',
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        fig.update_yaxes(title_text="Call Count", secondary_y=False)
        fig.update_yaxes(title_text="Percentage", secondary_y=True)

        return fig

    def plot_campaign_performance(self) -> go.Figure:
        """Create campaign performance chart"""
        campaign_data = self.analyzer.campaign_performance_summary()

        if campaign_data.empty:
            return self._empty_chart("No campaign data available")

        top_campaigns = campaign_data.head(10)

        fig = px.bar(
            top_campaigns,
            x='Campaign',
            y='Total Calls',
            color='Connectivity %',
            title='Campaign Performance Overview',
            color_continuous_scale='RdYlGn',
            text='Total Calls'
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)

        return fig

    def plot_hourly_heatmap(self) -> go.Figure:
        """Create hourly call volume heatmap"""
        if 'hour' not in self.data.columns or 'date' not in self.data.columns:
            return self._empty_chart("No datetime data available for heatmap")

        # Create pivot table
        hourly_data = pd.crosstab(self.data['date'], self.data['hour'])

        fig = px.imshow(
            hourly_data,
            title='Call Volume Heatmap (Date vs Hour)',
            labels=dict(x="Hour of Day", y="Date", color="Calls"),
            color_continuous_scale='Blues',
            aspect='auto'
        )

        return fig

    def plot_disposition_by_duration_heatmap(self) -> go.Figure:
        """Create disposition by duration heatmap"""
        disp_duration = self.analyzer.disposition_by_duration()

        if disp_duration.empty:
            return self._empty_chart("No disposition/duration data available")

        # Get top 15 dispositions
        top_disp = disp_duration.head(15).drop('Total', axis=1, errors='ignore')

        fig = px.imshow(
            top_disp,
            title='Disposition by Duration Heatmap',
            labels=dict(x="Duration Bucket", y="Disposition", color="Calls"),
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )

        return fig

    def plot_sales_funnel(self) -> go.Figure:
        """Create sales funnel visualization"""
        connectivity = self.analyzer.analyze_connectivity()

        if not connectivity:
            return self._empty_chart("No data available for funnel")

        total = connectivity['total_calls']
        connected = connectivity['connected_calls']

        # Get sales count if available
        sales = 0
        if 'is_sale' in self.data.columns:
            sales = int(self.data['is_sale'].sum())

        stages = ['Total Calls', 'Connected Calls', 'Sales']
        values = [total, connected, sales]

        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition='inside',
            textinfo='value+percent initial',
            marker=dict(
                color=[self.COLORS['primary'], self.COLORS['info'], self.COLORS['success']]
            )
        ))

        fig.update_layout(
            title='Call to Sale Funnel',
            showlegend=False
        )

        return fig

    def plot_kpi_cards(self) -> Dict:
        """Generate KPI data for dashboard cards"""
        connectivity = self.analyzer.analyze_connectivity()

        kpis = {
            'total_calls': {
                'value': connectivity.get('total_calls', 0),
                'label': 'Total Calls',
                'delta': None
            },
            'connected_calls': {
                'value': connectivity.get('connected_calls', 0),
                'label': 'Connected Calls',
                'delta': None
            },
            'connectivity_rate': {
                'value': f"{connectivity.get('connectivity_rate', 0):.1f}%",
                'label': 'Connectivity Rate',
                'delta': None
            },
            'avg_talk_time': {
                'value': f"{connectivity.get('avg_talk_time', 0):.0f}s",
                'label': 'Avg Talk Time',
                'delta': None
            }
        }

        if 'is_sale' in self.data.columns:
            total_sales = int(self.data['is_sale'].sum())
            conversion = round(total_sales / len(self.data) * 100, 2) if len(self.data) > 0 else 0
            kpis['total_sales'] = {
                'value': total_sales,
                'label': 'Total Sales',
                'delta': None
            }
            kpis['conversion_rate'] = {
                'value': f"{conversion}%",
                'label': 'Conversion Rate',
                'delta': None
            }

        return kpis

    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    def create_dashboard_charts(self) -> Dict[str, go.Figure]:
        """Create all charts for dashboard"""
        charts = {}

        charts['disposition_pie'] = self.plot_disposition_distribution()
        charts['connectivity_gauge'] = self.plot_connectivity_gauge()
        charts['connectivity_interval'] = self.plot_connectivity_by_interval()
        charts['connectivity_trunk'] = self.plot_connectivity_by_trunk()
        charts['duration_dist'] = self.plot_duration_distribution()
        charts['sales_duration'] = self.plot_sales_by_duration()
        charts['agent_performance'] = self.plot_agent_performance()
        charts['campaign_performance'] = self.plot_campaign_performance()
        charts['sales_funnel'] = self.plot_sales_funnel()
        charts['disposition_duration_heatmap'] = self.plot_disposition_by_duration_heatmap()

        return charts
