"""
CDR Data Loader Module
Handles loading and parsing of CDR data from various file formats
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import re


class CDRDataLoader:
    """Loads and parses CDR data from Excel and CSV files"""

    # Common column name mappings for different CDR formats
    COLUMN_MAPPINGS = {
        'agent': ['agent', 'agent_name', 'agentname', 'user', 'username', 'user_name',
                  'operator', 'rep', 'representative', 'agent_id', 'agentid', 'emp_name',
                  'employee', 'caller_id'],
        'disposition': ['disposition', 'status', 'call_status', 'callstatus', 'result',
                       'call_result', 'outcome', 'call_outcome', 'dispo', 'disp'],
        'duration': ['duration', 'talk_time', 'talktime', 'call_duration', 'callduration',
                    'talk_duration', 'connected_duration', 'billsec', 'bill_sec',
                    'duration_seconds', 'talk_secs', 'call_length'],
        'trunk': ['trunk', 'trunk_name', 'trunkname', 'gateway', 'channel', 'route',
                 'did', 'dnis', 'carrier', 'provider', 'trunk_id', 'line'],
        'campaign': ['campaign', 'campaign_name', 'campaignname', 'project', 'program',
                    'campaign_id', 'process', 'list', 'list_name'],
        'datetime': ['datetime', 'date_time', 'call_date', 'calldate', 'timestamp',
                    'call_time', 'start_time', 'starttime', 'call_datetime', 'date'],
        'phone': ['phone', 'phone_number', 'phonenumber', 'customer_phone', 'number',
                 'mobile', 'destination', 'called_number', 'ani', 'cli'],
        'ring_time': ['ring_time', 'ringtime', 'ring_duration', 'wait_time', 'queue_time'],
        'hold_time': ['hold_time', 'holdtime', 'hold_duration'],
        'call_type': ['call_type', 'calltype', 'direction', 'type', 'inbound_outbound'],
        'unique_id': ['unique_id', 'uniqueid', 'call_id', 'callid', 'id', 'record_id']
    }

    # Sale-related disposition keywords
    SALE_KEYWORDS = ['sale', 'sold', 'converted', 'successful', 'confirmed', 'booked',
                     'appointment', 'qualified', 'interested', 'hot', 'positive',
                     'payment', 'paid', 'closed', 'won', 'success', 'yes']

    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.file_path: Optional[str] = None
        self.load_errors: List[str] = []

    def load_file(self, file_path: str, sheet_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Load CDR data from file (Excel or CSV)

        Args:
            file_path: Path to the file
            sheet_name: Sheet name for Excel files (optional)

        Returns:
            Tuple of (success: bool, message: str)
        """
        self.load_errors = []
        self.file_path = file_path

        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"File not found: {file_path}"

            extension = path.suffix.lower()

            if extension == '.csv':
                self.raw_data = self._load_csv(file_path)
            elif extension in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
                self.raw_data = self._load_excel(file_path, sheet_name)
            else:
                return False, f"Unsupported file format: {extension}"

            if self.raw_data is None or self.raw_data.empty:
                return False, "File loaded but contains no data"

            # Process the data
            self._detect_columns()
            self._process_data()

            return True, f"Successfully loaded {len(self.raw_data)} records"

        except Exception as e:
            return False, f"Error loading file: {str(e)}"

    def load_from_upload(self, uploaded_file, file_name: str, sheet_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Load CDR data from uploaded file object (for Streamlit)

        Args:
            uploaded_file: File-like object
            file_name: Name of the file
            sheet_name: Sheet name for Excel files (optional)

        Returns:
            Tuple of (success: bool, message: str)
        """
        self.load_errors = []
        self.file_path = file_name

        try:
            extension = Path(file_name).suffix.lower()

            if extension == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        uploaded_file.seek(0)
                        self.raw_data = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif extension in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
                # Handle Excel files - ensure we get a DataFrame, not a dict
                uploaded_file.seek(0)

                if sheet_name:
                    # Specific sheet requested
                    self.raw_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                else:
                    # No sheet specified - read first sheet or handle dict
                    result = pd.read_excel(uploaded_file, sheet_name=0)

                    # If result is a dict (multiple sheets), get the first one
                    if isinstance(result, dict):
                        first_sheet = list(result.keys())[0]
                        self.raw_data = result[first_sheet]
                    else:
                        self.raw_data = result
            else:
                return False, f"Unsupported file format: {extension}"

            # Ensure we have a DataFrame
            if isinstance(self.raw_data, dict):
                # If still a dict, get first sheet
                first_key = list(self.raw_data.keys())[0]
                self.raw_data = self.raw_data[first_key]

            if self.raw_data is None:
                return False, "File loaded but contains no data"

            if not isinstance(self.raw_data, pd.DataFrame):
                return False, f"Unexpected data type: {type(self.raw_data)}"

            if self.raw_data.empty:
                return False, "File loaded but contains no data"

            # Process the data
            self._detect_columns()
            self._process_data()

            return True, f"Successfully loaded {len(self.raw_data)} records"

        except Exception as e:
            import traceback
            return False, f"Error loading file: {str(e)}\n{traceback.format_exc()}"

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        # Last resort - ignore errors
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')

    def _load_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load Excel file"""
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        return pd.read_excel(file_path)

    def get_excel_sheets(self, file_path: str) -> List[str]:
        """Get list of sheet names from Excel file"""
        try:
            xl = pd.ExcelFile(file_path)
            return xl.sheet_names
        except Exception:
            return []

    def _detect_columns(self):
        """Auto-detect column mappings based on column names"""
        if self.raw_data is None:
            return

        self.column_mapping = {}
        columns_lower = {col: col.lower().strip().replace(' ', '_') for col in self.raw_data.columns}

        for field, possible_names in self.COLUMN_MAPPINGS.items():
            for col, col_lower in columns_lower.items():
                if col_lower in possible_names or any(name in col_lower for name in possible_names):
                    self.column_mapping[field] = col
                    break

    def _process_data(self):
        """Process raw data into standardized format"""
        if self.raw_data is None:
            return

        self.processed_data = self.raw_data.copy()

        # Standardize column names
        rename_map = {v: k for k, v in self.column_mapping.items()}
        self.processed_data = self.processed_data.rename(columns=rename_map)

        # Process duration column
        if 'duration' in self.processed_data.columns:
            self.processed_data['duration_seconds'] = self.processed_data['duration'].apply(
                self._parse_duration
            )
        else:
            # Try to find any duration-like column
            for col in self.processed_data.columns:
                if 'duration' in col.lower() or 'time' in col.lower():
                    self.processed_data['duration_seconds'] = self.processed_data[col].apply(
                        self._parse_duration
                    )
                    break

        # Process datetime column
        if 'datetime' in self.processed_data.columns:
            self.processed_data['call_datetime'] = pd.to_datetime(
                self.processed_data['datetime'], errors='coerce'
            )
            # Extract time intervals
            self.processed_data['hour'] = self.processed_data['call_datetime'].dt.hour
            self.processed_data['date'] = self.processed_data['call_datetime'].dt.date
            self.processed_data['interval'] = self.processed_data['hour'].apply(
                lambda x: f"{x:02d}:00-{x:02d}:59" if pd.notna(x) else "Unknown"
            )

        # Identify sale dispositions
        if 'disposition' in self.processed_data.columns:
            self.processed_data['is_sale'] = self.processed_data['disposition'].apply(
                self._is_sale_disposition
            )

        # Create duration buckets
        if 'duration_seconds' in self.processed_data.columns:
            self.processed_data['duration_bucket'] = self.processed_data['duration_seconds'].apply(
                self._get_duration_bucket
            )

    def _parse_duration(self, value) -> float:
        """Parse duration value to seconds"""
        if pd.isna(value):
            return 0.0

        # Already numeric
        if isinstance(value, (int, float)):
            return float(value)

        value_str = str(value).strip()

        # Handle HH:MM:SS format
        if ':' in value_str:
            parts = value_str.split(':')
            try:
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(float, parts)
                    return m * 60 + s
            except ValueError:
                pass

        # Try direct numeric conversion
        try:
            return float(re.sub(r'[^\d.]', '', value_str))
        except ValueError:
            return 0.0

    def _is_sale_disposition(self, disposition) -> bool:
        """Check if disposition indicates a sale"""
        if pd.isna(disposition):
            return False

        disp_lower = str(disposition).lower()
        return any(keyword in disp_lower for keyword in self.SALE_KEYWORDS)

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

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get processed data"""
        return self.processed_data

    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get raw data"""
        return self.raw_data

    def get_column_mapping(self) -> Dict[str, str]:
        """Get detected column mappings"""
        return self.column_mapping

    def get_available_columns(self) -> List[str]:
        """Get list of available columns in processed data"""
        if self.processed_data is not None:
            return list(self.processed_data.columns)
        return []

    def update_column_mapping(self, mapping: Dict[str, str]):
        """Update column mapping and reprocess data"""
        self.column_mapping = mapping
        self._process_data()

    def get_unique_values(self, column: str) -> List:
        """Get unique values from a column"""
        if self.processed_data is not None and column in self.processed_data.columns:
            return self.processed_data[column].dropna().unique().tolist()
        return []

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of loaded data"""
        if self.processed_data is None:
            return {}

        stats = {
            'total_records': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'detected_mappings': self.column_mapping,
        }

        if 'duration_seconds' in self.processed_data.columns:
            stats['avg_duration'] = self.processed_data['duration_seconds'].mean()
            stats['max_duration'] = self.processed_data['duration_seconds'].max()
            stats['min_duration'] = self.processed_data['duration_seconds'].min()

        if 'disposition' in self.processed_data.columns:
            stats['disposition_count'] = self.processed_data['disposition'].nunique()

        if 'agent' in self.processed_data.columns:
            stats['agent_count'] = self.processed_data['agent'].nunique()

        if 'trunk' in self.processed_data.columns:
            stats['trunk_count'] = self.processed_data['trunk'].nunique()

        if 'campaign' in self.processed_data.columns:
            stats['campaign_count'] = self.processed_data['campaign'].nunique()

        return stats
