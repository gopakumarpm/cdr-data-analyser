"""
Generate sample CDR data for testing the analyzer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


def generate_sample_cdr(num_records: int = 5000) -> pd.DataFrame:
    """Generate realistic sample CDR data"""

    # Define possible values
    agents = [f"Agent_{i:03d}" for i in range(1, 21)]  # 20 agents

    campaigns = [
        "Sales_Campaign_Q1",
        "Lead_Generation_2024",
        "Customer_Retention",
        "New_Product_Launch",
        "Survey_Campaign",
        "Collection_Drive"
    ]

    trunks = [
        "TRUNK_PRIMARY",
        "TRUNK_BACKUP",
        "SIP_GATEWAY_1",
        "SIP_GATEWAY_2",
        "VOIP_LINE_A",
        "VOIP_LINE_B"
    ]

    # Realistic dispositions with weights
    dispositions = {
        'Sale': 5,
        'Interested': 8,
        'Callback Scheduled': 10,
        'Not Interested': 15,
        'Do Not Call': 5,
        'Voicemail': 12,
        'No Answer': 20,
        'Busy': 10,
        'Wrong Number': 5,
        'Disconnected': 3,
        'Fax/Modem': 2,
        'Language Barrier': 3,
        'Transferred': 2
    }

    disp_list = []
    for disp, weight in dispositions.items():
        disp_list.extend([disp] * weight)

    # Generate data
    data = []
    start_date = datetime(2024, 1, 1, 8, 0, 0)

    for i in range(num_records):
        # Random datetime within business hours over 3 months
        days_offset = random.randint(0, 90)
        hour = random.choices(
            range(8, 21),
            weights=[5, 8, 10, 12, 15, 15, 12, 10, 8, 8, 6, 4, 2]
        )[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        call_datetime = start_date + timedelta(
            days=days_offset,
            hours=hour - 8,
            minutes=minute,
            seconds=second
        )

        # Select disposition
        disposition = random.choice(disp_list)

        # Duration based on disposition
        if disposition in ['No Answer', 'Busy', 'Disconnected', 'Fax/Modem']:
            duration = 0
        elif disposition in ['Wrong Number', 'Voicemail']:
            duration = random.randint(5, 30)
        elif disposition == 'Sale':
            # Sales typically have longer durations, but some are short
            duration_type = random.choices(
                ['short', 'medium', 'long'],
                weights=[10, 30, 60]
            )[0]
            if duration_type == 'short':
                duration = random.randint(0, 60)
            elif duration_type == 'medium':
                duration = random.randint(120, 300)
            else:
                duration = random.randint(300, 900)
        elif disposition in ['Interested', 'Callback Scheduled']:
            duration = random.randint(60, 300)
        elif disposition == 'Transferred':
            duration = random.randint(30, 120)
        else:
            duration = random.randint(20, 180)

        # Generate phone number
        phone = f"+1{random.randint(200, 999)}{random.randint(1000000, 9999999)}"

        record = {
            'call_id': f"CDR_{i+1:08d}",
            'datetime': call_datetime,
            'agent': random.choice(agents),
            'campaign': random.choice(campaigns),
            'trunk': random.choice(trunks),
            'phone_number': phone,
            'disposition': disposition,
            'duration': duration,
            'ring_time': random.randint(3, 30) if duration > 0 else random.randint(15, 60),
            'hold_time': random.randint(0, 30) if duration > 60 else 0,
            'call_type': 'Outbound'
        }

        data.append(record)

    df = pd.DataFrame(data)

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def main():
    """Generate and save sample data"""
    print("Generating sample CDR data...")

    # Generate 5000 records
    df = generate_sample_cdr(5000)

    # Save as CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_cdr_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save as Excel
    xlsx_path = os.path.join(os.path.dirname(__file__), 'sample_cdr_data.xlsx')
    df.to_excel(xlsx_path, index=False, sheet_name='CDR Data')
    print(f"Saved Excel: {xlsx_path}")

    # Print summary
    print("\n--- Sample Data Summary ---")
    print(f"Total Records: {len(df)}")
    print(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Agents: {df['agent'].nunique()}")
    print(f"Campaigns: {df['campaign'].nunique()}")
    print(f"Trunks: {df['trunk'].nunique()}")
    print(f"\nDisposition Distribution:")
    print(df['disposition'].value_counts())

    print("\nDone! You can now use these files to test the CDR Analyser.")


if __name__ == "__main__":
    main()
