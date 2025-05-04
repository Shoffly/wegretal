import streamlit as st
import pandas as pd
from google.cloud import bigquery
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
from google.oauth2 import service_account
from posthog import Posthog
import io

# Initialize PostHog
posthog = Posthog(
    project_api_key='phc_iY1kjQZ5ib5oy0PU2fRIqJZ5323jewSS5fVDNyhe7RY',
    host='https://us.i.posthog.com'
)

# Authentication credentials
CREDENTIALS = {
    "admin": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
    "user": "04f8996da763b7a969b1028ee3007569eaf3a635486ddab211d512c85b9df8fb",  # user
    "dina.teilab@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "mai.sobhy@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "mostafa.sayed@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "ahmed.hassan@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "mohamed.youssef@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "ahmed.nagy@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "adel.abuelella@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "ammar.abdelbaset@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "youssef.mohamed@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "abdallah.hazem@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "mohamed.abdelgalil@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
    "mohanad.elgarhy@sylndr.com": hashlib.sha256("sylndr123".encode()).hexdigest(),
}


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in CREDENTIALS and \
                hashlib.sha256(st.session_state["password"].encode()).hexdigest() == CREDENTIALS[
            st.session_state["username"]]:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = st.session_state["username"]  # Store the username

            # First identify the user
            posthog.identify(
                st.session_state["username"],  # Use email as distinct_id
                {
                    'email': st.session_state["username"],
                    'name': st.session_state["username"].split('@')[0].replace('.', ' ').title(),
                    'last_login': datetime.now().isoformat()
                }
            )

            # Then capture the login event
            posthog.capture(
                st.session_state["username"],
                '$login',
                {
                    'app': 'Campaign Analysis',
                    'login_method': 'password',
                    'success': True
                }
            )

            del st.session_state["password"]  # Don't store the password
            del st.session_state["username"]  # Don't store the username
        else:
            st.session_state["password_correct"] = False
            # Track failed login attempt with more details
            if "username" in st.session_state:
                posthog.capture(
                    st.session_state["username"],
                    '$login_failed',
                    {
                        'app': 'Campaign Analysis',
                        'login_method': 'password',
                        'reason': 'invalid_credentials',
                        'attempted_email': st.session_state["username"]
                    }
                )

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input fields for username and password
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=password_entered)

    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")

    return False


def clean_audience_data(df):
    """
    Clean and validate the audience data.
    Returns cleaned DataFrame or raises an error if data is invalid.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Clean column names - remove any whitespace and special characters
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required_columns = ['User ID', 'Sent Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Clean User IDs
    df['User ID'] = df['User ID'].astype(str).str.strip()

    # Parse dates - try multiple formats
    def parse_date(date_str):
        date_formats = [
            "%d %b '%y, %H:%M %Z",  # e.g., "09 Apr '25, 17:12 EET"
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y"
        ]

        if pd.isna(date_str):
            return None

        date_str = str(date_str).strip()

        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue

        try:
            # Try pandas default parser as last resort
            return pd.to_datetime(date_str)
        except (ValueError, TypeError):
            return None

    df['Sent Date'] = df['Sent Date'].apply(parse_date)

    # Remove rows with invalid dates
    invalid_dates = df['Sent Date'].isna()
    if invalid_dates.any():
        st.warning(f"Removed {invalid_dates.sum()} rows with invalid dates")
        df = df[~invalid_dates]

    # Ensure we have valid data
    if len(df) == 0:
        raise ValueError("No valid data rows after cleaning")

    # Convert numeric columns if they exist
    numeric_columns = ['Total Clicks', 'Total Conversions']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def analyze_campaign_data(client, audience_df):
    """
    Analyze user activity before and after campaign dates.
    Returns a DataFrame with activity metrics.
    """
    try:
        # Clean and validate the audience data
        audience_df = clean_audience_data(audience_df)

        # Get all unique user IDs
        user_ids = audience_df['User ID'].unique().tolist()

        if not user_ids:
            raise ValueError("No valid user IDs found in the data")

        # Get the overall date range needed
        min_sent_date = audience_df['Sent Date'].min() - timedelta(days=7)
        max_sent_date = audience_df['Sent Date'].max() + timedelta(days=7)

        # Query all activity data at once
        activity_query = f"""
        SELECT 
            client_id,
            action_date,
            action_name,
            COUNT(*) as action_count
        FROM reporting.retail_user_activity
        WHERE client_id IN UNNEST(@user_ids)
        AND action_date BETWEEN @min_date AND @max_date
        GROUP BY client_id, action_date, action_name
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("user_ids", "STRING", user_ids),
                bigquery.ScalarQueryParameter("min_date", "TIMESTAMP", min_sent_date),
                bigquery.ScalarQueryParameter("max_date", "TIMESTAMP", max_sent_date),
            ]
        )

        # Load all activity data
        with st.spinner("Loading activity data..."):
            activity_df = client.query(activity_query, job_config=job_config).to_dataframe()

        if activity_df.empty:
            st.warning("No activity data found for the specified users and date range")
            return pd.DataFrame()

        results = []

        # Process each user
        for _, row in audience_df.iterrows():
            user_id = row['User ID']
            sent_date = row['Sent Date']

            # Calculate date ranges
            before_start = sent_date - timedelta(days=7)
            before_end = sent_date
            after_start = sent_date
            after_end = sent_date + timedelta(days=7)

            # Filter activity data for this user
            user_activity = activity_df[activity_df['client_id'] == user_id].copy()
            user_activity['action_date'] = pd.to_datetime(user_activity['action_date'])

            # Get before period data
            before_data = user_activity[
                (user_activity['action_date'] >= before_start) &
                (user_activity['action_date'] <= before_end)
                ]

            # Get after period data
            after_data = user_activity[
                (user_activity['action_date'] >= after_start) &
                (user_activity['action_date'] <= after_end)
                ]

            # Prepare results
            result = {
                'User ID': user_id,
                'Sent Date': sent_date,
                'Message Status': row.get('Latest Message Status', 'Unknown'),
                'Total Clicks': row.get('Total Clicks', 0),
                'Total Conversions': row.get('Total Conversions', 0)
            }

            # Get all unique actions for this user
            all_actions = pd.concat([
                before_data['action_name'] if not before_data.empty else pd.Series(),
                after_data['action_name'] if not after_data.empty else pd.Series()
            ]).unique()

            # Calculate metrics for each action
            for action in all_actions:
                before_count = before_data[before_data['action_name'] == action][
                    'action_count'].sum() if not before_data.empty else 0
                after_count = after_data[after_data['action_name'] == action][
                    'action_count'].sum() if not after_data.empty else 0

                result[f'{action} (Before)'] = before_count
                result[f'{action} (After)'] = after_count
                result[f'{action} (Change)'] = after_count - before_count

            results.append(result)

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Error in analyze_campaign_data: {str(e)}")
        raise


def main():
    st.set_page_config(
        page_title="Campaign Analysis Tool",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Campaign Analysis Tool")

    if not check_password():
        return

    # Track page view
    if "current_user" in st.session_state:
        posthog.capture(
            st.session_state["current_user"],
            'page_view',
            {
                'page': 'campaign_analysis',
                'timestamp': datetime.now().isoformat()
            }
        )

    # File upload
    uploaded_file = st.file_uploader("Upload Campaign Audience CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            # Show sample format
            st.info("""
            Expected CSV format:
            - Required columns: 'User ID', 'Sent Date'
            - Optional columns: 'Latest Message Status', 'Total Clicks', 'Total Conversions'
            - Date format example: '09 Apr '25, 17:12 EET' or 'YYYY-MM-DD'
            """)

            # Read the CSV file
            audience_df = pd.read_csv(uploaded_file)

            # Display raw data
            st.subheader("Raw Data Preview")
            st.dataframe(audience_df.head())

            # Get BigQuery credentials
            try:
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["service_account"]
                )
            except (KeyError, FileNotFoundError):
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        'service_account.json'
                    )
                except FileNotFoundError:
                    st.error("No credentials found for BigQuery access")
                    return

            # Create BigQuery client
            client = bigquery.Client(credentials=credentials)

            # Analyze the data
            with st.spinner("Analyzing campaign data..."):
                results_df = analyze_campaign_data(client, audience_df)

            if results_df.empty:
                st.warning("No results to display")
                return

            # Display results
            st.subheader("Campaign Analysis Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Users", len(results_df))
            with col2:
                total_clicks = results_df['Total Clicks'].sum()
                st.metric("Total Clicks", total_clicks)
            with col3:
                total_conversions = results_df['Total Conversions'].sum()
                st.metric("Total Conversions", total_conversions)
            with col4:
                conversion_rate = (total_conversions / len(results_df)) * 100 if len(results_df) > 0 else 0
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")

            # Action metrics
            st.subheader("Action Metrics")

            # Get all action columns
            action_columns = [col for col in results_df.columns if '(Change)' in col]

            for action_col in action_columns:
                action_name = action_col.replace(' (Change)', '')
                total_before = results_df[f'{action_name} (Before)'].sum()
                total_after = results_df[f'{action_name} (After)'].sum()
                change = results_df[action_col].sum()
                change_pct = ((total_after - total_before) / total_before * 100) if total_before > 0 else 0

                st.metric(
                    action_name,
                    f"{total_after:,} (After) vs {total_before:,} (Before)",
                    f"{change:+,} ({change_pct:+.1f}%)"
                )

            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                "campaign_analysis_results.csv",
                "text/csv",
                key='download-csv'
            )

        except Exception as e:
            st.error(f"Error analyzing campaign data: {str(e)}")


if __name__ == "__main__":
    main()
