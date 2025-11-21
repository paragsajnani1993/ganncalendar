import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, date
from streamlit_calendar import calendar 
import pandas_market_calendars as mcal

# ==========================================
# 1. ZIGZAG CALCULATION ENGINE
# ==========================================
def calculate_zigzag(df, deviation_percent=5):
    df = df.dropna()
    if len(df) < 2: return []

    highs = df['High'].values.flatten()
    lows = df['Low'].values.flatten()
    all_dates = df.index 
    
    deviation = deviation_percent / 100.0
    
    current_trend = 0 
    last_pivot_price = highs[0]
    last_pivot_index = 0
    pivots = [] 

    # Helper to add pivot
    def add_p(idx, price, ptype):
        pivots.append({
            'Date': all_dates[idx], 
            'Price': price, 
            'Type': ptype,
            'Index': idx 
        })

    for i in range(1, len(df)):
        curr_high = highs[i]
        curr_low = lows[i]
        
        if current_trend == 0:
            if curr_high > last_pivot_price * (1 + deviation):
                current_trend = 1 
                add_p(last_pivot_index, lows[last_pivot_index], 'Low')
                last_pivot_price = curr_high
                last_pivot_index = i
            elif curr_low < last_pivot_price * (1 - deviation):
                current_trend = -1 
                add_p(last_pivot_index, highs[last_pivot_index], 'High')
                last_pivot_price = curr_low
                last_pivot_index = i
        elif current_trend == 1:
            if curr_high > last_pivot_price:
                last_pivot_price = curr_high
                last_pivot_index = i
            elif curr_low < last_pivot_price * (1 - deviation):
                add_p(last_pivot_index, last_pivot_price, 'High')
                current_trend = -1 
                last_pivot_price = curr_low
                last_pivot_index = i
        elif current_trend == -1:
            if curr_low < last_pivot_price:
                last_pivot_price = curr_low
                last_pivot_index = i
            elif curr_high > last_pivot_price * (1 + deviation):
                add_p(last_pivot_index, last_pivot_price, 'Low')
                current_trend = 1 
                last_pivot_price = curr_high
                last_pivot_index = i
    
    if current_trend == 1:
        add_p(last_pivot_index, last_pivot_price, 'High (Current)')
    elif current_trend == -1:
        add_p(last_pivot_index, last_pivot_price, 'Low (Current)')

    return pivots

# ==========================================
# 2. EVENT GENERATION (DYNAMIC EXCHANGE)
# ==========================================
def generate_calendar_events(pivots, df_index, exchange_code):
    cycles = [36, 81, 99, 144, 360, 411, 822, 1644, 720, 1440]
    events = []
    
    # --- A. Get the Correct Calendar ---
    # Only fetch the calendar selected by the user
    try:
        market_cal = mcal.get_calendar(exchange_code)
    except:
        # Fallback to NYSE if code is wrong, but dropdown prevents this
        market_cal = mcal.get_calendar('NYSE')
    
    # --- B. Generate Future Schedule ---
    last_hist_date = df_index[-1]
    future_start = last_hist_date + timedelta(days=1)
    future_end = last_hist_date + timedelta(days=365 * 10) 
    
    # Get valid trading days for the specific exchange
    future_schedule = market_cal.schedule(start_date=future_start, end_date=future_end)
    future_dates = mcal.date_range(future_schedule, frequency='1D')
    
    # --- C. Build Master Timeline ---
    history_dates_naive = df_index.tz_localize(None)
    future_dates_naive = future_dates.tz_localize(None)
    master_timeline = history_dates_naive.append(future_dates_naive)
    total_capacity = len(master_timeline)

    for p in pivots:
        origin_date = pd.to_datetime(p['Date']).tz_localize(None)
        origin_str = origin_date.strftime('%d-%b-%y')
        pivot_type = p['Type']
        origin_idx = p['Index'] 
        
        for c in cycles:
            # 1. CALENDAR DAYS (Orange)
            target_date_cal = origin_date + timedelta(days=c)
            events.append({
                "title": f"{c}D (Cal) from {pivot_type} {origin_str}",
                "start": target_date_cal.strftime("%Y-%m-%d"),
                "backgroundColor": "#FF8C00", 
                "borderColor": "#FF8C00",
                "allDay": True
            })
            
            # 2. TRADING BARS (Blue) - Exchange Specific
            target_idx = origin_idx + c
            
            if target_idx < total_capacity:
                target_date_bar = master_timeline[target_idx]
                events.append({
                    "title": f"{c}B (Bars) from {pivot_type} {origin_str}",
                    "start": target_date_bar.strftime("%Y-%m-%d"),
                    "backgroundColor": "#1E90FF", 
                    "borderColor": "#1E90FF",
                    "allDay": True
                })
            
    return events

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Gann Cycle Calendar", layout="wide")
st.title("🗓️ ZigZag & Gann Time Cycle Calendar")

if 'calendar_events' not in st.session_state:
    st.session_state['calendar_events'] = None
if 'pivot_data' not in st.session_state:
    st.session_state['pivot_data'] = None

with st.sidebar:
    st.header("Configuration")
    
    # Ticker Input
    ticker_input = st.text_input("Stock Ticker", value="GC=F")
    st.caption("Gold: `GC=F` (Global), `GOLDBEES.NS` (India)")
    
    # Exchange Selection
    st.write("---")
    st.write("**Step 1: Select Holiday Calendar**")
    exchange_select = st.selectbox(
        "Exchange / Market",
        options=["NSE", "CMES", "NYSE", "LSE", "JPX"],
        index=1, # Default to CMES for Gold
        format_func=lambda x: {
            "NSE": "NSE (India - Nifty/Stocks)",
            "CMES": "CME (Global - Gold/Oil/BTC)",
            "NYSE": "NYSE (USA - Apple/Tesla)",
            "LSE": "LSE (London)",
            "JPX": "JPX (Japan)"
        }.get(x, x)
    )
    
    
    st.write("---")
    period_select = st.selectbox("History Length", ["2y", "5y", "10y", "max"], index=2)
    deviation_input = st.number_input("Pivot Deviation (%)", value=5)
    
    if st.button("Generate Analysis", type="primary"):
        with st.spinner(f"Fetching data for {ticker_input} using {exchange_select} calendar..."):
            try:
                df = yf.download(ticker_input, period=period_select, interval="1d", progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if df.empty:
                    st.error("No data found.")
                else:
                    pivot_list = calculate_zigzag(df, deviation_percent=deviation_input)
                    
                    if not pivot_list:
                        st.warning("No pivots found.")
                    else:
                        # Pass the selected exchange code to the generator
                        events = generate_calendar_events(pivot_list, df.index, exchange_select)
                        
                        st.session_state['pivot_data'] = pivot_list
                        st.session_state['calendar_events'] = events
                        
            except Exception as e:
                st.error(f"Error: {e}")

# Main Display
if st.session_state['calendar_events']:
    calendar_events = st.session_state['calendar_events']
    calendar_options = {
        "initialView": "dayGridMonth",
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,listYear"
        },
        "height": 700,
    }
    
    st.subheader(f"Cycle Projections for {ticker_input}")
    calendar(events=calendar_events, options=calendar_options)
    
    st.divider()
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("📋 Upcoming Key Dates")
        events_df = pd.DataFrame(calendar_events)
        if not events_df.empty:
            events_df['Date'] = pd.to_datetime(events_df['start'])
            today = pd.Timestamp.now().normalize()
            future_events = events_df[events_df['Date'] >= today].sort_values('Date')
            
            st.dataframe(
                future_events[['start', 'title']], 
                use_container_width=True,
                hide_index=True,
                column_config={"start": "Event Date", "title": "Description"},
                height=400
            )

    with col_b:
        st.subheader("📜 All Identified Pivots")
        if st.session_state['pivot_data']:
            p_df = pd.DataFrame(st.session_state['pivot_data'])
            p_df['Date'] = pd.to_datetime(p_df['Date']).dt.strftime('%Y-%m-%d')
            p_df_sorted = p_df.iloc[::-1]

            st.dataframe(
                p_df_sorted[['Date', 'Price', 'Type']], 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": "Pivot Date",
                    "Price": st.column_config.NumberColumn("Pivot Price", format="%.2f"),
                    "Type": "Pivot Type"
                },
                height=400
            )
            
            csv = p_df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Pivots CSV",
                data=csv,
                file_name=f'{ticker_input}_pivots.csv',
                mime='text/csv',
            )