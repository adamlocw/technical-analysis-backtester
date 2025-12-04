import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import timedelta

# -----------------------------------------------------------------------------
# 1. 介面翻譯與設定 (UI & Translation)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Crypto/Stock Backtester", layout="wide")

TRANSLATIONS = {
    'English': {
        'title': 'Technical Analysis Backtester',
        'sidebar_title': 'Configuration',
        'ticker_label': 'Ticker Symbol (e.g., BTC-USD, AAPL)',
        'period_label': 'Data Period',
        'error_ticker': 'Error: Could not fetch data for symbol "{symbol}". Please check inputs.',
        'tab_macd': 'MACD Divergence',
        'tab_trend': 'Trendline Breakout',
        'tab_summary': 'Backtest Summary',
        'btn_run': 'Run Analysis',
        'loading': 'Fetching data and calculating...',
        'col_date': 'Signal Date',
        'col_type': 'Type',
        'col_entry_date': 'Entry Date',
        'col_entry_price': 'Entry Price',
        'col_exit_date': 'Exit Date',
        'col_exit_price': 'Exit Price',
        'col_return': 'Return (%)',
        'total_signals': 'Total Signals',
        'win_rate': 'Win Rate',
        'reliability': 'Reliability Score',
        'rating_tooltip': '>80%: ★★★★★\n70-80%: ★★★★☆\n60-70%: ★★★☆☆\n50-60%: ★★☆☆☆\n<50%: ★☆☆☆☆',
        'bullish': 'Bullish',
        'bearish': 'Bearish',
        'chart_price': 'Price & Signals',
        'chart_macd': 'MACD Oscillator'
    },
    '繁體中文': {
        'title': '技術分析回測應用程式',
        'sidebar_title': '參數設定',
        'ticker_label': '標的代碼 (例如 BTC-USD, AAPL)',
        'period_label': '資料期間',
        'error_ticker': '錯誤：無法取得 "{symbol}" 的資料。請檢查代碼是否正確。',
        'tab_macd': 'MACD 背馳策略',
        'tab_trend': '趨勢線突破策略',
        'tab_summary': '回測摘要',
        'btn_run': '執行分析',
        'loading': '正在擷取資料並計算中...',
        'col_date': '訊號觸發日',
        'col_type': '訊號類型',
        'col_entry_date': '進場日期',
        'col_entry_price': '進場價格',
        'col_exit_date': '出場日期',
        'col_exit_price': '出場價格',
        'col_return': '報酬率 (%)',
        'total_signals': '總訊號數',
        'win_rate': '勝率',
        'reliability': '可靠度評分',
        'rating_tooltip': '>80%: ★★★★★\n70-80%: ★★★★☆\n60-70%: ★★★☆☆\n50-60%: ★★☆☆☆\n<50%: ★☆☆☆☆',
        'bullish': '看漲 (做多)',
        'bearish': '看跌 (做空)',
        # 修改：圖表標題強制使用英文以避免亂碼
        'chart_price': 'Price & Signals',
        'chart_macd': 'MACD Oscillator'
    }
}

# -----------------------------------------------------------------------------
# 2. 核心邏輯函式 (Core Logic)
# -----------------------------------------------------------------------------

@st.cache_data
def get_stock_data(ticker, period="1y"):
    """
    Download OHLCV data from yfinance.
    Supports standard periods (1y, 2y, 5y) and calculates start dates for others (3y, 4y).
    """
    try:
        # yfinance standard valid periods that are years-based
        standard_periods = ['1y', '2y', '5y', '10y', 'max']
        
        if period in standard_periods:
            df = yf.download(ticker, period=period, progress=False, multi_level_index=False)
        else:
            # Handle custom years like '3y', '4y' by calculating start date
            try:
                years = int(period.replace('y', ''))
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
                df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            except ValueError:
                # Fallback to 1y if parsing fails
                df = yf.download(ticker, period="1y", progress=False, multi_level_index=False)

        if df.empty:
            return None
        # Ensure standard column names
        df.columns = [c.capitalize() for c in df.columns]
        return df
    except Exception as e:
        return None

def calculate_star_rating(win_rate):
    """Return star rating string based on win rate percentage."""
    if win_rate >= 80: return "★★★★★"
    elif win_rate >= 70: return "★★★★☆"
    elif win_rate >= 60: return "★★★☆☆"
    elif win_rate >= 50: return "★★☆☆☆"
    else: return "★☆☆☆☆"

def perform_backtest(df, signals, hold_days=10):
    """
    Generic backtest engine.
    signals: list of dicts {'index': int, 'type': 'Bullish'/'Bearish', 'date': timestamp}
    Returns DataFrame of trades.
    """
    trades = []
    
    for sig in signals:
        idx = sig['index']
        # Check if we have enough data for entry (next day) and exit (next day + hold)
        if idx + 1 >= len(df) or idx + 1 + hold_days >= len(df):
            continue
            
        entry_idx = idx + 1
        exit_idx = idx + 1 + hold_days
        
        entry_date = df.index[entry_idx]
        entry_price = df['Open'].iloc[entry_idx]
        
        exit_date = df.index[exit_idx]
        exit_price = df['Close'].iloc[exit_idx]
        
        pct_return = 0.0
        if sig['type'] == 'Bullish':
            pct_return = (exit_price - entry_price) / entry_price * 100
        else: # Bearish
            pct_return = (entry_price - exit_price) / entry_price * 100
            
        trades.append({
            'Signal Date': sig['date'],
            'Type': sig['type'],
            'Entry Date': entry_date,
            'Entry Price': entry_price,
            'Exit Date': exit_date,
            'Exit Price': exit_price,
            'Return (%)': pct_return
        })
        
    return pd.DataFrame(trades)

# --- Strategy 1: MACD Divergence ---

def strategy_macd_divergence(df):
    """
    Detect MACD Divergences.
    Logic:
    1. Calculate MACD.
    2. Find local extrema (peaks/valleys) in Price (Low/High) and MACD (Hist or Line).
    3. Bullish Div: Price Lows are Lower, MACD Lows are Higher.
    4. Bearish Div: Price Highs are Higher, MACD Highs are Lower.
    """
    # 1. Calc MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    # Column names typically: MACD_12_26_9, MACDh_12_26_9 (Hist), MACDs_12_26_9 (Signal)
    macd_col = 'MACD_12_26_9'
    hist_col = 'MACDh_12_26_9' 
    
    # 2. Find Extrema (Order=5 means checking 5 candles before and 5 after)
    # Note: Using order=5 implies a lag of 5 days to confirm the signal in real-time.
    # We will simulate the signal trigger at index + 5.
    order = 5
    lookback = 20
    
    # Find local minima (indices)
    price_lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    macd_lows_idx = argrelextrema(df[macd_col].values, np.less, order=order)[0]
    
    # Find local maxima (indices)
    price_highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    macd_highs_idx = argrelextrema(df[macd_col].values, np.greater, order=order)[0]
    
    signals = []
    
    # Check Bullish Divergence
    # Iterate through confirmed Price Lows
    for i in range(1, len(price_lows_idx)):
        curr_idx = price_lows_idx[i]
        prev_idx = price_lows_idx[i-1]
        
        # Check if points are within lookback range to consider them "connected"
        if (curr_idx - prev_idx) > lookback:
            continue
            
        # Price Lower Low
        price_lower_low = df['Low'].iloc[curr_idx] < df['Low'].iloc[prev_idx]
        
        # Find corresponding MACD lows (simplification: find closest MACD low to the price low)
        # We look for a MACD low within a small window of the price low
        curr_macd_val = df[macd_col].iloc[curr_idx]
        prev_macd_val = df[macd_col].iloc[prev_idx]
        
        # Strict logic: actually verify if MACD formed a Higher Low around these times
        # Here we simplify: if MACD at current Price Low > MACD at previous Price Low
        macd_higher_low = curr_macd_val > prev_macd_val
        
        if price_lower_low and macd_higher_low and curr_macd_val < 0:
            # Signal trigger is delayed by 'order' days because we need to wait to confirm it's a low
            signal_idx = curr_idx + order
            if signal_idx < len(df):
                signals.append({
                    'index': signal_idx,
                    'type': 'Bullish',
                    'date': df.index[signal_idx],
                    'price_idx': curr_idx # For plotting the anchor
                })

    # Check Bearish Divergence
    for i in range(1, len(price_highs_idx)):
        curr_idx = price_highs_idx[i]
        prev_idx = price_highs_idx[i-1]
        
        if (curr_idx - prev_idx) > lookback:
            continue
            
        # Price Higher High
        price_higher_high = df['High'].iloc[curr_idx] > df['High'].iloc[prev_idx]
        
        # MACD Lower High
        curr_macd_val = df[macd_col].iloc[curr_idx]
        prev_macd_val = df[macd_col].iloc[prev_idx]
        
        macd_lower_high = curr_macd_val < prev_macd_val
        
        if price_higher_high and macd_lower_high and curr_macd_val > 0:
            signal_idx = curr_idx + order
            if signal_idx < len(df):
                signals.append({
                    'index': signal_idx,
                    'type': 'Bearish',
                    'date': df.index[signal_idx],
                    'price_idx': curr_idx
                })
                
    return signals, df

# --- Strategy 2: Trendline Breakout ---

def strategy_trendline_breakout(df):
    """
    Detect Trendline Breakouts using rolling regression on pivots.
    """
    order = 5
    lookback_window = 50 # How far back to scan for points to fit line
    
    # Identify pivots (Highs/Lows) across the whole dataset first
    # (In a real streaming app, this would be recalculated daily, 
    # but for vectorization we find all potential pivots first)
    highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    
    signals = []
    
    # We iterate through the DF starting after enough data exists
    start_scan = lookback_window + order
    
    # Optimization: Only check days where a breakout might occur is expensive loop in python.
    # We will check every day.
    
    # To save time, we only run logic if we have enough points in the window.
    # We maintain a list of 'active' trendlines? No, let's recalculate simply.
    
    # Iterate days
    for i in range(start_scan, len(df)):
        current_date = df.index[i]
        current_close = df['Close'].iloc[i]
        
        # 1. Bearish Breakout (Price falls below Uptrend Support)
        # Find recent lows within [i - lookback, i - 1]
        recent_lows = [x for x in lows_idx if (i - lookback_window) <= x < i]
        
        if len(recent_lows) >= 2:
            # Get values
            x = np.array(recent_lows)
            y = df['Low'].iloc[recent_lows].values
            
            # Linear Regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Conditions for a valid uptrend line: Positive slope
            if slope > 0:
                # Expected support value at current day i
                support_val = slope * i + intercept
                
                # Check for breakdown (Close < Support)
                # Filter: The previous day's close was above support (to avoid repeated signals)
                prev_support_val = slope * (i-1) + intercept
                if current_close < support_val and df['Close'].iloc[i-1] >= prev_support_val:
                     signals.append({
                        'index': i,
                        'type': 'Bearish',
                        'date': current_date,
                        'slope': slope,
                        'intercept': intercept
                    })

        # 2. Bullish Breakout (Price rises above Downtrend Resistance)
        recent_highs = [x for x in highs_idx if (i - lookback_window) <= x < i]
        
        if len(recent_highs) >= 2:
            x = np.array(recent_highs)
            y = df['High'].iloc[recent_highs].values
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Condition for downtrend line: Negative slope
            if slope < 0:
                res_val = slope * i + intercept
                prev_res_val = slope * (i-1) + intercept
                
                if current_close > res_val and df['Close'].iloc[i-1] <= prev_res_val:
                    signals.append({
                        'index': i,
                        'type': 'Bullish',
                        'date': current_date,
                        'slope': slope,
                        'intercept': intercept
                    })

    return signals

# -----------------------------------------------------------------------------
# 3. 主應用程式流程 (Main App Flow)
# -----------------------------------------------------------------------------

def main():
    # --- Sidebar ---
    st.sidebar.caption("v0.5")
    lang_opt = st.sidebar.selectbox("Language / 語言", ['English', '繁體中文'])
    txt = TRANSLATIONS[lang_opt]
    
    st.sidebar.title(txt['sidebar_title'])
    
    # Ticker Selection
    default_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'TSLA', 'NVDA']
    ticker_input = st.sidebar.selectbox(txt['ticker_label'], default_tickers)
    # Allow custom input via text if not in list (Streamlit UX pattern)
    custom_ticker = st.sidebar.text_input(f"Or type custom: {txt['ticker_label']}", "")
    
    final_ticker = custom_ticker.upper().strip() if custom_ticker else ticker_input
    
    # Period Selection
    # Update: Allows selection of 1y, 2y, 3y, 4y, 5y
    period_options = ['1y', '2y', '3y', '4y', '5y']
    period = st.sidebar.selectbox(txt['period_label'], period_options, index=0)
    
    st.title(f"{txt['title']} - {final_ticker}")
    
    # --- Data Fetching ---
    with st.spinner(txt['loading']):
        df = get_stock_data(final_ticker, period)
    
    if df is None or len(df) < 50:
        st.error(txt['error_ticker'].format(symbol=final_ticker))
        return

    # --- Run Analysis ---
    macd_signals, df_macd = strategy_macd_divergence(df.copy())
    trend_signals = strategy_trendline_breakout(df.copy())
    
    macd_trades = perform_backtest(df, macd_signals)
    trend_trades = perform_backtest(df, trend_signals)
    
    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([txt['tab_macd'], txt['tab_trend'], txt['tab_summary']])
    
    # --- TAB 1: MACD ---
    with tab1:
        st.subheader(f"MACD (12,26,9) - {final_ticker}")
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Price
        ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
        ax1.set_ylabel('Price')
        ax1.set_title(txt['chart_price']) # Uses English even in ZH mode now
        ax1.grid(True, alpha=0.3)
        
        # MACD
        ax2.plot(df_macd.index, df_macd['MACD_12_26_9'], label='MACD', color='blue')
        ax2.plot(df_macd.index, df_macd['MACDs_12_26_9'], label='Signal', color='orange')
        ax2.bar(df_macd.index, df_macd['MACDh_12_26_9'], label='Hist', color='gray', alpha=0.3)
        ax2.set_title(txt['chart_macd']) # Uses English even in ZH mode now
        ax2.grid(True, alpha=0.3)
        
        # Plot Signals
        for sig in macd_signals:
            date = sig['date']
            price = df.loc[date, 'Close']
            
            if sig['type'] == 'Bullish':
                # Green Up Arrow
                ax1.plot(date, price, marker='^', color='green', markersize=10, linestyle='None')
                # Draw line connecting divergence points if needed (advanced)
            else:
                # Red Down Arrow
                ax1.plot(date, price, marker='v', color='red', markersize=10, linestyle='None')

        # Mobile Optimization: Tight layout and container width
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Table
        if not macd_trades.empty:
            # Format table columns for display
            disp_cols = {
                'Signal Date': txt['col_date'],
                'Type': txt['col_type'],
                'Entry Date': txt['col_entry_date'],
                'Entry Price': txt['col_entry_price'],
                'Exit Date': txt['col_exit_date'],
                'Exit Price': txt['col_exit_price'],
                'Return (%)': txt['col_return']
            }
            
            # Map type text
            display_df = macd_trades.copy()
            display_df['Type'] = display_df['Type'].map({'Bullish': txt['bullish'], 'Bearish': txt['bearish']})
            display_df = display_df.rename(columns=disp_cols)
            
            # Format dates and floats
            for col in [txt['col_date'], txt['col_entry_date'], txt['col_exit_date']]:
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
            
            # Sort by Date DESCENDING (Newest first)
            display_df = display_df.sort_values(by=txt['col_date'], ascending=False).reset_index(drop=True)
            
            # Calculate dynamic height: (rows + 1 header) * 35px per row approx
            table_height = (len(display_df) + 1) * 35 + 3

            st.dataframe(
                display_df.style.format({
                    txt['col_entry_price']: "{:.2f}",
                    txt['col_exit_price']: "{:.2f}",
                    txt['col_return']: "{:.2f}%"
                }),
                use_container_width=True,
                height=table_height
            )
        else:
            st.info("No signals detected.")

    # --- TAB 2: Trendline ---
    with tab2:
        st.subheader(f"Trendline Breakout - {final_ticker}")
        
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
        ax.set_title(txt['chart_price']) # Uses English even in ZH mode now
        ax.grid(True, alpha=0.3)
        
        # Plot Signals and specific trendlines associated with them
        for sig in trend_signals:
            date = sig['date']
            price = df.loc[date, 'Close']
            idx = sig['index']
            slope = sig['slope']
            intercept = sig['intercept']
            
            # Draw the trendline segment (lookback window)
            # x coords for line: (idx - 50) to (idx)
            x_vals = np.arange(idx - 50, idx + 5) # Extend slightly past signal
            # Convert integer indices back to dates for plotting is tricky in matplotlib with dates
            # Standard matplotlib dates are floats. 
            # Alternative: simpler visualization - just mark the breakout point
            
            if sig['type'] == 'Bullish':
                ax.plot(date, price, marker='^', color='green', markersize=10)
                # Visualize Trendline (approximate for visual context)
                # We calculate price points relative to the index
                y_vals = slope * x_vals + intercept
                valid_dates = [df.index[min(max(0, i), len(df)-1)] for i in x_vals]
                ax.plot(valid_dates, y_vals, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
            else:
                ax.plot(date, price, marker='v', color='red', markersize=10)
                y_vals = slope * x_vals + intercept
                valid_dates = [df.index[min(max(0, i), len(df)-1)] for i in x_vals]
                ax.plot(valid_dates, y_vals, color='green', linestyle='--', alpha=0.5, linewidth=1)

        # Mobile Optimization
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        
        if not trend_trades.empty:
            disp_cols = {
                'Signal Date': txt['col_date'],
                'Type': txt['col_type'],
                'Entry Date': txt['col_entry_date'],
                'Entry Price': txt['col_entry_price'],
                'Exit Date': txt['col_exit_date'],
                'Exit Price': txt['col_exit_price'],
                'Return (%)': txt['col_return']
            }
            display_df_t = trend_trades.copy()
            display_df_t['Type'] = display_df_t['Type'].map({'Bullish': txt['bullish'], 'Bearish': txt['bearish']})
            display_df_t = display_df_t.rename(columns=disp_cols)
            
            for col in [txt['col_date'], txt['col_entry_date'], txt['col_exit_date']]:
                display_df_t[col] = display_df_t[col].dt.strftime('%Y-%m-%d')

            # Sort by Date DESCENDING (Newest first)
            display_df_t = display_df_t.sort_values(by=txt['col_date'], ascending=False).reset_index(drop=True)

            # Calculate dynamic height
            table_height_t = (len(display_df_t) + 1) * 35 + 3

            st.dataframe(
                display_df_t.style.format({
                    txt['col_entry_price']: "{:.2f}",
                    txt['col_exit_price']: "{:.2f}",
                    txt['col_return']: "{:.2f}%"
                }),
                use_container_width=True,
                height=table_height_t
            )
        else:
            st.info("No signals detected.")

    # --- TAB 3: Summary & Reliability ---
    with tab3:
        st.subheader(txt['tab_summary'])
        
        col1, col2 = st.columns(2)
        
        # Metrics Calculation
        def get_stats(trade_df):
            if trade_df.empty: return 0, 0, 0
            total = len(trade_df)
            wins = len(trade_df[trade_df['Return (%)'] > 0])
            rate = (wins / total) * 100
            return total, wins, rate

        # MACD Stats
        m_total, m_wins, m_rate = get_stats(macd_trades)
        m_stars = calculate_star_rating(m_rate) if m_total > 0 else "N/A"
        
        # Trend Stats
        t_total, t_wins, t_rate = get_stats(trend_trades)
        t_stars = calculate_star_rating(t_rate) if t_total > 0 else "N/A"

        with col1:
            st.markdown(f"### {txt['tab_macd']}")
            st.metric(txt['total_signals'], m_total)
            st.metric(txt['win_rate'], f"{m_rate:.1f}%")
            st.metric(txt['reliability'], m_stars, help=txt['rating_tooltip'])
            
        with col2:
            st.markdown(f"### {txt['tab_trend']}")
            st.metric(txt['total_signals'], t_total)
            st.metric(txt['win_rate'], f"{t_rate:.1f}%")
            st.metric(txt['reliability'], t_stars, help=txt['rating_tooltip'])

if __name__ == "__main__":
    main()
