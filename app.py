import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
# from scipy.stats import linregress # Removed in favor of np.polyfit
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
        'error_data_length': 'Warning: Not enough data ({length} candles) to calculate all indicators. Some charts may be empty.',
        'tab_vision': 'TrendVision Pro', 
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
        'chart_macd': 'MACD Oscillator',
        'setting_trend_title': 'Trendline Strategy Settings',
        'setting_order': 'Swing Detection Order',
        'setting_window': 'Regression Window (Candles)',
        'rec_order': 'Recommended: BTC/ETH=5, SOL=4',
        'rec_window': 'Recommended: BTC/ETH=50, SOL=40',
        'vision_title': 'TrendVision Pro: Live Trend Analysis',
        'legend_res': 'Resistance Trendline',
        'legend_sup': 'Support Trendline'
    },
    '繁體中文': {
        'title': '技術分析回測應用程式',
        'sidebar_title': '參數設定',
        'ticker_label': '標的代碼 (例如 BTC-USD, AAPL)',
        'period_label': '資料期間',
        'error_ticker': '錯誤：無法取得 "{symbol}" 的資料。請檢查代碼是否正確。',
        'error_data_length': '警告：資料不足 ({length} 根 K 線) 無法計算所有指標。部分圖表可能為空。',
        'tab_vision': 'TrendVision Pro', 
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
        'chart_price': 'Price & Signals',
        'chart_macd': 'MACD Oscillator',
        'setting_trend_title': '趨勢線策略設定',
        'setting_order': '擺盪偵測範圍 (Order)',
        'setting_window': '回歸窗口 (K線數量)',
        'rec_order': '推薦: BTC/ETH=5, SOL=4',
        'rec_window': '推薦: BTC/ETH=50, SOL=40',
        'vision_title': 'TrendVision Pro: 即時趨勢分析',
        'legend_res': '阻力趨勢線 (Resistance)',
        'legend_sup': '支撐趨勢線 (Support)'
    }
}

# -----------------------------------------------------------------------------
# 2. 核心邏輯函式 (Core Logic)
# -----------------------------------------------------------------------------

@st.cache_data
def get_stock_data(ticker, period="1y"):
    """
    Download OHLCV data from yfinance.
    Supports standard periods and custom year calculations.
    """
    try:
        # yfinance standard valid periods
        standard_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        
        if period in standard_periods:
            df = yf.download(ticker, period=period, progress=False, multi_level_index=False)
        else:
            # Handle custom years like '3y', '4y' by calculating start date
            try:
                years = int(period.replace('y', ''))
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
                df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            except ValueError:
                # Fallback
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
    """
    trades = []
    
    for sig in signals:
        idx = sig['index']
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
    Includes error handling for short data periods.
    """
    # Safeguard: Check if enough data exists for MACD (26+9 = 35 candles min approx)
    if len(df) < 35:
        return [], df

    try:
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is None or macd.empty:
            return [], df
            
        df = pd.concat([df, macd], axis=1)
        macd_col = 'MACD_12_26_9'
        hist_col = 'MACDh_12_26_9' 
        
        # Check if columns actually exist after concat
        if macd_col not in df.columns:
            return [], df

        order = 5
        lookback = 20
        
        # Ensure we have enough data for extrema finding
        if len(df) <= order * 2:
             return [], df

        price_lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
        macd_lows_idx = argrelextrema(df[macd_col].values, np.less, order=order)[0]
        
        price_highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
        macd_highs_idx = argrelextrema(df[macd_col].values, np.greater, order=order)[0]
        
        signals = []
        
        # Check Bullish Divergence
        for i in range(1, len(price_lows_idx)):
            curr_idx = price_lows_idx[i]
            prev_idx = price_lows_idx[i-1]
            
            if (curr_idx - prev_idx) > lookback:
                continue
                
            price_lower_low = df['Low'].iloc[curr_idx] < df['Low'].iloc[prev_idx]
            curr_macd_val = df[macd_col].iloc[curr_idx]
            prev_macd_val = df[macd_col].iloc[prev_idx]
            macd_higher_low = curr_macd_val > prev_macd_val
            
            if price_lower_low and macd_higher_low and curr_macd_val < 0:
                signal_idx = curr_idx + order
                if signal_idx < len(df):
                    signals.append({
                        'index': signal_idx,
                        'type': 'Bullish',
                        'date': df.index[signal_idx],
                        'price_idx': curr_idx
                    })

        # Check Bearish Divergence
        for i in range(1, len(price_highs_idx)):
            curr_idx = price_highs_idx[i]
            prev_idx = price_highs_idx[i-1]
            
            if (curr_idx - prev_idx) > lookback:
                continue
                
            price_higher_high = df['High'].iloc[curr_idx] > df['High'].iloc[prev_idx]
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
        
    except Exception as e:
        # Fallback if any calculation fails
        return [], df

# --- Strategy 2: Trendline Breakout (Backtest) ---

def strategy_trendline_breakout(df, order=5, lookback_window=50):
    """
    Detect Trendline Breakouts for backtesting.
    """
    if len(df) < order * 2:
        return []

    highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    
    signals = []
    start_scan = lookback_window + order
    
    # If data is shorter than lookback window, we can't backtest this strategy effectively
    if len(df) <= start_scan:
        return []

    for i in range(start_scan, len(df)):
        current_date = df.index[i]
        current_close = df['Close'].iloc[i]
        
        # Bearish Breakdown
        recent_lows = [x for x in lows_idx if (i - lookback_window) <= x < i]
        if len(recent_lows) >= 2:
            x = np.array(recent_lows)
            y = df['Low'].iloc[recent_lows].values
            slope, intercept = np.polyfit(x, y, 1)
            
            if slope > 0:
                support_val = slope * i + intercept
                prev_support_val = slope * (i-1) + intercept
                if current_close < support_val and df['Close'].iloc[i-1] >= prev_support_val:
                     signals.append({
                        'index': i,
                        'type': 'Bearish',
                        'date': current_date,
                        'slope': slope,
                        'intercept': intercept
                    })

        # Bullish Breakout
        recent_highs = [x for x in highs_idx if (i - lookback_window) <= x < i]
        if len(recent_highs) >= 2:
            x = np.array(recent_highs)
            y = df['High'].iloc[recent_highs].values
            slope, intercept = np.polyfit(x, y, 1)
            
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

# --- New: Calculate Current Trendlines for TrendVision Pro ---

def calculate_current_trendlines(df, order=5, lookback_window=50):
    """
    Calculate the LATEST valid support and resistance trendlines.
    Gracefully handles short data.
    """
    if len(df) < order * 2:
        return None, None

    # 1. Slice the dataframe to the most recent window for analysis
    current_idx = len(df) - 1
    # If data is shorter than window, look at whole dataframe
    start_idx = max(0, current_idx - lookback_window)
    
    # Find all pivots first
    highs_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    lows_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    
    # Filter pivots that are within the current active window [start_idx, current_idx]
    valid_highs = [x for x in highs_idx if x >= start_idx]
    valid_lows = [x for x in lows_idx if x >= start_idx]
    
    res_line = None
    sup_line = None
    
    # Calculate Resistance Trendline (connect recent highs)
    if len(valid_highs) >= 2:
        x = np.array(valid_highs)
        y = df['High'].iloc[valid_highs].values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Generate line coordinates for plotting
        x_plot_idx = np.arange(valid_highs[0], len(df))
        y_plot = slope * x_plot_idx + intercept
        res_line = (df.index[x_plot_idx], y_plot)
        
    # Calculate Support Trendline (connect recent lows)
    if len(valid_lows) >= 2:
        x = np.array(valid_lows)
        y = df['Low'].iloc[valid_lows].values
        slope, intercept = np.polyfit(x, y, 1)
        
        x_plot_idx = np.arange(valid_lows[0], len(df))
        y_plot = slope * x_plot_idx + intercept
        sup_line = (df.index[x_plot_idx], y_plot)
        
    return res_line, sup_line

# -----------------------------------------------------------------------------
# 3. 主應用程式流程 (Main App Flow)
# -----------------------------------------------------------------------------

def main():
    # --- Sidebar ---
    st.sidebar.caption("v0.8.2") # Bump version
    lang_opt = st.sidebar.selectbox("Language / 語言", ['English', '繁體中文'])
    txt = TRANSLATIONS[lang_opt]
    
    st.sidebar.title(txt['sidebar_title'])
    
    # Ticker Selection
    default_tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'
    ]
    ticker_input = st.sidebar.selectbox(txt['ticker_label'], default_tickers)
    custom_ticker = st.sidebar.text_input(f"Or type custom: {txt['ticker_label']}", "")
    
    final_ticker = custom_ticker.upper().strip() if custom_ticker else ticker_input
    
    # Period Selection
    # Update: Removed 1mo
    period_options = ['3mo', '6mo', '1y', '2y', '3y', '4y', '5y']
    period = st.sidebar.selectbox(txt['period_label'], period_options, index=2) # Default 1y
    
    # --- Trendline Settings ---
    st.sidebar.markdown("---")
    st.sidebar.subheader(txt['setting_trend_title'])
    
    trend_order = st.sidebar.slider(
        txt['setting_order'], 
        min_value=3, 
        max_value=10, 
        value=5, 
        help=txt['rec_order']
    )
    
    trend_window = st.sidebar.slider(
        txt['setting_window'], 
        min_value=20, 
        max_value=100, 
        value=50, 
        help=txt['rec_window']
    )

    st.title(f"{txt['title']} - {final_ticker}")
    
    # --- Data Fetching ---
    with st.spinner(txt['loading']):
        df = get_stock_data(final_ticker, period)
    
    # Relaxed data check for 1mo period (approx 20-22 days)
    if df is None or len(df) < 5: 
        st.error(txt['error_ticker'].format(symbol=final_ticker))
        return
        
    # Warning for short data
    if len(df) < 35:
        st.warning(txt['error_data_length'].format(length=len(df)))

    # --- Run Analysis ---
    macd_signals, df_macd = strategy_macd_divergence(df.copy())
    trend_signals = strategy_trendline_breakout(df.copy(), order=trend_order, lookback_window=trend_window)
    
    macd_trades = perform_backtest(df, macd_signals)
    trend_trades = perform_backtest(df, trend_signals)
    
    # Calculate Current Trendlines for TrendVision
    res_line_curr, sup_line_curr = calculate_current_trendlines(df, order=trend_order, lookback_window=trend_window)
    
    # --- Tabs ---
    tab_vision, tab_trend, tab_macd, tab_summary = st.tabs([
        txt['tab_vision'], 
        txt['tab_trend'], 
        txt['tab_macd'], 
        txt['tab_summary']
    ])
    
    # --- TAB 1: TrendVision Pro ---
    with tab_vision:
        st.subheader(txt['vision_title'])
        
        # Plotting - Dark Style mimic
        fig_v, (ax1_v, ax2_v) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Price Chart
        ax1_v.plot(df.index, df['Close'], label='Close Price', color='white', alpha=0.8, linewidth=1.5)
        
        # Plot Resistance Trendline (Current)
        if res_line_curr:
            ax1_v.plot(res_line_curr[0], res_line_curr[1], color='red', linestyle='--', linewidth=2, label=txt['legend_res'])
            
        # Plot Support Trendline (Current)
        if sup_line_curr:
            ax1_v.plot(sup_line_curr[0], sup_line_curr[1], color='#00ff00', linestyle='--', linewidth=2, label=txt['legend_sup'])
            
        # Styling
        ax1_v.set_facecolor('#0E1117') 
        ax1_v.grid(True, color='#444444', alpha=0.3)
        ax1_v.legend(loc='upper left', facecolor='#0E1117', labelcolor='white')
        ax1_v.set_ylabel('Price', color='white')
        ax1_v.tick_params(colors='white')
        
        # MACD Chart below - Handle case where MACD data might be missing
        if 'MACD_12_26_9' in df_macd.columns:
            ax2_v.plot(df_macd.index, df_macd['MACD_12_26_9'], label='MACD', color='#2962FF')
            ax2_v.plot(df_macd.index, df_macd['MACDs_12_26_9'], label='Signal', color='#FF6D00')
            ax2_v.bar(df_macd.index, df_macd['MACDh_12_26_9'], label='Hist', color='gray', alpha=0.3)
        else:
            ax2_v.text(0.5, 0.5, "Insufficient data for MACD", color='white', ha='center')
        
        ax2_v.set_facecolor('#0E1117')
        ax2_v.grid(True, color='#444444', alpha=0.3)
        ax2_v.legend(loc='upper left', facecolor='#0E1117', labelcolor='white')
        ax2_v.tick_params(colors='white')
        
        fig_v.patch.set_facecolor('#0E1117')
        fig_v.tight_layout()
        st.pyplot(fig_v, use_container_width=True)

    # --- TAB 2: Trendline Breakout (Backtest) ---
    with tab_trend:
        st.subheader(f"Trendline Breakout - {final_ticker}")
        
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
        ax.set_title(txt['chart_price'])
        ax.grid(True, alpha=0.3)
        
        for sig in trend_signals:
            date = sig['date']
            price = df.loc[date, 'Close']
            idx = sig['index']
            slope = sig['slope']
            intercept = sig['intercept']
            
            x_vals = np.arange(idx - 50, idx + 5)
            
            if sig['type'] == 'Bullish':
                ax.plot(date, price, marker='^', color='green', markersize=10)
                y_vals = slope * x_vals + intercept
                # Validate dates
                valid_dates = [df.index[min(max(0, i), len(df)-1)] for i in x_vals]
                ax.plot(valid_dates, y_vals, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
            else:
                ax.plot(date, price, marker='v', color='red', markersize=10)
                y_vals = slope * x_vals + intercept
                valid_dates = [df.index[min(max(0, i), len(df)-1)] for i in x_vals]
                ax.plot(valid_dates, y_vals, color='green', linestyle='--', alpha=0.5, linewidth=1)

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

            display_df_t = display_df_t.sort_values(by=txt['col_date'], ascending=False).reset_index(drop=True)
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

    # --- TAB 3: MACD ---
    with tab_macd:
        st.subheader(f"MACD (12,26,9) - {final_ticker}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
        ax1.set_ylabel('Price')
        ax1.set_title(txt['chart_price'])
        ax1.grid(True, alpha=0.3)
        
        # Handle missing MACD columns
        if 'MACD_12_26_9' in df_macd.columns:
            ax2.plot(df_macd.index, df_macd['MACD_12_26_9'], label='MACD', color='blue')
            ax2.plot(df_macd.index, df_macd['MACDs_12_26_9'], label='Signal', color='orange')
            ax2.bar(df_macd.index, df_macd['MACDh_12_26_9'], label='Hist', color='gray', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for MACD", ha='center')

        ax2.set_title(txt['chart_macd'])
        ax2.grid(True, alpha=0.3)
        
        for sig in macd_signals:
            date = sig['date']
            price = df.loc[date, 'Close']
            
            if sig['type'] == 'Bullish':
                ax1.plot(date, price, marker='^', color='green', markersize=10, linestyle='None')
            else:
                ax1.plot(date, price, marker='v', color='red', markersize=10, linestyle='None')

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        if not macd_trades.empty:
            disp_cols = {
                'Signal Date': txt['col_date'],
                'Type': txt['col_type'],
                'Entry Date': txt['col_entry_date'],
                'Entry Price': txt['col_entry_price'],
                'Exit Date': txt['col_exit_date'],
                'Exit Price': txt['col_exit_price'],
                'Return (%)': txt['col_return']
            }
            
            display_df = macd_trades.copy()
            display_df['Type'] = display_df['Type'].map({'Bullish': txt['bullish'], 'Bearish': txt['bearish']})
            display_df = display_df.rename(columns=disp_cols)
            
            for col in [txt['col_date'], txt['col_entry_date'], txt['col_exit_date']]:
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
            
            display_df = display_df.sort_values(by=txt['col_date'], ascending=False).reset_index(drop=True)
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

    # --- TAB 4: Summary & Reliability ---
    with tab_summary:
        st.subheader(txt['tab_summary'])
        
        col1, col2 = st.columns(2)
        
        def get_stats(trade_df):
            if trade_df.empty: return 0, 0, 0
            total = len(trade_df)
            wins = len(trade_df[trade_df['Return (%)'] > 0])
            rate = (wins / total) * 100
            return total, wins, rate

        m_total, m_wins, m_rate = get_stats(macd_trades)
        m_stars = calculate_star_rating(m_rate) if m_total > 0 else "N/A"
        
        t_total, t_wins, t_rate = get_stats(trend_trades)
        t_stars = calculate_star_rating(t_rate) if t_total > 0 else "N/A"

        with col1:
            st.markdown(f"### {txt['tab_trend']}")
            st.metric(txt['total_signals'], t_total)
            st.metric(txt['win_rate'], f"{t_rate:.1f}%")
            st.metric(txt['reliability'], t_stars, help=txt['rating_tooltip'])
            
        with col2:
            st.markdown(f"### {txt['tab_macd']}")
            st.metric(txt['total_signals'], m_total)
            st.metric(txt['win_rate'], f"{m_rate:.1f}%")
            st.metric(txt['reliability'], m_stars, help=txt['rating_tooltip'])

if __name__ == "__main__":
    main()
