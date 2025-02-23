import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
from openai import OpenAI
#from config import OPENAI_API_KEY
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Parameters for stock data and technical indicators
TICKER = 'CAT'
START_DATE = None  # e.g., '2023-01-01'
END_DATE = None  # e.g., '2023-06-01'
PERIOD = '9mo'  # Used if start/end are None
INTERVAL = '1d'  # Data interval

BB_WINDOW = 20  # Bollinger Bands rolling window
BB_STD_DEV = 2  # Number of standard deviations for Bollinger Bands


# ---------------------------
# Helper Functions
# ---------------------------

def calculate_linear_trendline(data, window):
    trendline = []
    for i in range(len(data)):
        if i < window - 1:
            trendline.append(np.nan)
        else:
            y = data.iloc[i - window + 1:i + 1].values
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            trendline.append(model.predict(np.array([[window - 1]]))[0])
    return trendline


def compute_percent_changes(data):
    changes = {}
    last_date = data.index[-1]
    last_close = data['Close'].iloc[-1]
    milestones = {
        "1 Day": timedelta(days=1),
        "5 Days": timedelta(days=5),
        "1 Week": timedelta(days=7),
        "2 Weeks": timedelta(days=14),
        "3 Weeks": timedelta(days=21),
        "1 Month": timedelta(days=30),
        "3 Months": timedelta(days=90),
        "6 Months": timedelta(days=180),
        "1 Year": timedelta(days=365),
        "2 Years": timedelta(days=730),
        "5 Years": timedelta(days=1825),
        "All Time": None
    }
    for label, delta in milestones.items():
        if delta is not None:
            target_date = last_date - delta
            filtered = data[data.index <= target_date]
            if filtered.empty:
                changes[label] = None
            else:
                value = filtered.iloc[-1]['Close']
                percent_change = ((last_close - value) / value) * 100
                changes[label] = percent_change
        else:
            first_close = data['Close'].iloc[0]
            percent_change = ((last_close - first_close) / first_close) * 100
            changes[label] = percent_change
    return changes


def fetch_stock_data(ticker, period=PERIOD, interval=INTERVAL, start=START_DATE, end=END_DATE, retries=5, delay=10):
    cache_file = f"{ticker}_{period}_{interval}.csv"
    # Load from cache if available and recent
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not data.empty:
            return data

    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, start=start, end=end)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ["_".join(col).strip() for col in data.columns.values]
                rename_map = {}
                for col in data.columns:
                    if "Close" in col:
                        rename_map[col] = "Close"
                    elif "High" in col:
                        rename_map[col] = "High"
                    elif "Low" in col:
                        rename_map[col] = "Low"
                    elif "Open" in col:
                        rename_map[col] = "Open"
                    elif "Volume" in col:
                        rename_map[col] = "Volume"
                data.rename(columns=rename_map, inplace=True)
            if not data.empty:
                # Cache the data for future runs
                data.to_csv(cache_file)
                return data
            else:
                st.warning(f"No data returned for ticker {ticker}.")
                return None
        except Exception as e:
            st.error(f"Attempt {attempt+1} for ticker {ticker} failed: {e}")
            time.sleep(delay)
    st.error(f"Max retries exceeded for ticker {ticker}.")
    return None

data = fetch_stock_data("NVDA")

def detect_trend(data):
    close_prices = data['Close'].squeeze()
    data['SMA_50'] = ta.trend.sma_indicator(close_prices, window=50)
    data['SMA_200'] = ta.trend.sma_indicator(close_prices, window=200)
    if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
        return "📈 Uptrend"
    elif data['SMA_50'].iloc[-1] < data['SMA_200'].iloc[-1]:
        return "📉 Downtrend"
    else:
        return "➡️ Sideways Market"


def detect_patterns(data):
    data.loc[:, 'min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order=5)[0]]['Close'].copy()
    data.loc[:, 'max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order=5)[0]]['Close'].copy()
    data['bb_middle'] = data['Close'].rolling(window=BB_WINDOW).mean()
    data['bb_std'] = data['Close'].rolling(window=BB_WINDOW).std()
    data['bb_upper'] = data['bb_middle'] + BB_STD_DEV * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - BB_STD_DEV * data['bb_std']

    min_values = data['min'].dropna().values
    max_values = data['max'].dropna().values

    if len(min_values) > 2 and np.allclose(min_values[-2:], min_values[-1], atol=0.02):
        return "✅ Double Bottom (Bullish Reversal)"
    if len(max_values) > 2 and np.allclose(max_values[-2:], max_values[-1], atol=0.02):
        return "⚠️ Double Top (Bearish Reversal)"

    last_close = float(data['Close'].iloc[-1])
    max_series = data['max'].dropna()
    if not max_series.empty:
        max_value = float(max_series.max())
        if last_close > max_value:
            return "🚀 Breakout (Bullish Continuation)"
    if last_close > data['bb_upper'].iloc[-1]:
        return "🚀 Breakout above Upper Bollinger Band (Bullish)"
    elif last_close < data['bb_lower'].iloc[-1]:
        return "🔻 Breakdown below Lower Bollinger Band (Bearish)"
    if (last_close < data['SMA_50'].iloc[-1]) and (last_close > data['SMA_200'].iloc[-1]):
        return "📏 Triangle Formation (Breakout Possible)"
    return "📊 No Clear Pattern Detected"


def calculate_support_resistance(data):
    if 'min' not in data.columns or 'max' not in data.columns:
        data.loc[:, 'min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order=5)[0]]['Close'].copy()
        data.loc[:, 'max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order=5)[0]][
            'Close'].copy()
    local_mins = data['min'].dropna()
    local_maxs = data['max'].dropna()
    support = local_mins.mean() if not local_mins.empty else None
    resistance = local_maxs.mean() if not local_maxs.empty else None
    return support, resistance


# ---------------------------
# ChatGPT Query Functions
# ---------------------------

def get_additional_insights(trend, support, resistance, pattern):
    prompt = (
        f"The current analysis shows a trend of '{trend}', a support level of {support:.2f}, and a resistance level of {resistance:.2f}. "
        f"The detected pattern is '{pattern}'.\n"
        "Provide a brief summary of the trading signals indicated by these metrics."
    )
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        insights = response.choices[0].message.content
        return insights
    except Exception as e:
        st.error(f"Error fetching insights: {e}")
        return "No additional insights available."


def get_chart_interpretation(symbol, data, trend, pattern, support, resistance, percent_changes):
    prompt = (
        f"You are an experienced stock trader with over 20 years of experience. Analyze the technical state of {symbol} "
        "using the following details:\n\n"
        f"- Trend: {trend}\n"
        f"- Pattern: {pattern}\n"
        f"- 50-day SMA: {data['SMA_50'].iloc[-1]:.2f}\n"
        f"- 200-day SMA: {data['SMA_200'].iloc[-1]:.2f}\n"
        f"- MACD: {data['MACD'].iloc[-1]:.2f}\n"
        f"- RSI: {data['RSI'].iloc[-1]:.2f}\n"
        f"- Support Level: {support:.2f}\n"
        f"- Resistance Level: {resistance:.2f}\n"
        f"- 1 Month Percentage Change: {percent_changes['1 Month']:.2f}%\n\n"
        "Based on these metrics, please provide a detailed interpretation of the current market sentiment, "
        "potential short-term trends, and key trading signals. In your analysis, emphasize the implications of the "
        "identified trend and pattern on future price movements and any potential breakout or reversal scenarios."
    )
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        interpretation = response.choices[0].message.content
        return interpretation
    except Exception as e:
        st.error(f"Error fetching chart interpretation: {e}")
        return "No chart interpretation available."

def get_options_strategy(symbol, trend, pattern, support, resistance, investment_levels=[1000, 2000, 3000, 4000, 5000]):
    ticker_obj = yf.Ticker(symbol)
    expirations = ticker_obj.options
    if expirations:
        nearest_exp = expirations[0]
        opt_chain = ticker_obj.option_chain(nearest_exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        avg_iv_calls = calls[
            'impliedVolatility'].mean() if 'impliedVolatility' in calls.columns and not calls.empty else None
        avg_iv_puts = puts[
            'impliedVolatility'].mean() if 'impliedVolatility' in puts.columns and not puts.empty else None
        options_summary = f"Nearest expiration date: {nearest_exp}. "
        if avg_iv_calls is not None:
            options_summary += f"Avg. implied volatility (calls): {avg_iv_calls:.2f}. "
        if avg_iv_puts is not None:
            options_summary += f"Avg. implied volatility (puts): {avg_iv_puts:.2f}."
    else:
        options_summary = "No options chain data available."

    investment_str = ", ".join([f"${lvl}" for lvl in investment_levels])
    prompt = (
        f"You are a seasoned options trader with over 20 years of experience. Based on the technical analysis for {symbol}:\n"
        f"- Trend: {trend}\n"
        f"- Pattern: {pattern}\n"
        f"- Support Level: {support:.2f}\n"
        f"- Resistance Level: {resistance:.2f}\n"
        f"and the following options chain summary:\n{options_summary}\n\n"
        f"Provide explicit options trading strategy recommendations for the following investment amounts: {investment_str}.\n"
        f"For each level, detail an options trade strategy (calls, puts, or spreads) including entry and exit criteria, "
        f"risk analysis, and estimated potential profit (assuming full investment is used)."
    )
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        strategies = response.choices[0].message.content
        return strategies
    except Exception as e:
        st.error(f"Error fetching options strategy: {e}")
        return "No options strategy suggestions available."


def get_equity_strategies(symbol, trend, pattern, support, resistance):
    prompt = (
        f"You are a veteran stock trader with over 20 years of experience. Based on the current technical analysis of {symbol}:\n"
        f"- Trend: {trend}\n"
        f"- Pattern: {pattern}\n"
        f"- Support Level: {support:.2f}\n"
        f"- Resistance Level: {resistance:.2f}\n\n"
        f"Please provide detailed equity trading strategies for the following scenarios:\n"
        f"1. **Long Position Strategy:** Describe ideal entry criteria, exit strategy (including stop-loss and profit targets), "
        f"risk management, and expected profit.\n"
        f"2. **Short Position Strategy:** Explain the conditions for initiating a short position, including entry, stop-loss, "
        f"and profit targets.\n"
        f"3. **Swing Trade Strategy:** Outline the criteria for capturing short-to-medium term price swings, including entry and exit points, "
        f"risk analysis, and recommended holding timeframes.\n\n"
        f"Provide a detailed, trader-style explanation for each strategy."
    )
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        equity_strategies = response.choices[0].message.content
        return equity_strategies
    except Exception as e:
        st.error(f"Error fetching equity strategies: {e}")
        return "No equity strategy suggestions available."


# ---------------------------
# New Function: Options Chain Table Legend
# ---------------------------
def display_options_legend():
    legend = {
        "contractSymbol": "Unique identifier for the options contract.",
        "lastTradeDate": "The date and time of the last trade.",
        "strike": "The strike price of the option.",
        "lastPrice": "The last traded price of the option.",
        "bid": "The current bid price.",
        "ask": "The current ask price.",
        "change": "Price change since the previous close.",
        "percentChange": "Percentage change in price since the previous close.",
        "volume": "Number of contracts traded.",
        "openInterest": "Number of open contracts.",
        "impliedVolatility": "Expected volatility from option pricing models."
    }
    st.markdown("### Options Table Legend")
    for col, desc in legend.items():
        st.write(f"**{col}**: {desc}")


# ---------------------------
# New Function: Options Chain Table
# ---------------------------
def create_options_chain_table(symbol, support, resistance):
    ticker_obj = yf.Ticker(symbol)
    expirations = ticker_obj.options
    if not expirations:
        st.error("No options chain data available.")
        return None, None
    nearest_exp = expirations[0]
    opt_chain = ticker_obj.option_chain(nearest_exp)
    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()

    # Get current stock price
    current_price = ticker_obj.info.get("regularMarketPrice", None)
    if current_price is None:
        current_price = calls['lastPrice'].iloc[0]  # fallback if needed

    # Define a highlighting function for the "strike" column
    def highlight_strike(val):
        tol = 0.05  # 5% tolerance
        if pd.isna(val):
            return ""
        style = ""
        if abs(val - current_price) / current_price < tol:
            style = "background-color: yellow"
        elif support is not None and abs(val - support) / support < tol:
            style = "background-color: lightgreen"
        elif resistance is not None and abs(val - resistance) / resistance < tol:
            style = "background-color: lightcoral"
        return style

    # Apply styling to the 'strike' column for both calls and puts
    calls_styled = calls.style.applymap(highlight_strike, subset=["strike"])
    puts_styled = puts.style.applymap(highlight_strike, subset=["strike"])

    return calls_styled, puts_styled


# ---------------------------
# New Function: Create an interactive Plotly chart for equity technical analysis
# ---------------------------
def create_interactive_chart(data, support=None, resistance=None):
    fig = sp.make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            f"{TICKER} Price, MAs & Bollinger Bands",
            "MACD & Signal Line",
            "RSI (Relative Strength Index)",
            "Volume & Volume Trendline"
        )
    )
    # Row 1: Price, MAs, Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
                             name=f'{TICKER} Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['20_MA'], mode='lines',
                             name='20-Day MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['50_MA'], mode='lines',
                             name='50-Day MA', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines',
                             name='Upper Bollinger Band', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines',
                             name='Lower Bollinger Band', line=dict(color='red', dash='dash')), row=1, col=1)

    # Row 2: MACD and Signal Line
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines',
                             name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines',
                             name='Signal Line', line=dict(color='red')), row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines',
                             name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[70, 70],
                             mode='lines', name='Overbought', line=dict(color='grey', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[30, 30],
                             mode='lines', name='Oversold', line=dict(color='grey', dash='dash')), row=3, col=1)

    # Row 4: Volume and Trendlines
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                         marker=dict(color='grey'), opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['20_EMA_Volume'], mode='lines',
                             name='20-Day EMA Volume', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_Trendline'], mode='lines',
                             name='Volume Trendline', line=dict(color='red', dash='dash')), row=4, col=1)

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b %d",
        tickangle=-45,
        showgrid=True,
        ticklabelmode="period"
    )
    fig.update_layout(
        height=1200,
        title=f'{TICKER} Analysis',
        showlegend=True,
        xaxis_title='Date',
        yaxis_title='Value',
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig


# ---------------------------
# New Function: Create an interactive options chain chart
# ---------------------------
def create_options_chart(symbol, support, resistance):
    ticker_obj = yf.Ticker(symbol)
    expirations = ticker_obj.options
    if not expirations:
        st.error("No options chain data available.")
        return None
    nearest_exp = expirations[0]
    opt_chain = ticker_obj.option_chain(nearest_exp)
    calls = opt_chain.calls
    puts = opt_chain.puts
    current_price = ticker_obj.info.get("regularMarketPrice", None)
    if current_price is None:
        current_price = calls['lastPrice'].iloc[0]

    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Calls Options", "Puts Options"))

    fig.add_trace(go.Scatter(
        x=calls['strike'],
        y=calls['lastPrice'],
        mode='markers',
        name='Call Last Price',
        marker=dict(color='green', size=8)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=puts['strike'],
        y=puts['lastPrice'],
        mode='markers',
        name='Put Last Price',
        marker=dict(color='red', size=8)
    ), row=2, col=1)

    for row in [1, 2]:
        if current_price is not None:
            fig.add_vline(x=current_price, line=dict(color='blue', dash='dash'),
                          annotation_text="Current Price", row=row, col=1)
        if support is not None:
            fig.add_vline(x=support, line=dict(color='orange', dash='dot'),
                          annotation_text="Support", row=row, col=1)
        if resistance is not None:
            fig.add_vline(x=resistance, line=dict(color='purple', dash='dot'),
                          annotation_text="Resistance", row=row, col=1)

    fig.update_layout(
        title=f"Options Chain for {symbol} (Expiration: {nearest_exp})",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=800
    )
    return fig


# ---------------------------
# New Function: Company Financial Analysis
# ---------------------------
def get_financial_analysis(symbol):
    """
    Retrieves key financial metrics for the company using yfinance and then queries ChatGPT
    for a comprehensive fundamental analysis similar to what Warren Buffett might use.
    """
    ticker_obj = yf.Ticker(symbol)
    info = ticker_obj.info

    # Extract key financial ratios and metrics
    financials = {
        "Current Price": info.get("regularMarketPrice", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "Price-to-Book": info.get("priceToBook", "N/A"),
        "PEG Ratio": info.get("pegRatio", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "Return on Equity": info.get("returnOnEquity", "N/A"),
        "Total Revenue": info.get("totalRevenue", "N/A"),
        "Gross Profits": info.get("grossProfits", "N/A")
    }
    fin_df = pd.DataFrame(financials, index=[symbol])

    # Build a prompt with the financial data
    prompt = (
        f"You are a highly experienced investor similar to Warren Buffett. Analyze the following financial metrics for {symbol}:\n\n"
        f"{fin_df.to_string()}\n\n"
        "For each metric, explain its significance, what the number means for the company's financial health, and how it impacts the company's long-term investment potential."
    )
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        analysis = response.choices[0].message.content
        return fin_df, analysis
    except Exception as e:
        st.error(f"Error fetching financial analysis: {e}")
        return fin_df, "No financial analysis available."


# ---------------------------
# Streamlit UI
# ---------------------------
st.title(f"Analysis for {TICKER}")

data = fetch_stock_data(TICKER)
if data is not None:
    # Calculate technical indicators for equity chart
    data['20_MA'] = data['Close'].rolling(window=20).mean()
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['20_MA'] + (std_20 * 2)
    data['Lower_Band'] = data['20_MA'] - (std_20 * 2)
    data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['12_EMA'] - data['26_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['20_EMA_Volume'] = data['Volume'].ewm(span=20, adjust=False).mean()
    window = 20
    data['Volume_Trendline'] = calculate_linear_trendline(data['Volume'], window)

    # Compute percentage changes over key milestones
    percent_changes = compute_percent_changes(data)

    # Get technical analysis and interpretation outputs from ChatGPT
    trend = detect_trend(data)
    pattern = detect_patterns(data)
    support, resistance = calculate_support_resistance(data)
    insights = get_additional_insights(trend, support, resistance, pattern)
    interpretation = get_chart_interpretation(TICKER, data, trend, pattern, support, resistance, percent_changes)
    options_strategy = get_options_strategy(TICKER, trend, pattern, support, resistance)
    equity_strategies = get_equity_strategies(TICKER, trend, pattern, support, resistance)

    st.subheader("Technical Analysis")
    st.write(f"**Trend:** {trend}")
    st.write(f"**Pattern:** {pattern}")
    if support is not None:
        st.write(f"**Support Level:** {support:.2f}")
    if resistance is not None:
        st.write(f"**Resistance Level:** {resistance:.2f}")

    st.subheader("Percentage Change Over Key Milestones")
    for milestone, change in percent_changes.items():
        if change is not None:
            st.write(f"**{milestone}:** {change:.2f}%")
        else:
            st.write(f"**{milestone}:** Data not available")

    st.subheader("Additional Insights")
    st.write(insights)

    st.subheader("Chart Interpretation")
    st.markdown(
        f"<div style='font-size:18px; line-height:1.6;'>{interpretation.replace('\n', '<br>')}</div>",
        unsafe_allow_html=True
    )

    st.subheader("Equity Trading Strategy Recommendations")
    # Optionally, clean the text if there are any unwanted characters:
    formatted_equity_strategies = equity_strategies.replace("*", "").replace("∗", "").replace("\n", "<br>")
    st.markdown(
        f"<div style='font-size:18px; line-height:1.6;'>{formatted_equity_strategies}</div>",
        unsafe_allow_html=True
    )

    st.subheader("Options Trading Strategy Recommendations")
    formatted_options_strategy = options_strategy.replace("*", "").replace("∗", "").replace("\n", "<br>")
    st.markdown(
        f"<div style='font-size:18px; line-height:1.6;'>{formatted_options_strategy}</div>",
        unsafe_allow_html=True
    )

    st.subheader("Interactive Equity Chart")
    interactive_fig = create_interactive_chart(data, support, resistance)
    st.plotly_chart(interactive_fig, use_container_width=True)

    st.subheader("Interactive Options Chain Chart")
    options_fig = create_options_chart(TICKER, support, resistance)
    if options_fig is not None:
        st.plotly_chart(options_fig, use_container_width=True)

    st.subheader("Options Chain Table")
    calls_table, puts_table = create_options_chain_table(TICKER, support, resistance)
    if calls_table is not None:
        st.write("### Calls")
        st.dataframe(calls_table)
    if puts_table is not None:
        st.write("### Puts")
        st.dataframe(puts_table)

    st.subheader("Options Table Legend")
    display_options_legend()

    st.subheader("Company Financial Analysis")
    fin_df, fin_analysis = get_financial_analysis(TICKER)
    st.write("### Key Financial Ratios and Metrics")
    st.dataframe(fin_df)
    st.markdown("### Financial Analysis Interpretation")
    st.markdown(
        f"<div style='font-size:18px; line-height:1.6;'>{fin_analysis.replace('\n', '<br>')}</div>",
        unsafe_allow_html=True
    )

else:
    st.error("No data available to display.")

def create_pdf_report(output_text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    textobject = c.beginText(40, height - 40)
    textobject.setFont("Helvetica", 12)
    for line in output_text.splitlines():
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()

create_pdf_report(interpretation, "stock_analysis.pdf")