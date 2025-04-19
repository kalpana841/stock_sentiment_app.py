import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
import requests
import json
import re
import random
from typing import List, Dict, Tuple, Optional
import time

# Set page configuration
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-positive {
        color: #4CAF50;
        font-weight: 600;
    }
    .metric-negative {
        color: #F44336;
        font-weight: 600;
    }
    .metric-neutral {
        color: #FF9800;
        font-weight: 600;
    }
    .ticker-symbol {
        font-weight: 700;
        font-size: 1.2rem;
    }
    .news-item {
        padding: 0.75rem;
        border-bottom: 1px solid #f0f0f0;
    }
    .news-source {
        color: #666;
        font-size: 0.8rem;
    }
    .news-time {
        color: #999;
        font-size: 0.7rem;
    }
    .sentiment-positive {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
    }
    .sentiment-negative {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 3px solid #F44336;
    }
    .sentiment-neutral {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 3px solid #FF9800;
    }
    .recommendation-buy {
        background-color: rgba(76, 175, 80, 0.2);
        color: #1B5E20;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .recommendation-sell {
        background-color: rgba(244, 67, 54, 0.2);
        color: #B71C1C;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .recommendation-hold {
        background-color: rgba(255, 152, 0, 0.2);
        color: #E65100;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


class SentimentAnalyzer:
    """Class to analyze sentiment from text data"""
    
    def __init__(self):
        self.positive_words = [
            "up", "gain", "profit", "growth", "bullish", "positive", 
            "increase", "higher", "beat", "exceed", "strong", "opportunity",
            "outperform", "upgrade", "buy", "recommend", "optimistic"
        ]
        self.negative_words = [
            "down", "loss", "bearish", "negative", "decrease", "lower", 
            "miss", "fail", "risk", "concern", "weak", "trouble",
            "underperform", "downgrade", "sell", "avoid", "pessimistic"
        ]
    
    def analyze_text(self, text: str, ticker: Optional[str] = None) -> Dict:
        """
        Analyze sentiment of text using TextBlob and keyword matching
        
        Args:
            text: The text to analyze
            ticker: Optional stock ticker symbol
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Clean the text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Use TextBlob for sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        # Determine sentiment category
        if polarity > 0.2:
            sentiment = "positive"
        elif polarity < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Calculate score (0-100)
        score = 50 + (polarity * 50)
        score = max(0, min(100, score))
        
        # Extract keywords
        keywords = [word for word in text.split() 
                   if word in self.positive_words or word in self.negative_words]
        
        return {
            "sentiment": sentiment,
            "score": score,
            "polarity": polarity,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "keywords": keywords,
            "ticker": ticker,
            "summary": f"The text shows {sentiment} sentiment with a score of {score:.1f}/100."
        }


class StockDataFetcher:
    """Class to fetch and process stock data"""
    
    def __init__(self):
        self.cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch stock data for a given symbol and period
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock data
        """
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                # If no data, generate mock data
                data = self._generate_mock_data(symbol, period)
            else:
                # Add some additional columns
                data['Symbol'] = symbol
                data['Return'] = data['Close'].pct_change() * 100
                
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            # Generate mock data if fetching fails
            mock_data = self._generate_mock_data(symbol, period)
            self.cache[cache_key] = mock_data
            return mock_data
    
    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Generate mock stock data when real data can't be fetched"""
        # Determine base price based on symbol
        base_prices = {
            "AAPL": 170, "MSFT": 330, "GOOGL": 140, 
            "AMZN": 130, "TSLA": 240
        }
        base_price = base_prices.get(symbol, 100)
        
        # Determine number of days based on period
        days_map = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365, "ytd": datetime.now().timetuple().tm_yday
        }
        days = days_map.get(period, 30)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price data with some randomness
        np.random.seed(sum(ord(c) for c in symbol))  # Seed based on symbol for consistency
        
        # Create a random walk for prices
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove the initial base price
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': [p * (0.99 + 0.02 * np.random.random()) for p in prices],
            'High': [p * (1 + 0.02 * np.random.random()) for p in prices],
            'Low': [p * (0.98 + 0.02 * np.random.random()) for p in prices],
            'Close': prices,
            'Volume': [int(1e6 * (0.5 + np.random.random())) for _ in prices],
            'Symbol': symbol
        }, index=dates[:len(prices)])
        
        data['Return'] = data['Close'].pct_change() * 100
        
        return data
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information for a ticker symbol"""
        companies = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
            "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Content & Information"},
            "AMZN": {"name": "Amazon.com, Inc.", "sector": "Consumer Cyclical", "industry": "Internet Retail"},
            "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            "META": {"name": "Meta Platforms, Inc.", "sector": "Technology", "industry": "Internet Content & Information"},
            "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors"},
            "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks"},
            "V": {"name": "Visa Inc.", "sector": "Financial Services", "industry": "Credit Services"},
            "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive", "industry": "Discount Stores"},
            "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer Defensive", "industry": "Household Products"},
            "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Healthcare", "industry": "Healthcare Plans"},
            "HD": {"name": "Home Depot Inc.", "sector": "Consumer Cyclical", "industry": "Home Improvement Retail"},
            "BAC": {"name": "Bank of America Corp.", "sector": "Financial Services", "industry": "Banks"},
            "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            "DIS": {"name": "Walt Disney Co.", "sector": "Communication Services", "industry": "Entertainment"},
            "VZ": {"name": "Verizon Communications Inc.", "sector": "Communication Services", "industry": "Telecom Services"},
            "CSCO": {"name": "Cisco Systems Inc.", "sector": "Technology", "industry": "Communication Equipment"},
            "ADBE": {"name": "Adobe Inc.", "sector": "Technology", "industry": "Software"},
            "CRM": {"name": "Salesforce Inc.", "sector": "Technology", "industry": "Software"},
            "NFLX": {"name": "Netflix Inc.", "sector": "Communication Services", "industry": "Entertainment"},
            "INTC": {"name": "Intel Corporation", "sector": "Technology", "industry": "Semiconductors"},
            "AMD": {"name": "Advanced Micro Devices Inc.", "sector": "Technology", "industry": "Semiconductors"},
            "PYPL": {"name": "PayPal Holdings Inc.", "sector": "Financial Services", "industry": "Credit Services"},
            "SBUX": {"name": "Starbucks Corporation", "sector": "Consumer Cyclical", "industry": "Restaurants"},
            "NKE": {"name": "Nike Inc.", "sector": "Consumer Cyclical", "industry": "Footwear & Accessories"},
            "SPY": {"name": "SPDR S&P 500 ETF", "sector": "ETF", "industry": "Index Fund"},
            "QQQ": {"name": "Invesco QQQ Trust", "sector": "ETF", "industry": "Index Fund"},
            "IWM": {"name": "iShares Russell 2000 ETF", "sector": "ETF", "industry": "Index Fund"}
        }
        
        if symbol in companies:
            return companies[symbol]
        
        try:
            # Try to get real info from Yahoo Finance
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                "name": info.get("longName", f"{symbol} Inc."),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown")
            }
        except:
            # Return generic info if real info can't be fetched
            return {
                "name": f"{symbol} Inc.",
                "sector": "Unknown",
                "industry": "Unknown"
            }


class NewsFeedGenerator:
    """Class to generate news and social media data"""
    
    def __init__(self):
        self.sources = {
            "news": ["Bloomberg", "CNBC", "Reuters", "Wall Street Journal", "Financial Times", 
                    "MarketWatch", "Barron's", "The Economist", "Forbes", "Business Insider"],
            "twitter": ["Jim Cramer", "Stock Guru", "Trading Expert", "Market Analyst", "Wall Street Whiz",
                       "Investing Pro", "Finance Insider", "Market Maven", "Stock Whisperer", "Trading Shark"],
            "reddit": ["r/investing", "r/stocks", "r/wallstreetbets", "r/finance", "r/SecurityAnalysis",
                      "r/StockMarket", "r/options", "r/dividends", "r/ValueInvesting", "r/algotrading"]
        }
        
        self.templates = {
            "positive": [
                "{company} reports record quarterly revenue, exceeding analyst expectations by {percent}%.",
                "{company} announces new product line, stock jumps {percent}% in pre-market trading.",
                "Analysts upgrade {company} citing strong growth prospects and raise price target to ${price}.",
                "{company} expands into new markets, investors react positively pushing shares up {percent}%.",
                "{company} beats earnings estimates by ${amount} per share, raises full-year guidance.",
                "{company} announces {amount}% dividend increase, signaling confidence in future performance.",
                "Institutional investors increase stakes in {company} ahead of expected product launch.",
                "{company} secures major partnership with {partner}, shares rally on the news.",
                "Hedge fund manager predicts {company} could see {percent}% upside in the next quarter.",
                "{company}'s CEO purchases ${amount}M in company stock, showing insider confidence."
            ],
            "negative": [
                "{company} misses revenue targets by {percent}%, stock falls in after-hours trading.",
                "Analysts downgrade {company} citing competitive pressures and price target cut to ${price}.",
                "{company} announces restructuring plan, including {amount} job cuts across divisions.",
                "Regulatory concerns weigh on {company} as investigation into {issue} continues.",
                "{company} lowers guidance for next quarter, citing supply chain issues and inflation.",
                "Short seller report targets {company}, questioning {issue} and accounting practices.",
                "{company} faces lawsuit over {issue}, potentially impacting future earnings.",
                "Insider selling at {company} raises concerns as executives offload ${amount}M in shares.",
                "{company}'s market share drops {percent}% as competitors gain ground in key segments.",
                "Rising costs pressure {company}'s margins, profitability expected to decline next quarter."
            ],
            "neutral": [
                "{company} reports quarterly results in line with expectations, stock trades flat.",
                "{company} announces leadership transition plan as CEO prepares for retirement.",
                "Industry report shows {company} maintaining market share despite competitive environment.",
                "{company} to present at upcoming investor conference on {date}.",
                "{company} completes previously announced acquisition of {target} for ${amount}B.",
                "Analysts maintain neutral rating on {company} with price target of ${price}.",
                "{company} introduces minor updates to product line, no significant impact expected.",
                "Regulatory review of {company}'s proposed merger continues, decision expected by {date}.",
                "{company} maintains dividend at ${amount} per share, in line with previous quarters.",
                "{company} reaffirms full-year guidance, sees no major changes to business outlook."
            ]
        }
        
        self.companies = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan",
            "V": "Visa",
            "WMT": "Walmart",
            "JNJ": "Johnson & Johnson",
            "PG": "Procter & Gamble",
            "UNH": "UnitedHealth",
            "HD": "Home Depot",
            "BAC": "Bank of America",
            "PFE": "Pfizer",
            "DIS": "Disney",
            "VZ": "Verizon",
            "CSCO": "Cisco",
            "ADBE": "Adobe",
            "CRM": "Salesforce",
            "NFLX": "Netflix",
            "INTC": "Intel",
            "AMD": "AMD",
            "PYPL": "PayPal",
            "SBUX": "Starbucks",
            "NKE": "Nike"
        }
    
    def generate_feed(self, ticker: Optional[str] = None, count: int = 10) -> List[Dict]:
        """Generate a news and social media feed"""
        feed = []
        
        # If ticker is specified, focus on that ticker
        if ticker:
            tickers = [ticker]
        else:
            # Otherwise use a mix of popular tickers
            tickers = list(self.companies.keys())
        
        for _ in range(count):
            # Select random type, source, sentiment, and ticker
            feed_type = random.choice(list(self.sources.keys()))
            source = random.choice(self.sources[feed_type])
            sentiment = random.choice(["positive", "negative", "neutral"])
            
            if ticker:
                selected_ticker = ticker
            else:
                selected_ticker = random.choice(tickers)
                
            company = self.companies.get(selected_ticker, selected_ticker)
            
            # Select and format content template
            template = random.choice(self.templates[sentiment])
            
            # Generate random values for template placeholders
            percent = random.randint(5, 25)
            price = random.randint(50, 500)
            amount = random.randint(1, 10)
            partner = random.choice(list(self.companies.values()))
            target = random.choice(list(self.companies.values()))
            issue = random.choice(["privacy concerns", "product defects", "accounting irregularities", 
                                  "regulatory compliance", "data breach", "antitrust issues"])
            date = (datetime.now() + timedelta(days=random.randint(10, 60))).strftime("%B %d")
            
            # Format the content
            content = template.format(
                company=company,
                percent=percent,
                price=price,
                amount=amount,
                partner=partner,
                target=target,
                issue=issue,
                date=date
            )
            
            # Generate random time
            hours_ago = random.randint(0, 24)
            minutes_ago = random.randint(0, 59)
            if hours_ago == 0:
                time_str = f"{minutes_ago} minutes ago"
            else:
                time_str = f"{hours_ago} hours ago"
            
            feed.append({
                "type": feed_type,
                "source": source,
                "time": time_str,
                "content": content,
                "sentiment": sentiment,
                "ticker": selected_ticker
            })
        
        # Sort by recency
        feed.sort(key=lambda x: (
            int(x["time"].split()[0]) if "hour" in x["time"] else 0,
            int(x["time"].split()[0]) if "minute" in x["time"] else 0
        ))
        
        return feed
    
    def try_get_real_news(self, ticker: str, count: int = 5) -> List[Dict]:
        """Try to get real news for a ticker using a news API"""
        # This would use a real news API in production
        # For now, return realistic mock data
        return self.generate_feed(ticker, count)


class SentimentTrader:
    """Class to analyze sentiment and generate trading signals"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stock_fetcher = StockDataFetcher()
        self.news_generator = NewsFeedGenerator()
        
    def analyze_stock_sentiment(self, ticker: str, period: str = "1mo") -> Dict:
        """
        Analyze sentiment for a specific stock
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for analysis
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Get stock data
        stock_data = self.stock_fetcher.get_stock_data(ticker, period)
        
        # Get company info
        company_info = self.stock_fetcher.get_company_info(ticker)
        
        # Generate news feed for this ticker
        news_feed = self.news_generator.generate_feed(ticker, 20)
        
        # Analyze sentiment for each news item
        for item in news_feed:
            analysis = self.sentiment_analyzer.analyze_text(item["content"], ticker)
            item["sentiment_score"] = analysis["score"]
            item["keywords"] = analysis["keywords"]
        
        # Calculate sentiment scores
        sentiment_scores = [item["sentiment_score"] for item in news_feed]
        
        # Calculate overall sentiment metrics
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            positive_pct = len([s for s in sentiment_scores if s >= 60]) / len(sentiment_scores) * 100
            neutral_pct = len([s for s in sentiment_scores if 40 < s < 60]) / len(sentiment_scores) * 100
            negative_pct = len([s for s in sentiment_scores if s <= 40]) / len(sentiment_scores) * 100
        else:
            avg_score = 50
            positive_pct = neutral_pct = negative_pct = 33.33
        
        # Determine recommendation based on sentiment score
        if avg_score >= 75:
            recommendation = "Strong Buy"
        elif avg_score >= 60:
            recommendation = "Buy"
        elif avg_score >= 40:
            recommendation = "Hold"
        elif avg_score >= 25:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        # Calculate sentiment change (mock data)
        sentiment_change = random.uniform(-15, 15)
        
        # Generate sentiment data aligned with stock data dates
        sentiment_data = self._generate_sentiment_data(stock_data, avg_score)
        
        # Calculate correlation between sentiment and price
        if len(sentiment_data) > 5 and not stock_data.empty:
            price_changes = stock_data['Close'].pct_change().dropna().values
            sentiment_changes = pd.Series(sentiment_data).diff().dropna().values
            
            if len(price_changes) > 5 and len(sentiment_changes) > 5:
                min_len = min(len(price_changes), len(sentiment_changes))
                correlation = np.corrcoef(price_changes[:min_len], sentiment_changes[:min_len])[0, 1]
            else:
                correlation = 0
        else:
            correlation = 0
        
        return {
            "ticker": ticker,
            "company_name": company_info["name"],
            "sector": company_info["sector"],
            "industry": company_info["industry"],
            "sentiment_score": avg_score,
            "sentiment_change": sentiment_change,
            "positive_pct": positive_pct,
            "neutral_pct": neutral_pct,
            "negative_pct": negative_pct,
            "recommendation": recommendation,
            "news_feed": news_feed,
            "price_data": stock_data,
            "sentiment_data": sentiment_data,
            "price_sentiment_correlation": correlation
        }
    
    def _generate_sentiment_data(self, stock_data: pd.DataFrame, base_sentiment: float) -> List[float]:
        """Generate sentiment data aligned with stock data dates"""
        if stock_data.empty:
            return []
        
        # Create sentiment data with some correlation to price movements
        returns = stock_data['Return'].fillna(0).values
        sentiment_data = []
        
        # Base sentiment around the overall sentiment score
        sentiment_base = base_sentiment
        
        for ret in returns:
            # Sentiment is partially correlated with returns, plus some noise
            sentiment = sentiment_base + ret * 0.5 + np.random.normal(0, 5)
            sentiment = max(0, min(100, sentiment))  # Clamp between 0 and 100
            sentiment_data.append(sentiment)
            
            # Allow sentiment to drift slightly
            sentiment_base = 0.95 * sentiment_base + 0.05 * sentiment
        
        return sentiment_data
    
    def analyze_market_sentiment(self) -> Dict:
        """Analyze overall market sentiment"""
        # Get sentiment for major indices/stocks
        tickers = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        sentiment_data = {}
        
        for ticker in tickers:
            analysis = self.analyze_stock_sentiment(ticker, "5d")
            sentiment_data[ticker] = analysis["sentiment_score"]
        
        # Calculate market sentiment (weighted average)
        weights = {"SPY": 0.3, "QQQ": 0.2, "AAPL": 0.1, "MSFT": 0.1, 
                  "GOOGL": 0.1, "AMZN": 0.1, "TSLA": 0.1}
        
        market_score = sum(sentiment_data[t] * weights[t] for t in tickers)
        
        # Determine market mood
        if market_score >= 75:
            market_mood = "Strongly Bullish"
        elif market_score >= 60:
            market_mood = "Bullish"
        elif market_score >= 40:
            market_mood = "Neutral"
        elif market_score >= 25:
            market_mood = "Bearish"
        else:
            market_mood = "Strongly Bearish"
        
        # Generate mock change
        market_change = random.uniform(-5, 5)
        
        # Get news feed for market
        market_news = self.news_generator.generate_feed(None, 10)
        
        return {
            "market_score": market_score,
            "market_mood": market_mood,
            "market_change": market_change,
            "components": sentiment_data,
            "news_feed": market_news
        }
    
    def create_price_sentiment_chart(self, analysis: Dict) -> go.Figure:
        """Create an interactive chart showing price and sentiment"""
        stock_data = analysis["price_data"]
        sentiment_data = analysis["sentiment_data"]
        ticker = analysis["ticker"]
        company_name = analysis["company_name"]
        
        if stock_data.empty or not sentiment_data:
            # Create empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Price",
                line=dict(color='#1E88E5', width=2)
            ),
            secondary_y=False,
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=sentiment_data,
                name="Sentiment",
                line=dict(color='#4CAF50', width=2, dash='dot')
            ),
            secondary_y=True,
        )
        
        # Add shapes for sentiment zones
        fig.add_shape(
            type="rect",
            x0=stock_data.index[0],
            x1=stock_data.index[-1],
            y0=0,
            y1=40,
            line=dict(width=0),
            fillcolor="rgba(244, 67, 54, 0.1)",
            layer="below",
            yref="y2"
        )
        
        fig.add_shape(
            type="rect",
            x0=stock_data.index[0],
            x1=stock_data.index[-1],
            y0=40,
            y1=60,
            line=dict(width=0),
            fillcolor="rgba(255, 152, 0, 0.1)",
            layer="below",
            yref="y2"
        )
        
        fig.add_shape(
            type="rect",
            x0=stock_data.index[0],
            x1=stock_data.index[-1],
            y0=60,
            y1=100,
            line=dict(width=0),
            fillcolor="rgba(76, 175, 80, 0.1)",
            layer="below",
            yref="y2"
        )
        
        # Add a horizontal line at neutral sentiment (50)
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            x1=stock_data.index[-1],
            y0=50,
            y1=50,
            line=dict(color="rgba(0, 0, 0, 0.5)", width=1, dash="dash"),
            layer="below",
            yref="y2"
        )
        
        # Update layout
        fig.update_layout(
            title=f"{company_name} ({ticker}) - Price and Sentiment Analysis",
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[0, 100])
        
        return fig
    
    def create_sentiment_distribution_chart(self, analysis: Dict) -> go.Figure:
        """Create a chart showing sentiment distribution"""
        positive = analysis["positive_pct"]
        neutral = analysis["neutral_pct"]
        negative = analysis["negative_pct"]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[positive, neutral, negative],
            hole=.4,
            marker_colors=['#4CAF50', '#FF9800', '#F44336']
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        return fig


# Initialize the app
@st.cache_resource
def get_sentiment_trader():
    return SentimentTrader()

trader = get_sentiment_trader()

# Sidebar
st.sidebar.markdown('<div class="main-header">SentiTrade</div>', unsafe_allow_html=True)
st.sidebar.markdown("Real-time sentiment analysis for stock trading")

# Stock selection
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
all_tickers = list(trader.news_generator.companies.keys())

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    options=all_tickers,
    index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0
)

# Time period selection
period_options = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "Year to Date": "ytd"
}

selected_period_name = st.sidebar.selectbox(
    "Select Time Period",
    options=list(period_options.keys()),
    index=2  # Default to 1 Month
)
selected_period = period_options[selected_period_name]

# Add a refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_resource.clear()
    st.rerun()


# Disclaimer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="news-time">
This app uses a combination of real stock data and simulated sentiment analysis. 
It is for demonstration purposes only and should not be used for actual trading decisions.
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="main-header">Stock Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)

# Show a loading spinner while analyzing
with st.spinner(f"Analyzing sentiment for {selected_ticker}..."):
    # Get market sentiment
    market_sentiment = trader.analyze_market_sentiment()
    
    # Get stock sentiment
    stock_analysis = trader.analyze_stock_sentiment(selected_ticker, selected_period)

# Market Overview Section
st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)

# Market metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div>Market Sentiment</div>
        <div class="metric-{}" style="font-size: 1.5rem;">{}</div>
        <div>{:+.1f}% from yesterday</div>
    </div>
    """.format(
        "positive" if market_sentiment["market_score"] >= 60 else "negative" if market_sentiment["market_score"] <= 40 else "neutral",
        market_sentiment["market_mood"],
        market_sentiment["market_change"]
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div>S&P 500 Sentiment</div>
        <div class="metric-{}" style="font-size: 1.5rem;">{:.1f}</div>
        <div>Score out of 100</div>
    </div>
    """.format(
        "positive" if market_sentiment["components"]["SPY"] >= 60 else "negative" if market_sentiment["components"]["SPY"] <= 40 else "neutral",
        market_sentiment["components"]["SPY"]
    ), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div>NASDAQ Sentiment</div>
        <div class="metric-{}" style="font-size: 1.5rem;">{:.1f}</div>
        <div>Score out of 100</div>
    </div>
    """.format(
        "positive" if market_sentiment["components"]["QQQ"] >= 60 else "negative" if market_sentiment["components"]["QQQ"] <= 40 else "neutral",
        market_sentiment["components"]["QQQ"]
    ), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div>Trading Signal</div>
        <div class="recommendation-{}" style="font-size: 1.5rem;">{}</div>
        <div>Based on market sentiment</div>
    </div>
    """.format(
        "buy" if market_sentiment["market_score"] >= 60 else "sell" if market_sentiment["market_score"] <= 40 else "hold",
        "Buy" if market_sentiment["market_score"] >= 60 else "Sell" if market_sentiment["market_score"] <= 40 else "Hold"
    ), unsafe_allow_html=True)

# Stock Analysis Section
st.markdown(f'<div class="sub-header">{stock_analysis["company_name"]} ({selected_ticker}) Analysis</div>', unsafe_allow_html=True)

# Stock info and metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Current price
    current_price = stock_analysis["price_data"]["Close"].iloc[-1] if not stock_analysis["price_data"].empty else 0
    prev_price = stock_analysis["price_data"]["Close"].iloc[-2] if not stock_analysis["price_data"].empty and len(stock_analysis["price_data"]) > 1 else current_price
    price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
    
    st.markdown(f"""
    <div class="card">
        <div>Current Price</div>
        <div style="font-size: 1.5rem;">${current_price:.2f}</div>
        <div class="metric-{'positive' if price_change_pct >= 0 else 'negative'}">{price_change_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div>Sentiment Score</div>
        <div class="metric-{'positive' if stock_analysis['sentiment_score'] >= 60 else 'negative' if stock_analysis['sentiment_score'] <= 40 else 'neutral'}" style="font-size: 1.5rem;">{stock_analysis['sentiment_score']:.1f}</div>
        <div>{stock_analysis['sentiment_change']:+.1f}% change</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div>Price-Sentiment Correlation</div>
        <div style="font-size: 1.5rem;">{stock_analysis['price_sentiment_correlation']:.2f}</div>
        <div>{'Strong' if abs(stock_analysis['price_sentiment_correlation']) > 0.7 else 'Moderate' if abs(stock_analysis['price_sentiment_correlation']) > 0.3 else 'Weak'} correlation</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
        <div>Trading Signal</div>
        <div class="recommendation-{'buy' if stock_analysis['recommendation'] in ['Buy', 'Strong Buy'] else 'sell' if stock_analysis['recommendation'] in ['Sell', 'Strong Sell'] else 'hold'}" style="font-size: 1.5rem;">{stock_analysis['recommendation']}</div>
        <div>Based on sentiment analysis</div>
    </div>
    """, unsafe_allow_html=True)

# Charts
col1, col2 = st.columns([3, 1])

with col1:
    # Price and sentiment chart
    price_sentiment_chart = trader.create_price_sentiment_chart(stock_analysis)
    st.plotly_chart(price_sentiment_chart, use_container_width=True)

with col2:
    # Sentiment distribution chart
    sentiment_dist_chart = trader.create_sentiment_distribution_chart(stock_analysis)
    st.plotly_chart(sentiment_dist_chart, use_container_width=True)
    
    # Sentiment breakdown
    st.markdown("""
    <div class="card">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">Sentiment Breakdown</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div>Positive: {stock_analysis['positive_pct']:.1f}%</div>", unsafe_allow_html=True)
    st.progress(stock_analysis['positive_pct']/100)
    
    st.markdown(f"<div>Neutral: {stock_analysis['neutral_pct']:.1f}%</div>", unsafe_allow_html=True)
    st.progress(stock_analysis['neutral_pct']/100)
    
    st.markdown(f"<div>Negative: {stock_analysis['negative_pct']:.1f}%</div>", unsafe_allow_html=True)
    st.progress(stock_analysis['negative_pct']/100)
    
    st.markdown("</div>", unsafe_allow_html=True)

# News and Social Media Feed
st.markdown('<div class="sub-header">News & Social Media Sentiment</div>', unsafe_allow_html=True)

# Filter news for the selected ticker
ticker_news = [item for item in stock_analysis["news_feed"] if item["ticker"] == selected_ticker]

# Display news items
for item in ticker_news[:10]:  # Show top 10 news items
    sentiment_class = f"sentiment-{item['sentiment']}"
    icon = "📰" if item["type"] == "news" else "🐦" if item["type"] == "twitter" else "💬"
    
    st.markdown(f"""
    <div class="news-item {sentiment_class}">
        <div>
            <span class="ticker-symbol">${item['ticker']}</span> {icon} {item['content']}
        </div>
        <div>
            <span class="news-source">{item['source']}</span> · 
            <span class="news-time">{item['time']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Trading Recommendations
st.markdown('<div class="sub-header">Trading Recommendations</div>', unsafe_allow_html=True)

# Create columns for recommendations
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">Short-term Outlook (1-7 days)</div>
        <div class="recommendation-{}" style="display: inline-block; margin-bottom: 0.5rem;">{}</div>
        <div>
            <p>Based on recent sentiment trends and price action, the short-term outlook suggests a {} strategy.</p>
            <p>Key factors:</p>
            <ul>
                <li>Sentiment score: {:.1f}/100</li>
                <li>Recent sentiment change: {:+.1f}%</li>
                <li>Price-sentiment correlation: {:.2f}</li>
            </ul>
        </div>
    </div>
    """.format(
        "buy" if stock_analysis['sentiment_score'] >= 60 else "sell" if stock_analysis['sentiment_score'] <= 40 else "hold",
        stock_analysis['recommendation'],
        stock_analysis['recommendation'].lower(),
        stock_analysis['sentiment_score'],
        stock_analysis['sentiment_change'],
        stock_analysis['price_sentiment_correlation']
    ), unsafe_allow_html=True)

with col2:
    # Generate a slightly different long-term recommendation for variety
    long_term_score = min(100, max(0, stock_analysis['sentiment_score'] + random.uniform(-10, 10)))
    
    if long_term_score >= 75:
        long_term_rec = "Strong Buy"
    elif long_term_score >= 60:
        long_term_rec = "Buy"
    elif long_term_score >= 40:
        long_term_rec = "Hold"
    elif long_term_score >= 25:
        long_term_rec = "Sell"
    else:
        long_term_rec = "Strong Sell"
    
    rec_class = "buy" if long_term_rec in ["Buy", "Strong Buy"] else "sell" if long_term_rec in ["Sell", "Strong Sell"] else "hold"
    
    st.markdown(f"""
    <div class="card">
        <div style="font-weight: 600; margin-bottom: 0.5rem;">Long-term Outlook (1-3 months)</div>
        <div class="recommendation-{rec_class}" style="display: inline-block; margin-bottom: 0.5rem;">{long_term_rec}</div>
        <div>
            <p>The long-term sentiment analysis indicates a {long_term_rec.lower()} position may be appropriate.</p>
            <p>Key factors:</p>
            <ul>
                <li>Industry sentiment: {stock_analysis['industry']}</li>
                <li>Sector performance: {stock_analysis['sector']}</li>
                <li>Long-term sentiment score: {long_term_score:.1f}/100</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add a footer with timestamp
st.markdown(f"""
<div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# Run the app with: streamlit run stock_sentiment_app.py