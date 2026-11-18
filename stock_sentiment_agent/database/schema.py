"""SQLAlchemy ORM schema for all 9 tables in HengSeng Sentiment Price Prediction Agent."""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from sqlalchemy import (
        Boolean,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Index,
        Integer,
        String,
        Text,
        UniqueConstraint,
    )
    from sqlalchemy.orm import declarative_base, relationship

    Base = declarative_base()

    class Stock(Base):
        __tablename__ = "stocks"

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), unique=True, nullable=False, index=True)
        company_name = Column(String(120), nullable=False)
        exchange = Column(String(16), nullable=False, index=True)
        sector = Column(String(80), nullable=True)
        currency = Column(String(8), nullable=False, default="HKD")
        is_active = Column(Boolean, default=True, nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        social_posts = relationship("SocialPost", back_populates="stock", cascade="all, delete-orphan")
        news_articles = relationship("NewsArticle", back_populates="stock", cascade="all, delete-orphan")
        prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
        sentiments = relationship("SentimentResult", back_populates="stock", cascade="all, delete-orphan")
        features = relationship("FinancialFeature", back_populates="stock", cascade="all, delete-orphan")
        predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")

    class SocialPost(Base):
        __tablename__ = "social_posts"
        __table_args__ = (
            UniqueConstraint("source_platform", "post_id", name="uq_social_platform_post"),
            Index("ix_social_ticker_utc", "ticker", "utc_timestamp"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        company_name = Column(String(120), nullable=False)
        post_id = Column(String(128), nullable=False)
        post_text = Column(Text, nullable=False)
        cleaned_text = Column(Text, nullable=True)
        post_timestamp = Column(String(64), nullable=False)
        utc_timestamp = Column(DateTime, nullable=False)
        aligned_trading_date = Column(String(16), nullable=True, index=True)
        source_platform = Column(String(40), nullable=False, index=True)
        author_username = Column(String(120), nullable=True)
        author_follower_count = Column(Integer, nullable=True)
        likes = Column(Integer, default=0, nullable=False)
        comments = Column(Integer, default=0, nullable=False)
        reposts = Column(Integer, default=0, nullable=False)
        views = Column(Integer, default=0, nullable=False)
        post_url = Column(String(512), nullable=True)
        raw_payload = Column(Text, nullable=True)
        is_simulated = Column(Boolean, default=False, nullable=False)
        collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="social_posts")

    class NewsArticle(Base):
        __tablename__ = "news_articles"
        __table_args__ = (
            UniqueConstraint("article_url", name="uq_news_article_url"),
            Index("ix_news_ticker_pub", "ticker", "published_utc"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        headline = Column(String(400), nullable=False)
        article_url = Column(String(600), nullable=False)
        published_at = Column(String(64), nullable=True)
        published_utc = Column(DateTime, nullable=False)
        aligned_trading_date = Column(String(16), nullable=True, index=True)
        summary = Column(Text, nullable=True)
        source = Column(String(120), nullable=True)
        relevance_score = Column(Float, default=0.0, nullable=False)
        sentiment_label = Column(String(16), nullable=True)
        sentiment_score = Column(Float, nullable=True)
        is_simulated = Column(Boolean, default=False, nullable=False)
        collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="news_articles")

    class StockPrice(Base):
        __tablename__ = "stock_prices"
        __table_args__ = (
            UniqueConstraint("ticker", "trading_date", name="uq_price_ticker_date"),
            Index("ix_price_ticker_date", "ticker", "trading_date"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        trading_date = Column(String(16), nullable=False)
        open = Column(Float, nullable=False)
        high = Column(Float, nullable=False)
        low = Column(Float, nullable=False)
        close = Column(Float, nullable=False)
        volume = Column(Float, nullable=False)
        daily_return = Column(Float, nullable=True)
        is_simulated = Column(Boolean, default=False, nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="prices")

    class SentimentResult(Base):
        __tablename__ = "sentiment_results"
        __table_args__ = (
            UniqueConstraint("ticker", "trading_date", "source_scope", name="uq_sentiment_scope"),
            Index("ix_sentiment_ticker_date", "ticker", "trading_date"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        trading_date = Column(String(16), nullable=False)
        source_scope = Column(String(40), nullable=False)  # 'social', 'news', 'combined', or platform name
        total_items = Column(Integer, default=0, nullable=False)
        positive_count = Column(Integer, default=0, nullable=False)
        negative_count = Column(Integer, default=0, nullable=False)
        neutral_count = Column(Integer, default=0, nullable=False)
        positive_pct = Column(Float, default=0.0, nullable=False)
        negative_pct = Column(Float, default=0.0, nullable=False)
        neutral_pct = Column(Float, default=0.0, nullable=False)
        sentiment_score = Column(Float, default=0.0, nullable=False)
        engagement_weighted_score = Column(Float, default=0.0, nullable=False)
        bullish_probability = Column(Float, default=0.5, nullable=False)
        bearish_probability = Column(Float, default=0.5, nullable=False)
        confidence_score = Column(Float, default=0.0, nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="sentiments")

    class FinancialFeature(Base):
        __tablename__ = "financial_features"
        __table_args__ = (
            UniqueConstraint("ticker", "trading_date", name="uq_feature_ticker_date"),
            Index("ix_feature_ticker_date", "ticker", "trading_date"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        trading_date = Column(String(16), nullable=False)
        social_mentions = Column(Integer, default=0, nullable=False)
        social_sentiment = Column(Float, default=0.0, nullable=False)
        pos_neg_ratio = Column(Float, default=1.0, nullable=False)
        engagement_score = Column(Float, default=0.0, nullable=False)
        avg_author_followers = Column(Float, nullable=True)
        mention_growth = Column(Float, default=0.0, nullable=False)
        sentiment_change = Column(Float, default=0.0, nullable=False)
        news_sentiment = Column(Float, default=0.0, nullable=False)
        news_count = Column(Integer, default=0, nullable=False)
        news_relevance = Column(Float, default=0.0, nullable=False)
        news_recency = Column(Float, default=0.0, nullable=False)
        rsi_14 = Column(Float, nullable=True)
        macd = Column(Float, nullable=True)
        macd_signal = Column(Float, nullable=True)
        sma_5 = Column(Float, nullable=True)
        sma_10 = Column(Float, nullable=True)
        sma_20 = Column(Float, nullable=True)
        volatility_10 = Column(Float, nullable=True)
        support_level = Column(Float, nullable=True)
        resistance_level = Column(Float, nullable=True)
        target_next_close = Column(Float, nullable=True)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="features")

    class Prediction(Base):
        __tablename__ = "predictions"
        __table_args__ = (
            Index("ix_pred_ticker_target", "ticker", "forecast_target_date"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        prediction_date = Column(String(16), nullable=False)
        forecast_target_date = Column(String(16), nullable=False)
        last_close = Column(Float, nullable=False)
        predicted_close = Column(Float, nullable=False)
        actual_close = Column(Float, nullable=True)
        predicted_pct_change = Column(Float, nullable=False)
        model_name = Column(String(64), nullable=False)
        mae = Column(Float, nullable=True)
        rmse = Column(Float, nullable=True)
        mape = Column(Float, nullable=True)
        directional_accuracy = Column(Float, nullable=True)
        is_validated = Column(Boolean, default=False, nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        stock = relationship("Stock", back_populates="predictions")

    class ModelMetric(Base):
        __tablename__ = "model_metrics"
        __table_args__ = (
            Index("ix_metrics_ticker_model", "ticker", "model_name"),
        )

        id = Column(Integer, primary_key=True, autoincrement=True)
        ticker = Column(String(20), ForeignKey("stocks.ticker", ondelete="CASCADE"), nullable=False)
        model_name = Column(String(64), nullable=False)
        split_window = Column(String(64), nullable=False)
        mae = Column(Float, nullable=False)
        rmse = Column(Float, nullable=False)
        mape = Column(Float, nullable=False)
        directional_accuracy = Column(Float, nullable=False)
        is_best_model = Column(Boolean, default=False, nullable=False)
        feature_importances_json = Column(Text, nullable=True)
        evaluated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    class AgentLog(Base):
        __tablename__ = "agent_logs"

        id = Column(Integer, primary_key=True, autoincrement=True)
        agent_name = Column(String(64), nullable=False, index=True)
        status = Column(String(24), nullable=False)
        ticker = Column(String(20), nullable=True)
        message = Column(Text, nullable=False)
        records_processed = Column(Integer, default=0, nullable=False)
        duration_ms = Column(Float, default=0.0, nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

except ImportError:  # Allows stdlib-only environments to import module cleanly
    Base = None
    Stock = SocialPost = NewsArticle = StockPrice = None
    SentimentResult = FinancialFeature = Prediction = ModelMetric = AgentLog = None
