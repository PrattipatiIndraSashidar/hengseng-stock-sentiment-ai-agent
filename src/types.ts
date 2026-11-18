export type DashboardPageId =
  | 'overview'
  | 'hkex'
  | 'krx'
  | 'sentiment'
  | 'intelligence'
  | 'prediction'
  | 'reports';

export type DataCategory = 'LIVE DATA' | 'HISTORICAL DATA' | 'SIMULATED DEMO DATA';

export interface StockConfigItem {
  ticker: string;
  company_name: string;
  exchange: 'HKEX' | 'KRX' | string;
  sector: string;
  aliases: string[];
  enabled: boolean;
}

export interface NewsHighlight {
  headline: string;
  source: string;
  url: string;
  sentiment: 'Positive' | 'Negative' | 'Neutral' | string;
  published_utc: string;
}

export interface FinancialIntelligence {
  ticker: string;
  company_name: string;
  exchange: string;
  currency?: string;
  currency_symbol?: string;
  exchange_timezone?: string;
  latest_price: number;
  latest_trading_date: string;
  total_social_mentions: number;
  positive_mentions: number;
  negative_mentions: number;
  neutral_mentions: number;
  social_sentiment_score: number;
  news_sentiment_score: number;
  combined_sentiment_score: number;
  bullish_probability: number;
  bearish_probability: number;
  market_direction: 'Bullish' | 'Bearish' | 'Neutral';
  mentioned_trading_position: 'Long' | 'Short' | 'Hold' | 'Not mentioned';
  support_levels: number[];
  resistance_levels: number[];
  rsi_14: number;
  macd: number;
  macd_signal: number;
  moving_averages: {
    sma_5: number;
    sma_10: number;
    sma_20: number;
  };
  trading_volume: number;
  news_highlights: NewsHighlight[];
}

export interface PredictionRecord {
  ticker: string;
  prediction_date: string;
  forecast_target_date: string;
  last_close: number;
  predicted_close: number;
  actual_close: number | null;
  predicted_pct_change: number;
  model_name: string;
  artifact_path?: string;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  directional_accuracy: number | null;
  is_validated: boolean | number;
}

export interface WalkForwardPredictionPoint {
  prediction_date: string;
  last_close: number;
  predicted_close: number;
  actual_close: number;
}

export interface WalkForwardValidation {
  model_name: string;
  split_window: string;
  mae: number;
  rmse: number;
  mape: number;
  directional_accuracy: number;
  predictions: WalkForwardPredictionPoint[];
  feature_importances: Record<string, number>;
  is_best_model: boolean;
}

export interface SentimentAggregate {
  total_items: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  positive_pct: number;
  negative_pct: number;
  neutral_pct: number;
  sentiment_score: number;
  engagement_weighted_score: number;
  bullish_probability: number;
  bearish_probability: number;
  confidence_score: number;
}

export interface SocialPostItem {
  ticker: string;
  post_id: string;
  post_text: string;
  utc_timestamp: string;
  aligned_trading_date: string;
  source_platform: string;
  author_username: string | null;
  author_follower_count: number | null;
  likes: number;
  comments: number;
  reposts: number;
  views: number;
  sentiment_label: string;
  sentiment_score: number;
  engagement_weight: number;
  is_simulated: boolean;
  data_category?: DataCategory;
}

export interface NewsArticleItem {
  ticker: string;
  company_name?: string;
  headline: string;
  article_url: string;
  published_utc: string;
  aligned_trading_date: string;
  summary: string;
  source: string;
  relevance_score: number;
  sentiment_label: string;
  sentiment_score: number;
  is_simulated: boolean;
  data_category?: DataCategory;
}

export interface PriceBar {
  ticker: string;
  trading_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  daily_return: number | null;
  is_simulated: boolean;
  data_category?: DataCategory;
}

export interface SourceConnectionStatus {
  source: string;
  kind: string;
  status: 'connected' | 'unavailable' | string;
  data_category: DataCategory | string;
  reason: string;
}

export interface ConfigCheckItem {
  configured: boolean;
  required: boolean;
  status: string;
  message: string;
}

export interface StockSummary {
  stock: StockConfigItem;
  intelligence: FinancialIntelligence;
  prediction: PredictionRecord;
  validation: WalkForwardValidation[];
  sentiment: {
    engine: string;
    social_summary: SentimentAggregate;
    news_summary: SentimentAggregate;
    combined_summary: SentimentAggregate;
    platform_summaries: Record<string, SentimentAggregate>;
  };
  data_provenance: {
    social: Record<string, { status?: string; mode: string; data_category?: DataCategory; reason: string }>;
    news: { status?: string; mode: string; data_category?: DataCategory; reason: string };
    prices: { status?: string; provider?: string; mode: string; data_category?: DataCategory; latest_quote_category?: DataCategory; reason: string };
    categories?: {
      prices: DataCategory;
      latest_quote?: DataCategory;
      social: DataCategory;
      news: DataCategory;
    };
    genuine_posts_count?: number;
    simulated_posts_count?: number;
    genuine_news_count?: number;
    simulated_news_count?: number;
    features_used?: string[];
    validation_mode?: string;
    is_demo_model?: boolean;
    contains_simulated_data: boolean;
  };
  recent_posts: SocialPostItem[];
  recent_news: NewsArticleItem[];
  price_history: PriceBar[];
}

export interface AgentLogEntry {
  agent_name: string;
  status: string;
  ticker: string | null;
  message: string;
  records_processed: number;
  duration_ms: number;
  created_at: string;
}

export interface AssessmentSummary {
  generated_utc: string;
  social_sources: Array<{
    platform: string;
    exchange: string;
    actor_id: string;
    status: string;
    data_category: string;
    genuine_posts: number;
    reason: string;
  }>;
  exa_status: {
    source: string;
    status: string;
    data_category: string;
    genuine_articles: number;
    reason: string;
  };
  market_data_status: {
    providers: string[];
    status: string;
    stocks_with_genuine_prices: number;
    data_category: string;
    latest_quote_category: string;
    reason: string;
  };
  genuine_posts_collected: number;
  simulated_demo_posts: number;
  genuine_news_collected: number;
  simulated_demo_news: number;
  stocks_analyzed: number;
  validation_summary: {
    mean_best_directional_accuracy: number;
    mean_naive_directional_accuracy: number;
    models_beating_or_matching_naive_rmse: number;
    total_stocks_validated: number;
    per_stock: Array<{
      ticker: string;
      company_name: string;
      exchange: string;
      validation_mode: string;
      is_demo_model: boolean;
      best_model: string;
      best_mae: number;
      best_rmse: number;
      best_mape: number;
      best_directional_accuracy: number;
      naive_mae: number | null;
      naive_rmse: number | null;
      naive_directional_accuracy: number | null;
      split_window: string;
    }>;
  };
  remaining_limitations: string[];
}

export interface DashboardPayload {
  generated_utc: string;
  stocks_analyzed: number;
  configuration_validation?: {
    valid: boolean;
    live_social_ready: boolean;
    live_news_ready: boolean;
    checks: Record<string, ConfigCheckItem>;
  };
  api_connection_status?: SourceConnectionStatus[];
  assessment_summary?: AssessmentSummary;
  summaries: StockSummary[];
  prediction_history?: PredictionRecord[];
  report?: {
    rankings?: Record<string, Record<string, StockSummary[]>>;
    csv_path?: string;
    pdf_path?: string;
    assessment_report_path?: string;
  };
  logs?: AgentLogEntry[];
  environment?: {
    apifyConfigured: boolean;
    exaConfigured: boolean;
    databaseUrlConfigured?: boolean;
    modelPathConfigured?: boolean;
  };
}
