import React, { useMemo, useState } from 'react';
import {
  ArrowDownRight,
  ArrowUpRight,
  Download,
  FileText,
  Filter,
  Minus,
  Search,
  SlidersHorizontal,
} from 'lucide-react';
import type {
  AgentLogEntry,
  DashboardPayload,
  PredictionRecord,
  StockConfigItem,
  StockSummary,
} from '../types';
import {
  ActualVsPredictedChart,
  FeatureImportanceChart,
  PredictionErrorChart,
  PriceAndForecastChart,
  SentimentComparisonChart,
} from './TerminalCharts';
import { exportSummariesToCsv, exportSummariesToPdf } from '../utils/exportReports';

function DirectionText({ direction }: { direction: string }) {
  if (direction === 'Bullish') {
    return (
      <span className="inline-flex items-center gap-1 font-medium text-emerald-400">
        <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
        Bullish
      </span>
    );
  }
  if (direction === 'Bearish') {
    return (
      <span className="inline-flex items-center gap-1 font-medium text-rose-400">
        <ArrowDownRight className="h-3.5 w-3.5" aria-hidden="true" />
        Bearish
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 font-medium text-amber-400">
      <Minus className="h-3.5 w-3.5" aria-hidden="true" />
      Neutral
    </span>
  );
}

function CategoryLabel({ category }: { category?: string }) {
  const cat = category || 'SIMULATED DEMO DATA';
  const cls =
    cat === 'LIVE DATA'
      ? 'text-emerald-400'
      : cat === 'HISTORICAL DATA'
      ? 'text-sky-400'
      : 'text-amber-400';
  return <span className={`font-mono text-[11px] font-medium ${cls}`}>{cat}</span>;
}

function fmtPrice(val: number, exchange: string) {
  return exchange === 'HKEX' ? `HK$${val.toFixed(2)}` : `₩${Math.round(val).toLocaleString()}`;
}

export function OverviewPage({
  payload,
  onSelectStock,
}: {
  payload: DashboardPayload;
  onSelectStock: (ticker: string) => void;
}) {
  const summaries = payload.summaries;
  const totalPosts = summaries.reduce((acc, s) => acc + s.intelligence.total_social_mentions, 0);
  const totalNews = summaries.reduce((acc, s) => acc + s.recent_news.length, 0);
  const avgSentiment =
    summaries.reduce((acc, s) => acc + s.intelligence.combined_sentiment_score, 0) /
    Math.max(1, summaries.length);
  const bullishCount = summaries.filter((s) => s.intelligence.market_direction === 'Bullish').length;
  const bearishCount = summaries.filter((s) => s.intelligence.market_direction === 'Bearish').length;

  const sourceRows = payload.api_connection_status || [];
  const configChecks = payload.configuration_validation?.checks
    ? Object.entries(payload.configuration_validation.checks)
    : [];

  return (
    <div className="space-y-6">
      {/* KPI Strip */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Stocks Analyzed</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-white">
            {summaries.length}
          </div>
          <div className="mt-1 text-xs text-slate-500">HKEX 5 · KRX 5</div>
        </div>
        <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Social Posts</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-white">
            {payload.assessment_summary ? payload.assessment_summary.genuine_posts_collected : totalPosts}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {payload.assessment_summary && payload.assessment_summary.genuine_posts_collected === 0
              ? `Genuine · ${totalPosts} demo sample`
              : '4 regional platforms'}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Financial News</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-white">
            {payload.assessment_summary ? payload.assessment_summary.genuine_news_collected : totalNews}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {payload.assessment_summary && payload.assessment_summary.genuine_news_collected === 0
              ? `Genuine · ${totalNews} demo sample`
              : 'Exa financial feed'}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Market Sentiment</div>
          <div
            className={`mt-1.5 font-mono text-2xl font-semibold tabular-nums ${
              avgSentiment >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {avgSentiment >= 0 ? `+${avgSentiment.toFixed(3)}` : avgSentiment.toFixed(3)}
          </div>
          <div className="mt-1 text-xs text-slate-500">Weighted [-1, +1]</div>
        </div>
        <div className="col-span-2 rounded-lg border border-slate-800/90 bg-[#111827] p-4 lg:col-span-1">
          <div className="text-xs text-slate-400">Bullish vs Bearish</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-white">
            <span className="text-emerald-400">{bullishCount}</span>
            <span className="mx-1.5 text-slate-600">/</span>
            <span className="text-rose-400">{bearishCount}</span>
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {summaries.length - bullishCount - bearishCount} Neutral
          </div>
        </div>
      </div>

      {/* Primary Visual Anchor */}
      <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-white">Cross-Exchange Sentiment Distribution</h2>
            <p className="text-xs text-slate-400">
              Engagement-weighted social (55%) fused with financial news sentiment (45%). Provenance tagged per equity below.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span className="text-emerald-400">Bullish</span>
            <span className="text-amber-400">Neutral</span>
            <span className="text-rose-400">Bearish</span>
          </div>
        </div>
        <SentimentComparisonChart summaries={summaries} />
      </div>

      {/* Summary Equity Grid with Provenance */}
      <div className="rounded-lg border border-slate-800/90 bg-[#111827]">
        <div className="border-b border-slate-800/80 px-5 py-3.5 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-base font-semibold text-white">Monitored Equity Matrix</h2>
          <span className="text-xs text-slate-400">
            Provenance: HISTORICAL DATA (Market) · LIVE / SIMULATED DEMO DATA (Social & News)
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="px-4 py-3 font-medium">Ticker</th>
                <th className="px-4 py-3 font-medium">Company</th>
                <th className="px-4 py-3 font-medium">Exchange</th>
                <th className="px-4 py-3 text-right font-medium">Last Close</th>
                <th className="px-4 py-3 text-right font-medium">Pred n+1</th>
                <th className="px-4 py-3 text-right font-medium">% Change</th>
                <th className="px-4 py-3 text-right font-medium">Mentions</th>
                <th className="px-4 py-3 text-right font-medium">Sentiment</th>
                <th className="px-4 py-3 font-medium">Direction</th>
                <th className="px-4 py-3 font-medium">Price / Social Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {summaries.map((s) => {
                const i = s.intelligence;
                const p = s.prediction;
                const cats = s.data_provenance.categories;
                return (
                  <tr
                    key={i.ticker}
                    onClick={() => onSelectStock(i.ticker)}
                    className="cursor-pointer transition-colors hover:bg-slate-800/40"
                  >
                    <td className="px-4 py-3 font-mono font-semibold text-sky-300">{i.ticker}</td>
                    <td className="px-4 py-3 text-slate-200">{i.company_name}</td>
                    <td className="px-4 py-3 font-mono text-slate-400">{i.exchange}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                      {fmtPrice(i.latest_price, i.exchange)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-white">
                      {fmtPrice(p.predicted_close, i.exchange)}
                    </td>
                    <td
                      className={`px-4 py-3 text-right font-mono tabular-nums ${
                        p.predicted_pct_change >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {p.predicted_pct_change >= 0 ? '+' : ''}
                      {p.predicted_pct_change.toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">
                      {i.total_social_mentions}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                      {i.combined_sentiment_score >= 0 ? '+' : ''}
                      {i.combined_sentiment_score.toFixed(3)}
                    </td>
                    <td className="px-4 py-3">
                      <DirectionText direction={i.market_direction} />
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <CategoryLabel category={cats?.prices} />
                      <span className="mx-1 text-slate-600">/</span>
                      <CategoryLabel category={cats?.social} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Actual API Connection Status & Environment Diagnostics */}
      {(sourceRows.length > 0 || configChecks.length > 0) && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-800/90 bg-[#111827] overflow-x-auto">
            <div className="border-b border-slate-800 px-5 py-3.5">
              <h3 className="text-sm font-semibold text-white">API Connection Status by Data Source</h3>
            </div>
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Source</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                  <th className="px-4 py-2.5 font-medium">Data Category</th>
                  <th className="px-4 py-2.5 font-medium">Detail</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {sourceRows.map((src) => (
                  <tr key={src.source}>
                    <td className="px-4 py-2.5 font-medium text-slate-200 whitespace-nowrap">{src.source}</td>
                    <td
                      className={`px-4 py-2.5 font-mono whitespace-nowrap ${
                        src.status === 'connected' ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {src.status === 'connected' ? 'Connected' : 'Unavailable'}
                    </td>
                    <td className="px-4 py-2.5 whitespace-nowrap">
                      <CategoryLabel category={src.data_category} />
                    </td>
                    <td className="px-4 py-2.5 text-slate-400">{src.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="rounded-lg border border-slate-800/90 bg-[#111827]">
            <div className="border-b border-slate-800 px-5 py-3.5">
              <h3 className="text-sm font-semibold text-white">Environment Configuration Validation</h3>
            </div>
            <div className="divide-y divide-slate-800/70">
              {configChecks.map(([envKey, info]) => (
                <div key={envKey} className="px-5 py-3 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono font-semibold text-slate-200">{envKey}</span>
                    <span
                      className={`font-mono ${
                        info.status === 'configured' || info.status === 'ready'
                          ? 'text-emerald-400'
                          : info.status === 'sqlite_fallback'
                          ? 'text-sky-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {info.status}
                    </span>
                  </div>
                  <p className="mt-1 text-slate-400">{info.message}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function ExchangeMarketPage({
  exchange,
  summaries,
  onSelectStock,
}: {
  exchange: 'HKEX' | 'KRX';
  summaries: StockSummary[];
  onSelectStock: (ticker: string) => void;
}) {
  const [rankBy, setRankBy] = useState<'sentiment' | 'mentions' | 'bullish' | 'return'>('sentiment');
  const filtered = useMemo(() => {
    const sub = summaries.filter((s) => s.intelligence.exchange === exchange);
    return [...sub].sort((a, b) => {
      if (rankBy === 'mentions') return b.intelligence.total_social_mentions - a.intelligence.total_social_mentions;
      if (rankBy === 'bullish') return b.intelligence.bullish_probability - a.intelligence.bullish_probability;
      if (rankBy === 'return') return b.prediction.predicted_pct_change - a.prediction.predicted_pct_change;
      return b.intelligence.combined_sentiment_score - a.intelligence.combined_sentiment_score;
    });
  }, [summaries, exchange, rankBy]);

  const title = exchange === 'HKEX' ? 'Hong Kong Stock Exchange (HKEX)' : 'South Korea Exchange (KRX)';
  const sources = exchange === 'HKEX' ? 'LIHKG · Telegram' : 'Naver Finance · Daum Finance';
  const tz = exchange === 'HKEX' ? 'Asia/Hong_Kong (HKT)' : 'Asia/Seoul (KST)';

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="text-xs text-slate-400">
            5 equities · Sources: {sources} · Session TZ: {tz}
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-[#111827] p-1">
          {[
            { id: 'sentiment', label: 'By Sentiment' },
            { id: 'mentions', label: 'By Mentions' },
            { id: 'bullish', label: 'By Bullish Prob' },
            { id: 'return', label: 'By Pred %' },
          ].map((opt) => (
            <button
              key={opt.id}
              onClick={() => setRankBy(opt.id as typeof rankBy)}
              className={`whitespace-nowrap shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                rankBy === opt.id ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-slate-800/90 bg-[#111827] p-5">
        <div className="mb-3 flex items-center justify-between text-xs text-slate-400">
          <span>Sentiment Ranking Chart</span>
          <CategoryLabel category={filtered[0]?.data_provenance.categories?.social} />
        </div>
        <SentimentComparisonChart summaries={filtered} />
      </div>

      <div className="rounded-lg border border-slate-800/90 bg-[#111827] overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Rank</th>
              <th className="px-4 py-3 font-medium">Ticker</th>
              <th className="px-4 py-3 font-medium">Company</th>
              <th className="px-4 py-3 text-right font-medium">Mentions</th>
              <th className="px-4 py-3 text-right font-medium">Social Sent</th>
              <th className="px-4 py-3 text-right font-medium">News Sent</th>
              <th className="px-4 py-3 text-right font-medium">Bullish %</th>
              <th className="px-4 py-3 text-right font-medium">Current</th>
              <th className="px-4 py-3 text-right font-medium">Predicted n+1</th>
              <th className="px-4 py-3 font-medium">Signal</th>
              <th className="px-4 py-3 font-medium">Provenance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {filtered.map((s, idx) => {
              const i = s.intelligence;
              const p = s.prediction;
              const cats = s.data_provenance.categories;
              return (
                <tr
                  key={i.ticker}
                  onClick={() => onSelectStock(i.ticker)}
                  className="cursor-pointer transition-colors hover:bg-slate-800/40"
                >
                  <td className="px-4 py-3 font-mono text-slate-400">#{idx + 1}</td>
                  <td className="px-4 py-3 font-mono font-semibold text-sky-300">{i.ticker}</td>
                  <td className="px-4 py-3 text-slate-200">{i.company_name}</td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">
                    {i.total_social_mentions}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                    {i.social_sentiment_score >= 0 ? '+' : ''}
                    {i.social_sentiment_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                    {i.news_sentiment_score >= 0 ? '+' : ''}
                    {i.news_sentiment_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                    {(i.bullish_probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                    {fmtPrice(i.latest_price, i.exchange)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-white">
                    {fmtPrice(p.predicted_close, i.exchange)}{' '}
                    <span className={p.predicted_pct_change >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      ({p.predicted_pct_change >= 0 ? '+' : ''}
                      {p.predicted_pct_change.toFixed(2)}%)
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <DirectionText direction={i.market_direction} />
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <CategoryLabel category={cats?.prices} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SentimentAnalysisPage({ summaries }: { summaries: StockSummary[] }) {
  const [sourceFilter, setSourceFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  const platformRows = useMemo(() => {
    const list: Array<{
      ticker: string;
      company: string;
      exchange: string;
      platform: string;
      total: number;
      posPct: number;
      negPct: number;
      neuPct: number;
      weightedScore: number;
      dataCategory: string;
    }> = [];
    summaries.forEach((s) => {
      Object.entries(s.sentiment.platform_summaries).forEach(([plat, agg]) => {
        if (sourceFilter !== 'ALL' && plat !== sourceFilter) return;
        const cat = s.data_provenance.social[plat]?.data_category || s.data_provenance.categories?.social || 'SIMULATED DEMO DATA';
        list.push({
          ticker: s.intelligence.ticker,
          company: s.intelligence.company_name,
          exchange: s.intelligence.exchange,
          platform: plat,
          total: agg.total_items,
          posPct: agg.positive_pct,
          negPct: agg.negative_pct,
          neuPct: agg.neutral_pct,
          weightedScore: agg.engagement_weighted_score,
          dataCategory: cat,
        });
      });
    });
    return list;
  }, [summaries, sourceFilter]);

  const postsFeed = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return summaries
      .flatMap((s) => s.recent_posts)
      .filter((p) => (sourceFilter === 'ALL' ? true : p.source_platform === sourceFilter))
      .filter((p) =>
        !q ? true : p.post_text.toLowerCase().includes(q) || p.ticker.toLowerCase().includes(q)
      )
      .sort((a, b) => b.engagement_weight - a.engagement_weight)
      .slice(0, 20);
  }, [summaries, sourceFilter, searchQuery]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-[#111827] p-1">
          {['ALL', 'LIHKG', 'Telegram', 'Naver Finance', 'Daum Finance'].map((src) => (
            <button
              key={src}
              onClick={() => setSourceFilter(src)}
              className={`whitespace-nowrap shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                sourceFilter === src ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {src === 'ALL' ? 'All Sources' : src}
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter posts or ticker..."
            className="w-full rounded-lg border border-slate-800 bg-[#111827] py-2 pl-8 pr-3 text-xs text-slate-200 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Platform-Wise Sentiment Table */}
      <div className="rounded-lg border border-slate-800/90 bg-[#111827] overflow-x-auto">
        <div className="border-b border-slate-800 px-5 py-3.5">
          <h2 className="text-base font-semibold text-white">Platform-Wise Engagement-Weighted Sentiment</h2>
        </div>
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Ticker</th>
              <th className="px-4 py-3 font-medium">Platform</th>
              <th className="px-4 py-3 text-right font-medium">Posts</th>
              <th className="px-4 py-3 text-right font-medium">Positive %</th>
              <th className="px-4 py-3 text-right font-medium">Negative %</th>
              <th className="px-4 py-3 text-right font-medium">Neutral %</th>
              <th className="px-4 py-3 text-right font-medium">Weighted Score</th>
              <th className="px-4 py-3 font-medium">Provenance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {platformRows.map((row) => (
              <tr key={`${row.ticker}-${row.platform}`} className="hover:bg-slate-800/30">
                <td className="px-4 py-2.5 font-mono text-sky-300">
                  {row.ticker} <span className="text-slate-500 font-sans">· {row.company}</span>
                </td>
                <td className="px-4 py-2.5 text-slate-300">{row.platform}</td>
                <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-300">{row.total}</td>
                <td className="px-4 py-2.5 text-right font-mono tabular-nums text-emerald-400">
                  {row.posPct.toFixed(1)}%
                </td>
                <td className="px-4 py-2.5 text-right font-mono tabular-nums text-rose-400">
                  {row.negPct.toFixed(1)}%
                </td>
                <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-400">
                  {row.neuPct.toFixed(1)}%
                </td>
                <td
                  className={`px-4 py-2.5 text-right font-mono tabular-nums font-medium ${
                    row.weightedScore >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {row.weightedScore >= 0 ? '+' : ''}
                  {row.weightedScore.toFixed(3)}
                </td>
                <td className="px-4 py-2.5 whitespace-nowrap">
                  <CategoryLabel category={row.dataCategory} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* High-Engagement Multilingual Social Feed */}
      <div className="rounded-lg border border-slate-800/90 bg-[#111827]">
        <div className="border-b border-slate-800 px-5 py-3.5">
          <h2 className="text-base font-semibold text-white">High-Engagement Multilingual Posts</h2>
        </div>
        <div className="divide-y divide-slate-800/70">
          {postsFeed.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500">No social posts match the current filter.</div>
          ) : (
            postsFeed.map((post) => (
              <div key={`${post.source_platform}-${post.post_id}`} className="px-5 py-3.5">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
                  <span>
                    <strong className="font-mono text-sky-300">{post.ticker}</strong>
                    <span className="mx-1.5">·</span>
                    <span>{post.source_platform}</span>
                    <span className="mx-1.5">·</span>
                    <span>Session {post.aligned_trading_date}</span>
                    <span className="mx-1.5">·</span>
                    <CategoryLabel
                      category={post.data_category || (post.is_simulated ? 'SIMULATED DEMO DATA' : 'LIVE DATA')}
                    />
                  </span>
                  <span className="font-mono tabular-nums">
                    <span
                      className={
                        post.sentiment_label === 'Positive'
                          ? 'text-emerald-400'
                          : post.sentiment_label === 'Negative'
                          ? 'text-rose-400'
                          : 'text-amber-400'
                      }
                    >
                      {post.sentiment_label} ({post.sentiment_score >= 0 ? '+' : ''}
                      {post.sentiment_score.toFixed(2)})
                    </span>
                    <span className="mx-1.5 text-slate-600">·</span>
                    <span>Weight {post.engagement_weight.toFixed(2)}x</span>
                  </span>
                </div>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-200">{post.post_text}</p>
                <div className="mt-1.5 flex flex-wrap gap-4 font-mono text-[11px] tabular-nums text-slate-500">
                  <span>Likes {post.likes}</span>
                  <span>Comments {post.comments}</span>
                  <span>Reposts {post.reposts}</span>
                  <span>Views {post.views}</span>
                  <span>
                    Followers {post.author_follower_count !== null ? post.author_follower_count.toLocaleString() : 'N/A'}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export function StockIntelligencePage({
  summaries,
  selectedTicker,
  onSelectTicker,
}: {
  summaries: StockSummary[];
  selectedTicker: string;
  onSelectTicker: (ticker: string) => void;
}) {
  const item = summaries.find((s) => s.intelligence.ticker === selectedTicker) || summaries[0];
  if (!item) return null;

  const intel = item.intelligence;
  const pred = item.prediction;
  const currency = intel.currency || (intel.exchange === 'HKEX' ? 'HKD' : 'KRW');
  const cats = item.data_provenance.categories;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">
            {intel.company_name} <span className="font-mono text-sky-400">({intel.ticker})</span>
          </h2>
          <p className="text-xs text-slate-400">
            {intel.exchange} ({currency} · {intel.exchange_timezone || (intel.exchange === 'HKEX' ? 'Asia/Hong_Kong' : 'Asia/Seoul')}) · Latest session{' '}
            {intel.latest_trading_date} · Engine: {item.sentiment.engine}
          </p>
        </div>
        <select
          aria-label="Select Equity Ticker"
          value={intel.ticker}
          onChange={(e) => onSelectTicker(e.target.value)}
          className="rounded-lg border border-slate-700 bg-[#111827] px-3 py-2 font-mono text-xs text-slate-100 focus:border-sky-500 focus:outline-none"
        >
          {summaries.map((s) => (
            <option key={s.intelligence.ticker} value={s.intelligence.ticker}>
              {s.intelligence.ticker} — {s.intelligence.company_name}
            </option>
          ))}
        </select>
      </div>

      {/* Explicit 3-Category Provenance Bar for Selected Stock */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 text-xs">
        <div className="rounded-lg border border-slate-800 bg-[#111827] px-4 py-2.5 flex items-center justify-between">
          <span className="text-slate-400">Market Prices</span>
          <CategoryLabel category={cats?.prices || item.data_provenance.prices.data_category} />
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#111827] px-4 py-2.5 flex items-center justify-between">
          <span className="text-slate-400">Social Feed</span>
          <CategoryLabel category={cats?.social} />
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#111827] px-4 py-2.5 flex items-center justify-between">
          <span className="text-slate-400">Financial News</span>
          <CategoryLabel category={cats?.news || item.data_provenance.news.data_category} />
        </div>
      </div>

      {/* Key Intelligence Metrics */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Latest Close ({currency})</div>
          <div className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">
            {fmtPrice(intel.latest_price, intel.exchange)}
          </div>
          <div className="mt-1 font-mono text-xs text-slate-500">
            Vol {intel.trading_volume.toLocaleString()}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Next-Day Forecast ({pred.forecast_target_date})</div>
          <div className="mt-1 font-mono text-xl font-semibold tabular-nums text-white">
            {fmtPrice(pred.predicted_close, intel.exchange)}
          </div>
          <div
            className={`mt-1 font-mono text-xs tabular-nums ${
              pred.predicted_pct_change >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {pred.predicted_pct_change >= 0 ? '+' : ''}
            {pred.predicted_pct_change.toFixed(2)}% via {pred.model_name}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Combined Sentiment</div>
          <div
            className={`mt-1 font-mono text-xl font-semibold tabular-nums ${
              intel.combined_sentiment_score >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {intel.combined_sentiment_score >= 0 ? '+' : ''}
            {intel.combined_sentiment_score.toFixed(3)}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            Social {intel.social_sentiment_score.toFixed(2)} · News {intel.news_sentiment_score.toFixed(2)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
          <div className="text-xs text-slate-400">Market Direction & Position</div>
          <div className="mt-1 text-xl font-semibold">
            <DirectionText direction={intel.market_direction} />
          </div>
          <div className="mt-1 text-xs text-slate-400">
            Explicit Position: <strong className="text-slate-200">{intel.mentioned_trading_position}</strong>
          </div>
        </div>
      </div>

      {/* Price Action + Support/Resistance Chart */}
      <div className="rounded-lg border border-slate-800 bg-[#111827] p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-white">
            Price Action, Support/Resistance & n+1 Forecast{' '}
            <span className="ml-2">
              <CategoryLabel category={cats?.prices || 'HISTORICAL DATA'} />
            </span>
          </h3>
          <span className="font-mono text-xs text-slate-400">
            Support {fmtPrice(intel.support_levels[1] ?? intel.support_levels[0], intel.exchange)} · Resistance{' '}
            {fmtPrice(intel.resistance_levels[0], intel.exchange)}
          </span>
        </div>
        <PriceAndForecastChart
          history={item.price_history}
          predictedClose={pred.predicted_close}
          forecastDate={pred.forecast_target_date}
          support={intel.support_levels[1] ?? intel.support_levels[0]}
          resistance={intel.resistance_levels[0]}
          currency={currency}
        />
      </div>

      {/* Technical Indicators & News Highlights */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-5">
          <h3 className="mb-4 text-sm font-semibold text-white">Technical & Mention Breakdown</h3>
          <dl className="grid grid-cols-2 gap-y-3 gap-x-6 text-xs">
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <dt className="text-slate-400">RSI (14)</dt>
              <dd className="font-mono tabular-nums text-slate-200">{intel.rsi_14.toFixed(2)}</dd>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <dt className="text-slate-400">MACD / Signal</dt>
              <dd className="font-mono tabular-nums text-slate-200">
                {intel.macd.toFixed(2)} / {intel.macd_signal.toFixed(2)}
              </dd>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <dt className="text-slate-400">SMA 5 / 10 / 20</dt>
              <dd className="font-mono tabular-nums text-slate-200">
                {intel.moving_averages.sma_5.toFixed(1)} · {intel.moving_averages.sma_10.toFixed(1)} ·{' '}
                {intel.moving_averages.sma_20.toFixed(1)}
              </dd>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <dt className="text-slate-400">Mentions (+ / - / =)</dt>
              <dd className="font-mono tabular-nums text-slate-200">
                {intel.total_social_mentions} ({intel.positive_mentions} / {intel.negative_mentions} /{' '}
                {intel.neutral_mentions})
              </dd>
            </div>
            <div className="flex justify-between pb-1">
              <dt className="text-slate-400">Support S2 / S1</dt>
              <dd className="font-mono tabular-nums text-emerald-400">
                {intel.support_levels.map((v) => fmtPrice(v, intel.exchange)).join(' · ')}
              </dd>
            </div>
            <div className="flex justify-between pb-1">
              <dt className="text-slate-400">Resistance R1 / R2</dt>
              <dd className="font-mono tabular-nums text-rose-400">
                {intel.resistance_levels.map((v) => fmtPrice(v, intel.exchange)).join(' · ')}
              </dd>
            </div>
          </dl>
        </div>

        <div className="rounded-lg border border-slate-800 bg-[#111827] p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Financial News Highlights</h3>
            <CategoryLabel category={cats?.news || 'SIMULATED DEMO DATA'} />
          </div>
          <div className="divide-y divide-slate-800/80">
            {intel.news_highlights.map((news, idx) => (
              <div key={idx} className="py-2.5 first:pt-0 last:pb-0">
                <div className="text-xs font-medium text-slate-200">{news.headline}</div>
                <div className="mt-1 flex items-center justify-between text-[11px] text-slate-400">
                  <span>
                    {news.source} · {news.published_utc.slice(0, 10)}
                  </span>
                  <span
                    className={
                      news.sentiment === 'Positive'
                        ? 'text-emerald-400'
                        : news.sentiment === 'Negative'
                        ? 'text-rose-400'
                        : 'text-amber-400'
                    }
                  >
                    {news.sentiment}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function PricePredictionPage({
  summaries,
  predictionHistory,
  selectedTicker,
  onSelectTicker,
}: {
  summaries: StockSummary[];
  predictionHistory?: PredictionRecord[];
  selectedTicker: string;
  onSelectTicker: (ticker: string) => void;
}) {
  const item = summaries.find((s) => s.intelligence.ticker === selectedTicker) || summaries[0];
  if (!item) return null;

  const bestVal = item.validation.find((v) => v.is_best_model) || item.validation[item.validation.length - 1];
  const currency = item.intelligence.currency || (item.intelligence.exchange === 'HKEX' ? 'HKD' : 'KRW');
  const cats = item.data_provenance.categories;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Walk-Forward Out-of-Sample Model Validation</h2>
          <p className="text-xs text-slate-400">
            Mode:{' '}
            <span className={item.data_provenance.is_demo_model ? 'text-amber-400 font-medium' : 'text-emerald-400 font-medium'}>
              {item.data_provenance.validation_mode || 'Genuine Out-of-Sample Walk-Forward'}
            </span>{' '}
            · Window: <span className="font-mono text-slate-300">{bestVal?.split_window}</span> · Price source:{' '}
            <CategoryLabel category={cats?.prices || 'HISTORICAL DATA'} />
          </p>
          {item.data_provenance.features_used && (
            <p className="mt-1 font-mono text-[11px] text-slate-500">
              Model features ({item.data_provenance.features_used.length}): {item.data_provenance.features_used.join(', ')}
            </p>
          )}
        </div>
        <select
          aria-label="Select Ticker for Validation"
          value={item.intelligence.ticker}
          onChange={(e) => onSelectTicker(e.target.value)}
          className="rounded-lg border border-slate-700 bg-[#111827] px-3 py-2 font-mono text-xs text-slate-100 focus:border-sky-500 focus:outline-none"
        >
          {summaries.map((s) => (
            <option key={s.intelligence.ticker} value={s.intelligence.ticker}>
              {s.intelligence.ticker} — {s.intelligence.company_name}
            </option>
          ))}
        </select>
      </div>

      {/* Model Comparison Table */}
      <div className="rounded-lg border border-slate-800 bg-[#111827] overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Model</th>
              <th className="px-4 py-3 text-right font-medium">MAE ({currency})</th>
              <th className="px-4 py-3 text-right font-medium">RMSE ({currency})</th>
              <th className="px-4 py-3 text-right font-medium">MAPE</th>
              <th className="px-4 py-3 text-right font-medium">Directional Accuracy</th>
              <th className="px-4 py-3 font-medium">Selection</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {item.validation.map((v) => (
              <tr key={v.model_name} className={v.is_best_model ? 'bg-sky-950/20' : ''}>
                <td className="px-4 py-3 font-medium text-slate-200">{v.model_name}</td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">{v.mae.toFixed(3)}</td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">{v.rmse.toFixed(3)}</td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">{v.mape.toFixed(2)}%</td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-200">
                  {v.directional_accuracy.toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-xs">
                  {v.is_best_model ? (
                    <span className="font-medium text-sky-400">Selected Best Model</span>
                  ) : (
                    <span className="text-slate-500">Evaluated</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-5 lg:col-span-2 space-y-5">
          <div>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">
                Actual vs Predicted Out-of-Sample ({bestVal?.model_name})
              </h3>
              <div className="flex gap-4 text-xs">
                <span className="text-emerald-400">Actual Close</span>
                <span className="text-sky-400">Predicted Close</span>
              </div>
            </div>
            <ActualVsPredictedChart points={bestVal?.predictions || []} currency={currency} />
          </div>

          <div className="border-t border-slate-800 pt-4">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="text-xs font-semibold text-slate-300">Out-of-Sample Prediction Error (Predicted - Actual)</h4>
              <span className="font-mono text-[11px] text-slate-500">
                Artifact: {item.prediction.artifact_path || 'persisted'}
              </span>
            </div>
            <PredictionErrorChart points={bestVal?.predictions || []} currency={currency} />
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-[#111827] p-5">
          <h3 className="mb-3 text-sm font-semibold text-white">Top Feature Importances</h3>
          <FeatureImportanceChart importances={bestVal?.feature_importances || {}} />
        </div>
      </div>

      {/* Cross-Ticker Forecast Summary & Stored Prediction History */}
      <div className="rounded-lg border border-slate-800 bg-[#111827] overflow-x-auto">
        <div className="border-b border-slate-800 px-5 py-3.5 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Next-Day (n+1) Predictions & Reconciliation Status</h3>
          <span className="text-xs text-slate-400">
            Actuals reconcile automatically once the target trading session closes
          </span>
        </div>
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">Ticker</th>
              <th className="px-4 py-3 font-medium">Forecast Date</th>
              <th className="px-4 py-3 text-right font-medium">Last Close</th>
              <th className="px-4 py-3 text-right font-medium">Predicted Close</th>
              <th className="px-4 py-3 text-right font-medium">% Change</th>
              <th className="px-4 py-3 font-medium">Selected Model</th>
              <th className="px-4 py-3 text-right font-medium">Val RMSE</th>
              <th className="px-4 py-3 text-right font-medium">Actual Close</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {summaries.map((s) => {
              const histMatch = predictionHistory?.find(
                (h) => h.ticker === s.prediction.ticker && h.forecast_target_date === s.prediction.forecast_target_date
              );
              const actual = histMatch?.actual_close ?? s.prediction.actual_close;
              return (
                <tr key={s.prediction.ticker} className="hover:bg-slate-800/30">
                  <td className="px-4 py-2.5 font-mono text-sky-300">{s.prediction.ticker}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">{s.prediction.forecast_target_date}</td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-300">
                    {fmtPrice(s.prediction.last_close, s.intelligence.exchange)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-white">
                    {fmtPrice(s.prediction.predicted_close, s.intelligence.exchange)}
                  </td>
                  <td
                    className={`px-4 py-2.5 text-right font-mono tabular-nums ${
                      s.prediction.predicted_pct_change >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {s.prediction.predicted_pct_change >= 0 ? '+' : ''}
                    {s.prediction.predicted_pct_change.toFixed(2)}%
                  </td>
                  <td className="px-4 py-2.5 text-slate-300">{s.prediction.model_name}</td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-300">
                    {s.prediction.rmse?.toFixed(3) ?? 'N/A'}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-400">
                    {actual !== null && actual !== undefined
                      ? fmtPrice(actual, s.intelligence.exchange)
                      : 'Pending close'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ReportsPage({
  payload,
  onSaveConfig,
}: {
  payload: DashboardPayload;
  onSaveConfig: (stocks: StockConfigItem[]) => Promise<void>;
}) {
  const [exchangeFilter, setExchangeFilter] = useState<'ALL' | 'HKEX' | 'KRX'>('ALL');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [editableStocks, setEditableStocks] = useState<StockConfigItem[]>(() =>
    payload.summaries.map((s) => s.stock)
  );
  const [saving, setSaving] = useState(false);

  const filtered = useMemo(
    () =>
      exchangeFilter === 'ALL'
        ? payload.summaries
        : payload.summaries.filter((s) => s.intelligence.exchange === exchangeFilter),
    [payload.summaries, exchangeFilter]
  );

  const topTrending = [...filtered].sort(
    (a, b) => b.intelligence.total_social_mentions - a.intelligence.total_social_mentions
  )[0];
  const mostPositive = [...filtered].sort(
    (a, b) => b.intelligence.combined_sentiment_score - a.intelligence.combined_sentiment_score
  )[0];
  const mostNegative = [...filtered].sort(
    (a, b) => a.intelligence.combined_sentiment_score - b.intelligence.combined_sentiment_score
  )[0];

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSaveConfig(editableStocks);
      setShowConfigModal(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-slate-400" aria-hidden="true" />
          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-[#111827] p-1">
            {(['ALL', 'HKEX', 'KRX'] as const).map((ex) => (
              <button
                key={ex}
                onClick={() => setExchangeFilter(ex)}
                className={`whitespace-nowrap shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  exchangeFilter === ex ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {ex === 'ALL' ? 'All Exchanges' : ex}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => setShowConfigModal((v) => !v)}
            className="inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 rounded-lg border border-slate-700 bg-[#111827] px-3.5 py-2 text-xs font-medium text-slate-200 hover:border-slate-500"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
            Configure Tickers
          </button>
          <button
            onClick={() => exportSummariesToCsv(filtered, `hengseng_${exchangeFilter.toLowerCase()}_report.csv`)}
            className="inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 rounded-lg border border-slate-700 bg-[#111827] px-3.5 py-2 text-xs font-medium text-slate-200 hover:border-slate-500"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            Export CSV
          </button>
          <a
            href="/api/reports/about-pdf"
            download="HengSeng_About_This_Project.pdf"
            className="inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 rounded-lg border border-slate-700 bg-[#111827] px-3.5 py-2 text-xs font-medium text-slate-200 hover:border-slate-500"
          >
            <FileText className="h-3.5 w-3.5" aria-hidden="true" />
            Project Overview PDF
          </a>
          <button
            onClick={() =>
              exportSummariesToPdf(
                filtered,
                `HengSeng ${exchangeFilter} Sentiment & Price Report`,
                `hengseng_${exchangeFilter.toLowerCase()}_report.pdf`
              )
            }
            className="inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 rounded-lg bg-sky-600 px-3.5 py-2 text-xs font-medium text-white hover:bg-sky-500"
          >
            <FileText className="h-3.5 w-3.5" aria-hidden="true" />
            Export PDF
          </button>
        </div>
      </div>

      {/* Configurable Tickers Drawer */}
      {showConfigModal && (
        <div className="rounded-lg border border-slate-700 bg-[#111827] p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Configurable Stock Universe</h3>
              <p className="text-xs text-slate-400">Enable/disable tickers or edit company labels, then re-run pipeline.</p>
            </div>
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-lg bg-sky-600 px-4 py-2 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Configuration'}
            </button>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {editableStocks.map((st, idx) => (
              <label
                key={st.ticker}
                className="flex items-center justify-between rounded-md border border-slate-800 px-3 py-2 text-xs"
              >
                <span className="font-mono text-slate-200">
                  {st.ticker} <span className="font-sans text-slate-400">· {st.company_name} ({st.exchange})</span>
                </span>
                <input
                  type="checkbox"
                  checked={st.enabled}
                  onChange={(e) => {
                    const next = [...editableStocks];
                    next[idx] = { ...st, enabled: e.target.checked };
                    setEditableStocks(next);
                  }}
                  className="h-4 w-4 accent-sky-500"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Report Highlights */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {topTrending && (
          <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
            <div className="text-xs text-slate-400">Top Trending Stock</div>
            <div className="mt-1 font-mono text-base font-semibold text-white">
              {topTrending.intelligence.ticker} · {topTrending.intelligence.company_name}
            </div>
            <div className="mt-1 font-mono text-xs text-sky-400">
              {topTrending.intelligence.total_social_mentions} social mentions ·{' '}
              <CategoryLabel category={topTrending.data_provenance.categories?.social} />
            </div>
          </div>
        )}
        {mostPositive && (
          <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
            <div className="text-xs text-slate-400">Highest Positive Sentiment</div>
            <div className="mt-1 font-mono text-base font-semibold text-white">
              {mostPositive.intelligence.ticker} · {mostPositive.intelligence.company_name}
            </div>
            <div className="mt-1 font-mono text-xs text-emerald-400">
              Score {mostPositive.intelligence.combined_sentiment_score >= 0 ? '+' : ''}
              {mostPositive.intelligence.combined_sentiment_score.toFixed(3)}
            </div>
          </div>
        )}
        {mostNegative && (
          <div className="rounded-lg border border-slate-800 bg-[#111827] p-4">
            <div className="text-xs text-slate-400">Lowest / Bearish Sentiment</div>
            <div className="mt-1 font-mono text-base font-semibold text-white">
              {mostNegative.intelligence.ticker} · {mostNegative.intelligence.company_name}
            </div>
            <div className="mt-1 font-mono text-xs text-rose-400">
              Score {mostNegative.intelligence.combined_sentiment_score >= 0 ? '+' : ''}
              {mostNegative.intelligence.combined_sentiment_score.toFixed(3)}
            </div>
          </div>
        )}
      </div>

      {/* Agent Audit Logs from SQLite */}
      <div className="rounded-lg border border-slate-800 bg-[#111827] overflow-x-auto">
        <div className="border-b border-slate-800 px-5 py-3.5">
          <h3 className="text-sm font-semibold text-white">Multi-Agent Execution Audit Log (SQLite)</h3>
        </div>
        <table className="w-full text-left text-xs">
          <thead className="border-b border-slate-800 text-slate-400">
            <tr>
              <th className="px-4 py-2.5 font-medium">Agent</th>
              <th className="px-4 py-2.5 font-medium">Ticker</th>
              <th className="px-4 py-2.5 font-medium">Status</th>
              <th className="px-4 py-2.5 font-medium">Summary</th>
              <th className="px-4 py-2.5 text-right font-medium">Duration</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {(payload.logs || []).slice(0, 16).map((log: AgentLogEntry, idx: number) => (
              <tr key={idx}>
                <td className="px-4 py-2 font-mono text-slate-200">{log.agent_name}</td>
                <td className="px-4 py-2 font-mono text-sky-400">{log.ticker || 'ALL'}</td>
                <td className="px-4 py-2 text-emerald-400">{log.status}</td>
                <td className="px-4 py-2 text-slate-400">{log.message}</td>
                <td className="px-4 py-2 text-right font-mono tabular-nums text-slate-400">
                  {log.duration_ms.toFixed(1)} ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Priority 5: Final Internship Assessment Report */}
      {payload.assessment_summary && (
        <div className="rounded-lg border border-slate-800 bg-[#111827] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-white">Final Live Data Integration & Assessment Report</h3>
              <p className="text-xs text-slate-400">
                Audited live connectivity, genuine vs demo counts, walk-forward validation vs naive baseline, and limitations
              </p>
            </div>
            <span className="font-mono text-[11px] text-slate-400">
              Genuine Posts: {payload.assessment_summary.genuine_posts_collected} · Genuine News:{' '}
              {payload.assessment_summary.genuine_news_collected} · Genuine Equities:{' '}
              {payload.assessment_summary.market_data_status.stocks_with_genuine_prices}/
              {payload.assessment_summary.stocks_analyzed}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 text-xs">
            <div className="space-y-2">
              <div className="font-medium text-slate-300">Social, News & Market Providers</div>
              <div className="divide-y divide-slate-800/80 rounded-md border border-slate-800">
                {payload.assessment_summary.social_sources.map((src) => (
                  <div key={src.platform} className="flex items-center justify-between gap-3 px-3 py-2">
                    <span className="text-slate-200">
                      {src.platform} ({src.exchange}) <span className="font-mono text-slate-500">· {src.actor_id}</span>
                    </span>
                    <span className={`font-mono ${src.status === 'connected' ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {src.status} ({src.genuine_posts} genuine)
                    </span>
                  </div>
                ))}
                <div className="flex items-center justify-between gap-3 px-3 py-2">
                  <span className="text-slate-200">Exa Search API</span>
                  <span
                    className={`font-mono ${
                      payload.assessment_summary.exa_status.status === 'connected' ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {payload.assessment_summary.exa_status.status} (
                    {payload.assessment_summary.exa_status.genuine_articles} genuine)
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 px-3 py-2">
                  <span className="text-slate-200">
                    Market Data ({payload.assessment_summary.market_data_status.providers.join(', ')})
                  </span>
                  <span className="font-mono text-emerald-400">
                    {payload.assessment_summary.market_data_status.status} (
                    {payload.assessment_summary.market_data_status.stocks_with_genuine_prices}/
                    {payload.assessment_summary.stocks_analyzed})
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="font-medium text-slate-300">
                Walk-Forward Validation ({payload.assessment_summary.validation_summary.models_beating_or_matching_naive_rmse}/
                {payload.assessment_summary.validation_summary.total_stocks_validated} beat/match Naive RMSE · Mean Dir Acc{' '}
                {payload.assessment_summary.validation_summary.mean_best_directional_accuracy}%)
              </div>
              <ul className="space-y-1.5 rounded-md border border-slate-800 p-3 text-slate-400 list-disc pl-6">
                {payload.assessment_summary.remaining_limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
