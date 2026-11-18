import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  BarChart3,
  Building2,
  FileSpreadsheet,
  Globe2,
  LineChart,
  Menu,
  MessageSquareText,
  RefreshCw,
  X,
} from 'lucide-react';
import type { DashboardPageId, DashboardPayload, StockConfigItem } from './types';
import {
  ExchangeMarketPage,
  OverviewPage,
  PricePredictionPage,
  ReportsPage,
  SentimentAnalysisPage,
  StockIntelligencePage,
} from './components/DashboardPages';

const NAV_ITEMS: Array<{ id: DashboardPageId; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'hkex', label: 'Hong Kong Market', icon: Building2 },
  { id: 'krx', label: 'South Korea Market', icon: Globe2 },
  { id: 'sentiment', label: 'Sentiment Analysis', icon: MessageSquareText },
  { id: 'intelligence', label: 'Stock Intelligence', icon: Activity },
  { id: 'prediction', label: 'Price Prediction', icon: LineChart },
  { id: 'reports', label: 'Reports', icon: FileSpreadsheet },
];

export default function App() {
  const [page, setPage] = useState<DashboardPageId>('overview');
  const [payload, setPayload] = useState<DashboardPayload | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>('0700.HK');
  const [loading, setLoading] = useState<boolean>(true);
  const [runningPipeline, setRunningPipeline] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/dashboard');
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      const data: DashboardPayload = await res.json();
      setPayload(data);
      if (data.summaries?.length && !data.summaries.some((s) => s.intelligence.ticker === selectedTicker)) {
        setSelectedTicker(data.summaries[0].intelligence.ticker);
      }
    } catch (err: any) {
      setError(err?.message || 'Unable to load dashboard telemetry');
    } finally {
      setLoading(false);
    }
  }, [selectedTicker]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const handleRunPipeline = async () => {
    setRunningPipeline(true);
    setError(null);
    try {
      const res = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Pipeline failed (${res.status})`);
      }
      const data: DashboardPayload = await res.json();
      setPayload(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to execute agent pipeline');
    } finally {
      setRunningPipeline(false);
    }
  };

  const handleSelectStockFromTable = (ticker: string) => {
    setSelectedTicker(ticker);
    setPage('intelligence');
  };

  const handleSaveConfig = async (stocks: StockConfigItem[]) => {
    const res = await fetch('/api/stocks/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stocks }),
    });
    if (!res.ok) {
      throw new Error('Failed to save stock universe configuration');
    }
    await handleRunPipeline();
  };

  const currentNav = NAV_ITEMS.find((n) => n.id === page) || NAV_ITEMS[0];
  const anySimulated = Boolean(payload?.summaries?.some((s) => s.data_provenance?.contains_simulated_data));
  const unavailableSources = (payload?.api_connection_status || []).filter((s) => s.status === 'unavailable');

  return (
    <div className="min-h-screen bg-[#090D16] text-slate-100 flex">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0 border-r border-slate-800/80 bg-[#0C1220]">
        <div className="flex h-16 items-center px-5 border-b border-slate-800/80">
          <span className="font-display text-base font-bold tracking-tight text-white whitespace-nowrap shrink-0">
            HengSeng Terminal
          </span>
        </div>

        <nav aria-label="Dashboard Navigation" className="flex-1 space-y-1 px-3 py-4">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = page === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-sky-600/15 text-sky-300 border border-sky-500/30'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-slate-800/80 px-5 py-4 text-[11px] text-slate-500 space-y-1">
          <div>Exchanges: HKEX · KRX</div>
          <div>
            Apify: {payload?.environment?.apifyConfigured ? 'Configured' : 'Unavailable'} · Exa:{' '}
            {payload?.environment?.exaConfigured ? 'Configured' : 'Unavailable'}
          </div>
          <div>Prices: Yahoo Finance (Historical)</div>
        </div>
      </aside>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="fixed inset-0 bg-black/70" onClick={() => setMobileMenuOpen(false)} />
          <aside className="relative z-10 flex w-64 flex-col bg-[#0C1220] border-r border-slate-800">
            <div className="flex h-16 items-center justify-between px-5 border-b border-slate-800">
              <span className="font-display text-base font-bold text-white">HengSeng Terminal</span>
              <button
                onClick={() => setMobileMenuOpen(false)}
                aria-label="Close menu"
                className="rounded-md p-1.5 text-slate-400 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <nav className="flex-1 space-y-1 px-3 py-4">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = page === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setPage(item.id);
                      setMobileMenuOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-medium ${
                      active ? 'bg-sky-600/15 text-sky-300' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-64 flex flex-col min-w-0">
        {/* Top Bar Contract: Breadcrumb on left, 1 Primary CTA on right */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-8 border-b border-slate-800/80 bg-[#090D16]/95 px-6 backdrop-blur-sm">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation"
              className="lg:hidden rounded-md p-1.5 text-slate-400 hover:text-white"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-2 text-sm truncate">
              <span className="text-slate-400 whitespace-nowrap">HengSeng Terminal</span>
              <span className="text-slate-600">/</span>
              <h1 className="font-display font-semibold text-white truncate">{currentNav.label}</h1>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={handleRunPipeline}
              disabled={runningPipeline}
              className="inline-flex items-center gap-2 whitespace-nowrap shrink-0 rounded-lg bg-sky-600 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-sky-500 focus-visible:outline-2 focus-visible:outline-sky-400 disabled:opacity-60"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${runningPipeline ? 'animate-spin' : ''}`} aria-hidden="true" />
              <span>{runningPipeline ? 'Running Agents...' : 'Run Agent Pipeline'}</span>
            </button>
          </div>
        </header>

        <main className="flex-1 px-6 py-6 max-w-[1440px] w-full mx-auto space-y-6">
          {/* 3-Category Data Provenance & API Status Warning */}
          {anySimulated && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 px-4 py-3 text-xs text-amber-200/90 space-y-1">
              <div>
                <strong>API Provenance Warning:</strong>{' '}
                {unavailableSources.length > 0
                  ? `${unavailableSources.map((s) => s.source).join(', ')} returned unavailable status (missing APIFY_API_TOKEN / EXA_API_KEY or external restriction).`
                  : 'External social/news credentials are not configured.'}
              </div>
              <div className="text-amber-200/75">
                Categories active: <strong>HISTORICAL DATA</strong> (market prices),{' '}
                <strong>
                  {payload?.assessment_summary?.genuine_news_collected ? 'LIVE DATA' : 'SIMULATED DEMO DATA'}
                </strong>{' '}
                (Exa financial news), and{' '}
                <strong>
                  {payload?.assessment_summary?.genuine_posts_collected ? 'LIVE DATA' : 'SIMULATED DEMO DATA'}
                </strong>{' '}
                (social demonstration corpus). Simulated records are labeled and excluded from genuine ML training.
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-xs text-rose-200 flex items-center justify-between gap-4">
              <span>{error}</span>
              <button
                onClick={loadDashboard}
                className="whitespace-nowrap shrink-0 rounded-md border border-rose-400/40 px-3 py-1 text-xs font-medium hover:bg-rose-900/40"
              >
                Retry
              </button>
            </div>
          )}

          {loading && !payload ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
                {[1, 2, 3, 4, 5].map((k) => (
                  <div key={k} className="h-24 rounded-lg border border-slate-800 bg-[#111827] animate-pulse" />
                ))}
              </div>
              <div className="h-72 rounded-lg border border-slate-800 bg-[#111827] animate-pulse" />
            </div>
          ) : payload && payload.summaries?.length ? (
            <>
              {page === 'overview' && (
                <OverviewPage payload={payload} onSelectStock={handleSelectStockFromTable} />
              )}
              {page === 'hkex' && (
                <ExchangeMarketPage
                  exchange="HKEX"
                  summaries={payload.summaries}
                  onSelectStock={handleSelectStockFromTable}
                />
              )}
              {page === 'krx' && (
                <ExchangeMarketPage
                  exchange="KRX"
                  summaries={payload.summaries}
                  onSelectStock={handleSelectStockFromTable}
                />
              )}
              {page === 'sentiment' && <SentimentAnalysisPage summaries={payload.summaries} />}
              {page === 'intelligence' && (
                <StockIntelligencePage
                  summaries={payload.summaries}
                  selectedTicker={selectedTicker}
                  onSelectTicker={setSelectedTicker}
                />
              )}
              {page === 'prediction' && (
                <PricePredictionPage
                  summaries={payload.summaries}
                  predictionHistory={payload.prediction_history}
                  selectedTicker={selectedTicker}
                  onSelectTicker={setSelectedTicker}
                />
              )}
              {page === 'reports' && <ReportsPage payload={payload} onSaveConfig={handleSaveConfig} />}
            </>
          ) : (
            <div className="rounded-lg border border-slate-800 bg-[#111827] p-8 text-center">
              <p className="text-sm text-slate-300">No equity intelligence snapshots have been generated yet.</p>
              <button
                onClick={handleRunPipeline}
                className="mt-4 rounded-lg bg-sky-600 px-4 py-2 text-xs font-medium text-white hover:bg-sky-500"
              >
                Run Agent Pipeline
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
