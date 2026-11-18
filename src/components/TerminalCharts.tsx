import React, { useState } from 'react';
import type { PriceBar, StockSummary, WalkForwardPredictionPoint } from '../types';

export function SentimentComparisonChart({ summaries }: { summaries: StockSummary[] }) {
  const [hovered, setHovered] = useState<number | null>(null);
  if (!summaries.length) return null;

  const width = 760;
  const height = 270;
  const pad = { top: 24, right: 20, bottom: 48, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const zeroY = pad.top + innerH / 2;
  const maxAbs = Math.max(0.45, ...summaries.map((s) => Math.abs(s.intelligence.combined_sentiment_score)));
  const barW = Math.min(42, innerW / summaries.length - 12);

  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto select-none">
        {[-1, -0.5, 0, 0.5, 1].map((tick) => {
          const val = tick * maxAbs;
          const y = zeroY - (val / maxAbs) * (innerH / 2);
          return (
            <g key={tick}>
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={y}
                y2={y}
                stroke={tick === 0 ? '#475569' : '#1E293B'}
                strokeDasharray={tick === 0 ? undefined : '3 3'}
                strokeWidth={1}
              />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {summaries.map((s, idx) => {
          const score = s.intelligence.combined_sentiment_score;
          const cx = pad.left + ((idx + 0.5) * innerW) / summaries.length;
          const barH = (Math.abs(score) / maxAbs) * (innerH / 2);
          const y = score >= 0 ? zeroY - barH : zeroY;
          const dir = s.intelligence.market_direction;
          const color = dir === 'Bullish' ? '#10B981' : dir === 'Bearish' ? '#EF4444' : '#F59E0B';
          const isHover = hovered === idx;

          return (
            <g
              key={s.intelligence.ticker}
              onMouseEnter={() => setHovered(idx)}
              onMouseLeave={() => setHovered(null)}
              className="cursor-pointer"
            >
              <rect
                x={cx - barW / 2}
                y={y}
                width={barW}
                height={Math.max(2, barH)}
                rx={3}
                fill={color}
                fillOpacity={isHover ? 1 : 0.82}
              />
              <text
                x={cx}
                y={height - 20}
                textAnchor="middle"
                className="fill-slate-300 text-[10px] font-mono"
              >
                {s.intelligence.ticker.replace('.HK', '').replace('.KS', '')}
              </text>
              <text
                x={cx}
                y={height - 7}
                textAnchor="middle"
                className="fill-slate-500 text-[9px]"
              >
                {s.intelligence.exchange}
              </text>
            </g>
          );
        })}
      </svg>

      {hovered !== null && summaries[hovered] && (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 pt-2 text-xs text-slate-300">
          <span>
            <strong className="font-semibold text-white">{summaries[hovered].intelligence.company_name}</strong> (
            <span className="font-mono">{summaries[hovered].intelligence.ticker}</span>)
          </span>
          <span className="font-mono tabular-nums text-slate-400">
            Combined: <strong className="text-slate-100">{summaries[hovered].intelligence.combined_sentiment_score.toFixed(3)}</strong>
            {' · '}Social: {summaries[hovered].intelligence.social_sentiment_score.toFixed(3)}
            {' · '}News: {summaries[hovered].intelligence.news_sentiment_score.toFixed(3)}
            {' · '}Direction: {summaries[hovered].intelligence.market_direction}
          </span>
        </div>
      )}
    </div>
  );
}

export function PriceAndForecastChart({
  history,
  predictedClose,
  forecastDate,
  support,
  resistance,
  currency,
}: {
  history: PriceBar[];
  predictedClose: number;
  forecastDate: string;
  support: number;
  resistance: number;
  currency: string;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  if (!history.length) return null;

  const width = 760;
  const height = 280;
  const pad = { top: 20, right: 64, bottom: 38, left: 60 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const allVals = [...history.map((h) => h.close), predictedClose, support, resistance];
  const minVal = Math.min(...allVals) * 0.985;
  const maxVal = Math.max(...allVals) * 1.015;
  const range = Math.max(1e-6, maxVal - minVal);
  const totalPoints = history.length + 1;

  const xAt = (idx: number) => pad.left + (idx / Math.max(1, totalPoints - 1)) * innerW;
  const yAt = (v: number) => pad.top + innerH - ((v - minVal) / range) * innerH;

  const linePath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(1)} ${yAt(h.close).toFixed(1)}`)
    .join(' ');
  const lastX = xAt(history.length - 1);
  const lastY = yAt(history[history.length - 1].close);
  const predX = xAt(history.length);
  const predY = yAt(predictedClose);
  const decimals = currency === 'HKD' ? 2 : 0;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = minVal + t * range;
          const y = yAt(v);
          return (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
                {v.toFixed(decimals)}
              </text>
            </g>
          );
        })}

        {/* Support & Resistance Lines */}
        <line
          x1={pad.left}
          x2={width - pad.right}
          y1={yAt(support)}
          y2={yAt(support)}
          stroke="#10B981"
          strokeDasharray="4 4"
          strokeOpacity={0.65}
        />
        <text x={width - pad.right + 6} y={yAt(support) + 3} className="fill-emerald-400 text-[10px] font-mono">
          S {support.toFixed(decimals)}
        </text>

        <line
          x1={pad.left}
          x2={width - pad.right}
          y1={yAt(resistance)}
          y2={yAt(resistance)}
          stroke="#EF4444"
          strokeDasharray="4 4"
          strokeOpacity={0.65}
        />
        <text x={width - pad.right + 6} y={yAt(resistance) + 3} className="fill-rose-400 text-[10px] font-mono">
          R {resistance.toFixed(decimals)}
        </text>

        {/* Historical Close Curve */}
        <path d={linePath} fill="none" stroke="#38BDF8" strokeWidth={2} />

        {/* Forecast Connector */}
        <line
          x1={lastX}
          y1={lastY}
          x2={predX}
          y2={predY}
          stroke="#F59E0B"
          strokeWidth={2}
          strokeDasharray="4 3"
        />
        <circle cx={predX} cy={predY} r={4.5} fill="#F59E0B" />
        <text x={predX - 4} y={predY - 9} textAnchor="end" className="fill-amber-300 text-[10px] font-mono">
          n+1: {predictedClose.toFixed(decimals)}
        </text>

        {/* Interactive hover hitboxes */}
        {history.map((bar, i) => (
          <g key={bar.trading_date}>
            <circle
              cx={xAt(i)}
              cy={yAt(bar.close)}
              r={hoverIdx === i ? 4 : 2}
              fill={hoverIdx === i ? '#38BDF8' : 'transparent'}
            />
            <rect
              x={xAt(i) - innerW / (totalPoints * 2)}
              y={pad.top}
              width={Math.max(8, innerW / totalPoints)}
              height={innerH}
              fill="transparent"
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
            />
          </g>
        ))}

        <text x={pad.left} y={height - 10} className="fill-slate-500 text-[10px] font-mono">
          {history[0].trading_date}
        </text>
        <text x={lastX} y={height - 10} textAnchor="middle" className="fill-slate-400 text-[10px] font-mono">
          {history[history.length - 1].trading_date}
        </text>
        <text x={predX} y={height - 10} textAnchor="end" className="fill-amber-400 text-[10px] font-mono">
          {forecastDate}
        </text>
      </svg>

      {hoverIdx !== null && history[hoverIdx] && (
        <div className="mt-2 flex flex-wrap gap-4 border-t border-slate-800 pt-2 text-xs font-mono tabular-nums text-slate-400">
          <span className="text-slate-200">{history[hoverIdx].trading_date}</span>
          <span>O {history[hoverIdx].open.toFixed(decimals)}</span>
          <span>H {history[hoverIdx].high.toFixed(decimals)}</span>
          <span>L {history[hoverIdx].low.toFixed(decimals)}</span>
          <span className="text-sky-300">C {history[hoverIdx].close.toFixed(decimals)}</span>
          <span>Vol {history[hoverIdx].volume.toLocaleString()}</span>
        </div>
      )}
    </div>
  );
}

export function ActualVsPredictedChart({
  points,
  currency,
}: {
  points: WalkForwardPredictionPoint[];
  currency: string;
}) {
  if (!points.length) return null;
  const width = 680;
  const height = 240;
  const pad = { top: 20, right: 20, bottom: 36, left: 60 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const vals = points.flatMap((p) => [p.actual_close, p.predicted_close]);
  const minV = Math.min(...vals) * 0.99;
  const maxV = Math.max(...vals) * 1.01;
  const range = Math.max(1e-6, maxV - minV);
  const decimals = currency === 'HKD' ? 2 : 0;

  const xAt = (i: number) => pad.left + (i / Math.max(1, points.length - 1)) * innerW;
  const yAt = (v: number) => pad.top + innerH - ((v - minV) / range) * innerH;

  const actualPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yAt(p.actual_close)}`).join(' ');
  const predPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i)} ${yAt(p.predicted_close)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto select-none">
      {[0, 0.5, 1].map((t) => {
        const v = minV + t * range;
        const y = yAt(v);
        return (
          <g key={t}>
            <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
            <text x={pad.left - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
              {v.toFixed(decimals)}
            </text>
          </g>
        );
      })}
      <path d={actualPath} fill="none" stroke="#10B981" strokeWidth={2} />
      <path d={predPath} fill="none" stroke="#38BDF8" strokeWidth={2} strokeDasharray="5 3" />
      {points.map((p, i) => (
        <g key={p.prediction_date}>
          <circle cx={xAt(i)} cy={yAt(p.actual_close)} r={2.8} fill="#10B981" />
          <circle cx={xAt(i)} cy={yAt(p.predicted_close)} r={2.8} fill="#38BDF8" />
        </g>
      ))}
      <text x={pad.left} y={height - 10} className="fill-slate-500 text-[10px] font-mono">
        {points[0].prediction_date}
      </text>
      <text x={width - pad.right} y={height - 10} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
        {points[points.length - 1].prediction_date}
      </text>
    </svg>
  );
}

export function PredictionErrorChart({
  points,
  currency,
}: {
  points: WalkForwardPredictionPoint[];
  currency: string;
}) {
  if (!points.length) return null;
  const width = 680;
  const height = 180;
  const pad = { top: 18, right: 20, bottom: 32, left: 60 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const zeroY = pad.top + innerH / 2;
  const errors = points.map((p) => p.predicted_close - p.actual_close);
  const maxAbs = Math.max(0.01, ...errors.map((e) => Math.abs(e)));
  const decimals = currency === 'HKD' ? 2 : 0;
  const barW = Math.min(26, innerW / points.length - 6);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto select-none">
      {[-1, 0, 1].map((t) => {
        const val = t * maxAbs;
        const y = zeroY - (val / maxAbs) * (innerH / 2);
        return (
          <g key={t}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y}
              y2={y}
              stroke={t === 0 ? '#475569' : '#1E293B'}
              strokeDasharray={t === 0 ? undefined : '3 3'}
            />
            <text x={pad.left - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
              {val >= 0 ? `+${val.toFixed(decimals)}` : val.toFixed(decimals)}
            </text>
          </g>
        );
      })}
      {points.map((p, idx) => {
        const err = p.predicted_close - p.actual_close;
        const cx = pad.left + ((idx + 0.5) * innerW) / points.length;
        const barH = (Math.abs(err) / maxAbs) * (innerH / 2);
        const y = err >= 0 ? zeroY - barH : zeroY;
        return (
          <rect
            key={p.prediction_date}
            x={cx - barW / 2}
            y={y}
            width={Math.max(4, barW)}
            height={Math.max(2, barH)}
            rx={2}
            fill={err >= 0 ? '#38BDF8' : '#F59E0B'}
            fillOpacity={0.85}
          />
        );
      })}
      <text x={pad.left} y={height - 8} className="fill-slate-500 text-[10px] font-mono">
        {points[0].prediction_date}
      </text>
      <text x={width - pad.right} y={height - 8} textAnchor="end" className="fill-slate-500 text-[10px] font-mono">
        {points[points.length - 1].prediction_date}
      </text>
    </svg>
  );
}

export function FeatureImportanceChart({ importances }: { importances: Record<string, number> }) {
  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  if (!sorted.length) {
    return <p className="text-xs text-slate-500">Feature importances are not applicable to the naive baseline.</p>;
  }
  const maxImp = Math.max(0.01, ...sorted.map(([, v]) => v));

  return (
    <div className="space-y-2.5">
      {sorted.map(([feature, val]) => (
        <div key={feature} className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-slate-300">{feature}</span>
            <span className="font-mono tabular-nums text-slate-400">{(val * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-sm bg-slate-800">
            <div
              className="h-1.5 rounded-sm bg-sky-500"
              style={{ width: `${Math.min(100, (val / maxImp) * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
