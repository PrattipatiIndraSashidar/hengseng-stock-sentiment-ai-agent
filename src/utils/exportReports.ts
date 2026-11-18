import { jsPDF } from 'jspdf';
import type { StockSummary } from '../types';

export function exportSummariesToCsv(summaries: StockSummary[], filename = 'hengseng_sentiment_report.csv'): void {
  const headers = [
    'Ticker',
    'Company',
    'Exchange',
    'Currency',
    'Latest Close',
    'Predicted n+1 Close',
    'Predicted % Change',
    'Forecast Target Date',
    'Best ML Model',
    'Social Mentions',
    'Social Sentiment',
    'News Sentiment',
    'Combined Sentiment',
    'Bullish Prob',
    'Bearish Prob',
    'Market Direction',
    'Trading Position',
    'RSI 14',
    'MACD',
    'Support Level',
    'Resistance Level',
    'Validation RMSE',
    'Validation Dir Accuracy %',
    'Price Data Category',
    'Social Data Category',
    'News Data Category',
  ];

  const escape = (val: unknown) => {
    const str = String(val ?? '');
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const rows = summaries.map((s) => {
    const i = s.intelligence;
    const p = s.prediction;
    const cats = s.data_provenance.categories;
    return [
      i.ticker,
      i.company_name,
      i.exchange,
      i.currency || (i.exchange === 'HKEX' ? 'HKD' : 'KRW'),
      i.latest_price,
      p.predicted_close,
      p.predicted_pct_change,
      p.forecast_target_date,
      p.model_name,
      i.total_social_mentions,
      i.social_sentiment_score,
      i.news_sentiment_score,
      i.combined_sentiment_score,
      i.bullish_probability,
      i.bearish_probability,
      i.market_direction,
      i.mentioned_trading_position,
      i.rsi_14,
      i.macd,
      i.support_levels[1] ?? i.support_levels[0],
      i.resistance_levels[0],
      p.rmse ?? '',
      p.directional_accuracy ?? '',
      cats?.prices || s.data_provenance.prices.data_category || 'HISTORICAL DATA',
      cats?.social || 'SIMULATED DEMO DATA',
      cats?.news || s.data_provenance.news.data_category || 'SIMULATED DEMO DATA',
    ]
      .map(escape)
      .join(',');
  });

  const csvContent = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportSummariesToPdf(
  summaries: StockSummary[],
  reportTitle = 'HengSeng Sentiment & Price Prediction Report',
  filename = 'hengseng_sentiment_report.pdf'
): void {
  const doc = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();

  doc.setFillColor(9, 13, 22);
  doc.rect(0, 0, pageWidth, 64, 'F');
  doc.setTextColor(241, 245, 249);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.text(reportTitle, 32, 32);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(148, 163, 184);
  const anySim = summaries.some((s) => s.data_provenance.contains_simulated_data);
  doc.text(
    `Generated UTC: ${new Date().toISOString()}  |  Equities: ${summaries.length}  |  Provenance: ${
      anySim ? 'HISTORICAL DATA (Prices) + SIMULATED DEMO DATA (Social/News)' : 'LIVE DATA + HISTORICAL DATA'
    }`,
    32,
    49
  );

  const cols = [
    { label: 'Ticker', x: 28 },
    { label: 'Company', x: 88 },
    { label: 'Exch', x: 185 },
    { label: 'Close', x: 222 },
    { label: 'Pred n+1', x: 285 },
    { label: '% Chg', x: 350 },
    { label: 'Mentions', x: 398 },
    { label: 'Sent', x: 448 },
    { label: 'Direction', x: 495 },
    { label: 'Best Model', x: 552 },
    { label: 'Dir Acc', x: 642 },
    { label: 'Provenance (Price / Social / News)', x: 690 },
  ];

  let y = 92;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(30, 41, 59);
  cols.forEach((c) => doc.text(c.label, c.x, y));

  doc.setDrawColor(203, 213, 225);
  doc.line(28, y + 6, pageWidth - 28, y + 6);
  y += 22;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);

  summaries.forEach((s, idx) => {
    if (y > 545) {
      doc.addPage();
      y = 48;
    }
    if (idx % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(26, y - 11, pageWidth - 52, 18, 'F');
    }
    const i = s.intelligence;
    const p = s.prediction;
    const cats = s.data_provenance.categories;
    doc.setTextColor(15, 23, 42);
    doc.text(i.ticker, 28, y);
    doc.text(i.company_name.slice(0, 16), 88, y);
    doc.text(i.exchange, 185, y);
    doc.text(i.latest_price.toFixed(i.exchange === 'HKEX' ? 2 : 0), 222, y);
    doc.text(p.predicted_close.toFixed(i.exchange === 'HKEX' ? 2 : 0), 285, y);

    if (p.predicted_pct_change >= 0) {
      doc.setTextColor(5, 150, 105);
    } else {
      doc.setTextColor(220, 38, 38);
    }
    doc.text(`${p.predicted_pct_change >= 0 ? '+' : ''}${p.predicted_pct_change.toFixed(2)}%`, 350, y);

    doc.setTextColor(15, 23, 42);
    doc.text(String(i.total_social_mentions), 398, y);
    doc.text(`${i.combined_sentiment_score >= 0 ? '+' : ''}${i.combined_sentiment_score.toFixed(3)}`, 448, y);
    doc.text(i.market_direction, 495, y);
    doc.text(p.model_name.replace(' Regressor', ''), 552, y);
    doc.text(`${p.directional_accuracy ?? 0}%`, 642, y);

    const pShort = (cats?.prices || 'HISTORICAL').replace(' DATA', '');
    const sShort = (cats?.social || 'SIMULATED DEMO').replace(' DATA', '');
    const nShort = (cats?.news || 'SIMULATED DEMO').replace(' DATA', '');
    doc.text(`${pShort} / ${sShort} / ${nShort}`, 690, y);

    y += 20;
  });

  doc.save(filename);
}
