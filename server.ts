import express from 'express';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { promisify } from 'node:util';
import 'dotenv/config';

const execFileAsync = promisify(execFile);
const ROOT_DIR = process.cwd();
const DATA_DIR = path.join(ROOT_DIR, 'stock_sentiment_agent', 'data');
const CONFIG_PATH = path.join(ROOT_DIR, 'stock_sentiment_agent', 'config', 'tickers.json');
const REPORTS_DIR = path.join(ROOT_DIR, 'stock_sentiment_agent', 'reports');
const SNAPSHOT_PATH = path.join(DATA_DIR, 'latest_pipeline_output.json');
const DB_PATH = path.join(DATA_DIR, 'hengseng.db');

async function runPythonPipeline(ticker?: string): Promise<any> {
  const args = ['-m', 'stock_sentiment_agent.main'];
  if (ticker) {
    args.push('--ticker', ticker);
  }
  await execFileAsync('python3', args, {
    cwd: ROOT_DIR,
    env: { ...process.env },
    timeout: 120000,
    maxBuffer: 10 * 1024 * 1024,
  });
  if (fs.existsSync(SNAPSHOT_PATH)) {
    return JSON.parse(fs.readFileSync(SNAPSHOT_PATH, 'utf-8'));
  }
  throw new Error('Pipeline finished without generating latest_pipeline_output.json');
}

async function querySqliteTelemetry(limit = 40): Promise<{ logs: any[]; predictionHistory: any[] }> {
  if (!fs.existsSync(DB_PATH)) {
    return { logs: [], predictionHistory: [] };
  }
  const pyCode = `
import json, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.row_factory = sqlite3.Row
logs = [dict(r) for r in conn.execute(
    "SELECT agent_name, status, ticker, message, records_processed, duration_ms, created_at "
    "FROM agent_logs ORDER BY id DESC LIMIT ?", (int(sys.argv[2]),)
).fetchall()]
preds = [dict(r) for r in conn.execute(
    "SELECT ticker, prediction_date, forecast_target_date, last_close, predicted_close, "
    "actual_close, predicted_pct_change, model_name, mae, rmse, mape, directional_accuracy, is_validated "
    "FROM predictions ORDER BY id DESC LIMIT 40"
).fetchall()]
print(json.dumps({"logs": logs, "predictionHistory": preds}))
`;
  try {
    const { stdout } = await execFileAsync('python3', ['-c', pyCode, DB_PATH, String(limit)], {
      cwd: ROOT_DIR,
      timeout: 5000,
    });
    return JSON.parse(stdout || '{"logs":[],"predictionHistory":[]}');
  } catch {
    return { logs: [], predictionHistory: [] };
  }
}

function getSafeEnvStatus() {
  const apifyConfigured = Boolean(process.env.APIFY_API_TOKEN?.trim());
  const exaConfigured = Boolean(process.env.EXA_API_KEY?.trim());
  const databaseUrlConfigured = Boolean(process.env.DATABASE_URL?.trim());
  const modelPathConfigured = Boolean(process.env.MODEL_PATH?.trim());
  return {
    apifyConfigured,
    exaConfigured,
    databaseUrlConfigured,
    modelPathConfigured,
  };
}

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req, res) => {
    const env = getSafeEnvStatus();
    res.json({
      status: 'ok',
      ...env,
      snapshotExists: fs.existsSync(SNAPSHOT_PATH),
      databaseExists: fs.existsSync(DB_PATH),
    });
  });

  app.get('/api/diagnostics/apify', async (_req, res) => {
    try {
      const pyCode = `
import json
from stock_sentiment_agent.collectors.apify_collector import ApifyBaseCollector
c = ApifyBaseCollector()
probe = c.probe_sources()
test_runs = {}
targets = {
    "Daum_table": ("apify/web-scraper", {
        "startUrls": [{"url": "https://finance.daum.net/quotes/A005930#talks"}],
        "maxRequestsPerCrawl": 1,
        "waitUntil": ["networkidle2"],
        "pageFunction": "async function pageFunction(context) { await context.waitFor(3000); const tables = Array.from(document.querySelectorAll('table')); const talkRows = []; tables.forEach((tbl) => { const txt = tbl.innerText || ''; if (txt.includes('제목') || txt.includes('작성자') || tbl.querySelectorAll('tr').length > 4) { Array.from(tbl.querySelectorAll('tbody tr, tr')).slice(0, 12).forEach((tr) => { const tds = Array.from(tr.querySelectorAll('td')).map(td => td.innerText?.trim()); const a = tr.querySelector('a'); if (tds.length >= 3) talkRows.push({ tds, href: a ? a.getAttribute('href') : null, cls: tr.className }); }); } }); return [{ title: document.title, tableCount: tables.length, talkRowsCount: talkRows.length, talkRows: talkRows.slice(0, 8) }]; }"
    }),
}
for name, (actor, inp) in targets.items():
    items, err = c.run_actor_and_fetch_items(actor, inp, max_items=2, timeout_sec=45)
    test_runs[name] = {"actor": actor, "ok": err is None, "items_count": len(items), "items": items[:1], "error": err}
print(json.dumps({"auth": c.verify_authentication(), "probe": probe, "test_runs": test_runs}, ensure_ascii=False))
`;
      const { stdout } = await execFileAsync('python3', ['-c', pyCode], {
        cwd: ROOT_DIR,
        env: { ...process.env },
        timeout: 90000,
      });
      res.json(JSON.parse(stdout));
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Apify diagnostics failed' });
    }
  });

  app.get('/api/dashboard', async (_req, res) => {
    try {
      let data: any;
      if (fs.existsSync(SNAPSHOT_PATH)) {
        data = JSON.parse(fs.readFileSync(SNAPSHOT_PATH, 'utf-8'));
      } else {
        data = await runPythonPipeline();
      }
      const { logs, predictionHistory } = await querySqliteTelemetry(40);
      res.json({
        ...data,
        logs,
        prediction_history: predictionHistory,
        environment: getSafeEnvStatus(),
      });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to load dashboard data' });
    }
  });

  app.post('/api/pipeline/run', async (req, res) => {
    try {
      const ticker = typeof req.body?.ticker === 'string' ? req.body.ticker.trim() : undefined;
      const data = await runPythonPipeline(ticker || undefined);
      const { logs, predictionHistory } = await querySqliteTelemetry(40);
      res.json({
        ...data,
        logs,
        prediction_history: predictionHistory,
        environment: getSafeEnvStatus(),
      });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Pipeline execution failed' });
    }
  });

  app.get('/api/stocks/config', (_req, res) => {
    try {
      if (!fs.existsSync(CONFIG_PATH)) {
        return res.json([]);
      }
      const items = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
      res.json(items);
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to read ticker config' });
    }
  });

  app.post('/api/stocks/config', (req, res) => {
    try {
      const stocks = req.body?.stocks;
      if (!Array.isArray(stocks) || stocks.length === 0) {
        return res.status(400).json({ error: 'Request body must include a non-empty stocks array.' });
      }
      fs.writeFileSync(CONFIG_PATH, JSON.stringify(stocks, null, 2), 'utf-8');
      res.json({ status: 'saved', count: stocks.length });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to save ticker config' });
    }
  });

  app.get('/api/reports/csv', async (_req, res) => {
    const csvPath = path.join(REPORTS_DIR, 'hengseng_report.csv');
    if (!fs.existsSync(csvPath)) {
      await runPythonPipeline();
    }
    res.download(csvPath, 'hengseng_report.csv');
  });

  app.get('/api/reports/pdf', async (_req, res) => {
    const pdfPath = path.join(REPORTS_DIR, 'hengseng_report.pdf');
    if (!fs.existsSync(pdfPath)) {
      await runPythonPipeline();
    }
    res.download(pdfPath, 'hengseng_report.pdf');
  });

  app.get('/api/reports/about-pdf', async (_req, res) => {
    try {
      const aboutPdfPath = path.join(REPORTS_DIR, 'about_hengseng_project.pdf');
      if (!fs.existsSync(aboutPdfPath)) {
        await execFileAsync('node', [path.join(ROOT_DIR, 'scripts', 'generate_about_pdf.mjs')], {
          cwd: ROOT_DIR,
          timeout: 15000,
        });
      }
      res.download(aboutPdfPath, 'HengSeng_About_This_Project.pdf');
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to generate project overview PDF' });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const hmrDisabled = process.env.DISABLE_HMR === 'true';
    const wsDisabled = process.env.DISABLE_WS === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: !hmrDisabled,
        ws: wsDisabled ? false : { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const PORT = 3000;
  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`HengSeng Terminal running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
