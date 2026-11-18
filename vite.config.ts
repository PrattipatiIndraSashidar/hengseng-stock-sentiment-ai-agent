import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

function parseAppUrl(rawUrl?: string): URL | null {
  if (!rawUrl || rawUrl === 'MY_APP_URL') {
    return null;
  }
  try {
    const parsed = new URL(rawUrl.trim());
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

export default defineConfig(({mode}) => {
  const rootDir = import.meta.dirname;
  const env = loadEnv(mode, rootDir, '');
  const hmrDisabled = (process.env.DISABLE_HMR ?? env.DISABLE_HMR) === 'true';
  const wsDisabled = (process.env.DISABLE_WS ?? env.DISABLE_WS) === 'true';
  const appUrl = parseAppUrl(process.env.APP_URL ?? env.APP_URL);
  const isHttpsPreview = appUrl?.protocol === 'https:';
  const previewClientPort = appUrl
    ? appUrl.port
      ? Number(appUrl.port)
      : isHttpsPreview
        ? 443
        : 80
    : 3000;

  return {
    plugins: [
      react(),
      tailwindcss(),
      // Copies src/assets into dist/ so root-relative URLs like
      // "/src/assets/images/..." (e.g. generate_image output) still resolve
      // in production builds. Do not remove.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(rootDir, 'dist');
          const srcAssets = path.resolve(rootDir, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
        },
      },
    ],
    resolve: {
      alias: {
        '@': path.resolve(rootDir, '.'),
      },
    },
    server: {
      host: '0.0.0.0',
      port: 3000,
      ...(appUrl ? {allowedHosts: [appUrl.hostname]} : {}),
      // Disable file watching / HMR updates when DISABLE_HMR=true while keeping
      // the WebSocket transport bound to the Express HTTP server so /@vite/client
      // connects cleanly instead of probing unproxied port 24678.
      hmr: !hmrDisabled,
      ws: wsDisabled
        ? false
        : appUrl
          ? {
              protocol: isHttpsPreview ? 'wss' : 'ws',
              host: appUrl.hostname,
              port: 3000,
              clientPort: previewClientPort,
            }
          : {
              protocol: 'ws',
              port: 3000,
              clientPort: 3000,
            },
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: hmrDisabled ? null : {},
    },
  };
});
