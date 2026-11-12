import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies /api to the FastAPI backend so no CORS setup is needed in dev.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        embed: 'embed.html',
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
