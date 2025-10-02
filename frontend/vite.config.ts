import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 9243,
    open: true, // Automatically open browser when dev server starts
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9127',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:9127',
        ws: true,
      },
      '/configs': {
        target: 'http://127.0.0.1:9127',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
