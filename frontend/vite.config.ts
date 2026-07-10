import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['msi', 'msi.tail5c01da.ts.net', '100.95.69.79'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:18180',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
