/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        flux: {
          primary: '#6366f1',
          secondary: '#8b5cf6',
          accent: '#06b6d4',
          neutral: '#64748b',
        },
        consciousness: {
          thought: '#f59e0b',
          memory: '#10b981',
          curiosity: '#ef4444',
          dream: '#8b5cf6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
