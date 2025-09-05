/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './**/*.js'
  ],
  theme: {
    extend: {
      colors: {
        clay: {
          50: '#f8f9fa',
          100: '#e9ecef',
          200: '#dee2e6',
          300: '#ced4da',
          400: '#adb5bd',
          500: '#6c757d',
          600: '#495057',
          700: '#343a40',
          800: '#212529',
          900: '#1a1d20',
        }
      },
      boxShadow: {
        'clay-inset': 'inset 8px 8px 16px #d1d9e6, inset -8px -8px 16px #ffffff',
        'clay-outset': '8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff',
        'clay-pressed': 'inset 4px 4px 8px #d1d9e6, inset -4px -4px 8px #ffffff',
      }
    },
  },
  plugins: [],
}