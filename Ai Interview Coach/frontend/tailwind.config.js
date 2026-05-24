/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        apple: {
          bg: '#f5f5f7',
          text: '#1d1d1f',
          gray: '#86868b',
          'gray-light': '#f5f5f7',
          'gray-dark': '#1d1d1f',
        },
      },
      borderRadius: {
        apple: '32px',
      },
      fontFamily: {
        sf: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Display"', '"SF Pro Text"', 'Inter', '"Helvetica Neue"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
