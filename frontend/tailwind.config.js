/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pitch: {
          950: '#070D09',
          900: '#0B130E',
          800: '#14231A',
          700: '#1F382A',
          accent: '#10B981',
          neon: '#34D399'
        }
      }
    },
  },
  plugins: [],
}
