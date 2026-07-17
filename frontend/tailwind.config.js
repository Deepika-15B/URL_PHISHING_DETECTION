/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: { ink: "#0b1220", cyan: "#22d3ee", safe: "#22c55e", danger: "#f43f5e" },
      boxShadow: { glow: "0 12px 40px rgba(34,211,238,.14)" },
    },
  },
  plugins: [],
};
