import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const sans = Geist({ subsets: ["latin"], variable: "--sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--mono" });

export const metadata = {
  title: "SpriteSheet · generation bank",
  description:
    "56,000 pixel-art sprites per model from the three text-to-image models that run as native Google Sheets formulas, each prompt reviewed by eye.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
