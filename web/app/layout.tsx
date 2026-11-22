import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Footnote",
  description: "Cited answers from SEC filings (FinanceBench), with the retrieved evidence shown.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
