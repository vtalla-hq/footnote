import type { NextConfig } from "next";

// In production Vercel serves /api/* from the Python function (api/index.py, see vercel.json).
// In local dev, proxy /api/* to uvicorn on :8000.
const nextConfig: NextConfig = {
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};

export default nextConfig;
