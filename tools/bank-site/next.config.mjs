import { fileURLToPath } from "node:url";
import path from "node:path";

/** Static export: the whole site is HTML + JS + the data files in public/data. */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  turbopack: { root: path.dirname(fileURLToPath(import.meta.url)) },
};
export default nextConfig;
