import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths so the build works from any static host or subdirectory.
  base: './',
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    // OpenCV.js (with inlined wasm) is a single ~10MB chunk loaded only by the worker.
    chunkSizeWarningLimit: 15000,
  },
});
