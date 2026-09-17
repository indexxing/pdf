# Scan to PDF

A static site that scans pages with the device camera and exports them as a PDF. Everything runs in the browser; nothing is uploaded.

```sh
npm install
npm run dev      # http://localhost:5173
npm run build    # static output in dist/
```

Serve `dist/` from any static host. Camera access needs HTTPS (localhost is fine for development).

## How it works

- `src/scanner.worker.ts` runs OpenCV.js in a web worker. Each downscaled preview frame is searched for page-shaped quadrilaterals two ways: fitting quads to outlines (handles bent edges and corners hidden by fingers) and intersecting long straight edge lines (handles gaps, e.g. a white page in front of a white wall). Candidates are scored by how much of their outline lies on real edges, how smooth their interior is, size, distance from centre and colour (paper is near-neutral). The chosen page must be fully in frame, big enough, bright enough, sharp and steady; after ~0.8s the page is captured automatically. Short glitches (a few bad frames) don't reset the hold, so hand-held pages work.
- When the camera starts, the worker spends ~0.7s learning the empty scene's edges ("Getting ready"). Page-shaped outlines that were already there (windows, posters, whiteboards, monitors) are ignored afterwards. If someone is already holding a page up (it jitters), learning waits until it leaves; edges that disappear for a few seconds are forgotten, so a page that was lying there at the start is picked up once it's moved.
- Scans always come out portrait. Landscape scans are turned so their text reads upright (text lines share a left margin; ends are ragged). Pages whose content is genuinely landscape are turned with their top to the left. The Rotate button in the page preview turns a page 90° if the guess is wrong.
- Pages can be held up to a laptop webcam or laid on a desk under a phone camera. Front-facing/laptop cameras show a mirrored preview; scans are never mirrored.
- Open the site with `?debug` to see live detection values (issue, page area, sharpness, movement, fps).
- Captures use the full-resolution camera frame: perspective-corrected, snapped to Letter/A4/Legal proportions when close, then enhanced. Enhancement divides by an estimated paper background (removes shadows), then applies a tone curve that pushes faint marks such as pencil toward black while keeping paper white. Tuning constants are at the top of the worker.
- After a capture, the same page won't be captured again until it leaves the frame or a clearly different page replaces it. The shutter button (or Space) captures manually at any time.
- `src/pdf.ts` builds the PDF with jsPDF, one page per scan.
