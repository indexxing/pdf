/// <reference lib="webworker" />
import cvModule from '@techstark/opencv-js';
import type { Detection, DetectionIssue, Point, Quad, WorkerRequest, WorkerResponse } from './protocol';

type CV = typeof cvModule;
type Mat = InstanceType<CV['Mat']>;

// ---- Tuning ---------------------------------------------------------------

/** Pages smaller than this (fraction of the frame) aren't considered at all. */
const MIN_CANDIDATE_FRACTION = 0.04;
/** Pages smaller than this are detected but the user is asked to move closer. */
const TOO_FAR_FRACTION = 0.08;
/** Shapes covering more of the frame than this are the background, not a page. */
const MAX_CANDIDATE_FRACTION = 0.95;
/** Outline area / fitted quad area. Rejects merged blobs that only loosely resemble a quad. */
const MIN_SOLIDITY = 0.85;
/** Shortest straight edge (fraction of the frame's short side) used to build line-based quads. */
const MIN_SIDE_FRACTION = 0.12;
/** How many of the longest lines are combined into quads. */
const MAX_LINES = 14;
/** Fraction of a quad's visible perimeter that must lie on real edges. */
const MIN_EDGE_SUPPORT = 0.7;
/** Fraction of a quad's perimeter that must be inside the frame to be considered at all. */
const MIN_IN_FRAME = 0.6;
/** Interior edge density at which a candidate's score drops to ~37%. */
const INTERIOR_EDGE_SCALE = 0.012;
/** Mean HSV saturation (0..255) above which a shape is too colourful to be paper. */
const MAX_PAPER_SATURATION = 90;
/** Wait this long after the camera starts (exposure settling) before learning the empty scene. */
const BASELINE_SETTLE_MS = 700;
/** Mean corner movement per frame (px, detection scale) that marks a shape as hand-held rather than part of the room. */
const HAND_JITTER_PX = 1.2;
/** Background edges missing for this many frames are forgotten (a page that was lying there got moved). */
const BASELINE_FORGET_FRAMES = 150;
/** A candidate is background when this share of its currently visible outline was already in the empty scene. */
const BASELINE_MATCH = 0.55;
/** Corners closer than this (fraction of the short side) to the frame edge are out of bounds. */
const EDGE_MARGIN_FRACTION = 0.01;
/** Mean corner distance from the smoothed position (fraction of the diagonal) still considered steady. */
const STEADY_MOVEMENT = 0.015;
/** How long the page must stay steady and sharp before auto-capture. */
const HOLD_MS = 800;
/** Consecutive bad frames tolerated before the hold resets (hand-held pages flicker). */
const GRACE_FRAMES = 4;
/** Absolute floor for Laplacian variance on the detection frame. */
const MIN_SHARPNESS = 10;
/** A frame must be at least this fraction of the recent sharpest frame (autofocus settling). */
const RELATIVE_SHARPNESS = 0.45;
const MIN_BRIGHTNESS = 70;
/** Frames without a page needed before the same page can be captured again. */
const REARM_LOST_FRAMES = 6;
/** Signature difference (0..255 mean abs) that counts as a different page. */
const REARM_SIGNATURE_DIFF = 22;
/** Long side of the preview frames the page thread sends for detection. */
const DETECT_LONG_SIDE = 640;
/** Longest side of the processed page image. */
const MAX_OUTPUT_SIDE = 3200;
/** Common paper aspect ratios (long / short) to snap to when close. */
const PAPER_RATIOS = [11 / 8.5, Math.SQRT2, 14 / 8.5];
const PAPER_SNAP_TOLERANCE = 0.06;

// ---- State -----------------------------------------------------------------

let cv: CV;
/** Exponentially smoothed page position, used to judge steadiness. */
let smoothedQuad: Quad | null = null;
let steadySince = 0;
let missedFrames = 0;
let lastGood: Detection | null = null;
let sharpnessPeak = 0;
/** False right after a capture until the page is removed or replaced. */
let armed = true;
let lostFrames = 0;
let lastSignature: Uint8Array | null = null;
/** Edge map of the scene when the camera started. Shapes already present then are background. */
let baseline: Mat | null = null;
let baselineStartedAt = 0;
/** Per-pixel count of consecutive frames a background edge has been missing. */
let baselineMissing: Uint16Array | null = null;
/** Movement of the shape seen while calibrating, to tell a hand-held page from the room. */
let calibration = { last: null as Quad | null, movement: 0, frames: 0 };

function resetTracking() {
  baseline?.delete();
  baseline = null;
  baselineStartedAt = 0;
  baselineMissing = null;
  calibration = { last: null, movement: 0, frames: 0 };
  smoothedQuad = null;
  steadySince = 0;
  missedFrames = 0;
  lastGood = null;
  sharpnessPeak = 0;
}

const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(message, transfer);

async function loadOpenCv(): Promise<CV> {
  const mod = cvModule as unknown as CV | Promise<CV>;
  if (mod instanceof Promise) return await mod;
  if ((mod as CV).Mat) return mod as CV;
  await new Promise<void>((resolve) => {
    (mod as unknown as { onRuntimeInitialized: () => void }).onRuntimeInitialized = resolve;
  });
  return mod as CV;
}

loadOpenCv()
  .then((loaded) => {
    cv = loaded;
    post({ type: 'ready' });
  })
  .catch((err) => post({ type: 'error', message: String(err) }));

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'reset') {
      resetTracking();
    } else if (req.type === 'detect') {
      post({ type: 'detect', id: req.id, detection: detect(req.width, req.height, req.pixels) });
    } else {
      const result = processPage(req.width, req.height, req.pixels, req.quad);
      post(
        { type: 'process', id: req.id, width: result.width, height: result.height, pixels: result.pixels },
        [result.pixels],
      );
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

// ---- Helpers ---------------------------------------------------------------

/** Runs `fn` with a list that collects Mats and deletes them afterwards. */
function withMats<T>(fn: (track: <M extends { delete(): void }>(m: M) => M) => T): T {
  const mats: { delete(): void }[] = [];
  try {
    return fn((m) => {
      mats.push(m);
      return m;
    });
  } finally {
    for (const m of mats) m.delete();
  }
}

function rgbaMat(width: number, height: number, pixels: ArrayBuffer): Mat {
  const mat = new cv.Mat(height, width, cv.CV_8UC4);
  mat.data.set(new Uint8Array(pixels));
  return mat;
}

const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
const meanCornerDistance = (a: Quad, b: Quad) => a.reduce((sum, p, i) => sum + dist(p, b[i]), 0) / 4;
const cross = (a: Point, b: Point) => a.x * b.y - a.y * b.x;
const sub = (a: Point, b: Point): Point => ({ x: a.x - b.x, y: a.y - b.y });

/** Orders four points clockwise starting from the top-left. */
function orderCorners(points: Point[]): Quad {
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  const sorted = [...points].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  let start = 0;
  for (let i = 1; i < sorted.length; i++) if (sorted[i].x + sorted[i].y < sorted[start].x + sorted[start].y) start = i;
  return [0, 1, 2, 3].map((i) => sorted[(start + i) % 4]) as Quad;
}

function polygonArea(points: Point[]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) sum += cross(points[i], points[(i + 1) % points.length]);
  return Math.abs(sum) / 2;
}

function anglesAreReasonable(q: Quad): boolean {
  for (let i = 0; i < 4; i++) {
    const v1 = sub(q[(i + 3) % 4], q[i]);
    const v2 = sub(q[(i + 1) % 4], q[i]);
    const cos = (v1.x * v2.x + v1.y * v2.y) / (Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y));
    const deg = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
    if (deg < 45 || deg > 135) return false;
  }
  return true;
}

/**
 * Reduces a convex polygon to a quadrilateral by repeatedly dropping the edge whose removal
 * (extending its two neighbours until they meet) adds the least area. This straightens
 * slightly bent edges and restores corners that are rounded off or covered by fingers.
 */
function reduceToQuad(polygon: Point[]): Quad | null {
  const p = [...polygon];
  while (p.length > 4) {
    const n = p.length;
    let best = -1;
    let bestCost = Infinity;
    let bestPoint: Point | null = null;
    for (let i = 0; i < n; i++) {
      const a = p[(i + n - 1) % n];
      const b = p[i];
      const c = p[(i + 1) % n];
      const d = p[(i + 2) % n];
      // Ray from b continuing edge a→b, ray from c continuing edge d→c.
      const r = sub(b, a);
      const s = sub(c, d);
      const denom = cross(r, s);
      if (Math.abs(denom) < 1e-9) continue;
      const bc = sub(c, b);
      const t = cross(bc, s) / denom;
      const u = cross(bc, r) / denom;
      if (t < 0 || u < 0) continue;
      const x = { x: b.x + t * r.x, y: b.y + t * r.y };
      const cost = Math.abs(cross(sub(x, b), sub(c, b))) / 2;
      if (cost < bestCost) {
        best = i;
        bestCost = cost;
        bestPoint = x;
      }
    }
    if (best < 0 || !bestPoint) return null;
    if (best === n - 1) {
      p.splice(n - 1, 1);
      p[0] = bestPoint;
    } else {
      p.splice(best, 2, bestPoint);
    }
  }
  return p.length === 4 ? orderCorners(p) : null;
}

function matToPoints(mat: Mat): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < mat.rows; i++) pts.push({ x: mat.data32S[i * 2], y: mat.data32S[i * 2 + 1] });
  return pts;
}

interface Candidate {
  quad: Quad;
  area: number;
  score: number;
  inBounds: boolean;
}

interface Line {
  /** Unit normal and offset: points p on the line satisfy nx*p.x + ny*p.y = c. */
  nx: number;
  ny: number;
  c: number;
  /** Direction in [0, π). */
  angle: number;
  length: number;
  /** Midpoint of the longest segment on this line. */
  mid: Point;
}

const angleDiff = (a: number, b: number) => {
  const d = Math.abs(a - b) % Math.PI;
  return Math.min(d, Math.PI - d);
};

function isConvex(q: Quad): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const z = cross(sub(q[(i + 1) % 4], q[i]), sub(q[(i + 2) % 4], q[(i + 1) % 4]));
    if (Math.abs(z) < 1e-6) return false;
    if (sign === 0) sign = Math.sign(z);
    else if (Math.sign(z) !== sign) return false;
  }
  return true;
}

/** Hull-based quads: precise for pages whose outline is mostly intact (bent edges, rounded corners). */
function hullQuads(contours: InstanceType<CV['MatVector']>, minArea: number, maxArea: number): Quad[] {
  const quads: Quad[] = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    try {
      const area = cv.contourArea(contour);
      if (area < minArea || area > maxArea) continue;
      const quad = withMats((t) => {
        const hull = t(new cv.Mat());
        cv.convexHull(contour, hull, false, true);
        const approx = t(new cv.Mat());
        cv.approxPolyDP(hull, approx, 0.005 * cv.arcLength(hull, true), true);
        return reduceToQuad(matToPoints(approx));
      });
      // Reject merged blobs that only loosely resemble a quadrilateral.
      if (quad && area / polygonArea(quad) >= MIN_SOLIDITY) quads.push(quad);
    } finally {
      contour.delete();
    }
  }
  return quads;
}

/**
 * Line-based quads: groups straight outline segments into lines and intersects pairs of
 * roughly opposite lines. Works when the outline has gaps (fingers, a bright wall behind a
 * white page) or is merged with background shapes.
 */
function lineQuads(contours: InstanceType<CV['MatVector']>, minLength: number): Quad[] {
  const segments: Line[] = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    try {
      if (cv.arcLength(contour, true) < minLength * 2) continue;
      const pts = withMats((t) => {
        const approx = t(new cv.Mat());
        cv.approxPolyDP(contour, approx, 2.5, true);
        return matToPoints(approx);
      });
      for (let j = 0; j < pts.length; j++) {
        const a = pts[j];
        const b = pts[(j + 1) % pts.length];
        const length = dist(a, b);
        if (length < minLength) continue;
        const angle = (Math.atan2(b.y - a.y, b.x - a.x) + Math.PI) % Math.PI;
        const nx = -Math.sin(angle);
        const ny = Math.cos(angle);
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        segments.push({ nx, ny, c: nx * mid.x + ny * mid.y, angle, length, mid });
      }
    } finally {
      contour.delete();
    }
  }

  // Merge collinear segments (both sides of an edge, or an edge split by a finger).
  segments.sort((a, b) => b.length - a.length);
  const lines: Line[] = [];
  for (const s of segments) {
    const match = lines.find((l) => angleDiff(l.angle, s.angle) < (4 * Math.PI) / 180 && Math.abs(l.nx * s.mid.x + l.ny * s.mid.y - l.c) < 5);
    if (match) match.length += s.length;
    else lines.push({ ...s });
  }
  const top = lines.sort((a, b) => b.length - a.length).slice(0, MAX_LINES);

  const intersect = (a: Line, b: Line): Point | null => {
    const det = a.nx * b.ny - a.ny * b.nx;
    if (Math.abs(det) < 0.3) return null;
    return { x: (a.c * b.ny - b.c * a.ny) / det, y: (a.nx * b.c - b.nx * a.c) / det };
  };
  const PARALLEL = (30 * Math.PI) / 180;
  const quads: Quad[] = [];
  const n = top.length;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      for (let k = j + 1; k < n; k++)
        for (let l = k + 1; l < n; l++) {
          const [p, q, r, s] = [top[i], top[j], top[k], top[l]];
          for (const [a, b, c, d] of [
            [p, q, r, s],
            [p, r, q, s],
            [p, s, q, r],
          ]) {
            if (angleDiff(a.angle, b.angle) > PARALLEL || angleDiff(c.angle, d.angle) > PARALLEL) continue;
            if (angleDiff(a.angle, c.angle) < PARALLEL) continue;
            const corners = [intersect(a, c), intersect(c, b), intersect(b, d), intersect(d, a)];
            if (corners.some((x) => !x)) continue;
            quads.push(orderCorners(corners as Point[]));
          }
        }
  return quads;
}

/** Fraction of the quad's in-frame perimeter that lies on detected edges, and how much of it is in frame. */
function edgeSupport(edgeMap: Mat, quad: Quad, margin: number) {
  const w = edgeMap.cols;
  const h = edgeMap.rows;
  const data = edgeMap.data;
  let total = 0;
  let inside = 0;
  let hits = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const steps = Math.max(1, Math.ceil(dist(a, b) / 2));
    for (let s = 0; s < steps; s++) {
      const x = Math.round(a.x + ((b.x - a.x) * s) / steps);
      const y = Math.round(a.y + ((b.y - a.y) * s) / steps);
      total++;
      if (x < margin || y < margin || x >= w - margin || y >= h - margin) continue;
      inside++;
      if (data[y * w + x]) hits++;
    }
  }
  return { support: inside ? hits / inside : 0, inFrame: inside / total };
}

/** Share of the quad's inner area (inset from its border) covered by edges. Paper interiors are smooth once writing is closed away. */
function interiorEdgeDensity(canny: Mat, quad: Quad): number {
  const cx = quad.reduce((s, p) => s + p.x, 0) / 4;
  const cy = quad.reduce((s, p) => s + p.y, 0) / 4;
  const inner = quad.map((p) => ({ x: cx + (p.x - cx) * 0.9, y: cy + (p.y - cy) * 0.9 }));
  return withMats((track) => {
    const mask = track(cv.Mat.zeros(canny.rows, canny.cols, cv.CV_8UC1));
    const pts = track(cv.matFromArray(4, 1, cv.CV_32SC2, inner.flatMap((p) => [Math.round(p.x), Math.round(p.y)])));
    const polys = track(new cv.MatVector());
    polys.push_back(pts);
    cv.fillPoly(mask, polys, new cv.Scalar(255));
    const masked = track(new cv.Mat());
    cv.bitwise_and(canny, mask, masked);
    return cv.countNonZero(masked) / Math.max(1, cv.countNonZero(mask));
  });
}

function meanSaturation(hsv: Mat, quad: Quad): number {
  return withMats((track) => {
    const mask = track(cv.Mat.zeros(hsv.rows, hsv.cols, cv.CV_8UC1));
    const pts = track(cv.matFromArray(4, 1, cv.CV_32SC2, quad.flatMap((p) => [Math.round(p.x), Math.round(p.y)])));
    const polys = track(new cv.MatVector());
    polys.push_back(pts);
    cv.fillPoly(mask, polys, new cv.Scalar(255));
    return cv.mean(hsv, mask)[1];
  });
}

function signature(gray: Mat, quad: Quad): Uint8Array {
  return withMats((track) => {
    const small = track(new cv.Mat());
    warp(gray, quad, 36, 48, small, cv.INTER_AREA);
    cv.GaussianBlur(small, small, new cv.Size(5, 5), 0);
    cv.normalize(small, small, 0, 255, cv.NORM_MINMAX);
    return new Uint8Array(small.data);
  });
}

function signatureDiff(a: Uint8Array, b: Uint8Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total / a.length;
}

function warp(src: Mat, quad: Quad, width: number, height: number, dst: Mat, interpolation: number) {
  withMats((track) => {
    const from = track(cv.matFromArray(4, 1, cv.CV_32FC2, quad.flatMap((p) => [p.x, p.y])));
    const to = track(cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width, 0, width, height, 0, height]));
    const m = track(cv.getPerspectiveTransform(from, to));
    cv.warpPerspective(src, dst, m, new cv.Size(width, height), interpolation, cv.BORDER_REPLICATE);
  });
}

// ---- Detection -------------------------------------------------------------

interface Analysis {
  quad: Quad | null;
  issue: DetectionIssue;
  gray: Mat;
  supportMap: Mat;
  stats: Record<string, string | number>;
}

/** Looks at a single frame. `previous` (the smoothed page position) breaks ties between similar candidates. */
function analyze(src: Mat, previous: Quad | null, track: <M extends { delete(): void }>(m: M) => M): Analysis {
  const width = src.cols;
  const height = src.rows;
  const frameArea = width * height;
  const shortSide = Math.min(width, height);
  const margin = shortSide * EDGE_MARGIN_FRACTION;
  const gray = track(new cv.Mat());
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  // Close away writing so the page reads as one solid shape.
  const smooth = track(new cv.Mat());
  cv.GaussianBlur(gray, smooth, new cv.Size(5, 5), 0);
  cv.morphologyEx(smooth, smooth, cv.MORPH_CLOSE, track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(9, 9))));

  const canny = track(new cv.Mat());
  cv.Canny(smooth, canny, 30, 90);
  const edges = track(new cv.Mat());
  cv.dilate(canny, edges, track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))));
  const supportMap = track(new cv.Mat());
  cv.dilate(canny, supportMap, track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7))));
  // Brightness finds a page against dark surroundings even when its edges are soft.
  const bright = track(new cv.Mat());
  cv.threshold(smooth, bright, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

  const contoursOf = (binary: Mat) => {
    const contours = track(new cv.MatVector());
    cv.findContours(binary, contours, track(new cv.Mat()), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    return contours;
  };
  const edgeContours = contoursOf(edges);
  const minArea = frameArea * MIN_CANDIDATE_FRACTION;
  const maxArea = frameArea * MAX_CANDIDATE_FRACTION;
  const rawQuads = [
    ...hullQuads(edgeContours, minArea, maxArea),
    ...hullQuads(contoursOf(bright), minArea, maxArea),
    ...lineQuads(edgeContours, shortSide * MIN_SIDE_FRACTION),
  ];

  const halfDiagonal = Math.hypot(width, height) / 2;
  const background = baseline && baseline.cols === width && baseline.rows === height ? baseline : null;
  const candidates: Candidate[] = [];
  let ignored = 0;
  for (const quad of rawQuads) {
    const area = polygonArea(quad);
    if (area < minArea || area > maxArea || !isConvex(quad) || !anglesAreReasonable(quad)) continue;
    if (quad.some((p) => p.x < -width / 2 || p.y < -height / 2 || p.x > width * 1.5 || p.y > height * 1.5)) continue;
    const { support, inFrame } = edgeSupport(supportMap, quad, margin);
    if (inFrame < MIN_IN_FRAME || support < MIN_EDGE_SUPPORT) continue;
    // Windows, posters and monitors were already there when the camera started.
    if (background && edgeSupport(background, quad, margin).support >= support * BASELINE_MATCH) {
      ignored++;
      continue;
    }
    // Favour large, well-outlined, central shapes with a smooth interior. The interior term
    // rejects quads that stitch the page's edges to lines in the background.
    const cx = quad.reduce((s, p) => s + p.x, 0) / 4;
    const cy = quad.reduce((s, p) => s + p.y, 0) / 4;
    const offCenter = Math.hypot(cx - width / 2, cy - height / 2) / halfDiagonal;
    const smoothness = Math.exp(-interiorEdgeDensity(canny, quad) / INTERIOR_EDGE_SCALE);
    candidates.push({
      quad,
      area,
      score: area * support ** 2 * smoothness * (1 - 0.7 * offCenter),
      inBounds: quad.every((p) => p.x > margin && p.y > margin && p.x < width - margin && p.y < height - margin),
    });
  }
  candidates.sort((a, b) => b.score - a.score);
  const stats: Analysis['stats'] = { candidates: candidates.length, background: background ? ignored : 'learning' };

  // Paper is close to neutral; this rules out posters, skin and most furniture.
  const rgb = track(new cv.Mat());
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const hsv = track(new cv.Mat());
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const paperLike = (c: Candidate) => meanSaturation(hsv, c.quad) <= MAX_PAPER_SATURATION;

  let inside = candidates.filter((c) => c.inBounds).slice(0, 5).find(paperLike);
  if (inside && previous) {
    // Stick with the shape we were already tracking unless something clearly better appears.
    const tracked = candidates.find(
      (c) =>
        c.inBounds &&
        c.score >= inside!.score * 0.6 &&
        meanCornerDistance(c.quad, previous) < halfDiagonal * 0.06,
    );
    if (tracked && tracked !== inside && paperLike(tracked)) inside = tracked;
  }
  const clipped = candidates.filter((c) => !c.inBounds).slice(0, 5).find(paperLike);

  // A much larger shape running off the frame is more likely the page than a small one inside it.
  if (clipped && (!inside || clipped.area > inside.area * 2)) {
    return { quad: clipped.quad, issue: 'out-of-bounds', gray, supportMap, stats };
  }
  if (!inside) return { quad: null, issue: 'no-page', gray, supportMap, stats };

  const quad = inside.quad;
  stats.area = +(inside.area / frameArea).toFixed(3);
  if (inside.area < frameArea * TOO_FAR_FRACTION) return { quad, issue: 'too-far', gray, supportMap, stats };

  const roi = track(gray.roi(boundingRect(quad, width, height)));
  const brightness = cv.mean(roi)[0];
  stats.brightness = Math.round(brightness);
  if (brightness < MIN_BRIGHTNESS) return { quad, issue: 'too-dark', gray, supportMap, stats };

  const lap = track(new cv.Mat());
  cv.Laplacian(roi, lap, cv.CV_16S, 3);
  const mean = track(new cv.Mat());
  const std = track(new cv.Mat());
  cv.meanStdDev(lap, mean, std);
  const sharpness = std.data64F[0] ** 2;
  sharpnessPeak = Math.max(sharpness, sharpnessPeak * 0.95);
  stats.sharpness = Math.round(sharpness);
  stats.sharpnessPeak = Math.round(sharpnessPeak);
  if (sharpness < MIN_SHARPNESS || sharpness < sharpnessPeak * RELATIVE_SHARPNESS) {
    return { quad, issue: 'blurry', gray, supportMap, stats };
  }

  return { quad, issue: 'none', gray, supportMap, stats };
}

function boundingRect(quad: Quad, width: number, height: number) {
  const x = Math.max(0, Math.floor(Math.min(...quad.map((p) => p.x))));
  const y = Math.max(0, Math.floor(Math.min(...quad.map((p) => p.y))));
  const right = Math.min(width, Math.ceil(Math.max(...quad.map((p) => p.x))));
  const bottom = Math.min(height, Math.ceil(Math.max(...quad.map((p) => p.y))));
  return new cv.Rect(x, y, Math.max(1, right - x), Math.max(1, bottom - y));
}

function detect(width: number, height: number, pixels: ArrayBuffer): Detection {
  return withMats((track) => {
    const src = track(rgbaMat(width, height, pixels));
    const now = performance.now();
    if (baseline && (baseline.cols !== width || baseline.rows !== height)) resetTracking();
    if (!baselineStartedAt) baselineStartedAt = now;
    const { quad, issue, gray, supportMap, stats } = analyze(src, smoothedQuad, track);

    if (!baseline) {
      const shape = issue === 'out-of-bounds' ? null : quad;
      if (shape && calibration.last) {
        calibration.movement += meanCornerDistance(shape, calibration.last);
        calibration.frames++;
      }
      calibration.last = shape;
      if (now - baselineStartedAt < BASELINE_SETTLE_MS) {
        return { quad: null, issue: 'calibrating', steadiness: 0, debug: stats };
      }
      // Learn the scene now, unless someone is already holding a page up; then wait for it to leave.
      const handHeld = shape && calibration.frames > 0 && calibration.movement / calibration.frames > HAND_JITTER_PX;
      if (!handHeld) {
        baseline = supportMap.clone();
        baselineMissing = new Uint16Array(baseline.rows * baseline.cols);
      }
    } else {
      forgetVanishedEdges(supportMap);
    }

    if (issue !== 'none') {
      if (!quad || issue === 'out-of-bounds') {
        if (++lostFrames >= REARM_LOST_FRAMES) armed = true;
      } else {
        lostFrames = 0;
      }
      // Ride out brief glitches (a finger slipping, a blurry frame) without losing progress.
      if (lastGood && ++missedFrames <= GRACE_FRAMES) {
        const steadiness = lastGood.issue === 'none' && steadySince ? Math.min(0.99, (now - steadySince) / HOLD_MS) : 0;
        return { ...lastGood, steadiness, debug: { ...stats, raw: issue, grace: missedFrames } };
      }
      smoothedQuad = null;
      steadySince = 0;
      lastGood = null;
      return { quad, issue, steadiness: 0, debug: stats };
    }

    const page = quad!;
    missedFrames = 0;
    lostFrames = 0;
    const diagonal = Math.hypot(width, height);
    const movement = smoothedQuad
      ? meanCornerDistance(smoothedQuad, page) / diagonal
      : Infinity;
    smoothedQuad = smoothedQuad
      ? (smoothedQuad.map((p, i) => ({ x: p.x * 0.5 + page[i].x * 0.5, y: p.y * 0.5 + page[i].y * 0.5 })) as Quad)
      : page;
    stats.movement = Number.isFinite(movement) ? +movement.toFixed(4) : 'n/a';

    if (movement > STEADY_MOVEMENT) {
      steadySince = now;
      lastGood = { quad, issue: 'moving', steadiness: 0 };
      return { ...lastGood, debug: stats };
    }
    if (!steadySince) steadySince = now;

    if (!armed) {
      const sig = signature(gray, page);
      if (lastSignature && signatureDiff(sig, lastSignature) > REARM_SIGNATURE_DIFF) armed = true;
      else {
        lastGood = { quad, issue: 'duplicate', steadiness: 0 };
        return { ...lastGood, debug: stats };
      }
    }

    lastGood = { quad, issue: 'none', steadiness: Math.min(1, (now - steadySince) / HOLD_MS) };
    return { ...lastGood, debug: stats };
  });
}

function forgetVanishedEdges(current: Mat) {
  if (!baseline || !baselineMissing) return;
  const learned = baseline.data;
  const seen = current.data;
  for (let i = 0; i < learned.length; i++) {
    if (!learned[i]) continue;
    if (seen[i]) baselineMissing[i] = 0;
    else if (++baselineMissing[i] > BASELINE_FORGET_FRAMES) learned[i] = 0;
  }
}

// ---- Page processing -------------------------------------------------------

function outputSize(quad: Quad): { width: number; height: number } {
  let width = Math.max(dist(quad[0], quad[1]), dist(quad[3], quad[2]));
  let height = Math.max(dist(quad[0], quad[3]), dist(quad[1], quad[2]));

  const long = Math.max(width, height);
  const short = Math.min(width, height);
  const ratio = long / short;
  const snapped = PAPER_RATIOS.find((r) => Math.abs(ratio - r) / r < PAPER_SNAP_TOLERANCE);
  if (snapped) {
    if (height >= width) height = width * snapped;
    else width = height * snapped;
  }

  const scale = Math.min(1.5, MAX_OUTPUT_SIDE / Math.max(width, height));
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

/**
 * Re-finds the page in the exact frame being captured. A hand-held page can shift slightly
 * between the preview frame that triggered capture and this full-resolution frame.
 */
function refineQuad(src: Mat, quad: Quad): Quad {
  return withMats((track) => {
    const scale = Math.min(1, DETECT_LONG_SIDE / Math.max(src.cols, src.rows));
    const small = track(new cv.Mat());
    cv.resize(src, small, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);
    const hint = quad.map((p) => ({ x: p.x * scale, y: p.y * scale })) as Quad;
    const peak = sharpnessPeak;
    const fresh = analyze(small, hint, track).quad;
    sharpnessPeak = peak;
    if (!fresh) return quad;
    if (meanCornerDistance(fresh, hint) > Math.hypot(small.cols, small.rows) * 0.04) return quad;
    return fresh.map((p) => ({ x: p.x / scale, y: p.y / scale })) as Quad;
  });
}

function processPage(width: number, height: number, pixels: ArrayBuffer, givenQuad: Quad) {
  return withMats((track) => {
    const src = track(rgbaMat(width, height, pixels));
    const quad = refineQuad(src, givenQuad);
    const size = outputSize(quad);
    const warped = track(new cv.Mat());
    warp(src, quad, size.width, size.height, warped, cv.INTER_CUBIC);

    const gray = track(new cv.Mat());
    cv.cvtColor(warped, gray, cv.COLOR_RGBA2GRAY);

    // Trim a sliver so the page edge and background don't show.
    const inset = Math.round(Math.min(size.width, size.height) * 0.006);
    const cropped = track(gray.roi(new cv.Rect(inset, inset, size.width - inset * 2, size.height - inset * 2)));

    const enhanced = track(toPortrait(track(enhance(cropped))));
    const rgba = track(new cv.Mat());
    cv.cvtColor(enhanced, rgba, cv.COLOR_GRAY2RGBA);

    // Remember this page so auto-capture doesn't grab it twice.
    const detectGray = track(new cv.Mat());
    cv.cvtColor(src, detectGray, cv.COLOR_RGBA2GRAY);
    lastSignature = signature(detectGray, quad);
    armed = false;
    lostFrames = 0;
    steadySince = 0;

    const out = new Uint8Array(rgba.data).buffer;
    return { width: rgba.cols, height: rgba.rows, pixels: out };
  });
}

/** Returns a portrait copy of the page, turning landscape scans the way their text reads upright. */
function toPortrait(page: Mat): Mat {
  const out = new cv.Mat();
  if (page.rows >= page.cols) page.copyTo(out);
  else cv.rotate(page, out, withMats((track) => uprightRotation(page, track)));
  return out;
}

/**
 * Picks the quarter turn that makes a landscape scan's text upright. Text lines show up as
 * strong alternation in the ink profile across them. Lines share a left margin while their
 * ends are ragged, which tells us which side is the start of each line.
 */
function uprightRotation(page: Mat, track: <M extends { delete(): void }>(m: M) => M): number {
  const clockwise = cv.ROTATE_90_CLOCKWISE;
  const counterClockwise = cv.ROTATE_90_COUNTERCLOCKWISE;

  const small = track(new cv.Mat());
  cv.resize(page, small, new cv.Size(400, Math.round((page.rows * 400) / page.cols)), 0, 0, cv.INTER_AREA);
  const w = small.cols;
  const h = small.rows;
  const ink = track(new cv.Mat());
  cv.threshold(small, ink, 150, 255, cv.THRESH_BINARY_INV);
  // Drop ruled lines and table borders so only writing is left.
  for (const size of [new cv.Size(1, Math.round(h * 0.4)), new cv.Size(Math.round(w * 0.4), 1)]) {
    const lines = track(new cv.Mat());
    cv.morphologyEx(ink, lines, cv.MORPH_OPEN, track(cv.getStructuringElement(cv.MORPH_RECT, size)));
    cv.subtract(ink, lines, ink);
  }

  // Ignore the outer margin, where leftover page edges can sit.
  const mx = Math.round(w * 0.04);
  const my = Math.round(h * 0.04);
  const rows = new Float64Array(h);
  const cols = new Float64Array(w);
  const data = ink.data;
  let total = 0;
  for (let y = my; y < h - my; y++) {
    for (let x = mx; x < w - mx; x++) {
      if (!data[y * w + x]) continue;
      rows[y]++;
      cols[x]++;
      total++;
    }
  }
  if (total < w * h * 0.002) return clockwise;

  const alternation = (profile: Float64Array, from: number, to: number) => {
    let sum = 0;
    let diff = 0;
    for (let i = from; i < to; i++) {
      sum += profile[i];
      if (i > from) diff += (profile[i] - profile[i - 1]) ** 2;
    }
    const n = to - from;
    const mean = sum / n;
    return mean ? diff / n / (mean * mean) : 0;
  };
  // Text already runs along the long side: landscape content. Put its top on the left.
  if (alternation(rows, my, h - my) >= alternation(cols, mx, w - mx)) return counterClockwise;

  // Text runs vertically: a portrait page held sideways. Each band of inked columns is a line.
  const starts: number[] = [];
  const ends: number[] = [];
  for (let x = mx; x < w - mx; ) {
    if (!cols[x]) {
      x++;
      continue;
    }
    const bandStart = x;
    while (x < w - mx && cols[x]) x++;
    if (x - bandStart < 3) continue;
    let minY = h;
    let maxY = -1;
    for (let bx = bandStart; bx < x; bx++) {
      for (let y = my; y < h - my; y++) {
        if (!data[y * w + bx]) continue;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    starts.push(minY);
    ends.push(maxY);
  }
  if (starts.length < 3) return clockwise;

  const spread = (values: number[]) => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  };
  // Turning clockwise moves the bottom of the image to the left, so the bottom ends become line starts.
  return spread(ends) <= spread(starts) ? clockwise : counterClockwise;
}

/**
 * Flattens lighting and boosts strokes. Dividing by an estimated paper background removes
 * shadows and gradients; a tone curve then pushes faint marks (pencil) toward black while
 * leaving paper white.
 */
function enhance(gray: Mat): Mat {
  return withMats((track) => {
    // Background estimate at quarter scale: max-filter erases strokes, median smooths blotches.
    const small = track(new cv.Mat());
    cv.resize(gray, small, new cv.Size(Math.round(gray.cols / 4), Math.round(gray.rows / 4)), 0, 0, cv.INTER_AREA);
    const k = Math.max(5, Math.round(Math.min(small.cols, small.rows) / 45) | 1);
    cv.dilate(small, small, track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k))));
    cv.medianBlur(small, small, 7);
    const background = track(new cv.Mat());
    cv.resize(small, background, new cv.Size(gray.cols, gray.rows), 0, 0, cv.INTER_LINEAR);

    const normalized = track(new cv.Mat());
    cv.divide(gray, background, normalized, 255);
    cv.GaussianBlur(normalized, normalized, new cv.Size(3, 3), 0);

    // Paper noise floor: the median darkness (most of a page is blank paper).
    const histogram = new Uint32Array(256);
    const data = normalized.data;
    for (let i = 0; i < data.length; i += 3) histogram[255 - data[i]]++;
    const half = Math.ceil(data.length / 3) / 2;
    let floor = 0;
    for (let seen = 0; floor < 255 && (seen += histogram[floor]) < half; floor++);
    floor += 4;

    const TAU = 30; // lower = more aggressive darkening
    const KNEE = 6; // softens the transition out of the paper noise floor
    const lut = new Uint8Array(256);
    for (let v = 0; v < 256; v++) {
      const d = Math.max(0, 255 - v - floor);
      const soft = (d * d) / (d + KNEE) || 0;
      lut[v] = Math.round(255 * Math.exp(-soft / TAU));
    }
    const lutMat = track(cv.matFromArray(1, 256, cv.CV_8UC1, lut));
    const out = new cv.Mat();
    cv.LUT(normalized, lutMat, out);
    return out;
  });
}
