export interface Point {
  x: number;
  y: number;
}

/** Corners ordered top-left, top-right, bottom-right, bottom-left. */
export type Quad = [Point, Point, Point, Point];

export type DetectionIssue =
  | 'none' // page found, in bounds, sharp and steady
  | 'no-page'
  | 'out-of-bounds'
  | 'too-far'
  | 'too-dark'
  | 'blurry'
  | 'moving'
  | 'duplicate'
  | 'calibrating'; // learning what the scene looks like without a page

export interface Detection {
  /** Quad in the coordinate space of the frame that was sent. */
  quad: Quad | null;
  issue: DetectionIssue;
  /** 0..1, how long the page has been held steady and clear. */
  steadiness: number;
  /** Diagnostic values, shown when the page is opened with ?debug. */
  debug?: Record<string, string | number>;
}

export type WorkerRequest =
  | { type: 'reset' }
  | { type: 'detect'; id: number; width: number; height: number; pixels: ArrayBuffer }
  | {
      type: 'process';
      id: number;
      width: number;
      height: number;
      pixels: ArrayBuffer;
      quad: Quad;
    };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | { type: 'detect'; id: number; detection: Detection }
  | { type: 'process'; id: number; width: number; height: number; pixels: ArrayBuffer };
