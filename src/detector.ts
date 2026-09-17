import type { Detection, Quad, WorkerRequest, WorkerResponse } from './protocol';

export interface ProcessedPage {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}

type Pending = { resolve: (value: never) => void; reject: (err: Error) => void };

/** Promise-based client for the OpenCV worker. */
export class Detector {
  private worker = new Worker(new URL('./scanner.worker.ts', import.meta.url), { type: 'module' });
  private nextId = 1;
  private pending = new Map<number, Pending>();
  readonly ready: Promise<void>;

  constructor() {
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data;
        if (msg.type === 'ready') return resolve();
        if (msg.type === 'error') {
          const err = new Error(msg.message);
          reject(err);
          for (const p of this.pending.values()) p.reject(err);
          this.pending.clear();
          return;
        }
        const entry = this.pending.get(msg.id);
        if (!entry) return;
        this.pending.delete(msg.id);
        if (msg.type === 'detect') entry.resolve(msg.detection as never);
        else
          entry.resolve({
            width: msg.width,
            height: msg.height,
            pixels: new Uint8ClampedArray(msg.pixels),
          } as never);
      };
      this.worker.onerror = (event) => reject(new Error(event.message || 'Scanner worker failed to load'));
    });
  }

  detect(frame: ImageData): Promise<Detection> {
    const id = this.nextId++;
    return this.send<Detection>(
      { type: 'detect', id, width: frame.width, height: frame.height, pixels: frame.data.buffer as ArrayBuffer },
      id,
    );
  }

  process(frame: ImageData, quad: Quad): Promise<ProcessedPage> {
    const id = this.nextId++;
    return this.send<ProcessedPage>(
      { type: 'process', id, width: frame.width, height: frame.height, pixels: frame.data.buffer as ArrayBuffer, quad },
      id,
    );
  }

  /** Forgets the learned background and tracking state, e.g. when the camera restarts. */
  reset(): void {
    this.worker.postMessage({ type: 'reset' } satisfies WorkerRequest);
  }

  private send<T>(request: Extract<WorkerRequest, { pixels: ArrayBuffer }>, id: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: never) => void, reject });
      this.worker.postMessage(request, [request.pixels]);
    });
  }
}
