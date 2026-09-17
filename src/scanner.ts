import { describeCameraError, openCamera } from './camera';
import { Detector } from './detector';
import { exportPdf } from './pdf';
import { PageStore } from './pages';
import type { Detection, DetectionIssue, Quad } from './protocol';

/** Long side of frames sent for live detection. */
const DETECT_LONG_SIDE = 640;
const JPEG_QUALITY = 0.9;

const LABELS: Record<DetectionIssue, string> = {
  none: 'Hold steady',
  'no-page': 'No page in view',
  'out-of-bounds': 'Fit the whole page in view',
  'too-far': 'Move closer',
  'too-dark': 'Needs more light',
  blurry: 'Focusing',
  moving: 'Hold steady',
  duplicate: 'Captured. Next page',
  calibrating: 'Getting ready',
};

type StatusState = 'idle' | 'good' | 'bad';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Scanner {
  private readonly store = new PageStore();
  private detector: Detector | null = null;

  private video = $<HTMLVideoElement>('video');
  private overlay = $<HTMLCanvasElement>('overlay');
  private stage = $('stage');
  private statusEl = $('status');
  private statusFill = $('status-fill');
  private statusLabel = $('status-label');
  private list = $<HTMLOListElement>('page-list');
  private autoToggle = $<HTMLInputElement>('auto');

  private debugEl: HTMLPreElement | null = null;
  private frameCanvas = document.createElement('canvas');
  private frameCtx = this.frameCanvas.getContext('2d', { willReadFrequently: true })!;

  private running = false;
  private detecting = false;
  private processing = 0;
  private selectedId: number | null = null;
  private lastDetection: { detection: Detection; width: number; height: number; at: number } | null = null;

  constructor() {
    this.store.subscribe(() => this.renderPages());
    $('shutter').addEventListener('click', () => this.capture(true));
    $('export').addEventListener('click', () => this.export());
    $('back').addEventListener('click', () => this.select(null));
    $('delete').addEventListener('click', () => this.deleteSelected());
    $('rotate').addEventListener('click', () => this.rotateSelected());
    $('retry').addEventListener('click', () => this.start());
    this.autoToggle.addEventListener('change', () => this.statusFill.style.setProperty('--progress', '0'));
    new ResizeObserver(() => this.drawOverlay()).observe(this.stage);

    if (new URLSearchParams(location.search).has('debug')) {
      this.debugEl = document.createElement('pre');
      this.debugEl.className = 'debug';
      this.stage.append(this.debugEl);
    }

    document.addEventListener('keydown', (e) => {
      if (!this.running || e.target instanceof HTMLInputElement) return;
      if (e.key === 'Escape' && this.selectedId !== null) this.select(null);
      if (e.code === 'Space' && this.selectedId === null) {
        e.preventDefault();
        this.capture(true);
      }
    });
  }

  async start(): Promise<void> {
    $('stage-message').hidden = true;
    this.setStatus('idle', 'Starting camera');
    this.detector ??= new Detector();

    try {
      await openCamera(this.video);
    } catch (err) {
      return this.showError(describeCameraError(err));
    }
    $('controls').hidden = false;

    this.setStatus('idle', 'Loading scanner');
    try {
      await this.detector.ready;
    } catch (err) {
      this.detector = null;
      return this.showError(`The scanner failed to load. ${err instanceof Error ? err.message : ''}`);
    }

    // Learn the empty scene afresh each time the camera starts.
    this.detector.reset();
    this.running = true;
    this.setStatus('idle', LABELS.calibrating);
    requestAnimationFrame(this.tick);
  }

  // ---- Live detection ------------------------------------------------------

  private tick = () => {
    if (!this.running) return;
    requestAnimationFrame(this.tick);
    if (this.detecting || this.selectedId !== null || this.video.readyState < 2) return;
    this.detecting = true;
    this.detectOnce().finally(() => (this.detecting = false));
  };

  private async detectOnce() {
    const frame = this.grabFrame(DETECT_LONG_SIDE);
    const { width, height } = frame;
    let detection: Detection;
    try {
      detection = await this.detector!.detect(frame);
    } catch (err) {
      console.error(err);
      return;
    }
    if (this.selectedId !== null) return;

    const now = performance.now();
    if (this.debugEl) {
      const fps = this.lastDetection ? Math.round(1000 / (now - this.lastDetection.at)) : 0;
      const lines = { issue: detection.issue, steadiness: detection.steadiness.toFixed(2), fps, frame: `${width}x${height}`, ...detection.debug };
      this.debugEl.textContent = Object.entries(lines).map(([k, v]) => `${k}: ${v}`).join('\n');
    }
    this.lastDetection = { detection, width, height, at: now };
    this.drawOverlay();

    const auto = this.autoToggle.checked;
    const clear = detection.issue === 'none' || detection.issue === 'duplicate';
    const state = detection.issue === 'calibrating' ? 'idle' : clear ? 'good' : 'bad';
    const label =
      detection.issue === 'none' && !auto ? 'Ready. Tap capture' : LABELS[detection.issue];
    this.setStatus(state, label, auto && detection.issue === 'none' ? detection.steadiness : 0);

    if (auto && detection.issue === 'none' && detection.steadiness >= 1 && this.processing === 0) {
      this.capture(false);
    }
  }

  private grabFrame(longSide?: number): ImageData {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const scale = longSide ? Math.min(1, longSide / Math.max(vw, vh)) : 1;
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    if (this.frameCanvas.width !== w || this.frameCanvas.height !== h) {
      this.frameCanvas.width = w;
      this.frameCanvas.height = h;
    }
    this.frameCtx.drawImage(this.video, 0, 0, w, h);
    return this.frameCtx.getImageData(0, 0, w, h);
  }

  // ---- Capture -------------------------------------------------------------

  private async capture(manual: boolean) {
    if (!this.running || !this.detector || this.selectedId !== null) return;
    if (!manual && this.processing > 0) return;

    const frame = this.grabFrame();
    const recent = this.lastDetection && performance.now() - this.lastDetection.at < 600 ? this.lastDetection : null;
    let quad: Quad;
    if (recent?.detection.quad && (manual ? recent.detection.issue !== 'out-of-bounds' : true)) {
      const s = frame.width / recent.width;
      quad = recent.detection.quad.map((p) => ({ x: p.x * s, y: p.y * s })) as Quad;
    } else {
      // No usable page outline: keep the whole frame.
      const { width: w, height: h } = frame;
      quad = [
        { x: 0, y: 0 },
        { x: w, y: 0 },
        { x: w, y: h },
        { x: 0, y: h },
      ];
    }

    this.flash();
    this.processing++;
    this.setStatus('good', 'Captured', 1);
    this.renderPages();

    try {
      const result = await this.detector.process(frame, quad);
      const canvas = document.createElement('canvas');
      canvas.width = result.width;
      canvas.height = result.height;
      const imageData = new ImageData(result.pixels as Uint8ClampedArray<ArrayBuffer>, result.width, result.height);
      canvas.getContext('2d')!.putImageData(imageData, 0, 0);
      const blob = await encodeJpeg(canvas);
      this.store.add(blob, result.width, result.height);
      this.list.lastElementChild?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } catch (err) {
      console.error(err);
      this.setStatus('bad', 'Capture failed');
    } finally {
      this.processing--;
      this.renderPages();
    }
  }

  private flash() {
    const el = $('flash');
    el.classList.remove('on');
    void el.offsetWidth; // restart the animation
    el.classList.add('on');
  }

  // ---- Pages sidebar & preview --------------------------------------------

  private renderPages() {
    const pages = this.store.all;
    $('page-count').textContent = String(pages.length);
    $('pages-empty').hidden = pages.length > 0 || this.processing > 0;
    $<HTMLButtonElement>('export').disabled = pages.length === 0;

    const items = pages.map((page, index) => {
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.className = 'page-item';
      button.setAttribute('aria-label', `Page ${index + 1}`);
      if (page.id === this.selectedId) button.setAttribute('aria-current', 'true');
      button.addEventListener('click', () => this.select(page.id === this.selectedId ? null : page.id));

      const img = document.createElement('img');
      img.src = page.url;
      img.alt = '';
      img.style.aspectRatio = `${page.width} / ${page.height}`;

      const num = document.createElement('span');
      num.className = 'page-num';
      num.textContent = String(index + 1);

      button.append(img, num);
      li.append(button);
      return li;
    });

    for (let i = 0; i < this.processing; i++) {
      const li = document.createElement('li');
      li.innerHTML = `<div class="page-item is-processing"><div class="page-placeholder"></div><span class="page-num">Processing</span></div>`;
      items.push(li);
    }
    this.list.replaceChildren(...items);
  }

  private select(id: number | null) {
    this.selectedId = id;
    const page = id === null ? undefined : this.store.get(id);
    $('preview').hidden = !page;
    $('controls').hidden = !!page;
    if (this.debugEl) this.debugEl.hidden = !!page;

    if (page) {
      const index = this.store.all.indexOf(page);
      $('preview-title').textContent = `Page ${index + 1} of ${this.store.all.length}`;
      $<HTMLImageElement>('preview-img').src = page.url;
      this.setStatus('idle', 'Paused');
    } else {
      this.selectedId = null;
      this.lastDetection = null;
      this.drawOverlay();
      if (this.running) this.setStatus('bad', LABELS['no-page']);
    }
    this.renderPages();
  }

  private deleteSelected() {
    if (this.selectedId === null) return;
    const pages = this.store.all;
    const index = pages.findIndex((p) => p.id === this.selectedId);
    const neighbor = pages[index + 1] ?? pages[index - 1];
    this.store.remove(this.selectedId);
    this.select(neighbor?.id ?? null);
  }

  private async rotateSelected() {
    const page = this.selectedId === null ? undefined : this.store.get(this.selectedId);
    if (!page) return;
    const bitmap = await createImageBitmap(page.blob);
    const canvas = document.createElement('canvas');
    canvas.width = page.height;
    canvas.height = page.width;
    const ctx = canvas.getContext('2d')!;
    ctx.translate(canvas.width, 0);
    ctx.rotate(Math.PI / 2);
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    this.store.replace(page.id, await encodeJpeg(canvas), canvas.width, canvas.height);
    if (this.selectedId === page.id) this.select(page.id);
  }

  private async export() {
    const button = $<HTMLButtonElement>('export');
    button.disabled = true;
    button.textContent = 'Exporting…';
    try {
      await exportPdf([...this.store.all]);
    } catch (err) {
      console.error(err);
      alert('Could not create the PDF.');
    } finally {
      button.textContent = 'Export PDF';
      button.disabled = this.store.all.length === 0;
    }
  }

  // ---- Status & overlay ----------------------------------------------------

  private setStatus(state: StatusState, label: string, progress = 0) {
    this.statusEl.dataset.state = state;
    this.statusLabel.textContent = label;
    this.statusFill.style.setProperty('--progress', String(progress));
  }

  private showError(message: string) {
    this.running = false;
    $('controls').hidden = true;
    $('stage-message-text').textContent = message;
    $('stage-message').hidden = false;
    this.setStatus('bad', 'Unavailable');
  }

  private drawOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const { clientWidth: cw, clientHeight: ch } = this.stage;
    if (this.overlay.width !== Math.round(cw * dpr) || this.overlay.height !== Math.round(ch * dpr)) {
      this.overlay.width = Math.round(cw * dpr);
      this.overlay.height = Math.round(ch * dpr);
    }
    const ctx = this.overlay.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const last = this.lastDetection;
    const quad = last?.detection.quad;
    if (!last || !quad || this.selectedId !== null) return;

    // The video uses object-fit: contain; map frame coordinates into the displayed rectangle.
    const scale = Math.min(cw / last.width, ch / last.height);
    const ox = (cw - last.width * scale) / 2;
    const oy = (ch - last.height * scale) / 2;

    const good = last.detection.issue === 'none' || last.detection.issue === 'duplicate';
    const color = good ? '#2fbf71' : '#e5534b';
    ctx.beginPath();
    quad.forEach((p, i) => {
      const x = ox + p.x * scale;
      const y = oy + p.y * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = good ? 'rgba(47, 191, 113, 0.14)' : 'rgba(229, 83, 75, 0.12)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = color;
    ctx.stroke();
  }
}

function encodeJpeg(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('JPEG encoding failed'))), 'image/jpeg', JPEG_QUALITY),
  );
}
