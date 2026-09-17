export interface Page {
  id: number;
  blob: Blob;
  url: string;
  width: number;
  height: number;
}

type Listener = () => void;

/** In-memory list of scanned pages. */
export class PageStore {
  private items: Page[] = [];
  private listeners = new Set<Listener>();
  private nextId = 1;

  get all(): readonly Page[] {
    return this.items;
  }

  get(id: number): Page | undefined {
    return this.items.find((p) => p.id === id);
  }

  add(blob: Blob, width: number, height: number): Page {
    const page = { id: this.nextId++, blob, url: URL.createObjectURL(blob), width, height };
    this.items.push(page);
    this.emit();
    return page;
  }

  replace(id: number, blob: Blob, width: number, height: number): void {
    const page = this.get(id);
    if (!page) return;
    URL.revokeObjectURL(page.url);
    Object.assign(page, { blob, url: URL.createObjectURL(blob), width, height });
    this.emit();
  }

  remove(id: number): void {
    const page = this.get(id);
    if (!page) return;
    URL.revokeObjectURL(page.url);
    this.items = this.items.filter((p) => p !== page);
    this.emit();
  }

  subscribe(listener: Listener): void {
    this.listeners.add(listener);
  }

  private emit(): void {
    for (const l of this.listeners) l();
  }
}
