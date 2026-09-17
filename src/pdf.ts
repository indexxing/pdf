import type { Page } from './pages';

/** Letter width in points; page height follows the scan's aspect ratio. */
const PAGE_WIDTH_PT = 612;

export async function exportPdf(pages: Page[]): Promise<void> {
  const { jsPDF } = await import('jspdf');
  let doc: InstanceType<typeof jsPDF> | null = null;

  for (const page of pages) {
    const landscape = page.width > page.height;
    const shortSide = PAGE_WIDTH_PT;
    const longSide = (PAGE_WIDTH_PT * Math.max(page.width, page.height)) / Math.min(page.width, page.height);
    const format: [number, number] = landscape ? [longSide, shortSide] : [shortSide, longSide];
    const orientation = landscape ? 'landscape' : 'portrait';

    if (!doc) doc = new jsPDF({ unit: 'pt', format, orientation, compress: true });
    else doc.addPage(format, orientation);

    const bytes = new Uint8Array(await page.blob.arrayBuffer());
    doc.addImage(bytes, 'JPEG', 0, 0, format[0], format[1], undefined, 'NONE');
  }

  if (!doc) return;
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}.${pad(now.getMinutes())}`;
  doc.save(`Scan ${stamp}.pdf`);
}
