// Replaceable price service. Lookups are independent of recognition and never run in the camera loop.
import type { Conversion, Quote } from '../recognition/value';

export interface PriceService {
  /** Quotes per identity; null means no quote (unknown, never zero). */
  quotes(ids: readonly string[], signal?: AbortSignal): Promise<Map<string, Quote | null>>;
  readonly description: string;
  /** Currency of the quotes returned now (the selected market). */
  currency?(): string;
  /** USD->EUR rate recorded with the snapshot, if any. */
  usdToEur?(): Promise<number | null>;
}

type QuoteTable = Record<string, { amount: number; basis: string; converted?: Conversion }>;

/** Schema 1: one market (legacy bundled snapshot). Schema 2: USD and EUR markets. */
interface SnapshotFile {
  schema_version?: number;
  currency?: string;
  source?: string;
  snapshot_date: string;
  coverage?: string | Record<string, unknown>;
  quotes?: QuoteTable;
  markets?: Record<string, { currency: string; source: string; quotes: QuoteTable }>;
  fx?: { USD_EUR: number; date: string; source: string };
}

/** Prices from the bundled static snapshot, loaded lazily once. */
export class SnapshotPriceService implements PriceService {
  private file: Promise<SnapshotFile> | null = null;
  description = 'Bundled price snapshot';

  constructor(private readonly url: string, private readonly market: () => string = () => 'EUR') {}

  private table(f: SnapshotFile): { currency: string; source: string; quotes: QuoteTable } {
    if (f.markets) return f.markets[this.market()] ?? Object.values(f.markets)[0];
    return { currency: f.currency ?? 'USD', source: f.source ?? 'snapshot', quotes: f.quotes ?? {} };
  }

  private load(): Promise<SnapshotFile> {
    this.file ??= fetch(this.url).then((r) => {
      if (!r.ok) throw new Error(`Price snapshot unavailable (${r.status})`);
      return r.json() as Promise<SnapshotFile>;
    }).then((f) => {
      const t = this.table(f);
      this.description = `${t.source}, ${f.snapshot_date}`;
      return f;
    });
    this.file.catch(() => { this.file = null; });
    return this.file;
  }

  currency(): string {
    return this.market();
  }

  async usdToEur(): Promise<number | null> {
    return (await this.load()).fx?.USD_EUR ?? null;
  }

  async quotes(ids: readonly string[], signal?: AbortSignal): Promise<Map<string, Quote | null>> {
    const file = await this.load();
    signal?.throwIfAborted();
    const t = this.table(file);
    this.description = `${t.source}, ${file.snapshot_date}`;
    return new Map(ids.map((id) => {
      const q = t.quotes[id];
      return [id, q ? { amount: q.amount, currency: t.currency, basis: q.basis, asOf: file.snapshot_date, source: t.source, converted: q.converted } : null];
    }));
  }
}
