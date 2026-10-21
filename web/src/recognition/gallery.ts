// Exact cosine search over L2-normalized gallery rows. Scores are similarities, not probabilities.

export interface Hit {
  identity: string;
  row: number;
  score: number;
}

export function float16ToFloat32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

/** Reject zero, non-finite or unnormalized vectors before search. */
export function validateEmbedding(v: Float32Array, dimension: number): void {
  if (v.length !== dimension) throw new Error(`Embedding has ${v.length} values, expected ${dimension}`);
  let n = 0;
  for (let i = 0; i < v.length; i++) {
    if (!Number.isFinite(v[i])) throw new Error('Embedding contains non-finite values');
    n += v[i] * v[i];
  }
  if (Math.abs(Math.sqrt(n) - 1) > 0.01) throw new Error(`Embedding norm ${Math.sqrt(n).toFixed(3)} is not 1`);
}

export class Gallery {
  readonly rows: number;
  /** Optional compact storage: int8 values with one scale per row (about 4x smaller than float32). */
  private readonly q8: { values: Int8Array; scales: Float32Array } | null;

  constructor(readonly vectors: Float32Array, readonly dimension: number, readonly identities: readonly string[],
    q8: { values: Int8Array; scales: Float32Array } | null = null) {
    this.q8 = q8;
    this.rows = (q8 ? q8.values.length : vectors.length) / dimension;
    if (!Number.isInteger(this.rows) || this.rows !== identities.length) throw new Error('Gallery rows and identities differ');
  }

  static fromFloat16(buffer: ArrayBuffer, dimension: number, identities: readonly string[]): Gallery {
    return new Gallery(float16ToFloat32(new Uint16Array(buffer)), dimension, identities);
  }

  /** Int8 rows with a per-row scale, converted row by row (never holds the float32 copy). */
  static fromFloat16Int8(buffer: ArrayBuffer, dimension: number, identities: readonly string[]): Gallery {
    const half = new Uint16Array(buffer);
    const rows = half.length / dimension;
    const values = new Int8Array(half.length), scales = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      const row = float16ToFloat32(half.subarray(r * dimension, (r + 1) * dimension));
      let max = 0;
      for (let i = 0; i < dimension; i++) max = Math.max(max, Math.abs(row[i]));
      const scale = max / 127 || 1;
      scales[r] = scale;
      for (let i = 0; i < dimension; i++) values[r * dimension + i] = Math.round(row[i] / scale);
    }
    return new Gallery(new Float32Array(0), dimension, identities, { values, scales });
  }

  /** Top-k distinct identities (best template per identity), highest similarity first. `allowed`: rows to search (all when null). */
  search(query: Float32Array, k = 20, allowed: Uint8Array | null = null): Hit[] {
    validateEmbedding(query, this.dimension);
    const { vectors, dimension: d, rows, q8 } = this;
    // Keep a small candidate pool; multiple templates per identity are collapsed afterwards.
    const pool = Math.min(allowed ? allowed.reduce((n, a) => n + a, 0) : rows, k * 4);
    if (!pool) return [];
    const top: Hit[] = [];
    let floor = -Infinity;
    for (let r = 0; r < rows; r++) {
      if (allowed && !allowed[r]) continue;
      let s = 0;
      const o = r * d;
      if (q8) {
        const v = q8.values;
        for (let i = 0; i < d; i++) s += v[o + i] * query[i];
        s *= q8.scales[r];
      } else {
        for (let i = 0; i < d; i++) s += vectors[o + i] * query[i];
      }
      if (top.length < pool || s > floor) {
        top.push({ identity: this.identities[r], row: r, score: s });
        if (top.length > pool) {
          top.sort((a, b) => b.score - a.score);
          top.length = pool;
          floor = top[pool - 1].score;
        }
      }
    }
    top.sort((a, b) => b.score - a.score);
    const seen = new Set<string>();
    const hits: Hit[] = [];
    for (const h of top) {
      if (seen.has(h.identity)) continue;
      seen.add(h.identity);
      hits.push(h);
      if (hits.length === k) break;
    }
    return hits;
  }
}

/** Game of a gallery identity: `game:language:id`, or `back:game` for card backs. */
export function identityGame(identity: string): string {
  const [first, second] = identity.split(':');
  return first === 'back' ? second : first;
}

/** Rows whose game is in `games`. */
export function gameMask(identities: readonly string[], games: readonly string[]): Uint8Array {
  const wanted = new Set(games);
  return Uint8Array.from(identities, (id) => (wanted.has(identityGame(id)) ? 1 : 0));
}
