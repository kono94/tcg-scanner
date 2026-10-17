import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Gallery, validateEmbedding } from '../src/recognition/gallery';
import { EXPORT_DIR, hasFixtures, json, type EncoderFixture } from './fixtures';

describe.skipIf(!hasFixtures)('gallery search parity', () => {
  it('reproduces the Python top-5 on the fp16 gallery', () => {
    const meta = JSON.parse(readFileSync(join(EXPORT_DIR, 'gallery.json'), 'utf8')) as { identities: string[]; dimension: number };
    const raw = readFileSync(join(EXPORT_DIR, 'gallery.f16'));
    const gallery = Gallery.fromFloat16(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength), meta.dimension, meta.identities);
    expect(gallery.rows).toBe(24197);
    for (const f of json<{ fixtures: EncoderFixture[] }>('encoder.json').fixtures) {
      const hits = gallery.search(Float32Array.from(f.embedding_torch_cpu), 5);
      expect(hits.map((h) => h.identity)).toEqual(f.topk_torch_cpu.map((h) => h.identity));
      hits.forEach((h, i) => expect(Math.abs(h.score - f.topk_torch_cpu[i].score)).toBeLessThan(2e-3));
    }
  });
});

describe('embedding validation', () => {
  it('rejects zero and non-finite vectors', () => {
    expect(() => validateEmbedding(new Float32Array(4), 4)).toThrow();
    expect(() => validateEmbedding(Float32Array.of(NaN, 0, 0, 0), 4)).toThrow();
    expect(() => validateEmbedding(Float32Array.of(1, 0, 0, 0), 4)).not.toThrow();
  });

  it('collapses multiple templates of one identity', () => {
    const g = new Gallery(Float32Array.of(1, 0, 0.99, 0.141, 0, 1), 2, ['a', 'a', 'b']);
    expect(g.search(Float32Array.of(1, 0), 3).map((h) => h.identity)).toEqual(['a', 'b']);
  });
});

describe.skipIf(!hasFixtures)('int8 gallery rows', () => {
  it('keeps the top hits of the fp16 gallery', () => {
    const meta = JSON.parse(readFileSync(join(EXPORT_DIR, 'gallery.json'), 'utf8')) as { identities: string[]; dimension: number };
    const raw = readFileSync(join(EXPORT_DIR, 'gallery.f16'));
    const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const g16 = Gallery.fromFloat16(buf, meta.dimension, meta.identities);
    const g8 = Gallery.fromFloat16Int8(buf, meta.dimension, meta.identities);
    for (const f of json<{ fixtures: EncoderFixture[] }>('encoder.json').fixtures) {
      const q = Float32Array.from(f.embedding_torch_cpu);
      const a = g16.search(q, 5), b = g8.search(q, 5);
      expect(b[0].identity).toBe(a[0].identity);
      expect(Math.abs(b[0].score - a[0].score)).toBeLessThan(0.01);
    }
  });
});
