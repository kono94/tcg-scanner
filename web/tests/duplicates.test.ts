import { describe, expect, it } from 'vitest';
import { appearanceFingerprint, probableDuplicate, similarAppearance } from '../src/batch/duplicates';
import type { RGBImage } from '../src/vision/image';

function image(seed: number, exposure = 0): RGBImage {
  const width = 60, height = 80, data = new Uint8Array(width * height * 3);
  let value = seed;
  for (let by = 0; by < 8; by++) for (let bx = 0; bx < 6; bx++) {
    value = (value * 1664525 + 1013904223) >>> 0;
    const colour = [60 + value % 100, 60 + (value >>> 8) % 100, 60 + (value >>> 16) % 100];
    for (let y = by * 10; y < by * 10 + 10; y++) for (let x = bx * 10; x < bx * 10 + 10; x++)
      for (let c = 0; c < 3; c++) data[(y * width + x) * 3 + c] = colour[c] + exposure;
  }
  return { width, height, data };
}

describe('probable page duplicate evidence', () => {
  it('tolerates exposure changes but distinguishes different appearance', () => {
    const a = appearanceFingerprint(image(1)), bright = appearanceFingerprint(image(1, 30)), other = appearanceFingerprint(image(48));
    expect(similarAppearance(a, bright)).toBe(true);
    expect(similarAppearance(a, other)).toBe(false);
    expect(similarAppearance(a, 'incompatible')).toBe(false);
  });

  it('matches physical copies one-to-one rather than suppressing shared catalog IDs', () => {
    const [a, b] = [1, 48].map((s) => appearanceFingerprint(image(s)));
    expect(probableDuplicate([a, a, b], [b, a, a])).toBe(true);
    expect(probableDuplicate([a, a, b], [a, b, b])).toBe(false);
  });

  it('supports single cards, larger scenes and a small number of missing detections', () => {
    const cards = Array.from({ length: 14 }, (_, i) => appearanceFingerprint(image(i + 1)));
    expect(probableDuplicate([cards[0]], [cards[0]])).toBe(true);
    expect(probableDuplicate([], [])).toBe(false);
    expect(probableDuplicate(cards, [...cards].reverse())).toBe(true);
    expect(probableDuplicate(cards, cards.slice(0, 12))).toBe(true);
    expect(probableDuplicate(cards, cards.slice(0, 8))).toBe(false);
  });
});
