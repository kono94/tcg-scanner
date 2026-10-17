// Pixel containers and coordinate-space types shared by the worker and tests.

/** Interleaved RGB uint8 pixels (no alpha), row-major. */
export interface RGBImage {
  width: number;
  height: number;
  data: Uint8Array;
}

/** A point in a named coordinate space. Spaces are never mixed implicitly. */
export type Point = readonly [number, number];

/** Four physical card corners in capture (oriented source) pixels, ordered TL, TR, BR, BL. */
export type Quad = readonly [Point, Point, Point, Point];

export function rgbFromRGBA(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): RGBImage {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height; i++, j += 3) {
    data[j] = rgba[i * 4];
    data[j + 1] = rgba[i * 4 + 1];
    data[j + 2] = rgba[i * 4 + 2];
  }
  return { width, height, data };
}

export function rgbaFromRGB(image: RGBImage): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, j = 0; i < image.width * image.height; i++, j += 3) {
    out[i * 4] = image.data[j];
    out[i * 4 + 1] = image.data[j + 1];
    out[i * 4 + 2] = image.data[j + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}
