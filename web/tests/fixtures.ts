// Loads golden fixtures written by scripts/export_browser_models.py.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RGBImage } from '../src/vision/image';

const here = dirname(fileURLToPath(import.meta.url));
export const EXPORT_DIR = process.env.TCG_EXPORT_DIR ?? join(here, '../../runs/browser_export/gpu-20261017');
export const BUNDLE_DIR = join(here, '../public/bundle');
export const FIXTURES = join(EXPORT_DIR, 'fixtures');
export const hasFixtures = existsSync(join(FIXTURES, 'encoder.json'));

export function json<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURES, name), 'utf8')) as T;
}

export function f32(name: string): Float32Array {
  const b = readFileSync(join(FIXTURES, name));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

export function rgb(name: string, width: number, height: number): RGBImage {
  const b = readFileSync(join(FIXTURES, name));
  if (b.byteLength !== width * height * 3) throw new Error(`Unexpected size for ${name}`);
  return { width, height, data: new Uint8Array(b.buffer, b.byteOffset, b.byteLength) };
}

export interface EncoderFixture {
  name: string;
  expected_identity: string;
  source_rgb: string;
  source_size: [number, number];
  canonical_rgb: string;
  input: string;
  corners: [number, number][];
  embedding_torch_cpu: number[];
  topk_torch_cpu: { identity: string; index: number; score: number }[];
}

export interface DetectorFixture {
  kind: 'pose';
  name: string;
  input_size?: number;
  input: string;
  output: string;
  output_shape: number[];
  letterbox: { scale: number; pad_x: number; pad_y: number; source_size: [number, number] };
  true_corners: [number, number][][];
  ultralytics_corners?: [number, number][][];
  ultralytics_conf: number[];
}
