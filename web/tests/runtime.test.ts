// Cross-runtime parity: onnxruntime-web (WASM backend, as used by the browser fallback) vs PyTorch.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { decodeObb, letterboxFor } from '../src/vision/obb';
import { decodePose } from '../src/vision/pose';
import { encoderTensor } from '../src/vision/preprocess';
import { EXPORT_DIR, f32, hasFixtures, json, rgb, type DetectorFixture, type EncoderFixture } from './fixtures';

ort.env.wasm.numThreads = 1;

describe.skipIf(!hasFixtures || process.env.SKIP_RUNTIME === '1')('onnxruntime-web parity', () => {
  it('encodes TypeScript-preprocessed crops like PyTorch', async () => {
    const session = await ort.InferenceSession.create(readFileSync(join(EXPORT_DIR, 'encoder.onnx')), { executionProviders: ['wasm'] });
    for (const f of json<{ fixtures: EncoderFixture[] }>('encoder.json').fixtures.slice(0, 3)) {
      const input = new ort.Tensor('float32', encoderTensor(rgb(f.canonical_rgb, 224, 312)), [1, 3, 224, 224]);
      const out = (await session.run({ pixels: input })).embedding.data as Float32Array;
      const cos = f.embedding_torch_cpu.reduce((s, v, i) => s + v * out[i], 0);
      expect(cos).toBeGreaterThan(0.9999);
    }
    await session.release();
  });

  it('detects like PyTorch', async () => {
    const session = await ort.InferenceSession.create(readFileSync(join(EXPORT_DIR, 'detector.onnx')), { executionProviders: ['wasm'] });
    const f = json<{ fixtures: DetectorFixture[] }>('detector.json').fixtures.reduce((a, b) => (b.ultralytics_conf.length > a.ultralytics_conf.length ? b : a));
    const name = session.inputNames[0];
    const out = (await session.run({ [name]: new ort.Tensor('float32', f32(f.input), [1, 3, f.input_size ?? 320, f.input_size ?? 320]) }))[session.outputNames[0]];
    const expected = f32(f.output);
    let worst = 0;
    const data = out.data as Float32Array;
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(data[i] - expected[i]));
    expect(worst).toBeLessThan(0.01);
    const lb = letterboxFor(...f.letterbox.source_size, f.input_size ?? 640);
    const detections = f.kind === 'pose' ? decodePose(data, out.dims[2], lb, 0.25, 0.7, 300, false) : decodeObb(data, out.dims[2], lb);
    expect(detections.length).toBe(f.ultralytics_conf.length);
    await session.release();
  });
});
