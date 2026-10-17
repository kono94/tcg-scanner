// Host the pinned onnxruntime-web JS/WASM pairs next to the app so runtime and assets always match.
import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'node_modules/onnxruntime-web/dist');
const out = join(root, 'public/ort');
mkdirSync(out, { recursive: true });
const files = readdirSync(dist).filter((f) => /^ort-wasm-simd-threaded(\.jsep|\.asyncify)?\.(mjs|wasm)$/.test(f));
for (const f of files) copyFileSync(join(dist, f), join(out, f));
const version = JSON.parse(readFileSync(join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8')).version;
writeFileSync(join(out, 'VERSION'), version + '\n');
console.log(`onnxruntime-web ${version}: copied ${files.length} files to public/ort`);
