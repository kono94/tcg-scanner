// Copy local test videos into public/media (gitignored) for the camera mock.
// Only synthetic replays and held-out *test-split* excerpts: never training or validation sources.
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = join(dirname(fileURLToPath(import.meta.url)), '../..');
const out = join(repo, 'web/public/media');
const candidates = [
  { path: 'runs/card_detector/synthetic/synthetic_replay.mp4', name: 'Synthetic replay', note: 'Synthetic scenes with known identities; not a camera test' },
  { path: 'runs/video_mining/binder-search-op-slow/raw/4QxpRG4E6d0-300-312.webm', name: 'One Piece binder (test split)', note: 'Real 4K excerpt from a held-out test-split source; labels are AI-reviewed' },
  { path: 'runs/video_mining/binder-search-op-slow/raw/4QxpRG4E6d0-312-324.webm', name: 'One Piece binder 2 (test split)', note: 'Real 4K excerpt from a held-out test-split source; labels are AI-reviewed' },
];
mkdirSync(out, { recursive: true });
const videos = [];
for (const c of candidates) {
  const src = join(repo, c.path);
  if (!existsSync(src)) { console.warn(`missing: ${c.path}`); continue; }
  copyFileSync(src, join(out, basename(src)));
  videos.push({ name: c.name, url: `media/${basename(src)}`, note: c.note });
}
writeFileSync(join(out, 'manifest.json'), JSON.stringify({ videos }, null, 2) + '\n');
console.log(`${videos.length} test videos in public/media`);
