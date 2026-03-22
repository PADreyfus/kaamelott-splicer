/**
 * test/analyzer.test.mjs
 *
 * Run with:  npm test
 * Requires:  node-web-audio-api, music-metadata  (npm install)
 *
 * Unit tests are fast (synthetic data, no files).
 * Integration tests load real audio from data/ and may take several minutes.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { AudioContext } from 'node-web-audio-api';
import { parseBuffer } from 'music-metadata';

import {
  fft, blockFeatures, findMp3Sync,
  spectralMatchInSlice, buildJingleFp, analyzeBuffer,
  FFT_SIZE, N_BANDS, NCC_THRESHOLD,
} from '../lib/analyzer.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = join(ROOT, 'data');

// =====================================================================
// UNIT TESTS
// =====================================================================

test('fft: DC component of constant signal', () => {
  const N = 1024;
  const re = new Float32Array(N).fill(1.0);
  const im = new Float32Array(N);
  fft(re, im);
  // DC bin (index 0) magnitude should equal N
  assert.ok(Math.abs(re[0] - N) < 1e-3, `DC bin expected ${N}, got ${re[0]}`);
  // All other bins near zero
  for (let i = 1; i < N; i++) {
    assert.ok(
      Math.abs(re[i]) < 1e-2 && Math.abs(im[i]) < 1e-2,
      `Bin ${i} should be ~0, got re=${re[i]}, im=${im[i]}`
    );
  }
});

test('fft: single sinusoid produces correct frequency bin', () => {
  const N = 1024;
  const k = 4; // 4 cycles over the window
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) re[i] = Math.cos(2 * Math.PI * k * i / N);
  fft(re, im);
  // Peak should be at bin k (and N-k by symmetry)
  let maxMag = 0, maxBin = -1;
  for (let i = 0; i < N / 2; i++) {
    const mag = Math.hypot(re[i], im[i]);
    if (mag > maxMag) { maxMag = mag; maxBin = i; }
  }
  assert.equal(maxBin, k, `Peak bin expected ${k}, got ${maxBin}`);
});

test('blockFeatures: correct block count and band count', () => {
  const sr = 8000;
  const nBlocks = 5;
  const pcm = new Float32Array(nBlocks * FFT_SIZE); // silence
  const feats = blockFeatures(pcm, sr);
  assert.equal(feats.length, nBlocks, 'block count');
  for (const f of feats) {
    assert.equal(f.length, N_BANDS, 'band count per block');
  }
});

test('blockFeatures: silence produces low (but finite) values', () => {
  const sr = 8000;
  const pcm = new Float32Array(4 * FFT_SIZE); // all zeros
  const feats = blockFeatures(pcm, sr);
  for (const f of feats) {
    for (const v of f) {
      assert.ok(isFinite(v) && v >= 0, `band value should be finite non-negative, got ${v}`);
    }
  }
});

test('findMp3Sync: finds valid sync marker', () => {
  // Craft a minimal byte sequence with a valid sync pattern
  // 0xFF 0xFB (MPEG1, layer3, 192kbps, 44100Hz)
  const bytes = new Uint8Array([0x00, 0x00, 0xFF, 0xFB, 0x90, 0x00]);
  const result = findMp3Sync(bytes, 0);
  assert.equal(result, 2, `expected offset 2, got ${result}`);
});

test('findMp3Sync: returns -1 when no sync found', () => {
  const bytes = new Uint8Array([0x00, 0x01, 0x02, 0x03, 0x04, 0x05]);
  assert.equal(findMp3Sync(bytes, 0), -1);
});

test('findMp3Sync: respects start offset', () => {
  // Sync at offset 2, but we start searching at offset 4
  const bytes = new Uint8Array([0x00, 0x00, 0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00]);
  assert.equal(findMp3Sync(bytes, 4), -1, 'should not find sync before offset 4');
});

test('spectralMatchInSlice: returns null when jingleFp is null', () => {
  const pcm = new Float32Array(FFT_SIZE * 10);
  const result = spectralMatchInSlice(pcm, 8000, null);
  assert.equal(result, null);
});

test('spectralMatchInSlice: matches identical signal perfectly', () => {
  // Create a random-ish PCM signal, use it as both jingle and audio search target
  const sr = 8000;
  const jingleBlocks = 3;
  const jinglePcm = new Float32Array(FFT_SIZE * jingleBlocks);
  for (let i = 0; i < jinglePcm.length; i++) {
    jinglePcm[i] = Math.sin(i * 0.1) * 0.5; // deterministic signal
  }
  const jFeats = blockFeatures(jinglePcm, sr);
  let norm2 = 0;
  for (const f of jFeats) for (const v of f) norm2 += v * v;
  const fp = { features: jFeats, blockCount: jFeats.length, blockSec: FFT_SIZE / sr, norm: Math.sqrt(norm2) };

  // Embed jingle signal at offset 2 blocks within longer audio
  const padding = FFT_SIZE * 2;
  const audioPcm = new Float32Array(padding + jinglePcm.length + padding);
  audioPcm.set(jinglePcm, padding);

  const result = spectralMatchInSlice(audioPcm, sr, fp);
  assert.ok(result !== null, 'should find a match');
  assert.ok(result.score > NCC_THRESHOLD, `score ${result.score} should exceed threshold`);
});

// =====================================================================
// INTEGRATION TESTS  (load real audio files)
// =====================================================================

const JINGLE_PATH = join(DATA, 'jingle.wav');
const FULL_PATH   = join(DATA, 'Kaamelott Livre I - Tome 1 [b05Scfhi0dU].mp3');

/** Helper: get audio duration in seconds via music-metadata */
async function getAudioDuration(buf) {
  const meta = await parseBuffer(buf);
  return meta.format.duration;
}

/** Helper: create a 8 kHz AudioContext */
function makeCtx() { return new AudioContext({ sampleRate: 8000 }); }

test('integration: full file — detects ~50 episodes with correct first split points', { timeout: 10 * 60_000 }, async () => {
  const jingleBuf = await readFile(JINGLE_PATH);
  const fullBuf   = await readFile(FULL_PATH);

  const duration = await getAudioDuration(fullBuf);

  const jingleCtx = makeCtx();
  const fp = await buildJingleFp(jingleBuf.buffer, jingleCtx);
  jingleCtx.close();

  const result = await analyzeBuffer(
    fullBuf.buffer, duration, fp, makeCtx,
    (p) => process.stderr.write(`\r  full file: ${(p * 100).toFixed(0)}%  `),
    () => {},
  );
  process.stderr.write('\n');

  const nEpisodes = result.splitPoints.length - 1;
  const fmt = (s) => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
  console.log(`  full file split points (${nEpisodes} episodes): ${result.splitPoints.slice(0, 6).map(s => fmt(s)).join(', ')}${nEpisodes > 5 ? ', ...' : ''}`);

  // First four expected split points (jingle positions) in seconds
  //   0:00 → 0s  (always index 0 in splitPoints)
  //   3:13 → 193s
  //   6:04 → 364s
  //  10:01 → 601s
  const EXPECTED = [0, 193, 364, 601];
  for (let i = 0; i < EXPECTED.length; i++) {
    const got = result.splitPoints[i];
    assert.ok(
      got !== undefined && Math.abs(got - EXPECTED[i]) <= 1,
      `splitPoints[${i}] expected near ${EXPECTED[i]}s, got ${got?.toFixed(1)}s`
    );
  }

  // Around 50 episodes total
  assert.ok(
    nEpisodes >= 48 && nEpisodes <= 55,
    `expected 48–55 episodes, got ${nEpisodes}`
  );
});
