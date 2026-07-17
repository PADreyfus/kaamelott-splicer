/**
 * Dodo avec Kaamelott — load an MP3 compilation, cut it into episodes by
 * detecting the three-horns intro jingle, then play episodes randomly.
 *
 * Detection strategy (validated on Livre I Tome 1):
 *  - Every episode starts with the same ~2.6 s jingle, and the compilation
 *    itself starts with one at t=0. So the fingerprint is taken from the
 *    first JINGLE_SEC seconds of the loaded file itself — this avoids the
 *    codec-artifact mismatch of comparing a WAV fingerprint against MP3.
 *  - Spectral features (24 log-spaced bands, 1024-pt FFT blocks) are matched
 *    by normalized cross-correlation; local maxima above NCC_THRESHOLD that
 *    are at least MIN_GAP seconds apart become episode boundaries.
 *  - VBR MP3 seek fix: decodeAudioData gives a linear PCM timeline, but the
 *    <audio> element seeks by byte position. A frame index maps PCM time to
 *    byte offset, then to the element's linear seek time.
 */

const APP_VERSION = 'v17';   // bump on every deploy (also index.html ?v= and sw.js CACHE/SHELL)

const JINGLE_SEC = 2.6;      // length of the three-horns blast (fingerprint)
const SKIP_INTRO_SEC = 5.3;  // full intro incl. musical sting (measured: episodes
                             // are block-identical up to ~5.3 s, diverge at ~5.5)
const NCC_THRESHOLD = 0.65;  // minimum correlation to accept a jingle match
const MIN_GAP = 120;         // minimum episode length in seconds
const MIN_EPISODE = 5;       // discard segments shorter than this

const FFT_SIZE = 1024;
const N_BANDS = 24;

// ===================== FFT / SPECTRAL FEATURES =====================

// Cooley-Tukey radix-2 in-place FFT. re/im must be Float32Array of length 2^n.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cRe = 1, cIm = 0;
      for (let j = 0; j < (len >> 1); j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const k = i + j + (len >> 1);
        const vRe = re[k] * cRe - im[k] * cIm;
        const vIm = re[k] * cIm + im[k] * cRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[k] = uRe - vRe; im[k] = uIm - vIm;
        const nRe = cRe * wRe - cIm * wIm;
        cIm = cRe * wIm + cIm * wRe; cRe = nRe;
      }
    }
  }
}

// Block-level log-magnitude spectral features: one Float32Array(N_BANDS)
// per non-overlapping FFT_SIZE-sample block, 24 log-spaced bands 50–4000 Hz.
function blockFeatures(pcm, sr) {
  const halfN = FFT_SIZE >> 1;
  const loHz = 50, hiHz = Math.min(4000, sr / 2);
  const loBin = Math.max(1, Math.round(loHz * FFT_SIZE / sr));
  const hiBin = Math.min(halfN - 1, Math.round(hiHz * FFT_SIZE / sr));
  const edges = new Uint16Array(N_BANDS + 1);
  for (let b = 0; b <= N_BANDS; b++) {
    edges[b] = Math.round(loBin * Math.pow(hiBin / loBin, b / N_BANDS));
  }
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  const nBlocks = Math.floor(pcm.length / FFT_SIZE);
  const feats = [];
  for (let b = 0; b < nBlocks; b++) {
    const off = b * FFT_SIZE;
    for (let i = 0; i < FFT_SIZE; i++) {
      const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (FFT_SIZE - 1)));
      re[i] = pcm[off + i] * w;
      im[i] = 0;
    }
    fft(re, im);
    const bands = new Float32Array(N_BANDS);
    for (let band = 0; band < N_BANDS; band++) {
      const lo = edges[band], hi = edges[band + 1];
      let sum = 0;
      for (let k = lo; k <= hi; k++) sum += Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      bands[band] = Math.log1p(sum / (hi - lo + 1));
    }
    feats.push(bands);
  }
  return feats;
}

// ===================== MP3 BYTE↔TIME MAPPING (VBR) =====================

function findMp3Sync(bytes, off) {
  for (let i = Math.max(0, off); i < bytes.length - 3; i++) {
    if (bytes[i] !== 0xFF || (bytes[i + 1] & 0xE0) !== 0xE0) continue;
    const v = (bytes[i + 1] >> 3) & 3, l = (bytes[i + 1] >> 1) & 3,
          br = (bytes[i + 2] >> 4) & 0xF, sr = (bytes[i + 2] >> 2) & 3;
    if (v !== 1 && l !== 0 && br > 0 && br < 15 && sr < 3) return i;
  }
  return -1;
}

// Parse the Xing/Info VBR header: { nFrames, fileBytes, sr, spf, syncOff } or null.
function parseXingHeader(bytes) {
  let start = 0;
  if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) {
    const bodySize = ((bytes[6] & 0x7F) << 21) | ((bytes[7] & 0x7F) << 14) |
                     ((bytes[8] & 0x7F) << 7) | (bytes[9] & 0x7F);
    start = 10 + bodySize;
  }
  const syncOff = findMp3Sync(bytes, start);
  if (syncOff < 0 || syncOff + 120 > bytes.length) return null;

  const b1 = bytes[syncOff + 1], b2 = bytes[syncOff + 2], b3 = bytes[syncOff + 3];
  const mpegVer = (b1 >> 3) & 3;
  const layer = (b1 >> 1) & 3;
  if (layer !== 1 || mpegVer === 1) return null;
  const srIdx = (b2 >> 2) & 3;
  if (srIdx === 3) return null;
  const srTable = [[44100, 48000, 32000], [22050, 24000, 16000], [11025, 12000, 8000]];
  const sr = srTable[mpegVer === 3 ? 0 : mpegVer === 2 ? 1 : 2][srIdx];
  const spf = mpegVer === 3 ? 1152 : 576;
  const isMono = ((b3 >> 6) & 3) === 3;
  const sideInfoSize = mpegVer === 3 ? (isMono ? 17 : 32) : (isMono ? 9 : 17);
  const xOff = syncOff + 4 + sideInfoSize;
  if (xOff + 8 > bytes.length) return null;

  const tag = String.fromCharCode(bytes[xOff], bytes[xOff + 1], bytes[xOff + 2], bytes[xOff + 3]);
  if (tag !== 'Xing' && tag !== 'Info') return null;

  const flags = (bytes[xOff + 4] << 24) | (bytes[xOff + 5] << 16) |
                (bytes[xOff + 6] << 8) | bytes[xOff + 7];
  let off = xOff + 8, nFrames = null, fileBytes = null;
  if (flags & 0x01) {
    nFrames = (bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3];
    off += 4;
  }
  if (flags & 0x02) {
    fileBytes = (bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3];
  }
  return { nFrames, fileBytes, sr, spf, syncOff };
}

// Walk every MP3 frame and record (byteOffset, timeSec) every 100 frames,
// so linear interpolation gives an accurate PCM-time → byte mapping.
function buildFrameIndex(bytes, startByte) {
  const BITRATES = [
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
  ];
  const SR_TABLE = [[44100, 48000, 32000], [22050, 24000, 16000], [11025, 12000, 8000]];
  const byteArr = [startByte], timeArr = [0];
  let off = startByte, frameTime = 0, frameCount = 0;
  while (off + 4 <= bytes.length) {
    if (bytes[off] !== 0xFF || (bytes[off + 1] & 0xE0) !== 0xE0) { off++; continue; }
    const b1 = bytes[off + 1], b2 = bytes[off + 2];
    const mpegVer = (b1 >> 3) & 3, layer = (b1 >> 1) & 3;
    const brIdx = (b2 >> 4) & 0xF, srIdx = (b2 >> 2) & 3, padding = (b2 >> 1) & 1;
    if (mpegVer === 1 || layer !== 1 || srIdx === 3 || brIdx === 0 || brIdx === 15) { off++; continue; }
    const mIdx = mpegVer === 3 ? 0 : mpegVer === 2 ? 1 : 2;
    const sr = SR_TABLE[mIdx][srIdx];
    const spf = mpegVer === 3 ? 1152 : 576;
    const br = BITRATES[mpegVer === 3 ? 0 : 1][brIdx] * 1000;
    const frameSize = Math.floor(144 * br / sr) + padding;
    if (frameSize < 24 || off + frameSize > bytes.length) break;
    off += frameSize; frameTime += spf / sr; frameCount++;
    if (frameCount % 100 === 0) { byteArr.push(off); timeArr.push(frameTime); }
  }
  if (byteArr[byteArr.length - 1] !== off) { byteArr.push(off); timeArr.push(frameTime); }
  return { byteTable: new Int32Array(byteArr), timeTable: new Float64Array(timeArr) };
}

// ===================== ANALYSIS =====================

async function analyzeFile(fileAB, onProgress, onStatus) {
  const bytes = new Uint8Array(fileAB);

  const xing = parseXingHeader(bytes);
  const audioStartByte = xing?.syncOff ?? Math.max(0, findMp3Sync(bytes, 0));
  // The frame index maps byte offsets ↔ decoder-clock time; used only to
  // slice the file into decodable chunks and anchor each chunk's start time.
  const { byteTable, timeTable } = buildFrameIndex(bytes, audioStartByte);

  const ctx = new AudioContext();
  const sr = ctx.sampleRate;
  const blockSec = FFT_SIZE / sr;

  // Decoding a multi-hour MP3 in one decodeAudioData call needs gigabytes of
  // PCM and crashes the tab. Instead, slice the file at MP3 frame boundaries
  // into ~5 min chunks and detect jingles chunk by chunk. Each detection is
  // anchored to the chunk's absolute start time from the frame index, so the
  // per-chunk decoder padding never accumulates into a drift. Chunks overlap
  // by OVERLAP_SEC so a jingle sitting on a boundary is fully inside at least
  // one chunk; duplicates from the overlap are removed by the MIN_GAP
  // suppression below.
  const CHUNK_SEC = 300;
  const OVERLAP_SEC = 8;
  const indexEnd = timeTable[timeTable.length - 1];

  // Largest frame-index entry with timeTable[i] <= t
  const idxAtTime = t => {
    let lo = 0, hi = timeTable.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (timeTable[mid] <= t) lo = mid; else hi = mid - 1;
    }
    return lo;
  };

  const chunks = [];
  for (let t = 0; t === 0 || t < indexEnd; t += CHUNK_SEC) {
    const i0 = idxAtTime(Math.max(0, t - OVERLAP_SEC));
    const i1 = idxAtTime(t + CHUNK_SEC + OVERLAP_SEC) + 1;
    const last = i1 >= byteTable.length;
    chunks.push({
      startByte: byteTable[i0],
      endByte: last ? bytes.length : byteTable[i1],
      startTime: timeTable[i0],
      endTime: last ? indexEnd : timeTable[i1],
    });
  }

  let jFeats = null, jNorm = 0, J = 0;
  const candidates = [];

  for (let ci = 0; ci < chunks.length; ci++) {
    onStatus(`Analyse ${ci + 1}/${chunks.length}…`);
    let buf;
    try {
      buf = await ctx.decodeAudioData(fileAB.slice(chunks[ci].startByte, chunks[ci].endByte));
    } catch (e) {
      continue; // undecodable slice (e.g. trailing garbage) — skip it
    }
    const pcm = buf.getChannelData(0);

    // Fingerprint = the three-horns jingle that opens the compilation:
    // the first JINGLE_SEC seconds of the file itself.
    if (ci === 0) {
      const fpSamples = Math.min(pcm.length, Math.floor(JINGLE_SEC * sr));
      jFeats = blockFeatures(pcm.subarray(0, fpSamples), sr);
      if (jFeats.length < 2) throw new Error('Fichier trop court');
      let jNorm2 = 0;
      for (const f of jFeats) for (const v of f) jNorm2 += v * v;
      jNorm = Math.sqrt(jNorm2);
      J = jFeats.length;
    }

    // Rescale in-chunk offsets so decoded-chunk seconds line up with the
    // frame-index seconds the chunk anchors use (they can differ slightly
    // from decoder padding at the chunk edges).
    const walkedSpan = chunks[ci].endTime - chunks[ci].startTime;
    const timeScale = (buf.duration > 0 && walkedSpan > 0) ? walkedSpan / buf.duration : 1;

    const feats = blockFeatures(pcm, sr);
    const N = feats.length;
    if (N >= J) {
      const scores = new Float32Array(N - J + 1);
      for (let o = 0; o <= N - J; o++) {
        let dot = 0, aNorm2 = 0;
        for (let t = 0; t < J; t++) {
          const jf = jFeats[t], af = feats[o + t];
          for (let b = 0; b < N_BANDS; b++) { dot += jf[b] * af[b]; aNorm2 += af[b] * af[b]; }
        }
        scores[o] = dot / (jNorm * Math.sqrt(aNorm2) + 1e-9);
      }
      // Local maxima above threshold, anchored to the chunk's absolute time
      for (let o = 0; o < scores.length; o++) {
        if (scores[o] < NCC_THRESHOLD) continue;
        let isMax = true;
        for (let d = 1; d <= J && isMax; d++) {
          if (o - d >= 0 && scores[o - d] >= scores[o]) isMax = false;
          if (o + d < scores.length && scores[o + d] > scores[o]) isMax = false;
        }
        if (isMax) candidates.push({ time: chunks[ci].startTime + o * blockSec * timeScale, score: scores[o] });
      }
    }

    onProgress((ci + 1) / chunks.length);
    await new Promise(r => setTimeout(r)); // let the UI breathe
  }
  if (!jFeats) throw new Error('Décodage impossible');

  // Greedy non-maximum suppression: accepted peaks at least MIN_GAP apart.
  // This also removes duplicate detections from the chunk overlap regions.
  candidates.sort((a, b) => b.score - a.score);
  const accepted = [];
  for (const c of candidates) {
    if (accepted.every(a => Math.abs(a.time - c.time) >= MIN_GAP)) accepted.push(c);
  }
  accepted.sort((a, b) => a.time - b.time);

  // Split times are on the decoder clock (verified against user-checked
  // jingle positions 3:18 / 6:14 / 10:22 on the full file).
  const totalDur = timeTable[timeTable.length - 1];
  const splits = [
    0,
    ...accepted
      .map(a => a.time)
      .filter(t => t >= MIN_GAP && t <= totalDur - MIN_GAP), // a final jingle with no episode after it is not a cut
    totalDur,
  ];

  // Each split time → byte offset via the frame index. Episodes are played by
  // decoding their exact byte slice — never by seeking the <audio> element,
  // whose VBR seek lands 15–40 s off on this file (measured in Chrome).
  const byteAtTime = t => {
    let lo = 0, hi = timeTable.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (timeTable[mid] <= t) lo = mid; else hi = mid - 1;
    }
    if (lo >= timeTable.length - 1) return byteTable[lo];
    const frac = (t - timeTable[lo]) / (timeTable[lo + 1] - timeTable[lo]);
    return Math.round(byteTable[lo] + frac * (byteTable[lo + 1] - byteTable[lo]));
  };
  const splitBytes = splits.map(byteAtTime);
  splitBytes[splitBytes.length - 1] = bytes.length;
  // The first slice must not start at the Xing/Info metadata frame: it
  // declares the FULL file's duration, so the browser would never fire
  // 'ended' for that episode. Start at the first real audio frame instead.
  if (xing) splitBytes[0] = findMp3Sync(bytes, xing.syncOff + 4);

  try { ctx.close(); } catch (e) {}
  onProgress(1);
  return { splitPoints: splits, splitBytes, totalDuration: totalDur };
}

// ===================== APP STATE =====================

let compilations = [];   // { id, name, totalDuration }
let episodes = [];       // { id, compilationId, compilationName, index, startSec, endSec, duration, label, startByte, endByte, url? }
let sourceFiles = {};    // compilationId → File (episodes are byte slices of it)
let currentEp = null;
let playing = false;
let autoPlay = true;
let skipIntro = localStorage.getItem('skipIntro') === '1';
let sleepUntil = null; // epoch ms; playback pauses once passed

// ONE audio element for everything. With the screen locked, Android only lets
// a page CONTINUE its existing playback session — play() on a second/swapped
// element is treated as a new session and blocked (tested: Galaxy S25). So
// the next episode is fully pre-downloaded to a blob and, at 'ended', this
// same element's src flips to that in-memory blob — the continuation pattern
// web radios use.
const audio = document.createElement('audio');
audio.preload = 'auto';
let progressRAF = null;

const $ = id => document.getElementById(id);

function fmt(s) {
  if (!isFinite(s) || s < 0) return '0:00';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
}

// ===================== PLAYBACK =====================

function stopPlay() {
  if (progressRAF) { cancelAnimationFrame(progressRAF); progressRAF = null; }
  audio.pause();
  playing = false;
  render();
}

// Each episode plays as its own MP3: either a hosted file (ep.src) or a byte
// slice of a locally loaded compilation. Playback always starts at 0, so the
// <audio> element never seeks — Chrome's VBR seek is 15–40 s off on the full
// compilation (measured), byte slices are exact.
function episodeURL(ep) {
  if (ep.blobUrl) return ep.blobUrl; // fully downloaded copy (see preloadNext)
  if (ep.src) return ep.src;
  if (!ep.url) {
    const file = sourceFiles[ep.compilationId];
    if (!file) return null;
    ep.url = URL.createObjectURL(file.slice(ep.startByte, ep.endByte, 'audio/mpeg'));
  }
  return ep.url;
}

function nextEpOf(ep) {
  if (!episodes.length) return null;
  const i = ep ? episodes.findIndex(e => e.id === ep.id) : -1;
  return episodes[(i + 1) % episodes.length];
}

// ============== GAPLESS STREAM (MediaSource) ==============
// Every episode is appended into ONE continuous stream on ONE element. The
// playhead glides across episode boundaries — no 'ended' event, no src
// change, no play() call at the transition, so a locked Android has nothing
// to block. (The src-flip-at-'ended' approach still stopped on a locked
// Galaxy S25 even with correct durations and predownloaded audio.)
const MSE_OK = typeof MediaSource !== 'undefined' && MediaSource.isTypeSupported('audio/mpeg');

let ms = null, sb = null;
let segments = [];     // [{ep, start, end}] on the stream timeline
let appending = false; // one appendBuffer in flight at a time

function currentSegment() {
  const t = audio.currentTime;
  for (let i = segments.length - 1; i >= 0; i--) {
    if (t >= segments[i].start - 0.05) return segments[i];
  }
  return segments[0] || null;
}

// A fetch on a locked phone can hang forever (network suspended, promise
// never settles). The timeout makes it reject, which releases the
// `appending` flag so the watchdog can retry — otherwise the whole
// buffering pipeline deadlocks and playback stops at the buffer's edge.
const FETCH_TIMEOUT_MS = 20_000;
const AHEAD_SEC = 300;       // keep ~5 min appended beyond the playhead
const PREFETCH_COUNT = 2;    // episodes kept as bytes in RAM, ready to append

// Trim a slice to whole MP3 frames. Episode slices can start/end mid-frame;
// in sequence mode the demuxer stitches one file's partial tail against the
// next file's head into a garbage packet. Playing through it usually works,
// but when the decoder is STARVED at that exact timestamp (playhead stalled
// at the buffer edge — the locked-phone case) it decodes the garbage
// immediately and kills the pipeline: PIPELINE_ERROR_DECODE, media error 3,
// stream dead. Appending only whole frames makes that packet impossible.
function trimToFrames(ab) {
  const b = new Uint8Array(ab);
  const start = findMp3Sync(b, 0);
  if (start < 0) return ab;
  let off = start, lastEnd = start;
  while (off + 4 <= b.length) {
    if (b[off] !== 0xFF || (b[off + 1] & 0xE0) !== 0xE0) { off++; continue; }
    const b1 = b[off + 1], b2 = b[off + 2];
    const ver = (b1 >> 3) & 3, layer = (b1 >> 1) & 3;
    const brIdx = (b2 >> 4) & 0xF, srIdx = (b2 >> 2) & 3, pad = (b2 >> 1) & 1;
    if (ver === 1 || layer !== 1 || srIdx === 3 || brIdx === 0 || brIdx === 15) { off++; continue; }
    const BR = ver === 3
      ? [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
      : [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0];
    const SR = [[44100, 48000, 32000], [22050, 24000, 16000], [11025, 12000, 8000]][ver === 3 ? 0 : ver === 2 ? 1 : 2];
    const size = Math.floor((ver === 3 ? 144 : 72) * BR[brIdx] * 1000 / SR[srIdx]) + pad;
    if (size < 24) { off++; continue; }
    if (off + size > b.length) break; // trailing partial frame — drop it
    off += size; lastEnd = off;
  }
  return (start === 0 && lastEnd === b.length) ? ab : ab.slice(start, lastEnd);
}

async function epBytes(ep) {
  if (ep._bytes) return ep._bytes;
  let ab;
  if (ep.src) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
    try {
      const resp = await fetch(ep.src, { signal: ctl.signal });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      ab = await resp.arrayBuffer();
    } finally {
      clearTimeout(timer);
    }
  } else {
    const file = sourceFiles[ep.compilationId];
    if (!file) throw new Error('fichier source absent');
    ab = await file.slice(ep.startByte, ep.endByte).arrayBuffer();
  }
  ep._bytes = trimToFrames(ab);
  return ep._bytes;
}

// Keep the next PREFETCH_COUNT episodes' bytes in RAM so appending never
// depends on the network at the moment it's needed (locked phones may
// suspend it). RAM survives locking; the network doesn't.
let prefetching = false;
async function refillPrefetch() {
  if (prefetching || !segments.length) return;
  prefetching = true;
  try {
    let ep = segments[segments.length - 1].ep;
    for (let i = 0; i < PREFETCH_COUNT; i++) {
      ep = nextEpOf(ep);
      if (!ep || segments.some(s => s.ep.id === ep.id)) break;
      if (!ep._bytes) await epBytes(ep);
    }
  } catch (e) {
    // network gone — the watchdog will call us again
  } finally {
    prefetching = false;
  }
}

async function appendEpisode(ep) {
  if (!sb || appending || segments.some(s => s.ep.id === ep.id)) return;
  appending = true;
  try {
    const bytes = await epBytes(ep);
    if (!sb || ms?.readyState !== 'open' || sb.updating) return;
    const startAt = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0;
    if (!bytes.byteLength) {
      // Empty/corrupt file (it happened: ep07.mp3 shipped as 0 bytes) — record
      // a zero-length segment so the chain steps over it instead of stalling.
      segments.push({ ep, start: startAt, end: startAt });
      return;
    }
    await new Promise((ok, no) => {
      const onErr = () => no(new Error('append error'));
      sb.addEventListener('updateend', () => { sb.removeEventListener('error', onErr); ok(); }, { once: true });
      sb.addEventListener('error', onErr, { once: true });
      sb.appendBuffer(bytes);
    });
    const end = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : startAt;
    segments.push({ ep, start: startAt, end });
    delete ep._bytes; // free the download once it lives in the SourceBuffer
  } catch (e) {
    // Out of SourceBuffer quota: free played audio now so the retry can fit.
    if (e.name === 'QuotaExceededError') trimBehind(30);
    console.warn('append:', e.message);
  } finally {
    appending = false;
  }
  refillPrefetch();
}

function trimBehind(keepSec) {
  if (!sb || sb.updating || !sb.buffered.length) return;
  const start = sb.buffered.start(0);
  const upTo = audio.currentTime - keepSec;
  if (upTo <= start) return;
  try {
    sb.remove(start, upTo);
    segments = segments.filter(s => s.end > upTo);
  } catch (e) {}
}

// The one entry point that keeps the stream fed: append episodes until
// AHEAD_SEC of audio sits beyond the playhead, and keep the RAM prefetch
// full. Called from timeupdate, from 'waiting', and from the watchdog.
function ensureBuffered() {
  if (!MSE_OK || !sb || !autoPlay || !segments.length) return;
  const end = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0;
  if (end - audio.currentTime < AHEAD_SEC) {
    const nxt = nextEpOf(segments[segments.length - 1].ep);
    if (nxt && !segments.some(s => s.ep.id === nxt.id)) appendEpisode(nxt);
  }
  refillPrefetch();
}

// Watchdog: timeupdate stops firing the moment playback stalls, and a single
// 'waiting' event is not enough if the retry it triggers also fails. This
// interval keeps retrying — background throttling slows it to ~1/min on a
// locked phone, which is still enough to recover.
setInterval(() => { if (playing && currentEp) ensureBuffered(); }, 5000);

function teardownStream() {
  if (ms && ms.readyState === 'open') { try { ms.endOfStream(); } catch (e) {} }
  ms = null; sb = null; segments = [];
}

function playEp(ep) {
  if (progressRAF) { cancelAnimationFrame(progressRAF); progressRAF = null; }
  currentEp = ep;
  if (MSE_OK) {
    teardownStream();
    ms = new MediaSource();
    audio.src = URL.createObjectURL(ms);
    ms.addEventListener('sourceopen', () => {
      URL.revokeObjectURL(audio.src);
      sb = ms.addSourceBuffer('audio/mpeg');
      // Each MP3's internal timeline starts at 0 — sequence mode stacks
      // appends one after another instead of overlapping them.
      sb.mode = 'sequence';
      appendEpisode(ep).then(() => {
        if (skipIntro && segments.length) audio.currentTime = segments[0].start + SKIP_INTRO_SEC;
        ensureBuffered(); // start stacking the following episodes right away
      });
    }, { once: true });
  } else {
    const url = episodeURL(ep); // legacy fallback (no MediaSource, e.g. old iOS)
    if (!url) return;
    audio.src = url;
    if (skipIntro) {
      audio.addEventListener('loadedmetadata', () => { audio.currentTime = SKIP_INTRO_SEC; }, { once: true });
    }
  }
  audio.play().then(() => {
    playing = true;
    render();
    trackProgress();
    updateMediaSession();
  }).catch(err => {
    console.warn('play() refusé:', err.message);
    playing = false;
    render();
  });
}

// Keep `playing` and the ▶/⏸ buttons in sync with what the element really
// does (lock-screen controls, OS interruptions, play() resolving late).
audio.onplay = () => { playing = true; render(); };
audio.onpause = () => { if (!audio.ended) { playing = false; render(); } };
// Legacy fallback only: chain on 'ended' (MSE never ends its stream).
audio.onended = () => {
  if (MSE_OK) { stopPlay(); return; }
  if (!autoPlay || sleepExpired()) { stopPlay(); return; }
  playNext();
};

// timeupdate keeps firing during locked-screen audio playback (unlike RAF,
// which freezes) — it drives everything that must work while locked:
// episode-boundary bookkeeping, buffering ahead, sleep timer, memory trim.
audio.ontimeupdate = () => {
  if (!currentEp) return;
  if (playing && sleepExpired()) { togglePause(); return; }
  if (!MSE_OK || !segments.length) return;

  // Crossed into the next segment? Update state — playback itself just glides.
  const seg = currentSegment();
  if (seg && seg.ep.id !== currentEp.id) {
    currentEp = seg.ep;
    if (skipIntro && audio.currentTime < seg.start + SKIP_INTRO_SEC - 0.3) {
      audio.currentTime = seg.start + SKIP_INTRO_SEC;
    }
    updateMediaSession();
    render();
  }

  // Auto off: stop at the end of the current episode.
  const last = segments[segments.length - 1];
  if (!autoPlay && seg && seg.ep.id === last.ep.id && audio.currentTime >= seg.end - 0.1) {
    stopPlay();
    return;
  }

  // Keep AHEAD_SEC of audio appended beyond the playhead and the RAM
  // prefetch full — this is what survives a locked screen.
  ensureBuffered();

  // Trim played audio so a whole night stays flat in memory; drop segment
  // records that are entirely behind the trim point.
  if (sb && !sb.updating && sb.buffered.length &&
      audio.currentTime - sb.buffered.start(0) > 600) {
    trimBehind(120);
  }
};
// If we ever stall at the buffer's edge (append missed its window), retry —
// the moment data lands, playback resumes on its own.
audio.onwaiting = ensureBuffered;
audio.onstalled = ensureBuffered;
// A decode error kills the whole MediaSource pipeline (readyState 'ended',
// no more appends possible). Rebuild the stream from the current episode —
// best effort: with the screen locked the play() may be blocked, but with
// whole-frame appends this path should not trigger at all.
audio.onerror = () => {
  if (!MSE_OK || !currentEp || !ms) return;
  console.warn('media error', audio.error?.code, '— rebuilding stream');
  playEp(currentEp);
};

function segElapsed() {
  const seg = MSE_OK ? currentSegment() : null;
  return seg ? audio.currentTime - seg.start : audio.currentTime;
}

function setProgressUI(elapsed) {
  const pct = Math.min(100, Math.max(0, elapsed / currentEp.duration * 100));
  $('pFill').style.width = pct + '%';
  $('pKnob').style.left = pct + '%';
  $('tElapsed').textContent = fmt(elapsed);
}

function trackProgress() {
  const tick = () => {
    if (!playing || !currentEp) return;
    if (!seeking) setProgressUI(segElapsed());
    progressRAF = requestAnimationFrame(tick);
  };
  progressRAF = requestAnimationFrame(tick);
}

// Touch/drag seeking on the progress bar. Handlers live on the bar itself
// with pointer capture — document-level listeners missed parts of the touch
// sequence on mobile. Position is applied on release; a seek inside an
// episode blob is small and accurate (unlike the full VBR compilation,
// which must never be seeked).
let seeking = false;
{
  const bar = $('seekBar');
  const posToTime = e => {
    const r = bar.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    return frac * (currentEp ? currentEp.duration : 0);
  };
  bar.addEventListener('pointerdown', e => {
    if (!currentEp) return;
    e.preventDefault();
    seeking = true;
    bar.classList.add('dragging');
    try { bar.setPointerCapture(e.pointerId); } catch (err) {}
    setProgressUI(posToTime(e));
  });
  bar.addEventListener('pointermove', e => {
    if (seeking) { e.preventDefault(); setProgressUI(posToTime(e)); }
  });
  const finish = e => {
    if (!seeking) return;
    seeking = false;
    bar.classList.remove('dragging');
    if (currentEp) {
      const t = Math.max(0, Math.min(posToTime(e), currentEp.duration - 1));
      // In MSE mode the element timeline is the whole stream: offset into
      // the current episode's segment.
      const seg = MSE_OK ? currentSegment() : null;
      audio.currentTime = seg ? seg.start + t : t;
      setProgressUI(t);
    }
  };
  bar.addEventListener('pointerup', finish);
  bar.addEventListener('pointercancel', finish);
}

// Playback follows the (shuffled) list order.
function playNext() {
  const nxt = nextEpOf(currentEp);
  if (nxt) playEp(nxt);
}

function shuffleEpisodes() {
  for (let i = episodes.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [episodes[i], episodes[j]] = [episodes[j], episodes[i]];
  }
  render();
  if (!playing) playNext();
  else refillPrefetch(); // order changed — predownload the new next episodes
}

// Sleep timer: when armed, playback pauses once the deadline passes.
function sleepExpired() {
  if (sleepUntil === null || Date.now() < sleepUntil) return false;
  sleepUntil = null;
  render();
  return true;
}

function toggleSleep() {
  sleepUntil = sleepUntil === null ? Date.now() + 30 * 60_000 : null;
  render();
}

function updateMediaSession() {
  if (!('mediaSession' in navigator) || !currentEp) return;
  navigator.mediaSession.metadata = new MediaMetadata({
    title: currentEp.label, artist: 'Kaamelott', album: currentEp.compilationName,
  });
  navigator.mediaSession.setActionHandler('play', togglePause);
  navigator.mediaSession.setActionHandler('pause', togglePause);
  navigator.mediaSession.setActionHandler('nexttrack', playNext);
}

function togglePause() {
  if (!currentEp) { playNext(); return; }
  if (playing) {
    audio.pause();
    if (progressRAF) { cancelAnimationFrame(progressRAF); progressRAF = null; }
    playing = false;
  } else {
    audio.play().then(() => { playing = true; trackProgress(); }).catch(() => {});
  }
  render();
}

function deleteEp(id) {
  const ep = episodes.find(e => e.id === id);
  if (!ep) return;
  if (!confirm(`Supprimer définitivement ${ep.label} ?`)) return;
  if (currentEp?.id === id) { stopPlay(); currentEp = null; }
  if (ep.url) URL.revokeObjectURL(ep.url);
  episodes = episodes.filter(e => e.id !== id);
  if (ep.file) {
    // Remember locally right away (works even without a token / while the
    // Pages redeploy is pending), then delete on the server.
    let deleted = [];
    try { deleted = JSON.parse(localStorage.getItem('deletedEps')) || []; } catch (e) {}
    if (!deleted.includes(ep.file)) deleted.push(ep.file);
    localStorage.setItem('deletedEps', JSON.stringify(deleted));
    serverDeleteEpisode(ep.file, ep.label)
      .then(done => { if (done) console.log(ep.file + ' supprimé du serveur'); })
      .catch(err => alert('Suppression serveur échouée: ' + err.message +
        '\n(L\'épisode reste masqué sur cet appareil.)'));
  }
  render();
}

// ===================== FILE LOADING =====================

async function handleFiles(files) {
  $('loadArea').style.display = '';
  $('uploadArea').style.display = 'none';
  for (let fi = 0; fi < files.length; fi++) {
    const file = files[fi];
    $('loadTitle').textContent = `Analyse ${fi + 1}/${files.length}`;
    $('loadSub').textContent = file.name;
    $('loadFill').style.width = '0%';
    try {
      const cid = 'c' + Date.now() + '_' + fi;
      sourceFiles[cid] = file;
      const ab = await file.arrayBuffer();
      const res = await analyzeFile(
        ab,
        p => { $('loadFill').style.width = Math.round(p * 100) + '%'; },
        s => { $('loadSub').textContent = s; },
      );
      const name = file.name.replace(/\.[^.]+$/, '');
      compilations.push({ id: cid, name, totalDuration: res.totalDuration });
      for (let i = 0; i < res.splitPoints.length - 1; i++) {
        const s = res.splitPoints[i], e = res.splitPoints[i + 1];
        if (e - s < MIN_EPISODE) continue;
        episodes.push({
          id: cid + '_e' + i, compilationId: cid, compilationName: name,
          index: i, startSec: s, endSec: e, duration: e - s,
          startByte: res.splitBytes[i], endByte: res.splitBytes[i + 1],
          label: 'Épisode ' + (i + 1),
        });
      }
      $('loadSub').textContent = (res.splitPoints.length - 1) + ' épisodes détectés';
      await new Promise(r => setTimeout(r, 400));
    } catch (err) {
      alert('Erreur: ' + err.message);
    }
  }
  $('loadArea').style.display = 'none';
  render();
}

// ===================== RENDER =====================

// While the hosted-episodes check is in flight, show neither the upload card
// nor an empty player — avoids flashing "load a file" on slow connections.
let hostedChecked = false;

function render() {
  const has = episodes.length > 0;
  $('uploadArea').style.display = (has || !hostedChecked) ? 'none' : '';
  $('playerArea').style.display = has ? '' : 'none';

  // Now playing
  if (currentEp) {
    $('npCard').style.display = '';
    $('npTitle').textContent = currentEp.label;
    $('npSub').textContent = currentEp.compilationName;
    $('tTotal').textContent = fmt(currentEp.duration);
  } else {
    $('npCard').style.display = 'none';
  }
  $('bPlay').textContent = playing ? '⏸' : '▶';
  $('bAuto').classList.toggle('on', autoPlay);
  $('bSkipIntro').classList.toggle('on', skipIntro);
  $('bSleep').classList.toggle('on', sleepUntil !== null);
  $('bSleep').textContent = sleepUntil !== null
    ? '⏾ ' + fmt(Math.max(0, (sleepUntil - Date.now()) / 1000))
    : '⏾ 30 min';

  // Episode list
  $('epCount').textContent = episodes.length + ' épisode' + (episodes.length > 1 ? 's' : '');
  const list = $('epList');
  list.innerHTML = '';
  // Render in list order — playback follows it, and 🔀 reorders it.
  let lastComp = null;
  episodes.forEach(ep => {
    if (ep.compilationId !== lastComp) {
      lastComp = ep.compilationId;
      const comp = compilations.find(c => c.id === ep.compilationId);
      const header = document.createElement('div');
      header.className = 'comp-header';
      header.textContent = '🏰 ' + (comp ? comp.name : ep.compilationName);
      list.appendChild(header);
    }
    const row = document.createElement('div');
    row.className = 'ep-row' + (currentEp?.id === ep.id ? ' playing' : '');
    const meta = ep.src
      ? fmt(ep.duration)
      : `${fmt(ep.startSec)} → ${fmt(ep.endSec)} · ${fmt(ep.duration)}`;
    row.innerHTML =
      `<button class="btn-play" data-play="${ep.id}">▶</button>` +
      `<div class="ep-info"><div class="ep-name">${ep.label}</div>` +
      `<div class="ep-meta">${meta}</div></div>` +
      `<button class="btn-del" data-del="${ep.id}" title="Supprimer">✕</button>`;
    list.appendChild(row);
  });
}

// ===================== EVENTS =====================

document.addEventListener('click', e => {
  const t = e.target;
  if (t.dataset.play) {
    const ep = episodes.find(x => x.id === t.dataset.play);
    if (ep) playEp(ep);
  }
  if (t.dataset.del) deleteEp(t.dataset.del);
});

$('fileInput').addEventListener('change', e => {
  if (e.target.files.length) handleFiles(e.target.files);
  e.target.value = '';
});
$('bLoad').onclick = () => $('fileInput').click();
$('bMore').onclick = () => $('fileInput').click();
$('bPlay').onclick = togglePause;
$('bNext').onclick = playNext;
$('bShuffle').onclick = shuffleEpisodes;
$('bAuto').onclick = () => { autoPlay = !autoPlay; render(); };
$('bSkipIntro').onclick = () => {
  skipIntro = !skipIntro;
  localStorage.setItem('skipIntro', skipIntro ? '1' : '0');
  render();
};
$('bSleep').onclick = toggleSleep;
// Live countdown on the sleep button while armed (also fires the pause when
// the deadline passes with playback stopped or the tab in the background).
setInterval(() => {
  if (sleepUntil === null) return;
  if (Date.now() >= sleepUntil && playing) { togglePause(); }
  const b = $('bSleep');
  b.textContent = '⏾ ' + fmt(Math.max(0, (sleepUntil - Date.now()) / 1000));
  if (Date.now() >= sleepUntil && !playing) { sleepUntil = null; render(); }
}, 1000);

// ===================== HOSTED EPISODES =====================
// If the site ships pre-cut episodes (episodes/index.json), load them so the
// player is ready immediately — no file upload or analysis needed.
async function loadHostedEpisodes() {
  let idx;
  try {
    const resp = await fetch('episodes/index.json', { cache: 'no-cache' });
    if (!resp.ok) { hostedChecked = true; render(); return; }
    idx = await resp.json();
  } catch (e) {
    hostedChecked = true; render(); return; // no hosted episodes — upload mode
  }
  let deleted = [];
  try { deleted = JSON.parse(localStorage.getItem('deletedEps')) || []; } catch (e) {}
  const cid = 'hosted';
  compilations.push({ id: cid, name: idx.compilation, totalDuration: 0 });
  idx.episodes.forEach((e, i) => {
    if (deleted.includes(e.file)) return;
    episodes.push({
      id: cid + '_' + e.file, compilationId: cid, compilationName: idx.compilation,
      index: i, startSec: 0, endSec: e.duration, duration: e.duration,
      // ?v= busts browser caches of stale audio (v=4: ep07.mp3 was truncated
      // to 0 bytes by the v14 retag — an appended empty episode stalled the
      // stream at its buffer edge, "stops after 2-3 episodes" on locked phones)
      label: e.label, src: 'episodes/' + e.file + '?v=4', file: e.file,
    });
  });
  hostedChecked = true;
  render();
}

// ===================== SERVER-SIDE DELETE =====================
// Deleting a hosted episode commits the removal to the GitHub repo (episode
// file + its index.json entry), so it is gone for every device once Pages
// redeploys. Needs a fine-grained personal access token (Contents:
// read/write on this repo), asked once and kept in localStorage.
const GH_REPO = 'PADreyfus/kaamelott-splicer';

function ghToken() {
  let t = localStorage.getItem('ghToken');
  if (!t) {
    t = prompt(
      'Pour supprimer définitivement (sur le serveur), colle un token GitHub\n' +
      '(github.com → Settings → Developer settings → Fine-grained tokens,\n' +
      'repo kaamelott-splicer, permission "Contents: Read and write").\n\n' +
      'Annuler = suppression locale seulement (cet appareil).'
    );
    if (t) localStorage.setItem('ghToken', t.trim());
  }
  return t ? t.trim() : null;
}

async function ghApi(method, path, body) {
  const resp = await fetch(`https://api.github.com/repos/${GH_REPO}/contents/${path}`, {
    method,
    headers: {
      Authorization: 'Bearer ' + localStorage.getItem('ghToken'),
      Accept: 'application/vnd.github+json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (resp.status === 401 || resp.status === 403) {
    localStorage.removeItem('ghToken'); // bad token — re-ask next time
    throw new Error('Token GitHub invalide ou expiré');
  }
  if (!resp.ok && resp.status !== 404) throw new Error('GitHub API: ' + resp.status);
  return resp.status === 404 ? null : resp.json();
}

async function serverDeleteEpisode(file, label) {
  const token = ghToken();
  if (!token) return false;
  // Remove the entry from index.json
  const idxMeta = await ghApi('GET', 'episodes/index.json');
  if (idxMeta) {
    const idx = JSON.parse(atob(idxMeta.content.replace(/\n/g, '')));
    const before = idx.episodes.length;
    idx.episodes = idx.episodes.filter(e => e.file !== file);
    if (idx.episodes.length !== before) {
      await ghApi('PUT', 'episodes/index.json', {
        message: `Delete ${label} (${file})`,
        content: btoa(JSON.stringify(idx, null, 1)),
        sha: idxMeta.sha,
      });
    }
  }
  // Delete the episode file itself
  const fileMeta = await ghApi('GET', 'episodes/' + file);
  if (fileMeta) {
    await ghApi('DELETE', 'episodes/' + file, {
      message: `Delete ${label} audio (${file})`,
      sha: fileMeta.sha,
    });
  }
  return true;
}

// ===================== PWA INSTALL =====================
// The service worker + manifest make the site installable on Android. The
// installed app is the same page in a standalone window; playback while the
// phone is locked is unchanged (that's handled by the MediaSource stream).
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}

// Chrome fires beforeinstallprompt only while the app is NOT installed and
// installability criteria are met — so the button shows exactly when an
// install is possible, and never inside the installed app itself.
let installPrompt = null;
window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  installPrompt = e;
  $('bInstall').style.display = '';
});
$('bInstall').onclick = async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice.catch(() => {});
  installPrompt = null;
  $('bInstall').style.display = 'none';
};
window.addEventListener('appinstalled', () => {
  installPrompt = null;
  $('bInstall').style.display = 'none';
});

$('appVersion').textContent = APP_VERSION;

render();
loadHostedEpisodes();
