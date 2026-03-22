/**
 * analyzer.mjs — pure algorithm extracted from app.js, with no DOM/browser globals.
 * All functions are injectable: AudioContext is passed as a parameter or factory.
 */

export const FFT_SIZE = 1024;
export const N_BANDS = 24;
export const NCC_THRESHOLD = 0.45;

// ===================== SPECTRUM / FFT UTILITIES =====================
// Cooley-Tukey radix-2 in-place FFT. re/im must be Float32Array of length 2^n.
export function fft(re, im) {
  const n = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // butterfly stages
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

// Compute block-level log-magnitude spectral features from PCM.
// Returns an array of Float32Array(N_BANDS) — one per non-overlapping block.
export function blockFeatures(pcm, sr) {
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
      for (let k = lo; k <= hi; k++) sum += Math.sqrt(re[k]*re[k] + im[k]*im[k]);
      const cnt = hi - lo + 1;
      bands[band] = Math.log1p(sum / cnt);
    }
    feats.push(bands);
  }
  return feats;
}

// ===================== JINGLE FINGERPRINT =====================

/**
 * Build a jingle fingerprint from a raw ArrayBuffer (WAV or decodable audio).
 * @param {ArrayBuffer} jingleAB
 * @param {AudioContext} audioCtx - pre-created AudioContext instance
 * @returns {{ features: Float32Array[], blockCount: number, blockSec: number, norm: number }}
 */
export async function buildJingleFp(jingleAB, audioCtx) {
  const buf = await audioCtx.decodeAudioData(jingleAB.slice(0));
  const pcm = buf.getChannelData(0);
  const features = blockFeatures(pcm, buf.sampleRate);
  if (features.length < 2) throw new Error('Jingle too short');
  let norm2 = 0;
  for (const f of features) for (let i = 0; i < N_BANDS; i++) norm2 += f[i] * f[i];
  return {
    features,
    blockCount: features.length,
    blockSec: FFT_SIZE / buf.sampleRate,
    norm: Math.sqrt(norm2),
  };
}

// ===================== SPECTRAL MATCHING =====================

/**
 * Slide jingle fingerprint over audio PCM, return best NCC match.
 * @param {Float32Array} audioPcm
 * @param {number} audioSR
 * @param {{ features, blockCount, blockSec, norm }} jingleFp
 * @returns {{ timeSec: number, score: number } | null}
 */
export function spectralMatchInSlice(audioPcm, audioSR, jingleFp) {
  if (!jingleFp || jingleFp.blockCount < 2) return null;
  const audioFeats = blockFeatures(audioPcm, audioSR);
  const J = jingleFp.blockCount;
  const N = audioFeats.length;
  if (N < J) return null;
  const jFeats = jingleFp.features;
  const jNorm = jingleFp.norm;
  let bestScore = -1, bestOffset = -1;
  for (let o = 0; o <= N - J; o++) {
    let dot = 0;
    for (let t = 0; t < J; t++) {
      const jf = jFeats[t], af = audioFeats[o + t];
      for (let b = 0; b < N_BANDS; b++) dot += jf[b] * af[b];
    }
    let aNorm2 = 0;
    for (let t = 0; t < J; t++) {
      const af = audioFeats[o + t];
      for (let b = 0; b < N_BANDS; b++) aNorm2 += af[b] * af[b];
    }
    const score = dot / (jNorm * Math.sqrt(aNorm2) + 1e-9);
    if (score > bestScore) { bestScore = score; bestOffset = o; }
  }
  if (bestScore < NCC_THRESHOLD) return null;
  return { timeSec: bestOffset * jingleFp.blockSec, score: bestScore };
}

// ===================== MP3 SYNC =====================
export function findMp3Sync(bytes, off) {
  for (let i = Math.max(0, off); i < bytes.length - 3; i++) {
    if (bytes[i] !== 0xFF || (bytes[i+1] & 0xE0) !== 0xE0) continue;
    const v = (bytes[i+1] >> 3) & 3, l = (bytes[i+1] >> 1) & 3,
          br = (bytes[i+2] >> 4) & 0xF, sr = (bytes[i+2] >> 2) & 3;
    if (v !== 1 && l !== 0 && br > 0 && br < 15 && sr < 3) return i;
  }
  return -1;
}

// ===================== XING/VBR HEADER SUPPORT =====================

/**
 * Parse the Xing/Info VBR header from an MP3 file.
 * Returns { nFrames, fileBytes, toc, sr, spf, syncOff } or null.
 * syncOff is the byte offset of the first audio frame (Xing or regular).
 */
function parseXingHeader(bytes) {
  // Skip ID3v2 tag if present
  let start = 0;
  if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) {
    const bodySize = ((bytes[6] & 0x7F) << 21) | ((bytes[7] & 0x7F) << 14) |
                     ((bytes[8] & 0x7F) <<  7) |  (bytes[9] & 0x7F);
    start = 10 + bodySize;
  }

  const syncOff = findMp3Sync(bytes, start);
  if (syncOff < 0 || syncOff + 120 > bytes.length) return null;

  const b1 = bytes[syncOff + 1], b2 = bytes[syncOff + 2], b3 = bytes[syncOff + 3];
  const mpegVer = (b1 >> 3) & 3;  // 3=MPEG1, 2=MPEG2, 0=MPEG2.5, 1=reserved
  const layer   = (b1 >> 1) & 3;  // 1=Layer3 (MP3)
  if (layer !== 1 || mpegVer === 1) return null;

  const srIdx = (b2 >> 2) & 3;
  if (srIdx === 3) return null;
  const srTable = [[44100, 48000, 32000], [22050, 24000, 16000], [11025, 12000, 8000]];
  const sr  = srTable[mpegVer === 3 ? 0 : mpegVer === 2 ? 1 : 2][srIdx];
  const spf = mpegVer === 3 ? 1152 : 576;

  const isMono       = ((b3 >> 6) & 3) === 3;
  const sideInfoSize = mpegVer === 3 ? (isMono ? 17 : 32) : (isMono ? 9 : 17);

  const xOff = syncOff + 4 + sideInfoSize;
  if (xOff + 8 > bytes.length) return null;

  // Check for "Xing" or "Info" tag
  const t0 = bytes[xOff], t1 = bytes[xOff+1], t2 = bytes[xOff+2], t3 = bytes[xOff+3];
  const isXing = t0===0x58 && t1===0x69 && t2===0x6E && t3===0x67;
  const isInfo = t0===0x49 && t1===0x6E && t2===0x66 && t3===0x6F;
  if (!isXing && !isInfo) return null;

  const flags = (bytes[xOff+4] << 24) | (bytes[xOff+5] << 16) |
                (bytes[xOff+6] <<  8) |  bytes[xOff+7];

  let off = xOff + 8, nFrames = null, fileBytes = null, toc = null;
  if (flags & 0x01) {
    nFrames = (bytes[off] << 24) | (bytes[off+1] << 16) | (bytes[off+2] << 8) | bytes[off+3];
    off += 4;
  }
  if (flags & 0x02) {
    fileBytes = (bytes[off] << 24) | (bytes[off+1] << 16) | (bytes[off+2] << 8) | bytes[off+3];
    off += 4;
  }
  if ((flags & 0x04) && off + 100 <= bytes.length) {
    toc = bytes.slice(off, off + 100);
  }

  return { nFrames, fileBytes, toc, sr, spf, syncOff };
}

// ===================== FRAME-ACCURATE BYTE↔TIME INDEX =====================

/**
 * Walk every MP3 frame from startByte and build a compact lookup table.
 * Records (byteOffset, timeSec) every INDEX_INTERVAL frames so that
 * linear interpolation gives sub-frame accuracy for any seek position.
 *
 * Returns { byteTable: Int32Array, timeTable: Float64Array }.
 */
function buildFrameIndex(bytes, startByte) {
  // Bitrate tables in kbps: index 0 = MPEG1 Layer3, index 1 = MPEG2/2.5 Layer3
  const BITRATE_TABLE = [
    [0,32,40,48,56,64,80,96,112,128,160,192,224,256,320,0],
    [0, 8,16,24,32,40,48,56, 64, 80, 96,112,128,144,160,0],
  ];
  const SR_TABLE = [[44100,48000,32000],[22050,24000,16000],[11025,12000,8000]];
  const INDEX_INTERVAL = 100; // record every 100 frames (~2.6 s at 44.1 kHz)

  const byteArr = [startByte];
  const timeArr = [0];

  let off = startByte;
  let frameTime = 0;
  let frameCount = 0;

  while (off + 4 <= bytes.length) {
    if (bytes[off] !== 0xFF || (bytes[off + 1] & 0xE0) !== 0xE0) { off++; continue; }

    const b1 = bytes[off + 1], b2 = bytes[off + 2];
    const mpegVer = (b1 >> 3) & 3; // 3=MPEG1, 2=MPEG2, 0=MPEG2.5, 1=reserved
    const layer   = (b1 >> 1) & 3; // 1=Layer3
    const brIdx   = (b2 >> 4) & 0xF;
    const srIdx   = (b2 >> 2) & 3;
    const padding = (b2 >> 1) & 1;

    if (mpegVer === 1 || layer !== 1 || srIdx === 3 || brIdx === 0 || brIdx === 15) {
      off++; continue;
    }

    const mIdx = mpegVer === 3 ? 0 : mpegVer === 2 ? 1 : 2;
    const sr  = SR_TABLE[mIdx][srIdx];
    const spf = mpegVer === 3 ? 1152 : 576;
    const br  = BITRATE_TABLE[mpegVer === 3 ? 0 : 1][brIdx] * 1000; // bps

    const frameSize = Math.floor(144 * br / sr) + padding;
    if (frameSize < 24 || off + frameSize > bytes.length) break;

    off += frameSize;
    frameTime += spf / sr;
    frameCount++;

    if (frameCount % INDEX_INTERVAL === 0) {
      byteArr.push(off);
      timeArr.push(frameTime);
    }
  }

  // Always include a final sentinel so interpolation works at the end
  if (byteArr[byteArr.length - 1] !== off) {
    byteArr.push(off);
    timeArr.push(frameTime);
  }

  return {
    byteTable: new Int32Array(byteArr),
    timeTable: new Float64Array(timeArr),
  };
}

// ===================== MAIN ANALYSIS =====================

/**
 * Analyze an MP3 file buffer and detect episode split points.
 *
 * Correctly handles:
 *  - ID3v2 tags at the start of the file (excluded from bps computation)
 *  - VBR files with a Xing/Info header + TOC (used for accurate byte↔time mapping)
 *
 * @param {ArrayBuffer} fileAB - entire MP3 file
 * @param {number} totalDuration - total duration in seconds (from metadata, used as fallback)
 * @param {{ features, blockCount, blockSec, norm } | null} jingleFp - fingerprint or null
 * @param {() => AudioContext} createAudioContext - factory for AudioContext instances
 * @param {(progress: number) => void} onProgress - called with 0..1
 * @param {(status: string) => void} onStatus - called with human-readable status
 * @returns {Promise<{ splitPoints: number[], totalDuration: number }>}
 */
export async function analyzeBuffer(
  fileAB, totalDuration, jingleFp, createAudioContext,
  onProgress = () => {}, onStatus = () => {}
) {
  const bytes = new Uint8Array(fileAB);
  const totalBytes = bytes.length;

  // Parse Xing/Info VBR header (if present) for accurate timing.
  const xing = parseXingHeader(bytes);

  // Duration: prefer Xing frame-count-based duration (most accurate).
  const dur = (xing?.nFrames > 0 ? xing.nFrames * xing.spf / xing.sr : null) ?? totalDuration;

  // audioStartByte: byte offset of the first audio frame (after ID3 tags etc.)
  const audioStartByte = xing?.syncOff ?? findMp3Sync(bytes, 0) ?? 0;

  // Build a frame-accurate byte↔time index by walking every MP3 frame once.
  // This gives sub-frame precision (~26 ms at 44.1 kHz) regardless of VBR
  // variation, avoiding the systematic errors of a file-wide linear estimate.
  onStatus('Building frame index…');
  const { byteTable, timeTable } = buildFrameIndex(bytes, audioStartByte);

  /** Binary-search helper: largest index i where table[i] <= val. */
  function bisect(table, val) {
    let lo = 0, hi = table.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (table[mid] <= val) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  /** Convert file byte offset → audio time (frame-accurate). */
  function timeForByte(b) {
    const i = bisect(byteTable, b);
    if (i >= byteTable.length - 1) return timeTable[timeTable.length - 1];
    const frac = (b - byteTable[i]) / (byteTable[i + 1] - byteTable[i]);
    return timeTable[i] + frac * (timeTable[i + 1] - timeTable[i]);
  }

  /** Convert audio time → file byte offset (frame-accurate). */
  function byteForTime(t) {
    const i = bisect(timeTable, t);
    if (i >= timeTable.length - 1) return byteTable[byteTable.length - 1];
    const frac = (t - timeTable[i]) / (timeTable[i + 1] - timeTable[i]);
    return Math.round(byteTable[i] + frac * (byteTable[i + 1] - byteTable[i]));
  }

  const SLICE_SEC = 100;
  const ctx = createAudioContext();

  const jingles = [0];
  let lastJ = 0;
  let consecutiveFails = 0;

  for (let epIdx = 0; epIdx < Math.ceil(dur / 150); epIdx++) {
    const searchStart = lastJ + 155;
    if (searchStart >= dur - 10) break;
    const sliceStartSec = Math.max(0, searchStart - 5);
    const sliceStartByte = findMp3Sync(bytes, byteForTime(sliceStartSec));
    if (sliceStartByte < 0) break;
    const sliceEndByte = Math.min(totalBytes, byteForTime(sliceStartSec + SLICE_SEC) + 5000);
    const slice = fileAB.slice(sliceStartByte, sliceEndByte);
    onStatus('Jingle ' + (epIdx + 1) + ' — ' + sliceStartSec.toFixed(0) + 's');
    onProgress(searchStart / dur);
    try {
      const buf = await ctx.decodeAudioData(slice.slice(0));
      const pcm = buf.getChannelData(0);
      const actualSliceStart = timeForByte(sliceStartByte);
      let hitInSlice = -1;
      if (jingleFp) {
        const m = spectralMatchInSlice(pcm, ctx.sampleRate, jingleFp);
        if (m) { hitInSlice = m.timeSec; }
      }
      if (hitInSlice >= 0) {
        const absTime = actualSliceStart + hitInSlice;
        if (absTime > lastJ + 140 && absTime < lastJ + 260) {
          jingles.push(absTime);
          lastJ = absTime;
          consecutiveFails = 0;
          continue;
        }
      }
    } catch (e) { /* decode failed for this slice */ }
    consecutiveFails++;
    if (consecutiveFails >= 3) break;
    const fb = lastJ + 200;
    if (fb >= dur - 10) break;
    jingles.push(fb);
    lastJ = fb;
  }

  try { ctx.close(); } catch (e) {}

  const splits = [...jingles, dur];
  return { splitPoints: splits, totalDuration: dur };
}
