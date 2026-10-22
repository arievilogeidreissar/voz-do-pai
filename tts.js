// Voz natural masculina (Piper "Faber", pt-BR) rodando no próprio celular.
// O modelo é baixado uma vez e guardado no aparelho (OPFS).
import { createPiperPhonemize } from './piper-phonemize.js';

const MODEL_URL = 'https://huggingface.co/diffusionstudio/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx';
const PHON_URL = 'https://cdn.jsdelivr.net/npm/@diffusionstudio/piper-wasm@1.0.0/build/piper_phonemize';
const ORT_WASM = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.18.0/';
const FILES = { model: 'faber.onnx', config: 'faber.json' };

const ort = window.ort;
ort.env.wasm.wasmPaths = ORT_WASM;
ort.env.wasm.numThreads = 1;

// Guarda a voz no armazenamento do navegador (Cache Storage: Chrome, Android e iPhone)
const CACHE = 'voz-do-pai-v1';
const key = (name) => new Request('./_voz/' + name);
async function readFile(name) {
  try { const r = await (await caches.open(CACHE)).match(key(name)); return r ? await r.blob() : null; } catch (e) { return null; }
}
async function writeFile(name, blob) {
  await (await caches.open(CACHE)).put(key(name), new Response(blob));
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Falha no download (' + res.status + ')');
  const total = +(res.headers.get('Content-Length') || 0) || 63_000_000;
  const reader = res.body.getReader(); const parts = []; let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    parts.push(value); loaded += value.length; onProgress && onProgress(Math.min(1, loaded / total));
  }
  return new Blob(parts);
}

export function supported() {
  return !!(window.caches && window.ort);
}

export async function isDownloaded() {
  const [m, c] = await Promise.all([readFile(FILES.model), readFile(FILES.config)]);
  return !!(m && c && m.size > 1_000_000);
}

export async function download(onProgress) {
  try { await navigator.storage.persist(); } catch (e) {}
  const config = await fetchWithProgress(MODEL_URL + '.json');
  const model = await fetchWithProgress(MODEL_URL, onProgress);
  await writeFile(FILES.config, config);
  await writeFile(FILES.model, model);
}

let session = null, config = null;
async function load() {
  if (session) return;
  config = JSON.parse(await (await readFile(FILES.config)).text());
  const buf = await (await readFile(FILES.model)).arrayBuffer();
  session = await ort.InferenceSession.create(buf);
}

function phonemize(text) {
  return new Promise(async (resolve, reject) => {
    let done = false;
    try {
      const mod = await createPiperPhonemize({
        print: (line) => { if (!done) { done = true; resolve(JSON.parse(line).phoneme_ids); } },
        printErr: (line) => console.warn('[phonemize]', line),
        locateFile: (f) => f.endsWith('.wasm') ? PHON_URL + '.wasm' : f.endsWith('.data') ? PHON_URL + '.data' : f,
      });
      mod.callMain(['-l', config.espeak.voice, '--input', JSON.stringify([{ text }]), '--espeak_data', '/espeak-ng-data']);
      setTimeout(() => { if (!done) reject(new Error('Não consegui processar o texto.')); }, 20000);
    } catch (e) { reject(e); }
  });
}

// speed: 1 = normal; >1 mais rápido
export async function synth(text, speed = 1) {
  await load();
  const ids = await phonemize(text.replace(/\s+/g, ' ').trim());
  const inf = config.inference;
  const inputs = {
    input: new ort.Tensor('int64', ids, [1, ids.length]),
    input_lengths: new ort.Tensor('int64', [ids.length]),
    scales: new ort.Tensor('float32', [inf.noise_scale, inf.length_scale / speed, inf.noise_w]),
  };
  if (Object.keys(config.speaker_id_map || {}).length) inputs.sid = new ort.Tensor('int64', [0]);
  const { output } = await session.run(inputs);
  return { samples: output.data, sampleRate: config.audio.sample_rate };
}

function toInt16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

export function toWav(samples, sampleRate) {
  const pcm = toInt16(samples), v = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  pcm.forEach((s, i) => v.setInt16(44 + i * 2, s, true));
  return new Blob([v], { type: 'audio/wav' });
}

export function toMp3(samples, sampleRate) {
  const pcm = toInt16(samples), enc = new window.lamejs.Mp3Encoder(1, sampleRate, 64), parts = [];
  for (let i = 0; i < pcm.length; i += 1152) {
    const b = enc.encodeBuffer(pcm.subarray(i, i + 1152)); if (b.length) parts.push(new Uint8Array(b));
  }
  const end = enc.flush(); if (end.length) parts.push(new Uint8Array(end));
  return new Blob(parts, { type: 'audio/mpeg' });
}
