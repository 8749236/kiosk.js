const $ = (s) => document.querySelector(s);
const worker = new Worker('worker.js');
let seq = 0;
const pending = new Map();
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id); pending.delete(data.id);
  data.error ? p.reject(new Error(data.error)) : p.resolve(data);
};
worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || '后台线程出错')); pending.clear(); };
const call = (msg, transfer = []) => new Promise((resolve, reject) => {
  const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }, transfer);
});

// ---- small helpers
function recipientId(key, name) { // stable 32-bit id from key + name (FNV-1a)
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(key + '\n' + name)) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
const names = () => [...new Set($('#names').value.split('\n').map((s) => s.trim()).filter(Boolean))];

async function pixels(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }); }
  catch { throw new Error(`无法读取「${file.name}」，请确认它是图片`); }
  const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true }); ctx.drawImage(bmp, 0, 0); bmp.close();
  return ctx.getImageData(0, 0, c.width, c.height);
}
function toPng(rgba, w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  return new Promise((ok) => c.toBlob(ok, 'image/png'));
}

// ---- zip (stored, no compression: PNGs are already compressed)
const crcTable = Array.from({ length: 256 }, (_, n) => { for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0; });
function crc32(bytes) { let c = ~0; for (const b of bytes) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8); return ~c >>> 0; }
class Zip {
  constructor() { this.parts = []; this.central = []; this.offset = 0; }
  async add(name, blob) {
    const data = new Uint8Array(await blob.arrayBuffer()), nameBytes = new TextEncoder().encode(name), crc = crc32(data);
    const head = (sig, extra) => {
      const b = new DataView(new ArrayBuffer(extra ? 46 : 30)); let o = 0;
      const u16 = (v) => { b.setUint16(o, v, true); o += 2; }, u32 = (v) => { b.setUint32(o, v, true); o += 4; };
      u32(sig); if (extra) u16(20); u16(20); u16(0x0800); u16(0); u16(0); u16(0x21); u32(crc); u32(data.length); u32(data.length); u16(nameBytes.length); u16(0);
      if (extra) { u16(0); u16(0); u16(0); u32(0); u32(this.offset); }
      return b.buffer;
    };
    this.central.push(head(0x02014b50, true), nameBytes);
    this.parts.push(head(0x04034b50, false), nameBytes, blob);
    this.offset += 30 + nameBytes.length + data.length;
  }
  finish() {
    const size = this.central.reduce((n, p) => n + p.byteLength, 0), count = this.central.length / 2;
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, count, true); end.setUint16(10, count, true); end.setUint32(12, size, true); end.setUint32(16, this.offset, true);
    return new Blob([...this.parts, ...this.central, end.buffer], { type: 'application/zip' });
  }
}

// ---- remember key and names on this device only
try {
  $('#key').value = localStorage.getItem('tm.key') || '';
  $('#names').value = localStorage.getItem('tm.names') || '';
} catch {}
for (const id of ['key', 'names']) $('#' + id).addEventListener('input', () => { try { localStorage.setItem('tm.' + id, $('#' + id).value); } catch {} updateCount(); });
function updateCount() { const n = names().length; $('#count').textContent = n ? `${n} 人` : ''; }
updateCount();

// ---- tabs
for (const tab of document.querySelectorAll('[role=tab]')) tab.addEventListener('click', () => {
  for (const t of document.querySelectorAll('[role=tab]')) t.setAttribute('aria-selected', t === tab);
  for (const p of document.querySelectorAll('[role=tabpanel]')) p.hidden = p.id !== tab.getAttribute('aria-controls');
});

// ---- file pickers with drag and drop
for (const zone of document.querySelectorAll('.drop')) {
  const input = zone.querySelector('input'), label = zone.querySelector('.picked');
  const show = () => { label.textContent = [...input.files].map((f) => f.name).join('、') || ''; };
  input.addEventListener('change', show);
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault(); zone.classList.remove('over');
    const dt = new DataTransfer();
    for (const f of e.dataTransfer.files) if (input.multiple || !dt.files.length) dt.items.add(f);
    input.files = dt.files; show();
  });
}

function status(el, text, kind = '') { el.textContent = text; el.className = 'status ' + kind; }

// ---- embed
$('#embed-go').addEventListener('click', async () => {
  const out = $('#embed-status'), files = [...$('#embed-files').files], people = names(), key = $('#key').value;
  if (!files.length) return status(out, '请先选择要发的图', 'bad');
  if (!people.length) return status(out, '请填写至少一个支援者', 'bad');
  $('#embed-go').disabled = true;
  try {
    const zip = new Zip(), total = files.length * people.length; let done = 0;
    for (const file of files) {
      const img = await pixels(file), base = file.name.replace(/\.[^.]+$/, '');
      await call({ type: 'load', rgba: img.data.buffer, w: img.width, h: img.height }, [img.data.buffer]);
      for (const name of people) {
        status(out, `正在处理 ${++done} / ${total}…`);
        const r = await call({ type: 'embed', recipient: recipientId(key, name), key, strength: 1 });
        await zip.add(`${name.replace(/[\\/:*?"<>|]/g, '_')}/${base}.png`, await toPng(r.rgba, img.width, img.height));
      }
    }
    const csv = '﻿编号,支援者\n' + people.map((n) => `${recipientId(key, n)},"${n.replace(/"/g, '""')}"`).join('\n') + '\n';
    await zip.add('名单.csv', new Blob([csv]));
    const a = document.createElement('a'); a.href = URL.createObjectURL(zip.finish()); a.download = '带编号的图.zip'; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    status(out, `完成：${files.length} 张图 × ${people.length} 人，已打包下载。每人一个文件夹。`, 'good');
  } catch (e) { status(out, e.message, 'bad'); }
  $('#embed-go').disabled = false;
});

// ---- detect
$('#detect-go').addEventListener('click', async () => {
  const out = $('#detect-status'), o = $('#detect-original').files[0], s = $('#detect-suspect').files[0], key = $('#key').value;
  if (!o || !s) return status(out, '请选择原图和可疑的图', 'bad');
  $('#detect-go').disabled = true; status(out, '正在比对…');
  try {
    const [a, b] = [await pixels(o), await pixels(s)], people = names();
    const byId = new Map(people.map((n) => [recipientId(key, n), n]));
    const { result: r } = await call({ type: 'detect', orig: a.data.buffer, ow: a.width, oh: a.height, suspect: b.data.buffer, sw: b.width, sh: b.height, key, ids: [...byId.keys()] }, [a.data.buffer, b.data.buffer]);
    const who = r.recipient === null ? '' : byId.has(r.recipient) ? `「${byId.get(r.recipient)}」` : ` 编号 ${r.recipient}（不在当前名单里）`;
    const detail = `强度 ${r.strength.toFixed(1)}，图中保留了原图的 ${Math.round(r.coverage * 100)}%，对齐程度 ${r.alignment.toFixed(2)}${r.mirrored ? '，被镜像翻转过' : ''}`;
    if (r.verdict === 'match') status(out, `这是发给${who}的那一份。\n${detail}`, 'good');
    else if (r.verdict === 'weak') status(out, `有微弱痕迹指向${who}，但不足以确定。\n${detail}`, 'warn');
    else status(out, `没有找到水印。可能这张图不是从你发出的带编号版本流出的，或者密钥不对、原图选错了、图被改动得太厉害。\n${detail}`, 'bad');
  } catch (e) { status(out, e.message, 'bad'); }
  $('#detect-go').disabled = false;
});
