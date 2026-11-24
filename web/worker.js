// Runs the watermark code off the main thread so the page stays responsive.
let wasm, ready = (async () => {
  const res = await fetch('tracemark.wasm');
  wasm = (await WebAssembly.instantiate(await res.arrayBuffer(), {})).instance.exports;
})();

const mem = () => new Uint8Array(wasm.memory.buffer);
function put(bytes) {
  const ptr = wasm.tm_alloc(bytes.length);
  mem().set(bytes, ptr);
  return ptr;
}
const enc = new TextEncoder();
let original = null; // { rgba, w, h } kept between embed calls for the same picture

onmessage = async ({ data: m }) => {
  try {
    await ready;
    if (m.type === 'load') {
      original = { rgba: new Uint8Array(m.rgba), w: m.w, h: m.h };
      postMessage({ id: m.id });
    } else if (m.type === 'embed') {
      const { rgba, w, h } = original;
      const key = enc.encode(m.key);
      const p = put(rgba), k = put(key);
      wasm.tm_embed(p, w, h, m.recipient, k, key.length, m.strength);
      const out = mem().slice(p, p + rgba.length);
      wasm.tm_free(p, rgba.length); wasm.tm_free(k, key.length);
      postMessage({ id: m.id, rgba: out.buffer }, [out.buffer]);
    } else if (m.type === 'detect') {
      const o = new Uint8Array(m.orig), s = new Uint8Array(m.suspect);
      const key = enc.encode(m.key), ids = new Uint8Array(new Uint32Array(m.ids).buffer);
      const po = put(o), ps = put(s), k = put(key), pi = put(ids), out = wasm.tm_alloc(64);
      wasm.tm_detect(po, m.ow, m.oh, ps, m.sw, m.sh, k, key.length, pi, m.ids.length, out);
      const r = new Float64Array(wasm.memory.buffer.slice(out, out + 64));
      wasm.tm_free(po, o.length); wasm.tm_free(ps, s.length); wasm.tm_free(k, key.length); wasm.tm_free(pi, ids.length); wasm.tm_free(out, 64);
      postMessage({ id: m.id, result: {
        verdict: ['none', 'weak', 'match'][r[0]], recipient: r[1] < 0 ? null : r[1], repaired: r[2],
        strength: r[3], coverage: r[4], alignment: r[5], mirrored: !!r[6], readDirectly: !!r[7] } });
    }
  } catch (e) {
    postMessage({ id: m.id, error: String(e && e.message || e) });
  }
};
