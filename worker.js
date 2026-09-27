/* Pyodide host.  Runs pipeline.py -- the same file you can import locally -- inside a
 * worker so the UI thread never blocks on OpenCV.  That separation is what
 * keeps a slider drag at 60fps: even a 40ms run cannot stutter the thumb.
 *
 * Protocol
 *   in   {cmd:'init',   image}
 *        {cmd:'source', image}
 *        {cmd:'run',    id, ops, selected, all}
 *        {cmd:'export', id, ops}
 *   out  {type:'status', text}
 *        {type:'ready',  schema, source}
 *        {type:'source', source}
 *        {type:'result', id, meta, preview, thumbs}
 *        {type:'png',    id, data}
 *        {type:'error',  id, message}
 */

"use strict";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.0/full/";
const HOME = "/home/pyodide";

importScripts(PYODIDE + "pyodide.js");

let pyodide = null;
let bridge = null;

const status = (text) => postMessage({ type: "status", text });

/* Copy a numpy array out of the wasm heap into a transferable ArrayBuffer.
 * getBuffer() is a view, not a copy, and it is only valid until release() --
 * and the heap may be reallocated by the next Python call, so we copy now. */
function takeBuffer(proxy) {
  const view = proxy.getBuffer("u8");
  try {
    return new Uint8Array(view.data).buffer;
  } finally {
    view.release();
    proxy.destroy();
  }
}

async function copyToFS(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const name = path.split("/").pop();
  pyodide.FS.writeFile(`${HOME}/${name}`, new Uint8Array(await res.arrayBuffer()));
  return name;
}

/* `image` is a path under images/ relative to the page, e.g.
 * images/bear__00000.jpg.  Only images/ is ever fetched -- never masks/ or
 * previews/, which hold the ground truth. */
async function loadSource(image) {
  if (!/^images\/[\w.-]+$/.test(image)) throw new Error(`not an image path: ${image}`);
  const name = await copyToFS(image);
  return bridge.set_source(`${HOME}/${name}`, image);
}

async function init({ image }) {
  status("loading Python…");
  pyodide = await loadPyodide({ indexURL: PYODIDE });

  // Only what pipeline.py imports.  scikit-image drags scipy along with it,
  // so the first load is a sizeable download; the browser caches it after.
  status("loading numpy, opencv, scikit-image (first load is slow)…");
  await pyodide.loadPackage(["numpy", "opencv-python", "scikit-image"]);

  status("loading pipeline…");
  await Promise.all([copyToFS("pipeline.py"), copyToFS("bridge.py")]);

  pyodide.runPython(`
import sys
if ${JSON.stringify(HOME)} not in sys.path:
    sys.path.insert(0, ${JSON.stringify(HOME)})
`);
  bridge = pyodide.pyimport("bridge");

  status("decoding source…");
  const source = await loadSource(image);

  postMessage({ type: "ready", schema: bridge.schema(), source });
}

function handleRun({ id, ops, selected, all }) {
  const started = performance.now();
  const meta = JSON.parse(bridge.run(JSON.stringify(ops), selected, !!all));

  const [pw, ph] = JSON.parse(bridge.preview_shape());
  const preview = { width: pw, height: ph, data: takeBuffer(bridge.preview_array()) };

  const thumbs = meta.fresh_thumbs.map((index) => {
    const [w, h] = JSON.parse(bridge.thumb_shape(index));
    return { index, width: w, height: h, data: takeBuffer(bridge.thumb_array(index)) };
  });

  meta.total_ms = Math.round((performance.now() - started) * 10) / 10;
  postMessage({ type: "result", id, meta, preview, thumbs },
              [preview.data, ...thumbs.map((t) => t.data)]);
}

function handleExport({ id, ops }) {
  const data = takeBuffer(bridge.export_png(JSON.stringify(ops)));
  postMessage({ type: "png", id, data }, [data]);
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.cmd === "init") await init(msg);
    else if (msg.cmd === "source") postMessage({ type: "source", source: await loadSource(msg.image) });
    else if (msg.cmd === "run") handleRun(msg);
    else if (msg.cmd === "export") handleExport(msg);
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String(err && err.message || err) });
  }
};
