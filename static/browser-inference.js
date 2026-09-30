/* Standalone Pages transport: no prediction API or image upload. */
(() => {
  const sessions = new Map();
  let engine;
  const base = new URL('./', document.baseURI);
  const manifest = fetch(new URL('models.json', base), { cache: 'no-cache' }).then(response => {
    if (!response.ok) throw new Error('模型設定載入失敗，請重新整理。');
    return response.json();
  });
  ort.env.wasm.wasmPaths = new URL('static/vendor/ort/', base).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = true;

  async function modelBytes(info, signal, progress) {
    const url = new URL(info.file, base);
    url.searchParams.set('v', info.sha256);
    let cache;
    try { cache = await caches.open('cofe-log-models-v1'); } catch (_) { /* Cache is optional. */ }
    const cached = await cache?.match(url);
    if (cached) { progress('載入模型…'); return cached.arrayBuffer(); }
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error('模型下載失敗，請檢查網路後再試。');
    const reader = response.body.getReader();
    const chunks = []; let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); received += value.length;
      progress(`下載模型 ${Math.min(100, Math.round(received / info.bytes * 100))}%`);
    }
    const buffer = new Uint8Array(received); let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    if (cache) {
      try { await cache.put(url, new Response(buffer, { headers: { 'Content-Type': 'application/octet-stream' } })); }
      catch (_) { /* Storage limits must not prevent inference. */ }
    }
    return buffer;
  }

  async function sessionFor(info, signal, progress) {
    if (!sessions.has(info.key)) {
      const entry = { listeners: new Set(), message: '準備模型…' };
      const report = message => {
        entry.message = message;
        entry.listeners.forEach(listener => listener(message));
      };
      // Background loading and inference share one download. An individual
      // image request must not cancel a model that another request is using.
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 600000);
      entry.promise = (async () => {
        const model = modelBytes(info, controller.signal, report);
        let runtime = Promise.resolve();
        if (typeof DecompressionStream !== 'undefined') {
          if (!engine) {
            engine = (async () => {
              const compressed = await modelBytes(info.runtime, controller.signal,
                message => report(message.replace('模型', '辨識功能')));
              const stream = new Response(compressed).body.pipeThrough(new DecompressionStream('gzip'));
              ort.env.wasm.wasmBinary = await new Response(stream).arrayBuffer();
            })();
            engine.catch(() => { engine = undefined; });
          }
          runtime = engine;
        }
        const [bytes] = await Promise.all([model, runtime]);
        report('載入模型…');
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
        report('模型已就緒');
        return session;
      })();
      sessions.set(info.key, entry);
      entry.promise.then(() => clearTimeout(timeout), () => {
        clearTimeout(timeout); controller.abort(); sessions.delete(info.key);
      });
    }
    const entry = sessions.get(info.key);
    entry.listeners.add(progress);
    progress(entry.message);
    try { return await entry.promise; }
    finally { entry.listeners.delete(progress); }
  }

  // Pillow-compatible separable bilinear resize with antialiasing and
  // 22-bit coefficients. This preserves the original training preprocessing.
  function coefficients(sourceSize, targetSize) {
    const scale = sourceSize / targetSize;
    const support = Math.max(1, scale);
    return Array.from({ length: targetSize }, (_, index) => {
      const center = (index + .5) * scale;
      const first = Math.max(0, Math.trunc(center - support + .5));
      const last = Math.min(sourceSize, Math.trunc(center + support + .5));
      const weights = []; let total = 0;
      for (let x = first; x < last; x++) {
        const weight = Math.max(0, 1 - Math.abs((x + .5 - center) / support));
        weights.push(weight); total += weight;
      }
      return { first, weights: weights.map(value => Math.round(value / total * 4194304)) };
    });
  }

  async function prepareImage(file, size) {
    const bitmap = await createImageBitmap(file);
    try {
      const { width, height } = bitmap;
      const side = Math.max(width, height);
      if (!width || !height || width * height > 20000000 || side > 8192) {
        throw new Error('照片尺寸過大，請縮小至 8192 像素以內、總像素不超過 2000 萬。');
      }
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.fillStyle = '#000'; context.fillRect(0, 0, width, height); context.drawImage(bitmap, 0, 0);
      const pixels = context.getImageData(0, 0, width, height).data;
      const left = Math.floor((side - width) / 2), top = Math.floor((side - height) / 2);
      const weights = coefficients(side, size);
      const horizontal = new Uint8Array(size * side * 3);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < size; x++) {
          const { first, weights: values } = weights[x];
          for (let channel = 0; channel < 3; channel++) {
            let sum = 2097152;
            for (let i = 0; i < values.length; i++) {
              const sourceX = first + i - left;
              if (sourceX >= 0 && sourceX < width) sum += pixels[(y * width + sourceX) * 4 + channel] * values[i];
            }
            horizontal[((y + top) * size + x) * 3 + channel] = Math.min(255, Math.max(0, Math.floor(sum / 4194304)));
          }
        }
      }
      const data = new Float32Array(3 * size * size);
      for (let y = 0; y < size; y++) {
        const { first, weights: values } = weights[y];
        for (let x = 0; x < size; x++) {
          for (let channel = 0; channel < 3; channel++) {
            let sum = 2097152;
            for (let i = 0; i < values.length; i++) sum += horizontal[((first + i) * size + x) * 3 + channel] * values[i];
            const value = Math.min(255, Math.max(0, Math.floor(sum / 4194304)));
            data[channel * size * size + y * size + x] = (value / 255 - .5) / .5;
          }
        }
      }
      return data;
    } finally { bitmap.close(); }
  }

  window.browserInference = {
    prepareImage,
    async preload(modelKey, progress = () => {}) {
      const info = (await manifest).find(model => model.key === modelKey);
      if (!info) throw new Error('找不到所選模型。');
      return sessionFor(info, undefined, progress);
    },
    async predict(file, modelKey, signal, progress) {
      const info = (await manifest).find(model => model.key === modelKey);
      if (!info) throw new Error('找不到所選模型。');
      const session = await sessionFor(info, signal, progress);
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      progress('辨識中…');
      const image = new ort.Tensor('float32', await prepareImage(file, info.img_size), [1, 3, info.img_size, info.img_size]);
      const output = await session.run({ image });
      const logits = Array.from(output.logits.data);
      image.dispose(); output.logits.dispose();
      const exp = logits.map(value => Math.exp(value - Math.max(...logits)));
      const sum = exp.reduce((a, b) => a + b, 0);
      const probabilities = info.classes.map((label, index) => ({ label, probability: exp[index] / sum }));
      return { prediction: probabilities.reduce((best, item) => item.probability > best.probability ? item : best), probabilities };
    }
  };
})();
