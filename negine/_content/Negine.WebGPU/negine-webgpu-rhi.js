// negine-webgpu-rhi.js — JS-часть RHI на WebGPU (Negine.WebGPU/Rhi).
//
// Объекты WebGPU живут здесь, C# видит их как целочисленные id. Создание — прямые вызовы (JSImport), команды кадра
// C# пишет в массив int'ов, submitFrame исполняет их одним вызовом.
//
// WebGPU требует знать при создании layout'а группы тип выборки текстуры (глубину и 32-битный float без
// float32-filterable нельзя фильтровать), а рефлексия шейдера этого не знает. Поэтому настоящий GPUBindGroupLayout
// строится по тому, что привязано в группу, а GPURenderPipeline/GPUComputePipeline — при отрисовке, под layout'ы
// привязанных групп (кэш по их ключам). Сэмплер такой текстуры подменяется своим nearest-вариантом.

const SAMPLER_SHIFT = 48, TEXTURE_SHIFT = 16;
const ALL_STAGES = 1 | 2 | 4; // VERTEX | FRAGMENT | COMPUTE

// Коды команд — синхронно с WgpuOp (WgpuInterop.cs)
const OP = {
    BEGIN_RENDER_PASS: 1, END_RENDER_PASS: 2, SET_PIPELINE: 3, SET_BIND_GROUP: 4, SET_VERTEX_BUFFER: 5,
    SET_INDEX_BUFFER: 6, SET_VIEWPORT: 7, SET_SCISSOR: 8, DRAW: 9, DRAW_INDEXED: 10, BEGIN_COMPUTE_PASS: 11,
    END_COMPUTE_PASS: 12, SET_COMPUTE_PIPELINE: 13, DISPATCH: 14, COPY_BUFFER: 15, COPY_TEXTURE: 16,
};

// id не переиспользуются: кэши построены по id
const objs = [null];
function add(o) { objs.push(o); return objs.length - 1; }

const reported = new Set();
function report(message) {
    if (reported.has(message) || reported.size > 64) return;
    reported.add(message);
    console.error('[WebGPU RHI] ' + message);
}

// ============================ Устройство и холст ============================

let gpuDevice = null, context = null, canvas = null, format = 'bgra8unorm', description = 'WebGPU';
let float32Filterable = false;
const device = () => gpuDevice;

/** Вызывается из main.js до запуска .NET: запрос адаптера и устройства асинхронный. */
export async function initialize(canvasElement) {
    if (!navigator.gpu)
        throw new Error('Браузер не поддерживает WebGPU');

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter)
        throw new Error('Не удалось получить WebGPU-адаптер');

    const requiredFeatures = [];
    if (adapter.features.has('float32-filterable')) {
        requiredFeatures.push('float32-filterable');
        float32Filterable = true;
    }

    gpuDevice = await adapter.requestDevice({ requiredFeatures });
    gpuDevice.lost.then(info => console.error('[WebGPU] устройство потеряно: ' + info.message));
    gpuDevice.addEventListener('uncapturederror', e => report(e.error.message));

    const info = adapter.info ?? {};
    description = `WebGPU | ${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim();

    canvas = canvasElement;
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: gpuDevice, format, alphaMode: 'opaque',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
}

export function canvasFormat() { return format; }
export function deviceDescription() { return description; }

export function resizeCanvas(width, height) {
    canvas.width = Math.max(1, width);
    canvas.height = Math.max(1, height);
}

// ============================ Буферы ============================

export function createBuffer(size, usage, label) {
    return add({ kind: 'buffer', gpu: device().createBuffer({ size: Math.max(16, (size + 3) & ~3), usage, label }) });
}

export function writeBuffer(id, offset, view) {
    const bytes = view.slice();
    // writeBuffer требует размер, кратный 4
    const padded = (bytes.byteLength & 3) ? padTo4(bytes) : bytes;
    device().queue.writeBuffer(objs[id].gpu, offset, padded);
}

function padTo4(bytes) {
    const out = new Uint8Array((bytes.byteLength + 3) & ~3);
    out.set(bytes);
    return out;
}

/** Освобождение — после отправки кадра: на объект могут ссылаться его команды. */
const pendingDestroy = [];
export function destroy(id) { pendingDestroy.push(id); }

function flushDestroy() {
    for (const id of pendingDestroy) {
        const o = objs[id];
        if (o?.kind === 'buffer' || o?.kind === 'texture') o.gpu.destroy();
        objs[id] = null;
    }
    pendingDestroy.length = 0;
}

// ============================ Текстуры ============================

export function createTexture(width, height, layers, mipLevels, samples, format, usage, dimension, label) {
    const texture = device().createTexture({
        size: { width, height, depthOrArrayLayers: layers }, format, usage, mipLevelCount: mipLevels, sampleCount: samples, label,
    });
    return add({ kind: 'texture', gpu: texture, format, width, height, layers, mipLevels, samples, dimension });
}

export function createView(textureId, dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount) {
    const t = objs[textureId];
    return add({
        kind: 'view', texture: t, dimension,
        gpu: t.gpu.createView({ dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount }),
    });
}

export function writeTexture(id, mip, layer, width, height, bytesPerRow, view) {
    device().queue.writeTexture(
        { texture: objs[id].gpu, mipLevel: mip, origin: { x: 0, y: 0, z: layer } },
        view.slice(), { bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: 1 });
}

function isUnfilterable(format) {
    return format.startsWith('depth') || (!float32Filterable && format.endsWith('32float'));
}

// Мипмапы: полноэкранный треугольник из уровня i-1 в уровень i (в WebGPU своей генерации нет)
const mipPipelines = new Map();
let mipModule = null, mipSampler = null;
export function generateMips(id) {
    const t = objs[id];
    if (t.mipLevels <= 1) return;
    const dev = device();
    mipModule ??= dev.createShaderModule({ code: `
        struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
        @vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
            let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
            return Out(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
        }
        @group(0) @binding(0) var src: texture_2d<f32>;
        @group(0) @binding(1) var smp: sampler;
        @fragment fn fs(o: Out) -> @location(0) vec4f { return textureSampleLevel(src, smp, o.uv, 0.0); }` });
    mipSampler ??= dev.createSampler({ minFilter: 'linear', magFilter: 'linear' });

    let pipeline = mipPipelines.get(t.format);
    if (!pipeline) {
        pipeline = dev.createRenderPipeline({
            layout: 'auto',
            vertex: { module: mipModule, entryPoint: 'vs' },
            fragment: { module: mipModule, entryPoint: 'fs', targets: [{ format: t.format }] },
        });
        mipPipelines.set(t.format, pipeline);
    }

    const encoder = dev.createCommandEncoder();
    for (let layer = 0; layer < t.layers; layer++) {
        const viewOf = level => t.gpu.createView({ dimension: '2d', baseMipLevel: level, mipLevelCount: 1, baseArrayLayer: layer, arrayLayerCount: 1 });
        for (let mip = 1; mip < t.mipLevels; mip++) {
            const pass = encoder.beginRenderPass({ colorAttachments: [{ view: viewOf(mip), loadOp: 'clear', storeOp: 'store' }] });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, dev.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: viewOf(mip - 1) }, { binding: 1, resource: mipSampler }],
            }));
            pass.draw(3);
            pass.end();
        }
    }
    dev.queue.submit([encoder.finish()]);
}

// ============================ Сэмплеры и шейдеры ============================

export function createSampler(json) {
    const desc = JSON.parse(json);
    const dev = device();
    // nearest-вариант — для текстур, которые нельзя фильтровать (layout группы требует non-filtering)
    const nearest = { ...desc, minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest', maxAnisotropy: 1 };
    if (desc.maxAnisotropy > 1 && (desc.minFilter !== 'linear' || desc.magFilter !== 'linear' || desc.mipmapFilter !== 'linear'))
        desc.maxAnisotropy = 1; // анизотропия в WebGPU — только со всеми фильтрами linear
    return add({ kind: 'sampler', gpu: dev.createSampler(desc), nearest: dev.createSampler(nearest), comparison: !!desc.compare });
}

export function createShader(json) {
    const desc = JSON.parse(json);
    const module = stage => stage && { module: device().createShaderModule({ code: stage.code, label: desc.label }), entryPoint: stage.entryPoint };
    return add({
        kind: 'shader', label: desc.label, groupCount: desc.groupCount, usedGroups: new Set(desc.usedGroups),
        vertex: module(desc.vertex), fragment: module(desc.fragment), compute: module(desc.compute),
    });
}

// ============================ Группы привязок ============================

export function createBindGroupLayout(json) {
    return add({ kind: 'layout', entries: JSON.parse(json) });
}

const layoutCache = new Map(); // ключ (JSON записей) -> GPUBindGroupLayout
let emptyGroup = null;

function gpuLayout(entries) {
    const key = JSON.stringify(entries);
    let layout = layoutCache.get(key);
    if (!layout) {
        layout = device().createBindGroupLayout({ entries });
        layoutCache.set(key, { layout, key });
        return layoutCache.get(key);
    }
    return layout;
}

/** Пустая группа — для групп, которые шейдер не использует, но которые стоят в layout'е пайплайна до используемых. */
function empty() {
    if (!emptyGroup) {
        const layout = gpuLayout([]);
        emptyGroup = { gpu: device().createBindGroup({ layout: layout.layout, entries: [] }), layoutKey: layout.key, layout: layout.layout };
    }
    return emptyGroup;
}

export function createBindGroup(layoutId, json) {
    const description = objs[layoutId].entries;
    const resources = JSON.parse(json);
    const byBinding = new Map(resources.map(r => [r.binding, r]));

    // Текстуры, которые нельзя фильтровать, — по binding'у их сэмплера (sN — пара tN)
    const unfilterable = new Set();
    for (const r of resources) {
        if (r.view === undefined) continue;
        const v = objs[r.view];
        if (v && v.texture && isUnfilterable(v.texture.format)) unfilterable.add(r.binding - TEXTURE_SHIFT + SAMPLER_SHIFT);
    }

    const layoutEntries = [], entries = [];
    for (const d of description) {
        const r = byBinding.get(d.binding);
        const entry = { binding: d.binding, visibility: ALL_STAGES };
        switch (d.kind) {
            case 'uniform':
            case 'read-only-storage':
            case 'storage':
                entry.buffer = { type: d.kind };
                entries.push({ binding: d.binding, resource: { buffer: objs[r.buffer].gpu, offset: r.offset, size: r.size } });
                break;
            case 'texture': {
                const v = objs[r.view];
                entry.texture = {
                    sampleType: isUnfilterable(v.texture.format) ? 'unfilterable-float' : 'float',
                    viewDimension: v.dimension, multisampled: v.texture.samples > 1,
                };
                entries.push({ binding: d.binding, resource: v.gpu });
                break;
            }
            case 'sampler': {
                const s = objs[r.sampler];
                const nearest = unfilterable.has(d.binding);
                entry.sampler = { type: s.comparison ? 'comparison' : nearest ? 'non-filtering' : 'filtering' };
                entries.push({ binding: d.binding, resource: nearest ? s.nearest : s.gpu });
                break;
            }
            case 'storageTexture': {
                const v = objs[r.view];
                // Писать в storage-текстуру из вершинного шейдера нельзя
                entry.visibility = 2 | 4;
                entry.storageTexture = { access: 'write-only', format: d.format, viewDimension: v.dimension };
                entries.push({ binding: d.binding, resource: v.gpu });
                break;
            }
        }
        layoutEntries.push(entry);
    }

    const layout = gpuLayout(layoutEntries);
    return add({ kind: 'group', gpu: device().createBindGroup({ layout: layout.layout, entries }), layout: layout.layout, layoutKey: layout.key });
}

// ============================ Пайплайны ============================

export function createRenderPipeline(json) {
    const desc = JSON.parse(json);
    return add({ kind: 'render', desc, shader: objs[desc.shader], cache: new Map() });
}

export function createComputePipeline(json) {
    const desc = JSON.parse(json);
    return add({ kind: 'compute', desc, shader: objs[desc.shader], cache: new Map() });
}

/** Группы layout'а пайплайна: привязанные — там, где шейдер их использует, остальные — пустые. */
function pipelineGroups(shader, bound) {
    const groups = [];
    for (let g = 0; g < shader.groupCount; g++) {
        const group = shader.usedGroups.has(g) ? bound[g] : empty();
        if (!group) {
            report(`${shader.label}: group ${g} is not bound`);
            return null;
        }
        groups.push(group);
    }
    return groups;
}

function realizeRender(p, groups) {
    const key = groups.map(g => g.layoutKey).join('|');
    let pipeline = p.cache.get(key);
    if (pipeline) return pipeline;

    const d = p.desc, s = p.shader;
    const desc = {
        label: d.label,
        layout: device().createPipelineLayout({ bindGroupLayouts: groups.map(g => g.layout) }),
        vertex: { ...s.vertex, buffers: d.buffers },
        primitive: { topology: d.topology, cullMode: d.cullMode, frontFace: d.frontFace },
        multisample: { count: d.samples },
    };
    if (s.fragment)
        desc.fragment = { ...s.fragment, targets: d.targets };
    if (d.depthStencil)
        desc.depthStencil = d.depthStencil;
    pipeline = device().createRenderPipeline(desc);
    p.cache.set(key, pipeline);
    return pipeline;
}

function realizeCompute(p, groups) {
    const key = groups.map(g => g.layoutKey).join('|');
    let pipeline = p.cache.get(key);
    if (!pipeline) {
        pipeline = device().createComputePipeline({
            label: p.desc.label,
            layout: device().createPipelineLayout({ bindGroupLayouts: groups.map(g => g.layout) }),
            compute: p.shader.compute,
        });
        p.cache.set(key, pipeline);
    }
    return pipeline;
}

// ============================ Разрешение MSAA-глубины ============================
// В WebGPU своего нет: полноэкранный треугольник пишет глубину нулевого сэмпла (как блит в OpenGL)

const depthResolvePipelines = new Map();
let depthResolveModule = null;
function resolveDepth(encoder, source, destination) {
    const dev = device();
    depthResolveModule ??= dev.createShaderModule({ code: `
        @group(0) @binding(0) var depth: texture_depth_multisampled_2d;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
            return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
        }
        @fragment fn fs(@builtin(position) position: vec4f) -> @builtin(frag_depth) f32 {
            return textureLoad(depth, vec2i(position.xy), 0);
        }` });
    const format = destination.texture.format;
    let pipeline = depthResolvePipelines.get(format);
    if (!pipeline) {
        pipeline = dev.createRenderPipeline({
            layout: 'auto',
            vertex: { module: depthResolveModule, entryPoint: 'vs' },
            fragment: { module: depthResolveModule, entryPoint: 'fs', targets: [] },
            depthStencil: { format, depthWriteEnabled: true, depthCompare: 'always' },
        });
        depthResolvePipelines.set(format, pipeline);
    }
    const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: { view: destination.gpu, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: source.gpu }] }));
    pass.draw(3);
    pass.end();
}

// ============================ Кадр ============================

const f32 = new Float32Array(1), i32 = new Int32Array(f32.buffer);
function asFloat(bits) { i32[0] = bits; return f32[0]; }

/** Один вызов на кадр: команды кадра в один command buffer. Холст показывает браузер. */
export function submitFrame(commandsView, count) {
    const c = commandsView.slice(0, count);
    const dev = device();
    const encoder = dev.createCommandEncoder();
    let canvasView = null;
    const viewOf = id => id === 0 ? (canvasView ??= context.getCurrentTexture().createView()) : objs[id].gpu;

    let pass = null, compute = null, pipeline = null, depthResolve = null;
    let bound = [], vertexBuffers = [], indexBuffer = null;

    const prepare = isCompute => {
        const groups = pipelineGroups(pipeline.shader, bound);
        if (!groups) return false;
        const target = isCompute ? compute : pass;
        target.setPipeline(isCompute ? realizeCompute(pipeline, groups) : realizeRender(pipeline, groups));
        groups.forEach((g, i) => target.setBindGroup(i, g.gpu));
        return true;
    };

    let i = 0;
    while (i < count) {
        switch (c[i++]) {
            case OP.BEGIN_RENDER_PASS: {
                const colorAttachments = [];
                for (let n = c[i++]; n > 0; n--) {
                    const [view, clear, store, r, g, b, a, resolve] = c.subarray(i, i + 8); i += 8;
                    colorAttachments.push({
                        view: viewOf(view), loadOp: clear ? 'clear' : 'load', storeOp: store ? 'store' : 'discard',
                        clearValue: { r: asFloat(r), g: asFloat(g), b: asFloat(b), a: asFloat(a) },
                        resolveTarget: resolve >= 0 ? viewOf(resolve) : undefined,
                    });
                }
                const desc = { colorAttachments };
                depthResolve = null;
                if (c[i++]) {
                    const [view, clear, store, clearDepth, resolve] = c.subarray(i, i + 5); i += 5;
                    const depth = objs[view];
                    desc.depthStencilAttachment = {
                        view: depth.gpu, depthLoadOp: clear ? 'clear' : 'load', depthStoreOp: store || resolve >= 0 ? 'store' : 'discard',
                        depthClearValue: asFloat(clearDepth),
                    };
                    if (depth.texture.format.includes('stencil')) {
                        desc.depthStencilAttachment.stencilLoadOp = 'clear';
                        desc.depthStencilAttachment.stencilStoreOp = 'discard';
                    }
                    if (resolve >= 0) depthResolve = [depth, objs[resolve]];
                }
                pass = encoder.beginRenderPass(desc);
                bound = []; vertexBuffers = []; indexBuffer = null; pipeline = null;
                break;
            }
            case OP.END_RENDER_PASS:
                pass.end();
                pass = null;
                if (depthResolve) resolveDepth(encoder, depthResolve[0], depthResolve[1]);
                depthResolve = null;
                break;
            case OP.SET_PIPELINE:
            case OP.SET_COMPUTE_PIPELINE:
                pipeline = objs[c[i++]];
                break;
            case OP.SET_BIND_GROUP:
                bound[c[i]] = objs[c[i + 1]]; i += 2;
                break;
            case OP.SET_VERTEX_BUFFER:
                pass.setVertexBuffer(c[i], objs[c[i + 1]].gpu, c[i + 2]); i += 3;
                break;
            case OP.SET_INDEX_BUFFER:
                pass.setIndexBuffer(objs[c[i]].gpu, c[i + 1] ? 'uint32' : 'uint16', c[i + 2]); i += 3;
                break;
            case OP.SET_VIEWPORT:
                pass.setViewport(asFloat(c[i]), asFloat(c[i + 1]), asFloat(c[i + 2]), asFloat(c[i + 3]), asFloat(c[i + 4]), asFloat(c[i + 5]));
                i += 6;
                break;
            case OP.SET_SCISSOR:
                pass.setScissorRect(c[i], c[i + 1], c[i + 2], c[i + 3]); i += 4;
                break;
            case OP.DRAW:
                if (prepare(false)) pass.draw(c[i], c[i + 1], c[i + 2], c[i + 3]);
                i += 4;
                break;
            case OP.DRAW_INDEXED:
                if (prepare(false)) pass.drawIndexed(c[i], c[i + 1], c[i + 2], c[i + 3], c[i + 4]);
                i += 5;
                break;
            case OP.BEGIN_COMPUTE_PASS:
                compute = encoder.beginComputePass();
                bound = []; pipeline = null;
                break;
            case OP.END_COMPUTE_PASS:
                compute.end();
                compute = null;
                break;
            case OP.DISPATCH:
                if (prepare(true)) compute.dispatchWorkgroups(c[i], c[i + 1], c[i + 2]);
                i += 3;
                break;
            case OP.COPY_BUFFER:
                encoder.copyBufferToBuffer(objs[c[i]].gpu, c[i + 1], objs[c[i + 2]].gpu, c[i + 3], c[i + 4]); i += 5;
                break;
            case OP.COPY_TEXTURE: {
                const src = objs[c[i]], dst = objs[c[i + 1]]; i += 2;
                encoder.copyTextureToTexture({ texture: src.texture.gpu }, { texture: dst.texture.gpu },
                    { width: src.texture.width, height: src.texture.height });
                break;
            }
            default:
                report(`unknown command ${c[i - 1]} at ${i - 1}`);
                i = count;
        }
    }

    dev.queue.submit([encoder.finish()]);
    flushDestroy();
}
