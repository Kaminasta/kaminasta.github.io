// Точка входа веб-демо: WebGPU-устройство создаётся до запуска .NET (запрос асинхронный),
// затем C# строит движок в Main, а кадры идут через requestAnimationFrame -> Program.Frame.
import { dotnet } from './_framework/dotnet.js';
import * as webgpuRhi from './_content/Negine.WebGPU/negine-webgpu-rhi.js';

const canvas = document.getElementById('canvas');

// Загрузочный экран: скачивание ресурсов .NET — до 80% полосы, запуск движка — остальное
const loader = document.getElementById('loader');
const bar = document.getElementById('bar');
const caption = document.getElementById('caption');

function setProgress(fraction, text) {
    bar.style.width = `${Math.round(Math.min(Math.max(fraction, 0), 1) * 100)}%`;
    if (text) caption.textContent = text;
}

function canvasSize() {
    const dpr = window.devicePixelRatio || 1;
    return [Math.max(1, Math.round(canvas.clientWidth * dpr)), Math.max(1, Math.round(canvas.clientHeight * dpr))];
}

/** События DOM -> WebInputEvents (C#). Координаты — в пикселях фреймбуфера. */
function bindInput(input) {
    const dpr = () => window.devicePixelRatio || 1;
    canvas.addEventListener('mousemove', e => input.MouseMove(e.offsetX * dpr(), e.offsetY * dpr()));
    canvas.addEventListener('mousedown', e => { canvas.focus(); input.MouseButton(e.button, true); });
    window.addEventListener('mouseup', e => input.MouseButton(e.button, false));
    canvas.addEventListener('contextmenu', e => e.preventDefault()); // правая кнопка — поворот камеры
    canvas.addEventListener('wheel', e => { e.preventDefault(); input.MouseWheel(-e.deltaX / 100, -e.deltaY / 100); }, { passive: false });

    canvas.addEventListener('keydown', e => {
        input.Key(e.code, true);
        if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) input.Char(e.key);
        if (!e.ctrlKey || e.code === 'Tab') e.preventDefault(); // не прокручивать страницу пробелом и стрелками
    });
    canvas.addEventListener('keyup', e => input.Key(e.code, false));
    canvas.focus();
}

try {
    [canvas.width, canvas.height] = canvasSize();
    setProgress(0.02, 'ЗАГРУЗКА');
    await webgpuRhi.initialize(canvas);

    const { setModuleImports, getAssemblyExports, getConfig, runMain } = await dotnet
        .withModuleConfig({ onDownloadResourceProgress: (loaded, total) => total > 0 && setProgress(0.05 + 0.75 * loaded / total) })
        .withApplicationArguments()
        .create();
    setProgress(0.85, 'ЗАПУСК');
    setModuleImports('negine-webgpu-rhi', webgpuRhi);
    setModuleImports('main.js', { canvasSize: () => canvasSize() });

    const exports = await getAssemblyExports(getConfig().mainAssemblyName);
    const program = exports.Negine.Demo.Web.Program;

    await runMain();
    setProgress(1, '');
    loader.classList.add('done'); // плавно гаснет (index.html)
    setTimeout(() => loader.remove(), 700);
    bindInput(exports.Negine.Demo.Web.Inputs.WebInputEvents);

    let last = performance.now();
    const frame = now => {
        const [w, h] = canvasSize();
        program.Frame((now - last) / 1000, w, h);
        last = now;
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
} catch (e) {
    caption.classList.add('error');
    caption.textContent = 'Ошибка: ' + (e?.message ?? e);
    console.error(e);
}
