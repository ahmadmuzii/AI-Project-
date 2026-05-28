import { useRef, useEffect } from 'react';
import { useTheme } from '../context/ThemeContext';

function hash21(px, py) {
  let h = (px * 374761393 + py * 668265263) | 0;
  h = ((h ^ (h >> 13)) * 1274126177) | 0;
  return ((h ^ (h >> 16)) & 0x7fffffff) / 0x7fffffff;
}

function vNoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const a = hash21(ix, iy), b = hash21(ix + 1, iy);
  const c = hash21(ix, iy + 1), d = hash21(ix + 1, iy + 1);
  return a + (b - a) * sx + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}

function fbm(x, y, n = 3) {
  let val = 0, amp = 1, freq = 1, maxAmp = 0;
  for (let i = 0; i < n; i++) {
    val += amp * vNoise(x * freq, y * freq);
    maxAmp += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return val / maxAmp;
}

function curlFlow(x, y, t, scale) {
  const e = 0.3;
  const s = scale;
  const n = (px, py) => fbm(px / s, py / s) + t * 0.00015;
  const dx = (n(x + e, y) - n(x - e, y)) / (2 * e);
  const dy = (n(x, y + e) - n(x, y - e)) / (2 * e);
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 0.0001) return { x: 0, y: 0 };
  return { x: dy / len, y: -dx / len };
}

const DARK_PALS = [
  [59, 130, 246], [139, 92, 246], [99, 102, 241],
  [168, 85, 247], [6, 182, 212], [56, 189, 248],
];
const LIGHT_PALS = [
  [148, 163, 184], [100, 116, 139], [180, 190, 210],
  [160, 170, 195], [200, 210, 230],
];

const LAYERS = [
  { scale: 300, count: 1500, speedMul: 0.3, size: 4, alphaMul: 0.25, maxSpeed: 1.2 },
  { scale: 180, count: 1000, speedMul: 0.5, size: 6, alphaMul: 0.45, maxSpeed: 1.8 },
  { scale: 100, count: 500, speedMul: 0.8, size: 9, alphaMul: 0.65, maxSpeed: 2.5 },
];

const VS = `#version 300 es
in vec2 a_corner;
in vec2 a_pos;
in vec2 a_prev;
in vec3 a_color;
in float a_alpha;
in float a_size;
uniform vec2 u_resolution;
out vec3 v_color;
out float v_alpha;
out vec2 v_uv;
void main() {
  vec2 dir = a_pos - a_prev;
  float speed = length(dir);
  vec2 ndir = speed > 0.001 ? dir / speed : vec2(0.0, 1.0);
  vec2 perp = vec2(-ndir.y, ndir.x);
  float stretch = 1.0 + speed * 3.0;
  vec2 offset = a_corner.x * ndir * a_size * stretch
              + a_corner.y * perp * a_size * 0.5;
  vec2 pos = (a_pos + offset) / u_resolution * 2.0 - 1.0;
  pos.y = -pos.y;
  gl_Position = vec4(pos, 0.0, 1.0);
  v_color = a_color;
  v_alpha = a_alpha;
  v_uv = a_corner;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_alpha;
in vec2 v_uv;
out vec4 fragColor;
void main() {
  float d = length(v_uv);
  float a = 1.0 - smoothstep(0.0, 0.5, d);
  if (a < 0.01) discard;
  fragColor = vec4(v_color * v_alpha * a * 0.15, 0.0);
}`;

function compileShader(gl, type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function createProgram(gl, vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(prog));
    return null;
  }
  return prog;
}

export default function ParticleField() {
  const canvasRef = useRef(null);
  const { isDark } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
    });
    if (!gl) return;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VS);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FS);
    const program = createProgram(gl, vs, fs);
    gl.useProgram(program);

    const aCornerLoc = gl.getAttribLocation(program, 'a_corner');
    const aPosLoc = gl.getAttribLocation(program, 'a_pos');
    const aPrevLoc = gl.getAttribLocation(program, 'a_prev');
    const aColorLoc = gl.getAttribLocation(program, 'a_color');
    const aAlphaLoc = gl.getAttribLocation(program, 'a_alpha');
    const aSizeLoc = gl.getAttribLocation(program, 'a_size');
    const uResLoc = gl.getUniformLocation(program, 'u_resolution');

    const verts = new Float32Array([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]);
    const idxs = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const vertBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(aCornerLoc);
    gl.vertexAttribPointer(aCornerLoc, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(aCornerLoc, 0);

    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idxs, gl.STATIC_DRAW);

    const FLOATS_PER_PARTICLE = 9;
    const MAX_PARTICLES = 3000;
    const instData = new Float32Array(MAX_PARTICLES * FLOATS_PER_PARTICLE);
    const instBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, instData.byteLength, gl.DYNAMIC_DRAW);

    const stride = FLOATS_PER_PARTICLE * 4;
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(aPosLoc, 1);

    gl.enableVertexAttribArray(aPrevLoc);
    gl.vertexAttribPointer(aPrevLoc, 2, gl.FLOAT, false, stride, 8);
    gl.vertexAttribDivisor(aPrevLoc, 1);

    gl.enableVertexAttribArray(aColorLoc);
    gl.vertexAttribPointer(aColorLoc, 3, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(aColorLoc, 1);

    gl.enableVertexAttribArray(aAlphaLoc);
    gl.vertexAttribPointer(aAlphaLoc, 1, gl.FLOAT, false, stride, 28);
    gl.vertexAttribDivisor(aAlphaLoc, 1);

    gl.enableVertexAttribArray(aSizeLoc);
    gl.vertexAttribPointer(aSizeLoc, 1, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(aSizeLoc, 1);

    gl.bindVertexArray(null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);

    let animating = true;
    let W, H;
    const cursor = {
      x: -2000, y: -2000, tx: -2000, ty: -2000,
      vx: 0, vy: 0, lx: -2000, ly: -2000,
    };
    const cursorRadius = 280;
    const cursorForce = 3.0;
    let t = 0;
    let particles = [];
    let particleCount = 0;

    const isDarkMode = isDark;
    const pals = isDarkMode ? DARK_PALS : LIGHT_PALS;

    function initParticles() {
      particles = [];
      LAYERS.forEach((cfg, li) => {
        for (let i = 0; i < cfg.count; i++) {
          const x = Math.random() * (W || 1920);
          const y = Math.random() * (H || 1080);
          const a = Math.random() * Math.PI * 2;
          const spd = (Math.random() * 0.3 + 0.1) * cfg.speedMul;
          particles.push({
            x, y, px: x, py: y,
            vx: Math.cos(a) * spd, vy: Math.sin(a) * spd,
            alpha: Math.random() * 0.3 + 0.7,
            color: pals[Math.floor(Math.random() * pals.length)],
            size: cfg.size * (Math.random() * 0.4 + 0.8),
            li, cfg,
          });
        }
      });
      particleCount = particles.length;
    }

    function resize() {
      W = Math.floor(window.innerWidth);
      H = Math.floor(window.innerHeight);
      canvas.width = W;
      canvas.height = H;
      gl.viewport(0, 0, W, H);
      gl.uniform2f(uResLoc, W, H);
    }
    resize();
    initParticles();
    window.addEventListener('resize', resize);

    function onMove(e) {
      cursor.tx = e.clientX;
      cursor.ty = e.clientY;
    }
    window.addEventListener('mousemove', onMove);

    let prevTime = 0;

    function draw(time) {
      if (!animating) return;
      const dt = Math.min(16.67, time - prevTime || 16.67);
      prevTime = time;
      t += dt * 0.06;

      cursor.lx = cursor.x;
      cursor.ly = cursor.y;
      cursor.x += (cursor.tx - cursor.x) * 0.08;
      cursor.y += (cursor.ty - cursor.y) * 0.08;
      cursor.vx = (cursor.x - cursor.lx) / 16;
      cursor.vy = (cursor.y - cursor.ly) / 16;

      const breath = 0.8 + 0.2 * Math.sin(t * 0.002 + 0.94);

      for (let i = 0; i < particleCount; i++) {
        const p = particles[i];
        const cfg = p.cfg;
        const li = p.li;

        const flow = curlFlow(p.x, p.y, t, cfg.scale);
        const flowSpeed = cfg.speedMul * 0.35;
        p.vx += flow.x * flowSpeed * 0.06;
        p.vy += flow.y * flowSpeed * 0.06;

        const dx = p.x - cursor.x;
        const dy = p.y - cursor.y;
        const distSq = dx * dx + dy * dy;
        const maxDist = cursorRadius + li * 40;
        const maxDistSq = maxDist * maxDist;
        if (distSq < maxDistSq && distSq > 1) {
          const dist = Math.sqrt(distSq);
          const strength = cursorForce * (1 - dist / maxDist) * (1 + li * 0.3);
          const nx = -dy / dist;
          const ny = dx / dist;
          p.vx += nx * strength * 0.04;
          p.vy += ny * strength * 0.04;
          p.vx += cursor.vx * strength * 0.008;
          p.vy += cursor.vy * strength * 0.008;
        }

        const drag = 0.96;
        p.vx *= drag;
        p.vy *= drag;
        p.vx += (Math.random() - 0.5) * 0.0015;
        p.vy += (Math.random() - 0.5) * 0.0015;

        const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (speed > cfg.maxSpeed) {
          p.vx = (p.vx / speed) * cfg.maxSpeed;
          p.vy = (p.vy / speed) * cfg.maxSpeed;
        }

        p.px = p.x;
        p.py = p.y;
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < -20) p.x = W + 20;
        else if (p.x > W + 20) p.x = -20;
        if (p.y < -20) p.y = H + 20;
        else if (p.y > H + 20) p.y = -20;

        const speedNorm = Math.min(1, speed / cfg.maxSpeed);
        const a = p.alpha * cfg.alphaMul * breath * (0.2 + 0.8 * speedNorm);

        const off = i * FLOATS_PER_PARTICLE;
        instData[off] = p.x;
        instData[off + 1] = p.y;
        instData[off + 2] = p.px;
        instData[off + 3] = p.py;
        instData[off + 4] = p.color[0] / 255;
        instData[off + 5] = p.color[1] / 255;
        instData[off + 6] = p.color[2] / 255;
        instData[off + 7] = a;
        instData[off + 8] = p.size;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, instData.subarray(0, particleCount * FLOATS_PER_PARTICLE));

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.bindVertexArray(vao);
      gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, particleCount);

      requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);

    return () => {
      animating = false;
      gl.bindVertexArray(null);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMove);
    };
  }, [isDark]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: -1,
      }}
    />
  );
}
