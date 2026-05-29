import { useRef, useEffect } from 'react';
import { useTheme } from '../context/ThemeContext';

const GRADIENT_VS = `#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const GRADIENT_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform float u_time;
uniform bool u_dark;
out vec4 fragColor;
void main() {
  float t = u_time * 0.02;
  vec2 uv = v_uv;

  float v1 = sin(uv.x * 1.5 + t * 0.3 + sin(uv.y * 1.2 + t * 0.2));
  float v2 = cos(uv.y * 1.8 + t * 0.4 + sin(uv.x * 0.9 + t * 0.5));
  float v3 = sin((uv.x + uv.y) * 1.1 + t * 0.5);

  vec3 c1, c2, c3;
  if (u_dark) {
    c1 = vec3(0.35, 0.10, 0.55);
    c2 = vec3(0.08, 0.15, 0.45);
    c3 = vec3(0.02, 0.02, 0.10);
  } else {
    c1 = vec3(0.85, 0.45, 0.25);
    c2 = vec3(0.25, 0.55, 0.85);
    c3 = vec3(0.95, 0.90, 0.85);
  }

  vec3 color = mix(c1, c2, v1 * 0.5 + 0.5);
  color = mix(color, c3, v2 * 0.3 + 0.35);
  color += v3 * 0.04;

  fragColor = vec4(color, 1.0);
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
      premultipliedAlpha: true,
      antialias: false,
    });
    if (!gl) return;

    const gvs = compileShader(gl, gl.VERTEX_SHADER, GRADIENT_VS);
    const gfs = compileShader(gl, gl.FRAGMENT_SHADER, GRADIENT_FS);
    const gradProgram = createProgram(gl, gvs, gfs);

    const gPosLoc = gl.getAttribLocation(gradProgram, 'a_pos');
    const gTimeLoc = gl.getUniformLocation(gradProgram, 'u_time');
    const gDarkLoc = gl.getUniformLocation(gradProgram, 'u_dark');

    const gVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const gradVAO = gl.createVertexArray();
    gl.bindVertexArray(gradVAO);
    const gVertBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, gVertBuf);
    gl.bufferData(gl.ARRAY_BUFFER, gVerts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(gPosLoc);
    gl.vertexAttribPointer(gPosLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    let animating = true;
    let W, H;
    let t = 0;
    const isDarkMode = isDark;

    function resize() {
      W = Math.floor(window.innerWidth);
      H = Math.floor(window.innerHeight);
      canvas.width = W;
      canvas.height = H;
      gl.viewport(0, 0, W, H);
    }
    resize();
    window.addEventListener('resize', resize);

    let prevTime = 0;

    function draw(time) {
      if (!animating) return;
      const dt = Math.min(16.67, time - prevTime || 16.67);
      prevTime = time;
      t += dt * 0.06;

      gl.clearColor(0, 0, 0, isDarkMode ? 1 : 0.9);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(gradProgram);
      gl.uniform1f(gTimeLoc, t);
      gl.uniform1i(gDarkLoc, isDarkMode ? 1 : 0);
      gl.bindVertexArray(gradVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);

    return () => {
      animating = false;
      gl.bindVertexArray(null);
      gl.deleteProgram(gradProgram);
      gl.deleteShader(gvs);
      gl.deleteShader(gfs);
      window.removeEventListener('resize', resize);
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
