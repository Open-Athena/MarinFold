/* Rotatable backbone preview for browsers without a WebGL context. */
window.createBackboneFallback = async function createBackboneFallback(
  element,
  protein,
  isCurrent,
) {
  const url = protein.pdbFallbackUrl || protein.structureUrl;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`PDB ${response.status}`);
  const pdb = await response.text();
  const all = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("ATOM  ") || line.slice(12, 16).trim() !== "CA")
      continue;
    all.push({
      chain: line[21],
      x: Number(line.slice(30, 38)),
      y: Number(line.slice(38, 46)),
      z: Number(line.slice(46, 54)),
    });
  }
  if (!all.length) throw new Error("No Cα atoms in structure");
  const chain =
    protein.viewerChain &&
    all.some((point) => point.chain === protein.viewerChain)
      ? protein.viewerChain
      : all[0].chain;
  const points = all.filter((point) => point.chain === chain);
  if (points.length < 2) throw new Error("Backbone has too few Cα atoms");
  const center = points
    .reduce(
      (sum, point) => [sum[0] + point.x, sum[1] + point.y, sum[2] + point.z],
      [0, 0, 0],
    )
    .map((v) => v / points.length);
  const coords = points.map((point) => [
    point.x - center[0],
    point.y - center[1],
    point.z - center[2],
  ]);
  const radius = Math.max(...coords.map(([x, y, z]) => Math.hypot(x, y, z)));
  const canvas = document.createElement("canvas");
  canvas.className = "backbone-canvas";
  canvas.setAttribute(
    "aria-label",
    `Interactive 3D backbone of ${protein.label}`,
  );
  if (!isCurrent()) throw new Error("Selection changed");
  element.replaceChildren(canvas);
  let angleX = -0.25,
    angleY = 0.65,
    zoom = 1,
    dragging = false,
    lastX = 0,
    lastY = 0;
  let disposed = false;
  const resize = () => {
    if (disposed) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(element.clientWidth * scale));
    canvas.height = Math.max(1, Math.floor(element.clientHeight * scale));
    draw();
  };
  const draw = () => {
    const ctx = canvas.getContext("2d");
    const w = canvas.width,
      h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const gradient = ctx.createRadialGradient(
      w / 2,
      h / 2,
      5,
      w / 2,
      h / 2,
      Math.max(w, h) * 0.65,
    );
    gradient.addColorStop(0, "#24433b");
    gradient.addColorStop(1, "#0d201d");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
    const base = (Math.min(w, h) * 0.4 * zoom) / Math.max(radius, 1);
    const sx = Math.sin(angleX),
      cx = Math.cos(angleX),
      sy = Math.sin(angleY),
      cy = Math.cos(angleY);
    const projected = coords.map(([x, y, z]) => {
      const xx = x * cy + z * sy,
        zz = z * cy - x * sy;
      const yy = y * cx - zz * sx,
        depth = y * sx + zz * cx;
      const perspective = 1 / (1 + depth / (radius * 4 + 1));
      return [
        w / 2 + xx * base * perspective,
        h / 2 - yy * base * perspective,
        depth,
      ];
    });
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const pixelRatio = window.devicePixelRatio || 1;
    for (let i = 1; i < projected.length; i++) {
      const [ax, ay] = projected[i - 1],
        [bx, by] = projected[i];
      const [x0, y0, z0] = coords[i - 1],
        [x1, y1, z1] = coords[i];
      if (Math.hypot(x1 - x0, y1 - y0, z1 - z0) > 8) continue;
      const hue = 145 + (45 * i) / projected.length;
      ctx.strokeStyle = `hsl(${hue} 64% ${49 + (13 * i) / projected.length}%)`;
      ctx.lineWidth = 2.6 * pixelRatio;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.fillStyle = "#b7ebce";
    for (
      let i = 0;
      i < projected.length;
      i += Math.max(1, Math.floor(projected.length / 110))
    ) {
      ctx.beginPath();
      ctx.arc(
        projected[i][0],
        projected[i][1],
        1.8 * pixelRatio,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    ctx.fillStyle = "#8eafa2";
    ctx.font = `${10 * pixelRatio}px monospace`;
    ctx.fillText(
      `${points.length} Cα atoms · drag to rotate`,
      18 * pixelRatio,
      h - 17 * pixelRatio,
    );
  };
  const onDown = (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  };
  const onMove = (event) => {
    if (!dragging) return;
    angleY += (event.clientX - lastX) * 0.008;
    angleX += (event.clientY - lastY) * 0.008;
    lastX = event.clientX;
    lastY = event.clientY;
    draw();
  };
  const onUp = () => {
    dragging = false;
  };
  const onWheel = (event) => {
    event.preventDefault();
    zoom = Math.max(0.4, Math.min(3, zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
    draw();
  };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("resize", resize);
  resize();
  return () => {
    disposed = true;
    window.removeEventListener("resize", resize);
  };
};
