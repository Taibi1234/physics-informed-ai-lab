const SVG_NS = "http://www.w3.org/2000/svg";

export const MODELS = {
  data_only: { label: "Data only", color: "var(--data-only)" },
  data_ic: { label: "Data + IC", color: "var(--data-ic)" },
  pinn: { label: "PINN", color: "var(--pinn)" },
};

export function svgElement(tag, attributes = {}, text = "") {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => {
    element.setAttribute(name, String(value));
  });
  if (text !== "") element.textContent = text;
  return element;
}

function clear(svg) {
  svg.replaceChildren();
}

function line(svg, x1, y1, x2, y2, className) {
  svg.append(svgElement("line", { x1, y1, x2, y2, class: className }));
}

function text(svg, x, y, value, className, anchor = "start") {
  svg.append(svgElement("text", { x, y, class: className, "text-anchor": anchor }, value));
}

function pathFromPoints(points, xScale, yScale, valueIndex = 1) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point[0]).toFixed(2)},${yScale(point[valueIndex]).toFixed(2)}`)
    .join(" ");
}

function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

function niceStep(span, targetTicks = 5) {
  const rawStep = Math.abs(span) / targetTicks || 1;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const niceNormalized = normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1;
  return niceNormalized * magnitude;
}

function ticks(domainMin, domainMax, targetTicks = 5) {
  const step = niceStep(domainMax - domainMin, targetTicks);
  const start = Math.ceil(domainMin / step) * step;
  const values = [];
  for (let value = start; value <= domainMax + step * 1e-8; value += step) {
    values.push(Number(value.toFixed(10)));
  }
  return values;
}

function formatTick(value) {
  if (Math.abs(value) >= 10) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

export function renderHero(container, trajectory) {
  if (!trajectory) return;
  const svg = svgElement("svg", {
    viewBox: "0 0 600 280",
    role: "img",
    "aria-label": "Recorded analytical underdamped-oscillator trajectory from time zero to four pi",
  });
  const points = trajectory.points.map((point) => [point[0], point[1]]);
  const width = 600;
  const height = 280;
  const margin = { top: 20, right: 14, bottom: 36, left: 42 };
  const xMax = points.at(-1)[0];
  const x = linearScale(0, xMax, margin.left, width - margin.right);
  const y = linearScale(-0.7, 1.05, height - margin.bottom, margin.top);

  ticks(-0.6, 1, 4).forEach((tick) => {
    line(svg, margin.left, y(tick), width - margin.right, y(tick), "chart-grid");
    text(svg, margin.left - 8, y(tick) + 4, formatTick(tick), "chart-tick", "end");
  });
  [0, Math.PI, Math.PI * 2, Math.PI * 3, Math.PI * 4].forEach((tick) => {
    text(svg, x(tick), height - 10, tick === 0 ? "0" : `${Math.round(tick / Math.PI)}π`, "chart-tick", "middle");
  });
  line(svg, margin.left, y(0), width - margin.right, y(0), "chart-axis");
  svg.append(svgElement("path", {
    d: pathFromPoints(points, x, y),
    class: "series-line",
    stroke: "var(--pinn)",
    "stroke-width": 3,
  }));
  container.replaceChildren(svg);
}

export function renderTrajectory(svg, options) {
  const { series, showExact, view, inspectIndex } = options;
  clear(svg);
  if (!series.length) return;

  const width = 900;
  const height = 430;
  const margin = { top: 24, right: 26, bottom: 60, left: 72 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const reference = series[0].points.map((point) => [point[0], point[1]]);
  const xMax = reference.at(-1)[0];
  const x = linearScale(0, xMax, margin.left, width - margin.right);

  const displaySeries = series.map((item) => ({
    ...item,
    displayPoints: item.points.map((point) => [
      point[0],
      view === "error" ? Math.abs(point[2] - point[1]) : point[2],
    ]),
  }));
  const allValues = displaySeries.flatMap((item) => item.displayPoints.map((point) => point[1]));
  if (showExact && view === "trajectory") allValues.push(...reference.map((point) => point[1]));

  let yMin = view === "error" ? 0 : Math.min(...allValues);
  let yMax = Math.max(...allValues);
  const padding = Math.max((yMax - yMin) * 0.1, view === "error" ? 0.01 : 0.05);
  if (view !== "error") yMin -= padding;
  yMax += padding;
  const y = linearScale(yMin, yMax, height - margin.bottom, margin.top);

  ticks(yMin, yMax, 5).forEach((tick) => {
    line(svg, margin.left, y(tick), width - margin.right, y(tick), "chart-grid");
    text(svg, margin.left - 12, y(tick) + 4, formatTick(tick), "chart-tick", "end");
  });
  ticks(0, xMax, 4).forEach((tick) => {
    line(svg, x(tick), margin.top, x(tick), height - margin.bottom, "chart-grid");
    text(svg, x(tick), height - 32, tick.toFixed(2), "chart-tick", "middle");
  });
  line(svg, margin.left, height - margin.bottom, width - margin.right, height - margin.bottom, "chart-axis");
  line(svg, margin.left, margin.top, margin.left, height - margin.bottom, "chart-axis");
  text(svg, margin.left + plotWidth / 2, height - 6, "TIME t", "chart-axis-title", "middle");
  const yTitle = view === "error" ? "ABSOLUTE POSITION ERROR" : "POSITION x(t)";
  const yTitleElement = svgElement("text", {
    x: 18,
    y: margin.top + plotHeight / 2,
    class: "chart-axis-title",
    "text-anchor": "middle",
    transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`,
  }, yTitle);
  svg.append(yTitleElement);

  if (showExact && view === "trajectory") {
    svg.append(svgElement("path", { d: pathFromPoints(reference, x, y), class: "reference-line" }));
  }
  displaySeries.forEach((item) => {
    svg.append(svgElement("path", {
      d: pathFromPoints(item.displayPoints, x, y),
      class: "series-line",
      stroke: item.color,
      "data-mode": item.mode,
    }));
  });

  const safeIndex = Math.max(0, Math.min(inspectIndex, reference.length - 1));
  const inspectTime = reference[safeIndex][0];
  const inspectX = x(inspectTime);
  line(svg, inspectX, margin.top, inspectX, height - margin.bottom, "inspection-line");
  if (showExact && view === "trajectory") {
    svg.append(svgElement("circle", {
      cx: inspectX,
      cy: y(reference[safeIndex][1]),
      r: 5,
      fill: "var(--text)",
      class: "inspection-point",
    }));
  }
  displaySeries.forEach((item) => {
    svg.append(svgElement("circle", {
      cx: inspectX,
      cy: y(item.displayPoints[safeIndex][1]),
      r: 5,
      fill: item.color,
      class: "inspection-point",
    }));
  });

  svg.append(svgElement("rect", {
    id: "chart-hit-area",
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "transparent",
    "aria-hidden": "true",
  }));
  svg.dataset.xMax = String(xMax);
  svg.dataset.marginLeft = String(margin.left);
  svg.dataset.plotWidth = String(plotWidth);
}

export function renderAggregate(svg, summary, metric) {
  clear(svg);
  const width = 720;
  const height = 340;
  const margin = { top: 36, right: 84, bottom: 54, left: 112 };
  const field = metric === "position" ? "position_rmse_mean" : "physics_residual_rmse_mean";
  const maxValue = Math.max(...summary.map((row) => row[field]));
  const xMax = maxValue * 1.18;
  const x = linearScale(0, xMax, margin.left, width - margin.right);
  const rowHeight = 72;

  ticks(0, xMax, 4).forEach((tick) => {
    line(svg, x(tick), margin.top - 10, x(tick), height - margin.bottom, "chart-grid");
    text(svg, x(tick), height - 24, formatTick(tick), "chart-tick", "middle");
  });
  summary.forEach((row, index) => {
    const y = margin.top + index * rowHeight;
    const value = row[field];
    const barWidth = x(value) - x(0);
    text(svg, 0, y + 26, MODELS[row.mode].label, "bar-label");
    svg.append(svgElement("rect", { x: margin.left, y, width: width - margin.left - margin.right, height: 38, class: "bar-track" }));
    svg.append(svgElement("rect", { x: margin.left, y, width: Math.max(1, barWidth), height: 38, fill: MODELS[row.mode].color }));
    text(svg, Math.min(x(value) + 10, width - 72), y + 25, value.toFixed(4), "bar-value");
    if (metric === "position") {
      text(svg, 0, y + 45, `± ${row.position_rmse_std.toFixed(4)} population std`, "bar-note");
    } else {
      text(svg, 0, y + 45, `${row.runs} recorded runs`, "bar-note");
    }
  });
  text(svg, margin.left, height - 2, metric === "position" ? "MEAN POSITION RMSE" : "MEAN PHYSICS-RESIDUAL RMSE", "chart-axis-title");
}

export function renderWeightChart(svg, rows, metric) {
  clear(svg);
  const width = 820;
  const height = 380;
  const margin = { top: 30, right: 34, bottom: 62, left: 76 };
  const meanField = metric === "position" ? "position_mean" : "residual_mean";
  const stdField = metric === "position" ? "position_std" : "residual_std";
  const maxValue = Math.max(...rows.map((row) => row[meanField] + row[stdField]));
  const yMax = maxValue * 1.18;
  const xPositions = rows.map((_, index) => margin.left + 90 + index * ((width - margin.left - margin.right - 180) / Math.max(rows.length - 1, 1)));
  const y = linearScale(0, yMax, height - margin.bottom, margin.top);

  ticks(0, yMax, 5).forEach((tick) => {
    line(svg, margin.left, y(tick), width - margin.right, y(tick), "chart-grid");
    text(svg, margin.left - 12, y(tick) + 4, formatTick(tick), "chart-tick", "end");
  });
  line(svg, margin.left, height - margin.bottom, width - margin.right, height - margin.bottom, "chart-axis");

  const path = rows.map((row, index) => `${index === 0 ? "M" : "L"}${xPositions[index]},${y(row[meanField])}`).join(" ");
  svg.append(svgElement("path", { d: path, class: "series-line", stroke: "var(--pinn)", "stroke-width": 3 }));

  rows.forEach((row, index) => {
    const x = xPositions[index];
    const mean = row[meanField];
    const std = row[stdField];
    line(svg, x, y(mean - std), x, y(mean + std), "error-bar");
    line(svg, x - 7, y(mean - std), x + 7, y(mean - std), "error-bar");
    line(svg, x - 7, y(mean + std), x + 7, y(mean + std), "error-bar");
    svg.append(svgElement("circle", { cx: x, cy: y(mean), r: 6, class: "weight-point" }));
    text(svg, x, height - 30, String(row.physics_weight), "chart-tick", "middle");
  });
  text(svg, margin.left + (width - margin.left - margin.right) / 2, height - 4, "PHYSICS-Loss WEIGHT λ", "chart-axis-title", "middle");
  const title = metric === "position" ? "MEAN POSITION RMSE" : "MEAN PHYSICS-RESIDUAL RMSE";
  svg.append(svgElement("text", {
    x: 18,
    y: margin.top + (height - margin.top - margin.bottom) / 2,
    class: "chart-axis-title",
    "text-anchor": "middle",
    transform: `rotate(-90 18 ${margin.top + (height - margin.top - margin.bottom) / 2})`,
  }, title));
}
