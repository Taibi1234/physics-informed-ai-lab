import {
  MODELS,
  renderAggregate as drawAggregate,
  renderHero,
  renderTrajectory,
  renderWeightChart,
} from "./charts.js";

const MODEL_ORDER = ["data_only", "data_ic", "pinn"];
const state = {
  data: null,
  seed: 42,
  modes: new Set(MODEL_ORDER),
  showExact: true,
  view: "trajectory",
  inspectIndex: 0,
  tooltipVisible: false,
  aggregateMetric: "position",
  weightMetric: "position",
};

const byId = (id) => document.getElementById(id);
const trajectorySvg = byId("trajectory-chart");
const tooltip = byId("trajectory-tooltip");
const scrubber = byId("time-scrubber");

function setPressed(id, pressed) {
  byId(id).setAttribute("aria-pressed", String(pressed));
}

function currentSeries() {
  if (!state.data) return [];
  return state.data.trajectories
    .filter((series) => series.seed === state.seed && state.modes.has(series.mode))
    .sort((a, b) => MODEL_ORDER.indexOf(a.mode) - MODEL_ORDER.indexOf(b.mode))
    .map((series) => ({
      ...series,
      label: MODELS[series.mode].label,
      color: MODELS[series.mode].color,
    }));
}

function renderExplorer() {
  const series = currentSeries();
  if (!series.length) return;
  const pointCount = series[0].points.length - 1;
  state.inspectIndex = Math.max(0, Math.min(state.inspectIndex, pointCount));
  scrubber.max = String(pointCount);
  scrubber.value = String(state.inspectIndex);

  byId("run-label").textContent = `DAMPED STUDY / SEED ${state.seed}`;
  byId("metric-seed").textContent = `Seed ${state.seed}`;
  byId("trajectory-title").textContent = state.view === "error" ? "Absolute prediction error" : "Position over time";
  byId("chart-caption").textContent = state.view === "error"
    ? "Absolute error is calculated directly as |predicted position − analytical position| at each recorded grid point."
    : "Solid colored lines are model predictions. The dashed line is the analytical reference.";

  renderTrajectory(trajectorySvg, {
    series,
    showExact: state.showExact,
    view: state.view,
    inspectIndex: state.inspectIndex,
  });
  updateInspector();
  renderRunMetrics();

  document.querySelectorAll(".seed-button").forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.seed) === state.seed));
  });
  document.querySelectorAll(".model-toggle").forEach((button) => {
    const selected = state.modes.has(button.dataset.mode);
    button.setAttribute("aria-pressed", String(selected));
    button.querySelector(".toggle-state").textContent = selected ? "ON" : "OFF";
    button.disabled = selected && state.modes.size === 1;
    button.title = button.disabled ? "At least one model must remain visible" : "";
  });
  setPressed("exact-toggle", state.showExact);
  byId("exact-toggle").querySelector(".toggle-state").textContent = state.showExact ? "ON" : "OFF";
  byId("exact-toggle").disabled = state.view === "error";
  setPressed("view-trajectory", state.view === "trajectory");
  setPressed("view-error", state.view === "error");
}

function updateInspector() {
  const series = currentSeries();
  if (!series.length) return;
  const reference = series[0].points[state.inspectIndex];
  const time = reference[0];
  byId("time-output").textContent = `t = ${time.toFixed(3)}`;
  scrubber.setAttribute("aria-valuetext", `Time ${time.toFixed(3)}`);

  if (!state.tooltipVisible) {
    tooltip.hidden = true;
    return;
  }

  const rows = [];
  if (state.view === "trajectory" && state.showExact) {
    rows.push(`<div class="tooltip-row"><span>Analytical</span><strong>${reference[1].toFixed(4)}</strong></div>`);
  }
  series.forEach((item) => {
    const point = item.points[state.inspectIndex];
    const value = state.view === "error" ? Math.abs(point[2] - point[1]) : point[2];
    rows.push(`<div class="tooltip-row"><span><i class="tooltip-swatch" style="background:${MODELS[item.mode].color}"></i>${MODELS[item.mode].label}</span><strong>${value.toFixed(4)}</strong></div>`);
  });
  tooltip.innerHTML = `<div class="tooltip-row"><span>TIME</span><strong>${time.toFixed(4)}</strong></div>${rows.join("")}`;
  tooltip.hidden = false;

  const svgRect = trajectorySvg.getBoundingClientRect();
  const wrapRect = byId("trajectory-wrap").getBoundingClientRect();
  const marginLeft = Number(trajectorySvg.dataset.marginLeft || 72);
  const plotWidth = Number(trajectorySvg.dataset.plotWidth || 800);
  const ratio = state.inspectIndex / Math.max(series[0].points.length - 1, 1);
  const xInSvg = marginLeft + ratio * plotWidth;
  const x = svgRect.left - wrapRect.left + (xInSvg / 900) * svgRect.width;
  const placeLeft = ratio > 0.58;
  tooltip.style.left = `${x + (placeLeft ? -8 : 8)}px`;
  tooltip.style.transform = placeLeft ? "translateX(-100%)" : "none";
}

function renderRunMetrics() {
  const allRuns = state.data.runs.filter((run) => run.seed === state.seed);
  const selectedRuns = allRuns
    .filter((run) => state.modes.has(run.mode))
    .sort((a, b) => MODEL_ORDER.indexOf(a.mode) - MODEL_ORDER.indexOf(b.mode));
  const metrics = [
    { field: "position_rmse", label: "POSITION RMSE", short: "RMSE" },
    { field: "position_mae", label: "POSITION MAE", short: "MAE" },
    { field: "physics_residual_rmse", label: "PHYSICS-RESIDUAL RMSE", short: "PHYS." },
  ];
  const best = Object.fromEntries(metrics.map((metric) => [
    metric.field,
    Math.min(...allRuns.map((run) => run[metric.field])),
  ]));

  byId("run-metrics").innerHTML = selectedRuns.map((run) => `
    <article class="run-metric">
      <div class="run-metric-head"><span class="series-key series-${run.mode === "data_only" ? "data" : run.mode === "data_ic" ? "ic" : "pinn"}"></span>${MODELS[run.mode].label}</div>
      <div class="metric-values">
        ${metrics.map((metric) => `
          <div><span title="${metric.label}">${metric.short}</span><strong class="${run[metric.field] === best[metric.field] ? "best-value" : ""}">${run[metric.field].toFixed(4)}</strong></div>
        `).join("")}
      </div>
    </article>
  `).join("");
}

function renderAggregate() {
  const summary = [...state.data.summary].sort((a, b) => MODEL_ORDER.indexOf(a.mode) - MODEL_ORDER.indexOf(b.mode));
  const metric = state.aggregateMetric;
  const field = metric === "position" ? "position_rmse_mean" : "physics_residual_rmse_mean";
  byId("aggregate-title").textContent = metric === "position" ? "Mean position error by model" : "Mean equation residual by model";
  setPressed("aggregate-position", metric === "position");
  setPressed("aggregate-residual", metric === "residual");
  renderAggregateChart(summary, metric);
  byId("aggregate-table").innerHTML = `<table><caption>Five-run aggregate metrics</caption><thead><tr><th>Model</th><th>${metric === "position" ? "Position RMSE" : "Residual RMSE"}</th></tr></thead><tbody>${summary.map((row) => `<tr><td>${MODELS[row.mode].label}</td><td>${row[field].toFixed(8)}</td></tr>`).join("")}</tbody></table>`;
}

function renderAggregateChart(summary, metric) {
  drawAggregate(byId("aggregate-chart"), summary, metric);
}

function renderWeightStudy() {
  const rows = state.data.undamped_weight_sweep;
  const metric = state.weightMetric;
  const meanField = metric === "position" ? "position_mean" : "residual_mean";
  const stdField = metric === "position" ? "position_std" : "residual_std";
  byId("weight-chart-title").textContent = metric === "position" ? "Position RMSE vs. physics weight" : "Residual RMSE vs. physics weight";
  setPressed("weight-position", metric === "position");
  setPressed("weight-residual", metric === "residual");
  renderWeightChart(byId("weight-chart"), rows, metric);
  byId("weight-values").innerHTML = rows.map((row) => `
    <div class="weight-value"><span>λ = ${row.physics_weight}</span><strong>${row[meanField].toFixed(4)}</strong><small>± ${row[stdField].toFixed(4)} population std</small></div>
  `).join("");
}

function populateStaticData() {
  const summary = state.data.summary;
  const bestPosition = summary.reduce((best, row) => row.position_rmse_mean < best.position_rmse_mean ? row : best);
  const bestResidual = summary.reduce((best, row) => row.physics_residual_rmse_mean < best.physics_residual_rmse_mean ? row : best);
  byId("best-position-value").textContent = bestPosition.position_rmse_mean.toFixed(4);
  byId("best-residual-value").textContent = bestResidual.physics_residual_rmse_mean.toFixed(4);
  byId("solver-error").textContent = Number(state.data.config.reference_solver_max_abs_error).toExponential(3);

  const weightTen = state.data.undamped_weight_sweep.find((row) => row.physics_weight === 10);
  byId("weight-ten-residual").textContent = weightTen.residual_mean.toFixed(4);
  byId("weight-ten-std").textContent = weightTen.position_std.toFixed(4);

  const seeds = [...new Set(state.data.runs.map((run) => run.seed))].sort((a, b) => a - b);
  if (!seeds.includes(state.seed)) state.seed = seeds[0];
  byId("seed-controls").innerHTML = seeds.map((seed) => `<button class="seed-button" type="button" data-seed="${seed}" aria-pressed="${seed === state.seed}">${seed}</button>`).join("");

  const heroTrajectory = state.data.trajectories.find((series) => series.seed === state.seed && series.mode === "data_only");
  renderHero(byId("hero-chart"), heroTrajectory);
}

function validatePayload(payload) {
  if (!payload || !Array.isArray(payload.runs) || !Array.isArray(payload.trajectories) || !Array.isArray(payload.summary) || !Array.isArray(payload.undamped_weight_sweep)) {
    throw new Error("Experiment data has an invalid structure");
  }
  if (!payload.runs.length || !payload.trajectories.length || !payload.summary.length || !payload.undamped_weight_sweep.length) {
    throw new Error("Experiment data is empty");
  }
}

async function loadData() {
  byId("data-loading").hidden = false;
  byId("data-error").hidden = true;
  byId("explorer-shell").hidden = true;
  byId("weight-grid").hidden = true;
  try {
    const response = await fetch("./data/experiment_data.json");
    if (!response.ok) throw new Error(`Data request returned ${response.status}`);
    const payload = await response.json();
    validatePayload(payload);
    state.data = payload;
    populateStaticData();
    renderExplorer();
    renderAggregate();
    renderWeightStudy();
    byId("data-loading").hidden = true;
    byId("explorer-shell").hidden = false;
    byId("weight-grid").hidden = false;
  } catch (error) {
    byId("data-loading").hidden = true;
    byId("data-error").hidden = false;
    console.error("Unable to load experiment data:", error);
  }
}

function downloadShownData() {
  const rows = [["seed", "mode", "time", "exact_position", "predicted_position"]];
  currentSeries().forEach((series) => {
    series.points.forEach((point) => {
      rows.push([state.seed, series.mode, point[0], point[1], point[2]]);
    });
  });
  const csv = rows.map((row) => row.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `damped_seed_${state.seed}_shown_models.csv`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

byId("seed-controls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-seed]");
  if (!button || !state.data) return;
  state.seed = Number(button.dataset.seed);
  state.inspectIndex = 0;
  renderExplorer();
  const heroTrajectory = state.data.trajectories.find((series) => series.seed === state.seed && series.mode === "data_only");
  renderHero(byId("hero-chart"), heroTrajectory);
});

byId("model-controls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button || !state.data) return;
  const mode = button.dataset.mode;
  if (state.modes.has(mode)) {
    if (state.modes.size === 1) return;
    state.modes.delete(mode);
  } else {
    state.modes.add(mode);
  }
  renderExplorer();
});

byId("exact-toggle").addEventListener("click", () => {
  if (state.view === "error") return;
  state.showExact = !state.showExact;
  renderExplorer();
});
byId("view-trajectory").addEventListener("click", () => { state.view = "trajectory"; renderExplorer(); });
byId("view-error").addEventListener("click", () => { state.view = "error"; renderExplorer(); });
byId("reset-explorer").addEventListener("click", () => {
  state.seed = 42;
  state.modes = new Set(MODEL_ORDER);
  state.showExact = true;
  state.view = "trajectory";
  state.inspectIndex = 0;
  state.tooltipVisible = false;
  if (!state.data.runs.some((run) => run.seed === state.seed)) state.seed = state.data.runs[0].seed;
  renderExplorer();
  const heroTrajectory = state.data.trajectories.find((series) => series.seed === state.seed && series.mode === "data_only");
  renderHero(byId("hero-chart"), heroTrajectory);
});
byId("aggregate-position").addEventListener("click", () => { state.aggregateMetric = "position"; renderAggregate(); });
byId("aggregate-residual").addEventListener("click", () => { state.aggregateMetric = "residual"; renderAggregate(); });
byId("weight-position").addEventListener("click", () => { state.weightMetric = "position"; renderWeightStudy(); });
byId("weight-residual").addEventListener("click", () => { state.weightMetric = "residual"; renderWeightStudy(); });
byId("download-run").addEventListener("click", downloadShownData);
byId("retry-data").addEventListener("click", loadData);

scrubber.addEventListener("input", () => {
  state.inspectIndex = Number(scrubber.value);
  state.tooltipVisible = true;
  renderExplorer();
});
let pointerFrame = null;
let pointerClientX = 0;
trajectorySvg.addEventListener("pointermove", (event) => {
  pointerClientX = event.clientX;
  state.tooltipVisible = true;
  if (pointerFrame !== null) return;
  pointerFrame = requestAnimationFrame(() => {
    pointerFrame = null;
    if (!state.data || !state.tooltipVisible) return;
    const rect = trajectorySvg.getBoundingClientRect();
    const svgX = ((pointerClientX - rect.left) / rect.width) * 900;
    const marginLeft = Number(trajectorySvg.dataset.marginLeft || 72);
    const plotWidth = Number(trajectorySvg.dataset.plotWidth || 800);
    const ratio = Math.max(0, Math.min(1, (svgX - marginLeft) / plotWidth));
    const series = currentSeries();
    state.inspectIndex = Math.round(ratio * (series[0].points.length - 1));
    scrubber.value = String(state.inspectIndex);
    renderTrajectory(trajectorySvg, { series, showExact: state.showExact, view: state.view, inspectIndex: state.inspectIndex });
    updateInspector();
  });
});
trajectorySvg.addEventListener("pointerleave", () => {
  state.tooltipVisible = false;
  tooltip.hidden = true;
});
window.addEventListener("resize", () => {
  if (state.tooltipVisible) updateInspector();
});

loadData();
