# Physics-Informed AI Lab website

A static, GitHub Pages-compatible experiment explorer built from the project's recorded artifacts. It uses HTML, CSS, vanilla JavaScript, and SVG charts; no backend, database, package installation, or build step is required.

## Repository placement

The website files belong in `docs/`. The data-export script belongs at `scripts/export_website_data.py`.

```text
docs/
├── index.html
├── styles.css
├── src/
│   ├── main.js
│   └── charts.js
├── data/
│   └── experiment_data.json
└── .nojekyll
scripts/
└── export_website_data.py
```

## Regenerate the website data

Run from the repository root after changing the experiment artifacts:

```powershell
python scripts\export_website_data.py
```

The exporter validates required columns, finite numeric values, unique runs, increasing trajectory time, and matching run/trajectory coverage. It does not alter experiment results.

## Preview locally

From the repository root:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/docs/`. Do not double-click `index.html`: browsers normally block the JSON request when the page is opened through `file://`.

Stop the server with `Ctrl+C`.

## Publish with GitHub Pages

1. Commit and push the `docs/` site and export script to `main`.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose branch **main**, folder **/docs**, and save.
5. Open `https://taibi1234.github.io/physics-informed-ai-lab/` after deployment finishes.

## Data sources

- `reports/advanced_damped_oscillator/damped_results.csv`
- `reports/advanced_damped_oscillator/damped_summary.csv`
- `reports/advanced_damped_oscillator/damped_predictions.csv`
- `reports/advanced_damped_oscillator/damped_config.json`
- `reports/weight_sweep/pinn_weight_summary.csv`
