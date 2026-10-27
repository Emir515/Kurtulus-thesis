"""Build the standalone 'Thesis Notebooks Explained' page.

1. Collects the figures the page refers to from the executed notebooks and outputs/ into reports/explainer/figs/
2. Writes notebooks_explained_standalone.html with every figure embedded (one file, about 10 MB)

Run after the notebooks:  python reports/explainer/build_explainer.py
Edit the text in notebooks_explained.html; figs/ and the standalone file are generated (not in git).
"""
import base64, io, json, os, re, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB = os.path.join(ROOT, 'notebooks')
A5 = os.path.join(ROOT, 'outputs', 'nb05_artifacts')
FIG = os.path.join(HERE, 'figs')
N1, N3, N4 = '01_data_exploration.ipynb', '03_dataset_analysis.ipynb', '04_anomaly_detection_cluster_analysis.ipynb'

# figure file -> (notebook, text that identifies the plotting cell, index of the image in that cell) or a saved PNG
SOURCES = {
    'nb01_timeline.png': (N1, 'Failure & Maintenance Timeline', 0),
    'nb01_sensors.png': (N1, 'Key Sensors (hourly means)', 0),
    'nb01_distributions.png': (N1, 'Normal vs Failure (train', 0),
    'nb01_correlation.png': (N1, 'Sensor Correlation Matrix', 0),
    'nb03_scores_year.png': (N3, 'Failure Probability (Jun 2024', 0),
    'nb03_train_test.png': (N3, 'RF Probability: Train vs Test', 0),
    'nb03_hotspots.png': (N3, 'RF Hotspots', 0),
    'nb03_separation.png': (N3, 'Feature Discriminability by Documented Sensor Group', 0),
    'nb03_within_group.png': (N3, 'Within-Group Correlation (different sensors)', 0),
    'nb03_aggregation.png': (N3, 'Mean |Correlation| Between Aggregation Types', 0),
    'nb03_per_sensor.png': (N3, 'Per-Sensor Aggregation Correlation', 0),
    'nb03_standardisation.png': (N3, 'Per-Channel Mean (after z-score', 0),
    'nb03_coverage.png': (N3, 'Temporal Coverage: Windows per Day', 0),
    'nb03_hour.png': (N3, 'RF Probability by Hour of Day', 0),
    'nb04_activation_count.png': (N4, 'Feature Activation Count: Normal vs Failure', 0),
    'nb04_thresholds.png': (N4, 'Threshold Sensitivity Analysis', 0),
    'nb04_monthly.png': (N4, 'Monthly anomaly rate: drift check', 0),
    'nb04_association.png': (N4, 'Statistical association: anomalies vs failures', 0),
    'nb04_coactivation_top40.png': (N4, 'Feature Co-Activation: Jaccard Similarity (top 40', 0),
    'nb04_coactivation.png': (N4, 'Jaccard Similarity Matrix Ordered by Feature Cluster', 0),
    'nb04_cohesion.png': (N4, 'Cluster Cohesion: How Tightly', 0),
    'nb04_heuristics.png': (N4, 'STATISTICAL COMPARISON', 0),
    'nb04_binary_continuous.png': (N4, 'Binary vs Continuous: Do binary groups', 0),
    'nb04_opstate.png': (N4, 'Operational state when each cluster fires', 0),
    'nb04_phases.png': (N4, 'Drift-controlled FC enrichment by phase', 0),
    'nb04_examples.png': (N4, 'Drift-controlled FC enrichment by phase', 1),
    'nb04_days.png': (N4, 'Feature Cluster Activation: All Failure Days vs All Normal Days', 0),
    'nb04_volatility.png': (N4, 'Volatility lead-up aligned to canonical failure onsets', 0),
    'nb04_drift.png': (N4, 'FC activation far from failures: train vs test', 0),
    'nb05_overview.png': 'fc_overview.png',
    'nb05_scorecard.png': 'fc_scorecard.png',
    'nb05_prior_alarms.png': 'prior_model_fc_enrichment.png',
    'nb05_failure_type.png': 'fc_failure_type.png',
    'nb05_timeline.png': 'fc_timeline.png',
    'nb05_deep_fc7.png': 'fc7_deep_dive.png',
    'nb05_deep_fc8.png': 'fc8_deep_dive.png',
    'nb05_raw_fc5.png': 'fc5_raw_signals.png',
    'nb05_raw_fc7.png': 'fc7_raw_signals.png',
    'nb05_raw_fc8.png': 'fc8_raw_signals.png',
    'nb05_rf_importance.png': 'rf_importance_by_fc.png',
    'nb05_air_system.png': 'daily_physics_timeline.png',
    'nb05_profiles.png': 'failure_profiles.png',
}

_nb_cache = {}


def cell_image(nb_file, key, k):
    if nb_file not in _nb_cache:
        _nb_cache[nb_file] = json.load(open(os.path.join(NB, nb_file), encoding='utf-8'))
    hits = [c for c in _nb_cache[nb_file]['cells'] if c['cell_type'] == 'code' and key in ''.join(c['source'])]
    if len(hits) != 1:
        sys.exit(f'{nb_file}: expected one cell containing "{key}", found {len(hits)}')
    imgs = [o['data']['image/png'] for o in hits[0].get('outputs', []) if 'data' in o and 'image/png' in o['data']]
    return base64.b64decode(imgs[k])


def save(png, name, max_w=1500):
    im = Image.open(io.BytesIO(png)).convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(os.path.join(FIG, name), optimize=True)


os.makedirs(FIG, exist_ok=True)
for name, src in SOURCES.items():
    png = cell_image(*src) if isinstance(src, tuple) else open(os.path.join(A5, src), 'rb').read()
    save(png, name)

page = open(os.path.join(HERE, 'notebooks_explained.html'), encoding='utf-8').read()
missing = set(re.findall(r'src="figs/([^"]+\.png)"', page)) - set(SOURCES)
if missing:
    sys.exit(f'figures referenced but not built: {sorted(missing)}')
page = re.sub(r'src="figs/([^"]+\.png)"',
              lambda m: 'src="data:image/png;base64,' + base64.b64encode(open(os.path.join(FIG, m.group(1)), 'rb').read()).decode() + '"',
              page)
out = os.path.join(HERE, 'notebooks_explained_standalone.html')
open(out, 'w', encoding='utf-8').write(page)
print(f'{len(SOURCES)} figures in reports/explainer/figs/, standalone page {os.path.getsize(out) / 1e6:.1f} MB')
