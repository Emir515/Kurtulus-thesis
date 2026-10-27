"""Build the material for the Wiener Linien expert review from the current NB04/NB05 outputs.

Writes to outputs/expert_review/:
  WL_Expert_Review_NB04.pdf        13-page review document (clusters, evidence, raw signals, air system, questions)
  NB04_Workflow_and_Metrics.pdf    3-page backup: NB04 workflow and validation metrics
  expert_verification_sheet.xlsx   copy of the sheet from NB05 (the experts fill this one in)

Run after NB04 and NB05:  python reports/generate_expert_review.py
The filled sheet goes back to outputs/nb05_artifacts/expert_verification_sheet_FILLED.csv for NB05.
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.stats import fisher_exact

sys.stdout.reconfigure(encoding='utf-8')
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
A4, A5 = os.path.join(OUT, 'nb04_artifacts'), os.path.join(OUT, 'nb05_artifacts')
REVIEW = os.path.join(OUT, 'expert_review')
os.makedirs(REVIEW, exist_ok=True)

# ── Data ──
clusters = {int(k): v for k, v in json.load(open(os.path.join(A4, 'heuristic_a.json'))).items()}
fc_ids = sorted(clusters)
constants = json.load(open(os.path.join(A4, 'constants.json')))
all_sup = pd.read_parquet(os.path.join(A4, 'all_sup.parquet'))
all_sup['window_end'] = pd.to_datetime(all_sup['window_end'])
anom = pd.read_parquet(os.path.join(A4, 'anom_activations.parquet'))
anom_meta = pd.read_parquet(os.path.join(A4, 'anom_meta.parquet'))
idx = np.load(os.path.join(A4, 'anom_orig_idx.npy'))
anom.index = anom_meta.index = pd.Index(idx)
fail_events = pd.read_parquet(os.path.join(A4, 'fail_events.parquet'))
temporal = pd.read_csv(os.path.join(A4, 'fc_temporal.csv')).set_index('FC')
location = pd.read_csv(os.path.join(A4, 'fc_location.csv')).set_index('FC')
drift = pd.read_csv(os.path.join(A4, 'fc_drift.csv')).set_index('FC')
scores = pd.read_csv(os.path.join(A5, 'fc_scorecard.csv')).set_index('FC')
ftype = pd.read_csv(os.path.join(A5, 'fc_failure_type.csv'), index_col=0)
prior = {m: pd.read_csv(os.path.join(A5, f)).set_index('FC') for m, f in
         (('RF', 'prior_rf_fc_enrichment.csv'), ('LSTM-AE', 'prior_lstmae_fc_enrichment.csv'))}

n_win, n_anom = len(all_sup), len(anom)
n_anom_train = int((anom_meta['split'] == 'train').sum())
lab = anom_meta['label']
n_fail_anom, n_fail = int(lab.sum()), int(all_sup['label'].sum())
odds, _ = fisher_exact([[n_fail_anom, n_anom - n_fail_anom], [n_fail - n_fail_anom, (n_win - n_anom) - (n_fail - n_fail_anom)]], alternative='greater')
rate_anom, rate_non = n_fail_anom / n_anom, (n_fail - n_fail_anom) / (n_win - n_anom)

fc_rows = []
for fc in fc_ids:
    act = anom[clusters[fc]].any(axis=1)
    a, b = int((act & (lab == 1)).sum()), int((act & (lab == 0)).sum())
    _, p = fisher_exact([[a, b], [n_fail_anom - a, (n_anom - n_fail_anom) - b]], alternative='greater')
    fc_rows.append({'FC': f'FC{fc}', 'windows': int(act.sum()), 'fail_rate': 100 * a / max(a + b, 1),
                    'p_adj': min(1, p * len(fc_ids))})
fcs = pd.DataFrame(fc_rows).set_index('FC')
sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

ABBR = [('BRAKE_CYLINDER_PRESSURE', 'BC_P'), ('SPRING_BRAKE_PRESSURE', 'SB_P'), ('SPRING_BRAKE_ACTIVE', 'SB_ACT'),
        ('PROPORTIONAL_VALVE_PRESSURE', 'PV_P'), ('PNEUMATIC_BRAKING_FORCE', 'PBF'), ('LOAD_PRESSURE', 'LP'),
        ('LOAD_SIGNAL', 'LSIG'), ('MAIN_RESERVOIR_PRESSURE', 'MR_P'), ('ENERGY_BRAKING_RESISTANCE', 'EBR'),
        ('COMPRESSOR_RUNNING', 'COMP_RUN'), ('_BOGIE1', '_B1'), ('_BOGIE2', '_B2')]


def short(name):
    for a, b in ABBR:
        name = name.replace(a, b)
    return name


def sensors_of(fc):
    out = []
    for f in clusters[fc]:
        base = f.rsplit('_', 1)[0] if f.rsplit('_', 1)[-1] in ('min', 'max', 'mean', 'flips', 's') else f
        base = base.replace('_active', '')
        if base not in out:
            out.append(base)
    return out


NAVY, GREY, RED, ORANGE, GREEN, TEAL = '#1f3b57', '#6b7280', '#c0392b', '#e67e22', '#2e8b57', '#17becf'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
FC_RANGE = f'FC1-FC{max(fc_ids)}'


def title(fig, main, sub=None):
    fig.text(0.04, 0.95, main, fontsize=17, fontweight='bold', color=NAVY, va='top')
    if sub:
        fig.text(0.04, 0.905, sub, fontsize=10.5, color=GREY, va='top')


def footer(fig, n):
    fig.text(0.04, 0.025, f'Emirhan Kurtulus, TU Wien, Master Thesis | Feature clusters {FC_RANGE} are data-driven and carry no subsystem label',
             fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, str(n), fontsize=8, color=GREY, ha='right')


def text_block(fig, x, y, lines, size=10, gap=0.034):
    for item in lines:
        if isinstance(item, tuple):
            txt, style = item
        else:
            txt, style = item, {}
        fig.text(x, y, txt, fontsize=style.get('size', size), fontweight=style.get('weight', 'normal'),
                 color=style.get('color', 'black'), va='top')
        y -= style.get('gap', gap)
    return y


pattern_col = {'early-warning': GREEN, 'short-term precursor': TEAL, 'detection-only': RED,
               'weak / mixed': ORANGE, 'background': GREY}
keep = scores[scores.verdict == 'KEEP'].sort_values('lift_train', ascending=False)
robust = keep[keep.test_check == 'holds']
train_only = keep[keep.test_check != 'holds']
precursor = temporal[temporal.pattern == 'short-term precursor'].index.tolist()
detection = temporal[temporal.pattern == 'detection-only'].index.tolist()
drift_fcs = drift[drift.drift_flag].index.tolist()
both_periods = ['FC2', 'FC6']  # NB04 §4.4 stratified contrast (> 5 pp in train and test)
n_tr, n_te = int((fail_events.split == 'train').sum()), int((fail_events.split == 'test').sum())
types = fail_events.event_type.value_counts().to_dict()


def verdict_text(k):
    v = scores.loc[k, 'verdict']
    return f'{v} / {scores.loc[k, "test_check"]}' if v == 'KEEP' else v


# ════════════════════════ Expert review PDF ════════════════════════
LAND = (11.69, 8.27)
page = 0
with PdfPages(os.path.join(REVIEW, 'WL_Expert_Review_NB04.pdf')) as pdf:
    # Page: overview
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Predictive Maintenance: Feature Cluster Analysis', 'Expert review document for Wiener Linien | Emirhan Kurtulus, TU Wien, Master Thesis')
    text_block(fig, 0.05, 0.84, [
        ('OVERVIEW', {'weight': 'bold', 'color': NAVY, 'size': 12, 'gap': 0.045}),
        f'Data: 101 sensor channels of one MetroAT vehicle, June 2024 - June 2025, aggregated into {n_win:,} windows of 6 min',
        f'Recorded failures: {len(fail_events)} events ({n_tr} in the training period until 11 Feb 2025, {n_te} in the test period after): '
        f'{types.get("Brake System Failure", 0)} brake, {types.get("Compressor Module Failure", 0)} compressor, {types.get("Leveling System Failure", 0)} leveling',
        f'Anomaly detection: a sensor feature is "activated" if |z| > {constants["Z_THRESH"]:.0f} (vs. normal train data); '
        f'a window is anomalous if >= {constants["MIN_ACTIVATED"]} features are activated',
        f'Anomalous windows: {n_anom:,} ({100 * n_anom / n_win:.1f}%), {odds:.0f}x more likely to be failure-related than other windows',
        f'Feature clusters: {len(fc_ids)} ({FC_RANGE}), {sum(map(len, clusters.values()))} features that fire together, '
        f'found on the {n_anom_train:,} anomalous windows of the training period only',
        ('', {'gap': 0.02}),
        ('IMPORTANT', {'weight': 'bold', 'color': NAVY, 'size': 12, 'gap': 0.045}),
        'The clusters are found from the data alone. They are NOT assigned to subsystems by us:',
        'deciding which subsystem a cluster belongs to, and whether it makes physical sense, is the purpose of this review.',
        'The test period (from 12 Feb 2025) was not used to form or select the clusters; it only serves as a check.',
        ('', {'gap': 0.02}),
        ('WHAT WE ASK YOU', {'weight': 'bold', 'color': NAVY, 'size': 12, 'gap': 0.045}),
        '1. Do the sensors in each cluster belong together physically? (full sensor lists on page 3)',
        '2. Which subsystem does each cluster belong to?',
        '3. Which clusters correspond to known failure modes?',
        '4. Are clusters that mostly appear in false alarms (page 6) normal operating patterns, e.g. braking manoeuvres?',
        '5. Which clusters should be kept for building subsystem-specific datasets?',
        ('', {'gap': 0.02}),
        ('Answers go into the accompanying sheet: expert_verification_sheet.xlsx (one row per cluster).', {'color': GREY}),
    ])
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: cluster table
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, f'All {len(fc_ids)} Feature Clusters', 'Sensors, location, failure association, temporal pattern and event-level verdict')
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.75]); ax.axis('off')
    rows, colors = [], []
    for fc in fc_ids:
        k = f'FC{fc}'
        sens = sensors_of(fc)
        side = 'A' if location.loc[k, 'side_A_share'] == 1 else 'B' if location.loc[k, 'side_B_share'] == 1 else 'A+B'
        rows.append([k, len(clusters[fc]), len(sens), ', '.join(short(s) for s in sens[:3]) + (' ...' if len(sens) > 3 else ''),
                     side, f'{fcs.loc[k, "fail_rate"]:.1f}% {sig(fcs.loc[k, "p_adj"])}', temporal.loc[k, 'pattern'],
                     verdict_text(k), 'yes' if drift.loc[k, 'drift_flag'] else ''])
        colors.append(pattern_col[temporal.loc[k, 'pattern']])
    tab = ax.table(cellText=rows, colLabels=['FC', 'Feats', 'Sensors', 'Main sensors', 'Chain', 'Fail rate',
                                             'Temporal pattern', 'Verdict / test', 'Drift'],
                   colWidths=[0.05, 0.05, 0.06, 0.38, 0.05, 0.09, 0.14, 0.10, 0.05], loc='upper center', cellLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(7.4); tab.scale(1, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        if r == 0:
            cell.set_facecolor(NAVY); cell.get_text().set_color('white'); cell.get_text().set_fontweight('bold')
        elif c == 6:
            cell.get_text().set_color(colors[r - 1]); cell.get_text().set_fontweight('bold')
        elif c == 7 and rows[r - 1][7] == 'KEEP / holds':
            cell.get_text().set_fontweight('bold'); cell.set_facecolor('#e8f5ee')
    fig.text(0.04, 0.085, f'Fail rate = share of failure-labelled windows among the anomalous windows where the FC fires '
             f'(baseline {100 * rate_anom:.1f}%); *** p<0.001, ** p<0.01, * p<0.05 (Bonferroni).  Chain A = CW1/MW1/MW2, B = CW2/MW3/MW4.',
             fontsize=6.8, color=GREY)
    fig.text(0.04, 0.068, f'Verdict: KEEP if the cluster, used as an alarm, covers >= 3 of the {n_tr} training-period failures at >= 1.5x the chance rate; '
             f'"holds" = the same rule is also met on the {n_te} test-period failures (highlighted).',
             fontsize=6.8, color=GREY)
    fig.text(0.04, 0.051, 'Abbreviations: BC_P brake cylinder, SB spring brake, PV proportional valve, PBF pneumatic braking force, LP load pressure, '
             'LSIG load signal, MR main reservoir, EBR energy braking resistance, COMP_RUN compressor running, B1/B2 bogie.',
             fontsize=6.8, color=GREY)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: full sensor lists
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Full Sensor List per Cluster', 'All features of each cluster (sensor name + window statistic) for question 1: do they belong together?')
    lines = []
    for fc in fc_ids:
        by_sensor = {}
        for f in clusters[fc]:
            stat = f.rsplit('_', 1)[-1]
            base = f.rsplit('_', 1)[0] if stat in ('min', 'max', 'mean', 'flips', 's') else f
            if stat == 's':
                base, stat = base.replace('_active', ''), 'active_s'
            by_sensor.setdefault(base, []).append(stat)
        lines.append((f'FC{fc}', '; '.join(f'{short(s)} ({"/".join(v)})' for s, v in by_sensor.items())))
    col_x, y0, y = [0.04, 0.52], 0.86, None
    half = (len(lines) + 1) // 2
    for col, chunk in enumerate((lines[:half], lines[half:])):
        y = y0
        for k, txt in chunk:
            fig.text(col_x[col], y, k, fontsize=8.2, fontweight='bold', color=NAVY, va='top')
            wrapped, cur = [], ''
            for part in txt.split('; '):
                if len(cur) + len(part) > 70 and cur:
                    wrapped.append(cur); cur = part
                else:
                    cur = f'{cur}; {part}' if cur else part
            wrapped.append(cur)
            fig.text(col_x[col] + 0.04, y, '\n'.join(wrapped), fontsize=6.8, va='top', linespacing=1.35)
            y -= 0.022 + 0.0185 * len(wrapped)
    fig.text(0.04, 0.05, 'Statistic per 6-min window: min / max / mean of an analog signal; flips = number of switches and active_s = seconds active of an on/off signal.',
             fontsize=7, color=GREY)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: anomalies vs failures
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Are Anomalies Linked to Failures?', 'Statistical evidence')
    ax = fig.add_axes([0.06, 0.15, 0.27, 0.65])
    vals = [100 * rate_anom, 100 * rate_non, 100 * n_fail / n_win]
    ax.bar(['Anomalous\nwindows', 'Other\nwindows', 'All\nwindows'], vals, color=[RED, GREY, NAVY])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Failure-labelled windows (%)'); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'Odds ratio {odds:.1f} (Fisher, p < 1e-200)\n{n_fail_anom} of {n_fail} failure windows ({100 * n_fail_anom / n_fail:.0f}%) are anomalous',
                 fontsize=9.5)
    ax = fig.add_axes([0.40, 0.15, 0.57, 0.65])
    fr = fcs['fail_rate']
    ax.bar(fr.index, fr.values, color=[RED if p < 0.001 else ORANGE if p < 0.05 else GREY for p in fcs['p_adj']])
    ax.axhline(100 * rate_anom, color='black', ls='--', lw=1)
    ax.text(0.99, 0.97, f'dashed line = baseline {100 * rate_anom:.1f}%', transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.set_xticks(range(len(fr))); ax.set_xticklabels(fr.index, rotation=90, fontsize=8)
    ax.set_ylabel('Failure rate (%)'); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Failure rate per cluster (red p<0.001, orange p<0.05, grey not significant)', fontsize=9.5)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: temporal patterns
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'When Do Clusters Fire Relative to a Failure?',
          'Activation rate relative to normal periods of the same period (train/test), which corrects for the test-period drift')
    ax = fig.add_axes([0.06, 0.14, 0.40, 0.68])
    enr = temporal[['enrich_early', 'enrich_imminent', 'enrich_during']].astype(float)
    im = ax.imshow(np.log2(enr.clip(1 / 16, 16)).values, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    for i in range(len(enr)):
        for j in range(3):
            v = enr.iloc[i, j]
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(np.log2(np.clip(v, 1 / 16, 16))) > 2 else 'black')
    ax.set_yticks(range(len(enr))); ax.set_yticklabels(enr.index, fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['6 d to 1 d\nbefore', 'last 24 h\nbefore', 'during\nfailure'])
    fig.colorbar(im, ax=ax, fraction=0.04, label='log2 ratio to normal rate')
    ax = fig.add_axes([0.56, 0.14, 0.40, 0.66])
    d = temporal.assign(excess=temporal.events_24h - temporal.expected_24h).sort_values('excess', ascending=False)
    ax.barh(d.index, d.events_24h, color=[pattern_col[p] for p in d.pattern], alpha=0.85)
    ax.scatter(d.expected_24h, d.index, marker='|', s=160, color='black')
    ax.invert_yaxis(); ax.tick_params(axis='y', labelsize=7.5)
    ax.set_xlabel(f'Failures with a cluster activation in the 24 h before onset (of {len(fail_events)}); | = expected by chance')
    ax.legend(handles=[Patch(color=c, label=p) for p, c in pattern_col.items()], fontsize=7, loc='lower left',
              bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.06, 0.075, 'No cluster fires significantly more often 1-6 days before failures (no multi-day early warning).',
             fontsize=9, color=NAVY)
    fig.text(0.06, 0.05, f'{", ".join(precursor) or "None"}: short-term precursors (24 h).   '
             f'{len(detection)} clusters fire mainly DURING failures (detection).', fontsize=9, color=NAVY)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: prior-model alarms
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Which Clusters Sit Behind the Existing Models\' Alarms?',
          'Alarms of the Random Forest and LSTM autoencoder baselines (test period): correct early warnings vs. false alarms')
    for k, (m, t) in enumerate(prior.items()):
        ax = fig.add_axes([0.07 + 0.47 * k, 0.14, 0.38, 0.69])
        y = np.arange(len(t))
        ax.barh(y - 0.2, t['rate_in_early_hits_%'], 0.4, color=GREEN, label='in correct early warnings')
        ax.barh(y + 0.2, t['rate_in_false_alarms_%'], 0.4, color=RED, label='in false alarms')
        for i, c in enumerate(t['characterises'].fillna('')):
            if c:
                ax.text(max(t.iloc[i][['rate_in_early_hits_%', 'rate_in_false_alarms_%']]) + 1, i, c, va='center', fontsize=6.8,
                        color=GREEN if c == 'early hits' else RED, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7.5); ax.invert_yaxis()
        ax.set_xlabel('% of alarm windows in which the cluster is active'); ax.set_title(m, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.07, 0.06, 'Labels mark clusters significantly over-represented (Fisher test, Bonferroni-corrected).',
             fontsize=8.5, color=GREY)
    fig.text(0.07, 0.042, 'Clusters typical for false alarms may describe normal operation; clusters typical for correct warnings '
             'may describe real degradation.', fontsize=8.5, color=GREY)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: failure types
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Clusters vs. Failure Types Recorded by Wiener Linien',
          'Activation in the 24 h before and during each failure type, relative to normal periods (1 = no difference)')
    ax = fig.add_axes([0.12, 0.10, 0.45, 0.76])
    ft = ftype.astype(float)
    im = ax.imshow(np.log2(ft.clip(1 / 32, 32)).values, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    for i in range(len(ft)):
        for j in range(ft.shape[1]):
            v = ft.iloc[i, j]
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(np.log2(np.clip(v, 1 / 32, 32))) > 2.7 else 'black')
    ax.set_yticks(range(len(ft))); ax.set_yticklabels(ft.index, fontsize=7.5)
    ax.set_xticks(range(ft.shape[1])); ax.set_xticklabels([c.replace(' Failure', '\nfailure') for c in ft.columns], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label='log2 ratio')
    text_block(fig, 0.64, 0.80, [
        ('How to read', {'weight': 'bold', 'color': NAVY, 'size': 11}),
        'Each cell: how much more often a cluster fires',
        'around failures of that type than in normal',
        'operation. Failure types are taken from the',
        'Wiener Linien records, not from the clusters.',
        ('', {'gap': 0.02}),
        ('Caution', {'weight': 'bold', 'color': NAVY, 'size': 11}),
        f'Only {types.get("Leveling System Failure", 0)} leveling and {types.get("Compressor Module Failure", 0)} compressor failures:',
        'values for these types rest on few events.',
    ], size=9.5)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: example traces (one per test-robust cluster, most robust first)
    examples = [(int(k[2:]), str(temporal.loc[k, 'pattern']).upper(), pattern_col[temporal.loc[k, 'pattern']]) for k in robust.index[:3]]
    train_normal = all_sup[(all_sup.split == 'train') & (all_sup.label == 0)]
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Examples Around a Recorded Failure',
          'The clusters that also hold on the test period; z-scores of the cluster features, markers = windows in which the cluster fires')
    n_ex = max(len(examples), 1)
    h = 0.74 / n_ex
    for k, (fc, name, col) in enumerate(examples):
        ax = fig.add_axes([0.06, 0.86 - h * (k + 1) + 0.035, 0.90, h - 0.07])
        feats = clusters[fc]
        act = pd.Series(False, index=all_sup.index)
        act.loc[idx] = anom[feats].any(axis=1).values
        best = max(fail_events.index, key=lambda e: int((act & (all_sup.ev_id == e) & all_sup.phase.isin(['early', 'imminent', 'during'])).sum()))
        ev = fail_events.loc[best]
        view = (all_sup.window_end >= ev.start - pd.Timedelta(days=3)) & (all_sup.window_end <= ev.end + pd.Timedelta(days=1))
        t_view = all_sup.loc[view, 'window_end']
        gap = t_view.diff() > pd.Timedelta(hours=1)  # no line across operational breaks
        for f in feats[:6]:
            z = ((all_sup.loc[view, f] - train_normal[f].mean()) / (train_normal[f].std() or 1)).where(~gap)
            ax.plot(t_view, z, lw=0.8, alpha=0.8, label=short(f))
        ax.axvspan(ev.start - pd.Timedelta(hours=24), ev.start, color='gold', alpha=0.15, label='last 24 h')
        ax.axvspan(ev.start, ev.end, color='red', alpha=0.15, label='recorded failure')
        fires = all_sup.loc[view & act, 'window_end']
        ax.scatter(fires, np.full(len(fires), ax.get_ylim()[1] * 0.9), marker='v', s=16, color=col)
        ax.set_title(f'[{name}] FC{fc}: {ev.event_type} on {ev.start:%Y-%m-%d %H:%M}', fontsize=9.5, fontweight='bold', color=col, loc='left')
        ax.set_ylabel('z-score', fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=4, loc='lower left'); ax.spines[['top', 'right']].set_visible(False)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Page: values behind the robust clusters (NB05 7.1)
    top10 = pd.read_csv(os.path.join(A5, 'fc_top10_features.csv'))
    vals = top10[top10.FC.isin(robust.index)]
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Values Behind the Robust Clusters',
          'Top features of the clusters that hold on the test period: normal range vs. value when the cluster fires')
    ax = fig.add_axes([0.04, 0.22, 0.92, 0.64]); ax.axis('off')
    rows = [[r.FC, short(r.feature), f'{100 * r.active_share:.0f}%', f'{r.normal_median:.3g}',
             f'{r.normal_p05:.3g} - {r.normal_p95:.3g}', f'{r.fc_active_median:.3g}', f'{r.shift_in_std:+.1f}']
            for r in vals.itertuples()]
    tab = ax.table(cellText=rows, colLabels=['FC', 'Feature', 'Active when FC fires', 'Normal median', 'Normal range (5-95%)',
                                             'Median when FC fires', 'Shift (std)'],
                   colWidths=[0.06, 0.30, 0.13, 0.11, 0.15, 0.13, 0.09], loc='upper center', cellLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.45)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        if r == 0:
            cell.set_facecolor(NAVY); cell.get_text().set_color('white'); cell.get_text().set_fontweight('bold')
    text_block(fig, 0.05, 0.46, [
        'Normal = training windows without a failure label (the reference for the anomaly detection). Values are in the scaled units of the public MetroAT data;',
        'the shift column gives the difference in normal standard deviations (negative = lower than normal). All 19 clusters: fc_top10_features.csv.',
    ], size=8.5, gap=0.026)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

    # Pages: raw 1-second signals (NB05 7.2)
    raw_imgs = {k: os.path.join(A5, f'{k.lower()}_raw_signals.png') for k in robust.index}
    raw_imgs = {k: v for k, v in raw_imgs.items() if os.path.exists(v)}
    raw_note = ('Raw 1 Hz signals read back from the MetroAT files, from 24 h before onset to the end of the failure.' + chr(10) +
                'Orange markers = 6-min windows where the cluster fires; red = recorded failure; gaps = no data (train not recording). ' + chr(10) + 'FC7 and FC8 fire mostly while standing: low spring-brake pressure while standing can also mean the parking brake is applied.')
    imgs = {k: plt.imread(v) for k, v in raw_imgs.items()}
    keys = sorted(imgs, key=lambda k: imgs[k].shape[0] / imgs[k].shape[1])  # widest (flattest) image first
    for chunk in (keys[:2], keys[2:]):
        if not chunk:
            continue
        page += 1
        fig = plt.figure(figsize=LAND)
        title(fig, 'Raw 1-Second Signals Before a Failure', 'Back to the underlying values of the robust clusters')
        rects = [[0.04, 0.67, 0.92, 0.20], [0.04, 0.12, 0.92, 0.52]] if len(chunk) == 2 else [[0.04, 0.115, 0.92, 0.76]]
        for k, rect in zip(chunk, rects):
            ax = fig.add_axes(rect); ax.imshow(imgs[k]); ax.axis('off')
        fig.text(0.04, 0.098, raw_note, fontsize=7.8, color=GREY, linespacing=1.5, va='top')
        footer(fig, page)
        pdf.savefig(fig); plt.close(fig)

    # Page: daily compressor work and air-pressure rates (NB05 7.4 / 7.4b)
    phys_png = os.path.join(A5, 'daily_physics_timeline.png')
    if os.path.exists(phys_png):
        page += 1
        fig = plt.figure(figsize=LAND)
        title(fig, 'Compressor Work and Air Pressure Over the Whole Year',
              'Daily values computed from the raw 1-second data, per side (CW1 solid, CW2 dashed)')
        ax = fig.add_axes([0.04, 0.17, 0.92, 0.70]); ax.imshow(plt.imread(phys_png)); ax.axis('off')
        text_block(fig, 0.05, 0.155, [
            'Over the training months the pressure loss when standing (bottom) rises every month (0.0009 to 0.0046 per minute), and the CW2 side fills more slowly',
            'while its compressor runs more. After the 8-week revision of 17-22 April 2025 CW2 fills normally again, but CW1 fills more slowly; compressor failures',
            'follow on 18 April, 5 and 13 May. In the days just before single failures these quantities show no consistent change (event-aligned profiles, NB05 7.5).',
        ], size=8.4, gap=0.024)
        footer(fig, page)
        pdf.savefig(fig); plt.close(fig)

    # Page: summary
    page += 1
    fig = plt.figure(figsize=LAND)
    title(fig, 'Summary & Questions for Expert Review')
    robust_lines = [f'   {k}: {", ".join(short(s) for s in sensors_of(int(k[2:]))[:3])}{" ..." if len(sensors_of(int(k[2:]))) > 3 else ""} | '
                    f'{r.lift_train:.1f}x chance on train failures, {r.lift_test:.1f}x on test failures, {int(r.false_alarms)} false alarms'
                    for k, r in robust.iterrows()]
    text_block(fig, 0.05, 0.86, [
        ('OBSERVATIONS (not conclusions)', {'weight': 'bold', 'color': NAVY, 'size': 12, 'gap': 0.042}),
        f'1. {len(fc_ids)} clusters from {sum(map(len, clusters.values()))} features; {int((fcs.p_adj < 0.05).sum())} are significantly linked to failure-labelled windows.',
        '2. No cluster gives a multi-day early warning (1-6 days before). '
        + (f'{" and ".join(precursor)} often fire in the last 24 h before a failure (short-term precursors).' if precursor else ''),
        f'3. {len(detection)} clusters fire mainly during failures (useful for diagnosis, not prediction).',
        f'4. Only {" and ".join(both_periods)} are more active on failure days in both the training and the test period.',
        f'5. {len(keep)} clusters pass the event-level check on the training-period failures, but only {len(robust)} also hold on the test period:',
        *robust_lines,
        f'6. The other {len(train_only)} drop to chance level in the test period; {int(train_only.drift_flag.sum())} of them are flagged as possibly drift-driven.',
        f'7. {len(drift_fcs)} clusters became much more active in 2025 without failures nearby (possible drift).',
        ('', {'gap': 0.015}),
        ('QUESTIONS FOR WIENER LINIEN', {'weight': 'bold', 'color': NAVY, 'size': 12, 'gap': 0.042}),
        '1. Do the sensors in each cluster belong together physically (page 3)?',
        '2. Which subsystem does each cluster belong to?',
        '3. Which clusters correspond to known failure modes (page 7)? Is it plausible that '
        + (f'{" / ".join(robust.index)} change before failures?' if len(robust) else 'they change before failures?'),
        '4. Are clusters typical for false alarms (page 6) normal operating patterns, e.g. braking manoeuvres?',
        '5. Was anything changed on the vehicle in spring 2025 that could explain the drift?',
        '6. Which clusters should be kept for the subsystem datasets?',
        '7. FC7 and FC8: spring-brake pressure drops to about 0 (raw signals pages), but both fire mostly while the train stands (88% / 92%).',
        '    Is this a real pressure loss, a signal dropout of the car, or simply the parking (spring) brake applied, e.g. a train parked because of the fault?',
        '8. Air system over the year (page before): is the growing pressure loss a known leak? Was the CW2 compressor repaired or exchanged',
        '    in the April 2025 revision, and could the CW1 side have degraded after it?',
    ], size=9.4, gap=0.0285)
    footer(fig, page)
    pdf.savefig(fig); plt.close(fig)

# ════════════════════════ Workflow PDF ════════════════════════
PORT = (8.27, 11.69)


def box(ax, x, y, w, h, head, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=color, ec='none', alpha=0.13))
    ax.add_patch(FancyBboxPatch((x, y), 0.008, h, boxstyle='square,pad=0', fc=color, ec='none'))
    ax.text(x + 0.022, y + h - 0.012, head, fontsize=9.5, fontweight='bold', color=color, va='top')
    ax.text(x + 0.022, y + h - 0.036, body, fontsize=7.8, va='top', linespacing=1.45)


# Heuristic A vs B metrics from the NB04 run of 2026-10-27 (§3.1 output)
hA = {'clusters': len(fc_ids), 'sil': 0.547, 'ari': 0.457, 'nmi': 0.769, 'ami': 0.689, 'co': 87.3, 'maxfr': 62.7, 'spread': 13.14}
hB = {'clusters': 13, 'sil': 0.529, 'maxfr': 14.7, 'spread': 3.29}
n_feat = len(json.load(open(os.path.join(A4, 'feature_cols.json'))))
with PdfPages(os.path.join(REVIEW, 'NB04_Workflow_and_Metrics.pdf')) as pdf:
    fig = plt.figure(figsize=PORT)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.965, 'NB04: Anomaly Detection & Feature Clustering', fontsize=16, fontweight='bold', color=NAVY, va='top')
    ax.text(0.06, 0.94, 'Step-by-step workflow', fontsize=10.5, color=GREY, va='top')
    steps = [
        ('INPUT', f'{n_win:,} windows (6 min) x {n_feat} sensor features from NB02a (_sum dropped; speed, temperature\nand days-since features excluded from detection)', NAVY),
        ('STEP 0: Canonical failure events', f'{len(fail_events)} events from the raw failure flag ({n_tr} train / {n_te} test), shared with NB05.\n'
         'Each window gets a phase: early (6 d to 1 d before), imminent (last 24 h), during, normal', NAVY),
        ('STEP 1 (Sec. 1): Z-score anomaly detection', f'z = (value - train-normal mean) / train-normal std  |  activated if |z| > 3  |  anomalous if >= 5\n'
         f'{n_win:,} windows -> {n_anom:,} anomalous ({100 * n_anom / n_win:.1f}%), {n_anom_train:,} of them in the training period', '#2563eb'),
        ('STEP 2 (Sec. 2): Binary activation matrix', f'{n_anom_train:,} TRAIN-period anomalous windows x features, 1 = feature deviates beyond 3 std\n'
         f'features active in >= 1% of these windows: {sum(map(len, clusters.values()))}', '#2563eb'),
        ('STEP 3 (Sec. 3): Feature clustering (train period only)', '"Which sensor features fire TOGETHER?"  Jaccard distance between columns,\n'
         f'average linkage, cut t = 0.7 -> {len(fc_ids)} feature clusters ({FC_RANGE}); the test period is never used', '#7c3aed'),
        ('STEP 4 (Sec. 3.1-3.2): Validate the clusters', 'Cohesion, same-sensor redundancy, sensor location (control chain, bogie),\n'
         f'Jaccard vs. correlation clustering (agreement {hA["co"]:.0f}%), binary vs. continuous structure', '#7c3aed'),
        ('STEP 5 (Sec. 4.1-4.3): Failure association & temporal pattern', 'Fisher tests; drift-controlled enrichment per phase; event-level test against chance\n'
         f'-> 0 early-warning, {len(precursor)} short-term precursor, {len(detection)} detection-only clusters', RED),
        ('STEP 6 (Sec. 4.4-4.6): Failure days, volatility, drift', 'Failure-related vs. normal days (stratified by period); within-day variance\n'
         f'before onset (Kendall tau); per-cluster drift flag ({len(drift_fcs)} clusters)', RED),
        ('OUTPUT -> NB05', f'{len(fc_ids)} data-driven feature clusters + temporal, location and drift tables.\n'
         'NB05: event-level scoring (verdict on train events, test check), expert sheet (RQ1) -> subsystem datasets (RQ2)', GREEN),
    ]
    y = 0.885
    for head, body, col in steps:
        box(ax, 0.06, y - 0.075, 0.88, 0.075, head, body, col)
        y -= 0.093
    ax.text(0.06, 0.042, 'Clusters are derived from sensor co-activation only. No subsystem knowledge is used;',
            fontsize=7.8, color=GREY)
    ax.text(0.06, 0.028, 'subsystem assignment is done by the Wiener Linien experts.', fontsize=7.8, color=GREY)
    pdf.savefig(fig); plt.close(fig)

    metrics = [
        ('1. Silhouette score', 's(i) = [b(i) - a(i)] / max(a(i), b(i));  a = mean distance to own cluster, b = to nearest other cluster',
         '-1 to +1, higher = better separation', f'A: {hA["sil"]:.3f} | B: {hB["sil"]:.3f} -> both well separated, A slightly tighter'),
        ('2. Adjusted Rand Index (ARI)', 'ARI = (RI - E[RI]) / (max(RI) - E[RI]);  RI = share of feature pairs treated alike by both methods',
         '-1 to +1, 1 = identical, 0 = random', f'ARI = {hA["ari"]:.3f}: agreement well beyond chance'),
        ('3. Normalised / adjusted mutual information', 'NMI = 2 I(A;B) / [H(A) + H(B)];  AMI corrects NMI for chance',
         '0 to 1, 1 = identical', f'NMI = {hA["nmi"]:.3f}, AMI = {hA["ami"]:.3f}: core structure shared'),
        ('4. Pairwise co-membership agreement', 'share of feature pairs that are either together in both methods or apart in both',
         '0-100%', f'{hA["co"]:.1f}%: almost 9 of 10 feature pairs grouped identically'),
        ('5. Failure-rate spread and maximum', 'std of the cluster failure rates; highest failure rate of any cluster',
         'higher = better separation of failure-prone and benign clusters', f'spread A {hA["spread"]:.2f} vs. B {hB["spread"]:.2f}; max A {hA["maxfr"]:.1f}% vs. B {hB["maxfr"]:.1f}%'),
        ('6. Phase enrichment (drift-controlled)', 'activations in a phase / activations expected at the normal rate of the SAME period (train or test)',
         '1 = no difference; > 1.5 counted as enriched', 'corrects for the test period, where all clusters fire more often'),
        ('7. Event-level test', 'number of failures with an activation in a phase vs. number expected by chance (Poisson-binomial tail)',
         'p < 0.05 counted as significant', 'robust to autocorrelation within an event and to failure-type-specific clusters'),
        ('8. Volatility lead-up', 'within-day std of z-scored cluster features, days -3 to -1 before onset / same-period normal days',
         '1 = no change; Kendall tau over days -7 to -1 = trend', 'only FC18: 1.25x with tau 0.71 (p 0.03); FC2 rises only on the onset day'),
    ]
    fig = plt.figure(figsize=PORT)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.965, 'Metrics Explained', fontsize=16, fontweight='bold', color=NAVY, va='top')
    ax.text(0.06, 0.94, 'How the clusters and their temporal patterns are validated', fontsize=10.5, color=GREY, va='top')
    y = 0.90
    for head, formula, rng, result in metrics:
        box(ax, 0.06, y - 0.092, 0.88, 0.092, head, f'Formula: {formula}\nRange: {rng}\nResult: {result}', NAVY)
        y -= 0.106
    pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=PORT)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.965, 'Conclusion', fontsize=16, fontweight='bold', color=NAVY, va='top')
    tab_ax = fig.add_axes([0.08, 0.66, 0.84, 0.25]); tab_ax.axis('off')
    rows = [['Clusters', hA['clusters'], hB['clusters'], '-'], ['Silhouette', hA['sil'], hB['sil'], 'A'],
            ['ARI / NMI (agreement)', f'{hA["ari"]} / {hA["nmi"]}', '', 'agree'], ['Co-membership', f'{hA["co"]}%', '', 'agree'],
            ['Max failure rate', f'{hA["maxfr"]}%', f'{hB["maxfr"]}%', 'A'], ['Failure-rate spread', hA['spread'], hB['spread'], 'A']]
    t = tab_ax.table(cellText=rows, colLabels=['Metric', 'A (Jaccard)', 'B (correlation)', 'Winner'],
                     loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d1d5db')
        if r == 0:
            cell.set_facecolor(NAVY); cell.get_text().set_color('white'); cell.get_text().set_fontweight('bold')
    y = 0.60
    for head, body, col in [
        ('CLUSTERING: use Heuristic A (Jaccard / binary)', f'Higher silhouette and much better separation of failure-prone and benign clusters; {hA["co"]:.0f}% agreement\n'
         f'with the correlation-based grouping, so binary activation keeps the structure; finer granularity ({hA["clusters"]} vs. {hB["clusters"]}).', '#7c3aed'),
        ('TEMPORAL PATTERN: detection rather than multi-day prediction', f'After correcting for drift, no cluster fires more often 1-6 days before failures.\n'
         f'{" and ".join(precursor) or "No cluster"} are short-term (24 h) precursors; {len(detection)} clusters mainly fire during failures.', RED),
        ('DRIFT: compare within periods', f'Anomaly rate far from failures: 10.9% (train) vs. 27.4% (test). {len(drift_fcs)} clusters are flagged\n'
         'as possibly drift-driven; pooled train+test comparisons overstate failure effects.', ORANGE),
        ('NEXT: NB05 and expert verification', f'{len(fc_ids)} clusters, no subsystem labels -> event-level scoring (only {", ".join(robust.index)} hold on the test period),\n'
         'prior-model alarms, expert verification sheet (RQ1) -> subsystem datasets for the model comparison (RQ2).', GREEN),
    ]:
        box(ax, 0.06, y - 0.085, 0.88, 0.085, head, body, col)
        y -= 0.1
    pdf.savefig(fig); plt.close(fig)

for f in ('expert_verification_sheet.xlsx', 'expert_verification_sheet.csv'):
    if os.path.exists(os.path.join(A5, f)):
        shutil.copy2(os.path.join(A5, f), os.path.join(REVIEW, f))
print(f'Wrote outputs/expert_review/: WL_Expert_Review_NB04.pdf ({page} pages), NB04_Workflow_and_Metrics.pdf (3 pages), expert sheet copy')
