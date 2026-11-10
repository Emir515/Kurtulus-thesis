"""Short progress update for the supervisor: one page per discussed point, with the key figure and 2-3 lines.

Same look as the expert review document. Built from the current NB04/NB05 outputs.
Run after NB04 and NB05:  python reports/generate_progress_update.py
Writes outputs/expert_review/Progress_Update.pdf
"""
import base64, io, json, os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
A4, A5 = os.path.join(OUT, 'nb04_artifacts'), os.path.join(OUT, 'nb05_artifacts')
REVIEW = os.path.join(OUT, 'expert_review')
os.makedirs(REVIEW, exist_ok=True)
PDF = os.path.join(REVIEW, 'Progress_Update.pdf')

NAVY, GREY, RED, GREEN, TEAL = '#1f3b57', '#6b7280', '#c0392b', '#2e8b57', '#17becf'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
LAND = (11.69, 8.27)

# ── Data ──
clusters = {int(k): v for k, v in json.load(open(os.path.join(A4, 'heuristic_a.json'))).items()}
fail_events = pd.read_parquet(os.path.join(A4, 'fail_events.parquet'))
scores = pd.read_csv(os.path.join(A5, 'fc_scorecard.csv')).set_index('FC')
perm = pd.read_csv(os.path.join(A5, 'permutation_test.csv')).set_index('FC')
top10 = pd.read_csv(os.path.join(A5, 'fc_top10_features.csv'))
rf_imp = pd.read_csv(os.path.join(A5, 'rf_importance_by_fc.csv')).set_index('group')
n_fc, n_feat = len(clusters), sum(len(v) for v in clusters.values())
n_tr, n_te = int((fail_events.split == 'train').sum()), int((fail_events.split == 'test').sum())
keep = scores[scores.verdict == 'KEEP']
robust = keep[keep.test_check == 'holds'].sort_values('lift_train', ascending=False)
sig_test = perm[perm.significant_test].index.tolist()
robust_imp = rf_imp.reindex(robust.index)['importance'].sum()
fc_imp = rf_imp.loc[rf_imp.index.str.startswith('FC'), 'importance'].sum()

nb04 = json.load(open(os.path.join(ROOT, 'notebooks', '04_anomaly_detection_cluster_analysis.ipynb'), encoding='utf-8'))['cells']


def nb04_image(key):
    c = [c for c in nb04 if c['cell_type'] == 'code' and key in ''.join(c['source'])][0]
    png = [o['data']['image/png'] for o in c['outputs'] if 'image/png' in o.get('data', {})][0]
    return mpimg.imread(io.BytesIO(base64.b64decode(png)), format='png')


def img(path):
    return mpimg.imread(path)


def page(pdf, n, title, points, images, heights=None):
    """Title, up to four short result lines, then one or more figures side by side."""
    fig = plt.figure(figsize=LAND)
    fig.text(0.04, 0.95, title, fontsize=17, fontweight='bold', color=NAVY, va='top')
    y = 0.885
    for p in points:
        fig.text(0.05, y, '•  ' + p, fontsize=10.2, va='top', wrap=True)
        y -= 0.045 * (1 + p.count('\n'))
    top = y - 0.01
    k = len(images)
    for i, im in enumerate(images):
        ax = fig.add_axes([0.04 + i * 0.92 / k, 0.06, 0.92 / k - 0.01, top - 0.07])
        ax.imshow(im)
        ax.axis('off')
    fig.text(0.04, 0.025, 'Emirhan Kurtulus, Master Thesis, TU Wien', fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, str(n), fontsize=8, color=GREY, ha='right')
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(PDF) as pdf:
    # 1. Overview and index
    fig = plt.figure(figsize=LAND)
    fig.text(0.04, 0.95, 'Progress Update: Results', fontsize=17, fontweight='bold', color=NAVY, va='top')
    fig.text(0.04, 0.905, 'Recurring anomaly patterns with clustering | Emirhan Kurtulus, Master Thesis TU Wien',
             fontsize=10.5, color=GREY, va='top')
    lines = [
        ('IN SHORT', {'weight': 'bold', 'color': NAVY, 'size': 12}),
        f'I found {n_fc} clusters (FC1-FC{n_fc}, {n_feat} features) of sensors that become anomalous together, using only the training period.',
        f'{len(keep)} of them are closer to the {n_tr} training failures than chance would be (checked with a permutation test),',
        f'but only {len(robust)} still work on the {n_te} test failures: {", ".join(robust.index)} ({" and ".join(sig_test)} also significant there).',
        'None of them warns several days ahead, but FC5 and FC7 fire a few hours before failures.',
        'From March 2025 the data changes a lot (more anomalies without failures), and the air system changes around the April 2025 revision.',
    ]
    y = 0.84
    for t in lines:
        txt, st = t if isinstance(t, tuple) else (t, {})
        fig.text(0.05, y, txt, fontsize=st.get('size', 10.2), fontweight=st.get('weight', 'normal'), color=st.get('color', 'black'), va='top')
        y -= 0.04
    rows = [
        ('Topic', 'What I found', 'Page'),
        ('Where does the noise decrease? (thresholds)', 'z > 3 and >= 5 features is a stable choice (Youden J)', '2'),
        ('Try 2 heuristic approaches; is it valid?', 'I kept Jaccard; the clusters also show in the real values', '3'),
        ('Match to failure events', f'{len(keep)} significant on train, {len(robust)} hold on test ({", ".join(robust.index)})', '4'),
        ('Top-10 features with numerical statistics; same sensor twice?', 'values per cluster; _sum dropped, min/max kept apart', '5'),
        ('Variance going up / shifting before failures?', 'no clear change in the last days, rather slow changes over months', '6'),
        ('Back-calculations, underlying values', 'compressor work and air pressure per day; raw 1 Hz signals', '7-8'),
        ('Feature importance: feature -> event -> failure', f'RF relies on FC18/FC6; robust clusters only {100 * robust_imp:.1f}%', '9'),
    ]
    y0 = y - 0.03
    for i, (a, b, c) in enumerate(rows):
        yy = y0 - i * 0.043
        w = 'bold' if i == 0 else 'normal'
        col = NAVY if i == 0 else 'black'
        fig.text(0.05, yy, a, fontsize=9.6, fontweight=w, color=col, va='top')
        fig.text(0.47, yy, b, fontsize=9.6, fontweight=w, color=col, va='top')
        fig.text(0.94, yy, c, fontsize=9.6, fontweight=w, color=col, va='top', ha='right')
        if i == 0:
            fig.add_artist(plt.Line2D([0.05, 0.94], [yy - 0.03, yy - 0.03], color=NAVY, lw=0.8))
    fig.text(0.04, 0.025, 'Emirhan Kurtulus, Master Thesis, TU Wien', fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, '1', fontsize=8, color=GREY, ha='right')
    pdf.savefig(fig); plt.close(fig)

    # 2. Noise / thresholds
    page(pdf, 2, 'Where Does the Noise Decrease? Choosing the Thresholds', [
        "A window counts as anomalous if at least 5 features are more than 3 standard deviations from normal (training period only).",
        "I compared 30 settings with Youden's J: mine is 3rd (58.6 vs 58.7 for the best), and the difference (0.18, 95% CI -0.34 to 0.82) is not significant.",
        'In normal windows there is a dip at 5-9 anomalous features, so a minimum anywhere from 4 to 10 gives almost the same result.',
    ], [nb04_image('1.2b Youden')])

    # 3. Two heuristics and validity
    page(pdf, 3, 'Two Heuristic Approaches and Validity of the Clusters', [
        'I tried Jaccard (features anomalous together) and correlation (values move together); they agree more than chance (ARI 0.46).',
        'With the other method\'s distance, the Jaccard clusters still hold (silhouette 0.47) and the correlation clusters do not (0.26), so I kept Jaccard.',
        'All 16 clusters with more than one feature also show up in the real sensor values (both measures >= 0.3).',
    ], [img(os.path.join(A4, 'heuristic_comparison.png')), img(os.path.join(A4, 'binary_vs_continuous_validation.png'))])

    # 4. Match to failure events
    r = robust
    page(pdf, 4, 'Matching the Clusters to the Failure Events', [
        f'I used each cluster as an alarm and compared it with the failures (lift = alarms near failures vs chance), deciding only on the {n_tr} training failures.',
        f'{len(keep)} clusters are significant on training (permutation test, BH-corrected); on the {n_te} test failures only {len(r)} still work:',
        '   ' + ' | '.join(f'{k}: lift {r.loc[k, "lift_train"]:.1f} train / {r.loc[k, "lift_test"]:.1f} test, {int(r.loc[k, "false_alarms"])} false alarm{"" if int(r.loc[k, "false_alarms"]) == 1 else "s"}' for k in r.index),
    ], [img(os.path.join(A5, 'fc_scorecard.png'))])

    # 5. Top features with values (table)
    fig = plt.figure(figsize=LAND)
    fig.text(0.04, 0.95, 'Top Features with Numerical Statistics', fontsize=17, fontweight='bold', color=NAVY, va='top')
    fig.text(0.05, 0.885, '•  For the three clusters that still work on the test period: normal range of each feature vs its value when the cluster fires.',
             fontsize=10.2, va='top')
    fig.text(0.05, 0.84, '•  Same sensor twice: I dropped the _sum features (almost a copy of the mean); when I cluster per sensor instead, FC5 stays,\n    but FC7 only exists if min and max are kept apart.',
             fontsize=10.2, va='top')
    t = top10[top10.FC.isin(robust.index)]
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.64]); ax.axis('off')
    cells = [[r.FC, r.feature, f'{100 * r.active_share:.0f}%', f'{r.normal_median:.3g}', f'{r.normal_p05:.3g} - {r.normal_p95:.3g}',
              f'{r.fc_active_median:.3g}', f'{r.shift_in_std:+.1f}'] for r in t.itertuples()]
    tab = ax.table(cellText=cells, colLabels=['Cluster', 'Feature', 'Active when\ncluster fires', 'Normal\nmedian', 'Normal range\n(5-95%)',
                                              'Median when\ncluster fires', 'Shift\n(std)'],
                   loc='upper center', cellLoc='left', colWidths=[0.07, 0.38, 0.1, 0.09, 0.13, 0.11, 0.07])
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.9)
    for (i, j), c in tab.get_celld().items():
        c.set_edgecolor('#d0d4da')
        if i == 0:
            c.set_facecolor('#dce6f1'); c.set_text_props(weight='bold')
    fig.text(0.04, 0.025, 'Emirhan Kurtulus, Master Thesis, TU Wien | values are in the scaled units of the public MetroAT data',
             fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, '5', fontsize=8, color=GREY, ha='right')
    pdf.savefig(fig); plt.close(fig)

    # 6. Variance / change before failures
    page(pdf, 6, 'Does the Variance Go Up or Shift Before Failures?', [
        'I lined up the failures of each type at their start and compared every failure with its own days -14 to -8 (so the slow drift does not matter).',
        'Nothing changes consistently in the last 3 days (Wilcoxon, BH); the closest is FC5 before brake failures (q = 0.051).',
        'For the variance within a day, only FC18 goes up a bit before failures (1.25x); the others only change on the failure day itself.',
    ], [img(os.path.join(A5, 'failure_profiles.png'))])

    # 7. Back-calculation
    page(pdf, 7, 'Back-Calculation: Compressor Work and Air Pressure per Day', [
        'From the raw 1 Hz data I calculated per day and side: compressor on-share, starts per hour, fill rate and pressure loss while standing.',
        'The pressure loss while standing goes up every month in the training period (about 5x), and the CW2 side fills more slowly and runs more.',
        'Around the revision of 17-22 April 2025 the two sides swap: CW2 fills faster again and CW1 slower. I would like to ask the expert about this.',
    ], [img(os.path.join(A5, 'daily_physics_timeline.png'))])

    # 8. Underlying values (raw signals): two figures stacked
    fig = plt.figure(figsize=LAND)
    fig.text(0.04, 0.95, 'Underlying Values: Raw 1-Second Signals Before a Failure', fontsize=17, fontweight='bold', color=NAVY, va='top')
    for k, p in enumerate([
            'FC5 (top): the CW2 compressor switches on and off more and more often in the hours before the failure on 10 May 2025.',
            'FC7 (bottom): the CW1 spring-brake and load pressure drop to 0 at exactly the same seconds, hours before the failure on 14 Dec 2024',
            '   (not sure yet if this is a real pressure loss or the signal dropping out).']):
        fig.text(0.05, 0.885 - k * 0.045, ('•  ' if not p.startswith('   ') else '') + p, fontsize=10.2, va='top')
    for (path, box) in ((os.path.join(A5, 'fc5_raw_signals.png'), [0.04, 0.6, 0.92, 0.14]),
                        (os.path.join(A5, 'fc7_raw_signals.png'), [0.04, 0.06, 0.92, 0.52])):
        ax = fig.add_axes(box); ax.imshow(img(path)); ax.axis('off')
    fig.text(0.04, 0.025, 'Emirhan Kurtulus, Master Thesis, TU Wien', fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, '8', fontsize=8, color=GREY, ha='right')
    pdf.savefig(fig); plt.close(fig)

    # 9. Feature importance: re-drawn from rf_importance_by_fc.csv with plain labels for the non-cluster groups
    fig = plt.figure(figsize=LAND)
    fig.text(0.04, 0.95, 'Feature Importance: Feature -> Event -> Failure', fontsize=17, fontweight='bold', color=NAVY, va='top')
    for k, p in enumerate([
            f'I summed the importance of the Random Forest baseline per cluster: {100 * fc_imp:.0f}% is on cluster features, mostly FC18 and FC6.',
            f'The clusters that still work on the test period (FC5, FC7, FC8, green) get only {100 * robust_imp:.1f}%: they might work, but the RF hardly uses them.',
            'Grey: the RF used all 340 features, so part of its importance is on features that are not in any cluster (rarely anomalous,',
            '   the _sum copies of the mean values, and speed / temperature / days since the last failure or revision).',
            f'About {100 * rf_imp.loc["sensor feature, no FC", "importance"]:.0f}% is on features that rarely become anomalous, so the clusters do not include them.',
            'FC2, FC6 and FC18 show up a lot in the RF false alarms (test period), so they might be worth leaving out later.',
            'This is still with the old baseline models; I will redo it after re-tuning them.']):
        fig.text(0.05, 0.885 - k * 0.042, ('' if p.startswith('   ') else '•  ') + p, fontsize=10.2, va='top')
    rename = {'sensor feature, no FC': 'not in a cluster (rarely anomalous)', '_sum (dropped in NB04)': '_sum copies of the mean',
              'context / asset': 'speed / temperature / days since'}
    imp = rf_imp['importance'].sort_values()
    labels = [rename.get(g, g) for g in imp.index]
    colors = ['#9aa0a6' if g in rename else (GREEN if g in robust.index else '#1f77b4') for g in imp.index]
    ax = fig.add_axes([0.3, 0.1, 0.62, 0.45])
    ax.barh(range(len(imp)), 100 * imp.values, color=colors)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(labels, fontsize=8.5)
    for i, v in enumerate(imp.values):
        if v >= 0.005:
            ax.text(100 * v + 0.3, i, f'{100 * v:.1f}%', va='center', fontsize=7.5, color=GREY)
    ax.set_xlabel('Share of the Random Forest importance (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color='#1f77b4'),
                       plt.Rectangle((0, 0), 1, 1, color='#9aa0a6')],
              labels=['cluster that still works on the test period', 'other cluster', 'not in any cluster'], loc='lower right', fontsize=8.5, frameon=False)
    fig.text(0.04, 0.025, 'Emirhan Kurtulus, Master Thesis, TU Wien', fontsize=7.5, color=GREY)
    fig.text(0.96, 0.025, '9', fontsize=8, color=GREY, ha='right')
    pdf.savefig(fig); plt.close(fig)

print(f'Wrote {PDF}')
