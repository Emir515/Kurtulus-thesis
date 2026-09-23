import os, sys, json, glob, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'outputs')

TS_COL = 'TIMESTAMP'
FAIL_COL = 'TRAIN_IS_IN_FAILURE'
MAINT_COL = 'TRAIN_IS_IN_MAINTENANCE'
SEQ_WINDOW_S = 1402
SUBSAMPLE = 4
seq_len_sub = SEQ_WINDOW_S // SUBSAMPLE
n_channels = 95

# Load metadata
with open(os.path.join(OUT_DIR, 'unsupervised', 'metadata.json')) as f:
    meta = json.load(f)
UNSUP_CHANNELS = meta['channels']
channel_stats = meta['channel_stats']

# Load RF model
saved = joblib.load(os.path.join(OUT_DIR, 'supervised', 'rf_model.joblib'))
rf_model = saved['model']
thr = saved['threshold']

# ── RF: predict on train + test ──
print('Loading supervised data...')
train = pd.read_parquet(os.path.join(OUT_DIR, 'supervised', 'train.parquet'))
test = pd.read_parquet(os.path.join(OUT_DIR, 'supervised', 'test.parquet'))
train['window_end'] = pd.to_datetime(train['window_end'])
test['window_end'] = pd.to_datetime(test['window_end'])
feature_cols = [c for c in train.columns if c not in {'timestamp', 'window_end', 'label'}]

train['prob'] = rf_model.predict_proba(train[feature_cols])[:, 1]
test['prob'] = rf_model.predict_proba(test[feature_cols])[:, 1]
train['fail'] = train['label']
test['fail'] = test['label']

rf_all = pd.concat([
    train[['window_end', 'prob', 'fail']],
    test[['window_end', 'prob', 'fail']]
]).sort_values('window_end').set_index('window_end')
del train, test; gc.collect()
print(f'RF combined: {len(rf_all)} windows')

# ── LSTM-AE model ──
class LSTMAutoencoder(nn.Module):
    def __init__(self, nc, hd):
        super().__init__()
        self.encoder = nn.LSTM(nc, hd, batch_first=True)
        self.decoder = nn.LSTM(hd, hd, batch_first=True)
        self.output_layer = nn.Linear(hd, nc)
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.output_layer(dec_out)

ae = LSTMAutoencoder(95, 64)
ae.load_state_dict(torch.load(os.path.join(OUT_DIR, 'unsupervised', 'lstm_ae_model.pt'), weights_only=True))
ae.eval()

def score_sequences(data_glob):
    files = sorted(glob.glob(data_glob, recursive=True))
    unsup_cols = [TS_COL, FAIL_COL, MAINT_COL] + UNSUP_CHANNELS
    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        avail = [c for c in unsup_cols if c in df.columns]
        chunks.append(df[avail])
        del df
    raw = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    raw[TS_COL] = pd.to_datetime(raw[TS_COL])
    raw.sort_values(TS_COL, inplace=True)
    raw.reset_index(drop=True, inplace=True)
    clean = raw[~raw[MAINT_COL].astype(bool)].copy()
    del raw; gc.collect()
    for c in UNSUP_CHANNELS:
        clean[c] = (clean[c] - channel_stats[c]['mean']) / channel_stats[c]['std']
    clean = clean.set_index(TS_COL).sort_index()
    resampler = clean.resample(f'{SEQ_WINDOW_S}s', origin='epoch')

    times, labels, seqs = [], [], []
    for ws, group in resampler:
        if len(group) < int(SEQ_WINDOW_S * 0.8):
            continue
        vals = group[UNSUP_CHANNELS].values
        if len(vals) >= SEQ_WINDOW_S:
            sub = vals[::SUBSAMPLE][:seq_len_sub]
        else:
            x_old = np.linspace(0, 1, len(vals))
            x_new = np.linspace(0, 1, SEQ_WINDOW_S)
            full = np.zeros((SEQ_WINDOW_S, n_channels), dtype=np.float32)
            for ch in range(n_channels):
                full[:, ch] = np.interp(x_new, x_old, vals[:, ch])
            sub = full[::SUBSAMPLE][:seq_len_sub]
        if len(sub) < seq_len_sub:
            sub = np.vstack([sub, np.zeros((seq_len_sub - len(sub), n_channels), dtype=np.float32)])
        seqs.append(np.nan_to_num(sub, 0.0).astype(np.float32))
        times.append(ws + pd.Timedelta(seconds=SEQ_WINDOW_S))
        has_fail = group[FAIL_COL].astype(bool).any() if FAIL_COL in group.columns else False
        labels.append(int(has_fail))
    del clean; gc.collect()

    seqs_arr = np.array(seqs, dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.FloatTensor(seqs_arr), torch.FloatTensor(seqs_arr)), batch_size=128)
    errors = []
    with torch.no_grad():
        for xb, _ in loader:
            recon = ae(xb)
            errors.append(((xb - recon) ** 2).mean(dim=(1, 2)).numpy())
    scores = np.concatenate(errors)
    return pd.DataFrame({'window_end': times, 'score': scores, 'fail': labels})

print('Scoring train sequences with LSTM-AE...')
train_ae = score_sequences(os.path.join(ROOT, 'train', '**', '*.parquet'))
print(f'Train LSTM-AE: {len(train_ae)} sequences')

print('Loading test LSTM-AE scores...')
test_ae = pd.read_csv(os.path.join(OUT_DIR, 'unsupervised', 'test_predictions_lstmae.csv'))
test_ae = test_ae.rename(columns={'anomaly_score_lstmae': 'score', 'true_failure_label': 'fail'})

ae_all = pd.concat([
    train_ae[['window_end', 'score', 'fail']],
    test_ae[['window_end', 'score', 'fail']]
])
ae_all['window_end'] = pd.to_datetime(ae_all['window_end'])
ae_all = ae_all.sort_values('window_end').set_index('window_end')
print(f'LSTM-AE combined: {len(ae_all)} sequences')

# ── Rolling averages ──
rf_roll = rf_all['prob'].rolling(50, center=True, min_periods=1).mean()
ae_roll = ae_all['score'].rolling(20, center=True, min_periods=1).mean()

def get_failure_spans(series):
    spans = []
    in_fail = False
    for t, v in series.items():
        if v == 1 and not in_fail:
            start = t
            in_fail = True
        elif v == 0 and in_fail:
            spans.append((start, t))
            in_fail = False
    if in_fail:
        spans.append((start, series.index[-1]))
    return spans

rf_spans = get_failure_spans(rf_all['fail'])
ae_spans = get_failure_spans(ae_all['fail'])
split_date = pd.Timestamp('2025-02-12')

# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(26, 10), sharex=True)

ax = axes[0]
ax.plot(rf_roll.index, rf_roll.values, color='steelblue', linewidth=0.7, alpha=0.85, label='P(failure) rolling avg')
ax.axhline(y=thr, color='green', linestyle='--', linewidth=1.2, label=f'Threshold={thr:.2f}')
ax.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5, label='Train/Test split')
for i, (s, e) in enumerate(rf_spans):
    ax.axvspan(s, e, alpha=0.3, color='red', label='Failure period' if i == 0 else '_')
ax.axvspan(rf_all.index.min(), split_date, alpha=0.05, color='blue')
ax.axvspan(split_date, rf_all.index.max(), alpha=0.05, color='orange')
ax.set_ylabel('P(failure)', fontsize=12)
ax.set_title('Random Forest \u2014 Failure Probability (Jun 2024 \u2013 Jun 2025)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ae_roll.index, ae_roll.values, color='darkorange', linewidth=0.7, alpha=0.85, label='Recon. error rolling avg')
ax.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5, label='Train/Test split')
for i, (s, e) in enumerate(ae_spans):
    ax.axvspan(s, e, alpha=0.3, color='red', label='Failure period' if i == 0 else '_')
ax.axvspan(ae_all.index.min(), split_date, alpha=0.05, color='blue')
ax.axvspan(split_date, ae_all.index.max(), alpha=0.05, color='orange')
ax.set_ylabel('Reconstruction Error', fontsize=12)
ax.set_title('LSTM-AE \u2014 Anomaly Score (Jun 2024 \u2013 Jun 2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.legend(loc='upper left', fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'anomaly_timeline_full.png'), dpi=150, bbox_inches='tight')
print('Saved anomaly_timeline_full.png')
