"""
Event-level scoring of activation series against the canonical failure events.
Shared by NB05 (official scorecard and permutation test) and NB06 (robustness checks), so both use the same rules:

- alarm episodes: active windows less than 30 min apart are merged; episodes need at least 2 windows (persistence filter)
- early hit: an episode overlapping [onset - 6 d, onset - 1 d] of an event; late hit: overlapping [onset - 1 d, end]
- false alarm: an episode overlapping no event window (post-failure episodes within 24 h after the end are counted apart)
- lift: near-event episodes / (episodes x chance level), chance level = share of windows near an event, per period
- permutation test: circular shift of the activation series along the time-ordered windows of one period
"""
import numpy as np
import pandas as pd

NEAR_PHASES = ('early', 'imminent', 'during')


def p_holm(p):
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out


def p_bh(p):
    """Benjamini-Hochberg (1995) adjusted p-values (q-values)."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[o] * m / (m - np.arange(m)))
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out


class EventScorer:
    """Scores activation series (boolean per window of all_sup) against fail_events."""

    def __init__(self, all_sup, fail_events, split_date, window_s, gap=pd.Timedelta(minutes=30)):
        self.all_sup = all_sup
        self.fail_events = fail_events
        self.split_date = pd.Timestamp(split_date)
        self.gap = gap
        self.starts = fail_events['start']
        self.ends = fail_events['end']
        near = all_sup['phase'].isin(NEAR_PHASES)
        self.chance = {sp: near[all_sup['split'] == sp].mean() for sp in ('train', 'test')}
        self.normal_days = (all_sup['phase'] == 'normal').sum() * window_s / 86400
        # vectorised scoring (permutation test)
        self.ev_start = fail_events['start'].values.astype('datetime64[ns]')
        self.ev_end = fail_events['end'].values.astype('datetime64[ns]')
        self.ev_split = fail_events['split'].values
        self.win_time = all_sup['window_end'].values.astype('datetime64[ns]')
        self.order = {sp: np.flatnonzero(all_sup['split'].values == sp) for sp in ('train', 'test')}
        for sp in self.order:
            self.order[sp] = self.order[sp][np.argsort(self.win_time[self.order[sp]])]
        self.win_per_day = {sp: len(self.order[sp]) / (np.ptp(self.win_time[self.order[sp]]) / np.timedelta64(1, 'D'))
                            for sp in self.order}

    # ── episode scoring (scorecard rules) ──
    @staticmethod
    def make_episodes(times, gap):
        t = pd.Series(np.sort(pd.to_datetime(times).values))
        if t.empty:
            return pd.DataFrame(columns=['start', 'end', 'n'])
        gid = (t.diff() > gap).cumsum()
        return t.groupby(gid).agg(start='min', end='max', n='size').reset_index(drop=True)

    def classify_episode(self, s, e):
        starts, ends, idx = self.starts, self.ends, self.fail_events.index
        early = [i for i in idx if s <= starts[i] - pd.Timedelta(days=1) and e >= starts[i] - pd.Timedelta(days=6)]
        late = [i for i in idx if i not in early and s <= ends[i] and e >= starts[i] - pd.Timedelta(days=1)]
        post = [i for i in idx if s <= ends[i] + pd.Timedelta(hours=24) and e >= ends[i]]
        kind = 'early_hit' if early else 'late_hit' if late else 'post_failure' if post else 'false_alarm'
        return kind, early, late

    def score_episodes(self, eps):
        out = eps.copy()
        res = [self.classify_episode(r.start, r.end) for r in eps.itertuples()]
        out['kind'] = [r[0] for r in res]
        out['early_events'] = [r[1] for r in res]
        out['late_events'] = [r[2] for r in res]
        return out

    def score_activation(self, act):
        """Lift and coverage per period plus the lift-rule verdict and test check for one activation series."""
        eps = self.score_episodes(self.make_episodes(self.all_sup.loc[act, 'window_end'], self.gap))
        eps = eps[eps['n'] >= 2]
        ev = {e for l in eps['early_events'] for e in l} | {e for l in eps['late_events'] for e in l}
        split = np.where(eps['start'] < self.split_date, 'train', 'test')
        is_near = eps['kind'].isin(['early_hit', 'late_hit']).values
        out = {}
        for sp in ('train', 'test'):
            exp = (split == sp).sum() * self.chance[sp]
            out[f'lift_{sp}'] = is_near[split == sp].sum() / exp if exp > 0 else np.nan
            out[f'covered_{sp}'] = sum(self.fail_events.loc[e, 'split'] == sp for e in ev)
        out['verdict'] = 'KEEP' if out['lift_train'] >= 1.5 and out['covered_train'] >= 3 else 'REVIEW'
        out['test_check'] = 'holds' if out['lift_test'] >= 1.5 and out['covered_test'] >= 3 else 'fails'
        return out

    # ── vectorised scoring (same rules) ──
    def fast_score(self, times, sp):
        """Lift and covered events of one period for the active window times (vectorised)."""
        if len(times) == 0:
            return 0.0, 0
        gap_ns = np.timedelta64(int(self.gap.total_seconds()), 's')
        d1, d6 = np.timedelta64(1, 'D'), np.timedelta64(6, 'D')
        t = np.sort(times)
        brk = np.flatnonzero(np.diff(t) > gap_ns)
        first = np.r_[0, brk + 1]
        last = np.r_[brk, len(t) - 1]
        keep = (last - first + 1) >= 2
        s, e = t[first][keep], t[last][keep]
        if len(s) == 0:
            return 0.0, 0
        in_sp = (s < np.datetime64(self.split_date)) if sp == 'train' else (s >= np.datetime64(self.split_date))
        s, e = s[in_sp], e[in_sp]
        if len(s) == 0:
            return 0.0, 0
        early = (s[:, None] <= self.ev_start[None, :] - d1) & (e[:, None] >= self.ev_start[None, :] - d6)
        late = (~early) & (s[:, None] <= self.ev_end[None, :]) & (e[:, None] >= self.ev_start[None, :] - d1)
        near = (early | late).any(axis=1)
        covered = (early | late).any(axis=0) & (self.ev_split == sp)
        return near.sum() / (len(s) * self.chance[sp]), int(covered.sum())
