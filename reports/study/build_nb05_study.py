"""Build a self-contained study guide for NB05: every output and figure with a plain-language explanation
and the reader's own questions highlighted next to the matching output.

Run after NB05:  python reports/study/build_nb05_study.py
Writes reports/study/nb05_study_guide.html (figures embedded) and nb05_study_guide.pdf (via Microsoft Edge).
"""
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_common import Notebook, render, make_pdf

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'nb05_study_guide.html')
nb = Notebook(os.path.join(ROOT, 'notebooks', '05_pattern_discovery_validation.ipynb'))

# ── content: (section id, title, key(s) of the code cell(s) or None, explanation dict, options) ──
BLOCKS = [
    ('s0', 'Part 0 · Setup', None, None, None),
    ('n0', 'Output 0 · What NB05 starts from', 'import os, json, warnings', dict(
        what='Loads the results of NB04: the 64,522 windows, the 12,022 unusual windows, the 19 groups (FC1–FC19, 126 features) and the official list of 21 failures (10 in the training months, 11 in the test months).',
        why='NB05 does not build anything new. It <b>tests</b> the NB04 groups against the failures and prepares the material for the expert.'), None),

    ('s1', 'Part 1 · The groups at a glance', None, None, None),
    ('n1', 'Output 1 · Size, activity, failure share and cohesion of every group', ['def sensor_name(f):', 'fig, axes = plt.subplots(1, 3, figsize=(20, 8))'], dict(
        what='One row per group: number of features and sensors, how often it fires, how often its windows are failure windows, and how tight it is (cohesion).',
        read='<b>activation_rate</b> = share of the 12,022 unusual windows in which the group fires. <b>failure_rate</b> = share of those windows with label 1. '
             '<b>cohesion</b> = how often the group\'s features fire together (1 = always; below 0.5 = "loose"). '
             'Plot: left = size (grey = single feature), middle = activation rate, right = failure-label rate (dashed line = average 6%; red = more than twice the average, grey = below it).',
        shows='<ul><li><b>FC19, FC2, FC1</b> fire most often (41%, 33%, 23% of unusual windows) but have low failure shares (5–10%).</li>'
              '<li><b>FC7</b> fires rarely (1%) but 63% of its windows are failure windows; <b>FC8</b> 34%, <b>FC6</b> 22%, <b>FC5</b> 15%.</li>'
              '<li>Six groups are loose (FC11, FC13–FC17, mostly load sensors); FC1 (0.98) and FC19 (0.91) are the tightest.</li></ul>',
        why='First overview of which groups are frequent "background" and which are rare and failure-related.',
        careful='These shares count windows from all months. Whether a group really warns is decided in Part 3 on failures, not windows.'), None),

    ('s2', 'Part 2 · Failures and phases', None, None, None),
    ('n2', 'Output 2 · How the label relates to the phases', '# The failure list itself is shown in NB04', dict(
        what='Cross-table of the time phase of each window (from NB04) and the label of NB02a (1 = failure running or starting within 6.2 h).',
        read='Rows = phases, columns = label 0/1. Below: how many unusual windows fall in each phase.',
        shows='<ul><li>All <b>543 "during"</b> windows and <b>659</b> of the "imminent" (last 24 h) windows have label 1: together the 1,202 label-1 windows.</li>'
              '<li>The <b>early</b> phase (6 to 1 days before) has <b>no</b> label-1 windows.</li>'
              '<li>Most unusual windows (6,466) are in normal times.</li></ul>',
        why='The label only covers the last 6.2 h and the failure itself, so it is <b>not</b> an early-warning target. That is why NB05 scores early hits with the phases, not with the label.'), None),

    ('s3', 'Part 3 · Does each group warn? (the official verdict)', None, None, None),
    ('n3', 'Output 3.0 · Event-level scorecard', 'import sys', dict(
        what='Treats every group as an alarm system and scores it against the <b>21 failures</b> (Helm 2025; RAMS 2026).',
        read='<ul><li><b>Alarm episode</b> = group-active windows less than 30 min apart, merged; at least 2 windows.</li>'
             '<li><b>early_hits</b> = alarms 6 days to 1 day before a failure; <b>late_hits</b> = last 24 h or during.</li>'
             '<li><b>false_alarms</b> = alarms far from any failure.</li>'
             '<li><b>lift</b> = share of alarms near a failure ÷ chance (19.4% train, 41.9% test).</li>'
             '<li><b>verdict</b> = KEEP/REVIEW on the 10 training failures; <b>test_check</b> = holds/fails on the 11 test failures.</li></ul>',
        shows='<ul><li><b>15 KEEP, 4 REVIEW</b> (FC1, FC2, FC14, FC19) on the training failures.</li>'
              '<li>On the test failures only <b>3 hold: FC5, FC7, FC8</b>.</li>'
              '<li><b>FC7</b>: 17 alarms, only 1 false alarm, lift 4.4 (train) / 2.4 (test). <b>FC5</b>: 38 alarms, 5 false alarms, lift 3.7 / 1.9.</li>'
              '<li><b>FC1</b>: 673 alarms, 421 false alarms, lift 1.2: it covers almost every failure, but only because it fires all the time.</li>'
              '<li>Without the 2-window rule there would be 6,283 instead of 2,663 false alarms.</li></ul>',
        why='Counting <b>failures</b> instead of windows is the fair way to judge a warning; lift corrects for groups that fire constantly.',
        careful='The "near a failure" zone covers 19.4% of the training time and 41.9% of the test time (many failures close together in 2025), so chance is measured per period.'), None),
    ('n4', 'Output 3.0b · The scorecard plot (train decides, test checks)', '# Scorecard in two panels', dict(
        what='The same scorecard as a picture, in two panels.',
        read='Left = training months (the decision), right = test months (the check). Each dot = one group. x = failures covered, y = lift. '
             'Dashed lines: lift 1.5 and 3 failures; dotted line: chance (lift 1). Dot size = number of alarms. Colour = timing pattern from NB04. Black edge = KEEP (left) / holds (right).',
        shows='Left: 15 dots with a black edge. Right: almost all dots drop to around lift 1; only <b>FC7, FC5 and FC8</b> stay above 1.5. FC7 and FC5 are small, turquoise (warn before) and high in both panels.',
        why='Shows in one picture that most groups do not carry over to 2025, and which ones do.'), None),
    ('n5', 'Output 3.1 · Permutation test (official verdict)', '3.1 Permutation test (circular shift', dict(
        what='Replaces the fixed "lift ≥ 1.5" with a statistical test. Each group\'s alarms are shifted in time 2,000 times (at least 7 days), the lift is recomputed, and we check how often a random shift does as well as the real timing.',
        read='<b>p_train</b> = share of shifts at least as good as the real lift; <b>p_bh_train</b> = corrected for testing 19 groups (Benjamini–Hochberg). '
             'KEEP = p_bh_train &lt; 0.05 and ≥ 3 training failures. <b>p_bh_test / significant_test</b> = the same on the test months (check only).',
        shows='<ul><li>Exactly the <b>same 15 KEEP</b> as the lift rule, so the 1.5 matched a proper statistical limit.</li>'
              '<li>On the test months <b>FC5 (q = 0.009)</b> and <b>FC7 (q = 0.038)</b> are significant; <b>FC8</b> just misses (q = 0.054).</li>'
              '<li>FC4 passes on training only just (q = 0.048) and fails on test.</li></ul>',
        why='Makes the verdict statistically grounded instead of resting on a chosen number.',
        careful='"≥ 3 failures" is still a chosen number; it means a group that only warns before a rare failure type (leveling: 2 cases) can never be KEEP. A data limitation.'), None),

    ('s4', 'Part 4 · What sits behind the old models\' alarms?', None, None, None),
    ('n6', 'Output 4 · Old models\' alarms and the groups behind them', ["rf = pd.read_csv(os.path.join(OUT, 'supervised', 'test_predictions_rf.", "win_times = all_sup['window_end']"], dict(
        what='Takes the alarms of the earlier Random Forest and LSTM-AE on the test months, splits them into correct early warnings and false alarms, and checks which of our groups are active in each.',
        read='First lines: how many alarm episodes, early hits and false alarms each model has. Tables: % of windows in which a group is active during early hits vs false alarms (Fisher test, Bonferroni). '
             'Heat map: left column = early hits, right column = false alarms; darker = more active. Look for rows with a big difference between the two columns.',
        shows='<ul><li><b>RF</b>: 240 episodes, early hits for 11/11 test failures, <b>109 false-alarm episodes</b>. LSTM-AE: 10/11, 209 false alarms.</li>'
              '<li>RF false alarms: <b>FC2, FC6, FC18</b> are over-represented (e.g. FC6 44% vs 20%). RF early hits: FC1, FC14, FC17.</li>'
              '<li>LSTM-AE: all significant groups lean towards early hits (strongest FC14, FC17, FC10, FC19).</li>'
              '<li>FC2 and FC18 play opposite roles in the two models; FC5 and FC7 are too rare to make a difference here.</li></ul>',
        why='Links our groups to the thesis problem (too many false alarms) and gives RQ2 an idea: groups behind false alarms might be left out. Named in the proposal and suggested by the RAMS paper.',
        careful='Old, untuned models: provisional, re-run after NB02b in Phase 2. The Fisher test counts windows, so p-values are too optimistic. Use as hints, not proof.'), None),

    ('s5', 'Part 5 · Groups vs the failure types of Wiener Linien', None, None, None),
    ('n7', 'Output 5 · Which groups react to which failure type?', "phase = all_sup['phase'].values", dict(
        what='For each failure type recorded by Wiener Linien (brake 13, compressor 6, leveling 2): how much more often each group fires in the last 24 h and during those failures than in normal times.',
        read='Table 1 / heat map: enrichment (1 = no difference; red = more active). Table 2: in how many failures of that type the group fires at all.',
        shows='<ul><li><b>FC7</b> 62× around brake failures; FC8 12×, FC18 8×, FC16 8×.</li>'
              '<li><b>FC14</b> and <b>FC17</b> 21–24× around the 2 leveling failures (very uncertain); FC5 reacts to all three types.</li>'
              '<li>Table 2: almost every group fires in almost every failure, so the enrichment is the more informative table.</li></ul>',
        why='Evidence for the expert, and a hint that different failure types show up in different sensor groups (the idea behind RQ2).',
        careful='These are Wiener Linien\'s labels for <b>failures</b>, not labels for groups: say "FC7 is most active around failures recorded as brake failures", never "FC7 is the brake group". Uses all 21 failures, descriptive only, not used for any decision.'), None),

    ('s6', 'Part 6 · The whole year in one picture', None, None, None),
    ('n8', 'Output 6 · Timeline of all groups and failures', 'fig, ax = plt.subplots(figsize=(24, 10))', dict(
        what='Every window in which a group fires, over the whole year.',
        read='Rows = groups; dots = active windows; red bands = failures; gold bands = early zone (6 to 1 days before); dashed line = train/test split. A good warning group has dots inside the gold/red bands and few elsewhere.',
        shows='<ul><li>FC1 and FC19 fire all year: noise.</li><li>FC2 "switches on" at the end of August 2024 and stays on: looks like a change in the MW4 load signal itself (question for the expert).</li>'
              '<li>From March 2025 many rows become much denser (FC3, FC4, FC9, FC10, FC14, FC17, FC18): the drift.</li>'
              '<li>FC7 is rare and its dots sit at or just before failures.</li><li>Late June 2025: FC3–FC5 fire continuously after the long failure.</li></ul>',
        why='A good figure for the expert meeting: they can recognise periods ("that was the revision").',
        careful='19 rows over a year: good for the overall picture, not for exact timings.'), dict(no_text=True)),

    ('s7', 'Part 7 · Deep dives into single groups', None, None, None),
    ('n9', 'Output 7 · Deep dives: FC7 and FC8 over the year', 'picks = []', dict(
        what='The strongest KEEP group and the strongest group of each timing type are chosen from the results: this time <b>FC7</b> and <b>FC8</b>. For each, the main features over the whole year.',
        read='Text: which feature is unusual how often when the group fires. Plots: one panel per feature; line = daily average, band = spread within the day, red = failures, gaps = days without data, dots = single days.',
        shows='<ul><li><b>FC7</b> = mainly <code>CW1_SPRING_BRAKE_PRESSURE_BOGIE2_min</code> (unusual in 99% of FC7 windows), sometimes with <code>CW1_LOAD_PRESSURE_BOGIE2_max</code> (49%).</li>'
              '<li>Both drop sharply on a few failure days (late August, early October, mid-February, late June).</li>'
              '<li><b>FC8</b> = the lowest spring-brake pressure on several cars (MW2, MW3, MW4…) dropping together; also on some days without failure (around 20 November, 30 December), mostly next to data gaps.</li></ul>',
        why='Shows what the group physically is in the data. Key expert question: is spring-brake pressure near 0 a fault, or the parked/switched-off train?'), None),
    ('n10', 'Output 7.1 · Top features per group, with values', "normal_ref = all_sup[(all_sup['split'] == 'train') & (all_sup['label']", dict(
        what='For the groups that hold on the test months (FC5, FC7, FC8): their most frequent features with their normal values and their values when the group fires (all 19 groups in <code>fc_top10_features.csv</code>). Supervisor request.',
        read='<b>active_share</b> = how often the feature is unusual when the group fires. <b>normal_median</b>, <b>normal range (p05–p95)</b> = typical value and the range of 90% of normal values. '
             '<b>fc_active_median</b> = typical value when the group fires. <b>shift_in_std</b> = how far that is from normal in "usual spreads".',
        shows='<ul><li><b>FC5</b>: CW2 compressor switches 4 times per 6 min instead of 0–1 (+5.6).</li>'
              '<li><b>FC7</b>: CW1 spring-brake pressure drops from 0.66 to <b>0.00</b> (−4.3); the load pressure only mildly (0.29 → 0.20, still inside its normal range).</li>'
              '<li><b>FC8</b>: spring-brake pressure near 0 on MW4 (−4.3), ≈ 0.10 on MW2/MW3, weaker on MW1 and CW2.</li></ul>',
        why='Turns the groups into concrete numbers the expert can judge.',
        careful='Values are in the scaled units of the public MetroAT data, not bar; the shift column is the easiest to compare.'), None),
    ('n11', 'Output 7.2 · The raw 1-second signals', 'import glob as _glob', dict(
        what='Goes back to the original 1-second data for FC5, FC7 and FC8, around the failure where each group fires most in the 24 h before. Supervisor request ("underlying values").',
        read='Blue = raw values (on/off signals: switches per 10 min); orange triangles = windows where the group fires; red = the failure; empty stretches = no data.',
        shows='<ul><li><b>FC5</b> (10 May 2025, leveling failure): the CW2 compressor switches more and more often during the day, up to 12 times in 10 min, <b>hours before</b> the failure.</li>'
              '<li><b>FC7</b> (14 Dec 2024, brake failure): the normal sawtooth disappears; spring-brake and load pressure jump to exactly 0 <b>at the same seconds</b>, hours before the recorded failure.</li>'
              '<li><b>FC8</b> (19 Aug 2024, brake failure): the pressure sits flat at about half its normal level on all four cars, with short drops to 0, the same on every car.</li></ul>',
        why='The expert can recognise patterns much better in raw signals than in our summaries.',
        careful='FC7: two different sensors dropping to exactly 0 at the same seconds could be a signal dropout rather than a real pressure loss. FC8: identical on all cars points to a whole-train state. Both are questions for the expert.'), None),
    ('n12', 'Output 7.3 · Where does the old Random Forest look?', 'import joblib', dict(
        what='The RF\'s feature importances (sum to 1), added up per group. Features outside the groups are shown as grey bars.',
        read='Longer bar = the RF relies more on it. Green = groups that hold on the test months; grey = outside the groups (other sensor features, <code>_sum</code> copies, context features).',
        shows='<ul><li>The RF relies most on <b>FC18</b> (18%), FC6 (10%) and FC14 (8%), and on sensor features outside the groups (27%).</li>'
              '<li>Our groups together: 54%. The robust groups FC5, FC7, FC8: only <b>2.4%</b>.</li>'
              '<li>FC18 and FC6 are also the groups behind the RF\'s false alarms (Part 4).</li></ul>',
        why='Shows that the flat model looks at noisy places and ignores the robust signals: motivation for RQ2.',
        careful='Untuned RF (provisional). Impurity importance favours features with many values and is split between near-copies; it shows where the model looks, not what causes failures.'), None),
    ('n13', 'Output 7.4 · Back-calculation: compressor work and air-pressure rates', "PHYS_COLS = ['TIMESTAMP'", dict(
        what='Four physical values per day and side (CW1, CW2), computed from the raw 1-second data with maintenance days left out. Supervisor request ("back-calculations").',
        read='<ul><li><b>on-share</b> = share of the time the compressor runs; <b>starts per hour</b> = how often it switches on;</li>'
             '<li><b>fill rate</b> = how fast the main tank pressure rises while it runs; <b>loss rate</b> = how fast it falls while the compressor is off and the train stands still (closest to a leak).</li>'
             '<li>Plot: blue = CW1, orange dashed = CW2; vertical lines = failures (grey brake, red compressor, purple leveling); green = maintenance; dotted = split. 3-day rolling median.</li></ul>',
        shows='<ul><li>Typical test day vs training day: <b>starts +18%</b> (1.7 → 2.0 per hour), <b>loss rate +22%</b>; on-share and fill rate about the same.</li>'
              '<li><b>Loss rate</b> climbs slowly from about 0.001 (summer 2024) to 0.005–0.006 (February 2025) on both sides.</li>'
              '<li>From October 2024 to April 2025 <b>CW2 fills much more slowly</b> (≈ 4× by spring) and runs more, with long running periods.</li>'
              '<li>Mid-August 2024: CW1 starts up to ~10 times per hour before the 19 August failure.</li></ul>',
        why='Gives a physical story behind the drift and FC5 (more air loss → more compressor work), as a hypothesis for the expert.',
        careful='Scaled units: compare directions and ratios. Negative loss rates (pressure rising while standing) are physically odd, maybe the other compressor refilling the shared tank: a question for the expert.'), None),
    ('n14', 'Output 7.4b · Long-term trends and the April 2025 maintenance', 'from scipy.stats import kendalltau, mannwhitneyu', dict(
        what='Tests the slow changes: a trend test on the monthly values of the training months, and a before/after test around the maintenance of 17–22 April 2025.',
        read='<b>Kendall\'s τ</b>: +1 = higher every month, −1 = lower every month, 0 = no direction (on monthly medians, because neighbouring days are too similar). '
             '<b>Mann–Whitney</b>: are the 30 days before and after the maintenance different?',
        shows='<ul><li>Loss rate rises <b>every month</b> June → February (τ = 1.00, ≈ 5×). CW2 fill rate falls (τ = −0.83, half the speed); CW2 runs somewhat more (τ = 0.56). CW1: no trend.</li>'
              '<li>After the April maintenance <b>the sides swap</b>: CW2 fills ≈ 3.7× faster and runs less, CW1 fills slower and runs more; standing loss is higher (these 30 days contain the May failures).</li>'
              '<li>RAMS 2026 reports a baseline shift after 17 April 2025, the same date.</li></ul>',
        why='Part of the test-period drift may come from work done in this maintenance rather than from developing faults: an important question for the expert.',
        careful='Only 9 monthly points; the before/after p-values are indicative, because days are not independent.'), None),
    ('n15', 'Output 7.5 · How do the values change in the days before a failure?', 'from scipy.stats import wilcoxon', dict(
        what='All failures of one type are lined up at their start (day 0) and followed from 14 days before to 3 days after. Every failure is measured against its own normal (days −14 to −8, green), which removes the slow drift. Supervisor request.',
        read='Columns = failure types, rows = the four physical values + FC5 activity. Thick line = average, band = 95% interval, thin lines = single failures. A warning would be a line moving away from 0 in the last days, with the band not touching 0. '
             'Table: Wilcoxon test of days −3 to −1 vs the baseline, BH-corrected; leveling (2 failures) not tested.',
        shows='<ul><li>No physical value changes consistently in the last days before a failure: the changes of 7.4/7.4b are <b>slow, over months</b>.</li>'
              '<li>Closest: FC5 activity before brake failures (p = 0.005, q = 0.051, just misses), but the change is tiny (+0.3 percentage points).</li></ul>',
        why='These values are better as slow condition indicators than as "failure tomorrow" alarms.',
        careful='Few failures (13, 6, 2), so "no change found" does not prove "no change exists"; failures close together share baseline days.'), None),

    ('s8', 'Part 8 · The expert sheet and the RQ1 metric', None, None, None),
    ('n16', 'Output 8 · Expert verification sheet', "EXPERT_COLS = ['expert_physically_coherent'", dict(
        what='One row per group with all evidence (size, cohesion, location, scorecard, permutation q-values, failure-type enrichment, RF false-alarm marker, features), saved as CSV and Excel.',
        read='Five empty columns for the expert: physically coherent (Y/N), subsystem, failure mode, keep for RQ2 (Y/N), comment.',
        shows='19 rows; the expert returns the file as <code>expert_verification_sheet_FILLED.csv</code>.',
        why='This is where subsystem names come from: the expert gives them, not us.',
        careful='The sheet shows all our evidence, which can steer the judgement. A "blind" first round (sensor names only, plus control groups) would make the RQ1 metric more trustworthy.'), None),
    ('n17', 'Output 8b · RQ1 verification metric (waiting for the expert)', 'def verification_metrics(filled):', dict(
        what='Once the filled sheet is back, this cell computes the RQ1 result promised in the proposal and prepares one dataset per expert-assigned subsystem for RQ2 (same windows, labels and split).',
        read='<b>FC confirmation rate</b> = share of groups confirmed as physically coherent (also by features); <b>assignment rate</b> = confirmed groups with a subsystem; <b>retention rate</b> = share of features kept for RQ2.',
        shows='For now: "Awaiting expert input".',
        why='Turns the expert\'s judgement into a number that answers RQ1. The RQ2 datasets are only built after the expert feedback, as planned.'), None),

    ('s9', 'Part 9 · Summary for the expert meeting', None, None, None),
    ('n18', 'Output 9 · Summary printout', "print('=' * 90)", dict(
        what='Everything in one printout: timing patterns, volatility, scorecard per group, old-model findings, drift flags and the five questions for the experts.',
        read='Read it as the one-page briefing for the meeting.',
        shows='<ul><li>Timing: 2 short-term precursors (FC5, FC7), 12 detection-only, 5 weak/mixed, no early warning several days ahead.</li>'
              '<li>Scorecard: 15 KEEP, of which FC7, FC5 and FC8 hold on the test months.</li>'
              '<li>Old models: FC2, FC6, FC18 behind RF false alarms.</li>'
              '<li>Drift: nine groups flagged.</li><li>Questions Q1–Q5 for the experts.</li></ul>',
        why='The basis for the expert meeting and the supervisor update.'), dict(no_text=True)),
    ('n19', 'Output 9b · Saved files', "fc_summary.to_csv(os.path.join(NB05_OUT, 'fc_summary.csv')", dict(
        what='The tables and figures NB05 saves in <code>outputs/nb05_artifacts/</code> (scorecard, permutation test, expert sheet, physics tables, plots).',
        why='NB06, the README and the expert material read these files.'), dict(no_text=True)),
]

# ── Your questions (asked while going through NB05) ──
QA = {
    'n1': [('What is cohesion?',
            'How often the features of a group are unusual <b>at the same time</b>, from 0 to 1. For every pair of features: of the windows where at least one is unusual, the share where both are (Jaccard); cohesion is the average over all pairs. '
            'FC1: 0.98 (almost always together); FC16: 0.38 (often separately). High cohesion means one tight pattern, <b>not</b> that the group is useful: FC1 is tight but not linked to failures.')],
    'n3': [('What is lift, briefly?',
            'How much more often a group fires near a failure than it would by chance. 1 = chance, 2 = twice as often, below 1 = less. FC7: 4.4 on the training months.'),
           ('A daily-life example for lift?',
            'It rains on 20 of 100 days (chance = 20%). Friend A says "rain tomorrow" on 10 days and is right on 8 → 80% ÷ 20% = <b>lift 4</b>. '
            'Friend B says "rain" every day → right on all rainy days, but only 20% of the time → <b>lift 1</b>, useless. FC7 is friend A; FC1 is friend B.'),
           ('Does lift count alarms "during" a failure, or only before?',
            'Both. "Near a failure" = the early window (6 to 1 days before), the last 24 h, and <b>during</b>. Not counted: after the failure (aftermath) and far away (false alarm). '
            'For "does it warn <b>before</b>?" look at the <b>pattern</b> column (short-term precursor: FC5, FC7).'),
           ('Is lift_train the percentage of alarms close to or on the failures in the training months?',
            'Almost: it is that percentage <b>divided by chance</b>. FC7: about 85% of its alarms are near a failure, ÷ 19.4% = 4.4. FC1: about 24% ÷ 19.4% = 1.2, barely better than chance. '
            'The plain percentage is <b>event_precision</b> (all months). In the test months chance is 41.9% because failures are closer together.'),
           ('How are KEEP and REVIEW decided?',
            'Only on the <b>10 training failures</b>. First rule: lift ≥ 1.5 and ≥ 3 failures covered. Official rule (§3.1): permutation test, BH-corrected q &lt; 0.05 and ≥ 3 failures. '
            'Both give the <b>same 15 KEEP</b> (REVIEW: FC1, FC2, FC14, FC19). The test months only <b>check</b> afterwards: FC5, FC7, FC8 hold.')],
    'n4': [('How do I read the two-panel plot?',
            'Left = training (decision), right = test (check). Each dot = a group; further right = more failures caught; higher = better than chance. '
            'Useful = above the 1.5 line and right of the 3 line. Small dot = few alarms. Turquoise = warns before; red = reacts during. Black edge = KEEP (left) / holds (right). '
            '<b>Look for dots that are high in both panels</b>: FC7 and FC5.')],
    'n5': [('What is the permutation test, in an easy way?',
            'We keep the failures where they are and move a group\'s alarms to random times (2,000 times, keeping their pattern of bursts and gaps). '
            'If the real alarms are closer to the failures than almost all random versions, the group is not lucky. The share of random versions that do as well is the p-value.'),
           ('What does a p-value do?',
            'The chance of a result this good if there were no real link. Below 0.05 = probably real. Because we test 19 groups, the p-values are corrected (BH), otherwise about 1 in 20 would pass by luck.'),
           ('A group may give a real signal for one failure but not for another, because that failure belongs to a different group?',
            'Yes, that is expected: different failures can show up in different groups. That is why a group only needs to cover ≥ 3 failures, not all of them, and why §5 looks at failure types. '
            'The limitation: a group that only warns before a rare type (leveling, 2 cases) can never reach 3 training failures.'),
           ('Should we add the breakdown types to the permutation test?',
            'No. Using the failure types to decide would let the labels steer the verdict (a back door for leakage and for subsystem labelling). Types stay descriptive only (§5); the experts decide what each group is.')],
    'n6': [('Is this part the back-calculation?',
            'No. This part zooms into the old models\' alarms. The back-calculation (going back from summaries to the physical values, e.g. compressor run time and pressure loss) is §7.4.'),
           ('Why do we need it, simply?',
            'The old models raise too many false alarms. §4 asks: when they raise an alarm, which of our groups are active? Groups mostly active during false alarms are the "toaster that sets off the smoke alarm" '
            '(FC2, FC6, FC18 for the RF). Leaving them out might reduce false alarms, which RQ2 can test. It is also in the proposal.'),
           ('What is an early hit?',
            'An alarm 1 to 6 days before a failure: early enough to plan a repair. Last day or during = late hit; within 24 h after = post-failure; nowhere near = false alarm. '
            'Example: failure on Friday → Monday alarm = early hit; Thursday evening = late hit; Saturday = post-failure; two weeks earlier = false alarm. The window follows Helm\'s thesis and the RAMS paper.'),
           ('The RAMS paper reports 14 false alarms, we have 109. Why?',
            'Different counting, not a worse model; with Stefan\'s method we also get about 14. (1) RAMS smooths the scores with a two-sided filter that uses <b>future</b> scores, so many short alarms melt into one (not deployable). '
            '(2) RAMS counts each upward crossing once; we count every alarm episode (30-min gap). (3) RAMS stops on 1 June 2025, we use the official end, 30 June (11 failures). '
            'Like a fire alarm beeping in 7 bursts: 1 false alarm for RAMS, 7 for us. For RQ2: report raw, causal and two-sided smoothing.')],
    'n7': [('What is enrichment?',
            'How many times more often something happens in a special situation than in normal times: enrichment = rate near failures ÷ rate in normal times. '
            'You sneeze 2 times a day normally and 20 times on pollen days → enrichment 10. FC7 fires in about 0.05% of normal windows and about 3% around brake failures → about 62. '
            'Unlike lift (alarms near failures vs chance), it compares activity rates; it counts windows, so a big number can come from a few failures.')],
    'n9': [('(Fixed after your question) Why did the lines look like smooth drops across missing days?',
            'The line was drawn across days without data. Now it breaks at gaps, single days are dots, and the y-labels show the full feature name (including <code>_min</code>/<code>_max</code>).')],
    'n10': [('What does shift_in_std show?',
             'How far the value moves from normal, in "usual spreads": (value when the group fires − normal average) ÷ normal spread. FC7: (0.00 − 0.66) ÷ 0.15 ≈ −4.3. '
             '0 = normal, ±2 clearly different, ±4 very different. It puts all sensors on one scale (like a fever of 38.0 °C: (38.0 − 36.8) ÷ 0.3 = +4).'),
            ('What does normal range (p05–p95) mean?',
             'The range of the middle 90% of normal values: 5% are lower than p05, 5% higher than p95. FC7\'s spring-brake pressure is normally 0.29–0.74; when FC7 fires it is 0.00, far below. '
             'It ignores the most extreme 5% on each side, which are often one-off glitches.')],
    'n12': [('What does "sensor feature, no FC" mean?',
             'Real sensor features that are in none of the 19 groups (132 features). They were left out in NB04 because they are unusual (z &gt; 3) in fewer than 1% of the training unusual windows. '
             'The RF still uses them for their <b>normal level</b> (e.g. a slightly higher load pressure), which our groups cannot capture because they only record clearly unusual moments.'),
            ('Why do we need these results, briefly?',
             '(1) The RF leans on FC18 and FC6, the groups behind its false alarms. (2) It hardly uses FC5, FC7, FC8, the groups that held on 2025. (3) It answers the meeting goal "feature → event → failure". '
             'In one sentence: the old model looks in the wrong places, our groups point to better ones (provisional until Phase 2).')],
    'n13': [('What are on-share, starts per hour, fill rate and loss rate?',
             'Four daily measures of how hard the compressor works. <b>On-share</b>: share of the time it runs. <b>Starts per hour</b>: how often it switches on. '
             '<b>Fill rate</b>: how fast the pressure rises while it runs. <b>Loss rate</b>: how fast the pressure falls while it is off and the train stands. '
             'Like a bicycle tyre with a slow puncture: it goes flat faster (loss ↑), you pump more often (starts ↑) and longer (on-share ↑), and pumping feels slower (fill ↓).'),
            ('Why are there green and blue in the plot?',
             'Green bands = maintenance (those days are left out). Blue used to mean both the CW1 line and leveling failures; leveling failures are now <b>purple</b>, so blue only means CW1.'),
            ('How do I interpret the plot?',
             '<ul><li><b>On-share:</b> both sides ≈ 30% in summer 2024; from January to mid-April 2025 CW2 runs more than CW1; after the April maintenance the order flips.</li>'
             '<li><b>Starts:</b> CW1 starts more often than CW2 from September to April, so CW2 runs a lot but in long periods; after April CW2 starts more, both 3–4 per hour in May.</li>'
             '<li><b>Fill rate:</b> equal in summer; from October CW2 fills much more slowly (≈ 4× by spring); after April both rise and come together.</li>'
             '<li><b>Loss rate:</b> slow climb on both sides from ≈ 0.001 to ≈ 0.005–0.006 by February; noisy after the split, CW2 peaks around the May failures.</li>'
             '<li>Story (hypotheses): slowly growing air loss; CW2 working harder and less effectively until April; single failures show no clear change right before them.</li></ul>')],
    'n14': [('(Fixed after your question) What exactly happened at the April maintenance?',
             'The text now uses the name in the data (8-week revision and legal inspection, 17–22 April 2025) and states that CW2 improved while CW1 got worse, with standing loss higher afterwards.')],
}

SUMMARY = """<h2>Summary in five sentences</h2>
<ol>
<li>NB05 treats each of the 19 groups as an alarm system and scores it on failures, not windows, with lift corrected for chance in each period.</li>
<li>On the 10 training failures 15 groups are KEEP (lift rule and permutation test agree); on the 11 test failures only FC5, FC7 and FC8 hold, and only FC5 and FC7 are significant.</li>
<li>FC5 (CW2 compressor switching too often) and FC7 (CW1 spring-brake pressure dropping to 0) fire in the hours before failures; FC8 shows spring-brake pressure dropping on several cars at once.</li>
<li>The old Random Forest relies on FC18 and FC6, the groups behind its false alarms, and hardly uses the robust groups: a motivation for RQ2.</li>
<li>The raw data shows slow changes over months (growing standing air loss, a weaker CW2 side until the April 2025 maintenance) rather than short-term warnings; all of it goes to the expert as observations and questions.</li>
</ol>"""

if __name__ == '__main__':
    render(nb, BLOCKS, QA, 'NB05 Study Guide',
           'Every output of <code>05_pattern_discovery_validation.ipynb</code>, with its figure, a plain explanation and the questions you asked. '
           'NB05 tests the 19 groups from NB04 against the failures (training months decide, test months check), looks at the old models, the raw signals and the physics of the air system, '
           'and prepares the sheet for the Wiener Linien expert.',
           SUMMARY, OUT)
    make_pdf(OUT, OUT.replace('.html', '.pdf'), os.path.join(tempfile.gettempdir(), 'study_print'))
