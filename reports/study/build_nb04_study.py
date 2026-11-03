"""Build a self-contained study guide for NB04: every output and figure with a plain-language explanation
and the reader's own questions highlighted next to the matching output.

Run after NB04:  python reports/study/build_nb04_study.py
Writes reports/study/nb04_study_guide.html (figures embedded) and nb04_study_guide.pdf (via Microsoft Edge).
"""
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_common import Notebook, render, make_pdf

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'nb04_study_guide.html')
nb = Notebook(os.path.join(ROOT, 'notebooks', '04_anomaly_detection_cluster_analysis.ipynb'))


# ── content: (section id, title, key of the code cell or None, explanation dict, options) ──
# explanation keys: what, read, shows, why, careful  (HTML allowed)
BLOCKS = [
    ('s0', 'Part 0 · Setup and the failure list', None, None, None),

    ('b1', 'Output 0.1 · Which features are used', 'import sys, os, gc, json', dict(
        what='The notebook loads the 6-minute windows from NB02a (340 features) and removes the features that should not be part of the anomaly search.',
        read='Read it as a small calculation: 340 features, minus 74 <code>_sum</code> features, minus 8 context/history features = <b>258 sensor features</b>.',
        shows='<ul><li><b>74 <code>_sum</code> features dropped:</b> the total is almost a copy of the average (NB03a §4), so keeping both would count each sensor twice.</li>'
              '<li><b>8 features excluded:</b> speed, outside temperature and the two "days since" counters. They describe conditions, not the machine. A hot day or a long stop is unusual, but not a fault (Helm 2025).</li>'
              '<li>64,522 windows, 1,202 of them tagged "trouble" (a failure is running or starts within 6.2 h).</li></ul>',
        why='The anomaly search should only react to the machine itself.',
        careful='Speed is not deleted: it is used again in §4.2 to see whether a group fires while moving or standing.'), None),

    ('b2', 'Output 0.2 · The official failure list and the time phases', 'Canonical failure-event table', dict(
        what='Builds the list of real failures from the raw "train is in failure" flag, and gives every window a <b>phase</b>: where it sits in time relative to the nearest failure.',
        read='<b>Failure list:</b> stretches of "in failure" of the same type less than 1 h apart are merged into one event. '
             '<b>Phases:</b> <i>normal</i> (far from failures), <i>buffer</i> (7–6 days before), <i>early</i> (6 days to 1 day before), <i>imminent</i> (last 24 h), <i>during</i> (while broken), <i>post</i> (24 h after).',
        shows='<ul><li><b>21 failures</b>: 10 in the training months, 11 in the test months (same as the RAMS paper). Types from Wiener Linien: 13 brake, 6 compressor, 2 leveling.</li>'
              '<li>A typical failure lasts 3.7 h; the longest almost 6 days (18–24 June 2025).</li>'
              '<li>66% of windows are "normal", 22% "early". The early window is large because every failure gets a 5-day early window.</li>'
              '<li>543 windows are "during": the same 543 in-failure windows inside the 1,202 trouble windows, a cross-check that the definitions fit.</li></ul>',
        why='NB04 and NB05 both use this one list and these phases, so "early warning", "false alarm" and "chance level" always mean the same thing.',
        careful='Because "early" covers 22% of all windows, a group that fires at random still lands in an early window quite often. That is why NB05 always compares with chance (the lift).'), None),

    ('b3', 'Output 0.3 · Quick preview', 'Quick dataset preview', dict(
        what='A check of the table after the reductions: 64,522 windows × 258 features, 1,202 trouble windows.',
        read='The notebook also shows the first five rows for one example sensor (min, max, mean of one brake cylinder pressure).',
        shows='Nothing to interpret, only a sanity check.', why='', careful=''), None),

    ('s1', 'Part 1 · Find the unusual windows', None, None, None),

    ('b4', 'Output 1.1 · z-scores', '1.1 Z-scores relative', dict(
        what='For every feature, the average and the usual spread are learned from the <b>40,842 normal windows of the training months</b>. Every window then gets a z-score per feature.',
        read='A z-score says how far a value is from normal, in units of "usual spread". z = 0 is perfectly normal; |z| &gt; 3 is very far off.',
        shows='64,522 windows × 258 features now have a z-score.',
        why='Only normal training windows are used as the reference, so failures and the test months do not pollute "normal".', careful=''), None),

    ('b5', 'Output 1.2 · Threshold experiment (training months only)', '1.2 Threshold experiment', dict(
        what='Tests 30 combinations of the two rules that decide "unusual": how far from normal (z from 2 to 4) and how many features at the same time (1 to 20).',
        read='<b>% flagged</b> = how many windows are called unusual. <b>Recall</b> = how many trouble windows are caught. <b>Precision</b> = how many of the flagged windows are trouble windows. '
             'Only the first table (training months) is used to choose; the test table is for information.',
        shows='<ul><li>A wide net (z &gt; 2, ≥ 1 feature) catches 95% of trouble windows but flags 67% of all windows.</li>'
              '<li>A narrow net (z &gt; 3.5, ≥ 20) flags 0.9% but catches only 22% of trouble windows.</li>'
              '<li>The chosen setting (z &gt; 3, ≥ 5) flags 12.9% and catches 70.7%. At z &gt; 3, minimum 3, 5 and 10 give almost the same result.</li>'
              '<li>In the test months the same setting flags 28.8%: the drift of 2025 again.</li></ul>',
        why='Shows that the choice is a balanced middle of a trade-off, made without looking at the test months.',
        careful='No setting is "best": every row trades catching trouble against flagging normal windows.'), None),

    ('b5b', "Output 1.2b · Youden's J: is our setting the best?", '1.2b Youden', dict(
        what="A standard score for choosing a detection threshold (Youden 1950): <b>J = share of trouble windows caught − share of normal windows wrongly flagged</b>. Computed for all 30 settings, training months only.",
        read="Higher J = better balance. The <b>95% interval</b> shows how uncertain J is: whole days are resampled 1,000 times (neighbouring windows are not independent). The plot shows the 12 best settings; orange = our setting.",
        shows="<ul><li>Our setting (z &gt; 3, min 5) is <b>3rd of 30</b>: J = 58.6 vs 58.7 for the best.</li>"
              "<li>The difference to the best is 0.18 points, and its 95% interval (−0.34 to 0.82) includes 0: <b>not significantly worse</b>.</li>"
              "<li>18 of 30 settings are within the uncertainty of the best, so there is a flat top area; among these we keep z &gt; 3, the standard three-sigma rule.</li></ul>",
        why="Turns the choice into a stated, citable criterion on training data instead of 'the middle of the table'.",
        careful="J treats missed trouble and false flags as equally bad, and the intervals are wide because the training months have only 10 failures."), None),

    ('b6', 'Output 1.3 · Normal reference', '1.3 Normal reference', dict(
        what='Looks only at normal training windows: how often does the rule fire when nothing is wrong?',
        read='<b>Per feature:</b> a bell-curve feature would exceed |z| &gt; 3 in 0.27% of normal windows. '
             '<b>Quantile table:</b> "q 0.90 = 10" means 90% of normal windows have 10 or fewer unusual features. '
             '<b>Plot:</b> bars = share of normal windows with exactly that many unusual features; the y-axis is logarithmic (each step = 10×).',
        shows='<ul><li>The features are <b>not</b> bell-shaped: 92 never fire, 51 fire in more than 1% of normal windows (up to 9%).</li>'
              '<li>Most normal windows have 0–3 unusual features; then there is a <b>dip at 5–9</b> and a <b>second hill at 10–13</b> (a group of about 12 features that keeps firing together).</li>'
              '<li>Any minimum from 4 to 10 flags almost the same share of normal windows (11–13%).</li></ul>',
        why='The minimum of 5 sits in the dip, so it is a stable, data-supported choice.',
        careful='About 12% of healthy windows still count as unusual, mostly the second hill. NB05 later sorts this noise out.'), None),

    ('b7', 'Output 1.4 · Apply the choice', '1.4 Apply the chosen thresholds', dict(
        what='Applies z &gt; 3 and at least 5 features to all windows.',
        read='<b>Left plot:</b> how many features are unusual per window, blue = normal windows, red = trouble windows; the green line is the minimum of 5. '
             '<b>Right plot:</b> every window over time; red = unusual.',
        shows='<ul><li><b>12,022 unusual windows</b> (18.6%). 94% of them have no trouble tag: unusual is not the same as broken.</li>'
              '<li>60% of trouble windows are unusual; 40% look normal to this method.</li>'
              '<li>Right plot: a horizontal red band at about 11–12 features through 2024 (the recurring group), and a denser red cloud from about May 2025 (drift).</li></ul>',
        why='These 12,022 windows are the raw material for the grouping in Part 3.',
        careful='The method is a wide net on purpose. Which groups really relate to failures is decided later, in NB05.'), None),

    ('b8', 'Output 1.5 · Monthly drift check', 'Distribution shift check', dict(
        what='Shows, month by month, how many windows are unusual, especially windows far from any failure.',
        read='<b>anomaly_rate_normal_phase</b> = unusual windows far from failures. If nothing changes, this should stay flat.',
        shows='<ul><li>June 2024 – February 2025: stable at about 7–15%.</li>'
              '<li>From March 2025 it rises: 23%, 17%, 21%, and <b>43% in June</b>.</li>'
              '<li>May: all windows 42% but far from failures only 21%, so much of May\'s increase is near its four failures. June is high even far from failures: drift.</li>'
              '<li>Far from failures: 10.9% in the training months vs 27.4% in the test months.</li></ul>',
        why='The 2024 "normal" stops fitting in 2025. This is why NB05 measures chance separately per period.',
        careful='Months are counted by the window start, so the last window of 30 June counts in June.'), None),

    ('b9', 'Output 1.6 · When does the drift start? (CUSUM)', '1.6 CUSUM change-point', dict(
        what='A standard change detector (CUSUM) keeps a running score of how much the daily unusual-rate stays above normal. When the score crosses a limit, it raises an alarm.',
        read='<b>Top plot:</b> daily unusual-rate far from failures (blue = training days, orange = test days). '
             '<b>Bottom plot:</b> the running score; the red dashed line is the alarm limit (h = 5 spreads); green band = the April 2025 revision. '
             '<b>Sensitivity table:</b> the same with 9 other settings.',
        shows='<ul><li>The first change after the split is detected on <b>20 March 2025</b> (starting early March), 16–26 March in 8 of 9 settings.</li>'
              '<li>On weekly data: alarm in the week of 10 March, slowly building up from mid-December 2024, and no false alarm in 2024.</li>'
              '<li>So the change starts about <b>six weeks before</b> the 17–22 April revision.</li>'
              '<li>Two daily alarms in 2024 are single extreme days (96% and 46%), not lasting changes.</li></ul>',
        why='Dates the drift objectively and raises a question for the expert: is the March change wear, and was the revision a response to it?',
        careful='This section only describes. It changes nothing in the windows or the groups.'), None),

    ('s2', 'Part 2 · Which features are unusual', None, None, None),

    ('b10', 'Output 2.1 · The most frequent unusual features', 'Extract activation profiles', dict(
        what='For every unusual window, a yes/no list of which of the 258 features are unusual: a "fingerprint". Only the 5,351 windows of the training months are used to build groups.',
        read='The top-30 list counts how often each feature is unusual within these 5,351 windows.',
        shows='<ul><li>The lowest and highest brake cylinder pressure on all 12 bogies is unusual in about a third of all unusual windows, at the same time. These later form FC1 (lowest) and FC19 (highest).</li>'
              '<li><code>MW4_LOAD_SIGNAL_min</code> alone fires in 36%: this becomes FC2.</li>'
              '<li>Several MW3 sensors appear together: later FC18.</li>'
              '<li>90 of 258 features are never unusual in these windows.</li></ul>',
        why='You can already see the outline of the later groups.', careful=''), None),

    ('b11', 'Output 2.2 · Which unusual features are near trouble?', 'Deeper look: which activated', dict(
        what='For the top-30 features: in how many of their unusual windows is there a trouble tag?',
        read='<b>Fail Rate</b> = share of the feature\'s unusual windows that are trouble windows. *** = above the average of 6.0%.',
        shows='<ul><li>The very frequent brake cylinder features are mostly <b>below</b> 6%: they fire a lot, but rarely near trouble.</li>'
              '<li>The MW3 sensors (spring brake, load, valve, braking force) are at 12–20%: much more often near trouble.</li></ul>',
        why='Frequent is not the same as useful. A first hint which features matter.',
        careful='This table counts all unusual windows (training and test months) and uses the trouble tags. It is descriptive only and not used to build the groups.'), None),

    ('b12', 'Output 2.3 · The same, per sensor', 'Group by physical sensor', dict(
        what='Adds up the unusual counts of all summaries (min, max, mean) of each physical sensor.',
        read='<b>Active/Total</b>: how many of the sensor\'s summaries are ever unusual. <b>Which</b>: which ones.',
        shows='Brake cylinder sensors fire with their lowest and highest value; MW3 spring brake and MW3 load pressure fire with all three; the MW4 load signal only with its lowest value.',
        why='Shows that the type of summary matters (a dip, a spike or a whole level shift).', careful=''), None),

    ('b13', 'Output 2.4 · Which features fire together (top 40)', 'Option A: Feature Co-Activation', dict(
        what='A heat map of how often two features are unusual <b>at the same time</b> (Jaccard similarity).',
        read='Every row and column is a feature. Bright = they nearly always fire together (1.0); dark = never together (0.0). The diagonal is always 1.',
        shows='Bright blocks are groups of features that fire together. These blocks are exactly what the clustering finds automatically in Part 3.',
        why='A visual preview of the groups.', careful=''), None),

    ('s3', 'Part 3 · Build the groups (FC1–FC19)', None, None, None),

    ('b14', 'Output 3.1 · Keep only features that fire often enough', 'Filter to features that are activated', dict(
        what='Only features that are unusual in at least 1% of the training unusual windows (≥ 53 windows) take part in the grouping.',
        read='', shows='126 of 258 features remain.',
        why='Features that almost never fire cannot form a reliable group.',
        careful='The 1% limit is a chosen value; its effect has not been tested separately.'), None),

    ('b15', 'Output 3.2 · Two ways of grouping', 'Heuristic A: Jaccard distance', dict(
        what='The features are grouped with hierarchical clustering. <b>Method A</b> (official) groups features that are unusual at the same time (Jaccard distance, average linkage, cut at 0.7). <b>Method B</b> (comparison) groups features whose raw values move together (correlation).',
        read='The list shows the 10 largest groups of method B.',
        shows='Method A gives 19 groups, method B 13 larger groups.',
        why='A second method checks that the groups are not just a product of one technique (compared in 3.1 below).', careful=''), None),

    ('b16', 'Output 3.3 · The 19 groups', 'All feature clusters (Heuristic A) with details', dict(
        what='The official result of NB04: <b>FC1–FC19</b>, with their size and their features.',
        read='<b>Feats</b> = number of features, <b>Sensors</b> = number of physical sensors, <b>Windows</b> = unusual windows in which the group fires, <b>Fail%</b> = share of those that are trouble windows.',
        shows='<ul><li>19 groups, 126 features; 3 groups are a single feature (FC2, FC4, FC5).</li>'
              '<li><b>FC7</b> stands out: it fires rarely (134 windows) but 63% of them are trouble windows.</li>'
              '<li>FC1 and FC19 are large and fire often, but with low trouble share (5–8%).</li></ul>',
        why='These groups go to NB05 (scored against the failures) and to the expert.',
        careful='The groups are named neutrally (FC numbers and sensor names). Which part of the train they belong to is for the expert to decide.'), None),

    ('b17', 'Output 3.4 · The groups in one heat map', 'Option C: Clustered Co-Activation', dict(
        what='The same similarity map as 2.4, but for all 126 features, sorted by group.',
        read='Blue lines mark the group borders. Bright squares on the diagonal = tight groups; bright spots off the diagonal = two groups that also fire together.',
        shows='Most groups form clear bright squares; a few (loose groups) are paler.', why='', careful=''), None),

    ('b18', 'Output 3.5 · Same sensor counted twice?', 'Same-sensor redundancy analysis', dict(
        what='Checks how many groups contain several summaries (min, mean, max) of the <b>same</b> sensor.',
        read='<b>cross-sensor</b> = each feature is a different sensor; <b>mixed</b> = different sensors plus repeated summaries; <b>same-sensor</b> = one sensor described several times.',
        shows='126 features come from 68 sensors; 35 features are repeated summaries. Only FC13 is a single sensor described three ways.',
        why='This was supervisor feedback. NB06 §3 tests what happens if each sensor is counted once (FC5 survives, FC7 only exists with the summaries kept apart).', careful=''), None),

    ('b19', 'Output 3.6 · How tight is each group? (cohesion)', 'Cluster Cohesion: Average', dict(
        what='For every group with 2+ features: how often its features fire together (average Jaccard similarity, 1 = always).',
        read='Bar height = cohesion; the label shows the number of features and the trouble share; colours = trouble share (red high, orange medium, green low). The dashed line is 0.5.',
        shows='Average cohesion 0.63. FC1 and FC19 are very tight (0.9+); six groups (FC11, FC13–FC17, mostly load sensors) are loose (0.38–0.49).',
        why='Loose groups are less of a single "pattern"; this is shown to the expert as a column on the sheet.',
        careful='Features that fire at very different rates cannot reach a high Jaccard value, even if they are related.'), None),

    ('b20', 'Output 3.7 · Where the sensors of each group are', 'def sensor_location(f):', dict(
        what='A neutral description of where each group\'s sensors sit: which cars, which side (A = CW1/MW1/MW2, B = CW2/MW3/MW4), which bogies, which signal types.',
        read='<b>side_A_share / side_B_share</b> = share of the group\'s features on each side of the train.',
        shows='Some groups cover the whole train (FC1, FC6, FC10, FC19), others one car or one side (FC5: CW2 only, FC7: CW1 bogie 2, FC18: MW3 bogie 2).',
        why='Helps the expert judge whether a group makes physical sense, without us naming a subsystem.',
        careful='Brake signals within one side are applied together, so part of a group\'s tightness can come from that rather than from a shared fault.'), None),

    ('b21', 'Output 3.8 · Comparison of the two grouping methods', 'Numerical comparison of Heuristic A vs Heuristic B', dict(
        what='Compares method A (official) with method B on several measures.',
        read='<b>Silhouette</b> = how well separated the groups are (higher = better); <i>own distance</i> judges each method by its own measure, <i>cross distance</i> by the other method’s measure. The black bars in the first panel are the chance levels. <b>Failure rate spread</b> = how differently the groups relate to trouble. <b>ARI/NMI</b> = how similar the two groupings are (1 = identical).',
        shows='<ul><li>Method A has slightly better separation (0.547 vs 0.529) and separates trouble-related from harmless groups much better (spread 13.1 vs 3.3).</li>'
              '<li>The two methods agree on 87% of feature pairs, but random groupings of these sizes would already agree on 77%, so this number is only moderately above chance. ARI (corrected for chance, 0 = random) is 0.46: moderate agreement, clearly above chance.</li></ul>',
        why='Supports choosing method A.',
        careful='The agreement is moderate, so the exact borders between groups depend on the method.'), None),

    ('b22', 'Output 3.9 · Are the groups also visible in the real values?', 'Numerical evidence: binary vs continuous', dict(
        what='The grouping only uses "unusual yes/no". This checks whether the groups are also visible in the real sensor values.',
        read='For every group: <b>binary cohesion</b> (how often its features are unusual together) and <b>continuous correlation</b> (how much their real values move together). Plot: one dot per group, labelled with its FC number; on the dashed line both values are equal.',
        shows='<ul><li>All 16 groups have both values of at least 0.3: the groups are also visible in the real values, so nothing important is lost.</li>'
              '<li>13 of 16 groups lie above the line: their values move together even more than they are unusual together (e.g. FC16: 0.38 vs 0.95).</li>'
              '<li>FC7 and FC9 lie below the line: their link shows mainly when something is unusual, which is exactly what the yes/no view catches.</li></ul>',
        why='Shows the yes/no simplification keeps the groups, and for some groups even finds a link the real values hide.',
        careful='Across groups the two measures are only weakly related (r = 0.31, p = 0.24, not significant), because they measure different things. A group that is tight in one view is not necessarily tight in the other.'), None),

    ('s4', 'Part 4 · First look at the failures', None, None, None),

    ('b23', 'Output 4.1 · Are unusual windows linked to failures?', 'Statistical association: anomalies vs failures', dict(
        what='Tests whether unusual windows contain more trouble windows than normal windows, overall and per group.',
        read='<b>Odds ratio</b> = how much more likely trouble is in unusual windows (1 = no difference). <b>p-value</b> = chance of this being luck (small = real).',
        shows='<ul><li>Overall: trouble is 6.0% in unusual windows vs 0.9% in others (odds ratio 7, extremely significant).</li>'
              '<li>15 of 19 groups are linked to trouble; FC7 most strongly (odds ratio 30). Not linked: FC1, FC4, FC10, FC15.</li></ul>',
        why='Confirms the unusual windows are related to failures at all.',
        careful='This test counts windows, and neighbouring windows are not independent, so the p-values are too optimistic. It also uses all months. The trustworthy test is NB05 §3.1 (event-level permutation test, training months decide).'), None),

    ('b24', 'Output 4.2 · Moving or standing?', 'Operational state analysis', dict(
        what='Checks in which driving state each group fires: standing, low speed or moving (from the train speed).',
        read='Bars = share of a group\'s windows in each state; red border = linked to trouble in 4.1.',
        shows='<ul><li>Unusual windows are mostly standing (56%), although only 23% of all windows are standing.</li>'
              '<li>FC1, FC2, FC6, FC7, FC8 and FC12 fire almost only while standing; FC10 and FC15 almost only while moving.</li></ul>',
        why='A group that only fires in one state may be a normal operating situation.',
        careful='FC7 and FC8 fire while standing. Spring brakes apply when the air is released, so very low spring-brake pressure while standing could simply be the parking brake: a question for the expert.'), None),

    ('b25', 'Output 4.3 · When does each group fire?', 'Temporal pattern analysis: drift-controlled', dict(
        what='For each group: how much more often it fires in each phase (early, last 24 h, during) than in normal times of the same period.',
        read='<b>enrich</b> = ratio to normal (2 = twice as often). <b>events_24h vs expected_24h</b> = in how many failures the group fired in the last 24 h, vs how many by chance. '
             'Left plot: enrichment per phase. Right plot: observed vs chance in the last 24 h. Below: example time courses.',
        shows='<ul><li><b>FC5</b> and <b>FC7</b> are "short-term precursors": they fire before failures in the last 24 h clearly more than chance (FC5: 15 failures vs 8 expected, lead about 14 h; FC7: 6 vs 1.7, lead about 4 h).</li>'
              '<li>Most groups are "detection-only": they react while the train is already broken.</li>'
              '<li>No group is a reliable early warning several days ahead.</li></ul>',
        why='Shows which groups could warn, and how early.',
        careful='Descriptive. The official decision is made in NB05 with the event-level permutation test.'), None),

    ('b26', 'Output 4.4 · Failure days vs normal days', 'Failure Day vs Normal Day comparison', dict(
        what='Compares how often each group fires on days with a failure (26 days) and on normal days (333 days). Supervisor request.',
        read='<b>Diff (pp)</b> = difference in percentage points. The second table splits by training and test months.',
        shows='<ul><li>Over the whole year FC2 and FC18 are clearly more active on failure days.</li>'
              '<li>Only <b>FC2 and FC6</b> are more active on failure days in <b>both</b> periods; for many groups the direction flips between periods (drift).</li></ul>',
        why='A simple day-level view that the supervisor asked for.',
        careful='Day-level and descriptive; mixing both periods can mislead, which is why the split table matters more.'), None),

    ('b27', 'Output 4.5 · Does the variation grow before failures?', 'Volatility lead-up analysis', dict(
        what='For six groups: does the day-to-day variation of their features rise in the days before a failure? Supervisor request.',
        read='<b>pre_vol_ratio</b> = variation in the 3 days before the failure vs normal days (1 = no change). <b>kendall_tau</b> = whether it rises steadily (1 = steady rise). Plots: average course around the failure day (red line).',
        shows='Only <b>FC18</b> shows a moderate rise before failures (ratio 1.25, steady rise τ = 0.71, p = 0.03). The others show no change, or a jump only on the failure day itself.',
        why='Answers the question whether failures announce themselves by growing variation: mostly no.', careful=''), None),

    ('b28', 'Output 4.6 · Drift per group', 'drift_rows = []', dict(
        what='For each group: how often it fires far from failures in the training months vs the test months.',
        read='<b>test_train_ratio</b> = how many times more often in the test months. <b>drift_flag</b> = ratio ≥ 2 and test rate ≥ 1%.',
        shows='Nine groups are flagged (e.g. FC10: 29× more often in 2025). FC5, FC7 and FC8 are not flagged.',
        why='Test-period results of flagged groups should be read with care.',
        careful='The flag rule (≥ 2× and ≥ 1%) is a chosen rule of thumb.'), None),

    ('s5', 'Part 5 · Save', None, None, None),

    ('b29', 'Output 5 · Files for NB05', 'Save all artifacts for NB05', dict(
        what='Saves the groups (<code>heuristic_a.json</code>), the failure list, the unusual windows and the tables for NB05.',
        read='', shows='', why='NB05 and NB06 start from these files.', careful=''), dict(no_text=True)),
]

# ── Your questions (asked while going through NB04), shown highlighted next to the matching output ──
QA = {
    'b2': [('What is "trouble"?',
            'A 6-minute window with <b>label 1</b> (in the notebooks: "failure window"). A window is trouble if a failure is <b>running</b> in it, '
            'or a failure <b>starts within the next 6.2 hours</b>. Example: a failure starts at 14:00 → a window at 08:00 is trouble, one at 07:00 is normal, one at 15:00 (during) is trouble. '
            'The 6.2 h comes from Helm\'s thesis and the RAMS paper and is kept unchanged. <b>Do not confuse:</b> 21 failures (real events) vs 1,202 trouble windows (one failure creates many).')],
    'b5': [('Why z > 3 and at least 5 features?',
            '<b>z &gt; 3</b>: the standard three-sigma rule from quality control (Shewhart, Montgomery); z &gt; 2 flags almost half of all windows, z &gt; 4 misses many trouble windows; '
            'NB06 shows the groups barely change at 2.5 or 3.5. <b>≥ 5 features</b>: ignores single glitches, sits in the natural dip of the normal reference (§1.3), and keeps the small useful groups '
            '(FC5 has 1 feature, FC7 has 2). Both started as conventions; the training-only experiment, Youden\'s J and NB06 make them grounded. '
            '<b>Limitation:</b> the data is not bell-shaped, so "3 sigma" is not equally strict for every feature.')],
    'b5b': [('Is there any other combination that is better?',
             'No setting is clearly better. The top settings (z &gt; 3.5/min 3 and z &gt; 4/min 3) score J = 58.7 vs 58.5 for ours: about 80 windows out of 41,000, i.e. random variation.'),
            ('If the examiner asks: why not the first two?',
             '<i>"Because they are not significantly better: the difference is 0.18 points and its 95% interval (−0.34 to 0.82) includes zero. Among the equally good settings I kept the standard three-sigma rule and the minimum that sits in the most stable region of the data."</i> '
             'Supporting points: in the bootstrap no setting is best more than 21% of the time ("winner\'s curse"); a minimum of 3 sits on a slope (13.6% → 12.7% → 12.2% flagged), 5 sits in the flat dip. '
             '<b>Never</b> argue with test-month results; that would be leakage.')],
    'b6': [('Would a higher minimum (10 or 12) not make more sense, since we look for related features?',
            'It would give <b>fewer</b> unusual windows, and mostly keep the windows of the large recurring group (the second hill at 10–13 features). '
            'It would <b>throw away</b> the windows where only a few features fire, and that is exactly where FC5 (1 feature) and FC7 (2 features), the two best warnings, live. '
            'The groups of related features are found by the <b>clustering</b> in Part 3, not by the minimum.')],
    'b9': [('What is CUSUM?',
            'A running score (Page 1954; Montgomery): every day adds how far the unusual-rate is <b>above</b> normal (minus a small allowance k); when the day is normal the score falls back towards 0. '
            'If the score climbs above the limit h, a lasting change is reported. Single odd days do not build up the score; a slow, lasting rise does.'),
           ('What does CUSUM improve for us?',
            '<ul><li>It <b>dates the drift objectively</b>: the change starts about 2 March 2025 (alarm 20 March), about 46 days before the April revision; weekly data shows a slow build-up from mid-December 2024.</li>'
            '<li>It separates <b>one-off extreme days</b> (25 Sep, 22 Oct 2024: no weekly alarm) from a <b>lasting change</b>.</li>'
            '<li>It justifies measuring chance <b>separately per period</b> in NB05, and gives the expert a concrete question (wear? response to the revision?).</li>'
            '<li>It does <b>not</b> change the windows or the groups.</li></ul>')],
    'b18': [('What do cross-sensor, same-sensor and mixed mean?',
             '<b>Cross-sensor:</b> every feature comes from a different physical sensor (they can be the same type: FC1 is 12 brake-cylinder sensors on 12 bogies). '
             '<b>Same-sensor:</b> one sensor described several ways (lowest, highest, average), e.g. FC13. '
             '<b>Mixed:</b> several sensors, some of them with several summaries, e.g. FC12 (main reservoir pressure on CW1 and CW2, each min/mean/max).')],
    'b19': [('The plot shows many failure percentages, but there are only 10 train and 11 test failures?',
             'The percentages are shares of <b>windows</b>, not numbers of failures. "2f · 63%" on FC7 = 2 features; 63% of the 134 windows where FC7 fires are trouble windows (84 windows, from only a few failures). '
             'How many failures a group covers is in NB05. (Cohesion uses the training months; the percentage uses all months; descriptive only.)'),
            ('So for FC6, 22% of its windows are failure windows?',
             'Yes: FC6 fires in 1,166 windows and 22% (about 256) are trouble windows, almost 4× the 6% average. But most of them are <b>during</b> failures (detection-only), so FC6 notices failures but does not warn early.')],
    'b21': [('What do silhouette "own measure" and "other measure" mean?',
             'Silhouette (−1 to 1) asks whether each feature is close to its own group and far from the others. <b>Own measure:</b> each method judged with its own distance: A 0.547, B 0.529, about equal. '
             '<b>Other measure</b> (the fairer test): A judged with correlation still 0.470, B judged with Jaccard only 0.256. '
             'Like two cooks who each like their own cake equally, but when they swap, only cake A is still liked. A\'s groups make sense both ways.'),
            ('What does "the exact borders differ (ARI 0.46)" mean?',
             'Like two teachers sorting students into teams: both keep friends together, but teacher B (13 teams) merges several of teacher A\'s 19 small teams. '
             'ARI: 1 = identical groupings, 0 = like random, 0.46 = same basic structure, different exact borders. An honest limitation, which is why NB06 checks robustness.'),
            ('Should we use correlation or stay with Jaccard?',
             'Stay with <b>Jaccard</b>: its groups also hold under correlation (0.47 vs 0.26), and it measures what we care about, sensors becoming <b>unusual together</b>, not normal co-movement (like passenger load). '
             'Do not argue mainly with "it separates failures better"; that would choose by the labels. Do not quote the 87% agreement without its chance level (77%).')],
    'b22': [('Why is FC16 so different (0.38 vs 0.95)?',
             'FC16 = the highest load pressure on bogie 2 of MW2 and MW4. Their values move together almost always (r = 0.95 even in normal windows) because both follow the <b>passenger load</b>. '
             'But each car\'s extreme peaks happen at different moments (MW2 unusual 101 times, MW4 71, both together 47). Also, with 101 vs 71 the highest possible Jaccard is 71/101 = 0.70. '
             'Like two people walking side by side: always the same direction, but they don\'t stumble at the same moment.'),
            ('What does Pearson r mean?',
             'How strongly two things go up and down together: +1 always together, 0 no link, −1 opposite (height and shoe size ≈ +0.7). '
             'In the plot box, r = 0.31 compares the two measures across the 16 groups: a weak link, and p = 0.24 means it could be luck.')],
    'b23': [('What do the p-value, Fisher\'s exact test and the chi-squared test do?',
             'All answer "could this be luck?". <b>p-value</b> = the chance of seeing this result if there were no real link; below 0.05 = probably real. '
             '<b>Fisher</b> computes that chance exactly from the 2×2 table (p = 2.7e-228, practically 0). <b>Chi-squared</b> compares expected (224) with observed (724) trouble windows among the unusual ones; a big gap = big χ². '
             'Two scales weighing the same thing: Fisher is exact (best for small numbers), chi-squared approximates (fine for large numbers).'),
            ('From which number is chi-squared "high"?',
             'For a 2×2 table: above <b>3.84</b> (p &lt; 0.05), 6.63 (p &lt; 0.01), 10.83 (p &lt; 0.001). Ours is 1,395. The limit depends on the table size, which is why one usually reads the p-value.'),
            ('Why is FC4 not significant, although its share is above the average?',
             'It has to be <b>clearly</b> above. FC4: 7.8% vs 6.0%, in only 490 windows, p = 0.065 (just above 0.05). FC5 has even fewer windows but 15.4%, p ≈ 0.0000000002. '
             'Like a coin: 60 heads in 100 can happen with a fair coin, 90 cannot. In NB05 FC4 just passes on the training months (q = 0.048) but fails on the test months: a weak, unstable signal.')],
    'b25': [('Which column is the most important?',
             '<b>p_24h</b>, read with <b>events_24h</b> and <b>expected_24h</b>: they count <b>failures</b> (not windows), compare with chance, and answer "does it warn before?". '
             'FC5: 15 vs 8.2 expected, p ≈ 0 ✅; FC7: 6 vs 1.7, p ≈ 0 ✅; FC14: 16 vs 16.4, p = 0.79 ❌. Second most useful: <b>pattern</b>. '
             'Careful with <b>enrich_…</b> (can look impressive, e.g. FC14\'s 5.5×) and with lead times of non-significant groups.')],
    'b26': [('How is the activation rate computed?',
             'Each calendar day is a <b>failure day</b> (it overlaps the 6.2 h before a failure up to its end; 26 days) or a <b>normal day</b> (333 days). '
             'Among the <b>unusual windows</b> of all failure days, the share in which the group fires is its failure-day rate; the same for normal days. '
             'Example: FC18 fires in 46% of the unusual windows on failure days vs 18% on normal days → +27.5 pp.')],
}

SUMMARY = """<h2>Summary in five sentences</h2>
<ol>
<li>NB04 uses 258 sensor features and compares every 6-minute window with normal behaviour in the training months (z-scores).</li>
<li>A window counts as unusual if at least 5 features are more than 3 spreads from normal; this follows the three-sigma rule, is in the top group by Youden's J (not significantly worse than the best) and sits in a stable dip of the data.</li>
<li>From early March 2025 the train looks unusual much more often (drift, dated by CUSUM), about six weeks before the April revision.</li>
<li>Features that are unusual together are grouped into 19 groups (FC1–FC19), using only the training months; Jaccard grouping was kept over correlation, and the groups are also visible in the real values.</li>
<li>A first look shows FC5 and FC7 fire before failures more than chance; the official test of all groups follows in NB05.</li>
</ol>"""

if __name__ == '__main__':
    render(nb, BLOCKS, QA, 'NB04 Study Guide',
           'Every output of <code>04_anomaly_detection_cluster_analysis.ipynb</code>, with its figure, a plain explanation and the questions you asked. '
           'NB04 finds the windows where the train behaves unusually and groups the sensors that become unusual together into 19 feature clusters (FC1–FC19). '
           'Whether a group really warns before failures is tested in NB05; NB06 checks the robustness.',
           SUMMARY, OUT)
    make_pdf(OUT, OUT.replace('.html', '.pdf'), os.path.join(tempfile.gettempdir(), 'study_print'))
