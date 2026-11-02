# Expert Interview Guide: Verification of the Feature Clusters (RQ1)

**Purpose:** verify the 19 data-driven feature clusters (FC1–FC19) with the Wiener Linien domain expert (RQ1), using a described and reproducible method (dean's feedback on the proposal: "method for interviews").

---

## 1. Method in short

| Element | Choice | Source / reason |
|---|---|---|
| Interview type | **Semi-structured expert interview**: fixed guide, open follow-up questions | Bogner, Littig & Menz (2009) |
| Guide development | Five phases: prerequisites, prior knowledge, preliminary guide, **pilot test**, final guide | Kallio et al. (2016) |
| Participant | **One key informant** (Wiener Linien expert for the pneumatic system) | Key-informant technique (Tremblay 1957) |
| Structure | **Two rounds**: round 1 blind (expert rates the clusters without our results), round 2 informed (our evidence is shown and discussed) | Prevents the expert from simply confirming our statistics |
| Validity check | **3–4 hidden control groups** (randomly assembled sensor groups) mixed in with the 19 FCs | Replaces agreement between several experts: shows whether the ratings separate real from random groups |
| Consistency check | **2–3 FCs asked twice** (in a different position) | Shows whether the ratings are stable |
| Recording | Filled sheet + notes; audio only with consent | Traceability |
| Analysis | Quantitative (share of verified FCs, control groups, consistency) + qualitative content analysis of the reasons (Kuckartz 2014; Mayring 2014) + triangulation with the data evidence | Mixed evidence for RQ1 |

**Limitation (stated in the thesis):** a single expert means the result is a verification by one domain expert, not a consensus; individual bias is possible. The blind round, the control groups and the consistency check reduce and make visible, but do not remove, this limitation.

---

## 2. Before the meeting

1. **Pilot test** the guide and the round-1 sheet with Andreas Steiner (≈ 30 min): are the questions clear, is the timing realistic? Adjust afterwards (Kallio phase 4).
2. Prepare the **round-1 sheet** (blind): for each of the 19 FCs plus the control groups, in random order: sensor list (names, cars, bogies), the cluster's activation plot over the year, and its raw-signal example where available. **No** verdicts, p-values, lifts, KEEP/REVIEW or failure-type columns. FCs and control groups get neutral codes (G01–G23), so the expert cannot tell them apart.
3. Prepare the **round-2 material**: the expert review PDF and the full verification sheet (verdict, p-values, test check, cohesion, failure-type enrichment).
4. Send the expert a short **information sheet** in advance: purpose, duration (≈ 90 min), use of the answers in the thesis, anonymity (role only, no name), optional audio recording.

---

## 3. Meeting procedure (≈ 90 minutes)

| Time | Part | What happens |
|---|---|---|
| 5 min | Introduction | Purpose, consent (notes / audio), how the groups were found (sensors that become unusual together), no results yet |
| 40 min | **Round 1: blind rating** | The expert rates each group G01–G23 with the questions in section 4 |
| 5 min | Break | |
| 30 min | **Round 2: informed discussion** | Our results are shown; questions in section 5 |
| 10 min | General questions | Questions in section 6 |

---

## 4. Round 1: questions per group (blind)

For each group the expert answers on the sheet:

| # | Question | Answer format |
|---|---|---|
| Q1 | Do these sensors **physically belong together** (same component, same pneumatic circuit, same control chain)? | yes / partly / no |
| Q2 | **Which subsystem** do they belong to? | free text (e.g. air supply, brake, leveling, other, unclear) |
| Q3 | Is there a **plausible situation** in which they become unusual together (a fault mode, or a normal operating situation such as parking)? | free text |
| Q4 | Would this group be **useful as an early warning**? | yes / no / unsure |
| Q5 | **Why?** (reason for Q1–Q4) | free text, the most important answer |
| Q6 | How **confident** are you? | low / medium / high |

The interviewer asks follow-up questions on Q3 and Q5 and does **not** comment on whether a group is a real FC or a control group.

---

## 5. Round 2: informed discussion

1. Here is what the data says for each FC (verdict, significance, test check, cohesion, failure types). **Where does it agree or disagree** with your rating from round 1?
2. **FC5, FC7, FC8** are the clusters that also warn before the test-period failures. Are they physically plausible warnings?
3. Which groups do you think are **normal operating situations** rather than faults (candidates: FC1, FC2, FC19, FC18)?
4. The **loose groups** (FC11, FC13–FC17, mostly load-pressure and load-signal sensors) fire together less consistently. Is that expected for these sensors?
5. Which of the 3–4 groups were the **control groups**, in your view? (Only after revealing which ones they were.)

---

## 6. General questions

1. **Change from March 2025:** the data changes from early March 2025, about six weeks before the 17–22 April revision, with a slow build-up from December 2024. Is a cause known (wear, a component change, a software or operating change)? Was the revision a response to it?
2. **The April revision:** which parts were replaced or adjusted? Does it create a "new normal"?
3. **FC7/FC8, spring-brake pressure near 0:** mostly while standing. Pressure loss, sensor dropout, or the parking brake (spring brakes apply when the air is released)?
4. **FC2, MW4 load signal minimum:** it is unusual in 9% of normal windows. Is this sensor known to drop out or behave unusually?
5. **Compressor sides:** after the revision the CW2 compressor recovers and CW1 degrades (NB05 §7.4). Is that consistent with what was done?
6. Which **sensors or relationships** are missing that you would expect in a warning for these failure types?

---

## 7. Analysis after the meeting

| Result | How it is computed | Use in the thesis |
|---|---|---|
| **Verified FCs** | Share of the 19 FCs rated "physically coherent" (Q1 = yes, or yes + partly), overall and for the KEEP FCs (NB05 §8, RQ1 verification metric) | Main RQ1 result |
| **Control-group check** | Q1 ratings of the control groups vs the FCs | Validity of the single-expert ratings |
| **Consistency** | Agreement of the repeated groups | Reliability of the ratings |
| **Subsystem assignment** | Q2 per FC → the subsystem datasets for RQ2 (`build_subsystem_datasets()` in NB05) | Input to Phase 2 |
| **Reasons** | Q3/Q5 and the round-2 notes, coded with structured qualitative content analysis (categories: physical coupling, fault mode, operating situation, sensor artefact, unclear) | Interpretation, discussion chapter |
| **Triangulation** | Expert answers vs data evidence (significance, failure types, raw signals, air-system back-calculation) | Strength of each conclusion |

---

## References

- Bogner, A., Littig, B., & Menz, W. (Eds.) (2009). *Interviewing Experts*. Palgrave Macmillan. https://doi.org/10.1057/9780230244276
- Kallio, H., Pietilä, A.-M., Johnson, M., & Kangasniemi, M. (2016). Systematic methodological review: developing a framework for a qualitative semi-structured interview guide. *Journal of Advanced Nursing*, 72(12), 2954–2965. https://doi.org/10.1111/jan.13031
- Tremblay, M.-A. (1957). The key informant technique: A nonethnographic application. *American Anthropologist*, 59(4), 688–701.
- Kuckartz, U. (2014). *Qualitative Text Analysis: A Guide to Methods, Practice and Using Software*. SAGE.
- Mayring, P. (2014). *Qualitative Content Analysis: Theoretical Foundation, Basic Procedures and Software Solution*. Klagenfurt.
