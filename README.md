# FYP Examiner Matching (SBERT + Harmony Search)

Matches each final-year project to the examiner whose expertise is closest in meaning, then uses
**Harmony Search** to give every student one examiner while keeping each lecturer's workload fair.

## Rules the system enforces

| Rule | How |
|---|---|
| One examiner per student | each position in the harmony holds one lecturer |
| Examiner ≠ supervisor | the supervisor is removed from that student's candidate list |
| Workload ≤ 20 per lecturer (supervise + examine, FYP1 + FYP2) | repair step moves students off full lecturers |
| Best match | maximise the total SBERT cosine score |
| Equal workload band (optional, on by default) | every lecturer within fair share ± spread |
| Fair workload | penalty `w × (L_max − L_min)` |

Objective: `Z = Σ s_ij·x_ij − w·(L_max − L_min)`

The 20-student limit is a **maximum**. Per-lecturer limits can be set with a `max_load` column.

## Files

| File | Purpose |
|---|---|
| `matcher.py` | engine: data cleaning, SBERT similarity, Harmony Search, CP-SAT benchmark, Excel export |
| `app.py` | Streamlit web app |
| `FYP_Examiner_Matching_Colab.ipynb` | Google Colab notebook (self-contained) |
| `sample_data/students.csv`, `sample_data/lecturers.csv` | fictional sample input / templates |
| `requirements.txt` | Python packages |

## Input format (CSV or Excel)

**students**: `student_id, name, fyp_level, project_title, project_description, supervisor`
(`project_description` is optional but improves matching; `fyp_level` = FYP1 or FYP2)

**lecturers**: `lecturer, expertise, max_load` (`max_load` is optional, default 20)

Supervisor names must match lecturer names. Case, dots and extra spaces are ignored, so `Dr. Chen` matches `dr chen`.

**Tip:** SBERT works best with real sentences. A 2–3 sentence project abstract and a short list of each
lecturer's research areas give clearer separation than bare titles. The fictional sample data uses
titles only, so its scores are low (about 0.1–0.4).

## Option A: Google Colab (free, easiest)
1. Go to https://colab.research.google.com → *File ▸ Upload notebook* → choose `FYP_Examiner_Matching_Colab.ipynb`.
2. *Runtime ▸ Run all*. It runs on sample data first.
3. To use your own data, set `USE_SAMPLE = False` in step 3, run again and upload your two files.
4. The last cell downloads `FYP_Examiner_Assignment.xlsx`.

## Option B: Web app on Streamlit Community Cloud (free, shareable link)
1. Create a GitHub repository and upload `app.py`, `matcher.py`, `requirements.txt` (and `sample_data/`).
2. Go to https://share.streamlit.io → *New app* → pick the repo, branch, and `app.py` → *Deploy*.
3. The first build takes a few minutes because of PyTorch and the SBERT model.

## Run locally
```powershell
pip install -r requirements.txt
```
```powershell
streamlit run app.py
```

## Harmony Search parameters
| Parameter | Default | Meaning |
|---|---|---|
| HMS | 20 | solutions kept in harmony memory |
| HMCR | 0.90 | chance of copying a student's examiner from memory |
| PAR | 0.30 | chance of pitch-adjusting that examiner to a neighbouring candidate in the student's ranked list |
| Bandwidth | 2 | how many ranks a pitch adjustment may move |
| Iterations | 2000 | new harmonies improvised |
| Local-search rate | 0.1 | share of new harmonies polished by swap / relocate / leveling moves (0 = classic HS) |
| Spread | ±1 | equal-workload band around the fair share (total supervision + students) / lecturers |

How a new harmony is made:
1. **Improvise**: memory consideration (HMCR), pitch adjustment (PAR), or a random pick.
2. **Build capacity-safe**: students are placed in random order. If the chosen examiner is full, the student gets their best examiner who still has room.
3. **Repair** the band minimum, then apply **local search** (hybrid HS) to a share of harmonies set by the local-search rate.
4. **Update memory**: replace the worst harmony if the new one is better.

Memory starts with one greedy solution and HMS−1 random ones. The final best harmony is always polished.

**Benchmark** (90 fictional students, 12 lecturers, SBERT MiniLM): hybrid HS reached the CP-SAT optimum
(Z = 20.0536, every lecturer at 15) in 6 out of 6 seeds, in about 6 s. Classic HS (`ls_rate = 0`) reached about 99.6–99.8%.
Turn on **CP-SAT** in the app to repeat this comparison on your own data, which is useful for the FYP report.
CP-SAT rounds scores to 3 decimals, so HS can occasionally show a tiny bit above 100%.

## Similarity models
* `all-MiniLM-L6-v2`: fast default
* `all-mpnet-base-v2`: more accurate, slower
* `paraphrase-multilingual-MiniLM-L12-v2`: use when titles are in Malay
* `TF-IDF`: keyword fallback, needs no download
