"""
FYP Examiner Matching
=====================
SBERT semantic similarity  +  Harmony Search optimisation
(+ optional CP-SAT exact solver as a benchmark).

Workflow (same as the poster):
  1. Input   : student projects + lecturer (panel) expertise
  2. SBERT   : convert text -> embeddings
  3. Cosine  : similarity score S[i, j] between student i and lecturer j
  4. Optimise: assign exactly ONE examiner to every student so that
                 - the examiner is not the student's own supervisor
                 - total workload (supervising + examining, FYP1 + FYP2)
                   of each lecturer  <=  max load (default 20)
                 - total matching score is maximised
                 - workload is balanced  (minimise L_max - L_min)

     Objective (maximise):
        Z = sum_i S[i, a_i]  -  w * (L_max - L_min)  -  P * (capacity overflow)

This file has no UI code, so it can be used from Streamlit, Colab or a script.
"""
from __future__ import annotations

import io
import re
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_MAX_LOAD = 20
OVERFLOW_PENALTY = 1000.0

SBERT_MODELS = [
    "all-MiniLM-L6-v2",          # fast, small (~90 MB) - good default
    "all-mpnet-base-v2",         # more accurate, slower (~420 MB)
    "paraphrase-multilingual-MiniLM-L12-v2",  # handles Malay + English
    "TF-IDF (no SBERT)",         # keyword fallback, no download needed
]

# ----------------------------------------------------------------------------
# 1. INPUT: read + clean data
# ----------------------------------------------------------------------------
STUDENT_ALIASES = {
    "student_id": ["student_id", "student_no", "matric", "matric_no", "matric_number", "id", "no"],
    "name": ["name", "student", "student_name"],
    "fyp_level": ["fyp_level", "fyp", "level", "course", "course_code"],
    "project_title": ["project_title", "title", "project", "project_name"],
    "project_description": ["project_description", "description", "abstract", "summary", "objective"],
    "supervisor": ["supervisor", "sv", "supervisor_name"],
}
LECTURER_ALIASES = {
    "lecturer": ["lecturer", "name", "lecturer_name", "panel", "panel_member", "examiner"],
    "expertise": ["expertise", "area", "research_area", "research_interest", "specialisation",
                  "specialization", "keywords"],
    "max_load": ["max_load", "capacity", "quota", "max_students"],
}


def norm_name(s) -> str:
    """Normalise a person's name so 'Dr. Chen ' == 'dr chen'."""
    s = re.sub(r"[. ]", " ", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def _rename(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    mapping = {}
    for std, alts in aliases.items():
        for a in alts:
            if a in df.columns and a not in mapping:
                mapping[a] = std
                break
    return df.rename(columns=mapping)


def read_table(file) -> pd.DataFrame:
    """Read CSV or Excel from a path or an uploaded file object."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def _fyp_level(v) -> str:
    s = str(v).upper()
    if s in ("", "NAN", "NONE"):
        return "FYP1"
    if "2" in s:
        return "FYP2"
    if "1" in s:
        return "FYP1"
    return str(v).strip()


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename(df, STUDENT_ALIASES)
    missing = [c for c in ("name", "project_title", "supervisor") if c not in df.columns]
    if missing:
        raise ValueError(f"Student file is missing column(s): {missing}. "
                         f"Required: name, project_title, supervisor.")
    df = df[df["name"].notna() & df["project_title"].notna()].copy()
    if "student_id" not in df.columns:
        df["student_id"] = [f"S{i + 1:03d}" for i in range(len(df))]
    if "fyp_level" not in df.columns:
        df["fyp_level"] = "FYP1"
    if "project_description" not in df.columns:
        df["project_description"] = ""
    df["fyp_level"] = df["fyp_level"].map(_fyp_level)
    for c in ("student_id", "name", "project_title", "project_description", "supervisor"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df[list(STUDENT_ALIASES)].reset_index(drop=True)


def clean_lecturers(df: pd.DataFrame, default_max_load: int = DEFAULT_MAX_LOAD) -> pd.DataFrame:
    df = _rename(df, LECTURER_ALIASES)
    missing = [c for c in ("lecturer", "expertise") if c not in df.columns]
    if missing:
        raise ValueError(f"Lecturer file is missing column(s): {missing}. "
                         f"Required: lecturer, expertise.")
    df = df[df["lecturer"].notna()].copy()
    if "max_load" not in df.columns:
        df["max_load"] = default_max_load
    df["max_load"] = pd.to_numeric(df["max_load"], errors="coerce").fillna(default_max_load).astype(int)
    df["lecturer"] = df["lecturer"].astype(str).str.strip()
    df["expertise"] = df["expertise"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset="lecturer", keep="first")
    return df[list(LECTURER_ALIASES)].reset_index(drop=True)


def student_text(row) -> str:
    t = row["project_title"]
    d = row.get("project_description", "")
    return f"{t}. {d}".strip() if d else t


# ----------------------------------------------------------------------------
# 2 + 3. SBERT embeddings + cosine similarity
# ----------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer  # imported lazily
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _tfidf_similarity(a_texts, b_texts) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(list(a_texts) + list(b_texts))
    A, B = X[: len(a_texts)], X[len(a_texts):]
    return (A @ B.T).toarray()  # rows are L2-normalised -> dot = cosine


def compute_similarity(student_texts, lecturer_texts, model_name=SBERT_MODELS[0]):
    """Return (S, method). S[i, j] = cosine(student_i, lecturer_j)."""
    if not model_name.upper().startswith("TF-IDF"):
        try:
            model = load_sbert(model_name)
            A = model.encode(list(student_texts), normalize_embeddings=True, show_progress_bar=False)
            B = model.encode(list(lecturer_texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(A @ B.T, dtype=float), f"SBERT ({model_name})"
        except ImportError:
            warnings.warn("sentence-transformers not installed - falling back to TF-IDF.")
    return _tfidf_similarity(student_texts, lecturer_texts), "TF-IDF cosine"


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ----------------------------------------------------------------------------
# 4. Optimisation problem
# ----------------------------------------------------------------------------
@dataclass
class Problem:
    S: np.ndarray            # n x m similarity
    sup_idx: np.ndarray      # n, index of supervisor in lecturer list (-1 = not a panel member)
    sup_load: np.ndarray     # m, number of students each lecturer supervises
    max_load: np.ndarray     # m, max total workload
    capacity: np.ndarray     # m, max number of students each lecturer may examine
    lower: np.ndarray        # m, min number of students each lecturer must examine (load band)
    fairness_weight: float
    domain: list = field(default_factory=list)   # allowed examiners per student, best first
    D: np.ndarray = None     # padded domain matrix  n x m
    dlen: np.ndarray = None  # domain sizes
    rank: np.ndarray = None  # n x m, rank of lecturer j for student i (-1 = not allowed)
    warnings: list = field(default_factory=list)
    band: tuple | None = None  # (L_low, L_high) workload band actually used, None = no band

    @property
    def n(self):
        return self.S.shape[0]

    @property
    def m(self):
        return self.S.shape[1]


def build_problem(students: pd.DataFrame, lecturers: pd.DataFrame, S: np.ndarray,
                  fairness_weight: float = 0.5, spread: int | None = 1) -> Problem:
    """
    spread : workload band around the fair share.
             fair share T = (total supervision + number of students) / number of lecturers
             every lecturer must end with  floor(T) - spread <= L_j <= ceil(T) + spread
             (never above their max_load). None = no band, only max_load.
             If the band is impossible it is widened automatically (with a warning).
    """
    n, m = S.shape
    idx = {norm_name(l): j for j, l in enumerate(lecturers["lecturer"])}
    sup_idx = np.array([idx.get(norm_name(s), -1) for s in students["supervisor"]], dtype=int)
    warns = []
    unknown = sorted({s for s, k in zip(students["supervisor"], sup_idx) if k < 0})
    if unknown:
        warns.append(f"Supervisor(s) not in the lecturer list (their supervision load is not counted): "
                     f"{', '.join(unknown)}")
    sup_load = np.bincount(sup_idx[sup_idx >= 0], minlength=m)
    max_load = lecturers["max_load"].to_numpy(dtype=int)
    capacity = np.maximum(max_load - sup_load, 0)
    for j in np.where(sup_load > max_load)[0]:
        warns.append(f"{lecturers['lecturer'][j]} already supervises {sup_load[j]} students "
                     f"(> max load {max_load[j]}); they will not be given any examining.")
    if capacity.sum() < n:
        warns.append(f"Not enough examiner capacity: {n} students need an examiner but only "
                     f"{capacity.sum()} slots remain. Increase max load or add lecturers.")

    lower, band = np.zeros(m, dtype=int), None
    if spread is not None and capacity.sum() >= n:
        T = (sup_load.sum() + n) / m
        s = int(spread)
        while True:
            lo_L, hi_L = int(np.floor(T)) - s, int(np.ceil(T)) + s
            hi = np.minimum(capacity, np.maximum(hi_L - sup_load, 0))
            lo = np.minimum(np.maximum(lo_L - sup_load, 0), hi)
            if lo.sum() <= n <= hi.sum():
                break
            s += 1
        if s != int(spread):
            warns.append(f"Workload band ±{spread} is not possible with this data; widened to ±{s}.")
        capacity, lower, band = hi, lo, (lo_L, hi_L)

    domain, D, rank = [], np.zeros((n, m), dtype=int), -np.ones((n, m), dtype=int)
    dlen = np.zeros(n, dtype=int)
    for i in range(n):
        order = [j for j in np.argsort(-S[i]) if j != sup_idx[i]]
        domain.append(np.array(order, dtype=int))
        dlen[i] = len(order)
        D[i, : len(order)] = order
        rank[i, order] = np.arange(len(order))
    return Problem(S, sup_idx, sup_load, max_load, capacity, lower, float(fairness_weight),
                   domain, D, dlen, rank, warns, band)


def evaluate(prob: Problem, a: np.ndarray):
    """Return dict with fitness (maximise), total similarity, load gap, violations."""
    sim = float(prob.S[np.arange(prob.n), a].sum())
    exam = np.bincount(a, minlength=prob.m)
    load = prob.sup_load + exam
    overflow = int(np.maximum(exam - prob.capacity, 0).sum() + np.maximum(prob.lower - exam, 0).sum())
    gap = int(load.max() - load.min())
    fitness = sim - prob.fairness_weight * gap - OVERFLOW_PENALTY * overflow
    return {"fitness": fitness, "similarity": sim, "gap": gap, "overflow": overflow,
            "L_max": int(load.max()), "L_min": int(load.min())}


def repair(prob: Problem, a: np.ndarray) -> np.ndarray:
    """
    Make a harmony feasible with the smallest loss of matching score:
      1. examiners above their maximum give away students to the best free alternative
      2. examiners below their minimum (workload band) take students from others
    """
    a = a.copy()
    S, allowed = prob.S, prob.rank >= 0
    exam = np.bincount(a, minlength=prob.m)
    # 1. overflow
    for j in np.where(exam > prob.capacity)[0]:
        while exam[j] > prob.capacity[j]:
            idx = np.where(a == j)[0]
            free = exam < prob.capacity
            free[j] = False
            cand = np.where(allowed[idx] & free, S[idx], -np.inf)
            k_best = cand.argmax(axis=1)
            gain = cand[np.arange(len(idx)), k_best]
            if not np.isfinite(gain).any():
                break  # cannot repair (not enough capacity)
            r = int(gain.argmax())  # smallest loss = largest new score - same old score
            i, k = idx[r], k_best[r]
            a[i] = k
            exam[j] -= 1
            exam[k] += 1
    # 2. underflow
    rows = np.arange(prob.n)
    for k in np.where(exam < prob.lower)[0]:
        while exam[k] < prob.lower[k]:
            donor_ok = (exam[a] > prob.lower[a]) & (a != k) & allowed[:, k]
            if not donor_ok.any():
                break
            loss = np.where(donor_ok, S[rows, a] - S[:, k], np.inf)
            i = int(loss.argmin())
            exam[a[i]] -= 1
            a[i] = k
            exam[k] += 1
    return a


def local_search(prob: Problem, a: np.ndarray, max_rounds: int | None = None) -> np.ndarray:
    """
    Hill-climbing polish used inside Harmony Search (hybrid HS).
      swap     : exchange the examiners of two students (workloads unchanged)
      relocate : give a student to another examiner with free room
    Only moves that respect all constraints and increase Z are accepted.
    """
    a = a.copy()
    n, rows = prob.n, np.arange(prob.n)
    S, allowed = prob.S, prob.rank >= 0
    best_f = evaluate(prob, a)["fitness"]
    for _ in range(max_rounds or 2 * n):
        improved = False
        cur = S[rows, a]
        # --- best swap
        X = S[:, a]                                   # X[i, i2] = S[i, a[i2]]
        ok = allowed[:, a]
        G = np.where(ok & ok.T, X + X.T - cur[:, None] - cur[None, :], -np.inf)
        i, i2 = np.unravel_index(int(G.argmax()), G.shape)
        if G[i, i2] > 1e-12:
            a[i], a[i2] = a[i2], a[i]
            best_f += G[i, i2]
            improved = True
        # --- best relocate (checked with the full objective because it changes workloads)
        exam = np.bincount(a, minlength=prob.m)
        R = np.where(allowed & (exam < prob.capacity)[None, :] & (exam[a] > prob.lower[a])[:, None],
                     S - S[rows, a][:, None], -np.inf)
        R[rows, a] = -np.inf
        i, k = np.unravel_index(int(R.argmax()), R.shape)
        if np.isfinite(R[i, k]):
            old = a[i]
            a[i] = k
            f = evaluate(prob, a)["fitness"]
            if f > best_f + 1e-12:
                best_f, improved = f, True
            else:
                a[i] = old
        # --- leveling chain (only when nothing else helps)
        if not improved and prob.fairness_weight > 0:
            b, f = _leveling_chain(prob, a)
            if f > best_f + 1e-12:
                a, best_f, improved = b, f, True
        if not improved:
            break
    return a


def _leveling_chain(prob: Problem, a: np.ndarray, max_moves: int = 50):
    """Repeatedly move the cheapest student from a busiest to a least-busy lecturer;
    return the best prefix of that chain. Needed because L_max - L_min only drops
    after several single moves."""
    b = a.copy()
    rows = np.arange(prob.n)
    allowed = prob.rank >= 0
    best_b, best_f = a.copy(), evaluate(prob, a)["fitness"]
    for _ in range(max_moves):
        exam = np.bincount(b, minlength=prob.m)
        load = prob.sup_load + exam
        if load.max() - load.min() < 2:
            break
        src = (load == load.max()) & (exam > prob.lower)
        dst = (load == load.min()) & (exam < prob.capacity)
        if not src.any() or not dst.any():
            break
        L = np.where(allowed & dst[None, :] & src[b][:, None], prob.S[rows, b][:, None] - prob.S, np.inf)
        i, k = np.unravel_index(int(L.argmin()), L.shape)
        if not np.isfinite(L[i, k]):
            break
        b[i] = k
        f = evaluate(prob, b)["fitness"]
        if f > best_f + 1e-12:
            best_b, best_f = b.copy(), f
    return best_b, best_f


def greedy_assignment(prob: Problem) -> np.ndarray:
    """Everyone gets their best non-supervisor match, then repair capacity."""
    return repair(prob, prob.D[:, 0].copy())


# ----------------------------------------------------------------------------
# 5. Harmony Search
# ----------------------------------------------------------------------------
@dataclass
class HSResult:
    assignment: np.ndarray
    metrics: dict
    history: list
    runtime: float
    method: str = "Harmony Search"


def harmony_search(prob: Problem, hms: int = 20, hmcr: float = 0.90, par: float = 0.30,
                   bandwidth: int = 2, max_iter: int = 2000, seed: int = 42,
                   seed_greedy: bool = True, ls_rate: float = 0.1, ls_rounds: int | None = None,
                   progress=None) -> HSResult:
    """
    Harmony   : integer vector a, a[i] = examiner index for student i
    HMCR      : prob. of taking a value from Harmony Memory (else random)
    PAR       : prob. of pitch-adjusting a memory value
                (move to a neighbouring candidate in the student's ranked list,
                 within +/- bandwidth ranks)
    Every new harmony is built so that capacity is never exceeded (and repaired
    for the minimum of the workload band), then replaces the worst harmony in
    memory if it is better.
    ls_rate   : fraction of new harmonies polished by local_search (hybrid HS);
                0 = classic HS. The final best harmony is always polished fully.
    ls_rounds : max improving moves per local search inside the loop (None = until no
                improving move is left).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = prob.n
    rows = np.arange(n)
    dom = [d.tolist() for d in prob.domain]
    cap = prob.capacity.tolist()
    need_lower_fix = bool(prob.lower.any())

    def build(ranks):
        """Turn chosen candidate ranks into a harmony that never exceeds capacity.
        Students are placed in random order; if the chosen examiner is already full,
        the student gets their best examiner that still has room."""
        ranks = ranks.tolist()
        exam = [0] * prob.m
        out = [0] * n
        broken = False
        for i in rng.permutation(n).tolist():
            k = dom[i][ranks[i]]
            if exam[k] >= cap[k]:
                for k2 in dom[i]:
                    if exam[k2] < cap[k2]:
                        k = k2
                        break
                else:
                    broken = True
            exam[k] += 1
            out[i] = k
        a = np.array(out)
        if broken or need_lower_fix:
            a = repair(prob, a)
        return a

    # --- initialise harmony memory: (optional) greedy + random harmonies
    HM = []
    if seed_greedy:
        HM.append(greedy_assignment(prob))
    while len(HM) < hms:
        HM.append(build(np.floor(rng.random(n) * prob.dlen).astype(int)))
    HM = np.array(HM)
    fit = np.array([evaluate(prob, h)["fitness"] for h in HM])
    history = [float(fit.max())]
    seen = {h.tobytes() for h in HM}

    step = max(1, max_iter // 100)
    for it in range(max_iter):
        # --- improvise: choose a candidate rank for every student
        use_mem = rng.random(n) < hmcr
        r = prob.rank[rows, HM[rng.integers(hms, size=n), rows]]       # memory consideration
        adj = use_mem & (rng.random(n) < par)                          # pitch adjustment
        r[adj] += rng.integers(-bandwidth, bandwidth + 1, size=int(adj.sum()))
        rnd = ~use_mem                                                 # random selection
        r[rnd] = np.floor(rng.random(int(rnd.sum())) * prob.dlen[rnd]).astype(int)
        r = np.clip(r, 0, prob.dlen - 1)

        new = build(r)
        if ls_rate > 0 and rng.random() < ls_rate:
            new = local_search(prob, new, max_rounds=ls_rounds)
        key = new.tobytes()
        if key not in seen:
            f = evaluate(prob, new)["fitness"]
            worst = int(fit.argmin())
            if f > fit[worst]:
                seen.discard(HM[worst].tobytes())
                HM[worst], fit[worst] = new, f
                seen.add(key)
        history.append(float(fit.max()))
        if progress and it % step == 0:
            progress(it / max_iter)

    best = local_search(prob, HM[int(fit.argmax())])
    history.append(max(history[-1], evaluate(prob, best)["fitness"]))
    if progress:
        progress(1.0)
    return HSResult(best, evaluate(prob, best), history, time.time() - t0)


# ----------------------------------------------------------------------------
# 6. Optional: CP-SAT exact solver (benchmark, needs `ortools`)
# ----------------------------------------------------------------------------
def solve_cpsat(prob: Problem, time_limit: float = 30.0, scale: int = 1000) -> HSResult | None:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None
    t0 = time.time()
    mdl = cp_model.CpModel()
    x = {(i, j): mdl.NewBoolVar(f"x_{i}_{j}") for i in range(prob.n) for j in prob.domain[i]}
    for i in range(prob.n):
        mdl.AddExactlyOne(x[i, j] for j in prob.domain[i])
    L = []
    for j in range(prob.m):
        exam = sum(x[i, j] for i in range(prob.n) if (i, j) in x)
        mdl.Add(exam <= int(prob.capacity[j]))
        mdl.Add(exam >= int(prob.lower[j]))
        Lj = mdl.NewIntVar(0, 10_000, f"L_{j}")
        mdl.Add(Lj == int(prob.sup_load[j]) + exam)
        L.append(Lj)
    Lmax, Lmin = mdl.NewIntVar(0, 10_000, "Lmax"), mdl.NewIntVar(0, 10_000, "Lmin")
    mdl.AddMaxEquality(Lmax, L)
    mdl.AddMinEquality(Lmin, L)
    mdl.Maximize(sum(int(round(prob.S[i, j] * scale)) * v for (i, j), v in x.items())
                 - int(round(prob.fairness_weight * scale)) * (Lmax - Lmin))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = 8
    status = solver.Solve(mdl)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    a = np.array([next(j for j in prob.domain[i] if solver.Value(x[i, j])) for i in range(prob.n)])
    label = "CP-SAT (optimal)" if status == cp_model.OPTIMAL else "CP-SAT (time limit, feasible)"
    return HSResult(a, evaluate(prob, a), [], time.time() - t0, label)


# ----------------------------------------------------------------------------
# 7. Output tables + Excel export
# ----------------------------------------------------------------------------
def assignment_table(students, lecturers, prob: Problem, a) -> pd.DataFrame:
    names = lecturers["lecturer"].to_numpy()
    rows = np.arange(prob.n)
    best = prob.D[:, 0]
    df = students[["student_id", "name", "fyp_level", "project_title", "supervisor"]].copy()
    df["examiner"] = names[a]
    df["matching_score"] = prob.S[rows, a].round(4)
    df["examiner_rank"] = prob.rank[rows, a] + 1          # 1 = best possible examiner
    df["best_possible_examiner"] = names[best]
    df["best_possible_score"] = prob.S[rows, best].round(4)
    return df


def workload_table(students, lecturers, prob: Problem, a) -> pd.DataFrame:
    m = prob.m
    lv = students["fyp_level"].to_numpy()
    out = pd.DataFrame({"lecturer": lecturers["lecturer"]})
    for lvl in sorted(set(lv)):
        k = prob.sup_idx[(lv == lvl) & (prob.sup_idx >= 0)]
        out[f"supervise_{lvl}"] = np.bincount(k, minlength=m)
    for lvl in sorted(set(lv)):
        out[f"examine_{lvl}"] = np.bincount(a[lv == lvl], minlength=m)
    out["total_supervise"] = prob.sup_load
    out["total_examine"] = np.bincount(a, minlength=m)
    out["total_workload"] = out["total_supervise"] + out["total_examine"]
    out["max_load"] = prob.max_load
    out["remaining"] = out["max_load"] - out["total_workload"]
    out["avg_matching_score"] = [round(float(prob.S[a == j, j].mean()), 4) if (a == j).any() else None
                                 for j in range(m)]
    return out


def similarity_table(students, lecturers, S) -> pd.DataFrame:
    df = pd.DataFrame(S.round(4), columns=lecturers["lecturer"])
    df.insert(0, "project_title", students["project_title"].values)
    df.insert(0, "name", students["name"].values)
    df.insert(0, "student_id", students["student_id"].values)
    return df


def summary_table(results: dict, method: str, n_students: int, n_lect: int,
                  band: tuple | None = None) -> pd.DataFrame:
    rows = [("Students", n_students), ("Lecturers", n_lect), ("Similarity method", method),
            ("Workload band (L_low - L_high)", f"{band[0]} - {band[1]}" if band else "none (max load only)")]
    for name, r in results.items():
        m = r.metrics
        rows += [(f"[{name}] objective Z", round(m["fitness"], 4)),
                 (f"[{name}] total matching score", round(m["similarity"], 4)),
                 (f"[{name}] average matching score", round(m["similarity"] / n_students, 4)),
                 (f"[{name}] workload L_max / L_min", f"{m['L_max']} / {m['L_min']}"),
                 (f"[{name}] fairness gap L_max - L_min", m["gap"]),
                 (f"[{name}] capacity violations", m["overflow"]),
                 (f"[{name}] runtime (s)", round(r.runtime, 2))]
    return pd.DataFrame(rows, columns=["item", "value"])


def export_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = "A2"
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 8. One-call pipeline (handy for Colab / scripts)
# ----------------------------------------------------------------------------
def run_pipeline(students_raw: pd.DataFrame, lecturers_raw: pd.DataFrame,
                 model_name: str = SBERT_MODELS[0], max_load: int = DEFAULT_MAX_LOAD,
                 fairness_weight: float = 0.5, spread: int | None = 1, hs_params: dict | None = None,
                 run_cpsat: bool = False, cpsat_time: float = 30.0):
    students = clean_students(students_raw)
    lecturers = clean_lecturers(lecturers_raw, max_load)
    S, method = compute_similarity([student_text(r) for _, r in students.iterrows()],
                                   lecturers["expertise"].tolist(), model_name)
    prob = build_problem(students, lecturers, S, fairness_weight, spread)
    results = {"Harmony Search": harmony_search(prob, **(hs_params or {}))}
    if run_cpsat:
        cp = solve_cpsat(prob, cpsat_time)
        if cp is not None:
            results["CP-SAT"] = cp
    hs = results["Harmony Search"]
    out = {
        "Assignment": assignment_table(students, lecturers, prob, hs.assignment),
        "Workload": workload_table(students, lecturers, prob, hs.assignment),
        "Similarity Matrix": similarity_table(students, lecturers, S),
        "Summary": summary_table(results, method, len(students), len(lecturers), prob.band),
    }
    return {"students": students, "lecturers": lecturers, "S": S, "method": method,
            "problem": prob, "results": results, "tables": out}


# ----------------------------------------------------------------------------
# 9. Sample data (fictional) so the system can be tried immediately
# ----------------------------------------------------------------------------
_SAMPLE_LECTURERS = [
    ("Dr Chen", "Cybersecurity, digital forensics, network security, phishing detection and threat analysis."),
    ("Dr Devi", "Web application development, database systems, software engineering and user interface design."),
    ("Dr Eva", "Artificial intelligence, data analytics, machine learning and information systems."),
    ("Dr Aminah", "Mobile application development, Android, iOS, cross-platform apps and usability."),
    ("Dr Faizal", "Internet of Things, embedded systems, sensors, Arduino and smart home automation."),
    ("Dr Lim", "Computer vision, image processing, deep learning and object detection."),
    ("Dr Kumar", "Natural language processing, text mining, sentiment analysis and chatbots."),
    ("Dr Siti", "E-learning, educational technology, gamification and multimedia learning."),
    ("Dr Hafiz", "Cloud computing, distributed systems, DevOps and blockchain applications."),
    ("Dr Wong", "Operations research, optimisation, scheduling, metaheuristics and decision support systems."),
    ("Dr Nurul", "Health informatics, medical decision support and healthcare information systems."),
    ("Dr Rahman", "Geographic information systems, spatial analysis, location-based services and mapping."),
]
_SAMPLE_TOPICS = {
    "Dr Chen": ["Intelligent Web-Based Forensic System for Detecting Phishing Threats",
                "Network Intrusion Detection Dashboard", "Malware Classification Tool",
                "Password Strength and Breach Checker", "Secure File Sharing System with Encryption"],
    "Dr Devi": ["Online Booking System with Database Management", "Inventory Management Web Portal",
                "Student Club Management System", "E-Commerce Website with Admin Panel",
                "Responsive Web Interface for Event Registration"],
    "Dr Eva": ["Machine Learning Model to Predict Student Dropout", "Sales Forecasting Analytics Dashboard",
               "Customer Churn Prediction Using Data Mining", "Crop Yield Prediction with Regression",
               "Credit Risk Classification with Random Forest"],
    "Dr Aminah": ["Android Mobile App for Campus Navigation", "Cross-Platform Flutter App for Food Ordering",
                  "Mobile Expense Tracker App", "iOS Fitness Tracking Application",
                  "Mobile App for Mosque Event Notification"],
    "Dr Faizal": ["IoT Smart Home Monitoring with Arduino", "Sensor-Based Flood Early Warning System",
                  "Smart Parking System Using IoT", "IoT Plant Watering Automation",
                  "Embedded Air Quality Monitoring Device"],
    "Dr Lim": ["Deep Learning Detection of Plant Leaf Disease from Images", "Face Recognition Attendance System",
               "Vehicle Plate Recognition Using Image Processing", "Object Detection for Retail Shelf Monitoring",
               "Fruit Ripeness Classification with CNN"],
    "Dr Kumar": ["Sentiment Analysis of Malaysian Product Reviews", "Chatbot for University Student Enquiries",
                 "Fake News Detection Using Text Mining", "Automatic Essay Scoring with NLP",
                 "Malay Text Summarisation Tool"],
    "Dr Siti": ["Gamified E-Learning Platform for Primary Mathematics", "Interactive Multimedia Courseware for Science",
                "Augmented Reality Learning App for Children", "Online Quiz System with Gamification",
                "Educational Game for Learning Jawi"],
    "Dr Hafiz": ["Blockchain-Based Certificate Verification System", "Cloud-Based Document Management Service",
                 "Serverless Microservices for Online Voting", "DevOps Pipeline for Web Deployment",
                 "Distributed Cloud Storage Monitoring Tool"],
    "Dr Wong": ["Exam Timetable Scheduling Using Harmony Search", "Vehicle Routing Optimisation for Delivery",
                "Genetic Algorithm for Nurse Rostering", "Decision Support System for Supplier Selection",
                "Course Timetabling with Metaheuristics"],
    "Dr Nurul": ["Clinic Appointment and Patient Record System", "Diabetes Risk Decision Support System",
                 "Telemedicine Platform for Rural Patients", "Hospital Bed Management Information System",
                 "Mental Health Self-Assessment App"],
    "Dr Rahman": ["GIS Mapping of Dengue Hotspots", "Location-Based Tourist Guide System",
                  "Spatial Analysis of Flood-Prone Areas", "Web GIS for Halal Restaurant Finder",
                  "Bus Tracking System with Real-Time Map"],
}
_CONTEXTS = ["for Malaysian SMEs", "for UiTM Students", "in Selangor", "for Local Communities",
             "for Small Businesses", "for Public Universities", "in Rural Areas", "for Government Agencies"]


def make_sample_data(n_students: int = 90, seed: int = 7, supervisor_same_area: float = 0.7):
    rng = np.random.default_rng(seed)
    lect = pd.DataFrame(_SAMPLE_LECTURERS, columns=["lecturer", "expertise"])
    lect["max_load"] = DEFAULT_MAX_LOAD
    names = lect["lecturer"].tolist()
    rows = []
    for k in range(n_students):
        area = names[k % len(names)]
        topic = _SAMPLE_TOPICS[area][rng.integers(len(_SAMPLE_TOPICS[area]))]
        title = f"{topic} {_CONTEXTS[rng.integers(len(_CONTEXTS))]}"
        sv = area if rng.random() < supervisor_same_area else names[rng.integers(len(names))]
        rows.append({
            "student_id": f"2024{100000 + k:06d}",
            "name": f"Student {k + 1:03d}",
            "fyp_level": "FYP1" if k % 2 == 0 else "FYP2",
            "project_title": title,
            "project_description": "",
            "supervisor": sv,
        })
    return pd.DataFrame(rows), lect
