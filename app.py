"""
Streamlit web app: FYP Examiner Matching (SBERT + Harmony Search)
Run locally :  streamlit run app.py
Deploy free :  Streamlit Community Cloud (see README.md)
"""
import numpy as np
import pandas as pd
import streamlit as st

import matcher as M

st.set_page_config(page_title="FYP Examiner Matching", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading SBERT model (first time only)...")
def get_model(name):
    return M.load_sbert(name)


@st.cache_data(show_spinner="Computing embeddings and cosine similarity...")
def get_similarity(student_texts, lecturer_texts, model_name):
    if not model_name.upper().startswith("TF-IDF"):
        try:
            get_model(model_name)  # warm cache
        except ImportError:
            pass
    return M.compute_similarity(list(student_texts), list(lecturer_texts), model_name)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Data source", ["Use sample data", "Upload my files"])

students_raw = lecturers_raw = None
if source == "Use sample data":
    n_sample = st.sidebar.slider("Number of sample students", 20, 110, 90, 5)
    students_raw, lecturers_raw = M.make_sample_data(n_sample)
else:
    fs = st.sidebar.file_uploader("Students (CSV / Excel)", type=["csv", "xlsx", "xls"])
    fl = st.sidebar.file_uploader("Lecturers (CSV / Excel)", type=["csv", "xlsx", "xls"])
    if fs is not None:
        students_raw = M.read_table(fs)
    if fl is not None:
        lecturers_raw = M.read_table(fl)
    s_tpl, l_tpl = M.make_sample_data(6)
    st.sidebar.download_button("⬇️ Student template", s_tpl.to_csv(index=False), "students_template.csv")
    st.sidebar.download_button("⬇️ Lecturer template", l_tpl.to_csv(index=False), "lecturers_template.csv")

st.sidebar.subheader("Matching")
model_name = st.sidebar.selectbox("Similarity model", M.SBERT_MODELS, index=0)
max_load = st.sidebar.number_input("Max workload per lecturer (supervise + examine, FYP1 + FYP2)",
                                   1, 100, M.DEFAULT_MAX_LOAD,
                                   help="Used when the lecturer file has no max_load column.")
use_band = st.sidebar.checkbox("Enforce equal-workload band", value=True,
                               help="Every lecturer's total must stay within ± spread of the fair share "
                                    "(total supervision + students) / lecturers.")
spread = st.sidebar.slider("Allowed spread around fair share (±)", 0, 10, 1) if use_band else None
fairness = st.sidebar.slider("Fairness weight  w  (penalty per unit of L_max − L_min)", 0.0, 3.0, 0.5, 0.05)

st.sidebar.subheader("Harmony Search")
hms = st.sidebar.slider("HMS (harmony memory size)", 5, 100, 20)
hmcr = st.sidebar.slider("HMCR (memory consideration rate)", 0.50, 0.99, 0.90, 0.01)
par = st.sidebar.slider("PAR (pitch adjustment rate)", 0.05, 0.90, 0.30, 0.05)
bw = st.sidebar.slider("Bandwidth (± ranks)", 1, 5, 2)
iters = st.sidebar.number_input("Max iterations", 100, 100_000, 2000, 500)
ls_rate = st.sidebar.slider("Local-search rate (hybrid HS, 0 = classic HS)", 0.0, 1.0, 0.1, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)

st.sidebar.subheader("Benchmark")
use_cp = st.sidebar.checkbox("Also solve with CP-SAT (exact) for comparison", value=False)
cp_time = st.sidebar.slider("CP-SAT time limit (s)", 5, 120, 20) if use_cp else 20

# ---------------------------------------------------------------- header
st.title("🎓 FYP Examiner Matching System")
st.caption("SBERT semantic similarity  →  cosine matching score  →  Harmony Search optimisation "
           "with workload balancing")

if students_raw is None or lecturers_raw is None:
    st.info("Upload both the **students** and **lecturers** files in the sidebar "
            "(or switch to *Use sample data*).\n\n"
            "**Students** columns: `student_id`, `name`, `fyp_level` (FYP1/FYP2), `project_title`, "
            "`project_description` (optional), `supervisor`\n\n"
            "**Lecturers** columns: `lecturer`, `expertise`, `max_load` (optional, default 20)")
    st.stop()

try:
    students = M.clean_students(students_raw)
    lecturers = M.clean_lecturers(lecturers_raw, int(max_load))
except ValueError as e:
    st.error(str(e))
    st.stop()

tab_data, tab_sim, tab_res, tab_load, tab_conv, tab_how = st.tabs(
    ["1 · Input", "2-3 · Similarity", "4 · Assignment", "5 · Workload", "6 · Convergence", "ℹ️ How it works"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(students))
    c2.metric("Lecturers", len(lecturers))
    c3.metric("Total capacity (lecturers × max load)", int(lecturers["max_load"].sum()))
    st.caption(f"Each student uses 2 workload slots (1 supervisor + 1 examiner) → "
               f"{2 * len(students)} slots needed.")
    st.subheader("Students")
    st.dataframe(students, width="stretch", hide_index=True)
    st.subheader("Lecturers / panel expertise")
    st.dataframe(lecturers, width="stretch", hide_index=True)

s_texts = tuple(M.student_text(r) for _, r in students.iterrows())
l_texts = tuple(lecturers["expertise"])
S, method = get_similarity(s_texts, l_texts, model_name)
if method.startswith("TF-IDF") and not model_name.upper().startswith("TF-IDF"):
    st.warning("`sentence-transformers` is not installed, so keyword TF-IDF similarity was used instead of SBERT. "
               "Run `pip install sentence-transformers` for semantic matching.")
prob = M.build_problem(students, lecturers, S, fairness, spread)
for w in prob.warnings:
    st.warning(w)
if prob.band:
    st.sidebar.caption(f"Workload band in use: every lecturer gets **{prob.band[0]}–{prob.band[1]}** students "
                       f"in total (max load still applies).")

with tab_sim:
    st.subheader(f"Matching scores — {method}")
    st.caption("Cosine(A, B) = A·B / (‖A‖ × ‖B‖). Higher = project is closer to the lecturer's expertise. "
               "The student's own supervisor is never chosen as examiner.")
    sim_df = M.similarity_table(students, lecturers, S)
    num_cols = list(lecturers["lecturer"])
    st.dataframe(sim_df.style.background_gradient(cmap="Greens", subset=num_cols, axis=None)
                 .format("{:.4f}", subset=num_cols),
                 width="stretch", hide_index=True, height=480)

# ---------------------------------------------------------------- run
run = st.button("🚀 Run Harmony Search", type="primary")
params_key = (s_texts, l_texts, model_name, int(max_load), spread, fairness, hms, hmcr, par, bw, int(iters),
              ls_rate, int(seed), use_cp, cp_time)
if run:
    bar = st.progress(0.0, text="Harmony Search running...")
    hs = M.harmony_search(prob, hms=hms, hmcr=hmcr, par=par, bandwidth=bw, max_iter=int(iters),
                          seed=int(seed), ls_rate=ls_rate,
                          progress=lambda p: bar.progress(p, text=f"Harmony Search {p:.0%}"))
    bar.empty()
    results = {"Harmony Search": hs}
    if use_cp:
        with st.spinner("Solving with CP-SAT..."):
            cp = M.solve_cpsat(prob, cp_time)
        if cp is None:
            st.warning("CP-SAT unavailable (install `ortools`) or no feasible solution found.")
        else:
            results["CP-SAT"] = cp
    greedy = M.greedy_assignment(prob)
    st.session_state["out"] = {"key": params_key, "results": results,
                               "greedy": M.evaluate(prob, greedy)}

out = st.session_state.get("out")
if not out or out["key"] != params_key:
    for t in (tab_res, tab_load, tab_conv):
        with t:
            st.info("Press **🚀 Run Harmony Search** to optimise the assignment.")
else:
    results = out["results"]
    hs = results["Harmony Search"]
    a = hs.assignment
    assign_df = M.assignment_table(students, lecturers, prob, a)
    load_df = M.workload_table(students, lecturers, prob, a)
    summary_df = M.summary_table(results, method, len(students), len(lecturers), prob.band)

    with tab_res:
        m = hs.metrics
        c = st.columns(5)
        c[0].metric("Objective Z", f"{m['fitness']:.3f}")
        c[1].metric("Avg matching score", f"{m['similarity'] / len(students):.4f}")
        c[2].metric("Workload L_max / L_min", f"{m['L_max']} / {m['L_min']}")
        c[3].metric("Capacity violations", m["overflow"])
        c[4].metric("Got best possible examiner", f"{(assign_df['examiner_rank'] == 1).mean():.0%}")
        if m["overflow"] == 0:
            st.success("All hard constraints satisfied: one examiner per student, examiner ≠ supervisor, "
                       "no lecturer above max load.")
        else:
            st.error("Some lecturers exceed their max load — add lecturers or raise the max load.")

        comp = [{"method": "Greedy (best match, then repair)", "objective Z": out["greedy"]["fitness"],
                 "total score": out["greedy"]["similarity"], "L_max − L_min": out["greedy"]["gap"],
                 "violations": out["greedy"]["overflow"], "runtime (s)": None}]
        for name, r in results.items():
            comp.append({"method": r.method, "objective Z": r.metrics["fitness"],
                         "total score": r.metrics["similarity"], "L_max − L_min": r.metrics["gap"],
                         "violations": r.metrics["overflow"], "runtime (s)": round(r.runtime, 2)})
        st.dataframe(pd.DataFrame(comp).round(4), hide_index=True, width="stretch")

        f_lvl = st.multiselect("Filter FYP level", sorted(assign_df["fyp_level"].unique()),
                               default=sorted(assign_df["fyp_level"].unique()))
        view = assign_df[assign_df["fyp_level"].isin(f_lvl)]
        st.dataframe(view.style.background_gradient(cmap="RdYlGn", subset=["matching_score"]),
                     width="stretch", hide_index=True, height=520)

    with tab_load:
        st.subheader("Workload per lecturer (supervise + examine, FYP1 + FYP2)")
        chart = load_df.set_index("lecturer")[["total_supervise", "total_examine"]]
        chart.columns = ["Supervising", "Examining"]
        st.bar_chart(chart, stack=True, color=["#4C78A8", "#F58518"])
        st.dataframe(load_df, width="stretch", hide_index=True)
        st.caption(f"L_j = supervising_j + examining_j.  Fairness gap = L_max − L_min = {hs.metrics['gap']}.")

    with tab_conv:
        st.subheader("Harmony Search convergence (best objective in memory)")
        st.line_chart(pd.DataFrame({"best objective Z": hs.history}))
        st.caption(f"{len(hs.history) - 1} iterations in {hs.runtime:.2f} s.")

    sheets = {"Assignment": assign_df, "Workload": load_df,
              "Similarity Matrix": M.similarity_table(students, lecturers, S), "Summary": summary_df}
    if "CP-SAT" in results:
        sheets["Assignment (CP-SAT)"] = M.assignment_table(students, lecturers, prob, results["CP-SAT"].assignment)
    st.download_button("⬇️ Download results (Excel)", M.export_excel(sheets),
                       "FYP_Examiner_Assignment.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_how:
    st.markdown(r"""
### Workflow
1. **Input** – student project title (+ optional description) and each lecturer's expertise.
2. **SBERT** – each text becomes a numeric vector (embedding) that captures its meaning.
3. **Cosine similarity** – $\text{cos}(A,B)=\dfrac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}$ gives a matching score $s_{ij}$.
4. **Harmony Search** chooses one examiner per student.

### Model
Decision: $x_{ij}=1$ if lecturer $j$ examines student $i$.

**Hard constraints**
- each student has exactly one examiner: $\sum_j x_{ij}=1$
- examiner ≠ supervisor
- workload $L_j = \text{supervising}_j + \sum_i x_{ij} \le$ max load (default **20**, FYP1 + FYP2)
- optional equal-workload band: $\lfloor T\rfloor - s \le L_j \le \lceil T\rceil + s$, where
  $T = (\text{total supervision} + \text{students}) / \text{lecturers}$

**Objective** (maximise)
$$Z=\sum_i\sum_j s_{ij}x_{ij}\;-\;w\,(L_{max}-L_{min})$$

### Harmony Search
| Parameter | Meaning |
|---|---|
| HMS | number of solutions kept in memory |
| HMCR | chance a student's examiner is copied from memory (else random) |
| PAR | chance the copied examiner is "pitch-adjusted" to a neighbour in that student's ranked candidate list (± bandwidth) |
| Iterations | number of new harmonies improvised |

| Local-search rate | share of new harmonies polished by swap / relocate / leveling moves (hybrid HS) |

Each new harmony is built **capacity-safe**: students are placed in random order, and if the chosen examiner is
full the student gets their best examiner who still has room. It is then repaired for the band minimum,
optionally polished by local search, and replaces the **worst** harmony in memory if it scores better.
Tick *CP-SAT* in the sidebar to compare against an exact solver.
""")
    st.markdown("#### Worked example from the poster (3-dimensional toy embeddings)")
    A = [0.8, 0.6, 0.2]
    toy = {"Dr Chen": [0.9, 0.5, 0.3], "Dr Devi": [0.3, 0.8, 0.7], "Dr Eva": [0.6, 0.4, 0.8]}
    st.dataframe(pd.DataFrame([{"Panel member": k, "embedding": str(v), "cosine with Student A": round(M.cosine(A, v), 4)}
                               for k, v in toy.items()]), hide_index=True)
