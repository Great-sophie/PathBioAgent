# PathBioAgent

**An evidence-driven LangGraph agent for multiscale morphology–molecular hypothesis analysis in computational pathology.**

PathBioAgent is a research prototype for querying, ranking, testing, and reporting morphology–molecular hypotheses derived from pathology foundation model representations across histological scales.

Rather than allowing a language model to generate scientific conclusions directly, PathBioAgent separates:

- natural-language scientific questions,
- deterministic evidence retrieval,
- predefined scientific decision rules,
- evidence-quality diagnostics,
- and auditable structured reporting.

The current v0.1 implementation operates on precomputed molecular-program visibility results from **8 pathology foundation models (PFMs)** across **3 histological magnifications: 2.5×, 10×, and 40×**.

> PathBioAgent is intended for exploratory computational pathology research.  
> It does **not** establish clinical biomarker validity, biological causality, or patient-level molecular diagnosis.

---

## Why PathBioAgent?

Computational pathology studies increasingly produce large collections of results across:

- foundation models,
- magnifications,
- molecular targets,
- cross-validation folds,
- robustness experiments,
- and sensitivity analyses.

A conventional workflow often requires manually inspecting many result tables before answering a seemingly simple question such as:

> Which molecular programs consistently prefer 10×?

or:

> Which programs show the most robust 40× morphology–molecular association?

PathBioAgent converts this process into an explicit evidence workflow:

```text
Scientific query
      ↓
Query interpretation
      ↓
Candidate search and ranking
      ↓
Real multiscale evidence retrieval
      ↓
Cross-model consensus
      ↓
Robustness analysis
      ↓
Evidence diagnostics
      ↓
Deterministic evidence gate
      ↓
Auditable Evidence Cards
      ↓
Structured JSON report