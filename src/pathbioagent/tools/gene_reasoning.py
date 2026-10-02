from typing import Dict, Any


def build_gene_reasoning(
    evidence: Dict[str, Any]
) -> str:

    gene = evidence["gene"]

    vis = evidence["visibility"]

    rna = evidence["rna_features"]


    score = vis[
        "median_visibility"
    ]

    scale = vis[
        "preferred_scale"
    ]


    tau = rna.get(
        "rna_cancer_tau_specificity"
    )

    eta = rna.get(
        "rna_cancer_eta_squared"
    )


    return (
        f"{gene} shows high morphology-derived "
        f"molecular visibility "
        f"(median correlation={score:.3f}). "
        f"The preferred scale is {scale}. "
        f"The gene also shows strong cancer-specific "
        f"expression structure "
        f"(tau={tau:.3f}; eta squared={eta:.3f}). "
        f"These RNA properties co-occur with high "
        f"morphology-derived molecular visibility "
        f"in this cohort, but they do not establish "
        f"the underlying mechanism. "
        f"Additional spatial, cell-type, or "
        f"histomorphological descriptors would be "
        f"needed to test why this gene is particularly "
        f"visible from morphology."
    )
