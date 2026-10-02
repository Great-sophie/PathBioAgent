from pathlib import Path
from typing import Dict, Any
import pandas as pd


DEFAULT_ROOT = (
    "/media/sophie/Data2T/"
    "Paper3_8model_gene_analysis/"
    "pathbioagent_discovery"
)


def load_gene_tables(
    root=DEFAULT_ROOT
):

    root = Path(root)

    master = pd.read_csv(
        root / "gene_discovery_master.csv"
    )

    rna = pd.read_csv(
        root / "gene_rna_features.csv"
    )

    return master, rna



def explain_gene(
    gene: str,
    root=DEFAULT_ROOT,
) -> Dict[str, Any]:

    master, rna = load_gene_tables(root)


    g1 = master[
        master["gene"] == gene
    ]


    g2 = rna[
        rna["gene"] == gene
    ]


    if len(g1) == 0:

        return {
            "error":
            f"{gene} not found"
        }


    visibility = g1.iloc[0]


    result = {

        "gene":
            gene,


        "visibility":
        {

            "median_visibility":
            float(
                visibility[
                "best_scale_median_pearson"
                ]
            ),


            "preferred_scale":
            visibility[
                "derived_median_optimum_scale"
            ],


            "vote_scale":
            visibility[
                "vote_preferred_scale"
            ],

        },


        "rna_features":
        {},


    }


    if len(g2):

        row = g2.iloc[0]


        keys = [
            "rna_mean_expression",
            "rna_variance_expression",
            "rna_cancer_eta_squared",
            "rna_cancer_tau_specificity",
            "rna_nonzero_fraction",
        ]


        for k in keys:

            if k in row:

                result[
                    "rna_features"
                ][k] = float(row[k])


    return result



if __name__ == "__main__":

    import sys

    gene = (
        sys.argv[1]
        if len(sys.argv)>1
        else "SFTPA1"
    )


    print(
        explain_gene(gene)
    )
