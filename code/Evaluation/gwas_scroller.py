import pandas as pd
import numpy as np
from scipy.stats import hypergeom

# Check the overlap and ranking of a set of genes compared to a GWAS study for a particular disease.
# Gwas studies can be found on gwas catalog.
# ppi_path is a list of coding genes, as gwas can include non-coding genes, it is important to filter gwas genes as
# genes obtained through ML on RNA-seq data will exclusively contain coding genes.
# We recommend to use the genes published in Geisy 2023.
def gwas(signature, ppi_path, gwas_path):
    coding = pd.read_csv(ppi_path, index_col="Symbol_A", header=0)
    coding = set([gene for gene in coding.index])
    # gwas files from the gwas_catalog are tsv
    df = pd.read_csv(gwas_path, index_col="MAPPED_GENE", header=0, delimiter="\t")
    df = df[df.index.notnull()]
    df = df.loc[df.loc[:, "P-VALUE"] < 10**(-8), :]
    coding_ra = set([i for i in df.index if any(gene.replace(" ", "") in coding for gene in i.split(",") + i.split("-"))])

    # Map the genes to the files of the coding genes for nomenclature issues and remove duplicates.
    signature = set([idx for idx in signature if idx in coding])

    retained = []
    rank = {}
    for gene in signature:
        for idx in df.index:
            # Gwas catalog can contain several names for a same gene, we check of one of them is the one we are looking for.
            if gene in idx:
                retained.append(gene)
                # 10^-8 usually is the significance threshold for gwas
                if df.loc[idx, "P-VALUE"].min() < 10**(-8):
                    # Get the Beta score of the gene in the gwas, when several occurrences of the genes are present,
                    # We take the one with the highest score.
                    if gene not in rank.keys():
                        rank[gene] = df.loc[idx, "OR or BETA"].max()
                break
    retained = set(retained)

    # We output the genes in the overlap with their Beta score and ensure the significance of the overlap
    return retained, rank, hypergeom.sf(len(retained)-1, len(coding), len(coding_ra), len(signature))
