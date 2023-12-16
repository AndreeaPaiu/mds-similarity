from scipy.stats import pearsonr


def pearsonr_similarity(a, b):
    c = pearsonr(a, b)
    if isinstance(c, float):
        return 1
    else:
        return 1 - c[0]