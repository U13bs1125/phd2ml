from scipy.stats import ttest_rel

def compare_models(results_df, model1, model2, metric="f1"):
    m1 = results_df[results_df["model"] == model1][metric]
    m2 = results_df[results_df["model"] == model2][metric]

    stat, p = ttest_rel(m1, m2)

    return {"t_stat": stat, "p_value": p}