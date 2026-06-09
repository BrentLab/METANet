import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import GroupKFold

parser = argparse.ArgumentParser(description="Creates TF-disjoint CV folds (no TF shared between train and test)")
parser.add_argument("--tissue", required=True, help="tissue")
parser.add_argument("--base_path", required=True, help="base path")
parser.add_argument("--model", required=True, help="model (e.g. lasso_excl_genie3_wgcna)")
parser.add_argument("--binding_threshold", type=int, default=10, help="binding threshold (default=10)")
parser.add_argument("--num_CV_folds", type=int, default=10, help="number of CV folds")
parser.add_argument("--seed", type=int, default=42, help="random seed for TF shuffle")
args = parser.parse_args()

# output directory -- mirrors create_cv_folds_4.py but writes to CV_folds_tf_disjoint
output_path = os.path.join(
    args.base_path, "results", args.tissue, args.model, "CV_folds_tf_disjoint"
)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# load data  (input_data path keeps the original 10.0_threshold_ format)
input_path = os.path.join(
    args.base_path, "input_data", args.tissue, args.model, f"{args.tissue}_{args.model}_nlogrank_minmax.txt"
)
df = pd.read_csv(input_path, sep="\t")

# preprocess (abs value on feature columns)
for col in df.columns[3:]:
    df[col] = df[col].abs()

# -------------------------------------------------------------------------
# TF-level splitting
# GroupKFold splits rows so that no TF appears in both train and test.
# GroupKFold is deterministic based on the order groups appear in the data,
# so we shuffle TFs first to randomise which TFs land in which fold.
# -------------------------------------------------------------------------
tfs = df["TF"].unique()
rng = np.random.default_rng(args.seed)
shuffled_tfs = rng.permutation(tfs)
tf_order = {tf: i for i, tf in enumerate(shuffled_tfs)}
df = df.iloc[df["TF"].map(tf_order).argsort()].reset_index(drop=True)

print(f"Total TFs: {len(tfs)}  |  Total edges: {len(df)}  |  Folds: {args.num_CV_folds}")

df_Y = df["LABEL"]
df_X = df.drop(columns=["LABEL"])
groups = df["TF"].values

gkf = GroupKFold(n_splits=args.num_CV_folds)

for fold, (train_idx, test_idx) in enumerate(gkf.split(df_X, df_Y, groups=groups)):
    df_fold_train = df.iloc[train_idx]
    df_fold_test  = df.iloc[test_idx]

    # sanity check: no TF overlap
    overlap = set(df_fold_train["TF"]) & set(df_fold_test["TF"])
    assert len(overlap) == 0, f"Fold {fold}: TF overlap detected: {overlap}"

    n_test_tfs  = df_fold_test["TF"].nunique()
    n_train_tfs = df_fold_train["TF"].nunique()
    pos_rate_train = df_fold_train["LABEL"].mean()
    pos_rate_test  = df_fold_test["LABEL"].mean()
    print(
        f"Fold {fold}: train_tfs={n_train_tfs}  test_tfs={n_test_tfs} | "
        f"train_edges={len(df_fold_train)}  test_edges={len(df_fold_test)} | "
        f"pos_rate_train={pos_rate_train:.3f}  pos_rate_test={pos_rate_test:.3f}"
    )

    df_fold_train.to_csv(os.path.join(output_path, f"fold{fold}_train_data.txt"), sep="\t", index=False)
    df_fold_test.to_csv(os.path.join(output_path,  f"fold{fold}_test_data.txt"),  sep="\t", index=False)

print(f"CV folds saved to {output_path}")
print("DONE")
