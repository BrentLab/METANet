import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import GroupKFold   # replaces StratifiedKFold for TF-disjoint inner CV
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import (accuracy_score, log_loss, average_precision_score,
                             roc_curve, precision_recall_curve, auc, f1_score,
                             confusion_matrix)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="XGBoost classifier train/test (TF-disjoint CV)")
parser.add_argument("--tissue", required=True, help="tissue")
parser.add_argument("--base_path", required=True, help="base path")
parser.add_argument("--model", required=True, help="model (e.g. METANet)")
parser.add_argument("--submodel", required=True, help="model (e.g. metanet_tf_disjoint)")
parser.add_argument("--binding_threshold", type=int, default=10, help="binding threshold in % (default=10)")
parser.add_argument("--n_cv_folds", type=int, default=10, help="number of CV folds")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args = parser.parse_args()

# set random seed
np.random.seed(seed=args.seed)

########################
### HYPEROPT CONFIGS ###
########################
# hyperopt xgboost parameter search space
space = {
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.loguniform('gamma', np.log(1e-3), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(100)),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(100)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': 500,  # actual tree count determined by early stopping
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}


# paths -- reads TF-disjoint folds created by create_cv_folds_tf_disjoint_4.py
run_path = os.path.join(args.base_path, "results", args.tissue, f"{args.model}_{args.binding_threshold}")
input_cv_path = os.path.join(run_path, "CV_folds_tf_disjoint")   # <-- only change vs xgboost_fixed_5.py
output_path = os.path.join(run_path, args.submodel)
output_cv_path = os.path.join(output_path, "CV")

# create dirs
if not os.path.exists(output_cv_path):
    os.makedirs(output_cv_path)


#####################
### HELPER FUNCTIONS
#####################
def get_sens_spec(y_true, y_pred):
    """Compute sensitivity (recall) and specificity from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


###############
### CV runs ###
###############
df_preds_list = []
for fold in range(args.n_cv_folds):
    print(f"fold: {fold}")

    # load CV data
    df_train = pd.read_csv(os.path.join(input_cv_path, f"fold{fold}_train_data.txt"), sep="\t")
    df_test = pd.read_csv(os.path.join(input_cv_path, f"fold{fold}_test_data.txt"), sep="\t")

    # sanity check: outer folds must be TF-disjoint
    overlap = set(df_train["TF"]) & set(df_test["TF"])
    assert len(overlap) == 0, f"Fold {fold}: TF overlap in outer split: {overlap}"

    # retain TF column for inner-CV grouping before it is dropped from the feature matrix
    train_tfs = df_train["TF"].values

    # split features vs response
    df_X_train = df_train.drop(columns=['TF','GENE','LABEL'])
    df_Y_train = df_train['LABEL']
    df_X_test = df_test.drop(columns=['TF','GENE','LABEL'])
    df_Y_test = df_test['LABEL']

    # class (im)balance
    class_balance_ratio = (len(df_Y_train) - sum(df_Y_train)) / sum(df_Y_train)
    print(f"class balance ratio: {class_balance_ratio}")
    # set scale_pos_weight
    space['scale_pos_weight'] = class_balance_ratio

    # list to collect per-hyperopt-iteration validation metrics
    val_scores = []
    trial_counter = [0]  # mutable counter for varying inner CV splits

    # -----------------------------------------------------------------------
    # hyperopt objective: inner CV uses TF-disjoint GroupKFold (3 splits).
    # GroupKFold is deterministic on input order, so rows are shuffled with a
    # trial-varying seed so different hyperopt trials see different partitions.
    # -----------------------------------------------------------------------
    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])

        rng_inner = np.random.default_rng(args.seed + trial_counter[0])
        trial_counter[0] += 1

        # shuffle rows so GroupKFold assigns TFs to inner splits randomly
        shuffled_idx = rng_inner.permutation(len(df_X_train))
        X_sh   = df_X_train.iloc[shuffled_idx]
        y_sh   = df_Y_train.iloc[shuffled_idx]
        tfs_sh = train_tfs[shuffled_idx]

        gkf_inner = GroupKFold(n_splits=3)
        fold_metrics = []

        for train_idx, val_idx in gkf_inner.split(X_sh, y_sh, groups=tfs_sh):
            X_tr  = X_sh.iloc[train_idx]
            X_val = X_sh.iloc[val_idx]
            y_tr  = y_sh.iloc[train_idx]
            y_val = y_sh.iloc[val_idx]

            model = XGBClassifier(**params, nthread=5, callbacks=[EarlyStopping(rounds=20)])
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            val_probs = model.predict_proba(X_val)[:, 1]
            val_preds = model.predict(X_val)

            val_logloss     = log_loss(y_val, val_probs)
            val_auprc       = average_precision_score(y_val, val_probs)
            fpr_v, tpr_v, _ = roc_curve(y_val, val_probs)
            val_auroc       = auc(fpr_v, tpr_v)
            val_sens, val_spec = get_sens_spec(y_val, val_preds)
            val_f1          = f1_score(y_val, val_preds)
            val_accuracy    = accuracy_score(y_val, val_preds)

            fold_metrics.append({
                'logloss':     val_logloss,
                'auprc':       val_auprc,
                'auroc':       val_auroc,
                'sensitivity': val_sens,
                'specificity': val_spec,
                'f1':          val_f1,
                'accuracy':    val_accuracy
            })

        # average across inner CV folds
        mean_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        val_scores.append(mean_metrics)

        return {'loss': -mean_metrics['auprc'], 'status': STATUS_OK}

    # hyperopt tuning
    print("hyperparameter tuning with hyperopt")
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=30,
                trials=trials,
                early_stop_fn=no_progress_loss(10))
    best['max_depth']        = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])
    # also include static hyperparameters not tuned and thus not returned by hyperopt
    best['n_estimators']     = 500  # early stopping determines actual tree count
    best['scale_pos_weight'] = class_balance_ratio
    best['objective']        = 'binary:logistic'
    best['eval_metric']      = 'logloss'

    print("Best hyperparameters found: ", best)

    # retrieve validation metrics at the best hyperopt iteration
    best_iter = np.argmin(trials.losses())
    best_val_metrics = val_scores[best_iter]
    print(f"Best inner-CV val metrics: {best_val_metrics}")

    # train final model with best hyperparam values
    print("training final model")
    final_model = XGBClassifier(**best, nthread=5, callbacks=[EarlyStopping(rounds=20)])
    final_model.fit(df_X_train, df_Y_train,
                    eval_set=[(df_X_test, df_Y_test)],
                    verbose=False)

    # update n_estimators to reflect actual trees used (early stopping may have stopped before 500)
    best['n_estimators'] = final_model.best_iteration + 1
    df_best = pd.DataFrame(best.items(), columns=['param', 'value'])
    # write best params
    df_best.to_csv(os.path.join(output_cv_path, f"fold{fold}_hyperparams.tsv"), sep="\t", index=False)

    # make predictions
    print("predicting")
    train_preds      = final_model.predict(df_X_train)
    train_probs      = final_model.predict_proba(df_X_train)[:, 1]

    test_preds       = final_model.predict(df_X_test)
    test_probs       = final_model.predict_proba(df_X_test)[:, 1]

    # evaluate train
    train_accuracy             = accuracy_score(df_Y_train, train_preds)
    train_logloss              = log_loss(df_Y_train, train_probs)
    train_auprc                = average_precision_score(df_Y_train, train_probs)
    train_f1                   = f1_score(df_Y_train, train_preds)
    train_sensitivity, train_specificity = get_sens_spec(df_Y_train, train_preds)
    print(f'Training Accuracy:    {train_accuracy:.4f}')
    print(f'Training Logloss:     {train_logloss:.4f}')
    print(f'Training AUPRC:       {train_auprc:.4f}')
    print(f'Training F1:          {train_f1:.4f}')
    print(f'Training Sensitivity: {train_sensitivity:.4f}')
    print(f'Training Specificity: {train_specificity:.4f}')

    # evaluate test
    test_accuracy              = accuracy_score(df_Y_test, test_preds)
    test_logloss               = log_loss(df_Y_test, test_probs)
    test_auprc                 = average_precision_score(df_Y_test, test_probs)
    test_f1                    = f1_score(df_Y_test, test_preds)
    test_sensitivity, test_specificity = get_sens_spec(df_Y_test, test_preds)
    print(f'Test Accuracy:        {test_accuracy:.4f}')
    print(f'Test Logloss:         {test_logloss:.4f}')
    print(f'Test AUPRC:           {test_auprc:.4f}')
    print(f'Test F1:              {test_f1:.4f}')
    print(f'Test Sensitivity:     {test_sensitivity:.4f}')
    print(f'Test Specificity:     {test_specificity:.4f}')

    # record predictions
    df_train_preds = df_train[['TF','GENE']].copy()
    df_train_preds['pred_prob'] = train_probs
    df_train_preds['pred']      = train_preds

    df_test_preds = df_test[['TF','GENE']].copy()
    df_test_preds['pred_prob'] = test_probs
    df_test_preds['pred']      = test_preds
    df_test_preds = df_test_preds.reset_index(drop=True)
    df_preds_list.append(df_test_preds)

    # plot AUROC
    print("plotting AUROC")
    fpr_train, tpr_train, _ = roc_curve(df_Y_train, train_probs)
    roc_auc_train            = auc(fpr_train, tpr_train)
    fpr_test,  tpr_test,  _ = roc_curve(df_Y_test, test_probs)
    roc_auc_test             = auc(fpr_test, tpr_test)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f"ROC train (AUC={roc_auc_train:.2f})")
    plt.plot(fpr_test,  tpr_test,  color='blue',       lw=2, label=f"ROC test  (AUC={roc_auc_test:.2f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{args.tissue}\nFold {fold} AUROC (TF-disjoint)")
    plt.legend()
    plt.savefig(os.path.join(output_cv_path, f"fold{fold}_auroc.png"), format="png", bbox_inches="tight")
    plt.close()

    # plot AUPRC
    print("plotting AUPRC")
    precision_train, recall_train, _ = precision_recall_curve(df_Y_train, train_probs)
    prc_auc_train                    = auc(recall_train, precision_train)
    precision_test,  recall_test,  _ = precision_recall_curve(df_Y_test,  test_probs)
    prc_auc_test                     = auc(recall_test,  precision_test)

    plt.figure(figsize=(10, 10))
    plt.plot(recall_train, precision_train, color='darkorange', lw=2, label=f"PRC train (AUC={prc_auc_train:.2f})")
    plt.plot(recall_test,  precision_test,  color='blue',       lw=2, label=f"PRC test  (AUC={prc_auc_test:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{args.tissue}\nFold {fold} AUPRC (TF-disjoint)")
    plt.legend()
    plt.savefig(os.path.join(output_cv_path, f"fold{fold}_auprc.png"), format="png", bbox_inches="tight")
    plt.close()

    print("saving")
    # write CV preds
    df_test_preds = df_test_preds.sort_values(by='pred_prob', ascending=False)
    df_test_preds.to_csv(os.path.join(output_cv_path, f"fold{fold}_test_preds.tsv"), sep="\t", index=False)
    # save model for downstream analysis (e.g. SHAP)
    final_model.save_model(os.path.join(output_cv_path, f"fold{fold}_model.json"))

    # save all metrics for this outer fold
    metrics = {
        'fold':                   fold,
        # --- training ---
        'train_accuracy':         train_accuracy,
        'train_logloss':          train_logloss,
        'train_auprc':            train_auprc,
        'train_auprc_curve':      prc_auc_train,
        'train_auroc':            roc_auc_train,
        'train_f1':               train_f1,
        'train_sensitivity':      train_sensitivity,
        'train_specificity':      train_specificity,
        # --- inner-CV validation (best hyperopt iteration) ---
        'val_logloss':            best_val_metrics['logloss'],
        'val_accuracy':           best_val_metrics['accuracy'],
        'val_auprc':              best_val_metrics['auprc'],
        'val_auroc':              best_val_metrics['auroc'],
        'val_f1':                 best_val_metrics['f1'],
        'val_sensitivity':        best_val_metrics['sensitivity'],
        'val_specificity':        best_val_metrics['specificity'],
        # --- test ---
        'test_accuracy':          test_accuracy,
        'test_logloss':           test_logloss,
        'test_auprc':             test_auprc,
        'test_auprc_curve':       prc_auc_test,
        'test_auroc':             roc_auc_test,
        'test_f1':                test_f1,
        'test_sensitivity':       test_sensitivity,
        'test_specificity':       test_specificity,
    }
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_cv_path, f"fold{fold}_metrics.tsv"), sep="\t", index=False)


# write final preds
df_preds = pd.concat(df_preds_list)
df_preds = df_preds.sort_values(by='pred_prob', ascending=False)
df_preds.to_csv(os.path.join(output_path, f"{args.tissue}_{args.submodel}_full.tsv"), sep="\t", index=False)
# tf-gene-pred_proba for evaluation
df_preds = df_preds.drop(columns=['pred'])
df_preds.to_csv(os.path.join(output_path, f"{args.tissue}_{args.submodel}.tsv"), sep="\t", index=False, header=False)

# summarize CV metrics across all outer folds
all_metrics = pd.concat([
    pd.read_csv(os.path.join(output_cv_path, f"fold{i}_metrics.tsv"), sep="\t")
    for i in range(args.n_cv_folds)
])
# append mean and std rows
numeric_metrics = all_metrics.drop(columns=['fold'])
mean_row = numeric_metrics.mean().to_frame().T
mean_row['fold'] = 'mean'
std_row = numeric_metrics.std().to_frame().T
std_row['fold'] = 'std'
all_metrics = pd.concat([all_metrics, mean_row, std_row], ignore_index=True)
all_metrics.to_csv(os.path.join(output_path, "cv_metrics_summary.tsv"), sep="\t", index=False)

print("DONE")
