import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import cross_validate

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Calcula todas las métricas requeridas
    """
    metrics=[]

    ti_tr = time.time()
    model.fit(X_train, y_train)
    tf_tr = time.time()
    t_tr = tf_tr - ti_tr

    ti_inf = time.time()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tf_inf = time.time()
    t_inf = tf_inf - ti_inf

    metrics.append({
        #"Modelo": ,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "MCC": matthews_corrcoef(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions)
    })
        
    return pd.DataFrame(metrics), predictions, probabilities, t_tr, t_inf

#función de métricas por grupo
def metrics_by_group(y_true, y_pred, y_proba, group):
    results = []
    for g in np.unique(group.dropna()):
        mask = (group == g)
        yt, yp = y_true[mask], y_pred[mask]
        ytr, ypr = y_true[mask], y_proba[mask]
        if len(np.unique(yt)) == 1:  #evita errores si solo hay una clase
            continue
        results.append({
            "Grupo": g,
            "Accuracy": accuracy_score(yt, yp),
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "F1-Score": f1_score(yt, yp, zero_division=0),
            "ROC-AUC": roc_auc_score(ytr, ypr),
            "PR-AUC": average_precision_score(ytr, ypr),
            "MCC": matthews_corrcoef(yt, yp),
            "Balanced Accuracy": balanced_accuracy_score(yt, yp)
        })
    return pd.DataFrame(results)

#McNemar tests
def run_mcnemar(y_true, pred1, pred2, name1, name2):
    both_correct = ((pred1 == y_true) & (pred2 == y_true)).sum()
    model1_correct_only = ((pred1 == y_true) & (pred2 != y_true)).sum()
    model2_correct_only = ((pred1 != y_true) & (pred2 == y_true)).sum()
    both_wrong = ((pred1 != y_true) & (pred2 != y_true)).sum()

    contingency = [[both_correct, model1_correct_only],
                   [model2_correct_only, both_wrong]]

    result = mcnemar(contingency, exact=False, correction=True)

    return {
        'Modelo A': name1,
        'Modelo B': name2,
        'Chi2': result.statistic,
        'p-value': result.pvalue
    }

#Intervalos de confianza para métricas
def metr_ic(y_test, y_pred, y_proba, n_bootstrap=1000, alpha=0.05):
    """
    Calcula intervalos de confianza para métricas de clasificación.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    rng = np.random.default_rng(42)  # reproducibilidad
    metricsic = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "ROC-AUC": [],
        "PR-AUC": [],
        "MCC": [],
        "Balanced Accuracy": []
    }

    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_res = y_test[idx]
        p_res = y_pred[idx]
        prob_res = y_proba[idx]

        metricsic["Accuracy"].append(accuracy_score(y_res, p_res))
        metricsic["Precision"].append(precision_score(y_res, p_res, zero_division=0))
        metricsic["Recall"].append(recall_score(y_res, p_res, zero_division=0))
        metricsic["F1-Score"].append(f1_score(y_res, p_res, zero_division=0))
        metricsic["ROC-AUC"].append(roc_auc_score(y_res, prob_res))
        metricsic["PR-AUC"].append(average_precision_score(y_res, prob_res))
        metricsic["MCC"].append(matthews_corrcoef(y_res, p_res))
        metricsic["Balanced Accuracy"].append(balanced_accuracy_score(y_res, p_res))

    #percentiles para IC
    metrics_ci = {}
    for metric, values in metricsic.items():
        lower = np.percentile(values, 100 * alpha/2)
        upper = np.percentile(values, 100 * (1 - alpha/2))
        metrics_ci[metric] = (lower, upper)

    return metrics_ci

def cv_results(model, X, y, model_name,cv, scoring):
    results = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    df_folds = pd.DataFrame(results)
    df_folds["Fold"] = range(1, len(df_folds)+1)
    df_folds["Modelo"] = model_name
    return df_folds