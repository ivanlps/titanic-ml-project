import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

joblib.load('../models/LogisticRegression_final.pkl')

#fairness
def demographic_disparity(df, group, pred_col='y_pred', reference=None):
    results = (
        df.groupby(group)[pred_col]
        .mean()
        .reset_index()
        .rename(columns={pred_col: 'P(Yp=1|G)'})
    )

    if reference is None:
        reference = results[group].iloc[0]  

    p_ref = results.loc[results[group] == reference, 'P(Yp=1|G)'].values[0]

    results['Diferencia'] = results['P(Yp=1|G)'] - p_ref
    results['Ratio'] = results['P(Yp=1|G)'] / p_ref

    return results, reference

def equal_opportunity(df, group, true_col='Survived', pred_col='y_pred'):
    #Calcula el True Positive Rate (TPR) por grupo.
    df_1s = df[df[true_col] == 1]  

    results = (
        df_1s.groupby(group)[pred_col]
        .mean()
        .reset_index()
        .rename(columns={pred_col: 'TPR'})
    )

    return results

def equalized_odds(df, group, true_col='Survived', pred_col='y_pred'):
    #Calcula TPR, FPR y Precisión por grupo.
    
    results = []

    for g, subset in df.groupby(group):
        tn, fp, fn, tp = confusion_matrix(subset[true_col], subset[pred_col], labels=[0,1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidad
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Especificidad
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        results.append({
            group: g,
            'TPR': tpr,
            'FPR': fpr,
            'Precision': precision
        })

    return pd.DataFrame(results)

def calibration_by_group(df, group, true_col='Survived', prob_col='y_prob', n_bins=10):
    #Calcula curvas de calibración por grupo usando bins de probabilidad
    
    results = {}

    for g, subset in df.groupby(group):
        y_true = subset[true_col]
        y_prob = subset[prob_col]

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        results[g] = pd.DataFrame({
            'mean_pred': mean_pred,
            'frac_pos': frac_pos
        })

    return results