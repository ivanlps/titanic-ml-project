from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class ModelExperiment:
    def __init__(self, model_name, model, param_grid, scoring='f1', random_state=42,refit_metric='f1'):
        self.model_name = model_name
        self.model = model
        self.param_grid = param_grid
        self.random_state = random_state
        self.results = {}
        self.best_estimator_ = None
        self.scoring = scoring
        self.refit_metric = refit_metric

    def run_experiment(self, X_train, y_train, X_test, y_test, cv_splits=5):
        print(f"=== {self.model_name} ===")

        # 1. GridSearch con validación cruzada estratificada
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=self.refit_metric
        )
        grid.fit(X_train, y_train)
        self.best_estimator_ = grid.best_estimator_

        # 2. Evaluación en conjunto de validación
        y_pred = self.best_estimator_.predict(X_test)
        y_proba = self.best_estimator_.predict_proba(X_test)[:, 1] if hasattr(self.best_estimator_, "predict_proba") else None

        # 3. Guardar resultados
        self.results['best_params'] = grid.best_params_
        self.results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.results['roc_auc'] = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        # 4. Persistencia del modelo
        joblib.dump(self.best_estimator_, f"../models/{self.model_name}_final.pkl")

        return self.results