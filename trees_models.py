##################################################################################################
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import shap
import random
from scipy import stats
from tqdm import tqdm
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, make_scorer

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

random.seed(123)

##################################################################################################
def run_regression(Y, X):
    # Añade una constante a las variables independientes
    X = sm.add_constant(X)
    
    
    # Ejecuta la regresión
    model = sm.OLS(Y, X)
    results = model.fit()
    
    # Crea un dataframe con los coeficientes y los p-valores
    results_df = pd.DataFrame({
        'coef': results.params,
        'p-value': results.pvalues
    })
    
    # Redondea los coeficientes a 4 decimales
    results_df['coef'] = results_df['coef'].round(4).astype(str)

    # Añade estrellas para los coeficientes significativos
    results_df.loc[results_df['p-value'] < 0.01, 'coef'] += '***'
    results_df.loc[(results_df['p-value'] >= 0.01) & (results_df['p-value'] < 0.05), 'coef'] += '**'
    results_df.loc[(results_df['p-value'] >= 0.05) & (results_df['p-value'] < 0.1), 'coef'] += '*'
    results_df["Explicativa"] = results_df.index

    return results_df[["Explicativa", 'coef']]

##################################################################################################
def xgboost_regression(X_train, Y_train, X_test, Y_test, loss_metric, trials_n):
    seed = 400
    np.random.seed(seed)

    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.2, 0.3, 0.4, 0.5])
        }

        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
            param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

        bst = xgb.XGBRegressor(**param)

        if loss_metric == "MSE":
            scores = cross_val_score(bst, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_squared_error')
        elif loss_metric == "MAE":
            scores = cross_val_score(bst, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_absolute_error')
        
        error = np.abs(scores.mean())
        return error

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials_n)

    error = study.best_trial.value
    parametros = study.best_trial.params

    xgboost_model = xgb.XGBRegressor(**parametros)
    xgboost_model = xgboost_model.fit(X_train, Y_train.values.ravel())

    feature_importance = pd.DataFrame(zip(X_train.columns, xgboost_model.feature_importances_ * 100), columns=["Explicativa", "Importance"])

    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X_train)

    feature_importance['SHAP'] = np.mean(shap_values, axis=0)
    feature_importance['SHAP_ABS'] = np.abs(np.mean(shap_values, axis=0))
    feature_importance = feature_importance.set_index("Explicativa")

    feature_importance[f"{loss_metric}_OUT"] = error

    if loss_metric == "MSE":
        feature_importance[f"{loss_metric}_IN"] = mean_squared_error(Y_test, xgboost_model.predict(X_test))
    elif loss_metric == "MAE":
        feature_importance[f"{loss_metric}_IN"] = np.abs(Y_test.values.ravel() - xgboost_model.predict(X_test)).mean()

    significant_features = list(feature_importance[feature_importance.Importance > 1].index)

    # Note: The function 'run_regression' is not provided in the original code snippet.
    # Assuming it's a custom function you have elsewhere to run a regression and retrieve coefficients/other stats.
    regression_results = run_regression(Y_train, X_train[significant_features])
    feature_importance = feature_importance.reset_index()
    feature_importance = feature_importance.merge(regression_results, "outer", "Explicativa")

    feature_importance = feature_importance.sort_values("Importance", ascending=False)
    feature_importance["PARAM"] = str(parametros)
    feature_importance.columns = [f"XGB_{i}" for i in feature_importance.columns]
    feature_importance = feature_importance.rename(columns={"XGB_Feature": "Explicativa"})

    return feature_importance


##################################################################################################
def random_forest_regression(X_train, Y_train, X_test, Y_test, loss_metric, trials_n):
    seed = 400
    np.random.seed(seed)

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }

        rf_regressor = RandomForestRegressor(**param, random_state=seed)

        if loss_metric == "MSE":
            scores = cross_val_score(rf_regressor, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_squared_error')
        elif loss_metric == "MAE":
            scores = cross_val_score(rf_regressor, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_absolute_error')
        
        error = np.abs(scores.mean())
        return error

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials_n)

    error = study.best_trial.value
    parametros = study.best_trial.params

    rf_model = RandomForestRegressor(**parametros, random_state=seed)
    rf_model = rf_model.fit(X_train, Y_train.values.ravel())

    feature_importance = pd.DataFrame(zip(X_train.columns, rf_model.feature_importances_ * 100), columns=["Explicativa", "Importance"])

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train)

    feature_importance['SHAP'] = np.mean(shap_values, axis=0)
    feature_importance['SHAP_ABS'] = np.abs(np.mean(shap_values, axis=0))
    feature_importance = feature_importance.set_index("Explicativa")

    feature_importance[f"{loss_metric}_OUT"] = error

    if loss_metric == "MSE":
        feature_importance[f"{loss_metric}_IN"] = mean_squared_error(Y_test, rf_model.predict(X_test))
    elif loss_metric == "MAE":
        feature_importance[f"{loss_metric}_IN"] = mean_absolute_error(Y_test, rf_model.predict(X_test))

    significant_features = list(feature_importance[feature_importance.Importance > 1].index)
    regression_results = run_regression(Y_train, X_train[significant_features])
    feature_importance = feature_importance.reset_index()
    feature_importance = feature_importance.merge(regression_results, "outer", "Explicativa")

    feature_importance = feature_importance.sort_values("Importance", ascending=False)
    feature_importance["PARAM"] = str(parametros)
    feature_importance.columns = [f"RF_{i}" for i in feature_importance.columns]
    feature_importance = feature_importance.rename(columns={"RF_Explicativa": "Explicativa"})

    return feature_importance

##################################################################################################
def catboost_regression(X_train, Y_train, X_test, Y_test, loss_metric, trials_n):
    perdida = loss_metric
    seed = 400  # Semilla para reproducibilidad
    np.random.seed(seed)  # Establecer la semilla para generación de números aleatorios

    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 100),  # Número de iteraciones
            'depth': trial.suggest_int('depth', 3, 10),  # Profundidad máxima del árbol
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # Tasa de aprendizaje
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2, 20),  # Regularización L2
            'border_count': trial.suggest_int('border_count', 20, 255),  # Número de contadores de borde
        }

        reg = CatBoostRegressor(**param, verbose=False, random_seed=seed)  # Crear el regresor CatBoost

        if perdida == "MAE":
            scores = cross_val_score(reg, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_absolute_error')  # Validación cruzada
        
        elif perdida == "MSE":
            scores = cross_val_score(reg, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_squared_error')

        error = np.abs(scores.mean())  # Valor absoluto del error

        return error

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))  # Crear un estudio de optimización de Optuna
    study.optimize(objective, n_trials=trials_n)  # Optimizar los hiperparámetros con Optuna

    error = study.best_trial.value  # Mejor valor de error
    parametros = study.best_trial.params  # Mejores hiperparámetros encontrados

    reg = CatBoostRegressor(**parametros, verbose=False, random_seed=seed)  # Crear un regresor CatBoost con los mejores hiperparámetros
    reg.fit(X_train, Y_train.values.ravel())  # Ajustar el modelo CatBoost

    importancias = reg.feature_importances_  # Importancia de las características según CatBoost
    importancias_normalizadas = 100.0 * importancias / np.sum(importancias)  # Importancia normalizada
    qebr = pd.DataFrame(zip(X_train.columns, importancias_normalizadas), columns=["Explicativa", "FE"])  # DataFrame con las características y su importancia

    explainer = shap.TreeExplainer(reg)  # Crear un objeto explainer de SHAP para CatBoost
    shap_values = explainer.shap_values(X_train)  # Calcular los valores SHAP

    qebr['SHAP'] = np.mean(shap_values, axis=0)  # Valor SHAP promedio
    qebr['SHAP_ABS'] = np.abs(np.mean(shap_values, axis=0))  # Valor absoluto del valor SHAP promedio
    qebr = qebr.set_index("Explicativa")  # Establecer la columna "Explicativa" como índice del DataFrame

    qebr[f"{perdida}_OUT"] = error  # Agregar el valor del error al DataFrame

    if perdida == "MAE":
        qebr[f"{perdida}_IN"] = mean_absolute_error(Y_test, reg.predict(X_test))
    elif perdida == "MSE":
        qebr[f"{perdida}_IN"] = mean_squared_error(Y_test, reg.predict(X_test))

    X_FE = list(qebr[qebr.FE > 1].index)  # Obtener las características con una importancia mayor que cero
    regression_results = run_regression(Y_train, X_train[X_FE])  # Realizar una regresión usando las características seleccionadas
    qebr = qebr.merge(regression_results, "outer", "Explicativa")  # Combinar los resultados de la regresión con el DataFrame qebr
    qebr = qebr.sort_values("FE", ascending=False)  # Ordenar el DataFrame por la importancia de las características
    qebr["PARAM"] = str(parametros)
    qebr.columns = [f"CAT_{i}" for i in qebr.columns]  # Renombrar las columnas del DataFrame
    qebr = qebr.rename(columns={"CAT_Explicativa": "Explicativa"})  # Renombrar la columna "CAT_Explicativa" a "Explicativa"
    
    return qebr

##################################################################################################
def lightgbm_regression(X_train, Y_train, X_test, Y_test, loss_metric, trials_n):
    perdida = loss_metric
    seed = 400  # Semilla para reproducibilidad
    np.random.seed(seed)  # Establecer la semilla para generación de números aleatorios

    def objective(trial):
        param = {
            'silent': True,  # Silenciar la salida del modelo
            'objective': 'regression',  # Función objetivo para regresión
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),  # Regularización L1
            'num_leaves': trial.suggest_int('num_leaves', 2, 100),  # Número de hojas
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),  # Muestreo de características
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),  # Muestreo de muestras
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  # Frecuencia de muestreo
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)  # Mínimo de muestras en las hojas
        }

        regressor = lgb.LGBMRegressor(**param)  # Crear el regresor LightGBM

        if perdida == "MAE":
            scores = cross_val_score(regressor, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_absolute_error')  # Validación cruzada
        
        elif perdida == "MSE":
            scores = cross_val_score(regressor, X_train, Y_train.values.ravel(), cv=3, scoring='neg_mean_squared_error')

        error = np.abs(scores.mean())  # Valor absoluto del error

        return mse
    
    mse = study.best_trial.value  # Mejor valor del error cuadrado medio
    parametros = study.best_trial.params  # Mejores hiperparámetros encontrados

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))  # Crear un estudio de optimización de Optuna
    study.optimize(objective, n_trials=trials_n)  # Optimizar los hiperparámetros con Optuna

    mse = study.best_trial.value  # Mejor valor del error cuadrado medio
    parametros = study.best_trial.params  # Mejores hiperparámetros encontrados

    lgbm = lgb.LGBMRegressor(**study.best_trial.params)  # Crear un regresor LightGBM con los mejores hiperparámetros
    lgbm = lgb.LGBMRegressor()  # Crear un regresor LightGBM por defecto
    lgbm = lgbm.fit(X_train, Y_train.values.ravel())  # Ajustar el modelo LightGBM

    qebr = pd.DataFrame(zip(X_train.columns, lgbm.feature_importances_ / np.sum(lgbm.feature_importances_) * 100), columns=["Explicativa", "FE"])  # Importancia de las características según LightGBM

    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_train)

    qebr['SHAP'] = np.mean(shap_values, axis=0)  # Valor SHAP promedio
    qebr['SHAP_ABS'] = np.abs(np.mean(shap_values, axis=0))  # Valor absoluto del valor SHAP promedio
    qebr = qebr.set_index("Explicativa")  # Establecer la columna "Explicativa" como índice del DataFrame

    qebr[f"{perdida}_OUT"] = mse  # Agregar el valor del error cuadrado medio al DataFrame
    qebr[f"{perdida}_IN"] = mean_squared_error(Y_test, lgbm.predict(X_test))

    X_FE = list(qebr[qebr.FE > 1].index)  # Obtener las características con una importancia mayor que cero
    regression_results = run_regression(Y_train, X_train[X_FE])  # Realizar una regresión usando las características seleccionadas
    qebr = qebr.merge(regression_results, "outer", "Explicativa")  # Combinar los resultados de la regresión con el DataFrame qebr
    qebr = qebr.sort_values("FE", ascending=False)  # Ordenar el DataFrame por la importancia de las características
    qebr["PARAM"] = str(parametros)
    qebr.columns = [f"LB_{i}" for i in qebr.columns]  # Renombrar las columnas del DataFrame
    qebr = qebr.rename(columns={"LB_Explicativa": "Explicativa"})  # Renombrar la columna "LB_Explicativa" a "Explicativa"

    return qebr
