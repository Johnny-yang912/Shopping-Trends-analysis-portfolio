# utils_shopping.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Literal
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# models - classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# models - regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, mean_absolute_error, mean_squared_error, r2_score
)

# ---------- 讀檔 / 基礎清理 ----------

def load_and_clean(path: str, drop_cols: Optional[list]=None) -> pd.DataFrame:
    """讀取 CSV，去重與基本缺失處理（示範版：丟掉含缺失列）。"""
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.drop_duplicates()
    df = df.dropna(how="any")
    return df

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """分離特徵/目標。"""
    assert target in df.columns, f"找不到目標欄位 {target}"
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

# ---------- 前處理 ----------

def make_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool=False
) -> ColumnTransformer:
    """
    - 類別欄：One-Hot (ignore unknown)
    - 數值欄：預設 passthrough；如 scale_numeric=True 則做 StandardScaler
    """
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_transformer = ("num", StandardScaler(), num_cols) if scale_numeric else ("num", "passthrough", num_cols)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            num_transformer,
        ]
    )

# ---------- 一站式訓練與評估 ----------

def train_and_eval(
    X: pd.DataFrame, y: pd.Series,
    task: Literal["classification","regression"]="classification",
    model: Optional[str]=None,
    test_size: float=0.2, random_state: int=42,
    scale_numeric: bool=False,
    tune: bool=False, n_iter: int=20
) -> Dict[str, Any]:
    """
    一行完成：切分 → 前處理 → 訓練 → 評估。
    task: "classification" 或 "regression"
    model:
      - 分類: "logreg" (預設), "rf"
      - 迴歸: "linreg" (預設), "rf"
    tune=True 時使用 RandomizedSearchCV 調參並印出最佳參數。
    """

    # 方便：若 y 是 "Yes"/"No" 之類，轉 1/0（只在分類任務做）
    if task == "classification" and y.dtype == "object":
        unique = set(map(str, y.unique()))
        yes_no_sets = [
            {"Yes","No"}, {"YES","NO"}, {"Y","N"}, {"True","False"}, {"TRUE","FALSE"}
        ]
        if any(unique == s for s in yes_no_sets):
            mapping = {list(s)[0]: 1 for s in yes_no_sets if "Yes" in s or "YES" in s or "Y" in s or "True" in s or "TRUE" in s}
            # 保險起見，明確列常見對應
            mapping = {"Yes":1,"No":0,"YES":1,"NO":0,"Y":1,"N":0,"True":1,"False":0,"TRUE":1,"FALSE":0}
            y = y.map(mapping).astype(int)

    pre = make_preprocessor(X, scale_numeric=scale_numeric)

    # 預設模型
    if task == "classification":
        model = model or "logreg"
        if model == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            param_dist = {
                "clf__C": np.logspace(-3, 2, 10),
                "clf__solver": ["liblinear","lbfgs"]
            }
            scorer, scoring_name = "f1", "F1"
        elif model == "rf":
            clf = RandomForestClassifier(random_state=random_state)
            param_dist = {
                "clf__n_estimators": [200,300,400,500],
                "clf__max_depth": [None,5,10,20,30],
                "clf__min_samples_split": [2,5,10]
            }
            scorer, scoring_name = "f1", "F1"
        else:
            raise ValueError("classification 模型請用 'logreg' 或 'rf'")
    else:
        model = model or "linreg"
        if model == "linreg":
            clf = LinearRegression()
            param_dist = None  # 線性迴歸無超參數
            scorer, scoring_name = "neg_root_mean_squared_error", "RMSE"
        elif model == "rf":
            clf = RandomForestRegressor(random_state=random_state)
            param_dist = {
                "clf__n_estimators": [200,300,400,500],
                "clf__max_depth": [None,5,10,20,30],
                "clf__min_samples_split": [2,5,10]
            }
            scorer, scoring_name = "neg_root_mean_squared_error", "RMSE"
        else:
            raise ValueError("regression 模型請用 'linreg' 或 'rf'")

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # 資料切分
    stratify = y if (task == "classification" and len(pd.Series(y).unique()) > 1) else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    # 是否調參
    if tune and param_dist:
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=n_iter,
            cv=3, n_jobs=-1, random_state=random_state, scoring=scorer
        )
        search.fit(Xtr, ytr)
        pipe = search.best_estimator_
        print(f"最佳參數 ({scoring_name}):", search.best_params_)
    else:
        pipe.fit(Xtr, ytr)

    # 評估
    if task == "classification":
        yhat = pipe.predict(Xte)
        metrics = {
            "accuracy": accuracy_score(yte, yhat),
            "precision": precision_score(yte, yhat, average="weighted", zero_division=0),
            "recall": recall_score(yte, yhat, average="weighted", zero_division=0),
            "f1": f1_score(yte, yhat, average="weighted", zero_division=0),
        }
        # AUC（僅二元且支援 predict_proba 時）
        try:
            if len(pd.Series(yte).unique()) == 2 and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(Xte)[:, 1]
                metrics["roc_auc"] = roc_auc_score(yte, proba)
        except Exception:
            pass
        metrics["report"] = classification_report(yte, pipe.predict(Xte), digits=4, zero_division=0)
    else:
        yhat = pipe.predict(Xte)
        try:
            from sklearn.metrics import root_mean_squared_error
            rmse_val = root_mean_squared_error(yte, yhat)
        except Exception:
            from sklearn.metrics import mean_squared_error
            rmse_val = np.sqrt(mean_squared_error(yte, yhat))

        metrics = {
            "mae": mean_absolute_error(yte, yhat),
            "rmse": rmse_val,
            "r2": r2_score(yte, yhat),
            }

    return {"model": pipe, "metrics": metrics}
