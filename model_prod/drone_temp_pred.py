import pandas as pd 
import numpy as np 
import seaborn as sns     

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib as plt  


data_dir = Path(__file__).parent
file_path = data_dir / "teachpandas" / "sample_drone_flights.csv"
if not file_path.exists():
    raise ValueError(f"File Not Found in {file_path}")
data = pd.read_csv(file_path)

def _clean_colnames(df):
    """ validate data cols"""
    df.columns = (df.columns.
                str.lower().
                str.strip().
                str.replace("(" ," ").
                str.replace(" ", "_").
                str.replace(")", " ").
                str.strip()
                )
    return df

def transform_training_data(df):
    """ Clean and Transform input data for model development"""
    data = df.copy()
    df_clean = _clean_colnames(data)

    for col in [col for col in df_clean if "datetime" in col]:
        df_clean[col] = pd.to_datetime(df_clean[col])
          
    df_clean = df_clean.sort_values("datetime_utc",kind="stable").dropna(subset=["datetime_utc"])         
    dropped_cols = [
        "time_millisecond",
        "wind_speed_mph",
        "wind_direction_degrees",
        "isvideo",
        "isphoto",
        "voltagecell1",
        "voltagecell2",
        "voltagecell3",
        "voltagecell4",
        "voltagecell5",
        "voltagecell6",
        "message", 
    ]
    missing = [col for col in dropped_cols if col not in df_clean.columns]
    if missing:
        raise ValueError(f"Input data is missing these columns {missing}")
    
    df_new = df_clean.drop(columns=dropped_cols)
    numerical_col = (df_new.drop(columns=["battery_temperature_f", "datetime_utc"]).select_dtypes(include="number").columns.tolist())
    categorical_col = (df_new.select_dtypes(include=["object", "string"]).columns.tolist())
    target = "battery_temperature_f"
    df_new[target] = pd.to_numeric(df_new["target"], errors="coerce")
    num_preprocess = Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler())
                                    ]
                            )
    cat_preprocess = Pipeline(steps=[("impute", SimpleImputer    (strategy="most_frequent")), 
                                    ("onehotencode", 
                                    OneHotEncoder(handle_unknown="ignore"))])

    col_transform = ColumnTransformer([("num", num_preprocess, numerical_col),
                                    ("cat", cat_preprocess, categorical_col)])
    comb_cols = numerical_col + categorical_col

    df_new["datetime_utc"] = pd.to_datetime(df_new["datetime_utc"])
    df_sorted = df_new.sort_values("datetime_utc")

    X = df_sorted[comb_cols].copy()
    y = df_sorted[target]
    return X, y, col_transform

def select_train_test_time_seres(X_orig, y_orig):
    """ Select and transform time series drone data """
    X = X_orig.copy()
    y = y_orig.copy()
    cutoff = int(len(X) * 0.8)
    X_dev = X.iloc[:cutoff]
    y_dev = y.iloc[:cutoff]
    X_test = X.iloc[cutoff:]
    y_test = y.iloc[cutoff:]
    return X_dev, y_dev, X_test, y_test
model_configs = {
    "Ridge": {
        "model": Ridge(solver="lsqr"),
        "params": {
            "model__alpha": [
                0.001, 0.01, 0.1, 1, 10, 100, 1000
            ],
        },
        "n_iter": 7,
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=1,
        ),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__max_features": [0.5, 0.8, 1.0],
        },
        "n_iter": 20,
    },

    "XGBoost": {
        "model": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        ),
        "params": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [2, 3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_lambda": [1, 5, 10],
        },
        "n_iter": 20,
    },
}

def build_crossvalidation_model(tscv, col_transform, model_configs, X_dev, y_dev):
    """ build and store cross validation results"""
        
    searches = {}
    results = []
    for name, config in model_configs.items():
        pipeline = Pipeline([
            ("preprocess", clone(col_transform)),
            ("model", clone(config["model"])),
        ])
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            scoring="neg_mean_absolute_error",
            cv=tscv,
            refit=True,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            error_score="raise",
        )
        search.fit(X_dev, y_dev)
        searches[name] = search

        results.append({
            "model": name,
            "cv_mae": -search.best_score_,
            "best_params": search.best_params_,
        })
    comparison = (
        pd.DataFrame(results)
        .sort_values("cv_mae")
        .reset_index(drop=True)
    )
    return comparison, searches


def export_holdout_plots(holdout, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(holdout.index, holdout["actual_f"], label="Actual")
    ax.plot(holdout.index, holdout["predicted_f"], label="Predicted", alpha=0.8)
    ax.set(
        title=f"{model_name} — holdout actual vs. predicted",
        xlabel="Holdout observation in chronological order",
        ylabel="Battery temperature (°F)",
    )
    ax.legend()
    fig.savefig(output_dir / "holdout_actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(holdout.index, holdout["residual_f"], linewidth=0.9)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        title=f"{model_name} — holdout residuals",
        xlabel="Holdout observation in chronological order",
        ylabel="Actual minus predicted (°F)",
    )
    fig.savefig(output_dir / "holdout_residuals.png", dpi=200)
    plt.close(fig)


def pipeline_orchestration(data, output_dir="model_outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, col_transform = transform_training_data(data)
    X_dev, y_dev, X_test, y_test = select_train_test_time_seres(X, y)

    comparison, searches = build_crossvalidation_model(
        TimeSeriesSplit(n_splits=5),
        col_transform,
        model_configs,
        X_dev,
        y_dev,
    )

    winner = comparison.iloc[0]
    model_name = winner["model"]
    best_pipeline = searches[model_name].best_estimator_

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    comparison.to_csv(output_dir / "cv_comparison.csv", index=False)

    # Evaluate the saved artifact using the original holdout features.
    loaded_pipeline = joblib.load(model_path)
    predicted = loaded_pipeline.predict(X_test)

    holdout = pd.DataFrame({
        "actual_f": y_test.to_numpy(),
        "predicted_f": predicted,
    })
    holdout["residual_f"] = holdout["actual_f"] - holdout["predicted_f"]
    holdout.index.name = "holdout_row"

    metrics = pd.DataFrame([{
        "model": model_name,
        "cv_mae_f": winner["cv_mae"],
        "holdout_rows": len(holdout),
        "holdout_mae_f": mean_absolute_error(y_test, predicted),
        "holdout_rmse_f": mean_squared_error(y_test, predicted) ** 0.5,
        "holdout_r2": r2_score(y_test, predicted),
    }])

    metrics.to_csv(output_dir / "holdout_metrics.csv", index=False)
    holdout.to_csv(output_dir / "holdout_predictions.csv")
    export_holdout_plots(holdout, model_name, output_dir)

    return comparison, metrics


def main():
    """ Orchestrate Pipeline to run code """
    print("Initializing Model Development ....")
    cv_comparison, holdout_metrics = pipeline_orchestration(data)

    print(cv_comparison.to_string(index=False))
    print(holdout_metrics.to_string(index=False))
    print("Model Development and Storage Complete")
if __name__ == "__main__":
    main()