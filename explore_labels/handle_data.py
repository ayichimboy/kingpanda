import pandas as pd       
import matplotlib.pyplot as plt   
from pathlib import Path         

def validation_process(file_path):
    """ Validate work process and data path """
    if not file_path.exists():
        raise FileExistsError("Specific File Type Doesn't Exist")
    if file_path.suffix not in [".csv", ".xlsx"]:
        raise ValueError("File type not present in filepath")
    df = pd.read_csv(file_path)
    if len(set(df.columns)) == len(df.columns):
        df = df.loc[:, df.columns.drop_duplicates()] 
    for col in [col for col in df.columns if "date" in col]:
        df[col] = pd.to_datetime(df[col])
        return df
    
    
def _clean_columns(df):
    """ clean column names for analysis"""
    dfx = df.copy()
    dfx.columns = dfx.columns.str.strip()
    return dfx 


def transform_cleaned_data(dfx):
    df = dfx.copy()
    df_clean = _clean_columns(df)
    selected_cols = [
        "datetime(utc)",
        "flycState",
        'current(A)',
        "battery_temperature(f)",
        "speed(mph)",
        "voltage(v)",
        "longitude",
        "latitude",
        "time(millisecond)",
    ]
    missing = [col for col in selected_cols if col not in df_clean.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    new_df = df_clean[selected_cols]
    df_sorted = new_df.sort_values(by=["datetime(utc)"])
    return df_sorted


def plot_save_file(dfx, dir="saved_plots"):
    """ Plot and save png file of drone fly state"""
    save_dir = Path(__file__).resolve().parent / "output_dir"
    save_dir.mkdir(parents=True, exist_ok=True)

    for flystate, rows in dfx.groupby("flycState", sort=False):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

        ax[0].scatter(
            rows["speed(mph)"],
            rows["battery_temperature(f)"],
            s=10,
            alpha=0.4,
        )
        ax[0].set(
            xlabel="Drone speed (mph)",
            ylabel="Battery temperature (°F)",
        )

        ax[1].scatter(rows["longitude"], rows["latitude"], s=10, alpha=0.4)
        ax[1].set(xlabel="Longitude", ylabel="Latitude")
        ax[1].tick_params(axis="x", labelrotation=30)
        ax[1].ticklabel_format(style="plain", useOffset=False)

        fig.suptitle(f"Drone Flight Mode: {flystate} (n={len(rows):,})")
        fig.savefig(save_dir / f"drone_flight_{flystate}.png", dpi=200)
        plt.close(fig)

    return save_dir.resolve()

            
def orchesttrate_pipeline():
    print("Starting Process ...")
    data_dir = Path(__file__).parent.parent
    file_path = data_dir / "teachpandas" / "sample_drone_flights.csv"
    df_val = validation_process(file_path)
    df_trans = transform_cleaned_data(df_val)
    plot_save_file(df_trans, dir="saved_plots")
    print("Process Completed ... Files Saved Successfully ✅✅")
if __name__=="__main__":
        orchesttrate_pipeline()