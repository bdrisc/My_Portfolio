import matplotlib
matplotlib.use('Agg')

import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from flask import send_file
from flask import Flask, request, jsonify
import pandas as pd
from utils.data_loader import load_data

dict_color = {
    "Four-Seam": "#FF0000",
    "Two-Seam": "#FF6666",
    "Sinker": "#FFA500",
    "ChangeUp": "#00CC66",
    "Slider": "#FFFF00",
    "Curveball": "#ADD8E6",
    "Cutter": "#8B0000",
    "Splitter": "#008080"
}

def standardize_pitch_types(df):
    pitch_map = {
        "FourSeamFastBall": "Four-Seam",
        "Four-Seam": "Four-Seam",
        "Fastball": "Four-Seam",
        "2-Seam": "Two-Seam",
        "TwoSeamFastBall": "Two-Seam",
        "Sinker": "Sinker",
        "ChangeUp": "ChangeUp",
        "Slider": "Slider",
        "Curveball": "Curveball",
        "Cutter": "Cutter",
        "Splitter": "Splitter"
    }
    df["TaggedPitchType"] = df["TaggedPitchType"].map(lambda x: pitch_map.get(x, x))
    return df

app = Flask(__name__)

DATA_PATH = r"C:\Users\brend\pitch_api\data\All Cape League Trackman.csv"
print("Loading data from:", DATA_PATH)
df = load_data(DATA_PATH)
df = standardize_pitch_types(df)
print("CSV loaded successfully with", len(df), "rows.")

@app.route("/")
def home():
    print("Home route hit")
    return "Cape League Pitch API is running"

@app.route("/api/pitches", methods=["GET"])
def get_pitches():
    print("API route hit")
    pitcher = request.args.get("pitcher")
    print("Pitcher:", pitcher)
    team = request.args.get("team")

    filtered = df.copy()

    if pitcher:
        filtered = filtered[filtered["Pitcher"].str.contains(pitcher, case=False, na=False)]
    if team:
        filtered = filtered[filtered["PitcherTeam"].str.contains(team, case=False, na=False)]

    if filtered.empty:
        return jsonify({"error": "No matching data found."}), 404

    summary = (
        filtered.groupby("TaggedPitchType")
        .agg({
            "Pitcher": "count",
            "RelSpeed": "mean",
            "SpinRate": "mean",
            "HorzBreak": "mean",
            "InducedVertBreak": "mean"
        })
        .rename(columns={"RelSpeed": "avg_velocity", "SpinRate": "avg_spin", "Pitcher": "pitch_count", "HorzBreak": "hz_break", "InducedVertBreak": "iv_break"})
        .reset_index()
    )

    response = {
        "query": {
            "pitcher": pitcher,
            "team": team
        },
        "results": summary.to_dict(orient="records")
    }
    print("Filtered rows:", len(filtered))

    return jsonify(response)

@app.route("/api/plot", methods=["GET"])
def plot_pitch_movement():
    pitcher = request.args.get("pitcher")
    if not pitcher:
        return jsonify({"error": "Missing 'pitcher' parameter"}), 400

    required_cols = ["Pitcher", "TaggedPitchType", "HorzBreak", "InducedVertBreak", "BatterSide", "PitcherThrows"]
    for col in required_cols:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    player_df = df[
        (df["Pitcher"].str.strip().str.contains(pitcher, case=False, na=False)) &
        df["TaggedPitchType"].notna() &
        df["BatterSide"].isin(["Right", "Left"])
    ].copy()

    if player_df.empty:
        return jsonify({"error": "No data found for this pitcher."}), 404

    pitch_types_used = player_df["TaggedPitchType"].unique()
    color_map = {pitch: dict_color.get(pitch, "black") for pitch in pitch_types_used}

    plt.figure(figsize=(7, 7))
    sns.set(style="white")

    sns.scatterplot(
        data=player_df,
        x="HorzBreak",
        y="InducedVertBreak",
        hue="TaggedPitchType",
        palette=color_map,
        s=80,
        edgecolor="none",
        alpha=0.7,
        legend=False
    )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    for tick in range(-20, 21, 10):
        plt.axhline(tick, color='#D3D3D3', linestyle='--', linewidth=0.75)
        plt.axvline(tick, color='#D3D3D3', linestyle='--', linewidth=0.75)

    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xticks(range(-20, 21, 10), fontsize=12)
    plt.yticks(range(-20, 21, 10), fontsize=12)
    plt.xlabel("Horizontal Break", fontsize=13)
    plt.ylabel("Induced Vertical Break", fontsize=13)
    plt.title(f"{pitcher} Pitch Movement", fontsize=16, fontweight='bold')
    plt.gca().set_aspect('equal', adjustable='box')

    PitcherThrows = player_df["PitcherThrows"].dropna().iloc[0] if not player_df["PitcherThrows"].dropna().empty else None
    if PitcherThrows == "Right":
        plt.text(-24, -24, 'Glove Side', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black'), fontsize=8)
        plt.text(24, -24, 'Arm Side', ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black'), fontsize=8)
    elif PitcherThrows == "Left":
        plt.text(24, -24, 'Glove Side', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black'), fontsize=8)
        plt.text(-24, -24, 'Arm Side', ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black'), fontsize=8)

    sns.despine()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
