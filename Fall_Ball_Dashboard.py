import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Polygon
import re
from scipy.stats import gaussian_kde
import io

st.set_page_config(layout="wide")

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-repeat: no-repeat;
            background-position: center top; 
            background-size: 62%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("images/stonehill_skyhawks_2012-pres_a.webp")

def load_data():
    file_path = "data/2025 SB Fall Ball Logging.xlsx"
    df = pd.read_excel(file_path)
    return df

df = load_data()


st.markdown("<h1 style='text-align: center;'>Fall Ball Player Dashboard</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;   /* default is much larger */
    }
    </style>
    """,
    unsafe_allow_html=True
)

player_type = st.sidebar.radio(
    "Select Player Type",
    ["Hitter", "Pitcher"]
)


st.sidebar.header("Select Player")
if player_type == "Hitter":
    all_players = sorted(df["Hitter"].dropna().unique())
else:
    all_players = sorted(df["Pitcher"].dropna().unique())

selected_player = st.sidebar.selectbox(player_type, all_players)

if player_type == "Hitter":
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "Basic Stats",
            "Quality of Contact",
            "Type of Contact",
            "Plate Skills",
            "Advanced Spray Chart",
            "Launch Angle EV Chart",
            "Hard-Hit & Barrel Locations",
            "Batted Ball Profile",
        ]
    )
else:
    page = st.sidebar.radio(
        "Select Dashboard Page",
        [
            "Basic Stats",
            "Advanced Stats",
            "Pitch Usage",
            "Contact Allowed",
            "Control and Whiffs",
            "Velocity Distribution",
            "Release Points",
            "Pitch Metrics",
            "Pitch Locations"
        ]
    )

if player_type == "Hitter":
    st.sidebar.header("Filters")
    df["Strike Zone Side"] = pd.to_numeric(df["Strike Zone Side"], errors="coerce")
    df["Strike Zone Height"] = pd.to_numeric(df["Strike Zone Height"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    handedness_options = ["All"] + sorted(df["T"].dropna().unique().tolist())
    handedness = st.sidebar.multiselect("Pitcher Throws", handedness_options, default=["All"])

    pitch_type_options = ["All"] + sorted(df["Pitch Type"].dropna().unique().tolist())
    pitch_types = st.sidebar.multiselect("Pitch Type", pitch_type_options, default=["All"])

    count_options = ["All"] + sorted(df["Count"].dropna().unique().tolist())
    counts = st.sidebar.multiselect("Count", count_options, default=["All"])

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    
else:
    st.sidebar.header("Filters")

    df["Strike Zone Side"] = pd.to_numeric(df["Strike Zone Side"], errors="coerce")
    df["Strike Zone Height"] = pd.to_numeric(df["Strike Zone Height"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    player_df = df[df["Pitcher"].str.strip() == selected_player.strip()]

    handedness_options = ["All"] + sorted(player_df["B"].dropna().unique().tolist())
    handedness = st.sidebar.multiselect("Batter Side", handedness_options, default=["All"])

    pitch_type_options = ["All"] + sorted(player_df["Pitch Type"].dropna().unique().tolist())
    pitch_types = st.sidebar.multiselect("Pitch Type", pitch_type_options, default=["All"])

    count_options = ["All"] + sorted(df["Count"].dropna().unique().tolist())
    counts = st.sidebar.multiselect("Count", count_options, default=["All"])

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

    filtered_df = player_df.copy()

    if "All" not in handedness:
        filtered_df = filtered_df[filtered_df["B"].isin(handedness)]

    if "All" not in pitch_types:
        filtered_df = filtered_df[filtered_df["Pitch Type"].isin(pitch_types)]

    if "All" not in counts:
        filtered_df = filtered_df[filtered_df["Count"].isin(counts)]

    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
    ]
    
st.sidebar.subheader("Pitch Location (Hitter/Catcher POV)")
zone_options = [
    "Up and Left", "Up and Middle", "Up and Right",
    "Middle Left", "Middle Middle", "Middle Right",
    "Low and Left", "Low and Middle", "Low and Right",
    "Out High", "Out Low", "Out Inside", "Out Outside"
]
zone_options_with_all = ["All"] + zone_options
selected_zones = st.sidebar.multiselect("Select Zones", zone_options_with_all, default=[])

def assign_zone(x, y):
    if -8.9 <= x < -3.96 and 34 <= y <= 42.5: return "Up and Right"
    if -3.96 <= x <= 3.96 and 34 <= y <= 42.5: return "Up and Middle"
    if 3.96 < x <= 8.9 and 34 <= y <= 42.5: return "Up and Left"

    if -8.9 <= x < -3.96 and 26 <= y < 34: return "Middle Right"
    if -3.96 <= x <= 3.96 and 26 <= y < 34: return "Middle Middle"
    if 3.96 < x <= 8.9 and 26 <= y < 34: return "Middle Left"

    if -8.9 <= x < -3.96 and 17.5 <= y < 26: return "Low and Right"
    if -3.96 <= x <= 3.96 and 17.5 <= y < 26: return "Low and Middle"
    if 3.96 < x <= 8.9 and 17.5 <= y < 26: return "Low and Left"

    if y > 42.5: return "Out High"
    if y < 17.5: return "Out Low"
    if x < -8.9: return "Out Right"
    if x > 8.9: return "Out Left"

    return "Other"

df["PitchZone"] = df.apply(lambda row: assign_zone(row["Strike Zone Side"], row["Strike Zone Height"]), axis=1)

if player_type == "Hitter":
    player_df = df[df["Hitter"] == selected_player]

    if "All" not in handedness:
        player_df = player_df[player_df["T"].isin(handedness)]

else:
    player_df = df[df["Pitcher"] == selected_player]

    if "All" not in handedness:
        player_df = player_df[player_df["B"].isin(handedness)]

if "All" not in pitch_types:
    player_df = player_df[player_df["Pitch Type"].isin(pitch_types)]
if "All" not in counts:
    player_df = player_df[player_df["Count"].isin(counts)]
    
player_df = player_df[player_df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]
filtered_zones = zone_options if "All" in selected_zones else selected_zones
if filtered_zones:
    player_df = player_df[player_df["PitchZone"].isin(filtered_zones)]

def calculate_basic_stats(player_df):
    stats = {}
    pa_mask = player_df["AB Result"].isin(["1B","2B","3B","HR","BB","K","HBP","SAC", "Out", "DP"])
    pa_df = player_df[pa_mask]
    stats["PA"] = len(pa_df)
    hbp = (pa_df["AB Result"]=="HBP").sum()
    stats["HBP"] = hbp
    sf = ((pa_df["AB Result"] == "SAC") & (pa_df["BBT"].isin(["LD", "OFFB"]))).sum()
    stats["BB"] = (pa_df["AB Result"]=="BB").sum()
    stats["SO"] = (pa_df["AB Result"]=="K").sum()
    AB = stats["PA"] - stats["BB"] - hbp - sf
    stats["AB"] = AB
    stats["H"] = pa_df["AB Result"].isin(["1B","2B","3B","HR"]).sum()
    stats["1B"] = (pa_df["AB Result"]=="1B").sum()
    stats["2B"] = (pa_df["AB Result"]=="2B").sum()
    stats["3B"] = (pa_df["AB Result"]=="3B").sum()
    stats["HR"] = (pa_df["AB Result"]=="HR").sum()
    rbi, runs = 0, 0
    for val in pa_df["Misc"].dropna():
        val = str(val).upper().strip()

        if "RBI" in val:
            match = re.search(r"(\d+)\s*RBI", val)
            if match:
                num = int(match.group(1))
            else:
                num = 1
            rbi += num

        if val.startswith("R") and not val.startswith("RBI"):
            runs += 1
            
    stats["RBI"] = rbi
    stats["R"] = runs
    TB = stats["1B"] + (2*stats["2B"]) + (3*stats["3B"]) + (4*stats["HR"])
    stats["AVG"] = f"{stats['H']/AB:.3f}" if AB > 0 else "0.000"
    stats["OBP"] = f"{(stats['H']+stats['BB']+hbp)/(AB+stats['BB']+hbp+sf):.3f}" if (AB+stats['BB']+hbp+sf) > 0 else "0.000"
    stats["SLG"] = f"{TB/AB:.3f}" if AB > 0 else "0.000"
    stats["OPS"] = f"{float(stats['OBP'])+float(stats['SLG']):.3f}"
    stats["ISO"] = f"{float(stats['SLG'])-float(stats['AVG']):.3f}"
    stats["BB%"] = round((stats["BB"]/stats["PA"])*100,1) if stats["PA"]>0 else 0
    stats["K%"] = round((stats["SO"]/stats["PA"])*100,1) if stats["PA"]>0 else 0
    stats["SB"] = (pa_df["Misc"]=="SB").sum()
    stats["CS"] = (pa_df["Misc"]=="CS").sum()
    return stats

def calculate_pitcher_basic_stats(player_df):
    """
    Calculate basic pitcher stats from pitch-by-pitch data.
    AB Result is used to calculate outs, strikeouts, hits, walks, HBP, and runs.
    """

    def count_outs(row):
        result = row["AB Result"]
        misc = str(row.get("Misc", ""))
        outs = 0

        if result == "DP":
            outs += 2
        elif result in ["Out", "K", "SAC"]:
            outs += 1

        if "PO" in misc:
            outs += 1

        return outs

    total_outs = player_df.apply(count_outs, axis=1).sum()

    full_innings = total_outs // 3
    remainder_outs = total_outs % 3
    IP = float(f"{full_innings}.{remainder_outs}")

    SO = (player_df["AB Result"] == "K").sum()
    BB = (player_df["AB Result"] == "BB").sum()
    HBP = (player_df["AB Result"] == "HBP").sum()
    WP = player_df["Misc"].fillna("").str.contains("WP").sum()

    hits = player_df["AB Result"].isin(["1B", "2B", "3B", "HR"]).sum()

    WHIP = (BB + hits) / (total_outs / 3) if total_outs > 0 else 0

    ER = player_df["Misc"].fillna("").str.contains("ER").sum()
    ERA = (ER * 9) / (total_outs / 3) if total_outs > 0 else 0

    stats = {
        "IP": IP,
        "ERA": f"{ERA:.2f}",
        "WHIP": f"{WHIP:.2f}",
        "SO": int(SO),
        "BB": int(BB),
        "HBP": int(HBP),
        "WP": int(WP),
        "H": int(hits),
        "ER": int(ER)
    }

    return stats

def is_barrel(ev, la):
    if pd.isna(ev) or pd.isna(la):
        return False
    if ev < 95.5:
        return False
    if ev == 95.5:
        return 26 <= la <= 30
    if ev == 96.5:
        return 25 <= la <= 31
    if ev == 97.5:
        return 24 <= la <= 33
    if 98.5 <= ev <= 112.5:
        lower = max(8, 24 - (ev - 99))
        upper = min(50, 33 + (ev - 99) * 2)
        return lower <= la <= upper
    if ev >= 113.5:
        return 8 <= la <= 50
    return False


def calculate_quality_contact_stats(player_df):
    stats = {}
    
    if player_df.empty:
        return {k: 0 for k in ["AVG EV", "Barrel%", "Hard-Hit%", "Max EV", "90th EV", "Air EV"]}

    batted_balls = player_df.dropna(subset=["ExitVelocity"])
    
    ev = batted_balls["ExitVelocity"] if not batted_balls.empty else pd.Series(dtype=float)
    
    stats["AVG EV"] = f"{ev.mean():.1f}" if not ev.empty else "0.0"
    stats["Max EV"] = f"{ev.max():.1f}" if not ev.empty else "0.0"
    stats["90th EV"] = f"{np.percentile(ev, 90):.1f}" if not ev.empty else "0.0"
    
    air_ev = batted_balls.loc[batted_balls["BBT"].isin(["LD", "OFFB", "IFFB"]), "ExitVelocity"]
    stats["Air EV"] = f"{air_ev.mean():.1f}" if not air_ev.empty else "0.0"
    
    batted_balls["is_barrel"] = batted_balls.apply(
        lambda row: is_barrel(row["ExitVelocity"], row["LaunchAngle"]), axis=1
    )
    stats["Barrel%"] = round(batted_balls["is_barrel"].mean() * 100, 1) if not batted_balls.empty else 0
    
    stats["Hard-Hit%"] = round((ev >= 92).mean() * 100, 1) if not ev.empty else 0
    
    return stats

def calculate_contact_type_stats(player_df):
    stats = {}
    if player_df.empty or "BBD" not in player_df.columns or "BBT" not in player_df.columns:
        return {k: 0 for k in ["GB%", "LD%", "OFFB%", "IFFB%", "Pull%", "Center%", "Oppo%"]}

    batted_balls = player_df.dropna(subset=["BBD", "BBT"])
    total = len(batted_balls)
    if total == 0:
        return {k: 0 for k in ["GB%", "LD%", "OFFB%", "IFFB%", "Pull%", "Center%", "Oppo%"]}

    stats["GB%"] = round((batted_balls["BBT"] == "GB").sum() / total * 100, 1)
    stats["LD%"] = round((batted_balls["BBT"] == "LD").sum() / total * 100, 1)
    stats["OFFB%"] = round((batted_balls["BBT"] == "OFFB").sum() / total * 100, 1)
    stats["IFFB%"] = round((batted_balls["BBT"] == "IFFB").sum() / total * 100, 1)

    stats["Pull%"] = round((batted_balls["BBD"] == "Pull").sum() / total * 100, 1)
    stats["Center%"] = round((batted_balls["BBD"] == "Center").sum() / total * 100, 1)
    stats["Oppo%"] = round((batted_balls["BBD"] == "Oppo").sum() / total * 100, 1)

    return stats

def calculate_plate_metrics(player_df):
    stats = {}
    if player_df.empty:
        return {k: 0 for k in ["Swing%", "Contact%", "Z-Contact%", "O-Contact%", "Z-Swing%", "Chase%", "Pitches/PA"]}
    total_pitches = len(player_df)
    pa_mask = player_df["AB Result"].notna()
    total_pa = pa_mask.sum()
    swings_mask = player_df["Pitch Result"].isin(["Contact", "Miss"])
    stats["Swing%"] = round(swings_mask.mean() * 100, 1)
    contacts_mask = player_df["Pitch Result"] == "Contact"
    stats["Contact%"] = round(contacts_mask.sum() / swings_mask.sum() * 100, 1) if swings_mask.sum() > 0 else 0
    in_zone = [
    "Up and Left", "Up and Middle", "Up and Right",
    "Middle Left", "Middle Middle", "Middle Right",
    "Low and Left", "Low and Middle", "Low and Right"
    ]
    z_mask = player_df["PitchZone"].isin(in_zone)
    o_mask = ~z_mask
    z_swings_mask = swings_mask & z_mask
    o_swings_mask = swings_mask & o_mask

    stats["Z-Contact%"] = round((contacts_mask & z_mask).sum() / z_swings_mask.sum() * 100, 1) if z_swings_mask.sum() > 0 else 0
    stats["O-Contact%"] = round((contacts_mask & o_mask).sum() / o_swings_mask.sum() * 100, 1) if o_swings_mask.sum() > 0 else 0
    stats["Z-Swing%"] = round((swings_mask & z_mask).sum() / z_mask.sum() * 100, 1) if z_mask.sum() > 0 else 0
    stats["Chase%"] = round((swings_mask & o_mask).sum() / o_mask.sum() * 100, 1) if o_mask.sum() > 0 else 0
    stats["Pitches/PA"] = round(total_pitches / total_pa, 2) if total_pa > 0 else 0
    return stats

def calculate_pitcher_advanced_stats(player_df, fip_constant=4.853):
    """
    Calculate advanced pitching stats from pitch-by-pitch data.
    """

    SO = (player_df["AB Result"] == "K").sum()
    BB = (player_df["AB Result"] == "BB").sum()
    HBP = (player_df["AB Result"] == "HBP").sum()
    HR = (player_df["AB Result"] == "HR").sum()

    hits = player_df["AB Result"].isin(["1B", "2B", "3B", "HR"]).sum()

    def count_outs(result):
        if result == "DP":
            return 2
        elif result in ["Out", "K", "SAC"]:
            return 1
        return 0

    total_outs = player_df["AB Result"].apply(count_outs).sum()
    IP = total_outs / 3 if total_outs > 0 else 0

    PA = player_df["AB Result"].notna().sum()
    AB = player_df[~player_df["AB Result"].isin(["BB", "HBP", "SAC"])].shape[0]

    TB = (
        (player_df["AB Result"] == "1B").sum()
        + 2 * (player_df["AB Result"] == "2B").sum()
        + 3 * (player_df["AB Result"] == "3B").sum()
        + 4 * (player_df["AB Result"] == "HR").sum()
    )

    K_perc = SO / PA if PA > 0 else 0
    BB_perc = BB / PA if PA > 0 else 0
    H9 = (hits * 9 / IP) if IP > 0 else 0
    HR9 = (HR * 9 / IP) if IP > 0 else 0
    K9 = (SO * 9 / IP) if IP > 0 else 0
    BB9 = (BB * 9 / IP) if IP > 0 else 0
    BAA = hits / AB if AB > 0 else 0
    SLG = TB / AB if AB > 0 else 0

    FIP = (((13 * HR) + 3 * (BB + HBP) - 2 * SO) / IP + fip_constant) if IP > 0 else 0

    stats = {
        "FIP": round(FIP, 2),
        "K%": round(K_perc * 100, 1),
        "BB%": round(BB_perc * 100, 1),
        "H/9": round(H9, 2),
        "HR/9": round(HR9, 2),
        "K/9": round(K9, 2),
        "BB/9": round(BB9, 2),
        "BAA": f"{BAA:.3f}",
        "SLG": f"{SLG:.3f}",
    }

    return stats

if player_df.empty:
    st.warning("No data for this player with current filters.")
else:
    st.markdown("""
    <style>
    .stat-card {
        background-color: rgba(255,255,255,0.65);
        border-radius: 10px;
        padding: 6px 6px;
        text-align: center;
        font-weight: bold;
        color: black;
        min-height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .stat-card h5 { margin:0; font-size: 30px; }
    .stat-card h3 { margin:0; font-size: 35px; }
    </style>
    """, unsafe_allow_html=True)

    if page == "Basic Stats":
        if player_type == "Hitter":
            basic_stats = calculate_basic_stats(player_df)
            stat_order = ["PA","AB","H","1B","2B","3B","HR","R","RBI","BB","HBP","SO",
                          "AVG","OBP","SLG","OPS","ISO","BB%","K%"]
        else:
            basic_stats = calculate_pitcher_basic_stats(player_df)
            stat_order = ["IP","ERA","WHIP","SO","BB","HBP","WP"]

        cols_per_row = 5
        stat_items = [stat for stat in stat_order if stat in basic_stats]
        for row_start in range(0, len(stat_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, stat in enumerate(stat_items[row_start:row_start+cols_per_row]):
                cols[idx].markdown(f"<div class='stat-card'><h5>{stat}</h5><h3>{basic_stats[stat]}</h3></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            
    elif page == "Quality of Contact":
        qc_stats = calculate_quality_contact_stats(player_df)
        metric_order = ["AVG EV","Barrel%","Hard-Hit%","Max EV","90th EV","Air EV"]
        cols_per_row = 3
        for row_start in range(0, len(metric_order), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, stat in enumerate(metric_order[row_start:row_start+cols_per_row]):
                cols[idx].markdown(
                    f"""
                    <div class='stat-card' style="margin-bottom:24px;">
                        <h5>{stat}</h5>
                        <h3>{qc_stats[stat]}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    elif page == "Type of Contact":
        contact_stats = calculate_contact_type_stats(player_df)
        metric_order = ["GB%","LD%","OFFB%","IFFB%","Pull%","Center%","Oppo%"]
        cols_per_row = 4
        for row_start in range(0, len(metric_order), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, stat in enumerate(metric_order[row_start:row_start+cols_per_row]):
                cols[idx].markdown(
                    f"""
                    <div class='stat-card' style="margin-bottom:24px;">
                        <h5>{stat}</h5>
                        <h3>{contact_stats[stat]}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    elif page == "Plate Skills":
        plate_metrics = calculate_plate_metrics(player_df)
        metric_order = ["Swing%","Contact%","Z-Contact%","O-Contact%","Z-Swing%","Chase%","Pitches/PA"]
        cols_per_row = 4
        for row_start in range(0, len(metric_order), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, stat in enumerate(metric_order[row_start:row_start+cols_per_row]):
                cols[idx].markdown(
                    f"""
                    <div class='stat-card' style="margin-bottom:24px;">
                        <h5>{stat}</h5>
                        <h3>{plate_metrics[stat]}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    elif page == "Advanced Spray Chart":

        plot_df = player_df.copy()

        plot_df = plot_df[~((plot_df["Pitch Result"]=="Contact") & (plot_df["AB Result"].isna()))]

        plot_df = plot_df.dropna(subset=['Distance','ExitDirection','ExitVelocity','AB Result'])

        plot_df['X'] = plot_df['Distance'] * np.sin(np.radians(plot_df['ExitDirection']))
        plot_df['Y'] = plot_df['Distance']

        marker_map = {"1B":"o","2B":"s","3B":"P","HR":"D","Out":"x","Error":"^"}
        
        def velocity_color(ev):
            if ev < 75: return "darkblue"
            elif ev < 85: return "lightblue"
            elif ev < 95: return "orangered"
            elif ev < 105: return "darkred"
            else: return "black"

        fig, ax = plt.subplots(figsize=(10,8))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        infield = np.array([[0,0],[-90,90],[0,180],[90,90],[0,0]])
        ax.plot(infield[:,0], infield[:,1], color='white', linewidth=2)

        ax.plot([0, -230], [0, 230], color='white', linestyle='--', linewidth=1.5)
        ax.plot([0, 230], [0, 230], color='white', linestyle='--', linewidth=1.5)

        x_wall = np.array([-230, 0, 230])
        y_wall = np.array([230, 390, 230])
        coeffs = np.polyfit(x_wall, y_wall, 2)
        x_vals = np.linspace(-230, 230, 300)
        y_vals = np.polyval(coeffs, x_vals)
        ax.plot(x_vals, y_vals, color='white', linewidth=2)

        for _, row in plot_df.iterrows():
            result = row.get("AB Result","Out")
            ev = row["ExitVelocity"]
            ax.scatter(row['X'], row['Y'],
                       marker=marker_map.get(result,"o"),
                       color=velocity_color(ev),
                       s=60,
                       edgecolor='black',
                       alpha=0.8)

        marker_handles = [Line2D([0],[0], marker=m, color='w', label=l, markerfacecolor='gray', markersize=6, markeredgecolor='black')
                          for l,m in marker_map.items()]
        color_map = [("EV < 75","darkblue"),("EV 75-85","lightblue"),("EV 85-95","orangered"),("EV 95-105","darkred")]
        color_handles = [Line2D([0],[0], marker='o', color='w', label=l, markerfacecolor=c, markersize=6, markeredgecolor='black')
                         for l,c in color_map]
        first_legend = ax.legend(handles=marker_handles, title="Play Result", loc='upper left', fontsize=8, title_fontsize=9)
        ax.add_artist(first_legend)
        ax.legend(handles=color_handles, title="Exit Velocity", loc='upper right', fontsize=8, title_fontsize=9)

        ax.set_xlim(-300,300)
        ax.set_ylim(0,420)
        ax.set_aspect('equal')

        st.pyplot(fig, bbox_inches='tight', transparent=True)

    elif page == "Launch Angle EV Chart":
        chart_df = player_df[player_df["AB Result"].isin(["1B","2B","3B","HR","Out","Error"])].copy()
        if chart_df.empty:
            st.warning("No data for this player with current filters.")
        else:
            from PIL import Image
            import numpy as np

            color_map = {"1B":"cyan","2B":"yellow","3B":"green","HR":"orangered","Out":"lightgrey","Error":"white"}
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection="polar")
            fig.subplots_adjust(top=1, bottom=0.01)
            angles = np.radians(np.linspace(-90,90,500))
            radius = np.full_like(angles,110)
            ax.fill_between(angles,0,radius,color="lightgray",alpha=.15)
            ax.grid(color='darkgray', linewidth=0.8, linestyle='--')
            ax.spines['polar'].set_color('black')

            for outcome, group in chart_df.groupby("AB Result"):
                ax.scatter(np.radians(group["LaunchAngle"]), group["ExitVelocity"], 
                           label=outcome, color=color_map.get(outcome,"black"),
                           edgecolor="black", s=70, alpha=0.9)

            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.set_ylim(0,110)
            ax.set_yticks([80, 100])
            ax.set_yticklabels([])

            for ytick, label_text, x_offset in zip([80, 100], ["80 mph", "100 mph"], [104, 103]):
                ax.text(np.radians(x_offset), ytick + 3, label_text, color='white', fontsize=10, va='center', ha='left')

            for ytick, label_text, x_offset in zip([80, 100], ["80 mph", "100 mph"], [-91, -91]):
                ax.text(np.radians(x_offset), ytick, label_text, color='white', fontsize=10, va='center', ha='right')

            ax.set_xticks(np.radians([-90,0,35,90]))
            ax.set_xticklabels(["-90°","0°","35°","90°"], fontsize=10, color='white')

            handles, labels = ax.get_legend_handles_labels()
            order = ["1B","2B","3B","HR","Out","Error"]
            handle_dict = dict(zip(labels, handles))
            ordered_handles = [handle_dict[o] for o in order if o in handle_dict]
            ordered_labels = [o for o in order if o in handle_dict]
            ax.legend(ordered_handles, ordered_labels, loc="upper right", bbox_to_anchor=(1.05,1))

            try:
                img = Image.open("images/baseball2.png").convert("RGBA")
                data = np.array(img)
                r, g, b, a = data.T
                mask = a > 0
                data[..., :-1][mask.T] = 255
                white_img = Image.fromarray(data)

                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                imagebox = OffsetImage(white_img, zoom=0.09)
                ab = AnnotationBbox(imagebox, (0.175,0.479), xycoords='axes fraction', frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                st.info("Baseball image not loaded: " + str(e))

            st.pyplot(fig, bbox_inches='tight', transparent=True)

    elif page == "Hard-Hit & Barrel Locations":
        plot_df = player_df.copy()

        hard_hit_df = plot_df[plot_df["ExitVelocity"] >= 92].dropna(subset=["Strike Zone Side", "Strike Zone Height"])

        def safe_is_barrel(row):
            if pd.notna(row["ExitVelocity"]) and pd.notna(row["LaunchAngle"]):
                return is_barrel(row["ExitVelocity"], row["LaunchAngle"])
            return False

        plot_df["is_barrel"] = plot_df.apply(safe_is_barrel, axis=1)

        barreled_df = plot_df[plot_df["is_barrel"]].dropna(
            subset=["Strike Zone Side", "Strike Zone Height"]
        )
        fig, ax = plt.subplots(figsize=(7, 5))

        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        if not hard_hit_df.empty:
            ax.scatter(
                hard_hit_df["Strike Zone Side"],
                hard_hit_df["Strike Zone Height"],
                color="salmon",
                edgecolor="black",
                s=90,
                label="Hard-Hit"
            )

        if not barreled_df.empty:
            ax.scatter(
                barreled_df["Strike Zone Side"],
                barreled_df["Strike Zone Height"],
                color="#800000",
                edgecolor="black",
                s=100,
                label="Barrel"
            )

        zone_bottom = 17.5
        zone_top = 42.5
        zone_half_width = 8.7
        strike_zone = patches.Rectangle(
            (-zone_half_width, zone_bottom),
            zone_half_width * 2,
            zone_top - zone_bottom,
            linewidth=2,
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.3
        )
        ax.add_patch(strike_zone)

        ax.set_xlim(-18, 18)
        ax.set_ylim(14, 46)
    
        ax.set_xlabel("Batter's Perspective", fontsize = 15, color='white')
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.grid(True, color='white', linestyle=':', linewidth=1)
        ax.legend(loc="upper right")

        st.pyplot(fig)

        if hard_hit_df.empty:
            st.info(f"No hard-hit balls found for {selected_player}.")

    elif page == "Batted Ball Profile":
        profile_df = player_df.copy()
        profile_df = profile_df[
            (profile_df["AB Result"].isin(["1B","2B","3B","HR","Out", "DP"])) &
            (profile_df["LaunchAngle"].between(-20, 80)) &
            (profile_df["ExitDirection"].between(-60, 60))
        ].dropna(subset=["LaunchAngle", "ExitDirection", "Hitter"])

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(9,7))

        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        x_min, x_max = -50, 50
        y_min, y_max = -40, 80
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for line in [10, 25, 50]:
            ax.axhline(line, color='gray', linestyle='--', linewidth=1)
        for line in [-15, 15]:
            ax.axvline(line, color='gray', linestyle='--', linewidth=1)

        if profile_df.empty or len(profile_df) < 3:

            ax.text(0, 20, "No batted ball data available", color='white', fontsize=16,
                    ha='center', va='center', fontweight='bold')
        else:
            sns.kdeplot(
                data=profile_df,
                x="ExitDirection",
                y="LaunchAngle",
                fill=True,
                cmap="RdBu_r",
                bw_adjust=0.7,
                thresh=0.05,
                levels=100,
                ax=ax
            )
            
        ax.set_ylabel("")
        ax.set_xlabel("Direction (Left to Right)", color='white', fontsize=12)

        quadrant_labels = [
            ("Line Drive", 16.5),
            ("Ground Ball", -12.5),
            ("Fly Ball", 45),
            ("Popup", 70)
        ]


        for label, y in quadrant_labels:
            ax.text(x_min - 16, y, label, va='center', ha='left', fontsize=13, color='white')

        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.5)

        ax.grid(True, linestyle='--', linewidth=1, alpha=0.3, color='white')

        st.pyplot(fig, transparent=True)

    if page == "Advanced Stats" and player_type == "Pitcher":
        advanced_stats = calculate_pitcher_advanced_stats(player_df)
        stat_order = ["FIP","K%","BB%","H/9","HR/9","K/9","BB/9","BAA","SLG"]

        cols_per_row = 5
        stat_items = [stat for stat in stat_order if stat in advanced_stats]
        for row_start in range(0, len(stat_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, stat in enumerate(stat_items[row_start:row_start+cols_per_row]):
                cols[idx].markdown(
                    f"<div class='stat-card'><h5>{stat}</h5><h3>{advanced_stats[stat]}</h3></div>", 
                    unsafe_allow_html=True
                )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if player_type == "Pitcher" and page == "Pitch Usage":

        usage_counts = player_df["Pitch Type"].value_counts()
        usage_percent = (usage_counts / usage_counts.sum() * 100).round(1)
        display_pitch_df = usage_percent.reset_index()
        display_pitch_df.columns = ["Pitch Type", "Usage%"]
        display_pitch_df = display_pitch_df.sort_values("Usage%", ascending=False)

        if not display_pitch_df.empty:
            for index, row in display_pitch_df.iterrows():
                pitch = row["Pitch Type"]
                usage = row["Usage%"]

                st.markdown(
                    f"""
                    <div style="margin-bottom: 15px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">{pitch}</div>
                        <div style="background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="
                                background-color: #6a0dad;
                                width: {usage}%;
                                padding: 20px 0;
                                border-radius: 10px;
                                text-align: right;
                                color: white;
                                font-weight: bold;
                                padding-right: 10px;
                                font-size: 18px;
                                white-space: nowrap;
                            ">
                                {usage}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No pitch data available for this player with current filters.")

    elif page == "Contact Allowed" and player_type == "Pitcher":
        def calculate_qoc_allowed(player_df):
            stats = {}
            if player_df.empty:
                return {k: 0 for k in ["AVG EV","Hard-Hit%","Barrel%","GB%","LD%","OFFB%","IFFB%"]}
            
            batted_df = player_df[player_df["AB Result"].isin(["1B","2B","3B","HR","Out","E","DP"])]
            if batted_df.empty:
                return {k: 0 for k in ["AVG EV","Hard-Hit%","Barrel%","GB%","LD%","OFFB%","IFFB%"]}
            
            total = batted_df["BBT"].notna().sum()

            ev = batted_df["ExitVelocity"].dropna()
            batted_df["is_barrel"] = batted_df.apply(
                lambda row: is_barrel(row["ExitVelocity"], row.get("LaunchAngle", np.nan)), axis=1
            )
            stats["AVG EV"] = round(ev.mean(), 1) if not ev.empty else 0
            stats["Hard-Hit%"] = round((batted_df["ExitVelocity"] >= 92).mean() * 100, 1)
            stats["Barrel%"] = round(batted_df["is_barrel"].mean() * 100, 1)

            if total > 0:
                stats["GB%"]   = round((batted_df["BBT"] == "GB").sum() / total * 100, 1)
                stats["LD%"]   = round((batted_df["BBT"] == "LD").sum() / total * 100, 1)
                stats["OFFB%"] = round((batted_df["BBT"] == "OFFB").sum() / total * 100, 1)
                stats["IFFB%"] = round((batted_df["BBT"] == "IFFB").sum() / total * 100, 1)
            else:
                stats["GB%"] = stats["LD%"] = stats["OFFB%"] = stats["IFFB%"] = 0

            return stats

        qoc_allowed_stats = calculate_qoc_allowed(player_df)
        metric_order = ["AVG EV","Hard-Hit%","Barrel%","GB%","LD%","OFFB%","IFFB%"]
        cols_per_row = 3
        for row_start in range(0, len(metric_order), cols_per_row):
            with st.container():
                cols = st.columns(cols_per_row)
                for idx, stat in enumerate(metric_order[row_start:row_start+cols_per_row]):
                    cols[idx].markdown(
                        f"<div class='stat-card'><h5>{stat}</h5><h3>{qoc_allowed_stats[stat]}</h3></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    elif page == "Control and Whiffs" and player_type == "Pitcher":

        def calculate_pitching_effectiveness(df):
            stats = {}

            if len(df) == 0:
                stats.update({
                    "Strike%": 0,
                    "Zone%": 0,
                    "Z-Whiff%": 0,
                    "O-Whiff%": 0,
                    "Whiff%": 0,
                    "Chase%": 0,
                    "CSW%": 0,
                    "Pitches/PA": 0
                })
                return stats

            total_pitches = len(df)

            called_strikes = df.apply(lambda x: 1 if (x['Ball/Strike'] == "Strike" and x['Pitch Result'] == "No Swing") else 0, axis=1)
            swung_miss = df['Pitch Result'].apply(lambda x: 1 if x == "Miss" else 0)

            in_zone = (df['Strike Zone Side'] >= -8.9) & (df['Strike Zone Side'] <= 8.9) & \
                      (df['Strike Zone Height'] >= 17.5) & (df['Strike Zone Height'] <= 42.5)

            strikes = df['Ball/Strike'].apply(lambda x: 1 if x == "Strike" else 0).sum()
            zone_pitches = in_zone.sum()

            swings_in_zone = df[in_zone & df['Pitch Result'].isin(["Miss", "Contact"])].shape[0]
            swings_out_zone = df[~in_zone & df['Pitch Result'].isin(["Miss", "Contact"])].shape[0]

            z_whiff = df[in_zone & (df['Pitch Result'] == "Miss")].shape[0]
            o_whiff = df[~in_zone & (df['Pitch Result'] == "Miss")].shape[0]
            total_whiff = swung_miss.sum()

            chase = (o_whiff / swings_out_zone * 100) if swings_out_zone > 0 else 0

            csw = ((called_strikes.sum() + total_whiff) / total_pitches) * 100

            pa_counts = df.groupby(['Hitter', 'AB Result']).size().sum()
            pitches_per_pa = total_pitches / pa_counts if pa_counts > 0 else 0

            stats["Strike%"] = round((strikes / total_pitches) * 100, 1)
            stats["Zone%"] = round((zone_pitches / total_pitches) * 100, 1)
            stats["Z-Whiff%"] = round((z_whiff / swings_in_zone) * 100, 1) if swings_in_zone > 0 else 0
            stats["O-Whiff%"] = round((o_whiff / swings_out_zone) * 100, 1) if swings_out_zone > 0 else 0
            stats["Whiff%"] = round((total_whiff / (swings_in_zone + swings_out_zone)) * 100, 1) if (swings_in_zone + swings_out_zone) > 0 else 0
            stats["Chase%"] = round(chase, 1)
            stats["CSW%"] = round(csw, 1)
            stats["Pitches/PA"] = round(pitches_per_pa, 2)

            return stats
        
        pitching_stats = calculate_pitching_effectiveness(player_df)

        metric_order = ["Strike%","Zone%","Z-Whiff%","O-Whiff%","Whiff%","Chase%","CSW%","Pitches/PA"]
        cols_per_row = 4
        for row_start in range(0, len(metric_order), cols_per_row):
            with st.container():
                cols = st.columns(cols_per_row)
                for idx, stat in enumerate(metric_order[row_start:row_start+cols_per_row]):
                    cols[idx].markdown(
                        f"<div class='stat-card'><h5>{stat}</h5><h3>{pitching_stats[stat]}</h3></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
    elif page == "Velocity Distribution" and player_type == "Pitcher":

        df_filtered = df[df["Pitcher"] == selected_player].copy()

        if "All" not in handedness:
            df_filtered = df_filtered[df_filtered["B"].isin(handedness)]
        if "All" not in pitch_types:
            df_filtered = df_filtered[df_filtered["Pitch Type"].isin(pitch_types)]
        if "All" not in counts:
            df_filtered = df_filtered[df_filtered["Count"].isin(counts)]
        df_filtered = df_filtered[
            (df_filtered["Date"] >= pd.to_datetime(date_range[0])) &
            (df_filtered["Date"] <= pd.to_datetime(date_range[1]))
        ]
        if selected_zones and "All" not in selected_zones:
            df_filtered = df_filtered[df_filtered["PitchZone"].isin(selected_zones)]
            
        df_filtered = df_filtered[df_filtered["Velocity"].notna()]
        df_filtered = df_filtered[df_filtered["Velocity"] >= 50]  

        pitch_counts = df_filtered["Pitch Type"].value_counts()
        valid_pitches = pitch_counts[pitch_counts >= 2].index
        df_filtered = df_filtered[df_filtered["Pitch Type"].isin(valid_pitches)]

        dict_color = {
            "Fastball": "#ff0000",
            "Slider": "#6a0dad",
            "Changeup": "#ffff00",
            "2-Seam": "#87cefa",
            "Splitter": "#ffa500",
            "Cutter": "#00008b",
            "Curveball": "#008000",
        }

        def velocity_dist(df, custom_floor=None, custom_ceiling=None):
            sorted_pitches = df["Pitch Type"].value_counts().sort_values(ascending=False).index.tolist()
            n = len(sorted_pitches)

            fig, axes = plt.subplots(n, 1, figsize=(8, .95*n), sharex=True, facecolor='none')
            if n == 1:
                axes = [axes]

            for ax, pitch_type in zip(axes, sorted_pitches):
                pitch_data = df[df["Pitch Type"] == pitch_type]["Velocity"]

                if pitch_data.nunique() == 1:
                    velocity = pitch_data.values[0]
                    ax.plot([velocity, velocity], [0, 1], linewidth=4, color=dict_color.get(pitch_type, "black"))
                else:
                    sns.kdeplot(
                        pitch_data, ax=ax, fill=True, alpha=0.6,
                        clip=(pitch_data.min(), pitch_data.max()),
                        color=dict_color.get(pitch_type, "black")
                    )

                avg_velocity = pitch_data.mean()
                ax.axvline(avg_velocity, color="white", linestyle="--")  
                x_min = custom_floor if custom_floor else math.floor(df["Velocity"].min() / 5) * 5 - 5
                x_max = custom_ceiling if custom_ceiling else math.ceil(df["Velocity"].max() / 5) * 5
                ax.set_xlim(x_min, x_max)
                
                ax.set_yticks([])
                ax.set_ylabel("")
                ax.grid(axis="x", linestyle="--", alpha=0.5, color="white")
                ax.set_facecolor('none')

                ax.text(-0.01, 0.5, pitch_type, transform=ax.transAxes, fontsize=12, va="center", ha="right",
                        fontweight="bold", color="white")

                ax.tick_params(axis="x", colors="white")
                if ax != axes[-1]:
                    ax.tick_params(axis="x", colors="none")

            axes[-1].set_xlabel("Velocity (mph)", fontsize=12, fontweight="bold", color="white")
            plt.tight_layout()
            return fig

        if not df_filtered.empty:
             fig = velocity_dist(df_filtered, custom_floor=68, custom_ceiling=99)
             st.pyplot(fig)
        else:
            st.info("No pitch data available for this pitcher.")

    elif page == "Release Points" and player_type == "Pitcher":

        df_filtered = df[df["Pitcher"].str.strip() == selected_player.strip()].copy()

        if "All" not in handedness:
            df_filtered = df_filtered[df_filtered["B"].isin(handedness)]
        if "All" not in pitch_types:
            df_filtered = df_filtered[df_filtered["Pitch Type"].isin(pitch_types)]
        if "All" not in counts:
            df_filtered = df_filtered[df_filtered["Count"].isin(counts)]
        df_filtered = df_filtered[
            (df_filtered["Date"] >= pd.to_datetime(date_range[0])) &
            (df_filtered["Date"] <= pd.to_datetime(date_range[1]))
        ]
        if selected_zones and "All" not in selected_zones:
            df_filtered = df_filtered[df_filtered["PitchZone"].isin(selected_zones)]

        df_filtered = df_filtered[
            df_filtered["Release Height"].notna() &
            df_filtered["Release Side"].notna() &
            df_filtered["Pitch Type"].notna()
        ]

        if df_filtered.empty:
            st.info("No release point data available for this pitcher.")
        else:

            dict_color = {
                "Fastball": "#ff0000",
                "Slider": "#6a0dad",
                "Changeup": "#ffff00",
                "2-Seam": "#87cefa",
                "Splitter": "#ffa500",
                "Cutter": "#00008b",
                "Curveball": "#008000",
            }
            unique_pitches = df_filtered["Pitch Type"].unique()
            color_map = {pitch: dict_color.get(pitch, "black") for pitch in unique_pitches}

            fig, ax = plt.subplots(figsize=(4, 7))

            sns.set_style("whitegrid", {'axes.facecolor': 'none', 'grid.color': 'none'})  # optional
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')

            sns.scatterplot(
                data=df_filtered,
                x="Release Side",
                y="Release Height",
                hue="Pitch Type",
                palette=color_map,
                s=60,
                edgecolor="none",
                alpha=0.9,
                ax=ax
            )

            plate_y = 1.2
            half_width = 1.15
            inner_y_offset = 0.35
            point_depth = 0.8
            plate = Polygon(
                [
                    [-half_width, plate_y],
                    [ half_width, plate_y],
                    [ half_width, plate_y - inner_y_offset],
                    [0.00, plate_y - point_depth],
                    [-half_width, plate_y - inner_y_offset]
                ],
                closed=True,
                facecolor='none',
                edgecolor='white',
                linewidth=1.5,
                zorder=1
            )
            ax.add_patch(plate)

            ax.axvline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(0, 7.5)

            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("")
            ax.set_ylabel("")
            leg = ax.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

            for text in leg.get_texts():
                text.set_color("white")

            sns.despine(fig=fig, ax=ax)

            st.pyplot(fig)

    elif page == "Pitch Metrics" and player_type == "Pitcher":
        player_df = df[df["Pitcher"].str.strip() == selected_player.strip()].copy()

        if "All" not in handedness:
            player_df = player_df[player_df["B"].isin(handedness)]
        if "All" not in pitch_types:
            player_df = player_df[player_df["Pitch Type"].isin(pitch_types)]
        if "All" not in counts:
            player_df = player_df[player_df["Count"].isin(counts)]
        player_df = player_df[
            (player_df["Date"] >= pd.to_datetime(date_range[0])) &
            (player_df["Date"] <= pd.to_datetime(date_range[1]))
        ]
        if selected_zones and "All" not in selected_zones:
            player_df = player_df[player_df["PitchZone"].isin(selected_zones)]

        all_counts = player_df.groupby("Pitch Type").size().rename("Count")

        metrics_df = player_df[player_df["HB (trajectory)"].notna() & player_df["VB (spin)"].notna()]

        if metrics_df.empty:
            st.info("No pitch movement data available for this pitcher.")
        else:
            dict_color = {
                "Fastball": "#ff0000",
                "Slider": "#6a0dad",
                "Changeup": "#ffff00",
                "2-Seam": "#87cefa",
                "Splitter": "#ffa500",
                "Cutter": "#00008b",
                "Curveball": "#008000",
            }
            pitch_types_used = metrics_df["Pitch Type"].unique()
            color_map = {pitch: dict_color.get(pitch, "white") for pitch in pitch_types_used}

            fig, ax = plt.subplots(figsize=(5, 3.25), dpi=200)
            fig.patch.set_alpha(0)

            ax.add_patch(
                plt.Rectangle((-30, -30), 60, 60, facecolor=(0.5, 0.5, 0.5, 0.35), zorder=0)
            )

            sns.scatterplot(
                data=metrics_df,
                x="HB (trajectory)",
                y="VB (spin)",
                hue="Pitch Type",
                palette=color_map,
                s=40,
                edgecolor="none",
                alpha=0.7,
                legend=False,
                ax=ax
            )

            ax.axhline(0, color='white', linestyle='--', linewidth=1.2)
            ax.axvline(0, color='white', linestyle='--', linewidth=1.2)
            for tick in range(-20, 21, 10):
                if tick != 0:
                    ax.axhline(tick, color='#555555', linestyle='--', linewidth=0.75)
                    ax.axvline(tick, color='#555555', linestyle='--', linewidth=0.75)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.set_xticks(range(-20, 21, 10))
            ax.set_yticks(range(-20, 21, 10))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect('equal', adjustable='box')

            ax.tick_params(colors="white", which="both")
            for spine in ax.spines.values():
                spine.set_color("white")
            sns.despine(fig=fig, ax=ax)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, transparent=True)
            buf.seek(0)
            st.image(buf)

            groups = metrics_df.groupby("Pitch Type")

            def mean_if_exists(colname):
                return groups[colname].mean() if colname in metrics_df.columns else pd.Series(index=groups.size().index, dtype=float)

            ivb_col = "VB (spin)"

            summary = pd.concat([
                all_counts,
                mean_if_exists("Velocity").rename("Velocity"),
                mean_if_exists(ivb_col).rename("IVB"),
                mean_if_exists("HB (trajectory)").rename("HB"),
                mean_if_exists("Total Spin").rename("Spin"),
                mean_if_exists("Spin Efficiency (release)").rename("Spin Eff"),
                mean_if_exists("Release Extension (ft)").rename("Ext"),
                mean_if_exists("Vertical Approach Angle").rename("VAA"),
                mean_if_exists("Release Height").rename("RelH"),
                mean_if_exists("Release Side").rename("RelS"),
            ], axis=1).reset_index().rename(columns={"index": "Pitch Type"})

            summary = summary.sort_values(by="Count", ascending=False).reset_index(drop=True)

            float_cols = ["Velocity", "IVB", "HB", "Ext", "VAA", "RelH", "RelS", "Spin Eff"]
            for c in float_cols:
                if c in summary.columns:
                    summary[c] = summary[c].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            if "Spin" in summary.columns:
                summary["Spin"] = summary["Spin"].map(lambda x: f"{int(round(float(x)))}" if pd.notna(x) else "")

            summary["Pitch Type"] = summary["Pitch Type"].apply(
                lambda p: f"<span style='color:{color_map.get(p, 'white')}'><b>{p}</b></span>"
            )

            st.write(
                f"""
                <div style='display: flex; justify-content: center;'>
                    <div style='font-size:25px; background-color: rgba(128,128,128,0.3); padding: 15px; border-radius: 10px;'>
                        {summary.to_html(escape=False, index=False, table_id="centered_table")}
                        <style>
                            #centered_table {{
                                margin-left: auto;
                                margin-right: auto;
                                text-align: center;
                            }}
                            #centered_table th, #centered_table td {{
                                text-align: center;
                            }}
                        </style>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif page == "Pitch Locations" and player_type == "Pitcher":

        player_df = df[df["Pitcher"].str.strip() == selected_player.strip()].copy()

        if "All" not in handedness:
            player_df = player_df[player_df["B"].isin(handedness)]
        if "All" not in pitch_types:
            player_df = player_df[player_df["Pitch Type"].isin(pitch_types)]
        if "All" not in counts:
            player_df = player_df[player_df["Count"].isin(counts)]
        player_df = player_df[
            (player_df["Date"] >= pd.to_datetime(date_range[0])) &
            (player_df["Date"] <= pd.to_datetime(date_range[1]))
        ]
        if selected_zones and "All" not in selected_zones:
            player_df = player_df[player_df["PitchZone"].isin(selected_zones)]

        player_df = player_df[
            player_df["Strike Zone Side"].notna() &
            player_df["Strike Zone Height"].notna() &
            player_df["Pitch Type"].notna() &
            player_df["B"].isin(["R", "L"])
        ].copy()

        if player_df.empty:
            st.info("No pitch location data available for this pitcher.")
        else:
            pitch_types = player_df["Pitch Type"].unique()
            batter_sides = ["R", "L"]

            cols = st.columns(4)
            col_idx = 0

            for pitch in pitch_types:
                for side in batter_sides:
                    subset = player_df[
                        (player_df["Pitch Type"] == pitch) &
                        (player_df["B"] == side)
                    ]

                    if len(subset) < 5:
                        continue

                    fig, ax = plt.subplots(figsize=(3, 4))
                    sns.set(style="white")

                    sns.kdeplot(
                        data=subset,
                        x="Strike Zone Side",
                        y="Strike Zone Height",
                        fill=True,
                        thresh=0.05,
                        cmap="RdBu_r",
                        levels=100,
                        alpha=0.9,
                        ax=ax
                    )

                    from matplotlib.patches import Rectangle
                    strike_zone = Rectangle(
                        (-9.96, 18), 19.92, 24,
                        linewidth=1.5, edgecolor='black', facecolor='none'
                    )
                    ax.add_patch(strike_zone)

                    ax.set_xlim(-18, 18)
                    ax.set_ylim(9, 54)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal', adjustable='box')

                    side_label = "RHH" if side == "R" else "LHH"
                    ax.set_title(f"{pitch} vs {side_label}", fontsize=16, color='white')

                    ax.tick_params(colors='white', which='both')
                    for spine in ax.spines.values():
                        spine.set_color('white')

                    sns.despine(fig=fig, ax=ax)
                    plt.tight_layout()

                    with cols[col_idx]:
                        st.pyplot(fig, transparent=True)

                    col_idx = (col_idx + 1) % 4
