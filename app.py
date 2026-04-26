import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSpatial Site Readiness Analyzer",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #4a90d9;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }
    .unit-badge {
        background: #e8f0fe;
        color: #1967d2;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .ethics-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=300, seed=42):
    np.random.seed(seed)

    cities = {
        "Ahmedabad": (23.0225, 72.5714),
        "Surat":     (21.1702, 72.8311),
        "Vadodara":  (22.3072, 73.1812),
        "Rajkot":    (22.3039, 70.8022),
        "Gandhinagar": (23.2156, 72.6369),
    }

    rows = []
    for i in range(n):
        city = np.random.choice(list(cities.keys()))
        base_lat, base_lon = cities[city]
        use_case = np.random.choice(["EV Charging", "Warehouse"], p=[0.55, 0.45])

        lat = base_lat + np.random.uniform(-0.15, 0.15)
        lon = base_lon + np.random.uniform(-0.15, 0.15)

        pop_density        = np.random.uniform(1000, 18000)
        road_access_score  = np.random.uniform(1, 10)
        competitor_count   = np.random.randint(0, 15)
        power_grid_km      = np.random.uniform(0.1, 8.0)
        highway_dist_km    = np.random.uniform(0.5, 20.0)
        avg_daily_traffic  = np.random.uniform(500, 50000)
        land_use           = np.random.choice(["Commercial", "Industrial", "Mixed", "Residential"])

        # Weighted readiness score (0-100)
        score = (
            0.30 * (pop_density / 18000) * 100 +
            0.25 * road_access_score * 10 +
            0.20 * (avg_daily_traffic / 50000) * 100 -
            0.12 * (competitor_count / 15) * 100 -
            0.08 * (highway_dist_km / 20) * 100 -
            0.05 * (power_grid_km / 8) * 100
        )
        score = np.clip(score + np.random.normal(0, 4), 10, 95)

        rows.append({
            "site_id":             f"SITE-{i+1:03d}",
            "city":                city,
            "use_case":            use_case,
            "latitude":            round(lat, 5),
            "longitude":           round(lon, 5),
            "population_density":  round(pop_density, 1),
            "road_access_score":   round(road_access_score, 2),
            "competitor_count":    competitor_count,
            "power_grid_km":       round(power_grid_km, 2),
            "highway_dist_km":     round(highway_dist_km, 2),
            "avg_daily_traffic":   round(avg_daily_traffic, 0),
            "land_use":            land_use,
            "readiness_score":     round(score, 2),
        })

    return pd.DataFrame(rows)


df = generate_dataset()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.title("🗺️ Filters")
st.sidebar.markdown("---")
selected_cities    = st.sidebar.multiselect("City", df["city"].unique(), default=list(df["city"].unique()))
selected_use_cases = st.sidebar.multiselect("Use Case", df["use_case"].unique(), default=list(df["use_case"].unique()))
score_range        = st.sidebar.slider("Min Readiness Score", 0, 100, 0)

filtered = df[
    df["city"].isin(selected_cities) &
    df["use_case"].isin(selected_use_cases) &
    (df["readiness_score"] >= score_range)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered)} of {len(df)} sites")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 🗺️ GeoSpatial Site Readiness Analyzer")
st.markdown("**Location Intelligence for EV Charging Stations & Warehouses** · Gujarat, India")
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — KPI METRIC CARDS  (Unit I)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Overview &nbsp;<span class="unit-badge"> DS Pipeline</span></div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Sites", len(filtered))
with c2:
    st.metric("Avg Readiness Score", f"{filtered['readiness_score'].mean():.1f}")
with c3:
    st.metric("EV Charging Sites", len(filtered[filtered["use_case"] == "EV Charging"]))
with c4:
    st.metric("Warehouse Sites", len(filtered[filtered["use_case"] == "Warehouse"]))
with c5:
    top = filtered.nlargest(1, "readiness_score")
    st.metric("Top Site Score", f"{top['readiness_score'].values[0]:.1f}" if len(top) else "—")

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 2 — EDA  (Unit I + Unit II Descriptive)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Exploratory Data Analysis &nbsp;<span class="unit-badge"> Descriptive Statistics</span></div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = {"EV Charging": "#4a90d9", "Warehouse": "#e67e22"}
    for uc, grp in filtered.groupby("use_case"):
        ax.hist(grp["readiness_score"], bins=20, alpha=0.65,
                label=uc, color=colors.get(uc, "gray"), edgecolor="white")
    ax.set_xlabel("Readiness Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Use Case")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_b:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    data_to_plot = [filtered[filtered["use_case"]==uc]["readiness_score"].values
                    for uc in ["EV Charging","Warehouse"]]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#4a90d9")
    if len(bp["boxes"]) > 1:
        bp["boxes"][1].set_facecolor("#e67e22")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["EV Charging", "Warehouse"])
    ax.set_ylabel("Readiness Score")
    ax.set_title("Score Spread: EV vs Warehouse")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_c:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    city_avg = filtered.groupby("city")["readiness_score"].mean().sort_values(ascending=True)
    bars = ax.barh(city_avg.index, city_avg.values, color="#4a90d9", edgecolor="white")
    for bar, val in zip(bars, city_avg.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Avg Readiness Score")
    ax.set_title("Avg Score by City")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Descriptive stats table + correlation heatmap
col_d, col_e = st.columns([1, 1])

with col_d:
    numeric_cols = ["population_density","road_access_score","competitor_count",
                    "highway_dist_km","avg_daily_traffic","readiness_score"]
    desc = filtered[numeric_cols].describe().T[["mean","std","min","max"]]
    desc["skew"] = filtered[numeric_cols].skew()
    desc["kurtosis"] = filtered[numeric_cols].kurtosis()
    desc = desc.round(2)
    st.markdown("**Descriptive Statistics (mean, std, skew, kurtosis)**")
    st.dataframe(desc, use_container_width=True)

with col_e:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    corr = filtered[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — INFERENTIAL STATISTICS (Unit II)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Statistical Analysis &nbsp;<span class="unit-badge"> Inferential Statistics</span></div>', unsafe_allow_html=True)

col_f, col_g, col_h = st.columns(3)

ev_scores  = filtered[filtered["use_case"] == "EV Charging"]["readiness_score"].dropna()
wh_scores  = filtered[filtered["use_case"] == "Warehouse"]["readiness_score"].dropna()

with col_f:
    if len(ev_scores) > 1 and len(wh_scores) > 1:
        t_stat, p_val = stats.ttest_ind(ev_scores, wh_scores)
        ci_ev = stats.t.interval(0.95, len(ev_scores)-1,
                                  loc=ev_scores.mean(), scale=stats.sem(ev_scores))
        ci_wh = stats.t.interval(0.95, len(wh_scores)-1,
                                  loc=wh_scores.mean(), scale=stats.sem(wh_scores))
        conclusion = "Reject H₀ — significant difference" if p_val < 0.05 else "Fail to reject H₀ — no significant difference"
        color = "#d32f2f" if p_val < 0.05 else "#388e3c"

        st.markdown("**Two-Sample T-Test: EV vs Warehouse**")
        st.markdown(f"- T-statistic: **{t_stat:.4f}**")
        st.markdown(f"- P-value: **{p_val:.4f}**")
        st.markdown(f"- EV 95% CI: **[{ci_ev[0]:.2f}, {ci_ev[1]:.2f}]**")
        st.markdown(f"- WH 95% CI: **[{ci_wh[0]:.2f}, {ci_wh[1]:.2f}]**")
        st.markdown(f"<span style='color:{color};font-weight:600'>→ {conclusion}</span>", unsafe_allow_html=True)

with col_g:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    means = [ev_scores.mean(), wh_scores.mean()]
    errors = [ev_scores.sem() * 1.96, wh_scores.sem() * 1.96]
    bars = ax.bar(["EV Charging", "Warehouse"], means, yerr=errors,
                  color=["#4a90d9", "#e67e22"], capsize=6,
                  error_kw=dict(elinewidth=1.5, ecolor="black"), edgecolor="white")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 1.2,
                f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean Readiness Score")
    ax.set_title("Mean Score with 95% CI")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_h:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.linspace(20, 90, 300)
    for scores, label, color in [(ev_scores, "EV Charging", "#4a90d9"),
                                  (wh_scores, "Warehouse", "#e67e22")]:
        mu, sigma = scores.mean(), scores.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color=color, label=f"{label} (μ={mu:.1f})", linewidth=2)
        ax.axvline(mu, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Readiness Score")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution (CLT)")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — MACHINE LEARNING (Unit III)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Machine Learning Models &nbsp;<span class="unit-badge">  Supervised + Unsupervised Learning</span></div>', unsafe_allow_html=True)

features = ["population_density","road_access_score","competitor_count",
            "highway_dist_km","avg_daily_traffic","power_grid_km"]
target   = "readiness_score"

ml_df = filtered[features + [target]].dropna()

col_i, col_j, col_k = st.columns(3)

# Linear Regression
with col_i:
    if len(ml_df) > 20:
        X = ml_df[features]
        y = ml_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_test, y_pred, alpha=0.6, color="#4a90d9", edgecolors="white", s=40)
        lims = [min(y_test.min(), y_pred.min())-2, max(y_test.max(), y_pred.max())+2]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(f"Linear Regression\nR²={r2:.3f}  MAE={mae:.2f}")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

with col_j:
    if len(ml_df) > 20:
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_}).sort_values("Coefficient")
        colors_coef = ["#d32f2f" if c < 0 else "#388e3c" for c in coef_df["Coefficient"]]
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Feature Importance\n(Regression Coefficients)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# KMeans clustering
with col_k:
    if len(ml_df) > 10:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(ml_df[features])

        # Elbow method
        inertias = []
        k_range = range(2, 8)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(list(k_range), inertias, "o-", color="#7b1fa2", linewidth=2, markersize=6)
        ax.axvline(3, color="#e67e22", linestyle="--", linewidth=1.5, label="k=3 chosen")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("K-Means Elbow Curve")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# Cluster scatter plot
if len(ml_df) > 10:
    km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    ml_df = ml_df.copy()
    ml_df["cluster"] = km_final.fit_predict(X_scaled)
    cluster_labels = {0: "High Potential", 1: "Medium Potential", 2: "Low Potential"}
    cluster_mean = ml_df.groupby("cluster")[target].mean()
    sorted_clusters = cluster_mean.sort_values(ascending=False)
    label_map = {sorted_clusters.index[0]: "High Potential",
                 sorted_clusters.index[1]: "Medium Potential",
                 sorted_clusters.index[2]: "Low Potential"}
    ml_df["zone"] = ml_df["cluster"].map(label_map)

    col_l, col_m = st.columns(2)
    with col_l:
        zone_colors = {"High Potential": "#388e3c", "Medium Potential": "#f9a825", "Low Potential": "#d32f2f"}
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for zone, grp in ml_df.groupby("zone"):
            ax.scatter(grp["population_density"], grp["readiness_score"],
                       label=zone, alpha=0.6, s=35,
                       color=zone_colors.get(zone, "gray"), edgecolors="white")
        ax.set_xlabel("Population Density")
        ax.set_ylabel("Readiness Score")
        ax.set_title("K-Means Clusters: Population vs Score")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_m:
        zone_count = ml_df["zone"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        wedge_colors = [zone_colors[z] for z in zone_count.index]
        wedges, texts, autotexts = ax.pie(
            zone_count.values, labels=zone_count.index,
            autopct="%1.1f%%", colors=wedge_colors,
            startangle=90, pctdistance=0.75,
            textprops={"fontsize": 9}
        )
        ax.set_title("Site Zone Distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 5 — GEOSPATIAL MAP (Unit IV)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Geospatial Site Map &nbsp;<span class="unit-badge"> Geospatial Visualization</span></div>', unsafe_allow_html=True)

try:
    import folium
    from streamlit_folium import st_folium

    map_df = filtered.copy()
    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                   tiles="CartoDB positron")

    use_case_colors = {"EV Charging": "blue", "Warehouse": "orange"}

    for _, row in map_df.iterrows():
        color = use_case_colors.get(row["use_case"], "gray")
        radius = 4 + (row["readiness_score"] / 20)
        popup_html = f"""
        <b>{row['site_id']}</b><br>
        Use Case: {row['use_case']}<br>
        City: {row['city']}<br>
        Score: <b>{row['readiness_score']:.1f}</b><br>
        Land Use: {row['land_use']}
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['site_id']} | Score: {row['readiness_score']:.1f}"
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;line-height:1.8">
        <b>Legend</b><br>
        <span style="color:#4a90d9">●</span> EV Charging<br>
        <span style="color:#e67e22">●</span> Warehouse<br>
        <i>Circle size = readiness score</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, use_container_width=True, height=480)

except ImportError:
    st.info("Install `folium` and `streamlit-folium` to view the interactive map. Showing coordinate preview instead.")
    map_preview = filtered[["site_id","city","use_case","latitude","longitude","readiness_score"]].head(20)
    st.dataframe(map_preview, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 6 — TOP SITES RANKED TABLE (Unit IV)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Top Recommended Sites &nbsp;<span class="unit-badge"> Visualization + Presentation</span></div>', unsafe_allow_html=True)

col_n, col_o = st.columns([1.4, 1])

with col_n:
    top_sites = (filtered
                 .nlargest(15, "readiness_score")
                 [["site_id","city","use_case","land_use",
                   "readiness_score","population_density","road_access_score"]]
                 .reset_index(drop=True))
    top_sites.index += 1
    st.markdown("**Top 15 Sites by Readiness Score**")
    st.dataframe(
        top_sites.style.background_gradient(subset=["readiness_score"], cmap="YlGn"),
        use_container_width=True, height=380
    )

with col_o:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    land_use_avg = filtered.groupby(["land_use","use_case"])["readiness_score"].mean().unstack()
    land_use_avg.plot(kind="bar", ax=ax, color=["#4a90d9","#e67e22"],
                      edgecolor="white", width=0.6)
    ax.set_xlabel("Land Use Type")
    ax.set_ylabel("Avg Readiness Score")
    ax.set_title("Avg Score by Land Use & Use Case")
    ax.legend(title="Use Case", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 7 — ETHICS (Unit IV)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Ethics & Fairness &nbsp;<span class="unit-badge"> Data Science Ethics</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="ethics-box">
<b>1. Data Bias</b> — Population density and traffic data may over-represent urban areas, systematically disadvantaging rural sites even when they are viable for warehouse use.<br><br>
<b>2. Fairness in Scoring</b> — The readiness score weights commercial viability over community need. A high-density low-income area may score high for EV stations without accounting for affordability access.<br><br>
<b>3. Location Data Privacy</b> — Candidate site coordinates combined with land-use labels could indirectly expose private property or business intelligence. Data should be aggregated at zone level before public sharing.<br><br>
<b>4. Algorithmic Accountability (The Five Cs)</b> — Consent, Clarity, Consistency, Control, and Consequences must guide how this model is used in real infrastructure decisions. No automated decision should be made without human review.<br><br>
<b>5. Diversity & Inclusion</b> — Infrastructure placement affects communities differently. EV charging access gaps in low-income or minority neighborhoods must be explicitly checked in the model output.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("GeoSpatial Site Readiness Analyzer · Mini Project · Data Science (CE0630) · Indus University · Semester VI")
