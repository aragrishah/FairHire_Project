import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px
import io
import re

# ─────────────────────────────────────────────────────────
# 1. PAGE CONFIG + GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="FairHire AI", layout="wide")

st.markdown("""
<style>
/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 1rem;
    margin-bottom: 1.6rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-logo {
    font-size: 1.15rem;
    font-weight: 900;
    color: #f1f5f9;
    padding: 12px 16px 12px 0;
    border-right: 1px solid #1e293b;
    margin-right: 6px;
    white-space: nowrap;
}
.nav-btn {
    background: none;
    border: none;
    color: #94a3b8;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 12px 14px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
}
.nav-btn:hover  { color: #f1f5f9; border-bottom-color: #3b82f6; }
.nav-btn.active { color: #3b82f6; border-bottom-color: #3b82f6; font-weight: 700; }

/* ── Section headers ── */
.section-header {
    font-size: 1.5rem; font-weight: 700;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #3b82f6;
    display: inline-block;
    margin-bottom: 0.2rem;
}

/* ── KPI metric cards ── */
[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #2d2d3f;
    border-radius: 10px; padding: 16px 20px;
}
[data-testid="metric-container"] label {
    font-size: 0.75rem !important; color: #9ca3af !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.9rem !important; font-weight: 800 !important; color: #f1f5f9 !important;
}

/* ── Resume upload card ── */
.resume-card {
    background: #1e293b; border: 1px solid #2d2d3f;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 14px;
}
.field-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 7px 0; border-bottom: 1px solid #2d2d3f; font-size: 0.87rem;
}
.field-label { color: #94a3b8; min-width: 160px; }
.field-value { color: #f1f5f9; font-weight: 600; text-align: right; }

hr { margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# 2. LOAD ASSETS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = pickle.load(open('outputs/hiring_model.pkl', 'rb'))
    encoders = pickle.load(open('outputs/encoders.pkl', 'rb'))
    df_raw   = pd.read_csv('synthetic_hiring_data.csv')
    df_raw   = df_raw.drop(columns=['candidate_id', 'age_group'], errors='ignore').fillna(0)

    df_encoded = df_raw.copy()
    for col, le in encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = le.transform(df_encoded[col].astype(str))

    feature_cols = [c for c in df_encoded.columns if c != 'hired']
    probs        = model.predict_proba(df_encoded[feature_cols])[:, 1]

    df_display = df_raw.copy()
    df_display['Hiring Probability'] = probs
    df_display['Decision'] = ["✅ SHORTLIST" if p > 0.5 else "❌ REJECT" for p in probs]

    return model, encoders, df_display, df_encoded, feature_cols

model, encoders, df_display, df_encoded, feature_cols = load_assets()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer      = get_explainer(model)
BIAS_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────
# 3. HEADER + NAVBAR
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1rem 0 0.2rem 0;'>
    <span style='font-size:2.2rem; font-weight:900; letter-spacing:-1px;'>FairHire AI</span><br>
    <span style='color:#64748b; font-size:0.9rem;'>XAI-Driven Hiring Dashboard — Transparent, Fair, Explainable Recruitment</span>
</div>
""", unsafe_allow_html=True)

# Nav pages
NAV_PAGES = [
    "Candidate Pool",
    "Pool Insights",
    "Leaderboard",
    "Fairness & Bias",
    "AI Explanation",
    "Candidate Comparison",
    "Resume Screening",
    "HR Report",
]

if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "Candidate Pool"

# Render navbar as Streamlit buttons in one row
nav_cols = st.columns(len(NAV_PAGES))
for i, page in enumerate(NAV_PAGES):
    active = st.session_state['nav_page'] == page
    btn_style = "primary" if active else "secondary"
    if nav_cols[i].button(page, key=f"nav_{page}", type=btn_style, use_container_width=True):
        st.session_state['nav_page'] = page
        st.rerun()

st.divider()
active_page = st.session_state['nav_page']


# ─────────────────────────────────────────────────────────
# SHARED: filter state
# ─────────────────────────────────────────────────────────
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = df_display.copy()

filtered_df = st.session_state['filtered_df']

display_cols = [
    'job_role_applied', 'gender', 'nationality', 'education_level',
    'university_tier', 'relevant_experience_years',
    'skill_match_score', 'interview_score', 'Hiring Probability', 'Decision'
]


# ═════════════════════════════════════════════════════════
# PAGE: CANDIDATE POOL
# ═════════════════════════════════════════════════════════
if active_page == "Candidate Pool":

    st.markdown('<div class="section-header">Candidate Filters</div>', unsafe_allow_html=True)
    st.caption("Use the filters below to narrow the candidate pool by role, demographics, and minimum score thresholds. "
               "All other sections of the dashboard update based on this filtered view.")
    st.markdown("")

    with st.form("filter_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: role_filter   = st.selectbox("Job Role",        ["All"] + sorted(df_display['job_role_applied'].unique().tolist()))
        with c2: gender_filter = st.selectbox("Gender",          ["All"] + sorted(df_display['gender'].unique().tolist()))
        with c3: nat_filter    = st.selectbox("Nationality",     ["All"] + sorted(df_display['nationality'].unique().tolist()))
        with c4: tier_filter   = st.selectbox("University Tier", ["All"] + sorted(df_display['university_tier'].unique().tolist()))

        c5, c6, c7 = st.columns(3)
        with c5: min_exp   = st.number_input("Min Relevant Experience (yrs)", 0.0, 40.0, 0.0, step=0.5)
        with c6: min_skill = st.slider("Min Skill Match Score", 0.0, 1.0, 0.0, step=0.05)
        with c7: min_int   = st.slider("Min Interview Score",   0.0, 100.0, 0.0, step=1.0)

        apply_btn = st.form_submit_button("Apply Filters", use_container_width=True)

    if apply_btn:
        fd = df_display.copy()
        if role_filter   != "All": fd = fd[fd['job_role_applied'] == role_filter]
        if gender_filter != "All": fd = fd[fd['gender']           == gender_filter]
        if nat_filter    != "All": fd = fd[fd['nationality']       == nat_filter]
        if tier_filter   != "All": fd = fd[fd['university_tier']   == tier_filter]
        fd = fd[(fd['relevant_experience_years'] >= min_exp) &
                (fd['skill_match_score']         >= min_skill) &
                (fd['interview_score']           >= min_int)]
        st.session_state['filtered_df'] = fd
        filtered_df = fd

    total_shown = len(filtered_df)
    st.info(f"Showing **{total_shown:,}** candidates" +
            (f" — filtered from {len(df_display):,} total" if total_shown < len(df_display) else " (all candidates)"))

    st.dataframe(
        filtered_df[display_cols].style
            .background_gradient(subset=['Hiring Probability'], cmap="RdYlGn")
            .format({"Hiring Probability": "{:.1%}", "skill_match_score": "{:.3f}",
                     "interview_score": "{:.1f}", "relevant_experience_years": "{:.1f}"}),
        use_container_width=True, height=500
    )


# ═════════════════════════════════════════════════════════
# PAGE: POOL INSIGHTS
# ═════════════════════════════════════════════════════════
elif active_page == "Pool Insights":

    st.markdown('<div class="section-header">Pool Insights</div>', unsafe_allow_html=True)
    st.caption("A high-level overview of the current filtered candidate pool — shortlist counts, average scores, "
               "and how hiring probability is distributed across candidates.")
    st.markdown("")

    if not filtered_df.empty:
        shortlisted   = (filtered_df['Hiring Probability'] > 0.5).sum()
        total         = len(filtered_df)
        rejected      = total - shortlisted

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Candidates",     f"{total:,}")
        k2.metric("Shortlisted",          f"{shortlisted:,}",        f"{shortlisted/total:.1%} rate")
        k3.metric("Rejected",             f"{rejected:,}",           f"{rejected/total:.1%} rate")
        k4.metric("Avg Hire Probability", f"{filtered_df['Hiring Probability'].mean():.1%}")
        k5.metric("Avg Skill Score",      f"{filtered_df['skill_match_score'].mean():.3f}")
        k6.metric("Avg Interview Score",  f"{filtered_df['interview_score'].mean():.1f}")

        st.markdown("""
        <style>
        div[data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] { color: #22c55e !important; }
        div[data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] { color: #ef4444 !important; }
        </style>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns([3, 1])

        with ch1:
            st.markdown("**Hiring Probability Distribution**")
            st.caption("Shows how the AI's confidence is spread across all candidates. "
                       "Candidates to the right of the red line (above 0.5) are shortlisted; those to the left are rejected.")
            fig_hist = px.histogram(filtered_df, x='Hiring Probability', nbins=30,
                                    color_discrete_sequence=['#3b82f6'],
                                    labels={'Hiring Probability': 'Probability'})
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#ef4444",
                               annotation_text="Threshold (0.5)", annotation_position="top right")
            fig_hist.update_layout(height=340, margin=dict(t=20, b=20), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch2:
            st.markdown("**Decision Split**")
            st.caption("Overall shortlist vs reject ratio for the current filtered pool.")
            fig_donut = go.Figure(go.Pie(
                labels=['Shortlisted', 'Rejected'], values=[shortlisted, rejected],
                hole=0.6, marker_colors=['#22c55e', '#ef4444'],
                textinfo='percent', showlegend=True
            ))
            fig_donut.update_layout(height=340, margin=dict(t=20, b=20, l=0, r=0),
                                    legend=dict(orientation='h', yanchor='bottom', y=-0.2))
            st.plotly_chart(fig_donut, use_container_width=True)

    else:
        st.warning("No candidates match the current filters. Go to Candidate Pool to adjust filters.")


# ═════════════════════════════════════════════════════════
# PAGE: LEADERBOARD
# ═════════════════════════════════════════════════════════
elif active_page == "Leaderboard":

    st.markdown('<div class="section-header">Top Candidates Leaderboard</div>', unsafe_allow_html=True)
    st.caption("Ranks the highest-probability candidates in the current filtered pool. "
               "Use this to quickly identify your best shortlist picks.")
    st.markdown("")

    if not filtered_df.empty:
        top_n  = st.slider("Show Top N Candidates", 5, 20, 10, step=5)
        top_df = (filtered_df[filtered_df['Hiring Probability'] > 0.5]
                  .sort_values('Hiring Probability', ascending=False)
                  .head(top_n).reset_index(drop=False))

        if top_df.empty:
            st.info("No shortlisted candidates in current filter. Go to Candidate Pool to adjust filters.")
        else:
            medals = ['🥇', '🥈', '🥉'] + ['🏅'] * 50
            for rank, row in top_df.iterrows():
                prob  = row['Hiring Probability']
                color = '#22c55e' if prob > 0.7 else '#f59e0b' if prob > 0.5 else '#ef4444'
                st.markdown(f"""
                <div style='display:flex; align-items:center; background:#1e293b;
                            border-radius:10px; padding:12px 18px; margin-bottom:8px;
                            border-left:4px solid {color};'>
                    <span style='font-size:1.3rem; width:36px;'>{medals[rank]}</span>
                    <div style='flex:1; margin-left:12px;'>
                        <b style='font-size:1rem;'>Candidate #{int(row['index'])}</b>
                        <span style='color:#94a3b8; margin-left:10px; font-size:0.85rem;'>
                            {row['job_role_applied']} · {row['gender']} · {row['nationality']}
                        </span><br>
                        <span style='font-size:0.81rem; color:#d1d5db;'>
                            {row['education_level']} ({row['university_tier']}) &nbsp;|&nbsp;
                            {row['relevant_experience_years']:.1f} yrs exp &nbsp;|&nbsp;
                            Skill: {row['skill_match_score']:.3f} &nbsp;|&nbsp;
                            Interview: {row['interview_score']:.1f}
                        </span>
                        <div style='background:#374151; border-radius:4px; height:5px; margin-top:7px;'>
                            <div style='background:{color}; width:{int(prob*100)}%; height:5px; border-radius:4px;'></div>
                        </div>
                    </div>
                    <div style='text-align:right; margin-left:16px; min-width:64px;'>
                        <b style='font-size:1.25rem; color:{color};'>{prob:.1%}</b><br>
                        <span style='font-size:0.72rem; color:#64748b;'>hire prob</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No candidates to show. Go to Candidate Pool to adjust filters.")


# ═════════════════════════════════════════════════════════
# PAGE: FAIRNESS & BIAS
# ═════════════════════════════════════════════════════════
elif active_page == "Fairness & Bias":

    st.markdown('<div class="section-header">Fairness & Bias Detection</div>', unsafe_allow_html=True)
    st.caption("Compares the AI shortlisting rate across demographic groups to detect unequal treatment. "
               "A gap above 15% triggers a bias alert based on the Demographic Parity fairness metric.")
    st.markdown("")

    if not filtered_df.empty:
        temp = filtered_df.copy()
        temp['Selected'] = temp['Hiring Probability'] > 0.5

        tab_g, tab_n, tab_e = st.tabs(["By Gender", "By Nationality", "By Education"])

        def render_fairness(df_in, group_col, tab):
            rates = (df_in.groupby(group_col)['Selected'].mean()
                     .reset_index().rename(columns={'Selected': 'Selection Rate'})
                     .sort_values('Selection Rate', ascending=False))
            with tab:
                if len(rates) < 2:
                    st.info(f"Only one {group_col.replace('_',' ')} group in current filter — "
                            "broaden your filters to compare across groups.")
                    return
                max_r = rates['Selection Rate'].max()
                min_r = rates['Selection Rate'].min()
                gap   = max_r - min_r

                mcols = st.columns(len(rates) + 1)
                for i, (_, row) in enumerate(rates.iterrows()):
                    mcols[i].metric(str(row[group_col]), f"{row['Selection Rate']:.1%}")
                mcols[-1].metric("Max Gap", f"{gap:.1%}")

                st.markdown("<br>", unsafe_allow_html=True)
                if gap > BIAS_THRESHOLD:
                    st.error(f"Bias Detected: Selection rate gap of {gap:.1%} exceeds the {BIAS_THRESHOLD:.0%} threshold.")
                else:
                    st.success(f"No Significant Bias: Gap of {gap:.1%} is within the acceptable {BIAS_THRESHOLD:.0%} limit.")

                bar_colors = ['#22c55e' if r == max_r else '#ef4444' if r == min_r else '#60a5fa'
                              for r in rates['Selection Rate']]
                fig = go.Figure(go.Bar(
                    x=rates[group_col], y=rates['Selection Rate'],
                    marker_color=bar_colors,
                    text=[f"{r:.1%}" for r in rates['Selection Rate']],
                    textposition='outside', width=0.5
                ))
                fig.add_hline(y=rates['Selection Rate'].mean(), line_dash="dot",
                              line_color="#fbbf24", annotation_text="Pool avg", annotation_position="right")
                fig.update_layout(
                    yaxis=dict(title="Selection Rate", range=[0, min(1.0, max_r*1.35)], tickformat='.0%'),
                    height=380, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Sensitive attributes are excluded from model predictions and used here only for compliance auditing.")

        render_fairness(temp, 'gender',          tab_g)
        render_fairness(temp, 'nationality',     tab_n)
        render_fairness(temp, 'education_level', tab_e)
    else:
        st.warning("No data available. Go to Candidate Pool to adjust filters.")


# ═════════════════════════════════════════════════════════
# PAGE: AI EXPLANATION
# ═════════════════════════════════════════════════════════
elif active_page == "AI Explanation":

    st.markdown('<div class="section-header">AI Explanation (SHAP)</div>', unsafe_allow_html=True)
    st.caption("Select any candidate to see exactly why the AI made its decision. "
               "Green bars are features that increased hire probability; red bars decreased it.")
    st.markdown("")

    if not filtered_df.empty:
        selected_idx = st.selectbox(
            "Select Candidate",
            options=filtered_df.index.tolist(),
            format_func=lambda x: (
                f"Candidate #{x}  |  {filtered_df.loc[x,'job_role_applied']}  |  "
                f"P(Hired) = {filtered_df.loc[x,'Hiring Probability']:.1%}  |  {filtered_df.loc[x,'Decision']}"
            )
        )

        row_encoded = df_encoded.loc[[selected_idx], feature_cols]
        sv = explainer.shap_values(row_encoded)
        if isinstance(sv, list):    shap_vals = np.array(sv[1]).flatten()
        elif sv.ndim == 3:          shap_vals = sv[0, :, 1]
        else:                       shap_vals = np.array(sv).flatten()

        shap_df = pd.DataFrame({"Feature": feature_cols, "Impact": shap_vals}).sort_values("Impact")

        col_shap, col_info = st.columns([3, 1])

        with col_shap:
            colors = ['#22c55e' if x > 0 else '#ef4444' for x in shap_df["Impact"]]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["Impact"], y=shap_df["Feature"], orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["Impact"]], textposition='outside'
            ))
            fig_shap.add_vline(x=0, line_color="#475569", line_dash="dot")
            fig_shap.update_layout(
                title="Feature Contributions to Hiring Decision",
                xaxis_title="SHAP Value  (green = increases hire chance  |  red = decreases hire chance)",
                height=530, plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=30, l=160, r=80)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_info:
            cand  = filtered_df.loc[selected_idx]
            prob  = cand['Hiring Probability']
            color = "#22c55e" if prob > 0.5 else "#ef4444"

            st.markdown(f"""
            <div style='text-align:center; padding:16px; border-radius:12px;
                        background:{color}18; border:2px solid {color}; margin-bottom:16px;'>
                <div style='font-size:2.4rem; font-weight:900; color:{color};'>{prob:.1%}</div>
                <div style='font-size:0.85rem; color:{color}; font-weight:600;'>
                    {"LIKELY HIRED" if prob > 0.5 else "UNLIKELY HIRED"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            details = {"Role": cand['job_role_applied'], "Gender": cand['gender'],
                       "Nationality": cand['nationality'], "Education": cand['education_level'],
                       "University": cand['university_tier'],
                       "Experience": f"{cand['relevant_experience_years']:.1f} yrs",
                       "Skill Score": f"{cand['skill_match_score']:.3f}",
                       "Interview": f"{cand['interview_score']:.1f}",
                       "Certifications": int(cand['num_certifications'])}
            for k, v in details.items():
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between;
                            padding:5px 0; border-bottom:1px solid #2d2d3f; font-size:0.84rem;'>
                    <span style='color:#9ca3af;'>{k}</span>
                    <span style='color:#f1f5f9; font-weight:600;'>{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            top_pos = shap_df[shap_df['Impact'] > 0].tail(3)['Feature'].tolist()[::-1]
            top_neg = shap_df[shap_df['Impact'] < 0].head(3)['Feature'].tolist()
            if top_pos:
                st.markdown("**Top Positive Factors**")
                for f in top_pos:
                    st.markdown(f"<div style='padding:3px 8px; margin:3px 0; background:#22c55e18; "
                                f"border-left:3px solid #22c55e; border-radius:4px; font-size:0.82rem;'>{f}</div>",
                                unsafe_allow_html=True)
            if top_neg:
                st.markdown("**Top Negative Factors**")
                for f in top_neg:
                    st.markdown(f"<div style='padding:3px 8px; margin:3px 0; background:#ef444418; "
                                f"border-left:3px solid #ef4444; border-radius:4px; font-size:0.82rem;'>{f}</div>",
                                unsafe_allow_html=True)
    else:
        st.warning("No candidates available. Go to Candidate Pool to adjust filters.")


# ═════════════════════════════════════════════════════════
# PAGE: CANDIDATE COMPARISON
# ═════════════════════════════════════════════════════════
elif active_page == "Candidate Comparison":

    st.markdown('<div class="section-header">Candidate Comparison</div>', unsafe_allow_html=True)
    st.caption("Select 2 to 5 candidates to compare their key scores side by side. "
               "Useful for making the final shortlist decision between close candidates.")
    st.markdown("")

    if not filtered_df.empty:
        selected_candidates = st.multiselect(
            "Select 2–5 Candidates to Compare",
            options=filtered_df.index.tolist(), max_selections=5,
            format_func=lambda x: f"#{x} — {filtered_df.loc[x,'job_role_applied']} ({filtered_df.loc[x,'Hiring Probability']:.1%})"
        )
        if len(selected_candidates) >= 2:
            compare_df = filtered_df.loc[selected_candidates]
            features_map = {
                'relevant_experience_years': ('Experience (yrs)', 1),
                'skill_match_score':         ('Skill Score',      100),
                'interview_score':           ('Interview Score',  1),
                'num_certifications':        ('Certifications',   1),
                'Hiring Probability':        ('Hire Probability %', 100),
            }
            fig_cmp = go.Figure()
            for i, idx in enumerate(selected_candidates):
                row    = compare_df.loc[idx]
                labels = [v[0] for v in features_map.values()]
                values = [row[col] * scale for col, (_, scale) in features_map.items()]
                fig_cmp.add_trace(go.Bar(
                    name=f"Candidate #{idx}", x=labels, y=values,
                    marker_color=px.colors.qualitative.Set2[i % 8],
                    text=[f"{v:.1f}" for v in values], textposition='outside'
                ))
            fig_cmp.update_layout(
                barmode='group', height=460,
                plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Score / Value",
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            best_idx = compare_df['Hiring Probability'].idxmax()
            st.success(f"Candidate #{best_idx} has the highest hiring probability "
                       f"({compare_df['Hiring Probability'].max():.1%}) among selected candidates.")
        else:
            st.info("Select at least 2 candidates to compare.")
    else:
        st.warning("No candidates available. Go to Candidate Pool to adjust filters.")


# ═════════════════════════════════════════════════════════
# PAGE: RESUME SCREENING  (BERT-based PDF extraction)
# ═════════════════════════════════════════════════════════
elif active_page == "Resume Screening":

    st.markdown('<div class="section-header">Resume Screening</div>', unsafe_allow_html=True)
    st.caption("Upload a candidate's resume PDF. BERT extracts key fields (skills, experience, education), "
               "maps them to model features, and predicts the candidate's hiring probability with a SHAP explanation.")
    st.markdown("")

    # ── Lazy imports so app loads fast even if not on this page ──
    try:
        import pdfplumber
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        BERT_AVAILABLE = True
    except ImportError:
        BERT_AVAILABLE = False

    if not BERT_AVAILABLE:
        st.warning("""**Missing dependencies for Resume Screening.** Install them with:
```
pip install pdfplumber transformers torch
```
Then restart the app.""")
    else:
        # ── Load BERT NER pipeline (cached) ──
        @st.cache_resource(show_spinner="Loading BERT model...")
        def load_bert():
            # Use a lightweight BERT NER model for entity extraction
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            return pipeline("ner", model=ner_model, tokenizer=tokenizer,
                            aggregation_strategy="simple")
        ner_pipe = load_bert()

        # ── Helper: extract text from PDF ──
        def extract_pdf_text(uploaded_file) -> str:
            text_chunks = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_chunks.append(t)
            return "\n".join(text_chunks)

        # ── Helper: rule-based field extraction from raw text ──
        def extract_fields(text: str) -> dict:
            text_lower = text.lower()

            # Experience years — look for patterns like "5 years", "3+ years", "8 yrs"
            exp_matches = re.findall(r'(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', text_lower)
            years_exp   = float(exp_matches[0]) if exp_matches else 0.0

            # Education level
            if any(k in text_lower for k in ['ph.d', 'phd', 'doctor']):
                edu = 'PhD'
            elif any(k in text_lower for k in ['master', 'msc', 'm.s', 'mba', 'm.tech']):
                edu = 'Master'
            elif any(k in text_lower for k in ['bachelor', 'b.sc', 'b.e', 'b.tech', 'undergraduate']):
                edu = 'Bachelor'
            else:
                edu = 'High School'

            # University tier heuristic (top schools → Tier 1)
            tier1_keywords = ['iit', 'mit', 'stanford', 'harvard', 'oxford', 'cambridge', 'nit']
            tier2_keywords = ['vit', 'manipal', 'bits', 'srm', 'psg', 'anna university']
            if any(k in text_lower for k in tier1_keywords):
                tier = 'Tier 1 (Top)'
            elif any(k in text_lower for k in tier2_keywords):
                tier = 'Tier 2'
            else:
                tier = 'Tier 3'

            # Certifications count
            cert_count = len(re.findall(
                r'\b(?:certified|certification|certificate|aws|gcp|azure|pmp|cissp|comptia|ceh|ccna)\b',
                text_lower
            ))

            # Leadership
            leadership = 1 if any(k in text_lower for k in [
                'led', 'managed', 'head of', 'team lead', 'director', 'manager', 'supervised'
            ]) else 0

            # Skill match — based on keyword density across common tech skills
            tech_keywords = [
                'python', 'java', 'sql', 'machine learning', 'deep learning', 'nlp',
                'cloud', 'aws', 'docker', 'kubernetes', 'react', 'node', 'tensorflow',
                'pytorch', 'data analysis', 'statistics', 'cybersecurity', 'networking',
                'agile', 'scrum', 'product management', 'excel', 'tableau', 'power bi'
            ]
            hits       = sum(1 for kw in tech_keywords if kw in text_lower)
            skill_score = min(1.0, hits / 10.0)

            # Field of study
            fos_map = {
                'computer science': 'Computer Science',
                'software': 'Computer Science',
                'data science': 'Computer Science',
                'engineering': 'Engineering',
                'business': 'Business',
                'management': 'Business',
                'mathematics': 'Engineering',
                'statistics': 'Engineering',
            }
            field = 'Other'
            for kw, val in fos_map.items():
                if kw in text_lower:
                    field = val
                    break

            # Previous jobs — count company/employer sections
            job_count = len(re.findall(
                r'\b(?:company|employer|organization|inc\.|ltd\.|pvt\.|llc)\b', text_lower
            ))
            job_count = max(1, min(job_count, 10))

            # Salary (current) — heuristic, often not on resume; default to 80000
            salary = 80000.0

            # Interview score and distance — not on resume, use pool averages
            avg_interview = float(df_display['interview_score'].mean())
            avg_distance  = float(df_display['distance_from_office'].mean())

            return {
                'years_of_experience':      years_exp,
                'relevant_experience_years': years_exp,
                'education_level':          edu,
                'university_tier':          tier,
                'num_certifications':       cert_count,
                'has_leadership_experience': leadership,
                'skill_match_score':        round(skill_score, 3),
                'field_of_study':           field,
                'num_previous_jobs':        job_count,
                'current_salary':           salary,
                'interview_score':          avg_interview,
                'distance_from_office':     avg_distance,
            }

        # ── Helper: BERT NER to pull named entities ──
        def run_bert_ner(text: str) -> dict:
            # Chunk text to stay within BERT 512-token limit
            words   = text.split()
            chunks  = [' '.join(words[i:i+400]) for i in range(0, len(words), 400)]
            all_ents = []
            for chunk in chunks[:5]:   # cap at 5 chunks
                try:
                    ents = ner_pipe(chunk)
                    all_ents.extend(ents)
                except Exception:
                    pass
            persons = [e['word'] for e in all_ents if e['entity_group'] == 'PER']
            orgs    = [e['word'] for e in all_ents if e['entity_group'] == 'ORG']
            locs    = [e['word'] for e in all_ents if e['entity_group'] == 'LOC']
            return {
                'Detected Names':         ', '.join(set(persons[:3])) or 'Not detected',
                'Detected Organizations': ', '.join(set(orgs[:5]))    or 'Not detected',
                'Detected Locations':     ', '.join(set(locs[:3]))    or 'Not detected',
            }

        # ── Helper: build feature row for model ──
        def build_feature_row(fields: dict, job_role: str, gender: str,
                              nationality: str, app_source: str) -> pd.DataFrame:
            row = {
                'age':                     30,
                'gender':                  gender,
                'nationality':             nationality,
                'education_level':         fields['education_level'],
                'field_of_study':          fields['field_of_study'],
                'university_tier':         fields['university_tier'],
                'years_of_experience':     fields['years_of_experience'],
                'relevant_experience_years': fields['relevant_experience_years'],
                'current_salary':          fields['current_salary'],
                'num_previous_jobs':       fields['num_previous_jobs'],
                'skill_match_score':       fields['skill_match_score'],
                'num_certifications':      fields['num_certifications'],
                'has_leadership_experience': fields['has_leadership_experience'],
                'interview_score':         fields['interview_score'],
                'distance_from_office':    fields['distance_from_office'],
                'job_role_applied':        job_role,
                'application_source':      app_source,
            }
            df_row = pd.DataFrame([row])
            for col, le in encoders.items():
                if col in df_row.columns:
                    val = str(df_row.at[0, col])
                    if val in le.classes_:
                        df_row[col] = le.transform([val])
                    else:
                        df_row[col] = le.transform([le.classes_[0]])
            return df_row[feature_cols]

        # ── UI ──
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF only)", type=["pdf"],
            help="The resume will be processed locally. No data is stored."
        )

        if uploaded_file:
            c_role, c_gender, c_nat, c_src = st.columns(4)
            with c_role:
                job_role = st.selectbox("Job Role Applying For",
                                        sorted(df_display['job_role_applied'].unique().tolist()))
            with c_gender:
                gender = st.selectbox("Gender", sorted(df_display['gender'].unique().tolist()))
            with c_nat:
                nationality = st.selectbox("Nationality", sorted(df_display['nationality'].unique().tolist()))
            with c_src:
                app_source = st.selectbox("Application Source",
                                          sorted(df_display['application_source'].unique().tolist()))

            if st.button("Analyse Resume", type="primary", use_container_width=True):
                with st.spinner("Extracting text from PDF..."):
                    raw_text = extract_pdf_text(uploaded_file)

                if not raw_text.strip():
                    st.error("Could not extract text from this PDF. Make sure it is not a scanned image-only PDF.")
                else:
                    with st.spinner("Running BERT Named Entity Recognition..."):
                        bert_entities = run_bert_ner(raw_text)

                    with st.spinner("Extracting structured fields..."):
                        fields = extract_fields(raw_text)

                    with st.spinner("Running hiring prediction..."):
                        feature_row  = build_feature_row(fields, job_role, gender, nationality, app_source)
                        hire_prob    = model.predict_proba(feature_row)[0, 1]
                        decision     = "SHORTLIST" if hire_prob > 0.5 else "REJECT"
                        dec_color    = "#22c55e" if hire_prob > 0.5 else "#ef4444"

                        sv_res = explainer.shap_values(feature_row)
                        if isinstance(sv_res, list): shap_vals_r = np.array(sv_res[1]).flatten()
                        elif sv_res.ndim == 3:        shap_vals_r = sv_res[0, :, 1]
                        else:                         shap_vals_r = np.array(sv_res).flatten()
                        shap_df_r = pd.DataFrame({"Feature": feature_cols, "Impact": shap_vals_r}).sort_values("Impact")

                    st.divider()

                    # ── Results layout ──
                    r1, r2, r3 = st.columns([1, 1.8, 2])

                    with r1:
                        st.markdown("**Prediction**")
                        st.markdown(f"""
                        <div style='text-align:center; padding:20px; border-radius:12px;
                                    background:{dec_color}18; border:2px solid {dec_color};'>
                            <div style='font-size:2.5rem; font-weight:900; color:{dec_color};'>{hire_prob:.1%}</div>
                            <div style='font-size:0.9rem; color:{dec_color}; font-weight:700; margin-top:4px;'>
                                {decision}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**BERT Detected Entities**")
                        for k, v in bert_entities.items():
                            st.markdown(f"""
                            <div class='field-row'>
                                <span class='field-label'>{k}</span>
                                <span class='field-value'>{v}</span>
                            </div>""", unsafe_allow_html=True)

                    with r2:
                        st.markdown("**Extracted Resume Fields**")
                        field_labels = {
                            'years_of_experience':       'Total Experience',
                            'relevant_experience_years': 'Relevant Experience',
                            'education_level':           'Education Level',
                            'university_tier':           'University Tier',
                            'field_of_study':            'Field of Study',
                            'skill_match_score':         'Skill Match Score',
                            'num_certifications':        'Certifications Found',
                            'has_leadership_experience': 'Leadership Experience',
                            'num_previous_jobs':         'Previous Jobs (est.)',
                        }
                        for key, label in field_labels.items():
                            val = fields[key]
                            if key == 'has_leadership_experience':
                                val = 'Yes' if val else 'No'
                            elif isinstance(val, float):
                                val = f"{val:.2f}" if key == 'skill_match_score' else f"{val:.1f}"
                            st.markdown(f"""
                            <div class='field-row'>
                                <span class='field-label'>{label}</span>
                                <span class='field-value'>{val}</span>
                            </div>""", unsafe_allow_html=True)

                    with r3:
                        st.markdown("**SHAP Feature Contributions**")
                        shap_colors = ['#22c55e' if x > 0 else '#ef4444' for x in shap_df_r["Impact"]]
                        fig_r = go.Figure(go.Bar(
                            x=shap_df_r["Impact"], y=shap_df_r["Feature"],
                            orientation='h', marker_color=shap_colors,
                            text=[f"{v:+.3f}" for v in shap_df_r["Impact"]],
                            textposition='outside'
                        ))
                        fig_r.add_vline(x=0, line_color="#475569", line_dash="dot")
                        fig_r.update_layout(
                            height=480, plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=10, b=10, l=10, r=60),
                            xaxis_title="SHAP Value"
                        )
                        st.plotly_chart(fig_r, use_container_width=True)

                    # Raw text expander
                    with st.expander("View Raw Extracted Resume Text"):
                        st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))
        else:
            st.info("Upload a PDF resume above to begin screening.")


# ═════════════════════════════════════════════════════════
# PAGE: HR REPORT
# ═════════════════════════════════════════════════════════
elif active_page == "HR Report":

    st.markdown('<div class="section-header">HR Summary Report</div>', unsafe_allow_html=True)
    st.caption("A concise summary of the current candidate pool — pool statistics, top insights, and fairness status — "
               "ready to share with hiring managers or HR leads.")
    st.markdown("")

    if not filtered_df.empty:
        shortlisted_count = (filtered_df['Hiring Probability'] > 0.5).sum()
        rejected_count    = len(filtered_df) - shortlisted_count
        top_role = filtered_df.groupby('job_role_applied')['Hiring Probability'].mean().idxmax()
        top_edu  = filtered_df.groupby('education_level')['Hiring Probability'].mean().idxmax()
        top_src  = filtered_df.groupby('application_source')['Hiring Probability'].mean().idxmax() \
                   if 'application_source' in filtered_df.columns else "N/A"
        top_tier = filtered_df.groupby('university_tier')['Hiring Probability'].mean().idxmax()

        gr = filtered_df.copy()
        gr['Selected'] = gr['Hiring Probability'] > 0.5
        g_gap = gr.groupby('gender')['Selected'].mean()
        g_gap = g_gap.max() - g_gap.min() if len(g_gap) > 1 else 0
        bias_str = f"{g_gap:.1%} gap — Bias Detected" if g_gap > BIAS_THRESHOLD else f"{g_gap:.1%} gap — Fair"
        bias_color = "#ef4444" if g_gap > BIAS_THRESHOLD else "#22c55e"

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div style='background:#1e293b; border-radius:12px; padding:20px 24px; border:1px solid #2d2d3f;'>
                <div style='font-size:1rem; font-weight:700; color:#3b82f6; margin-bottom:12px;'>Pool Summary</div>
                <table style='width:100%; font-size:0.88rem; border-collapse:collapse;'>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Total Candidates</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{len(filtered_df):,}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Shortlisted</td>
                        <td style='color:#22c55e;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{shortlisted_count:,} ({shortlisted_count/len(filtered_df):.1%})</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Rejected</td>
                        <td style='color:#ef4444;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{rejected_count:,} ({rejected_count/len(filtered_df):.1%})</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Avg Hiring Probability</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{filtered_df['Hiring Probability'].mean():.1%}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Avg Skill Score</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{filtered_df['skill_match_score'].mean():.3f}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;'>Avg Interview Score</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;'>{filtered_df['interview_score'].mean():.1f}</td></tr>
                </table>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div style='background:#1e293b; border-radius:12px; padding:20px 24px; border:1px solid #2d2d3f;'>
                <div style='font-size:1rem; font-weight:700; color:#3b82f6; margin-bottom:12px;'>Key Insights</div>
                <table style='width:100%; font-size:0.88rem; border-collapse:collapse;'>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Top Performing Role</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{top_role}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Top Education Level</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{top_edu}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Best Application Source</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{top_src}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;border-bottom:1px solid #2d2d3f;'>Top University Tier</td>
                        <td style='color:#f1f5f9;font-weight:700;text-align:right;border-bottom:1px solid #2d2d3f;'>{top_tier}</td></tr>
                    <tr><td style='color:#9ca3af;padding:6px 0;'>Gender Fairness</td>
                        <td style='color:{bias_color};font-weight:700;text-align:right;'>{bias_str}</td></tr>
                </table>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report_csv = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download Full HR Report (CSV)", data=report_csv,
                           file_name="hr_full_report.csv", mime="text/csv")

    else:
        st.warning("No candidates available. Go to Candidate Pool to adjust filters.")

st.divider()
st.caption("FairHire AI — XAI-Driven Hiring & HR Analytics | Powered by Random Forest + SHAP | Built with Streamlit")