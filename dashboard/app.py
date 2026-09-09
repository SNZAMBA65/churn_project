# ============================================================
# DASHBOARD STREAMLIT - Prédiction du Churn Client
# Projet #3 - Machine Learning · DPIA 1
# Auteur : Samir NZAMBA · Fonderie de l'Image
# ============================================================

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    f1_score, recall_score, precision_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Analytics · Telco",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}

.block-container {
    padding: 2rem 2.75rem !important;
    max-width: 1380px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c2541 0%, #0d1321 100%) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 1.1rem 0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.875rem !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    opacity: 0.55 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div > div { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox svg { fill: #ffffff !important; }
[data-testid="stSidebar"] ul[role="listbox"],
[data-testid="stSidebar"] div[role="listbox"] {
    background-color: #18213b !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] ul[role="listbox"] li,
[data-testid="stSidebar"] div[role="listbox"] div { color: #ffffff !important; }

[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 1.1rem 1.3rem 1rem 1.3rem;
    border: 1px solid rgba(128,128,128,0.15);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.page-banner {
    background: linear-gradient(135deg, #1c2541 0%, #3a4a7a 100%);
    border-radius: 14px;
    padding: 1.75rem 2.25rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.banner-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
    line-height: 1.2;
}
.banner-desc {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
}
.banner-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

.chart-card {
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.15);
    padding: 1.25rem 1.5rem 0.75rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 1.25rem;
}

.author-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.875rem 1rem;
    margin-top: 0.5rem;
}

.risk-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ─── Palette ──────────────────────────────────────────────────────────────────

BLEU    = "#1c2541"
ACCENT  = "#3a4a7a"
VERT    = "#2E7D32"
ORANGE  = "#DAA520"
ROUGE   = "#C1002A"
GRIS    = "#64748b"
FONT    = "Inter, system-ui, sans-serif"

PALETTE = [BLEU, ORANGE, ACCENT, VERT, ROUGE, "#6A1B9A", "#00838F"]
COLOR_CHURN = {"No": VERT, "Yes": ROUGE}

COLORSCALE_BLEU = [
    [0,   "#c7ccdb"],
    [0.5, "#5a6c9e"],
    [1,   BLEU],
]


def plo(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12),
        margin=dict(t=20, b=20, l=8, r=16),
        hoverlabel=dict(
            bgcolor="#0a1628", font_size=12,
            font_color="white", bordercolor="#1e293b",
        ),
    )
    base.update(kw)
    return base


def ax(**kw):
    d = dict(
        showgrid=False,
        linecolor="rgba(128,128,128,0.2)",
        tickcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11),
        title_font=dict(size=11),
    )
    d.update(kw)
    return d


def ay(**kw):
    d = dict(
        gridcolor="rgba(128,128,128,0.12)",
        gridwidth=1,
        linecolor="rgba(0,0,0,0)",
        tickcolor="rgba(0,0,0,0)",
        tickfont=dict(size=11),
        title_font=dict(size=11),
    )
    d.update(kw)
    return d


def chart(fig):
    st.plotly_chart(
        fig, use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        }
    )

# ─── Données et modèles ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(EXPORTS_DIR, 'dataset_clean.csv'))


@st.cache_data
def load_results():
    return pd.read_csv(os.path.join(EXPORTS_DIR, 'resultats_modeles.csv'))


@st.cache_data
def load_cv_results():
    try:
        return pd.read_csv(os.path.join(EXPORTS_DIR, 'validation_croisee.csv'))
    except FileNotFoundError:
        return None


@st.cache_resource
def load_models():
    model_files = {
        'Régression Logistique': 'regression_logistique.pkl',
        'Arbre de Décision': 'arbre_de_decision.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'LightGBM': 'lightgbm.pkl',
        'Random Forest Optimisé': 'random_forest_best.pkl',
        'XGBoost Optimisé': 'xgboost_best.pkl',
        'LightGBM Optimisé': 'lightgbm_best.pkl',
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    threshold_path = os.path.join(MODELS_DIR, 'optimal_threshold.pkl')
    optimal_threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5

    return models, scaler, feature_names, optimal_threshold


@st.cache_data
def prepare_test_data():
    df = load_data()
    df_encoded = df.copy()

    binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'PhoneService', 'PaperlessBilling', 'Churn']
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in binary_cols:
        df_encoded[col] = df_encoded[col].map(binary_map)

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cols, drop_first=True)

    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl')).transform(X_test)

    return X_test, X_test_scaled, y_test


df = load_data()
results = load_results()
cv_results = load_cv_results()
models, scaler, feature_names, optimal_threshold = load_models()
X_test, X_test_scaled, y_test = prepare_test_data()

NB_CLIENTS = len(df)
TAUX_CHURN = (df['Churn'] == 'Yes').mean() * 100
BEST_ROW = results.loc[results['AUC-ROC'].idxmax()]

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1rem;font-weight:700;"
        "letter-spacing:-0.01em;margin-bottom:0.1rem;'>"
        "📡 Churn Analytics</div>"
        "<div style='font-size:0.73rem;opacity:0.5;"
        "margin-bottom:1.25rem;'>Telco Customer · DPIA 1</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Section",
        ["Vue d'ensemble", "Analyse exploratoire", "Performance des modèles",
         "Qualité & Monitoring", "Simulateur"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if page == "Simulateur":
        st.markdown(
            "<div style='font-size:0.68rem;opacity:0.55;text-transform:uppercase;"
            "letter-spacing:0.09em;font-weight:600;margin-bottom:0.5rem;'>"
            "Paramètres de prédiction</div>",
            unsafe_allow_html=True
        )
        sim_model_choice = st.selectbox(
            "Modèle",
            list(models.keys()),
            index=list(models.keys()).index('XGBoost Optimisé') if 'XGBoost Optimisé' in models else 0
        )
        sim_use_optimal = st.checkbox(
            f"Seuil optimisé ({optimal_threshold:.2f})", value=True
        )
        st.markdown("---")

    st.markdown(
        "<div class='author-card'>"
        "<div style='font-size:0.78rem;font-weight:600;"
        "margin-bottom:0.3rem;'>Samir NZAMBA</div>"
        "<div style='font-size:0.7rem;opacity:0.65;line-height:1.6;'>"
        "Mastère DPIA 1<br>"
        "Directeur de Projet IA<br>"
        "Fonderie de l'Image<br>"
        "Projet #3 · Machine Learning"
        "</div></div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.68rem;opacity:0.45;line-height:1.9;'>"
        "Source · Telco Customer Churn (Kaggle)<br>"
        f"{NB_CLIENTS:,} clients · {df.shape[1]} variables<br>"
        f"Meilleur modèle : {BEST_ROW['Modèle']}<br>"
        f"AUC-ROC : {BEST_ROW['AUC-ROC']:.4f}"
        "</div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════
# VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

if page == "Vue d'ensemble":

    st.markdown(f"""
    <div class="page-banner">
        <div>
            <div class="banner-title">Prédiction du churn client</div>
            <div class="banner-desc">
                Modélisation du risque de résiliation · Telco Customer Dataset
                · {NB_CLIENTS:,} clients analysés
            </div>
        </div>
        <div class="banner-badge">● {BEST_ROW['Modèle']}</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clients analysés", f"{NB_CLIENTS:,}", "dataset Telco")
    k2.metric("Taux de churn", f"{TAUX_CHURN:.1f} %",
              f"{(df['Churn']=='Yes').sum():,} clients perdus")
    k3.metric("Mensualité moyenne", f"{df['MonthlyCharges'].mean():.0f} €",
              f"médiane {df['MonthlyCharges'].median():.0f} €")
    k4.metric("Ancienneté moyenne", f"{df['tenure'].mean():.0f} mois",
              f"médiane {df['tenure'].median():.0f} mois")
    k5.metric("Meilleur AUC-ROC", f"{BEST_ROW['AUC-ROC']:.4f}",
              f"Recall {BEST_ROW['Recall']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Objectif du projet")
        st.caption(
            "Anticiper les résiliations clients pour permettre des "
            "actions de rétention ciblées"
        )
        st.markdown(
            "Développer un modèle de Machine Learning capable de "
            "prédire le churn des clients d'un opérateur télécom, "
            "à partir de leurs caractéristiques démographiques, "
            "contractuelles et d'usage des services."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Approche méthodologique")
        st.caption("Pipeline complet, de la donnée brute au modèle déployé")

        etapes = [
            ("1", "Collecte & nettoyage",
             "7 043 clients, traitement des valeurs manquantes et incohérences", BLEU),
            ("2", "Analyse exploratoire",
             "Identification des facteurs de churn par segment client", ACCENT),
            ("3", "Modélisation",
             "5 algorithmes comparés : Régression Logistique, Arbre, "
             "Random Forest, XGBoost, LightGBM", VERT),
            ("4", "Optimisation",
             "SMOTE, GridSearchCV, validation croisée 5-fold, seuil de décision optimisé", ORANGE),
            ("5", "Monitoring",
             "Suivi de la qualité du modèle et du data drift via Evidently", ROUGE),
        ]
        for num, titre, desc, color in etapes:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:0.9rem;
                        align-items:flex-start;">
                <div style="min-width:1.8rem;height:1.8rem;
                            background:{color};border-radius:50%;
                            display:flex;align-items:center;
                            justify-content:center;color:white;
                            font-weight:700;font-size:0.8rem;">
                    {num}
                </div>
                <div>
                    <div style="font-weight:600;font-size:0.88rem;
                                margin-bottom:0.15rem;">{titre}</div>
                    <div style="font-size:0.8rem;opacity:0.65;
                                line-height:1.45;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f"### {BEST_ROW['Modèle']}")
        st.caption("Modèle retenu pour la mise en production")

        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC", f"{BEST_ROW['AUC-ROC']:.4f}")
        m2.metric("Accuracy", f"{BEST_ROW['Accuracy']:.1%}")
        m1.metric("Recall (Churné)", f"{BEST_ROW['Recall']:.1%}")
        m2.metric("F1-score", f"{BEST_ROW['F1']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Facteurs de risque majeurs")
        st.caption("Segments présentant le plus fort taux de churn")

        risk_factors = [
            ("Contrat mensuel", "42.7 %", ROUGE),
            ("Nouveau client (< 12 mois)", "47.7 %", ROUGE),
            ("Fiber optic", "41.9 %", ORANGE),
            ("Sans support technique", "41.6 %", ORANGE),
        ]
        for label, val, color in risk_factors:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        align-items:center;padding:0.55rem 0;
                        border-bottom:1px solid rgba(128,128,128,0.12);">
                <span style="font-size:0.88rem;">{label}</span>
                <span class="risk-pill" style="background:{color}1a;color:{color};">
                    {val}
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ANALYSE EXPLORATOIRE
# ═══════════════════════════════════════════════════════════════════

elif page == "Analyse exploratoire":

    st.markdown(f"""
    <div class="page-banner">
        <div>
            <div class="banner-title">Analyse exploratoire</div>
            <div class="banner-desc">
                Comportement des clients et facteurs associés au churn
                · {NB_CLIENTS:,} clients · {df.shape[1]} variables
            </div>
        </div>
        <div class="banner-badge">● Taux de churn {TAUX_CHURN:.1f} %</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution", "Démographie", "Contrat & Services", "Variables numériques"
    ])

    # ── Distribution ─────────────────────────────────────────────
    with tab1:
        churn_counts = df['Churn'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Distribution du churn")
            st.caption("Répartition des clients retenus et perdus")

            fig = go.Figure(go.Bar(
                x=churn_counts.index, y=churn_counts.values,
                marker_color=[COLOR_CHURN[c] for c in churn_counts.index],
                marker_opacity=0.88,
                marker_line=dict(color="rgba(0,0,0,0)"),
                text=churn_counts.values, textposition="outside",
                textfont=dict(size=14, family=FONT),
                hovertemplate="<b>%{x}</b><br>%{y:,} clients<extra></extra>",
            ))
            fig.update_layout(**plo(height=340, bargap=0.5))
            fig.update_xaxes(**ax(tickfont=dict(size=12)))
            fig.update_yaxes(**ay())
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Répartition (%)")
            st.caption("Proportion churné vs retenu")

            fig = go.Figure(go.Pie(
                labels=churn_counts.index, values=churn_counts.values,
                hole=0.65,
                marker=dict(
                    colors=[COLOR_CHURN[c] for c in churn_counts.index],
                    line=dict(color="white", width=4)
                ),
                textinfo="percent+label",
                textfont=dict(size=12, color="white", family=FONT),
                hovertemplate="<b>%{label}</b><br>%{value:,} clients<br>%{percent}<extra></extra>",
                rotation=90, pull=[0.03, 0.03],
            ))
            fig.update_layout(**plo(
                height=340, showlegend=False,
                annotations=[dict(
                    text=f"<b>{TAUX_CHURN:.0f} %</b><br>"
                         "<span style='font-size:11px;'>churn</span>",
                    x=0.5, y=0.5, font_size=18, showarrow=False, font=dict(family=FONT)
                )]
            ))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            f"**Dataset déséquilibré** : {100-TAUX_CHURN:.1f} % de clients retenus "
            f"contre {TAUX_CHURN:.1f} % de churn. Ce déséquilibre est traité via "
            f"SMOTE lors de l'entraînement des modèles."
        )

    # ── Démographie ──────────────────────────────────────────────
    with tab2:
        demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        demo_labels = {'gender': 'Genre', 'SeniorCitizen': 'Senior',
                        'Partner': 'Partenaire', 'Dependents': 'Dépendants'}

        col1, col2 = st.columns(2)
        for i, col_name in enumerate(demo_cols):
            churn_rate = df.groupby(col_name)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_rate.columns = [col_name, 'taux']

            target = col1 if i % 2 == 0 else col2
            with target:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f"### Churn par {demo_labels[col_name].lower()}")
                st.caption(f"Taux de résiliation selon {demo_labels[col_name].lower()}")

                fig = go.Figure(go.Bar(
                    x=churn_rate[col_name], y=churn_rate['taux'],
                    marker_color=BLEU, marker_opacity=0.85,
                    marker_line=dict(color="rgba(0,0,0,0)"),
                    text=churn_rate['taux'].apply(lambda x: f"{x:.1f} %"),
                    textposition="outside",
                    textfont=dict(size=12, family=FONT),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f} % de churn<extra></extra>",
                ))
                fig.update_layout(**plo(height=300, bargap=0.4))
                fig.update_xaxes(**ax(tickfont=dict(size=12)))
                fig.update_yaxes(**ay(range=[0, 55]))
                chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "Le genre n'influence pas le churn (26.9 % vs 26.2 %). Les seniors "
            "churne à 41.7 % contre 23.6 % pour les non-seniors. L'absence de "
            "partenaire ou de personnes à charge est associée à un taux plus élevé."
        )

    # ── Contrat & Services ───────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Taux de churn par type de contrat")
            st.caption("L'engagement contractuel est le facteur le plus discriminant")

            churn_contract = df.groupby('Contract')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_contract.columns = ['Contract', 'taux']

            fig = go.Figure(go.Bar(
                x=churn_contract['Contract'], y=churn_contract['taux'],
                marker_color=[ROUGE, ORANGE, VERT], marker_opacity=0.88,
                marker_line=dict(color="rgba(0,0,0,0)"),
                text=churn_contract['taux'].apply(lambda x: f"{x:.1f} %"),
                textposition="outside", textfont=dict(size=12, family=FONT),
                hovertemplate="<b>%{x}</b><br>%{y:.1f} % de churn<extra></extra>",
            ))
            fig.update_layout(**plo(height=360, bargap=0.45))
            fig.update_xaxes(**ax(tickfont=dict(size=11)))
            fig.update_yaxes(**ay(range=[0, 55]))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Taux de churn par ancienneté")
            st.caption("Les nouveaux clients constituent le segment le plus à risque")

            df_box = df.copy()
            df_box['tenure_group'] = pd.cut(
                df_box['tenure'], bins=[0, 12, 24, 48, 72],
                labels=['0-12 mois', '12-24 mois', '24-48 mois', '48-72 mois']
            )
            churn_tenure = df_box.groupby('tenure_group', observed=True)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_tenure.columns = ['Ancienneté', 'taux']

            fig = go.Figure(go.Bar(
                x=churn_tenure['Ancienneté'].astype(str), y=churn_tenure['taux'],
                marker_color=[ROUGE, ORANGE, "#e0b84a", VERT], marker_opacity=0.88,
                marker_line=dict(color="rgba(0,0,0,0)"),
                text=churn_tenure['taux'].apply(lambda x: f"{x:.1f} %"),
                textposition="outside", textfont=dict(size=12, family=FONT),
                hovertemplate="<b>%{x}</b><br>%{y:.1f} % de churn<extra></extra>",
            ))
            fig.update_layout(**plo(height=360, bargap=0.45))
            fig.update_xaxes(**ax(tickfont=dict(size=11)))
            fig.update_yaxes(**ay(range=[0, 55]))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Impact des services souscrits")
        st.caption("Taux de churn selon le type de service ou de paiement")

        service_cols = ['InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
        cols = st.columns(4)
        for i, col_name in enumerate(service_cols):
            churn_rate = df.groupby(col_name)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_rate.columns = [col_name, 'taux']

            with cols[i]:
                fig = go.Figure(go.Bar(
                    x=churn_rate[col_name], y=churn_rate['taux'],
                    marker_color=PALETTE[i % len(PALETTE)], marker_opacity=0.85,
                    marker_line=dict(color="rgba(0,0,0,0)"),
                    text=churn_rate['taux'].apply(lambda x: f"{x:.0f}%"),
                    textposition="outside", textfont=dict(size=10, family=FONT),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f} %<extra></extra>",
                ))
                fig.update_layout(**plo(height=300, bargap=0.4,
                                         margin=dict(t=30, b=20, l=4, r=4)))
                fig.update_xaxes(**ax(tickfont=dict(size=9), tickangle=-25))
                fig.update_yaxes(**ay(range=[0, 50], title_text=col_name, title_font=dict(size=10)))
                chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Variables numériques ─────────────────────────────────────
    with tab4:
        num_var = st.selectbox(
            "Variable numérique", ['tenure', 'MonthlyCharges', 'TotalCharges']
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f"### Distribution de {num_var}")
            st.caption("Comparaison entre clients retenus et churnés")

            fig = go.Figure()
            for val, label in [("No", "Resté"), ("Yes", "Churné")]:
                fig.add_trace(go.Histogram(
                    x=df[df['Churn'] == val][num_var], name=label,
                    marker_color=COLOR_CHURN[val], opacity=0.6, nbinsx=35,
                ))
            fig.update_layout(**plo(
                height=360, barmode="overlay",
                legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)")
            ))
            fig.update_xaxes(**ax(title_text=num_var))
            fig.update_yaxes(**ay(title_text="Nombre de clients"))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f"### Boxplot de {num_var}")
            st.caption("Médiane, dispersion et valeurs extrêmes")

            fig = go.Figure()
            for val, label in [("No", "Resté"), ("Yes", "Churné")]:
                fig.add_trace(go.Box(
                    y=df[df['Churn'] == val][num_var], name=label,
                    marker_color=COLOR_CHURN[val], boxmean=True,
                ))
            fig.update_layout(**plo(height=360, showlegend=False))
            fig.update_xaxes(**ax())
            fig.update_yaxes(**ay(title_text=num_var))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Statistiques descriptives")
        st.caption(f"Résumé statistique de {num_var} par statut de churn")
        st.dataframe(
            df.groupby('Churn')[num_var].describe().round(2),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE DES MODÈLES
# ═══════════════════════════════════════════════════════════════════

elif page == "Performance des modèles":

    st.markdown(f"""
    <div class="page-banner">
        <div>
            <div class="banner-title">Performance des modèles</div>
            <div class="banner-desc">
                Comparaison de {len(models)} modèles entraînés et évalués
                sur le jeu de test
            </div>
        </div>
        <div class="banner-badge">● Meilleur AUC-ROC {BEST_ROW['AUC-ROC']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Comparaison", "Courbes ROC", "Matrices de confusion",
        "Feature Importance", "Validation croisée"
    ])

    # ── Comparaison ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Tableau comparatif")
        st.caption("Métriques calculées sur le jeu de test, données jamais vues à l'entraînement")
        st.dataframe(
            results.style
            .highlight_max(
                subset=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
                color='rgba(28,37,65,0.18)'
            )
            .format({
                'Accuracy': '{:.2%}', 'Precision': '{:.2%}',
                'Recall': '{:.2%}', 'F1': '{:.4f}', 'AUC-ROC': '{:.4f}'
            }),
            use_container_width=True, height=320
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        col_titre, col_select = st.columns([3, 1])
        with col_titre:
            st.markdown("### Comparaison par métrique")
            st.caption("Classement des modèles selon la métrique sélectionnée")
        with col_select:
            metric_choice = st.selectbox(
                "Métrique", ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
                label_visibility="collapsed"
            )

        df_sorted = results.sort_values(metric_choice, ascending=True)
        fig = go.Figure(go.Bar(
            x=df_sorted[metric_choice], y=df_sorted['Modèle'],
            orientation='h',
            marker=dict(
                color=df_sorted[metric_choice], colorscale=COLORSCALE_BLEU,
                showscale=False, line=dict(color="rgba(0,0,0,0)")
            ),
            text=df_sorted[metric_choice].apply(lambda x: f"{x:.4f}"),
            textposition="outside", textfont=dict(size=11, family=FONT),
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        ))
        fig.update_layout(**plo(height=420))
        fig.update_xaxes(**ax(title_text=metric_choice))
        fig.update_yaxes(tickfont=dict(size=11), gridcolor="rgba(0,0,0,0)")
        chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Courbes ROC ──────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Courbes ROC — comparaison des modèles")
        st.caption(
            "Plus une courbe se rapproche du coin supérieur gauche, "
            "meilleur est le modèle"
        )

        fig = go.Figure()
        for i, (name, model) in enumerate(models.items()):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.4f})',
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.2),
                hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire (AUC=0.5)',
            line=dict(color=GRIS, width=1.3, dash='dash'),
        ))
        fig.update_layout(**plo(
            height=540,
            legend=dict(orientation="v", x=1.02, y=0.5, bgcolor="rgba(0,0,0,0)")
        ))
        fig.update_xaxes(**ax(title_text="Taux de Faux Positifs"))
        fig.update_yaxes(**ay(title_text="Taux de Vrais Positifs"))
        chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            "L'AUC mesure l'aire sous la courbe (1.0 = parfait, 0.5 = aléatoire). "
            "Les modèles boostés (XGBoost, LightGBM) dominent légèrement "
            "les modèles classiques sur ce jeu de données."
        )

    # ── Matrices de confusion ────────────────────────────────────
    with tab3:
        model_choice = st.selectbox("Modèle", list(models.keys()), key="cm_model")
        model = models[model_choice]
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        col1, col2 = st.columns([1, 1.1])

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f"### Matrice de confusion")
            st.caption(model_choice)

            fig = go.Figure(go.Heatmap(
                z=cm, x=['Resté', 'Churné'], y=['Resté', 'Churné'],
                colorscale=COLORSCALE_BLEU, showscale=False,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=18, color="white", family=FONT),
                hovertemplate="Réel: %{y}<br>Prédit: %{x}<br>%{z} clients<extra></extra>",
            ))
            fig.update_layout(**plo(height=400))
            fig.update_xaxes(title_text="Prédit", side="bottom")
            fig.update_yaxes(title_text="Réel", autorange="reversed")
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Détail des prédictions")
            st.caption("Lecture des quatre cas de la matrice")

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.2%}")
            m1.metric("Recall", f"{rec:.2%}")
            m2.metric("F1-score", f"{f1:.4f}")

            st.markdown("<br>", unsafe_allow_html=True)
            details = [
                ("Vrais Négatifs", tn, "Non-churners correctement identifiés", VERT),
                ("Faux Positifs", fp, "Non-churners classés comme churners", ORANGE),
                ("Faux Négatifs", fn, "Churners non détectés — cas critique", ROUGE),
                ("Vrais Positifs", tp, "Churners correctement détectés", BLEU),
            ]
            for label, val, desc, color in details:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            align-items:center;padding:0.5rem 0;
                            border-bottom:1px solid rgba(128,128,128,0.12);">
                    <div>
                        <div style="font-weight:600;font-size:0.85rem;">{label}</div>
                        <div style="font-size:0.75rem;opacity:0.6;">{desc}</div>
                    </div>
                    <span class="risk-pill" style="background:{color}1a;color:{color};">
                        {val}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature Importance ───────────────────────────────────────
    with tab4:
        tree_models = [
            name for name, m in models.items()
            if hasattr(m, 'feature_importances_')
        ]
        default_fi = 'XGBoost Optimisé' if 'XGBoost Optimisé' in tree_models else (
            tree_models[0] if tree_models else list(models.keys())[0]
        )

        st.markdown('', unsafe_allow_html=True)
        col_titre, col_select = st.columns([3, 1])
        with col_titre:
            st.markdown("### Top 15 features")
            st.caption("Variables les plus déterminantes dans la décision du modèle")
        with col_select:
            if tree_models:
                fi_model_name = st.selectbox(
                    "Modèle", tree_models,
                    index=tree_models.index(default_fi),
                    label_visibility="collapsed"
                )
            else:
                fi_model_name = list(models.keys())[0]

        fi_model = models[fi_model_name]

        if hasattr(fi_model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': fi_model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(15)

            fig = go.Figure(go.Bar(
                x=feature_imp['Importance'], y=feature_imp['Feature'],
                orientation='h',
                marker=dict(
                    color=feature_imp['Importance'], colorscale=COLORSCALE_BLEU,
                    showscale=False, line=dict(color="rgba(0,0,0,0)")
                ),
                text=feature_imp['Importance'].apply(lambda x: f"{x:.3f}"),
                textposition="outside", textfont=dict(size=10.5, family=FONT),
                hovertemplate="<b>%{y}</b><br>Importance %{x:.3f}<extra></extra>",
            ))
            fig.update_layout(**plo(height=520))
            fig.update_xaxes(**ax(title_text="Importance"))
            fig.update_yaxes(tickfont=dict(size=10.5), gridcolor="rgba(0,0,0,0)")
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.success(
                "**Facteurs dominants** : l'ancienneté (tenure) et le montant total "
                "facturé (TotalCharges) concentrent l'essentiel du pouvoir prédictif, "
                "suivis par la mensualité, le type de contrat et le type de connexion internet."
            )
        else:
            st.info("Ce modèle ne fournit pas de feature importance native.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Validation croisée ───────────────────────────────────────
    with tab5:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Validation croisée 5-fold")
        st.caption("Stabilité de l'AUC-ROC sur 5 découpages différents du jeu d'entraînement")

        if cv_results is not None:
            df_cv_sorted = cv_results.sort_values('AUC moyen', ascending=True)
            fig = go.Figure(go.Bar(
                x=df_cv_sorted['AUC moyen'], y=df_cv_sorted['Modèle'],
                orientation='h',
                error_x=dict(array=df_cv_sorted['Écart-type'], color=GRIS),
                marker=dict(
                    color=df_cv_sorted['AUC moyen'], colorscale=COLORSCALE_BLEU,
                    showscale=False, line=dict(color="rgba(0,0,0,0)")
                ),
                text=df_cv_sorted['AUC moyen'].apply(lambda x: f"{x:.4f}"),
                textposition="outside", textfont=dict(size=10.5, family=FONT),
                hovertemplate="<b>%{y}</b><br>AUC moyen %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(**plo(height=420))
            fig.update_xaxes(**ax(title_text="AUC-ROC moyen"))
            fig.update_yaxes(tickfont=dict(size=10.5), gridcolor="rgba(0,0,0,0)")
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Détail par modèle")
            st.dataframe(cv_results, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Résultats de validation croisée non disponibles.")
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# QUALITÉ & MONITORING
# ═══════════════════════════════════════════════════════════════════

elif page == "Qualité & Monitoring":

    st.markdown(f"""
    <div class="page-banner">
        <div>
            <div class="banner-title">Qualité & Monitoring</div>
            <div class="banner-desc">
                Robustesse du modèle, généralisation et suivi
                en conditions de production
            </div>
        </div>
        <div class="banner-badge">● Seuil optimisé {optimal_threshold:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Seuil de décision", "Courbe d'apprentissage", "Rapport Evidently"
    ])

    # ── Seuil de décision ────────────────────────────────────────
    with tab1:
        best_name = 'XGBoost Optimisé' if 'XGBoost Optimisé' in models else list(models.keys())[0]
        best_model = models[best_name]
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.01)
        f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
        recalls = [recall_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
        precisions = [precision_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f"### Optimisation du seuil - {best_name}")
        st.caption(
            "Par défaut, un modèle utilise un seuil de 0.50. Le seuil optimal "
            f"qui maximise le F1-score a été identifié à {optimal_threshold:.2f}"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1-score',
                                  line=dict(color=BLEU, width=2.5)))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, name='Recall',
                                  line=dict(color=ROUGE, width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, name='Precision',
                                  line=dict(color=VERT, width=2, dash='dash')))
        fig.add_vline(x=optimal_threshold, line_dash='dot', line_color=ORANGE,
                      annotation_text=f'Seuil optimal {optimal_threshold:.2f}',
                      annotation_font=dict(size=11, color=ORANGE, family=FONT))
        fig.add_vline(x=0.5, line_dash='dot', line_color=GRIS,
                      annotation_text='Seuil par défaut 0.50',
                      annotation_font=dict(size=10, color=GRIS, family=FONT),
                      annotation_position="bottom right")
        fig.update_layout(**plo(
            height=440,
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)")
        ))
        fig.update_xaxes(**ax(title_text="Seuil de décision"))
        fig.update_yaxes(**ay(title_text="Score"))
        chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        y_pred_05 = (y_proba >= 0.5).astype(int)
        y_pred_opt = (y_proba >= optimal_threshold).astype(int)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("##### Seuil 0.50 (défaut)")
            m1, m2, m3 = st.columns(3)
            m1.metric("F1-score", f"{f1_score(y_test, y_pred_05):.4f}")
            m2.metric("Recall", f"{recall_score(y_test, y_pred_05):.1%}")
            m3.metric("Precision", f"{precision_score(y_test, y_pred_05):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f"##### Seuil {optimal_threshold:.2f} (optimal)")
            m1, m2, m3 = st.columns(3)
            m1.metric("F1-score", f"{f1_score(y_test, y_pred_opt):.4f}")
            m2.metric("Recall", f"{recall_score(y_test, y_pred_opt):.1%}")
            m3.metric("Precision", f"{precision_score(y_test, y_pred_opt):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Courbe d'apprentissage ───────────────────────────────────
    with tab2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Capacité de généralisation")
        st.caption(
            "Comparaison des performances entre entraînement et validation : "
            "un écart faible indique l'absence de surapprentissage"
        )

        learning_curve_path = os.path.join(EXPORTS_DIR, '11_learning_curve.png')
        if os.path.exists(learning_curve_path):
            st.image(learning_curve_path, use_container_width=True)
        else:
            st.warning(
                "Image de la courbe d'apprentissage non trouvée. "
                "Générez-la depuis le notebook 02."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Rapport Evidently ─────────────────────────────────────────
    with tab3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Rapport de monitoring Evidently")
        st.caption(
            "Simulation d'un environnement de production : comparaison entre "
            "données d'entraînement (référence) et données de test (courant) "
            "pour détecter un éventuel data drift"
        )

        evidently_path = os.path.join(EXPORTS_DIR, 'evidently_report.html')
        if os.path.exists(evidently_path):
            with open(evidently_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.warning(
                "Rapport Evidently non trouvé. Générez-le depuis le "
                "notebook 02 (cellule Monitoring)."
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIMULATEUR
# ═══════════════════════════════════════════════════════════════════

elif page == "Simulateur":

    st.markdown(f"""
    <div class="page-banner">
        <div>
            <div class="banner-title">Simulateur de prédiction</div>
            <div class="banner-desc">
                Estimation du risque de churn pour un profil client donné
            </div>
        </div>
        <div class="banner-badge">● {sim_model_choice}</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### Profil client")
        st.caption("Renseignez les caractéristiques du client à évaluer")

        with st.expander("Informations démographiques", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Genre", ["Male", "Female"])
                senior = st.selectbox("Senior (65+)", ["No", "Yes"])
            with c2:
                partner = st.selectbox("Partenaire", ["Yes", "No"])
                dependents = st.selectbox("Dépendants", ["No", "Yes"])

        with st.expander("Services souscrits", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                phone = st.selectbox("Téléphone", ["Yes", "No"])
                multiple_lines = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes", "No internet service"])
            with c2:
                device_protection = st.selectbox("Protection appareil", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Support technique", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming films", ["No", "Yes", "No internet service"])

        with st.expander("Contrat & Facturation", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
                payment = st.selectbox(
                    "Mode de paiement",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
                )
            with c2:
                tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
                monthly = st.slider("Mensualité (€)", 18, 120, 65)
                total = st.slider("Total facturé (€)", 0, 9000, monthly * tenure)

        predict_btn = st.button("Lancer la prédiction", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if predict_btn:
            client = {
                'gender': 1 if gender == 'Male' else 0,
                'SeniorCitizen': 1 if senior == 'Yes' else 0,
                'Partner': 1 if partner == 'Yes' else 0,
                'Dependents': 1 if dependents == 'Yes' else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone == 'Yes' else 0,
                'PaperlessBilling': 1 if paperless == 'Yes' else 0,
                'MonthlyCharges': monthly,
                'TotalCharges': total,
                'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
                'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
                'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
                'InternetService_No': 1 if internet == 'No' else 0,
                'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
                'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
                'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
                'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
                'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
                'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
                'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
                'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
                'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
                'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
                'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
                'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
                'Contract_One year': 1 if contract == 'One year' else 0,
                'Contract_Two year': 1 if contract == 'Two year' else 0,
                'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
                'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
                'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
            }

            X_client = pd.DataFrame([client])[feature_names]
            X_input = scaler.transform(X_client)
            model = models[sim_model_choice]

            proba = model.predict_proba(X_input)[0][1]
            threshold_used = optimal_threshold if sim_use_optimal else 0.5
            prediction = int(proba >= threshold_used)

            if proba < 0.35:
                risk_level, color = "FAIBLE", VERT
            elif proba < 0.60:
                risk_level, color = "MODÉRÉ", ORANGE
            else:
                risk_level, color = "ÉLEVÉ", ROUGE

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;padding:1rem 0 0.5rem 0;">
                <div style="font-size:0.85rem;opacity:0.6;">Probabilité de churn</div>
                <div style="font-size:3rem;font-weight:800;margin:0.3rem 0;color:{color};">
                    {proba*100:.1f} %
                </div>
                <span class="risk-pill" style="background:{color}1a;color:{color};
                            font-size:0.9rem;padding:0.4rem 1.2rem;">
                    RISQUE {risk_level}
                </span>
                <div style="opacity:0.55;margin-top:0.75rem;font-size:0.82rem;">
                    Décision au seuil {threshold_used:.2f} :
                    {'Churn probable' if prediction else 'Client stable'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(46,125,50,0.12)'},
                        {'range': [35, 60], 'color': 'rgba(218,165,32,0.12)'},
                        {'range': [60, 100], 'color': 'rgba(193,0,42,0.12)'},
                    ],
                    'threshold': {'line': {'width': 3}, 'thickness': 0.8,
                                  'value': threshold_used * 100}
                }
            ))
            fig.update_layout(**plo(height=240, margin=dict(t=10, b=10)))
            chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("### Facteurs de risque détectés")

            risk_factors = []
            if tenure < 12:
                risk_factors.append(("Nouveau client", f"Ancienneté : {tenure} mois"))
            if contract == "Month-to-month":
                risk_factors.append(("Contrat mensuel", "Engagement faible"))
            if internet == "Fiber optic":
                risk_factors.append(("Fiber optic", "Service à risque élevé"))
            if online_security == "No":
                risk_factors.append(("Sans sécurité en ligne", "Facteur de risque"))
            if tech_support == "No":
                risk_factors.append(("Sans support technique", "Facteur de risque"))
            if payment == "Electronic check":
                risk_factors.append(("Chèque électronique", "Mode de paiement à risque"))
            if monthly > 80:
                risk_factors.append(("Mensualité élevée", f"{monthly} €/mois"))

            if risk_factors:
                for title, desc in risk_factors:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:0.5rem 0;border-bottom:1px solid rgba(128,128,128,0.12);">
                        <span style="font-weight:600;font-size:0.85rem;">{title}</span>
                        <span style="opacity:0.55;font-size:0.8rem;">{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Aucun facteur de risque majeur détecté pour ce profil.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center;padding:3.5rem 1rem;opacity:0.55;">
                Renseignez le profil client puis cliquez sur<br>
                <b>Lancer la prédiction</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)