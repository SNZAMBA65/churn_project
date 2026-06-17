# ============================================================
# DASHBOARD STREAMLIT - Prédiction du Churn Client
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, f1_score, recall_score,
    precision_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Churn Predictor - Telco",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    /* Fond général */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #16213e 100%);
        border-right: 1px solid #2d3561;
    }

    /* Titres */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Cartes métriques */
    .metric-card {
        background: linear-gradient(135deg, #1e2a45 0%, #2d3561 100%);
        border: 1px solid #3d4f7c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: 800;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.85em;
        color: #94a3b8;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 0.8em;
        color: #34d399;
        margin-top: 3px;
    }

    /* Badges */
    .badge-high {
        background: #dc2626;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-low {
        background: #16a34a;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }

    /* Plotly fond transparent */
    .js-plotly-plot { border-radius: 12px; }

    /* Divider */
    hr { border-color: #2d3561; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES DONNÉES ET MODÈLES (mis en cache)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('../exports/dataset_clean.csv')
    return df

@st.cache_data
def load_results():
    return pd.read_csv('../exports/resultats_modeles.csv')

@st.cache_resource
def load_models():
    models = {
        'Régression Logistique': joblib.load('../models/logistic_regression.pkl'),
        'Arbre de Décision': joblib.load('../models/decision_tree.pkl'),
        'Random Forest': joblib.load('../models/random_forest.pkl'),
        'Random Forest Optimisé': joblib.load('../models/random_forest_best.pkl'),
    }
    scaler = joblib.load('../models/scaler.pkl')
    feature_names = joblib.load('../models/feature_names.pkl')
    return models, scaler, feature_names

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
    scaler = joblib.load('../models/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)

    return X_test, X_test_scaled, y_test

df = load_data()
results = load_results()
models, scaler, feature_names = load_models()
X_test, X_test_scaled, y_test = prepare_test_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size:3em;'>📡</div>
        <div style='font-size:1.3em; font-weight:800; color:#60a5fa;'>
            Churn Predictor
        </div>
        <div style='font-size:0.8em; color:#64748b; margin-top:4px;'>
            Telco Customer Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🔍 Exploration", "📊 Modèles", "🎯 Simulateur"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div style='font-size:0.75em; color:#475569; padding:10px;'>
        <b style='color:#64748b;'>Dataset</b><br>
        Telco Customer Churn<br>
        7 043 clients · 20 variables<br><br>
        <b style='color:#64748b;'>Modèle retenu</b><br>
        Random Forest Optimisé<br>
        AUC-ROC : 0.8420
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 1 : ACCUEIL
# ============================================================
if page == "🏠 Accueil":

    st.markdown("""
    <h1 style='text-align:center; font-size:2.5em; margin-bottom:0;'>
        📡 Prédiction du Churn Client
    </h1>
    <p style='text-align:center; color:#64748b; font-size:1.1em; margin-top:8px;'>
        Analyse et modélisation du risque de résiliation — Telco Customer Dataset
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # KPIs principaux
    total = len(df)
    churned = (df['Churn'] == 'Yes').sum()
    retained = total - churned
    churn_rate = churned / total * 100
    avg_monthly = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, str(total), "Clients analysés", ""),
        (col2, f"{churn_rate:.1f}%", "Taux de churn", "⚠️ Déséquilibre détecté"),
        (col3, str(churned), "Clients perdus", ""),
        (col4, f"{avg_monthly:.0f}€", "Mensualité moyenne", ""),
        (col5, f"{avg_tenure:.0f} mois", "Ancienneté moyenne", ""),
    ]
    for col, val, label, delta in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
                <div class='metric-delta'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Résumé du projet
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 🎯 Objectif du projet")
        st.markdown("""
        <div style='background:#1e2a45; border-left:4px solid #60a5fa;
                    border-radius:8px; padding:18px; color:#cbd5e1; line-height:1.8;'>
        Développer un modèle de Machine Learning capable de prédire le churn
        des clients d'un opérateur télécom, afin d'anticiper les résiliations
        et de cibler les actions de rétention.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔬 Approche méthodologique")
        steps = [
            ("1", "Collecte & nettoyage", "Dataset Telco Kaggle, 7043 clients"),
            ("2", "Analyse exploratoire", "EDA avec Seaborn & Matplotlib"),
            ("3", "Modélisation ML", "Régression Logistique, Arbre, Random Forest"),
            ("4", "Optimisation", "GridSearchCV, validation croisée 5-fold"),
            ("5", "Dashboard", "Visualisation interactive Streamlit"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:8px 0;
                        background:#1a1f2e; border-radius:8px; padding:10px 15px;'>
                <div style='background:#3b82f6; color:white; border-radius:50%;
                            width:28px; height:28px; display:flex; align-items:center;
                            justify-content:center; font-weight:800; margin-right:12px;
                            flex-shrink:0;'>{num}</div>
                <div>
                    <div style='color:#e2e8f0; font-weight:600;'>{title}</div>
                    <div style='color:#64748b; font-size:0.85em;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🏆 Modèle retenu")
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e2a45,#2d3561);
                    border:2px solid #3b82f6; border-radius:12px; padding:20px;'>
            <div style='text-align:center; margin-bottom:15px;'>
                <span style='font-size:1.1em; color:#60a5fa; font-weight:700;'>
                    🌲 Random Forest Optimisé
                </span>
            </div>
        """, unsafe_allow_html=True)

        metrics = [
            ("AUC-ROC", "0.8420", "#60a5fa"),
            ("Accuracy", "76%", "#34d399"),
            ("Recall (Churné)", "79%", "#f59e0b"),
            ("F1-score", "0.64", "#a78bfa"),
        ]
        for label, val, color in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding:8px 0; border-bottom:1px solid #2d3561;'>
                <span style='color:#94a3b8;'>{label}</span>
                <span style='color:{color}; font-weight:700;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style='margin-top:15px; padding:10px; background:#0f1117;
                        border-radius:8px; text-align:center;'>
                <span style='color:#64748b; font-size:0.85em;'>
                    Hyperparamètres : n_estimators=200,
                    max_depth=10, min_samples_split=5
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Fichiers produits")
        files = [
            ("📓", "01_eda_preparation.ipynb", "EDA complète"),
            ("📓", "02_modelisation_validation.ipynb", "Modélisation"),
            ("🤖", "random_forest_best.pkl", "Modèle final"),
            ("📊", "resultats_modeles.csv", "Comparaison"),
        ]
        for icon, name, desc in files:
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:6px 0;
                        background:#1a1f2e; border-radius:6px; padding:8px 12px;'>
                <span style='margin-right:10px;'>{icon}</span>
                <div>
                    <div style='color:#e2e8f0; font-size:0.9em;'>{name}</div>
                    <div style='color:#64748b; font-size:0.78em;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 2 : EXPLORATION DES DONNÉES
# ============================================================
elif page == "🔍 Exploration":

    st.markdown("## 🔍 Exploration des Données")
    st.markdown(
        "<p style='color:#64748b;'>Analyse exploratoire du dataset Telco Customer Churn</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution", "👥 Démographie", "📋 Contrat & Services", "📈 Variables numériques"
    ])

    # ---- TAB 1 : Distribution du churn ----
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(go.Bar(
                x=churn_counts.index,
                y=churn_counts.values,
                marker_color=['#2ecc71', '#e74c3c'],
                text=churn_counts.values,
                textposition='outside',
                textfont=dict(size=16, color='white')
            ))
            fig.update_layout(
                title='Distribution du Churn',
                paper_bgcolor='#1e2a45',
                plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'),
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Pie(
                labels=churn_counts.index,
                values=churn_counts.values,
                marker_colors=['#2ecc71', '#e74c3c'],
                hole=0.45,
                textinfo='label+percent',
                textfont_size=14
            ))
            fig.update_layout(
                title='Répartition du Churn (%)',
                paper_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'),
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⚠️ Dataset déséquilibré : 73.5% de clients retenus vs 26.5% churné. "
            "Le paramètre class_weight='balanced' a été utilisé pour compenser."
        )

    # ---- TAB 2 : Démographie ----
    with tab2:
        demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        col1, col2 = st.columns(2)

        for i, col_name in enumerate(demo_cols):
            churn_rate = df.groupby(col_name)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_rate.columns = [col_name, 'Taux de churn (%)']

            fig = px.bar(
                churn_rate, x=col_name, y='Taux de churn (%)',
                color='Taux de churn (%)',
                color_continuous_scale='RdYlGn_r',
                text=churn_rate['Taux de churn (%)'].apply(lambda x: f'{x:.1f}%'),
                title=f'Churn par {col_name}'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), showlegend=False,
                coloraxis_showscale=False, height=350
            )
            if i % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='background:#1e2a45; border-left:4px solid #f59e0b;
                    border-radius:8px; padding:15px; color:#cbd5e1;'>
        <b>Observations :</b> Le genre n'influence pas le churn (26.9% vs 26.2%).
        Les seniors churne à 41.7%, les clients sans partenaire à 33.0%
        et sans dépendants à 31.3%.
        </div>
        """, unsafe_allow_html=True)

    # ---- TAB 3 : Contrat & Services ----
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            churn_contract = df.groupby('Contract')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_contract.columns = ['Contract', 'Taux (%)']
            fig = px.bar(
                churn_contract, x='Contract', y='Taux (%)',
                color='Taux (%)', color_continuous_scale='RdYlGn_r',
                text=churn_contract['Taux (%)'].apply(lambda x: f'{x:.1f}%'),
                title='Taux de churn par type de contrat'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), coloraxis_showscale=False, height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_box = df.copy()
            df_box['tenure_group'] = pd.cut(
                df_box['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-12 mois', '12-24 mois', '24-48 mois', '48-72 mois']
            )
            churn_tenure = df_box.groupby('tenure_group', observed=True)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_tenure.columns = ['Ancienneté', 'Taux (%)']
            fig = px.bar(
                churn_tenure, x='Ancienneté', y='Taux (%)',
                color='Taux (%)', color_continuous_scale='RdYlGn_r',
                text=churn_tenure['Taux (%)'].apply(lambda x: f'{x:.1f}%'),
                title="Taux de churn par ancienneté"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), coloraxis_showscale=False, height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        # Services
        st.markdown("#### Impact des services sur le churn")
        service_cols = ['InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
        col1, col2 = st.columns(2)
        for i, col_name in enumerate(service_cols):
            churn_rate = df.groupby(col_name)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            churn_rate.columns = [col_name, 'Taux (%)']
            fig = px.bar(
                churn_rate, x=col_name, y='Taux (%)',
                color='Taux (%)', color_continuous_scale='RdYlGn_r',
                text=churn_rate['Taux (%)'].apply(lambda x: f'{x:.1f}%'),
                title=f'{col_name}'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), coloraxis_showscale=False, height=320
            )
            if i % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 4 : Variables numériques ----
    with tab4:
        num_var = st.selectbox(
            "Choisir une variable numérique",
            ['tenure', 'MonthlyCharges', 'TotalCharges']
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x=num_var, color='Churn',
                color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'},
                barmode='overlay', opacity=0.7,
                title=f'Distribution de {num_var} par Churn',
                labels={'Churn': 'Churn'}
            )
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df, x='Churn', y=num_var, color='Churn',
                color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'},
                title=f'Boxplot de {num_var} par Churn'
            )
            fig.update_layout(
                paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'), height=380, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Stats descriptives
        st.markdown("#### Statistiques descriptives")
        stats = df.groupby('Churn')[num_var].describe().round(2)
        st.dataframe(stats, use_container_width=True)

# ============================================================
# PAGE 3 : PERFORMANCE DES MODÈLES
# ============================================================
elif page == "📊 Modèles":

    st.markdown("## 📊 Performance des Modèles")
    st.markdown(
        "<p style='color:#64748b;'>Comparaison et évaluation des modèles entraînés</p>",
        unsafe_allow_html=True
    )
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Comparaison", "📈 Courbes ROC", "🔢 Matrices de confusion", "🌲 Feature Importance"
    ])

    # ---- TAB 1 : Comparaison ----
    with tab1:
        st.markdown("#### Tableau comparatif des modèles")
        st.dataframe(
            results.style
            .highlight_max(subset=['Accuracy','Recall (Churné)','F1 (Churné)','AUC-ROC'],
                          color='#1e3a5f')
            .format({
                'Accuracy': '{:.2%}',
                'Precision (Churné)': '{:.2%}',
                'Recall (Churné)': '{:.2%}',
                'F1 (Churné)': '{:.4f}',
                'AUC-ROC': '{:.4f}'
            }),
            use_container_width=True, height=180
        )

        st.markdown("#### Visualisation comparative")
        metrics_to_plot = ['Accuracy', 'Precision (Churné)', 'Recall (Churné)', 'F1 (Churné)', 'AUC-ROC']
        colors = ['#3b82f6', '#f59e0b', '#a78bfa', '#2ecc71']

        fig = go.Figure()
        for i, row in results.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics_to_plot],
                theta=metrics_to_plot,
                fill='toself',
                name=row['Modèle'],
                line=dict(color=colors[i], width=2),
                opacity=0.6
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.5, 1]),
                bgcolor='#1a1f2e'
            ),
            paper_bgcolor='#1e2a45',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='#1a1f2e', bordercolor='#3d4f7c'),
            height=450,
            title='Radar des performances par modèle'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Barplot métriques
        metric_choice = st.selectbox(
            "Comparer les modèles sur une métrique",
            metrics_to_plot
        )
        fig = px.bar(
            results, x='Modèle', y=metric_choice,
            color='Modèle',
            color_discrete_sequence=colors,
            text=results[metric_choice].apply(lambda x: f'{x:.4f}'),
            title=f'Comparaison - {metric_choice}'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
            font=dict(color='#e2e8f0'), showlegend=False, height=380
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 2 : Courbes ROC ----
    with tab2:
        model_names = list(models.keys())
        colors_roc = ['#3b82f6', '#f59e0b', '#a78bfa', '#2ecc71']

        fig = go.Figure()
        for i, (name, model) in enumerate(models.items()):
            if 'Logistique' in name:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC={auc:.4f})',
                line=dict(color=colors_roc[i], width=2.5)
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aléatoire (AUC=0.5)',
            line=dict(color='#64748b', width=1.5, dash='dash')
        ))
        fig.update_layout(
            title='Courbes ROC - Comparaison des Modèles',
            xaxis_title='Taux de Faux Positifs',
            yaxis_title='Taux de Vrais Positifs (Recall)',
            paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='#1a1f2e', bordercolor='#3d4f7c'),
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='background:#1e2a45; border-left:4px solid #60a5fa;
                    border-radius:8px; padding:15px; color:#cbd5e1;'>
        <b>Lecture :</b> Plus la courbe est proche du coin supérieur gauche,
        meilleur est le modèle. L'AUC mesure l'aire sous la courbe
        (1.0 = parfait, 0.5 = aléatoire). Tous nos modèles dépassent 0.83.
        </div>
        """, unsafe_allow_html=True)

    # ---- TAB 3 : Matrices de confusion ----
    with tab3:
        model_choice = st.selectbox("Choisir un modèle", list(models.keys()))

        model = models[model_choice]
        if 'Logistique' in model_choice:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        col1, col2 = st.columns([1, 1.2])

        with col1:
            fig = px.imshow(
                cm,
                labels=dict(x='Prédit', y='Réel', color='Clients'),
                x=['Resté', 'Churné'],
                y=['Resté', 'Churné'],
                color_continuous_scale='Blues',
                text_auto=True,
                title=f'Matrice de confusion - {model_choice}'
            )
            fig.update_traces(textfont_size=20)
            fig.update_layout(
                paper_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Détail des prédictions")
            details = [
                ("✅ Vrais Négatifs (TN)", tn,
                 "Non-churners correctement identifiés", "#2ecc71"),
                ("⚠️ Faux Positifs (FP)", fp,
                 "Non-churners classés comme churners", "#f59e0b"),
                ("❌ Faux Négatifs (FN)", fn,
                 "Churners non détectés (cas critique)", "#e74c3c"),
                ("✅ Vrais Positifs (TP)", tp,
                 "Churners correctement détectés", "#3b82f6"),
            ]
            for label, val, desc, color in details:
                st.markdown(f"""
                <div style='background:#1a1f2e; border-left:4px solid {color};
                            border-radius:8px; padding:12px 15px; margin:8px 0;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#e2e8f0; font-weight:600;'>{label}</span>
                        <span style='color:{color}; font-weight:800;
                                     font-size:1.3em;'>{val}</span>
                    </div>
                    <div style='color:#64748b; font-size:0.82em; margin-top:3px;'>
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Métriques")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            for label, val in [
                ("Accuracy", f"{acc:.2%}"),
                ("Precision", f"{prec:.2%}"),
                ("Recall", f"{rec:.2%}"),
                ("F1-score", f"{f1:.4f}")
            ]:
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between;
                            padding:6px 0; border-bottom:1px solid #2d3561;'>
                    <span style='color:#94a3b8;'>{label}</span>
                    <span style='color:#60a5fa; font-weight:700;'>{val}</span>
                </div>
                """, unsafe_allow_html=True)

    # ---- TAB 4 : Feature Importance ----
    with tab4:
        rf_best = models['Random Forest Optimisé']
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_best.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)

        fig = px.bar(
            feature_imp.sort_values('Importance'),
            x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='RdYlGn',
            text=feature_imp.sort_values('Importance')['Importance'].apply(
                lambda x: f'{x:.3f}'
            ),
            title='Top 15 Features - Random Forest Optimisé'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            paper_bgcolor='#1e2a45', plot_bgcolor='#1e2a45',
            font=dict(color='#e2e8f0'), coloraxis_showscale=False,
            height=520, yaxis_title='', xaxis_title='Importance'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='background:#1e2a45; border-left:4px solid #2ecc71;
                    border-radius:8px; padding:15px; color:#cbd5e1;'>
        <b>Top 5 facteurs de churn :</b><br>
        1. <b>tenure</b> (0.175) — Ancienneté : les nouveaux clients churne davantage<br>
        2. <b>TotalCharges</b> (0.143) — Montant total facturé<br>
        3. <b>MonthlyCharges</b> (0.097) — Mensualité élevée = risque accru<br>
        4. <b>Contract_Two year</b> (0.094) — Engagement long = fidélité<br>
        5. <b>InternetService_Fiber optic</b> (0.070) — Service à risque élevé
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 4 : SIMULATEUR DE PRÉDICTION
# ============================================================
elif page == "🎯 Simulateur":

    st.markdown("## 🎯 Simulateur de Prédiction")
    st.markdown(
        "<p style='color:#64748b;'>Prédisez le risque de churn pour un client fictif</p>",
        unsafe_allow_html=True
    )
    st.divider()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### 📋 Profil client")

        with st.expander("👤 Informations démographiques", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Genre", ["Male", "Female"])
                senior = st.selectbox("Senior (65+)", ["No", "Yes"])
            with c2:
                partner = st.selectbox("Partenaire", ["Yes", "No"])
                dependents = st.selectbox("Dépendants", ["No", "Yes"])

        with st.expander("📱 Services souscrits", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                phone = st.selectbox("Téléphone", ["Yes", "No"])
                multiple_lines = st.selectbox(
                    "Lignes multiples", ["No", "Yes", "No phone service"]
                )
                internet = st.selectbox(
                    "Internet", ["Fiber optic", "DSL", "No"]
                )
                online_security = st.selectbox(
                    "Sécurité en ligne", ["No", "Yes", "No internet service"]
                )
                online_backup = st.selectbox(
                    "Sauvegarde en ligne", ["No", "Yes", "No internet service"]
                )
            with c2:
                device_protection = st.selectbox(
                    "Protection appareil", ["No", "Yes", "No internet service"]
                )
                tech_support = st.selectbox(
                    "Support technique", ["No", "Yes", "No internet service"]
                )
                streaming_tv = st.selectbox(
                    "Streaming TV", ["No", "Yes", "No internet service"]
                )
                streaming_movies = st.selectbox(
                    "Streaming films", ["No", "Yes", "No internet service"]
                )

        with st.expander("💳 Contrat & Facturation", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                contract = st.selectbox(
                    "Type de contrat",
                    ["Month-to-month", "One year", "Two year"]
                )
                paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
                payment = st.selectbox(
                    "Mode de paiement",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"]
                )
            with c2:
                tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
                monthly = st.slider("Mensualité (€)", 18, 120, 65)
                total = st.slider("Total facturé (€)", 0, 9000, monthly * tenure)

        model_choice = st.selectbox(
            "Modèle de prédiction",
            list(models.keys()),
            index=3
        )

        predict_btn = st.button("🔮 Lancer la prédiction", use_container_width=True)

    # ---- RÉSULTAT ----
    with col_result:
        st.markdown("### 📊 Résultat de la prédiction")

        if predict_btn:
            # Construction du vecteur client
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
            model = models[model_choice]

            if 'Logistique' in model_choice:
                X_input = scaler.transform(X_client)
            else:
                X_input = X_client

            proba = model.predict_proba(X_input)[0][1]
            prediction = model.predict(X_input)[0]

            # Niveau de risque
            if proba < 0.35:
                risk_level = "FAIBLE"
                risk_color = "#2ecc71"
                risk_icon = "✅"
                risk_msg = "Ce client présente un faible risque de résiliation."
            elif proba < 0.60:
                risk_level = "MODÉRÉ"
                risk_color = "#f59e0b"
                risk_icon = "⚠️"
                risk_msg = "Ce client présente un risque modéré. Une action préventive est conseillée."
            else:
                risk_level = "ÉLEVÉ"
                risk_color = "#e74c3c"
                risk_icon = "🚨"
                risk_msg = "Ce client est très susceptible de résilier. Intervention urgente recommandée."

            # Affichage résultat principal
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e2a45,#2d3561);
                        border:2px solid {risk_color}; border-radius:16px;
                        padding:30px; text-align:center; margin-bottom:20px;'>
                <div style='font-size:3.5em;'>{risk_icon}</div>
                <div style='font-size:1em; color:#94a3b8; margin:8px 0;'>
                    Probabilité de churn
                </div>
                <div style='font-size:3.8em; font-weight:900; color:{risk_color};'>
                    {proba*100:.1f}%
                </div>
                <div style='background:{risk_color}; color:white; border-radius:20px;
                            padding:6px 20px; display:inline-block; font-weight:700;
                            font-size:1.1em; margin-top:10px;'>
                    RISQUE {risk_level}
                </div>
                <div style='color:#94a3b8; margin-top:15px; font-size:0.9em;'>
                    {risk_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge de probabilité
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': '%', 'font': {'size': 40, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#e2e8f0'},
                    'bar': {'color': risk_color, 'thickness': 0.3},
                    'bgcolor': '#1a1f2e',
                    'bordercolor': '#3d4f7c',
                    'steps': [
                        {'range': [0, 35], 'color': '#1a3a2a'},
                        {'range': [35, 60], 'color': '#3a2e1a'},
                        {'range': [60, 100], 'color': '#3a1a1a'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.8,
                        'value': proba * 100
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='#1e2a45',
                font=dict(color='#e2e8f0'),
                height=280,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Facteurs de risque du profil
            st.markdown("#### 🔍 Facteurs de risque détectés")
            risk_factors = []
            if tenure < 12:
                risk_factors.append(("🕐", "Nouveau client", f"Ancienneté : {tenure} mois"))
            if contract == "Month-to-month":
                risk_factors.append(("📄", "Contrat mensuel", "Engagement faible"))
            if internet == "Fiber optic":
                risk_factors.append(("🌐", "Fiber optic", "Service à risque élevé"))
            if online_security == "No":
                risk_factors.append(("🔓", "Sans sécurité en ligne", "Facteur de risque"))
            if tech_support == "No":
                risk_factors.append(("🛠️", "Sans support technique", "Facteur de risque"))
            if payment == "Electronic check":
                risk_factors.append(("💳", "Chèque électronique", "Mode de paiement à risque"))
            if monthly > 80:
                risk_factors.append(("💰", "Mensualité élevée", f"{monthly}€/mois"))

            if risk_factors:
                for icon, title, desc in risk_factors:
                    st.markdown(f"""
                    <div style='background:#1a1f2e; border-left:3px solid #e74c3c;
                                border-radius:6px; padding:10px 14px; margin:5px 0;
                                display:flex; align-items:center;'>
                        <span style='margin-right:10px; font-size:1.2em;'>{icon}</span>
                        <div>
                            <span style='color:#e2e8f0; font-weight:600;'>{title}</span>
                            <span style='color:#64748b; font-size:0.85em;
                                         margin-left:8px;'>{desc}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Aucun facteur de risque majeur détecté pour ce profil.")

            # Recommandations
            st.markdown("#### 💡 Recommandations")
            if proba >= 0.35:
                recs = []
                if contract == "Month-to-month":
                    recs.append("Proposer une offre de passage en contrat annuel ou bi-annuel")
                if internet == "Fiber optic":
                    recs.append("Vérifier la qualité de service Fiber optic et proposer un geste commercial")
                if online_security == "No":
                    recs.append("Offrir un mois de sécurité en ligne gratuit")
                if tech_support == "No":
                    recs.append("Inclure le support technique dans le prochain renouvellement")
                if tenure < 12:
                    recs.append("Mettre en place un programme de fidélisation nouveaux clients")
                if not recs:
                    recs.append("Contacter le client pour un bilan de satisfaction")

                for rec in recs:
                    st.markdown(f"""
                    <div style='background:#1a2e1a; border-left:3px solid #2ecc71;
                                border-radius:6px; padding:10px 14px; margin:5px 0;
                                color:#a7f3d0;'>
                        💡 {rec}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#1a2e1a; border-left:3px solid #2ecc71;
                            border-radius:6px; padding:15px; color:#a7f3d0;'>
                    ✅ Ce client est stable. Continuer le suivi de routine.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background:#1e2a45; border:2px dashed #3d4f7c;
                        border-radius:16px; padding:60px 30px; text-align:center;'>
                <div style='font-size:3em; margin-bottom:15px;'>🔮</div>
                <div style='color:#64748b; font-size:1.1em;'>
                    Remplissez le profil client et cliquez sur<br>
                    <b style='color:#60a5fa;'>Lancer la prédiction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)