"""
Dashboard avanzado para comparar optimizadores, activaciones y arquitecturas
de una MLP entrenada con PCA.

Ejecución:
    streamlit run dashboard.py
"""

import os
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =================== CONFIGURACIÓN ===================
st.set_page_config(
    page_title="Análisis de Optimizadores y Arquitecturas - MLP con PCA",
    layout="wide"
)

# Tema oscuro elegante
st.markdown("""
<style>
body { background-color: #0E1117; color: #E0E0E0; }
.stApp { background-color: #0E1117; }
h1, h2, h3, h4, h5, h6 { color: #E0E0E0; font-family: 'Inter', sans-serif; }
.stDataFrame { background-color: #1E2228 !important; }
</style>
""", unsafe_allow_html=True)


# =================== FUNCIÓN DE CARGA ===================
def load_results_github(repo_owner="dev21-bit", repo_name="models"):
    """
    Carga automáticamente todos los archivos JSON de la raíz de un repositorio público de GitHub.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
    try:
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()

        json_files = [f for f in files if f["name"].endswith(".json")]
        if not json_files:
            st.warning("No se encontraron archivos JSON en la raíz del repositorio.")
            return pd.DataFrame()

        data = []
        for file in json_files:
            raw_url = file["download_url"]
            try:
                res = requests.get(raw_url)
                res.raise_for_status()
                j = json.loads(res.text)
                if isinstance(j.get("arch"), list):
                    j["arch"] = str(j["arch"])
                # Asegurar campos faltantes
                for key in ["train_loss", "val_f1"]:
                    if key not in j or not isinstance(j[key], list):
                        j[key] = []
                data.append(j)
            except Exception as e:
                st.error(f"Error leyendo {file['name']}: {e}")

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error al acceder al repositorio: {e}")
        return pd.DataFrame()


# =================== INTERFAZ PRINCIPAL ===================
st.title("Comparación de Optimizadores, Activaciones y Arquitecturas en una MLP (con PCA)")
st.caption("Visualización interactiva basada en los experimentos realizados sobre el dataset de estados mentales.")

df = load_results_github()
if df.empty:
    st.stop()

# =================== FILTROS ===================
optims = sorted(df["optimizer"].dropna().unique())
acts = sorted(df["activation"].dropna().unique())
archs = sorted(df["arch"].dropna().unique())

cols = st.columns(3)
optims_sel = cols[0].multiselect("Optimizadores:", optims, default=optims)
acts_sel = cols[1].multiselect("Activaciones:", acts, default=acts)
archs_sel = cols[2].multiselect("Arquitecturas:", archs, default=archs)

filtered = df[
    df["optimizer"].isin(optims_sel)
    & df["activation"].isin(acts_sel)
    & df["arch"].isin(archs_sel)
].copy()

if filtered.empty:
    st.warning("No hay resultados que coincidan con los filtros seleccionados.")
    st.stop()

# =================== TABLA RESUMEN ===================
st.subheader("Resultados Globales del Conjunto de Prueba")
st.dataframe(
    filtered[["optimizer", "activation", "arch", "test_acc", "test_f1", "runtime"]]
    .sort_values("test_f1", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)

# =================== VISUALIZACIONES ===================
st.markdown("---")
st.subheader("Comparaciones entre Combinaciones")

col1, col2 = st.columns(2)

with col1:
    fig_f1 = px.bar(
        filtered,
        x="optimizer",
        y="test_f1",
        color="activation",
        facet_col="arch",
        title="F1-score por combinación",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_f1.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_f1, use_container_width=True)

with col2:
    fig_acc = px.bar(
        filtered,
        x="optimizer",
        y="test_acc",
        color="activation",
        facet_col="arch",
        title="Accuracy por combinación",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_acc.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_acc, use_container_width=True)

# =================== DISPERSIÓN F1 VS ACC ===================
st.markdown("---")
st.subheader("Relación entre F1-score y Accuracy")

fig_scatter = px.scatter(
    filtered,
    x="test_acc",
    y="test_f1",
    color="optimizer",
    symbol="activation",
    size="runtime",
    hover_data=["arch"],
    title="Dispersión de desempeño entre Accuracy y F1",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
st.plotly_chart(fig_scatter, use_container_width=True)

# =================== HEATMAP ===================
st.markdown("---")
st.subheader("Mapa de calor: F1 promedio por Optimizador y Activación")

heatmap_data = (
    filtered.groupby(["optimizer", "activation"])["test_f1"]
    .mean()
    .reset_index()
    .pivot(index="activation", columns="optimizer", values="test_f1")
)

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=".2f",
    color_continuous_scale="viridis",
    title="F1-score promedio por combinación"
)
fig_heatmap.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
st.plotly_chart(fig_heatmap, use_container_width=True)

# =================== CURVAS DE ENTRENAMIENTO ===================
st.markdown("---")
st.subheader("Curvas de Entrenamiento Individuales")

filtered["combo"] = filtered.apply(lambda r: f"{r['optimizer']} | {r['activation']} | {r['arch']}", axis=1)
selected = st.selectbox("Selecciona una combinación:", filtered["combo"])

row = filtered.loc[filtered["combo"] == selected].iloc[0]
if row["train_loss"] and row["val_f1"]:
    epochs = list(range(1, len(row["train_loss"]) + 1))
    df_curve = pd.DataFrame({
        "Época": epochs,
        "Pérdida (train)": row["train_loss"],
        "F1 (val)": row["val_f1"]
    })
    fig_curve = px.line(
        df_curve,
        x="Época",
        y=["Pérdida (train)", "F1 (val)"],
        markers=True,
        title=f"Evolución del entrenamiento: {selected}",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_curve.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_curve, use_container_width=True)
else:
    st.warning("No se encontraron curvas de entrenamiento para esta combinación.")

# =================== PROMEDIOS GLOBALES ===================
st.markdown("---")
st.subheader("Pérdida y F1 Promedio Global")

valid_rows = filtered[filtered["train_loss"].apply(lambda x: len(x) > 0)]
if not valid_rows.empty:
    avg_loss = valid_rows["train_loss"].apply(lambda x: pd.Series(x)).mean(axis=0)
    avg_f1 = valid_rows["val_f1"].apply(lambda x: pd.Series(x)).mean(axis=0)
    epochs = list(range(1, len(avg_loss) + 1))

    fig_global = go.Figure()
    fig_global.add_trace(go.Scatter(x=epochs, y=avg_loss, mode="lines+markers", name="Pérdida (Promedio)"))
    fig_global.add_trace(go.Scatter(x=epochs, y=avg_f1, mode="lines+markers", name="F1 (Promedio)"))
    fig_global.update_layout(
        title="Evolución promedio global de entrenamiento",
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117"
    )
    st.plotly_chart(fig_global, use_container_width=True)
else:
    st.info("No hay curvas suficientes para calcular promedios globales.")

# =================== INTERPRETACIÓN FINAL ===================
st.markdown("---")
st.subheader("Interpretación Automática")

best = filtered.loc[filtered["test_f1"].idxmax()]
st.info(
    f"El mejor modelo fue entrenado con {best['optimizer']}, activación {best['activation']} "
    f"y arquitectura {best['arch']}, alcanzando un F1 = {best['test_f1']:.3f} "
    f"y Accuracy = {best['test_acc']:.3f}."
)
