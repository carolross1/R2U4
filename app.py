import os
import io
import glob
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from fpdf import FPDF

# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------
st.set_page_config(page_title="Habilidades Blandas — Análisis No Supervisado",
                    layout="wide")

# ---------------------------------------------------------------------
# Paleta de colores (usada en CSS, gráficas, Excel y PDF)
# ---------------------------------------------------------------------
COLOR_PRIMARIO = "#BDA8FF"
COLOR_PRIMARIO_CLARO = "#C3B4F0"
COLOR_FONDO_SUAVE = "#F5F1FC"
COLOR_BORDE = "#E5DBF8"
COLOR_TEXTO = "#3F3A56"

# Paleta semántica (alto/medio/bajo, óptimo/aceptable/necesita mejora) — tonos
# claros y suaves, en línea con el lila principal, evitando colores muy saturados
COLOR_ALTO = "#8C7AE6"     # lila (misma familia que la marca)
COLOR_MEDIO = "#ABDFFF"    # azul suave, contraste cálido
COLOR_BAJO = "#E08CA2"     # rosa empolvado, contraste suave


def hex_a_rgb(color_hex):
    color_hex = color_hex.lstrip("#")
    return tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))

MODELOS_DIR = "modelos"
CACHE_DIR = "cache"
os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PESOS = {
    "Comunicación": {
        "prefijo": "com",
        "subcolumnas": {
            "com_claridad_expresion": 0.25, "com_escucha_activa": 0.25,
            "com_comunicacion_escrita": 0.20, "com_retroalimentacion": 0.15,
            "com_adaptacion_audiencia": 0.15,
        },
    },
    "Colaboración": {
        "prefijo": "col",
        "subcolumnas": {
            "col_trabajo_equipo": 0.30, "col_cumplimiento_compromisos": 0.25,
            "col_disposicion_ayudar": 0.20, "col_contribucion_grupal": 0.25,
        },
    },
    "Liderazgo": {
        "prefijo": "lid",
        "subcolumnas": {
            "lid_toma_decisiones": 0.25, "lid_motivacion_equipo": 0.25,
            "lid_delegacion": 0.20, "lid_responsabilidad": 0.15,
            "lid_vision_estrategica": 0.15,
        },
    },
    "Resolución de Problemas": {
        "prefijo": "res",
        "subcolumnas": {
            "res_analisis_situacion": 0.25, "res_pensamiento_critico": 0.25,
            "res_creatividad_soluciones": 0.20, "res_efectividad_solucion": 0.20,
            "res_tiempo_resolucion": 0.10,
        },
    },
}

COLS_META = ["id_empleado", "nombre_completo", "area", "departamento", "fecha_evaluacion"]


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def hash_key(df: pd.DataFrame, habilidades: list, extra: str = "") -> str:
    """Llave estable para cachear resultados según datos + selección + modelo elegido."""
    m = hashlib.sha256()
    m.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    m.update("|".join(sorted(habilidades)).encode())
    m.update(extra.encode())
    return m.hexdigest()[:16]


def columnas_de(habilidades):
    cols = []
    for h in habilidades:
        cols += list(PESOS[h]["subcolumnas"].keys())
        cols.append(f"{PESOS[h]['prefijo']}_score")
    return cols


@st.cache_data(show_spinner=False)
def cargar_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def clasificar(valor):
    if valor >= 8:
        return "Óptimo"
    elif valor >= 6:
        return "Aceptable"
    return "Necesita mejora"


def estadistica_propia(df, habilidades):
    """'Algoritmo propio' (no ML): clasifica por umbrales fijos y resume."""
    filas = []
    for h in habilidades:
        col_score = f"{PESOS[h]['prefijo']}_score"
        clases = df[col_score].apply(clasificar)
        conteo = clases.value_counts().reindex(
            ["Óptimo", "Aceptable", "Necesita mejora"]).fillna(0).astype(int)
        filas.append({
            "Habilidad": h,
            "Promedio": round(df[col_score].mean(), 2),
            "Óptimo": conteo["Óptimo"],
            "Aceptable": conteo["Aceptable"],
            "Necesita mejora": conteo["Necesita mejora"],
        })
    return pd.DataFrame(filas)


def mejor_lider_por_area(df):
    if "lid_score" not in df.columns:
        return None
    idx = df.groupby("area")["lid_score"].idxmax()
    return df.loc[idx, ["area", "nombre_completo", "departamento", "lid_score"]] \
        .sort_values("lid_score", ascending=False).reset_index(drop=True)


def entrenar_modelo(df, habilidades):
    """Crea un modelo K-Means en memoria (no lo guarda en disco)."""
    score_cols = [f"{PESOS[h]['prefijo']}_score" for h in habilidades]
    X = df[score_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(Xs)
    return {"scaler": scaler, "kmeans": kmeans, "score_cols": score_cols, "habilidades": habilidades}


def cargar_modelo_archivo(nombre_archivo):
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    return joblib.load(ruta)


def predecir_con_modelo(paquete, df):
    score_cols = paquete["score_cols"]
    scaler, kmeans = paquete["scaler"], paquete["kmeans"]
    X = df[score_cols].values
    Xs = scaler.transform(X)
    labels = kmeans.predict(Xs)

    # Ordenar los 3 centroides DEL MODELO (no de los datos actuales) para que
    # la clasificación sea consistente aunque la nueva carga de datos no tenga
    # empleados en alguno de los 3 grupos (ej. una media muy alta o muy baja
    # que agrupe todo en 1 o 2 clusters nada más).
    centros_originales = scaler.inverse_transform(kmeans.cluster_centers_)
    promedio_centro = centros_originales.mean(axis=1)
    orden = np.argsort(promedio_centro)  # de menor a mayor desempeño, siempre 3 posiciones
    etiquetas = {int(orden[0]): "Bajo desempeño", int(orden[1]): "Desempeño medio", int(orden[2]): "Alto desempeño"}
    return np.array([etiquetas[l] for l in labels])


def guardar_modelo(paquete, version, n_registros):
    """Persiste el modelo en disco con un nombre/versión elegido por el usuario
    y agrega la entrada correspondiente al historial (con fecha)."""
    nombre_archivo = f"{slug(version)}.pkl"
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    joblib.dump(paquete, ruta)
    entrada = {
        "version": version,
        "archivo": nombre_archivo,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "habilidades": paquete["habilidades"],
        "n_registros": n_registros,
    }
    registrar_en_historial(entrada)
    return ruta, entrada


def df_a_excel_bytes(df, titulo="Datos filtrados — Habilidades Blandas"):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos", startrow=2)
        ws = writer.sheets["Datos"]
        n_cols = max(len(df.columns), 1)

        borde = Border(*(Side(style="thin", color="CFE4DF"),) * 4)

        # Título
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
        c_titulo = ws.cell(row=1, column=1, value=titulo)
        c_titulo.font = Font(size=13, bold=True, color="FFFFFF")
        c_titulo.fill = PatternFill("solid", fgColor="0F6B5C")
        c_titulo.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[1].height = 24

        # Subtítulo
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=n_cols)
        c_sub = ws.cell(
            row=2, column=1,
            value=f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  {len(df):,} registros",
        )
        c_sub.font = Font(size=9, italic=True, color="52606D")
        c_sub.alignment = Alignment(horizontal="left", indent=1)

        # Encabezado de columnas
        fila_encabezado = 3
        for j, col in enumerate(df.columns, start=1):
            c = ws.cell(row=fila_encabezado, column=j, value=col)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="0F6B5C")
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = borde

        # Filas de datos con rayado suave y bordes
        for i in range(len(df)):
            fila_excel = fila_encabezado + 1 + i
            for j in range(1, n_cols + 1):
                celda = ws.cell(row=fila_excel, column=j)
                celda.border = borde
                if i % 2 == 0:
                    celda.fill = PatternFill("solid", fgColor="EAF5F2")

        # Ancho de columna aproximado al contenido
        for j, col in enumerate(df.columns, start=1):
            largo_datos = df[col].astype(str).map(len).max() if len(df) else 0
            ancho = max(len(str(col)), largo_datos) + 2
            ws.column_dimensions[get_column_letter(j)].width = min(max(ancho, 10), 42)

        ws.freeze_panes = ws.cell(row=fila_encabezado + 1, column=1).coordinate

    return buf.getvalue()


def grafico_a_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return buf


PRIMARIO_RGB = hex_a_rgb(COLOR_PRIMARIO)
FONDO_RGB = hex_a_rgb(COLOR_FONDO_SUAVE)
TEXTO_RGB = hex_a_rgb(COLOR_TEXTO)


class PDFReporte(FPDF):
    def header(self):
        self.set_fill_color(*PRIMARIO_RGB)
        self.rect(0, 0, 210, 24, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_xy(12, 6)
        self.cell(0, 8, "Análisis de Habilidades Blandas", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_x(12)
        self.cell(0, 6, datetime.now().strftime("Generado: %Y-%m-%d %H:%M"))
        self.set_text_color(*TEXTO_RGB)
        self.set_y(30)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def tabla(self, df, titulo):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*PRIMARIO_RGB)
        self.cell(0, 9, titulo, ln=True)
        self.set_text_color(*TEXTO_RGB)

        cols = list(df.columns)
        ancho = 186 / len(cols)

        self.set_fill_color(*PRIMARIO_RGB)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for c in cols:
            self.cell(ancho, 7, str(c)[:24], border=0, fill=True, align="C")
        self.ln()

        self.set_text_color(*TEXTO_RGB)
        self.set_font("Helvetica", "", 8)
        for i, (_, fila) in enumerate(df.iterrows()):
            self.set_fill_color(*(FONDO_RGB if i % 2 == 0 else (255, 255, 255)))
            for c in cols:
                self.cell(ancho, 7, str(fila[c])[:24], border=0, fill=True)
            self.ln()
        self.ln(5)

    def imagen(self, img_bytes, w=170):
        x = (210 - w) / 2
        self.image(img_bytes, x=x, w=w)
        self.ln(5)


def generar_pdf(titulo, tabla_df, figuras, notas=""):
    pdf = PDFReporte()
    pdf.add_page()
    if notas:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXTO_RGB)
        pdf.multi_cell(0, 5, notas)
        pdf.ln(2)
    pdf.tabla(tabla_df, titulo)
    for fig_bytes in figuras:
        pdf.imagen(fig_bytes)
    return bytes(pdf.output(dest="S"))


# ---------------------------------------------------------------------
# Historial de modelos entrenados
# ---------------------------------------------------------------------
HISTORIAL_PATH = os.path.join(MODELOS_DIR, "historial.json")


def cargar_historial():
    if not os.path.exists(HISTORIAL_PATH):
        return []
    with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
        historial = json.load(f)
    return sorted(historial, key=lambda h: h["fecha"], reverse=True)


def registrar_en_historial(entrada):
    historial = cargar_historial()
    historial.append(entrada)
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)


def eliminar_modelo(archivo):
    """Borra el .pkl del modelo y su entrada del historial."""
    historial = cargar_historial()
    historial = [h for h in historial if h["archivo"] != archivo]
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)
    ruta = os.path.join(MODELOS_DIR, archivo)
    if os.path.exists(ruta):
        os.remove(ruta)


def slug(texto):
    t = texto.strip().lower()
    t = "".join(c if c.isalnum() else "_" for c in t)
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_") or "modelo"


# ---------------------------------------------------------------------
# Estilos propios (CSS)
# ---------------------------------------------------------------------
def inyectar_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: #FAFAFC; font-size: 1.1rem; }}

    div.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

    .hero {{
        background: linear-gradient(135deg, {COLOR_PRIMARIO} 0%, #9F82F7 100%);
        padding: 24px 32px; border-radius: 16px; color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(189, 168, 255, 0.3);
        text-align: center;
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .hero p {{ margin: 8px auto 0 auto; opacity: 0.95; font-size: 1.1rem; max-width: 800px; }}

    div[data-testid="stMetric"] {{
        background: #FFFFFF; border-radius: 16px; padding: 12px 16px;
        border: 1px solid #F0EEF8;
        box-shadow: 0 4px 12px rgba(189, 168, 255, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(189, 168, 255, 0.15);
    }}
    div[data-testid="stMetricLabel"] {{ font-weight: 600; color: #7B7B85; font-size: 1.1rem; }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_PRIMARIO}; font-weight: 700; font-size: 2.2rem; }}

    section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #F0EEF8; }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: {COLOR_PRIMARIO}; font-weight: 700; font-size: 1.4rem;
    }}
    section[data-testid="stSidebar"] div.block-container {{ padding-top: 2rem; }}

    /* Aumentar texto de listas, dropdowns y tablas */
    [data-baseweb="select"] {{ font-size: 1.1rem !important; }}
    [data-baseweb="menu"] {{ font-size: 1.1rem !important; }}
    .stMultiSelect [data-baseweb="tag"] {{ font-size: 1.05rem !important; padding: 6px 10px; }}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size: 1.1rem !important; }}
    label {{ font-size: 1.1rem !important; }}
    table {{ font-size: 1.1rem !important; }}
    th, td {{ font-size: 1.1rem !important; }}
    
    /* Intento de aumentar fuente en DataFrames (si aplica) */
    [data-testid="stDataFrame"] {{ font-size: 1.1rem !important; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px; border-bottom: none; margin-bottom: 16px; background: {COLOR_FONDO_SUAVE}; padding: 6px; border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 8px; padding: 8px 16px;
        font-weight: 600; color: #7B7B85; border: none; font-size: 1.1rem;
        margin: 0; transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {COLOR_PRIMARIO}; background: rgba(189,168,255,0.2); }}
    .stTabs [aria-selected="true"] {{
        background: #FFFFFF !important; color: {COLOR_PRIMARIO};
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}
    .stTabs {{ margin-top: 0px; }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 8px; border: 1px solid {COLOR_PRIMARIO}; font-weight: 600;
        padding: 0.5rem 1.4rem; transition: all 0.2s ease; font-size: 1.1rem;
    }}
    .stButton > button:hover {{ background: {COLOR_FONDO_SUAVE}; transform: translateY(-1px); }}
    .stDownloadButton > button {{ background: {COLOR_PRIMARIO}; color: white; border: none; box-shadow: 0 4px 10px rgba(189,168,255,0.3); }}
    .stDownloadButton > button:hover {{ background: #A287F4; transform: translateY(-1px); box-shadow: 0 6px 14px rgba(189,168,255,0.4); color: white; }}

    .badge {{
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        background: {COLOR_FONDO_SUAVE}; color: {COLOR_PRIMARIO}; font-size: 1.05rem;
        font-weight: 700; margin: 0 6px 8px 0; border: 1px solid {COLOR_BORDE};
    }}

    .card-indicador {{
        border-radius: 16px; padding: 16px 20px; background: white;
        border: 1px solid #F0EEF8; box-shadow: 0 4px 12px rgba(189, 168, 255, 0.08);
        margin-bottom: 14px; transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .card-indicador:hover {{
        transform: translateY(-3px); box-shadow: 0 8px 20px rgba(189, 168, 255, 0.15);
    }}
    .card-indicador .valor {{ font-size: 2.2rem; font-weight: 800; margin: 4px 0 0 0; line-height: 1.2; letter-spacing: -0.5px; }}
    .card-indicador .etiqueta {{ font-size: 1.05rem; color: #7B7B85; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
    .card-indicador .sub {{ font-size: 0.9rem; color: #A1A1AA; margin-top: 4px; }}

    .section-title {{
        background: {COLOR_FONDO_SUAVE}; display: inline-block;
        padding: 8px 16px; border-radius: 8px; color: {COLOR_PRIMARIO};
        margin: 16px 0 16px 0; font-weight: 700; font-size: 1.25rem;
    }}

    div[data-testid="stVerticalBlock"] {{ gap: 1.2rem; }}
    div[data-testid="stPlotlyChart"] {{ margin: 10px 0 16px 0; }}
    div[data-testid="column"] {{ padding: 0 12px; }}
    </style>
    """, unsafe_allow_html=True)


def tarjeta_indicador(color, etiqueta, valor, sub=""):
    st.markdown(f"""
    <div class="card-indicador" style="border-left: 5px solid {color}; padding-left: 15px;">
        <div class="etiqueta">{etiqueta}</div>
        <div class="valor" style="color:{color};">{valor}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def titulo_seccion(texto):
    st.markdown(f'<div class="section-title">{texto}</div>', unsafe_allow_html=True)


def aplicar_estilo_matplotlib():
    plt.rcParams.update({
        "axes.edgecolor": COLOR_BORDE, "axes.labelcolor": COLOR_TEXTO,
        "text.color": COLOR_TEXTO, "xtick.color": "#52606D", "ytick.color": "#52606D",
        "axes.grid": True, "grid.color": COLOR_FONDO_SUAVE, "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9.5,
    })


def estilo_titulo_mpl(ax, titulo):
    """Aplica el mismo estilo de título que diseno_plotly (tamaño, color y
    alineación a la izquierda) para que la versión estática del PDF se vea
    igual que la gráfica interactiva mostrada en pantalla."""
    ax.set_title(titulo, fontsize=13.5, color=COLOR_TEXTO, loc="left", fontweight="normal", pad=34)


def leyenda_arriba_mpl(ax, ncol=3):
    """Coloca la leyenda arriba del área de la gráfica, alineada a la
    izquierda, igual que la leyenda horizontal de Plotly (diseno_plotly)."""
    ax.legend(title="", frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
               ncol=ncol, fontsize=8.5, handletextpad=0.4, columnspacing=1.1)


def diseno_plotly(fig, titulo, altura=300, xlabel=None, ylabel=None, mostrar_leyenda=True):
    """Aplica un estilo consistente (tipografía, colores, márgenes compactos) a un
    plotly.graph_objects.Figure y lo devuelve listo para st.plotly_chart."""
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=COLOR_TEXTO, family="sans-serif"), x=0),
        height=altura,
        margin=dict(l=10, r=10, t=46, b=16),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=COLOR_TEXTO, size=12, family="sans-serif"),
        showlegend=mostrar_leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                     font=dict(size=11)),
        bargap=0.28, bargroupgap=0.12,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=COLOR_BORDE),
    )
    fig.update_xaxes(title=xlabel, gridcolor=COLOR_FONDO_SUAVE, zeroline=False,
                      showline=True, linecolor=COLOR_BORDE)
    fig.update_yaxes(title=ylabel, gridcolor=COLOR_FONDO_SUAVE, zeroline=False,
                      showline=True, linecolor=COLOR_BORDE)
    return fig


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
inyectar_css()
aplicar_estilo_matplotlib()

st.markdown("""
<div class="hero">
    <h1>Análisis No Supervisado de Habilidades Blandas</h1>
    <p>Carga tus datos, elige las habilidades a evaluar y descubre patrones de desempeño
    con un modelo de agrupación entrenable y reutilizable.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Guía rápida")
    with st.expander("¿Cómo se usa?", expanded=False):
        st.markdown(
            "1. Sube tu CSV (o usa el generado por `generar_dataset.py`).\n"
            "2. Elige un modelo guardado (fija las habilidades por ti) o "
            "'Crear nuevo modelo' para elegir tus propias 2 a 4 habilidades.\n"
            "3. Explora las pestañas de resultados y descarga lo que necesites."
        )

    st.markdown("### Cargar datos")
    archivo = st.file_uploader("Set de datos (.csv)", type=["csv"])

    st.markdown("### Modelo de agrupación")
    historial = cargar_historial()
    opciones_modelo = ["-- Seleccionar opción --", "-- Crear nuevo modelo --"] + [
        f"{h['version']}  ({h['fecha']})" for h in historial
    ]
    opcion_modelo_label = st.selectbox("Modelo previamente generado", options=opciones_modelo)
    
    if opcion_modelo_label == "-- Seleccionar opción --":
        st.session_state["modelo_entrenado"] = False
        entrada_historial = None
    elif opcion_modelo_label == "-- Crear nuevo modelo --":
        entrada_historial = None
    else:
        st.session_state["modelo_entrenado"] = False
        entrada_historial = historial[opciones_modelo.index(opcion_modelo_label) - 2]

    st.markdown("### Habilidades a analizar")
    
    def reiniciar_entrenamiento():
        st.session_state["modelo_entrenado"] = False

    if entrada_historial is not None:
        habilidades_sel = entrada_historial["habilidades"]
        st.multiselect(
            "Definidas por el modelo elegido (no editable)",
            options=list(PESOS.keys()),
            default=habilidades_sel,
            disabled=True,
        )
        st.caption(
            "Estas habilidades vienen fijadas por el modelo elegido arriba. "
            "Para elegir otras, cambia a '-- Crear nuevo modelo --'."
        )
    else:
        habilidades_sel = st.multiselect(
            "Selecciona entre 2 y 4 habilidades",
            options=list(PESOS.keys()),
            default=["Comunicación", "Liderazgo"],
            on_change=reiniciar_entrenamiento
        )
        
        if opcion_modelo_label == "-- Crear nuevo modelo --":
            if st.button("Agrupar y entrenar modelo", type="primary"):
                st.session_state["modelo_entrenado"] = True

if archivo is None:
    st.info("Carga un archivo CSV para comenzar. El dataset generado por "
            "`generar_dataset.py` tiene 5000 registros de ejemplo.")
    st.stop()

modelo_activo = True
if opcion_modelo_label == "-- Seleccionar opción --":
    modelo_activo = False
elif opcion_modelo_label == "-- Crear nuevo modelo --" and not st.session_state.get("modelo_entrenado", False):
    modelo_activo = False


if not (2 <= len(habilidades_sel) <= 4):
    st.warning("Selecciona entre 2 y 4 habilidades en el panel lateral.")
    st.stop()

st.markdown(
    " ".join(f'<span class="badge">{h}</span>' for h in habilidades_sel),
    unsafe_allow_html=True,
)

df_completo = cargar_csv(archivo.getvalue())
faltantes = [c for c in COLS_META + columnas_de(habilidades_sel) if c not in df_completo.columns]
if faltantes:
    st.error(f"El CSV no contiene las columnas esperadas: {faltantes}")
    st.stop()

cols_finales = COLS_META + columnas_de(habilidades_sel)
df_filtrado = df_completo[cols_finales].copy()

# --------------------- Caché de resultados por combinación -----------------
if "resultados" not in st.session_state:
    st.session_state["resultados"] = {}

if entrada_historial is not None:
    # Reutilizar un modelo ya guardado: sí conviene cachear (mismos datos+modelo -> mismo resultado)
    clave = hash_key(df_filtrado, habilidades_sel, extra=entrada_historial["archivo"])
    cache_path = os.path.join(CACHE_DIR, f"{clave}.pkl")
    if clave in st.session_state["resultados"]:
        resultado = st.session_state["resultados"][clave]
        resultado["_de_cache"] = True
    elif os.path.exists(cache_path):
        resultado = joblib.load(cache_path)
        resultado["_de_cache"] = True
        st.session_state["resultados"][clave] = resultado
    else:
        resultado = None
else:
    # "Entrenar nuevo modelo" NUNCA debe leerse de caché: cada vez que lo eliges
    # quieres un modelo recién entrenado, listo para guardarlo como una versión nueva.
    clave = hash_key(df_filtrado, habilidades_sel, extra=f"nuevo::{datetime.now().timestamp()}")
    cache_path = os.path.join(CACHE_DIR, f"{clave}.pkl")
    resultado = None

if resultado is None:
    with st.spinner("Calculando estadística base y procesando datos..."):
        stats_base = estadistica_propia(df_filtrado, habilidades_sel)
        lideres = mejor_lider_por_area(df_filtrado) if "Liderazgo" in habilidades_sel else None

        df_clusters = df_filtrado.copy()
        paquete = None
        version_actual, guardado = None, False

        if modelo_activo:
            if entrada_historial is None:
                paquete = entrenar_modelo(df_filtrado, habilidades_sel)
                version_actual, guardado = None, False
            else:
                if set(entrada_historial["habilidades"]) != set(habilidades_sel):
                    st.error(
                        f"El modelo '{entrada_historial['version']}' se entrenó con otras "
                        f"habilidades ({ ', '.join(entrada_historial['habilidades']) }). "
                        "Elige un modelo compatible o crea uno nuevo."
                    )
                    st.stop()
                paquete = cargar_modelo_archivo(entrada_historial["archivo"])
                version_actual, guardado = entrada_historial["version"], True

            clusters = predecir_con_modelo(paquete, df_filtrado)
            df_clusters["grupo"] = clusters

        resultado = {
            "stats_base": stats_base,
            "lideres": lideres,
            "df_clusters": df_clusters,
            "paquete": paquete,
            "version_actual": version_actual,
            "guardado": guardado,
            "_de_cache": False,
        }
        if entrada_historial is not None:
            # Solo persistimos en caché los resultados de un modelo ya guardado;
            # "entrenar nuevo modelo" siempre debe recalcularse desde cero.
            joblib.dump(resultado, cache_path)
            st.session_state["resultados"][clave] = resultado

stats_base = resultado["stats_base"]
lideres = resultado["lideres"]
df_clusters = resultado["df_clusters"]

if resultado.get("_de_cache"):
    st.caption("Resultados recuperados de caché (misma combinación de datos y habilidades).")

# --------------------- Clasificación propia por empleado (para comparar) ---
score_cols_sel = [f"{PESOS[h]['prefijo']}_score" for h in habilidades_sel]
df_clusters["score_compuesto"] = df_clusters[score_cols_sel].mean(axis=1)
df_clusters["clasificacion_propia"] = df_clusters["score_compuesto"].apply(clasificar)

MAPA_EQUIVALENCIA = {
    "Óptimo": "Alto desempeño",
    "Aceptable": "Desempeño medio",
    "Necesita mejora": "Bajo desempeño",
}
if modelo_activo:
    df_clusters["coincide"] = (
        df_clusters["clasificacion_propia"].map(MAPA_EQUIVALENCIA) == df_clusters["grupo"]
    )
    pct_coincidencia = df_clusters["coincide"].mean() * 100

    crosstab = pd.crosstab(df_clusters["clasificacion_propia"], df_clusters["grupo"])
    crosstab = crosstab.reindex(index=["Óptimo", "Aceptable", "Necesita mejora"],
                                 columns=["Alto desempeño", "Desempeño medio", "Bajo desempeño"]).fillna(0).astype(int)

    resumen_cluster = df_clusters["grupo"].value_counts().reindex(
        ["Alto desempeño", "Desempeño medio", "Bajo desempeño"]).fillna(0).astype(int)
else:
    pct_coincidencia = 0
    crosstab = None
    resumen_cluster = pd.Series({"Alto desempeño": 0, "Desempeño medio": 0, "Bajo desempeño": 0})
colores = {"Alto desempeño": COLOR_ALTO, "Desempeño medio": COLOR_MEDIO, "Bajo desempeño": COLOR_BAJO}
colores_propio = {"Óptimo": COLOR_ALTO, "Aceptable": COLOR_MEDIO, "Necesita mejora": COLOR_BAJO}
total_emp = len(df_filtrado)

m1, m2, m3, m4 = st.columns(4)
with m1:
    tarjeta_indicador(COLOR_PRIMARIO, "Empleados analizados", f"{total_emp:,}")
with m2:
    tarjeta_indicador(COLOR_PRIMARIO_CLARO, "Habilidades seleccionadas", len(habilidades_sel),
                       ", ".join(habilidades_sel))
with m3:
    if modelo_activo:
        pct_alto = resumen_cluster.get("Alto desempeño", 0) / total_emp * 100 if total_emp else 0
        tarjeta_indicador(COLOR_ALTO, "Alto desempeño (K-Means)",
                           f"{resumen_cluster.get('Alto desempeño', 0):,}", f"{pct_alto:.0f}% del total")
    else:
        tarjeta_indicador("#DDDDDD", "Alto desempeño (K-Means)", "-", "Modelo no creado")
with m4:
    if modelo_activo:
        tarjeta_indicador(COLOR_MEDIO, "Coincidencia propio vs. algoritmo", f"{pct_coincidencia:.0f}%")
    else:
        tarjeta_indicador("#DDDDDD", "Coincidencia propio vs. algoritmo", "-", "Modelo no creado")
tab_datos, tab_base, tab_algoritmo, tab_comparativa, tab_historial = st.tabs(
    ["Datos filtrados", "Estadística base", "Resultado del algoritmo", "Comparativa", "Historial de modelos"]
)

# --------------------- Tab 1: Datos -----------------------------------------
with tab_datos:
    titulo_seccion("Filtrar y explorar")

    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    with f1:
        areas_sel = st.multiselect("Área", options=sorted(df_filtrado["area"].unique()))
    with f2:
        base_depto = df_filtrado[df_filtrado["area"].isin(areas_sel)] if areas_sel else df_filtrado
        deptos_sel = st.multiselect("Departamento", options=sorted(base_depto["departamento"].unique()))
    with f3:
        busqueda = st.text_input("Buscar por nombre", "")

    df_vista = df_filtrado.copy()
    if areas_sel:
        df_vista = df_vista[df_vista["area"].isin(areas_sel)]
    if deptos_sel:
        df_vista = df_vista[df_vista["departamento"].isin(deptos_sel)]
    if busqueda.strip():
        df_vista = df_vista[df_vista["nombre_completo"].str.contains(busqueda.strip(), case=False, na=False)]

    d1, d2, d3 = st.columns(3)
    with d1:
        tarjeta_indicador(COLOR_PRIMARIO, "Registros mostrados", f"{len(df_vista):,}",
                           f"de {len(df_filtrado):,} totales")
    with d2:
        tarjeta_indicador(COLOR_PRIMARIO_CLARO, "Áreas en la vista", df_vista["area"].nunique())
    with d3:
        tarjeta_indicador(COLOR_PRIMARIO, "Departamentos en la vista", df_vista["departamento"].nunique())

    config_columnas = {
        col: st.column_config.ProgressColumn(col, min_value=0, max_value=10, format="%.2f")
        for col in score_cols_sel
    }
    # Usar pandas Styler para forzar un tamaño de fuente mayor en la tabla (si la versión de Streamlit lo soporta)
    df_styled = df_vista.head(300).style.set_properties(**{'font-size': '0.85rem'})
    st.dataframe(df_styled, use_container_width=True, height=380, column_config=config_columnas)
    st.caption(f"Mostrando hasta 300 de {len(df_vista):,} filas que cumplen el filtro actual.")

    st.download_button(
        "Descargar vista filtrada (Excel)",
        data=df_a_excel_bytes(df_vista, titulo="Datos filtrados — Habilidades Blandas"),
        file_name="datos_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --------------------- Tab 2: Estadística base (algoritmo propio) ----------
with tab_base:
    titulo_seccion("Clasificación por umbrales propios")
    st.caption("Óptimo ≥ 8 · Aceptable 6–8 · Necesita mejora < 6")

    b1, b2, b3 = st.columns(3)
    with b1:
        tarjeta_indicador(COLOR_ALTO, "Total Óptimo", f"{int(stats_base['Óptimo'].sum()):,}",
                           "sumado entre habilidades")
    with b2:
        tarjeta_indicador(COLOR_MEDIO, "Total Aceptable", f"{int(stats_base['Aceptable'].sum()):,}",
                           "sumado entre habilidades")
    with b3:
        tarjeta_indicador(COLOR_BAJO, "Total Necesita mejora", f"{int(stats_base['Necesita mejora'].sum()):,}",
                           "sumado entre habilidades")

    st.dataframe(stats_base.style.set_properties(**{'font-size': '0.85rem'}), use_container_width=True)

    fig1_plotly = go.Figure()
    for col, color in zip(["Óptimo", "Aceptable", "Necesita mejora"], [COLOR_ALTO, COLOR_MEDIO, COLOR_BAJO]):
        fig1_plotly.add_bar(
            x=stats_base["Habilidad"], y=stats_base[col], name=col,
            marker_color=color, text=stats_base[col], textposition="outside",
            marker_line_color="white", marker_line_width=1,
        )
    fig1_plotly = diseno_plotly(fig1_plotly, "Empleados por nivel y habilidad",
                                  altura=320, ylabel="Empleados")
    st.plotly_chart(fig1_plotly, use_container_width=True)

    # Versión estática (matplotlib) solo para el PDF, no se muestra en pantalla
    fig1, ax1 = plt.subplots(figsize=(7, 3.4))
    stats_base.set_index("Habilidad")[["Óptimo", "Aceptable", "Necesita mejora"]].plot(
        kind="bar", stacked=False, ax=ax1, color=[COLOR_ALTO, COLOR_MEDIO, COLOR_BAJO],
        width=0.75, edgecolor="white", linewidth=0.6)
    ax1.set_ylabel("Empleados")
    estilo_titulo_mpl(ax1, "Empleados por nivel y habilidad")
    leyenda_arriba_mpl(ax1, ncol=3)
    for cont in ax1.containers:
        ax1.bar_label(cont, fontsize=7.5, padding=2)
    plt.xticks(rotation=0)
    fig1.tight_layout()
    plt.close(fig1)

    if lideres is not None:
        titulo_seccion("Mejor líder por área")
        st.dataframe(lideres.style.set_properties(**{'font-size': '0.85rem'}), use_container_width=True)

    pdf_base = generar_pdf(
        "Estadística base (proceso propio)", stats_base,
        [grafico_a_bytes(fig1)],
        notas="Clasificación por umbrales: Óptimo >= 8, Aceptable 6-8, Necesita mejora < 6.")
    st.download_button("Descargar estadística base (PDF)", data=pdf_base,
                        file_name="estadistica_base.pdf", mime="application/pdf")

# --------------------- Tab 3: Estadística del algoritmo (clustering) -------
with tab_algoritmo:
    titulo_seccion("Agrupación no supervisada (K-Means)")
    if not modelo_activo:
        st.info("⚠️ Selecciona un modelo previamente generado o crea un nuevo modelo para habilitar esta sección.")
    else:

        g1, g2, g3 = st.columns(3)
        for col, grupo in zip([g1, g2, g3], ["Alto desempeño", "Desempeño medio", "Bajo desempeño"]):
            cantidad = resumen_cluster.get(grupo, 0)
            pct = cantidad / total_emp * 100 if total_emp else 0
            with col:
                tarjeta_indicador(colores[grupo], grupo, f"{cantidad:,}", f"{pct:.0f}% del total")

        c1, c2 = st.columns([1, 1.3], gap="large")
        with c1:
            orden_grupo = ["Alto desempeño", "Desempeño medio", "Bajo desempeño"]
            valores = [resumen_cluster.get(g, 0) for g in orden_grupo]
            fig2_plotly = go.Figure(go.Bar(
                x=valores, y=orden_grupo, orientation="h",
                marker_color=[colores[g] for g in orden_grupo],
                text=valores, texttemplate="%{text:,}", textposition="outside",
                marker_line_color="white", marker_line_width=1,
            ))
            fig2_plotly.update_yaxes(autorange="reversed")
            fig2_plotly = diseno_plotly(fig2_plotly, "Distribución de grupos", altura=300,
                                          xlabel="Empleados", mostrar_leyenda=False)
            fig2_plotly.update_xaxes(range=[0, max(valores) * 1.18])
            st.plotly_chart(fig2_plotly, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(4.6, 3.4))
            barras = ax2.barh(orden_grupo, valores, color=[colores[g] for g in orden_grupo])
            ax2.invert_yaxis()
            ax2.set_xlabel("Empleados")
            estilo_titulo_mpl(ax2, "Distribución de grupos")
            for b, v in zip(barras, valores):
                ax2.text(b.get_width() + max(valores) * 0.01, b.get_y() + b.get_height() / 2,
                          f"{v:,}", va="center", fontsize=8.5)
            fig2.tight_layout()
            plt.close(fig2)
        with c2:
            resumen_area = df_clusters.groupby(["area", "grupo"]).size().unstack(fill_value=0)
            orden_cols = [c for c in ["Alto desempeño", "Desempeño medio", "Bajo desempeño"] if c in resumen_area.columns]
            resumen_area = resumen_area[orden_cols]

            fig3_plotly = go.Figure()
            for col in resumen_area.columns:
                fig3_plotly.add_bar(x=resumen_area.index, y=resumen_area[col], name=col,
                                      marker_color=colores.get(col, "#999"),
                                      marker_line_color="white", marker_line_width=1)
            fig3_plotly = diseno_plotly(fig3_plotly, "Grupos por área", altura=320, ylabel="Empleados")
            st.plotly_chart(fig3_plotly, use_container_width=True)

            fig3, ax3 = plt.subplots(figsize=(7, 3.6))
            resumen_area.plot(kind="bar", stacked=False, ax=ax3,
                               color=[colores.get(c, "#999") for c in resumen_area.columns],
                               width=0.75, edgecolor="white", linewidth=0.5)
            plt.xticks(rotation=20)
            ax3.set_ylabel("Empleados")
            estilo_titulo_mpl(ax3, "Grupos por área")
            leyenda_arriba_mpl(ax3, ncol=3)
            fig3.tight_layout()
            plt.close(fig3)

        st.markdown("---")
        titulo_seccion("Explorar empleados por grupo")
        grupo_elegido = st.radio(
            "Elige un grupo para ver a sus empleados",
            options=["Alto desempeño", "Desempeño medio", "Bajo desempeño"],
            horizontal=True,
        )
        tabla_grupo = df_clusters[df_clusters["grupo"] == grupo_elegido][
            ["nombre_completo", "area", "departamento"] + score_cols_sel
        ]
        st.dataframe(
            tabla_grupo.head(200).style.set_properties(**{'font-size': '0.85rem'}), use_container_width=True, height=300,
            column_config={
                col: st.column_config.ProgressColumn(col, min_value=0, max_value=10, format="%.2f")
                for col in score_cols_sel
            },
        )
        st.caption(f"{len(tabla_grupo):,} empleados en '{grupo_elegido}' (mostrando hasta 200).")

        if resultado["guardado"]:
            st.caption(f"Modelo utilizado: {resultado['version_actual']} (guardado en el historial)")
        else:
            st.caption("Modelo creado en esta sesión — todavía no se ha guardado en el historial.")
            with st.form("guardar_modelo_form"):
                nombre_version = st.text_input(
                    "Nombre o versión para este modelo",
                    value=f"v{len(cargar_historial()) + 1}_{'_'.join(PESOS[h]['prefijo'] for h in habilidades_sel)}",
                )
                enviar = st.form_submit_button("Guardar modelo")
            if enviar:
                if not nombre_version.strip():
                    st.warning("Escribe un nombre o versión antes de guardar.")
                else:
                    ruta_guardada, entrada_nueva = guardar_modelo(
                        resultado["paquete"], nombre_version.strip(), len(df_filtrado)
                    )
                    resultado["guardado"] = True
                    resultado["version_actual"] = entrada_nueva["version"]
                    st.session_state["resultados"][clave] = resultado
                    st.success(f"Modelo guardado como '{entrada_nueva['version']}' ({entrada_nueva['fecha']}).")
                    st.rerun()

        tabla_pdf_cluster = resumen_cluster.rename("Empleados").reset_index().rename(columns={"index": "Grupo"})
        pdf_algoritmo = generar_pdf(
            "Estadística del algoritmo (K-Means)", tabla_pdf_cluster,
            [grafico_a_bytes(fig2), grafico_a_bytes(fig3)],
            notas="Agrupación no supervisada en 3 grupos según el score de las habilidades seleccionadas.")
        st.download_button("Descargar estadística del algoritmo (PDF)", data=pdf_algoritmo,
                            file_name="estadistica_algoritmo.pdf", mime="application/pdf")

    # --------------------- Tab 4: Comparativa (pantalla dividida) --------------
with tab_comparativa:
    if not modelo_activo:
        st.info("⚠️ Selecciona un modelo previamente generado o crea modelo para habilitar esta sección.")
    else:
        titulo_seccion("Estadística base (criterio propio) vs. estadística del algoritmo")
        st.caption(
            "Mismo empleado, dos formas de clasificarlo: a la izquierda tu regla por "
            "umbrales sobre el score combinado de las habilidades elegidas; a la derecha "
            "el grupo que le asignó K-Means. Abajo, el cruce de ambas."
        )

        tarjeta_indicador(COLOR_MEDIO, "Coincidencia entre criterio propio y el algoritmo",
                           f"{pct_coincidencia:.1f}%",
                           "Óptimo↔Alto · Aceptable↔Medio · Necesita mejora↔Bajo")

        izq, der = st.columns(2, gap="large")

        with izq:
            st.markdown("**Base (proceso propio)**")
            conteo_propio = df_clusters["clasificacion_propia"].value_counts().reindex(
                ["Óptimo", "Aceptable", "Necesita mejora"]).fillna(0).astype(int)
            st.dataframe(conteo_propio.rename("Empleados").reset_index().rename(columns={"index": "Clasificación"}).style.set_properties(**{'font-size': '0.85rem'}),
                         use_container_width=True)
            orden_propio = ["Óptimo", "Aceptable", "Necesita mejora"]
            valores4 = [conteo_propio.get(g, 0) for g in orden_propio]
            fig4_plotly = go.Figure(go.Bar(
                x=valores4, y=orden_propio, orientation="h",
                marker_color=[colores_propio[g] for g in orden_propio],
                text=valores4, texttemplate="%{text:,}", textposition="outside",
                marker_line_color="white", marker_line_width=1,
            ))
            fig4_plotly.update_yaxes(autorange="reversed")
            fig4_plotly = diseno_plotly(fig4_plotly, "Mi clasificación", altura=280,
                                          xlabel="Empleados", mostrar_leyenda=False)
            fig4_plotly.update_xaxes(range=[0, max(valores4 + [1]) * 1.18])
            st.plotly_chart(fig4_plotly, use_container_width=True)

            fig4, ax4 = plt.subplots(figsize=(4.6, 3.2))
            barras4 = ax4.barh(orden_propio, valores4, color=[colores_propio[g] for g in orden_propio])
            ax4.invert_yaxis()
            ax4.set_xlabel("Empleados")
            estilo_titulo_mpl(ax4, "Tu clasificación")
            for b, v in zip(barras4, valores4):
                ax4.text(b.get_width() + max(valores4 + [1]) * 0.01, b.get_y() + b.get_height() / 2,
                          f"{v:,}", va="center", fontsize=8.5)
            fig4.tight_layout()
            plt.close(fig4)

        with der:
            st.markdown("**Algoritmo (K-Means)**")
            st.dataframe(resumen_cluster.rename("Empleados").reset_index().rename(columns={"index": "Grupo"}).style.set_properties(**{'font-size': '0.85rem'}),
                         use_container_width=True)
            orden_grupo5 = ["Alto desempeño", "Desempeño medio", "Bajo desempeño"]
            valores5 = [resumen_cluster.get(g, 0) for g in orden_grupo5]
            fig5_plotly = go.Figure(go.Bar(
                x=valores5, y=orden_grupo5, orientation="h",
                marker_color=[colores[g] for g in orden_grupo5],
                text=valores5, texttemplate="%{text:,}", textposition="outside",
                marker_line_color="white", marker_line_width=1,
            ))
            fig5_plotly.update_yaxes(autorange="reversed")
            fig5_plotly = diseno_plotly(fig5_plotly, "Clasificación del algoritmo", altura=280,
                                          xlabel="Empleados", mostrar_leyenda=False)
            fig5_plotly.update_xaxes(range=[0, max(valores5 + [1]) * 1.18])
            st.plotly_chart(fig5_plotly, use_container_width=True)

            fig5, ax5 = plt.subplots(figsize=(4.6, 3.2))
            barras5 = ax5.barh(orden_grupo5, valores5, color=[colores[g] for g in orden_grupo5])
            ax5.invert_yaxis()
            ax5.set_xlabel("Empleados")
            estilo_titulo_mpl(ax5, "Clasificación del algoritmo")
            for b, v in zip(barras5, valores5):
                ax5.text(b.get_width() + max(valores5 + [1]) * 0.01, b.get_y() + b.get_height() / 2,
                          f"{v:,}", va="center", fontsize=8.5)
            fig5.tight_layout()
            plt.close(fig5)

        st.markdown("---")
        titulo_seccion("Cruce entre ambas clasificaciones")
        st.dataframe(crosstab.style.set_properties(**{'font-size': '0.85rem'}), use_container_width=True)

        fig6_plotly = go.Figure()
        for col in crosstab.columns:
            fig6_plotly.add_bar(x=crosstab.index, y=crosstab[col], name=col,
                                  marker_color=colores[col],
                                  marker_line_color="white", marker_line_width=1)
        fig6_plotly = diseno_plotly(fig6_plotly, "¿En qué grupo del algoritmo cae cada clasificación propia?",
                                      altura=340, xlabel="Tu clasificación", ylabel="Empleados")
        st.plotly_chart(fig6_plotly, use_container_width=True)

        fig6, ax6 = plt.subplots(figsize=(7.5, 3.6))
        crosstab.plot(kind="bar", stacked=False, ax=ax6,
                       color=[colores[c] for c in crosstab.columns],
                       width=0.75, edgecolor="white", linewidth=0.5)
        ax6.set_ylabel("Empleados")
        ax6.set_xlabel("Tu clasificación")
        leyenda_arriba_mpl(ax6, ncol=3)
        plt.xticks(rotation=0)
        estilo_titulo_mpl(ax6, "¿En qué grupo del algoritmo cae cada clasificación propia?")
        for cont in ax6.containers:
            ax6.bar_label(cont, fontsize=7, padding=2)
        fig6.tight_layout()
        plt.close(fig6)

        tabla_pdf_cruce = crosstab.reset_index().rename(columns={"clasificacion_propia": "Tu clasificación"})
        pdf_comparativa = generar_pdf(
            "Comparativa: estadística base vs. algoritmo", tabla_pdf_cruce,
            [grafico_a_bytes(fig4), grafico_a_bytes(fig5), grafico_a_bytes(fig6)],
            notas=(f"Coincidencia global entre el criterio propio y el algoritmo: "
                   f"{pct_coincidencia:.1f}%. Óptimo~Alto desempeño, "
                   f"Aceptable~Desempeño medio, Necesita mejora~Bajo desempeño."))
        st.download_button("Descargar comparativa (PDF)", data=pdf_comparativa,
                            file_name="comparativa.pdf", mime="application/pdf")

    # --------------------- Tab 5: Historial de modelos --------------------------
with tab_historial:
    titulo_seccion("Modelos guardados")
    historial_actual = cargar_historial()
    if not historial_actual:
        st.info(
            "Todavía no has guardado ningún modelo. Crea uno en la pestaña "
            "'Resultado del algoritmo' y usa el botón 'Guardar modelo' para que "
            "aparezca aquí con su fecha."
        )
    else:
        h1, h2, h3 = st.columns(3)
        with h1:
            tarjeta_indicador(COLOR_PRIMARIO, "Modelos guardados", len(historial_actual))
        with h2:
            tarjeta_indicador(COLOR_PRIMARIO_CLARO, "Último guardado", historial_actual[0]["fecha"])
        with h3:
            combinaciones = len(set(tuple(sorted(h["habilidades"])) for h in historial_actual))
            tarjeta_indicador(COLOR_PRIMARIO, "Combinaciones de habilidades distintas", combinaciones)

        st.markdown(
            '<div style="display:flex; font-weight:700; color:#52606D; '
            'font-size:0.82rem; padding: 4px 6px; border-bottom: 2px solid '
            f'{COLOR_BORDE};">'
            '<div style="flex:1.6;">Versión</div>'
            '<div style="flex:1.4;">Guardado</div>'
            '<div style="flex:2.4;">Habilidades</div>'
            '<div style="flex:1;">Registros</div>'
            '<div style="flex:1;"></div>'
            '</div>', unsafe_allow_html=True,
        )

        for h in historial_actual:
            fila = st.columns([1.6, 1.4, 2.4, 1, 1])
            fila[0].markdown(f"**{h['version']}**")
            fila[1].write(h["fecha"])
            fila[2].write(", ".join(h["habilidades"]))
            fila[3].write(f"{h['n_registros']:,}")
            if fila[4].button("Eliminar", key=f"del_{h['archivo']}"):
                st.session_state[f"confirmar_{h['archivo']}"] = True

            if st.session_state.get(f"confirmar_{h['archivo']}"):
                aviso, c1, c2 = st.columns([3, 1, 1])
                aviso.warning(f"¿Eliminar '{h['version']}' definitivamente? No se puede deshacer.")
                if c1.button("Sí, eliminar", key=f"si_{h['archivo']}", type="primary"):
                    eliminar_modelo(h["archivo"])
                    st.session_state.pop(f"confirmar_{h['archivo']}", None)
                    st.success(f"Modelo '{h['version']}' eliminado.")
                    st.rerun()
                if c2.button("Cancelar", key=f"no_{h['archivo']}"):
                    st.session_state.pop(f"confirmar_{h['archivo']}", None)
                    st.rerun()
            st.markdown(f'<hr style="margin:4px 0; border-color:{COLOR_BORDE};">', unsafe_allow_html=True)

        st.caption(
            "Estos modelos quedan disponibles en el selector 'Modelo previamente "
            "generado' del panel lateral. Elige uno para reutilizarlo con nuevas cargas de datos."
        )