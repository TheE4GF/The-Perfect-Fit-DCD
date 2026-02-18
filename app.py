# -*- coding: utf-8 -*-
"""
The Perfect Fit - Aplicación de Recomendación de Jugadores para Liga MX
Dashboard Streamlit con diagnóstico de equipos y recomendaciones basadas en clústeres.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuración de página
st.set_page_config(
    page_title="The Perfect Fit - Scouting Liga MX",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS: contenedor de gráficas sin fondo blanco y con bordes redondeados
st.markdown("""
<style>
    div[data-testid="stPlotlyChart"] {
        background-color: #0e1117 !important;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stPlotlyChart"] > div {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES (fidelidad al código original)
# =============================================================================

NEW_VARIABLES = [
    'creacion_peligro',
    'resiliencia',
    'peligro_ofensivo',
    'solidez_defensiva',
    'indice_faltas',
    'efectivida_puerta',
    'solidez_portero'
]

# Mapeo: métrica del radar (la más baja = debilidad) -> tipo de refuerzo a recomendar
METRICA_A_DEBILIDAD = {
    'creacion_peligro': 'Creador de Juego',
    'resiliencia': 'Regateador/Asistente',
    'peligro_ofensivo': 'Goleador',
    'solidez_defensiva': 'Defensor/Recuperador',
    'indice_faltas': 'Defensor/Recuperador',
    'efectivida_puerta': 'Goleador',
    'solidez_portero': 'Solidez Portero'
}

# Labels para el gráfico de radar (en español)
METRICAS_LABELS = {
    'creacion_peligro': 'Creación de Peligro',
    'resiliencia': 'Resiliencia',
    'peligro_ofensivo': 'Peligro Ofensivo',
    'solidez_defensiva': 'Solidez Defensiva',
    'indice_faltas': 'Índice de Faltas',
    'efectivida_puerta': 'Efectividad a Puerta',
    'solidez_portero': 'Solidez Portero'
}

# Lógica de recomendación (replicada de pruebaproyectofinalmodulov_final.py)
CLUSTER_MAPPING = {
    'Goleador': {'clusters': [4], 'metric': 'goals_total_p90', 'asc': False},
    'Creador de Juego': {'clusters': [1], 'metric': 'passes_key_p90', 'asc': False},
    'Defensor/Recuperador': {'clusters': [0, 2], 'metric': 'duels_won_p90', 'asc': False},
    'Regateador/Asistente': {'clusters': [5], 'metric': 'dribbles_success_p90', 'asc': False},
    'Solidez Portero': {'clusters': [3], 'metric': 'goals_saves_p90', 'asc': False}
}

# =============================================================================
# FUNCIONES DE CARGA (st.cache_data para optimización)
# =============================================================================

@st.cache_data
def cargar_diagnostico_equipos():
    """Carga el CSV de diagnóstico de equipos (Liga MX)."""
    path = os.path.join(os.path.dirname(__file__), 'df_final_diagnostico_equipos.csv')
    df = pd.read_csv(path)
    # Filtrar solo equipos de Liga MX
    if 'statistics_league_name' in df.columns:
        df = df[df['statistics_league_name'] == 'Liga MX'].copy()
    # Limpiar resiliencia si viene como string con %
    if 'resiliencia' in df.columns:
        df['resiliencia'] = df['resiliencia'].astype(str).str.replace('%', '', regex=False)
        df['resiliencia'] = pd.to_numeric(df['resiliencia'], errors='coerce').fillna(0)
    return df

@st.cache_data
def cargar_recomendacion_jugadores():
    """Carga el CSV de jugadores con clústeres y métricas p90."""
    path = os.path.join(os.path.dirname(__file__), 'df_final_recomendacion_jugadores.csv')
    df = pd.read_csv(path)
    # Asegurar que cluster sea numérico
    df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce')
    # Filtrar jugadores con cluster válido (0-5)
    df = df[df['cluster'].notna() & df['cluster'].isin([0, 1, 2, 3, 4, 5])].copy()
    df['cluster'] = df['cluster'].astype(int)
    return df

# =============================================================================
# FUNCIÓN DE RECOMENDACIÓN (replicada del código original)
# =============================================================================

def recomendar_refuerzos_integral(debilidad_prioritaria, presupuesto_max, n_opciones, df_players, 
                                   filtro_edad_min=None, filtro_edad_max=None, 
                                   filtro_contrato_max=None, filtro_nacionalidad=None,
                                   filtro_posicion=None):
    """
    Recomienda jugadores basado en debilidad prioritaria.
    Fidelidad a la función en pruebaproyectofinalmodulov_final.py
    """
    if debilidad_prioritaria not in CLUSTER_MAPPING:
        return pd.DataFrame(), f"Debilidad '{debilidad_prioritaria}' no reconocida."

    target_info = CLUSTER_MAPPING[debilidad_prioritaria]
    target_clusters = target_info['clusters']
    sort_metric = target_info['metric']
    sort_ascending = target_info['asc']

    candidatos = df_players[df_players['cluster'].isin(target_clusters)].copy()

    if candidatos.empty:
        return pd.DataFrame(), f"No se encontraron jugadores en los clústeres objetivo para '{debilidad_prioritaria}'."

    # Presupuesto
    if 'transfermarkt_market_value' in candidatos.columns:
        candidatos['transfermarkt_market_value'] = pd.to_numeric(candidatos['transfermarkt_market_value'], errors='coerce')
        candidatos = candidatos[candidatos['transfermarkt_market_value'].notna()]
        candidatos = candidatos[candidatos['transfermarkt_market_value'] <= presupuesto_max]

    if candidatos.empty:
        return pd.DataFrame(), f"No hay jugadores dentro del presupuesto de {presupuesto_max}M €."

    # Filtros adicionales
    if filtro_edad_min is not None:
        candidatos = candidatos[candidatos['player_age'].fillna(0) >= filtro_edad_min]
    if filtro_edad_max is not None:
        candidatos = candidatos[candidatos['player_age'].fillna(99) <= filtro_edad_max]
    if filtro_contrato_max is not None and 'contract_length_years' in candidatos.columns:
        candidatos['contract_length_years'] = pd.to_numeric(candidatos['contract_length_years'], errors='coerce')
        candidatos = candidatos[candidatos['contract_length_years'].fillna(999) <= filtro_contrato_max]
    if filtro_nacionalidad is not None and len(filtro_nacionalidad) > 0 and 'player_nationality' in candidatos.columns:
        candidatos = candidatos[candidatos['player_nationality'].fillna('').astype(str).str.strip().isin([str(n).strip() for n in filtro_nacionalidad])]
    if filtro_posicion is not None and len(filtro_posicion) > 0 and 'games_position' in candidatos.columns:
        candidatos = candidatos[candidatos['games_position'].fillna('').astype(str).str.strip().isin([str(p).strip() for p in filtro_posicion])]

    if candidatos.empty:
        return pd.DataFrame(), "No hay jugadores que cumplan los filtros aplicados."

    # Verificar que la métrica existe
    if sort_metric not in candidatos.columns:
        sort_metric = 'games_rating'  # fallback

    candidatos[sort_metric] = pd.to_numeric(candidatos[sort_metric], errors='coerce')
    candidatos_ordenados = candidatos.sort_values(by=sort_metric, ascending=sort_ascending)
    top_opciones = candidatos_ordenados.head(n_opciones)

    result_columns = [
        'player_name', 'team_name', 'league_name', 'player_age', 'player_nationality',
        'transfermarkt_market_value', 'games_position', sort_metric,
        'contract_length_years', 'transfermarkt_contract_ends'
    ]
    result_columns = [c for c in result_columns if c in top_opciones.columns]
    final_df = top_opciones[result_columns].copy()
    final_df = final_df.rename(columns={
        'player_name': 'Nombre',
        'team_name': 'Equipo',
        'league_name': 'Liga',
        'player_age': 'Edad',
        'player_nationality': 'Nacionalidad',
        'transfermarkt_market_value': 'Valor (M €)',
        'games_position': 'Posición',
        sort_metric: f'Métrica Clave (p90)',
        'contract_length_years': 'Años Contrato',
        'transfermarkt_contract_ends': 'Fin Contrato'
    })
    return final_df, None

# =============================================================================
# FUNCIÓN DE DETECCIÓN DE DEBILIDADES
# =============================================================================

def detectar_debilidades_equipo(team_row):
    """
    Identifica la debilidad principal del equipo (métrica normalizada más baja del radar).
    Retorna: (refuerzo_prioritario, lista de 3 claves de métrica ordenadas de más débil a menos).
    """
    metricas_validas = [v for v in NEW_VARIABLES if v in team_row.index]
    if not metricas_validas:
        return None, []

    valores = team_row[metricas_validas].astype(float)
    valores = valores.fillna(0)
    idx_min = valores.idxmin()
    debilidad_prioritaria = METRICA_A_DEBILIDAD.get(idx_min, 'Defensor/Recuperador')

    # Top 3 debilidades: claves de métrica (para mostrar nombre en gráfica + tipo de jugador)
    orden_metricas = valores.sort_values().index.tolist()
    top3_metricas = orden_metricas[:3]
    return debilidad_prioritaria, top3_metricas

# Tema oscuro para gráficas (sin fondo blanco; integrado con la página)
_CHART_THEME = {
    'paper_bgcolor': '#0e1117',
    'plot_bgcolor': '#1a1d24',
    'font_color': '#e8eaed',
    'gridcolor': '#3c4043',
    'linecolor': '#5f6368',
    'title_font_size': 15,
}

# =============================================================================
# GRÁFICA DE RADAR
# =============================================================================

def crear_grafico_desempeno_equipo(team_row, team_name):
    """Crea el gráfico de desempeño: Wins/Draws/Losses y Goles a favor vs en contra (por equipo)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Resultados (Victorias, Empates, Derrotas)', 'Goles a favor vs en contra'),
        horizontal_spacing=0.12,
        vertical_spacing=0.08
    )
    match_outcomes = ['Victorias', 'Empates', 'Derrotas']
    match_values = [
        int(team_row.get('Wins', 0)),
        int(team_row.get('Draws', 0)),
        int(team_row.get('Losses', 0))
    ]
    fig.add_trace(
        go.Bar(x=match_outcomes, y=match_values, marker_color=['#4caf50', '#ff9800', '#f44336'], showlegend=False),
        row=1, col=1
    )
    goals_labels = ['Goles a favor', 'Goles en contra']
    goals_values = [
        team_row.get('statistics_goals_for_total_total', 0),
        team_row.get('statistics_goals_against_total_total', 0)
    ]
    fig.add_trace(
        go.Bar(x=goals_labels, y=goals_values, marker_color=['#42a5f5', '#e57373'], showlegend=False),
        row=1, col=2
    )
    fig.update_layout(
        title_text=f'Desempeño de {team_name} en la temporada',
        title_font=dict(size=_CHART_THEME['title_font_size'], color=_CHART_THEME['font_color']),
        title_x=0.5,
        height=380,
        barmode='group',
        paper_bgcolor=_CHART_THEME['paper_bgcolor'],
        plot_bgcolor=_CHART_THEME['plot_bgcolor'],
        font=dict(family='Arial', color=_CHART_THEME['font_color'], size=12),
        margin=dict(t=60, b=50, l=50, r=30)
    )
    fig.update_xaxes(gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'])
    fig.update_yaxes(gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'])
    fig.update_annotations(font=dict(size=13, color=_CHART_THEME['font_color']))
    return fig


def crear_grafico_goles_por_minuto(team_row, team_name):
    """Crea el gráfico de goles anotados y recibidos por rango de minutos (por equipo)."""
    minute_ranges_labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    goals_for_values = []
    for mr in minute_ranges_labels:
        col_name = f'statistics_goals_for_minute_{mr}_total'
        val = team_row.get(col_name, 0)
        goals_for_values.append(0 if pd.isna(val) else float(val))
    goals_against_values = []
    for mr in minute_ranges_labels:
        col_name = f'statistics_goals_against_minute_{mr}_total'
        val = team_row.get(col_name, 0)
        goals_against_values.append(0 if pd.isna(val) else float(val))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Goles anotados por minuto', 'Goles recibidos por minuto'),
        horizontal_spacing=0.12,
        vertical_spacing=0.08
    )
    fig.add_trace(
        go.Bar(x=minute_ranges_labels, y=goals_for_values, name='Goles anotados', marker_color='#4caf50'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=minute_ranges_labels, y=goals_against_values, name='Goles recibidos', marker_color='#e57373'),
        row=1, col=2
    )
    fig.update_layout(
        title_text=f'Goles por minuto — {team_name}',
        title_font=dict(size=_CHART_THEME['title_font_size'], color=_CHART_THEME['font_color']),
        title_x=0.5,
        height=380,
        showlegend=False,
        paper_bgcolor=_CHART_THEME['paper_bgcolor'],
        plot_bgcolor=_CHART_THEME['plot_bgcolor'],
        font=dict(family='Arial', color=_CHART_THEME['font_color'], size=12),
        margin=dict(t=60, b=50, l=50, r=30)
    )
    fig.update_xaxes(title_text='Rango de minutos', gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'], row=1, col=1)
    fig.update_yaxes(title_text='Total goles', gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'], row=1, col=1)
    fig.update_xaxes(title_text='Rango de minutos', gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'], row=1, col=2)
    fig.update_yaxes(title_text='Total goles', gridcolor=_CHART_THEME['gridcolor'], linecolor=_CHART_THEME['linecolor'], zerolinecolor=_CHART_THEME['linecolor'], row=1, col=2)
    fig.update_annotations(font=dict(size=13, color=_CHART_THEME['font_color']))
    return fig


def crear_grafico_radar(team_row, team_name):
    """Crea el gráfico de radar de rendimiento del equipo con colores legibles."""
    metricas_validas = [v for v in NEW_VARIABLES if v in team_row.index]
    if not metricas_validas:
        return None

    values = team_row[metricas_validas].fillna(0).astype(float).tolist()
    values.append(values[0])  # Cerrar el polígono
    categories = [METRICAS_LABELS.get(m, m) for m in metricas_validas] + [METRICAS_LABELS.get(metricas_validas[0], metricas_validas[0])]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=team_name,
        line=dict(color='#0d47a1', width=2.5),
        fillcolor='rgba(25, 118, 210, 0.35)',
        marker=dict(size=10, color='#1565c0', line=dict(color='white', width=1)),
        opacity=0.9
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=_CHART_THEME['plot_bgcolor'],
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=_CHART_THEME['gridcolor'],
                tickfont=dict(size=12, color=_CHART_THEME['font_color']),
                linecolor=_CHART_THEME['linecolor']
            ),
            angularaxis=dict(
                direction='clockwise',
                tickfont=dict(size=14, color=_CHART_THEME['font_color'], family='Arial, sans-serif'),
                gridcolor=_CHART_THEME['gridcolor'],
                linecolor=_CHART_THEME['linecolor']
            )
        ),
        showlegend=False,
        title=dict(
            text=f'Análisis de Rendimiento: {team_name}',
            font=dict(size=18, color=_CHART_THEME['font_color'], family='Arial'),
            x=0.5
        ),
        paper_bgcolor=_CHART_THEME['paper_bgcolor'],
        plot_bgcolor=_CHART_THEME['plot_bgcolor'],
        margin=dict(l=100, r=100, t=80, b=80),
        height=500,
        font=dict(family='Arial', color=_CHART_THEME['font_color'], size=13)
    )
    return fig

# =============================================================================
# ASISTENTE GEMINI
# =============================================================================

def obtener_respuesta_gemini(prompt, historial=None):
    """Usa Gemini para responder preguntas del usuario sobre gráficas y recomendaciones."""
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "⚠️ No se encontró GOOGLE_API_KEY en st.secrets. Configura la clave en Streamlit Cloud o en .streamlit/secrets.toml localmente."

    system_instruction = """Eres un asistente experto en análisis de fútbol y scouting de jugadores. 
Tu rol es ayudar al usuario a entender las gráficas de diagnóstico de equipos de Liga MX y guiarlo en la elección de jugadores recomendados.
La app tiene 3 pasos: 1) Seleccionar equipo, 2) Ver gráfica de diagnóstico con debilidades, 3) Al presionar el botón, ver jugadores recomendados con filtros.

REGLAS DE ORO:
1.- FLUJO OBLIGATORIO: Si el usuario no ha presionado el botón de 'BUSCAR REFUERZOS', debes responder amablemente que primero seleccionen un equipo y presionen el botón para que puedas analizar a los candidatos reales.
2.- Solo recomienda o habla a fondo de los jugadores que aparecen en esta lista: {datos_jugadores[:2000]}
3.- MÉTRICAS: Usa términos como 'Percentiles', 'Clustering K-Means' y 'Métricas p90' para dar autoridad.
4.- SI NO HAY DATOS: Si la lista de jugadores está vacía, no inventes nombres. Pide que ejecuten la búsqueda.

Métricas del radar: creacion_peligro, resiliencia, peligro_ofensivo, solidez_defensiva, indice_faltas, efectivida_puerta, solidez_portero.

Contexto de fichajes:
1.- creacion_peligro: En caso de ser una debilidad, se requiere a un creador de juego, ya que es necesario alguien quien pueda generar jugadas de peligro.
2.- resiliencia: En caso de ser una debilidad, se requiere a un regateador/asistente, ya que el equipo tiene voluntad, pero es necesario alguien quien marque diferencia para aprovechar esa voluntad.
3.- peligro_ofensivo: En caso de ser debilidad, se requiere a un goleador, alguien quien sepa aprovechar las oportunidades.
4.- solidez_defensiva:  En caso de ser debilidad, se requiere a un defensor/recuperador, alguien quien sea firme en la saga defensiva y corrija los problemas de recibir goles.
5.- indice_faltas: En caso de ser debilidad, se requiere a un defensor/recuperador, alguien quien sea firme en la saga defensiva y corrija los problemas de la inseguridad cuando no se tiene el balon
6.- efectividad_puerta: En caso de ser debilidad, se requiere a un goleador, alguien quien sepa corregir los tiros a puerta, que sea bueno anotando goles.
7.- solidez_portero: En caso de ser debilidad, se requiere a un solidez portero, alguien quien sea seguro en la porteria

Explica de forma clara y concisa. Responde en el mismo idioma que use el usuario."""

    full_prompt = f"{system_instruction}\n\n{prompt}"

    # Opción 1: Obtener modelos disponibles vía REST y probar cada uno
    try:
        import requests
        for api_version in ("v1", "v1beta"):
            list_url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={api_key}"
            list_resp = requests.get(list_url, timeout=10)
            if list_resp.status_code != 200:
                continue
            models_data = list_resp.json()
            for m in models_data.get("models", []):
                name = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                gen_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{name}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}
                }
                try:
                    resp = requests.post(gen_url, json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            return text
                except Exception:
                    continue
    except Exception:
        pass

    # Opción 2: Lista fija de modelos por si ListModels falla
    for api_version in ("v1", "v1beta"):
        for model in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"):
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024}
                }
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text:
                        return text
            except Exception:
                continue

    # Opción 3: SDK google-genai (si la REST falló)
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_name in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"):
            try:
                response = client.models.generate_content(model=model_name, contents=full_prompt)
                if response and response.text:
                    return response.text
            except Exception:
                continue
    except ImportError:
        pass

    return _mensaje_error_gemini(
        "No se encontró ningún modelo compatible. Genera una nueva API key en "
        "Google AI Studio (aistudio.google.com) y actualízala en los secrets."
    )


def _mensaje_error_gemini(err_msg):
    """Mensaje amigable cuando falla la conexión a Gemini."""
    return (
        "⚠️ **No se pudo conectar con Gemini.**\n\n"
        f"*Error técnico:* {err_msg}\n\n"
        "**Qué puedes hacer:**\n"
        "1. Instala el nuevo SDK: `pip install google-genai`\n"
        "2. Verifica tu API key en [Google AI Studio](https://aistudio.google.com/).\n"
        "3. Genera una nueva API key si la actual es antigua.\n\n"
        "Mientras tanto, usa los filtros de la app para encontrar jugadores."
    )

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

def main():
    # Inicializar estado para flujo paso a paso
    if "equipo_seleccionado" not in st.session_state:
        st.session_state.equipo_seleccionado = None
    if "mostrar_diagnostico" not in st.session_state:
        st.session_state.mostrar_diagnostico = False
    if "mostrar_recomendaciones" not in st.session_state:
        st.session_state.mostrar_recomendaciones = False
    if "messages_gemini" not in st.session_state:
        st.session_state.messages_gemini = []

    st.title("⚽ The Perfect Fit")
    st.markdown("**Dashboard de Diagnóstico y Recomendación de Jugadores para Liga MX**")

    # Cargar datos
    df_diagnostico = cargar_diagnostico_equipos()
    df_jugadores = cargar_recomendacion_jugadores()

    if df_diagnostico.empty or df_jugadores.empty:
        st.error("No se pudieron cargar los archivos CSV. Verifica que existan df_final_diagnostico_equipos.csv y df_final_recomendacion_jugadores.csv")
        return

    equipos_opciones = df_diagnostico['team_name_x'].dropna().unique().tolist()

    # ========== PASO 1: Selección de equipo ==========
    st.subheader("📋 Paso 1: Selecciona un equipo")
    equipo_seleccionado = st.selectbox(
        "Equipo de Liga MX a analizar:",
        options=equipos_opciones,
        index=0 if equipos_opciones else 0,
        key="selector_equipo"
    )

    # Guardar selección
    st.session_state.equipo_seleccionado = equipo_seleccionado

    # Resetear diagnóstico y recomendaciones si cambian de equipo
    if "equipo_con_diagnostico" in st.session_state and st.session_state.equipo_con_diagnostico != equipo_seleccionado:
        st.session_state.mostrar_diagnostico = False
        st.session_state.mostrar_recomendaciones = False
    if "equipo_con_recomendaciones" in st.session_state and st.session_state.equipo_con_recomendaciones != equipo_seleccionado:
        st.session_state.mostrar_recomendaciones = False

    # ========== SECCIÓN: Estadísticas por equipo (solo al tener equipo seleccionado, antes de "Ver diagnóstico") ==========
    if equipo_seleccionado:
        st.divider()
        st.subheader(f"📈 Estadísticas de {equipo_seleccionado}")
        st.caption("Gráficas y datos del equipo seleccionado. Luego usa el botón inferior para ver el diagnóstico detallado.")

        team_row_preview = df_diagnostico[df_diagnostico['team_name_x'] == equipo_seleccionado].iloc[0]

        st.markdown("")  # Pequeño espacio antes de las gráficas
        # Bloque de gráficas: dos columnas con espaciado uniforme
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_desempeno = crear_grafico_desempeno_equipo(team_row_preview, equipo_seleccionado)
            st.plotly_chart(fig_desempeno, use_container_width=True, key="chart_desempeno")
        with col_chart2:
            fig_minutos = crear_grafico_goles_por_minuto(team_row_preview, equipo_seleccionado)
            st.plotly_chart(fig_minutos, use_container_width=True, key="chart_minutos")

        st.markdown("---")
        # Tabla: solo datos del equipo seleccionado (posesión y rating)
        st.markdown("**Posesión y rating de este equipo**")
        cols_tabla = ['team_name_x', 'ball_possession_total', 'games_rating']
        cols_disponibles = [c for c in cols_tabla if c in df_diagnostico.columns]
        if cols_disponibles:
            df_tabla = df_diagnostico[df_diagnostico['team_name_x'] == equipo_seleccionado][cols_disponibles].copy()
            if 'ball_possession_total' in df_tabla.columns:
                df_tabla['ball_possession_total'] = pd.to_numeric(df_tabla['ball_possession_total'], errors='coerce')
            if 'games_rating' in df_tabla.columns:
                df_tabla['games_rating'] = pd.to_numeric(df_tabla['games_rating'], errors='coerce')
            rename_map = {'team_name_x': 'Equipo', 'ball_possession_total': 'Posesión (%)', 'games_rating': 'Rating'}
            df_tabla = df_tabla.rename(columns={k: v for k, v in rename_map.items() if k in df_tabla.columns})
            if 'Posesión (%)' in df_tabla.columns and df_tabla['Posesión (%)'].max() <= 1.5:
                df_tabla['Posesión (%)'] = (df_tabla['Posesión (%)'] * 100).round(1)
            st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron las columnas ball_possession_total o games_rating en los datos.")

        st.divider()

    # Botón para ver el diagnóstico (el gráfico solo aparece después de elegir equipo)
    if st.button("📊 Ver diagnóstico del equipo", type="primary", use_container_width=True):
        st.session_state.mostrar_diagnostico = True
        st.session_state.equipo_con_diagnostico = equipo_seleccionado

    # ========== PASO 2: Gráfica de diagnóstico (solo si ya escogió equipo y confirmó) ==========
    if st.session_state.mostrar_diagnostico:
        team_row = df_diagnostico[df_diagnostico['team_name_x'] == equipo_seleccionado].iloc[0]
        debilidad_auto, top3_metricas = detectar_debilidades_equipo(team_row)
        debilidad_usar = debilidad_auto
        top3_refuerzos = [METRICA_A_DEBILIDAD.get(m, m) for m in top3_metricas] if top3_metricas else []

        st.divider()
        st.subheader(f"📊 Paso 2: Diagnóstico de {equipo_seleccionado}")

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_radar = crear_grafico_radar(team_row, equipo_seleccionado)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown("**Debilidades detectadas (métricas más bajas):**")
            for i, metrica_key in enumerate(top3_metricas, 1):
                nombre_metrica = METRICAS_LABELS.get(metrica_key, metrica_key)
                tipo_jugador = METRICA_A_DEBILIDAD.get(metrica_key, metrica_key)
                st.markdown(f"{i}. **{nombre_metrica}**")
                st.caption(f"→ Tipo de jugador que se necesita: {tipo_jugador}")

            st.info(f"**Refuerzo sugerido:** {debilidad_usar}")

        # Botón para mostrar jugadores recomendados
        st.divider()
        if st.button("🎯 Ver jugadores recomendados", type="primary", use_container_width=True, key="btn_recomendaciones"):
            st.session_state.mostrar_recomendaciones = True
            st.session_state.equipo_con_recomendaciones = equipo_seleccionado
    else:
        # Necesitamos estos valores para cuando no hay diagnóstico (p. ej. Gemini)
        team_row = df_diagnostico[df_diagnostico['team_name_x'] == equipo_seleccionado].iloc[0]
        debilidad_auto, top3_metricas = detectar_debilidades_equipo(team_row)
        debilidad_usar = debilidad_auto
        top3_refuerzos = [METRICA_A_DEBILIDAD.get(m, m) for m in top3_metricas] if top3_metricas else []

    # ========== PASO 3: Jugadores recomendados (solo al apretar el botón) ==========
    if st.session_state.mostrar_recomendaciones:
        st.divider()
        st.subheader("🎯 Paso 3: Jugadores recomendados")

        # Filtros en sidebar (visibles solo en este paso)
        with st.sidebar:
            st.header("🎯 Filtros de recomendación")
            presupuesto_mdd = st.slider("Presupuesto máximo (M €)", min_value=0.5, max_value=50.0, value=15.0, step=0.5)
            n_resultados = st.slider("Número de recomendaciones", min_value=3, max_value=20, value=10)

            debilidad_manual = st.selectbox(
                "Forzar tipo de refuerzo (opcional)",
                options=["Auto (según diagnóstico)", "Goleador", "Creador de Juego", "Defensor/Recuperador", "Regateador/Asistente", "Solidez Portero"],
                index=0
            )
            debilidad_usar = debilidad_auto if debilidad_manual == "Auto (según diagnóstico)" else debilidad_manual

            st.subheader("Filtros adicionales")
            usar_filtros = st.checkbox("Aplicar filtros extra", value=False)
            filtro_edad_min = None
            filtro_edad_max = None
            filtro_contrato_max = None
            filtro_nacionalidad = None
            filtro_posicion = None
            if usar_filtros:
                filtro_edad_min = st.number_input("Edad mínima", min_value=16, max_value=45, value=18)
                filtro_edad_max = st.number_input("Edad máxima", min_value=16, max_value=45, value=35)
                filtro_contrato_max = st.number_input("Máx. años de contrato restantes", min_value=0.0, max_value=5.0, value=5.0, step=0.5, help="Solo jugadores con ese tiempo o menos de contrato")
                posiciones_disponibles = sorted([p for p in df_jugadores['games_position'].dropna().astype(str).unique().tolist() if str(p).strip()])
                filtro_posicion = st.multiselect("Posición", options=posiciones_disponibles, default=None, help="Deja vacío para incluir todas (ej. Attacker, Defender, Goalkeeper)")
                nacionalidades_disponibles = sorted(df_jugadores['player_nationality'].dropna().astype(str).unique().tolist())
                filtro_nacionalidad = st.multiselect("Nacionalidad del jugador", options=nacionalidades_disponibles, default=None, help="Deja vacío para incluir todas")

        df_recomendados, error = recomendar_refuerzos_integral(
            debilidad_usar, presupuesto_mdd, n_resultados, df_jugadores,
            filtro_edad_min=filtro_edad_min,
            filtro_edad_max=filtro_edad_max,
            filtro_contrato_max=filtro_contrato_max,
            filtro_nacionalidad=filtro_nacionalidad,
            filtro_posicion=filtro_posicion
        )

        if error:
            st.warning(error)
        elif not df_recomendados.empty:
            st.dataframe(df_recomendados, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron jugadores que cumplan los criterios.")

    # ========== ASISTENTE GEMINI (siempre visible) ==========
    st.divider()
    st.subheader("🤖 Asistente IA (Gemini)")

    for msg in st.session_state.messages_gemini:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre las gráficas o los jugadores recomendados..."):
        st.session_state.messages_gemini.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        contexto = f"""
Equipo analizado: {equipo_seleccionado}
Debilidades detectadas: {', '.join(top3_refuerzos)}
Tipo de refuerzo aplicado: {debilidad_usar}
¿El usuario ya vio las recomendaciones de jugadores? {st.session_state.mostrar_recomendaciones}
"""
        respuesta = obtener_respuesta_gemini(contexto + "\n\nPregunta del usuario: " + prompt)

        with st.chat_message("assistant"):
            st.markdown(respuesta)

        st.session_state.messages_gemini.append({"role": "assistant", "content": respuesta})

if __name__ == "__main__":
    main()
