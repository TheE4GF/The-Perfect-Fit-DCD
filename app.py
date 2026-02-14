# -*- coding: utf-8 -*-
"""
The Perfect Fit - Aplicación de Recomendación de Jugadores para Liga MX
Dashboard Streamlit con diagnóstico de equipos y recomendaciones basadas en clústeres.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Configuración de página
st.set_page_config(
    page_title="The Perfect Fit - Scouting Liga MX",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                                   filtro_precio_max=None, filtro_contrato_min=None):
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
    if filtro_precio_max is not None:
        candidatos = candidatos[candidatos['transfermarkt_market_value'] <= filtro_precio_max]
    if filtro_contrato_min is not None and 'contract_length_years' in candidatos.columns:
        candidatos['contract_length_years'] = pd.to_numeric(candidatos['contract_length_years'], errors='coerce')
        candidatos = candidatos[candidatos['contract_length_years'].fillna(0) >= filtro_contrato_min]

    if candidatos.empty:
        return pd.DataFrame(), "No hay jugadores que cumplan los filtros aplicados."

    # Verificar que la métrica existe
    if sort_metric not in candidatos.columns:
        sort_metric = 'games_rating'  # fallback

    candidatos[sort_metric] = pd.to_numeric(candidatos[sort_metric], errors='coerce')
    candidatos_ordenados = candidatos.sort_values(by=sort_metric, ascending=sort_ascending)
    top_opciones = candidatos_ordenados.head(n_opciones)

    result_columns = [
        'player_name', 'team_name', 'league_name', 'player_age',
        'transfermarkt_market_value', 'games_position', sort_metric,
        'cluster', 'contract_length_years', 'transfermarkt_contract_ends'
    ]
    result_columns = [c for c in result_columns if c in top_opciones.columns]
    final_df = top_opciones[result_columns].copy()
    final_df = final_df.rename(columns={
        'player_name': 'Nombre',
        'team_name': 'Equipo',
        'league_name': 'Liga',
        'player_age': 'Edad',
        'transfermarkt_market_value': 'Valor (M €)',
        'games_position': 'Posición',
        sort_metric: f'Métrica Clave (p90)',
        'cluster': 'Clúster',
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
    Basado en la gráfica de radar del código original.
    """
    metricas_validas = [v for v in NEW_VARIABLES if v in team_row.index]
    if not metricas_validas:
        return None, []

    valores = team_row[metricas_validas].astype(float)
    valores = valores.fillna(0)
    idx_min = valores.idxmin()
    debilidad_prioritaria = METRICA_A_DEBILIDAD.get(idx_min, 'Defensor/Recuperador')

    # Top 3 debilidades (métricas más bajas)
    orden_debilidades = valores.sort_values().index.tolist()
    top3 = [METRICA_A_DEBILIDAD.get(m, m) for m in orden_debilidades[:3]]
    return debilidad_prioritaria, top3

# =============================================================================
# GRÁFICA DE RADAR
# =============================================================================

def crear_grafico_radar(team_row, team_name):
    """Crea el gráfico de radar de rendimiento del equipo."""
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
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        opacity=0.7
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='lightgray'),
            angularaxis=dict(direction='clockwise', tickfont_size=11)
        ),
        showlegend=False,
        title=dict(text=f'Análisis de Rendimiento: {team_name}', font=dict(size=16), x=0.5),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80),
        height=450
    )
    return fig

# =============================================================================
# ASISTENTE GEMINI
# =============================================================================

def obtener_respuesta_gemini(prompt, historial=None):
    """Usa Gemini para responder preguntas del usuario sobre gráficas y recomendaciones."""
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
        if not api_key:
            return "⚠️ No se encontró GOOGLE_API_KEY en st.secrets. Configura la clave en Streamlit Cloud o en .streamlit/secrets.toml localmente."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        system_instruction = """Eres un asistente experto en análisis de fútbol y scouting de jugadores. 
Tu rol es ayudar al usuario a entender las gráficas de diagnóstico de equipos de Liga MX y guiarlo en la elección de jugadores recomendados.
Usa las métricas: creacion_peligro, resiliencia, peligro_ofensivo, solidez_defensiva, indice_faltas, efectivida_puerta, solidez_portero.
Explica de forma clara y concisa. Responde en el mismo idioma que use el usuario."""

        response = model.generate_content(
            f"{system_instruction}\n\n{prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024,
            )
        )
        return response.text
    except ImportError:
        return "⚠️ Instala la librería: pip install google-generativeai"
    except Exception as e:
        return f"Error al conectar con Gemini: {str(e)}"

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

def main():
    st.title("⚽ The Perfect Fit")
    st.markdown("**Dashboard de Diagnóstico y Recomendación de Jugadores para Liga MX**")

    # Cargar datos
    df_diagnostico = cargar_diagnostico_equipos()
    df_jugadores = cargar_recomendacion_jugadores()

    if df_diagnostico.empty or df_jugadores.empty:
        st.error("No se pudieron cargar los archivos CSV. Verifica que existan df_final_diagnostico_equipos.csv y df_final_recomendacion_jugadores.csv")
        return

    # --- SIDEBAR ---
    st.sidebar.header("🎯 Filtros")

    equipos_opciones = df_diagnostico['team_name_x'].dropna().unique().tolist()
    equipo_seleccionado = st.sidebar.selectbox(
        "Equipo (Liga MX)",
        options=equipos_opciones,
        index=0 if equipos_opciones else 0
    )

    st.sidebar.subheader("Presupuesto y Resultados")
    presupuesto_mdd = st.sidebar.slider("Presupuesto máximo (M €)", min_value=0.5, max_value=50.0, value=15.0, step=0.5)
    n_resultados = st.sidebar.slider("Número de recomendaciones", min_value=3, max_value=20, value=10)

    st.sidebar.subheader("Filtros Adicionales")
    usar_filtros = st.sidebar.checkbox("Aplicar filtros extra", value=False)

    filtro_edad_min = None
    filtro_edad_max = None
    filtro_precio_max = None
    filtro_contrato_min = None

    if usar_filtros:
        filtro_edad_min = st.sidebar.number_input("Edad mínima", min_value=16, max_value=45, value=18)
        filtro_edad_max = st.sidebar.number_input("Edad máxima", min_value=16, max_value=45, value=35)
        filtro_precio_max = st.sidebar.number_input("Precio máximo adicional (M €)", min_value=0.5, max_value=100.0, value=20.0)
        filtro_contrato_min = st.sidebar.number_input("Mín. años de contrato restantes", min_value=0.0, max_value=5.0, value=0.0, step=0.5)

    debilidad_manual = st.sidebar.selectbox(
        "Forzar tipo de refuerzo (opcional)",
        options=["Auto (según diagnóstico)", "Goleador", "Creador de Juego", "Defensor/Recuperador", "Regateador/Asistente", "Solidez Portero"],
        index=0
    )

    # --- CONTENIDO PRINCIPAL ---
    team_row = df_diagnostico[df_diagnostico['team_name_x'] == equipo_seleccionado].iloc[0]
    debilidad_auto, top3_debilidades = detectar_debilidades_equipo(team_row)

    if debilidad_manual == "Auto (según diagnóstico)":
        debilidad_usar = debilidad_auto
    else:
        debilidad_usar = debilidad_manual

    # Columnas: Gráfica + Info
    col1, col2 = st.columns([1, 1])

    with col1:
        fig_radar = crear_grafico_radar(team_row, equipo_seleccionado)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader(f"📊 Diagnóstico: {equipo_seleccionado}")

        if 'Wins' in team_row.index:
            st.metric("Victorias", int(team_row.get('Wins', 0)))
            st.metric("Empates", int(team_row.get('Draws', 0)))
            st.metric("Derrotas", int(team_row.get('Losses', 0)))

        st.markdown("**Debilidades detectadas (métricas más bajas):**")
        for i, d in enumerate(top3_debilidades, 1):
            st.markdown(f"{i}. {d}")

        st.info(f"**Refuerzo sugerido:** {debilidad_usar}")

    # --- RECOMENDACIONES ---
    st.divider()
    st.subheader("🎯 Jugadores Recomendados")

    df_recomendados, error = recomendar_refuerzos_integral(
        debilidad_usar, presupuesto_mdd, n_resultados, df_jugadores,
        filtro_edad_min=filtro_edad_min,
        filtro_edad_max=filtro_edad_max,
        filtro_precio_max=filtro_precio_max,
        filtro_contrato_min=filtro_contrato_min
    )

    if error:
        st.warning(error)
    elif not df_recomendados.empty:
        st.dataframe(df_recomendados, use_container_width=True, hide_index=True)
    else:
        st.info("No se encontraron jugadores que cumplan los criterios.")

    # --- ASISTENTE GEMINI ---
    st.divider()
    st.subheader("🤖 Asistente IA (Gemini)")

    if "messages_gemini" not in st.session_state:
        st.session_state.messages_gemini = []

    for msg in st.session_state.messages_gemini:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre las gráficas o los jugadores recomendados..."):
        st.session_state.messages_gemini.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        contexto = f"""
Equipo analizado: {equipo_seleccionado}
Debilidades detectadas: {', '.join(top3_debilidades)}
Tipo de refuerzo aplicado: {debilidad_usar}
Número de jugadores recomendados mostrados: {len(df_recomendados) if not df_recomendados.empty else 0}
"""
        respuesta = obtener_respuesta_gemini(contexto + "\n\nPregunta del usuario: " + prompt)

        with st.chat_message("assistant"):
            st.markdown(respuesta)

        st.session_state.messages_gemini.append({"role": "assistant", "content": respuesta})

if __name__ == "__main__":
    main()
