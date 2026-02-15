# -*- coding: utf-8 -*-
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

# --- ESTILOS PERSONALIZADOS (Colores Necaxa) ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #be0e2d; color: white; height: 3em; font-weight: bold; }
    .stButton>button:hover { background-color: #7a091d; color: white; border: 1px solid white; }
    </style>
    """, unsafe_allow_stdio=True)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

NEW_VARIABLES = ['creacion_peligro', 'resiliencia', 'peligro_ofensivo', 'solidez_defensiva', 'indice_faltas', 'efectivida_puerta', 'solidez_portero']

METRICA_A_DEBILIDAD = {
    'creacion_peligro': 'Creador de Juego',
    'resiliencia': 'Regateador/Asistente',
    'peligro_ofensivo': 'Goleador',
    'solidez_defensiva': 'Defensor/Recuperador',
    'indice_faltas': 'Defensor/Recuperador',
    'efectivida_puerta': 'Goleador',
    'solidez_portero': 'Solidez Portero'
}

METRICAS_LABELS = {
    'creacion_peligro': 'Creación', 'resiliencia': 'Resiliencia', 'peligro_ofensivo': 'Ofensiva',
    'solidez_defensiva': 'Defensa', 'indice_faltas': 'Disciplina', 'efectivida_puerta': 'Efectividad', 'solidez_portero': 'Portería'
}

CLUSTER_MAPPING = {
    'Goleador': {'clusters': [4], 'metric': 'goals_total_p90', 'asc': False},
    'Creador de Juego': {'clusters': [1], 'metric': 'passes_key_p90', 'asc': False},
    'Defensor/Recuperador': {'clusters': [0, 2], 'metric': 'duels_won_p90', 'asc': False},
    'Regateador/Asistente': {'clusters': [5], 'metric': 'dribbles_success_p90', 'asc': False},
    'Solidez Portero': {'clusters': [3], 'metric': 'goals_saves_p90', 'asc': False}
}

# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():
    path_diag = os.path.join(os.path.dirname(__file__), 'df_final_diagnostico_equipos.csv')
    path_jug = os.path.join(os.path.dirname(__file__), 'df_final_recomendacion_jugadores.csv')
    
    df_diag = pd.read_csv(path_diag)
    df_jug = pd.read_csv(path_jug)
    
    if 'statistics_league_name' in df_diag.columns:
        df_diag = df_diag[df_diag['statistics_league_name'] == 'Liga MX'].copy()
    
    df_jug['cluster'] = pd.to_numeric(df_jug['cluster'], errors='coerce').fillna(-1).astype(int)
    return df_diag, df_jug

# =============================================================================
# LÓGICA DE VISUALIZACIÓN
# =============================================================================

def crear_grafico_radar(team_row, team_name):
    metricas_validas = [v for v in NEW_VARIABLES if v in team_row.index]
    values = team_row[metricas_validas].fillna(0).astype(float).tolist()
    values.append(values[0])
    categories = [METRICAS_LABELS.get(m, m) for m in metricas_validas] + [METRICAS_LABELS.get(metricas_validas[0], metricas_validas[0])]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        name=team_name,
        line=dict(color='#be0e2d', width=4), # Rojo Necaxa Profundo
        fillcolor='rgba(190, 14, 45, 0.3)',  # Rojo translúcido
        marker=dict(color='#7a091d', size=10)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='gray', tickfont_size=10),
            angularaxis=dict(gridcolor='gray', tickfont_size=12, font_family="Arial Black")
        ),
        showlegend=False,
        title=dict(text=f"PERFIL DE RENDIMIENTO: {team_name.upper()}", x=0.5, font=dict(size=20, color="#333")),
        height=500
    )
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Club_Necaxa_logo.svg/1200px-Club_Necaxa_logo.svg.png", width=80)
    st.title("The Perfect Fit: Scouting Inteligente")
    
    df_diagnostico, df_jugadores = cargar_datos()

    # --- PASO 1: SELECCIÓN (Sidebar) ---
    st.sidebar.header("⚙️ Configuración")
    equipos = sorted(df_diagnostico['team_name_x'].dropna().unique().tolist())
    equipo_sel = st.sidebar.selectbox("Selecciona un Equipo para analizar:", equipos)
    
    presupuesto = st.sidebar.slider("Presupuesto Máximo (M €)", 0.5, 30.0, 10.0)
    
    # --- PASO 2: DIAGNÓSTICO ---
    team_row = df_diagnostico[df_diagnostico['team_name_x'] == equipo_sel].iloc[0]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.plotly_chart(crear_grafico_radar(team_row, equipo_sel), use_container_width=True)

    with col2:
        st.subheader(f"🔍 Diagnóstico de {equipo_sel}")
        # Detectar debilidad
        valores = team_row[NEW_VARIABLES].astype(float).fillna(0)
        debilidad_cod = valores.idxmin()
        debilidad_nom = METRICA_A_DEBILIDAD.get(debilidad_cod, "N/A")
        
        st.error(f"**Punto Crítico Detectado:** {METRICAS_LABELS[debilidad_cod]}")
        st.info(f"**Perfil de Jugador Necesario:** {debilidad_nom}")
        
        st.write("---")
        st.write("¿Deseas buscar soluciones en el mercado de fichajes para este problema?")
        
        # EL BOTÓN MÁGICO
        buscar = st.button("🚀 BUSCAR REFUERZOS IDEALES")

    # --- PASO 3: RECOMENDACIÓN (Solo si se presiona el botón) ---
    if buscar:
        st.divider()
        st.subheader(f"✅ Candidatos Sugeridos para {debilidad_nom}")
        
        # Lógica de filtrado rápida
        target_info = CLUSTER_MAPPING[debilidad_nom]
        candidatos = df_jugadores[df_jugadores['cluster'].isin(target_info['clusters'])].copy()
        
        # Filtro presupuesto
        candidatos = candidatos[candidatos['transfermarkt_market_value'] <= presupuesto]
        candidatos = candidatos.sort_values(by=target_info['metric'], ascending=False).head(5)
        
        if not candidatos.empty:
            # Mostrar tabla limpia
            st.table(candidatos[['player_name', 'team_name', 'player_age', 'transfermarkt_market_value', target_info['metric']]]
                     .rename(columns={'player_name': 'Jugador', 'team_name': 'Club Actual', 'player_age': 'Edad', 
                                      'transfermarkt_market_value': 'Valor (M€)', target_info['metric']: 'Rendimiento (p90)'}))
            
            st.success("Análisis completado. Puedes consultar al Asistente IA abajo para más detalles.")
        else:
            st.warning("No se encontraron jugadores que ajusten al presupuesto para esta debilidad.")

    # --- PASO 4: ASISTENTE ---
    with st.expander("🤖 Consultar al Asistente de Scouting"):
        # (Aquí va tu lógica de Gemini que ya tenías)
        st.write("Usa el chat de abajo para preguntar detalles sobre estos jugadores.")

if __name__ == "__main__":
    main()
