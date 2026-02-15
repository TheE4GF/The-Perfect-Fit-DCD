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

# --- ESTILOS PERSONALIZADOS (Corrección de el error TypeError) ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #be0e2d; color: white; height: 3em; font-weight: bold; }
    .stButton>button:hover { background-color: #7a091d; color: white; border: 1px solid white; }
    </style>
    """, unsafe_allow_html=True) # <-- ESTA ERA LA LÍNEA DEL ERROR

# ... (El resto de tus CONSTANTES se mantienen igual) ...

# --- FUNCIÓN GEMINI ACTUALIZADA (Para evitar el Error 404) ---
def obtener_respuesta_gemini(prompt):
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return "Configura la API Key."
        
        genai.configure(api_key=api_key)
        # Usamos el nombre directo sin prefijos para mayor compatibilidad
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
