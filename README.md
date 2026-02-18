# ⚽ The Perfect Fit

**Dashboard de diagnóstico de equipos y recomendación de jugadores para Liga MX.**

Aplicación web construida con Streamlit que permite analizar el rendimiento de los equipos de la Liga MX, detectar debilidades a partir de métricas normalizadas y obtener recomendaciones de refuerzos basadas en clústeres de jugadores.

---

## 🎯 Características

- **Paso 1 — Selección de equipo:** Elige un equipo de Liga MX a analizar.
- **Estadísticas por equipo:** Gráficas de desempeño (victorias/empates/derrotas, goles a favor/en contra), goles por tramos de minutos y tabla de posesión y rating.
- **Paso 2 — Diagnóstico:** Gráfico radar con métricas normalizadas (creación de peligro, resiliencia, peligro ofensivo, solidez defensiva, etc.), debilidades detectadas y tipo de jugador que se necesita. Refuerzo sugerido automáticamente.
- **Paso 3 — Jugadores recomendados:** Listado de jugadores filtrados por tipo de refuerzo, presupuesto y filtros opcionales (edad, máximo años de contrato restantes, posición, nacionalidad).
- **Asistente IA (Gemini):** Chat para consultas sobre las gráficas y las recomendaciones (requiere API key de Google AI Studio).

---

## 📦 Requisitos

- Python 3.9+
- Dependencias listadas en `requirements.txt`

---

## 🚀 Instalación

1. **Clonar el repositorio** (o descargar el proyecto):

   ```bash
   git clone https://github.com/TU_USUARIO/the-perfect-fit.git
   cd the-perfect-fit
   ```

2. **Crear un entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar secretos (opcional, para el asistente Gemini):**

   - Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml`.
   - Añade tu API key de [Google AI Studio](https://aistudio.google.com/):

   ```toml
   GOOGLE_API_KEY = "tu_api_key_aqui"
   ```

5. **Ejecutar la aplicación:**

   ```bash
   streamlit run app.py
   ```

   Se abrirá en el navegador (por defecto `http://localhost:8501`).

---

## 📁 Estructura del proyecto

```
The Perfect Fit/
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
├── df_final_diagnostico_equipos.csv   # Datos de equipos Liga MX
├── df_final_recomendacion_jugadores.csv # Jugadores con clústeres y métricas
├── .streamlit/
│   └── secrets.toml.example       # Ejemplo de configuración para Gemini
└── pruebaproyectofinalmodulov_final.py  # Notebook/script origen del análisis
```

Los CSV deben estar en la misma carpeta que `app.py` para que la app cargue los datos correctamente.

---

## 🔧 Uso

1. Selecciona un equipo en el desplegable.
2. Revisa las estadísticas y gráficas del equipo (desempeño y goles por minuto).
3. Pulsa **"Ver diagnóstico del equipo"** para ver el radar y las debilidades.
4. Pulsa **"Ver jugadores recomendados"** y ajusta presupuesto y filtros en la barra lateral si lo deseas.
5. Usa el chat inferior para preguntar al asistente IA (si configuraste `GOOGLE_API_KEY`).

---

## 📊 Datos

- **Diagnóstico de equipos:** métricas agregadas por equipo (Liga MX), incluyendo variables normalizadas para el radar.
- **Recomendación de jugadores:** base de jugadores con clústeres (K-Means) y métricas p90; se recomiendan según el tipo de refuerzo asociado a la debilidad del equipo.

---

## 📄 Licencia

Proyecto de uso educativo / portfolio. Ajusta la licencia según tu preferencia.

---

## 👤 Autor

Erick Alejandro Guzmán Flores
