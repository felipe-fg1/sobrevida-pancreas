import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Sobrevida Cáncer de Páncreas", layout="wide")

# ==========================================
# 2. FUNCIÓN DE KAPLAN-MEIER (Adaptada para la app)
# ==========================================
@st.cache_data 
def cargar_datos():
    # IMPORTANTE: Reemplaza esto por tu archivo final procesado
    df = pd.read_csv("df_pacientes_final.csv")
    return df

def generar_grafico_km(df, curvas, incluir_PO, incluir_XP, unidad_tiempo, intervalo_x, x_max, fecha_fin_estudio='2018-12-31'):
    df_km = df.copy()
    
    if unidad_tiempo.lower() == 'y':
        divisor, etiqueta_eje, formato_hover = 365.25, 'Años', '<b>Años:</b> %{x:.1f}<br>'
    elif unidad_tiempo.lower() == 'm':
        divisor, etiqueta_eje, formato_hover = 30.44, 'Meses', '<b>Meses:</b> %{x:.1f}<br>'
    else:
        divisor, etiqueta_eje, formato_hover = 1, 'Días', '<b>Días:</b> %{x:.0f}<br>'
        
    df_km['MIN_FECHA_CP'] = pd.to_datetime(df_km['MIN_FECHA_CP'])
    df_km['FECHA_DEF'] = pd.to_datetime(df_km['FECHA_DEF'])
    fecha_cierre = pd.to_datetime(fecha_fin_estudio)
    
    grupos_validos = ['PP', 'PX']
    if incluir_PO:
        grupos_validos.append('PO')
    if incluir_XP:
        grupos_validos.append('XP')
        
    df_km = df_km[df_km['GRUPO'].isin(grupos_validos)].copy()
    
    if df_km.empty:
        return None

    # Lógica Grupo XP (Sesgo de sobrevida)
    if incluir_XP:
        mask_xp = df_km['GRUPO'] == 'XP'
        df_km.loc[mask_xp, 'MIN_FECHA_CP'] = df_km.loc[mask_xp, 'FECHA_DEF']
        df_km.loc[mask_xp, 'OPERADO'] = False
        df_km.loc[mask_xp, 'QUIMIOTERAPIA'] = False

    df_km['EVENTO'] = np.where(df_km['GRUPO'].isin(['PP', 'XP']), 1, 0)
    df_km['FECHA_TERMINO'] = df_km['FECHA_DEF'].fillna(fecha_cierre)
    df_km['TIEMPO_DIAS'] = (df_km['FECHA_TERMINO'] - df_km['MIN_FECHA_CP']).dt.days
    df_km['TIEMPO_DIAS'] = np.where(df_km['TIEMPO_DIAS'] <= 0, 1, df_km['TIEMPO_DIAS'])
    df_km['TIEMPO_ANALISIS'] = df_km['TIEMPO_DIAS'] / divisor
    
    texto_pub = 'Pertenecientes al Sistema Nacional de Servicios de Salud, SNSS'
    texto_priv = 'No Pertenecientes al Sistema Nacional de Servicios de Salud, SNSS'
    
    regs_norte = ['De Arica y Parinacota', 'De Tarapacá', 'De Antofagasta', 'De Atacama', 'De Coquimbo']
    regs_centro = ['De Valparaíso', "Del Libertador B. O'Higgins", 'Del Maule']
    regs_sur = ['De Ñuble', 'Del Bíobío', 'De La Araucanía', 'De Los Ríos', 'De Los Lagos']
    regs_austral = ['De Aisén del Gral. C. Ibáñez del Campo', 'De Magallanes y de La Antártica Chilena']
    
    config_curvas = {
        'global': {'mask': pd.Series(True, index=df_km.index), 'nombre': 'Cohorte Global', 'color': '#2c3e50'},
        'operados': {'mask': df_km['OPERADO'] == True, 'nombre': 'Operados', 'color': '#27ae60'},
        'no_operados': {'mask': df_km['OPERADO'] == False, 'nombre': 'No Operados', 'color': '#2ecc71', 'dash': 'dash'},
        'quimioterapia': {'mask': df_km['QUIMIOTERAPIA'] == True, 'nombre': 'Quimioterapia', 'color': '#d35400'},
        'sin_quimio': {'mask': df_km['QUIMIOTERAPIA'] == False, 'nombre': 'Sin Quimioterapia', 'color': '#e67e22', 'dash': 'dash'},
        'ambos': {'mask': (df_km['OPERADO'] == True) & (df_km['QUIMIOTERAPIA'] == True), 'nombre': 'Cirugía + Quimio', 'color': '#8e44ad'},
        'solo_cirugia': {'mask': (df_km['OPERADO'] == True) & (df_km['QUIMIOTERAPIA'] == False), 'nombre': 'Solo Cirugía', 'color': '#2980b9'},
        'solo_quimio': {'mask': (df_km['OPERADO'] == False) & (df_km['QUIMIOTERAPIA'] == True), 'nombre': 'Solo Quimioterapia', 'color': '#c0392b'},
        'ninguno': {'mask': (df_km['OPERADO'] == False) & (df_km['QUIMIOTERAPIA'] == False), 'nombre': 'Sin Tratamiento', 'color': '#7f8c8d', 'dash': 'dot'},
        'fonasa': {'mask': pd.to_numeric(df_km['PREVISION_DIAGNOSTICO'], errors='coerce') == 1, 'nombre': 'FONASA', 'color': '#2980b9'},
        'isapre': {'mask': pd.to_numeric(df_km['PREVISION_DIAGNOSTICO'], errors='coerce') == 2, 'nombre': 'ISAPRE', 'color': '#e67e22'},
        'ffaa': {'mask': pd.to_numeric(df_km['PREVISION_DIAGNOSTICO'], errors='coerce').isin([3, 4]), 'nombre': 'Fuerzas Armadas', 'color': '#8e44ad'},
        'tramo_a': {'mask': df_km['TRAMO_FONASA_DIAGNOSTICO'] == 'A', 'nombre': 'FONASA Tramo A', 'color': '#c0392b'},
        'tramo_b': {'mask': df_km['TRAMO_FONASA_DIAGNOSTICO'] == 'B', 'nombre': 'FONASA Tramo B', 'color': '#e67e22'},
        'tramo_c': {'mask': df_km['TRAMO_FONASA_DIAGNOSTICO'] == 'C', 'nombre': 'FONASA Tramo C', 'color': '#f1c40f'},
        'tramo_d': {'mask': df_km['TRAMO_FONASA_DIAGNOSTICO'] == 'D', 'nombre': 'FONASA Tramo D', 'color': '#27ae60'},
        'diag_publico': {'mask': df_km['TIPO_HOSPITAL_DIAGNOSTICO'] == texto_pub, 'nombre': 'Diag. Hospital Público', 'color': '#34495e'},
        'diag_privado': {'mask': df_km['TIPO_HOSPITAL_DIAGNOSTICO'] == texto_priv, 'nombre': 'Diag. Clínica/Privado', 'color': '#95a5a6'},
        'cirugia_publico': {'mask': (df_km['OPERADO'] == True) & (df_km['TIPO_HOSPITAL_CIRUGIA'] == texto_pub), 'nombre': 'Cirugía Hospital Público', 'color': '#2980b9'},
        'cirugia_privado': {'mask': (df_km['OPERADO'] == True) & (df_km['TIPO_HOSPITAL_CIRUGIA'] == texto_priv), 'nombre': 'Cirugía Clínica/Privado', 'color': '#e74c3c'},
        'mz_norte': {'mask': df_km['REGION_DIAGNOSTICO'].isin(regs_norte), 'nombre': 'Macrozona Norte', 'color': '#e74c3c'},
        'mz_centro': {'mask': df_km['REGION_DIAGNOSTICO'].isin(regs_centro), 'nombre': 'Macrozona Centro', 'color': '#f39c12'},
        'mz_metropolitana': {'mask': df_km['REGION_DIAGNOSTICO'] == 'Metropolitana de Santiago', 'nombre': 'RM', 'color': '#8e44ad'},
        'mz_sur': {'mask': df_km['REGION_DIAGNOSTICO'].isin(regs_sur), 'nombre': 'Macrozona Sur', 'color': '#27ae60'},
        'mz_austral': {'mask': df_km['REGION_DIAGNOSTICO'].isin(regs_austral), 'nombre': 'Macrozona Austral', 'color': '#2980b9'}
    }

    fig = go.Figure()
    
    for curva in curvas:
        if curva not in config_curvas: continue
        cfg = config_curvas[curva]
        df_subset = df_km[cfg['mask']].copy()
        
        if df_subset.empty: continue
            
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_subset['TIEMPO_ANALISIS'], event_observed=df_subset['EVENTO'])
        tabla = kmf.survival_function_
        
        fig.add_trace(go.Scatter(
            x=tabla.index, y=tabla.iloc[:, 0], mode='lines',
            name=f"{cfg['nombre']} (n={len(df_subset)})", 
            line_shape='hv', line=dict(color=cfg['color'], width=3, dash=cfg.get('dash', 'solid')),
            hovertemplate=formato_hover + '<b>Sobrevida:</b> %{y:.1%}<extra></extra>'
        ))

    limite_visual = x_max if unidad_tiempo.lower() == 'y' else (x_max * 12 if unidad_tiempo.lower() == 'm' else x_max * 365.25)
    
    fig.update_layout(
        title=None, 
        xaxis=dict(title=f'Tiempo ({etiqueta_eje})', dtick=intervalo_x, range=[0, limite_visual], showgrid=True, griddash='dash'),
        yaxis=dict(title='Probabilidad', tickformat='.0%', range=[0, 1.05], showgrid=True),
        template='plotly_white', hovermode='x unified', height=500, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig

# ==========================================
# 3. INTERFAZ Y MANEJO DE ESTADO (STREAMLIT)
# ==========================================

if 'fig_actual' not in st.session_state:
    st.session_state.fig_actual = None
if 'titulo_actual' not in st.session_state:
    st.session_state.titulo_actual = ""
if 'fig_anterior' not in st.session_state:
    st.session_state.fig_anterior = None
if 'titulo_anterior' not in st.session_state:
    st.session_state.titulo_anterior = ""

df_datos = cargar_datos()

presets = {
    "Cohorte Global": ['global'],
    "Operados vs No Operados": ['operados', 'no_operados'],
    "Quimioterapia vs No Quimioterapia": ['quimioterapia', 'sin_quimio'],
    "Esquemas Terapéuticos": ['ambos', 'solo_cirugia', 'solo_quimio', 'ninguno'],
    "Previsión": ['fonasa', 'isapre', 'ffaa'],
    "Tramos Fonasa": ['tramo_a', 'tramo_b', 'tramo_c', 'tramo_d'],
    "Tipo Hospital al Diagnóstico": ['diag_publico', 'diag_privado'],
    "Tipo Hospital Cirugía": ['cirugia_publico', 'cirugia_privado'],
    "Macrozonas Geográficas": ['mz_norte', 'mz_centro', 'mz_metropolitana', 'mz_sur', 'mz_austral'],
    "Personalizado": []
}

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.markdown("### Analizador de Sobrevida - Cáncer de Páncreas")
    st.markdown("---")
    st.header("1. Configuración Clínica")
    seleccion_preset = st.selectbox("Seleccione un escenario clínico:", list(presets.keys()))
    
    curvas_seleccionadas = presets[seleccion_preset]
    
    if seleccion_preset == "Personalizado":
        todas_las_curvas = [
            'global', 'operados', 'no_operados', 'quimioterapia', 'sin_quimio', 
            'ambos', 'solo_cirugia', 'solo_quimio', 'ninguno', 'fonasa', 'isapre', 'ffaa', 
            'tramo_a', 'tramo_b', 'tramo_c', 'tramo_d', 'diag_publico', 'diag_privado', 
            'cirugia_publico', 'cirugia_privado', 'mz_norte', 'mz_centro', 'mz_metropolitana', 'mz_sur', 'mz_austral'
        ]
        curvas_seleccionadas = st.multiselect("Elija las curvas a comparar:", todas_las_curvas, default=['global'])
        
    st.markdown("---")
    
    # Menú colapsable para configuración del gráfico
    with st.expander("⚙️ Configuración Avanzada del Eje"):
        incluir_po = st.checkbox("Incluir mortalidad por otras causas", value=True)
        incluir_xp = st.checkbox("Incluir Grupo XP (Sesgo de Sobrevida)", value=False, help="Agrega a los casos fallecidos sin egreso previo asumiendo 1 día de sobrevida.")
        unidad_t = st.selectbox("Unidad de Tiempo", options=['m', 'y', 'd'], index=0, format_func=lambda x: {'m': 'Meses', 'y': 'Años', 'd': 'Días'}[x])
        max_x = st.number_input("Mostrar hasta (Años)", min_value=1, max_value=20, value=5)
        paso_x = st.number_input("Intervalo del eje X", min_value=1, value=6)

    # Botón de acción principal
    if st.button("📊 Graficar / Actualizar", use_container_width=True, type="primary"):
        st.session_state.fig_anterior = st.session_state.fig_actual
        st.session_state.titulo_anterior = st.session_state.titulo_actual
        
        nueva_fig = generar_grafico_km(
            df_datos, curvas_seleccionadas, incluir_po, incluir_xp, unidad_t, paso_x, max_x
        )
        st.session_state.fig_actual = nueva_fig
        st.session_state.titulo_actual = seleccion_preset if seleccion_preset != "Personalizado" else "Curvas Personalizadas"

# --- ÁREA PRINCIPAL (GRÁFICOS) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🟢 Gráfico Actual: {st.session_state.titulo_actual}")
    if st.session_state.fig_actual:
        st.plotly_chart(st.session_state.fig_actual, use_container_width=True)
    else:
        st.info("Presione 'Graficar' en el panel izquierdo para generar la curva.")

with col2:
    if st.session_state.fig_anterior:
        st.subheader(f"⚪ Gráfico Anterior: {st.session_state.titulo_anterior}")
        st.plotly_chart(st.session_state.fig_anterior, use_container_width=True)
    else:
        st.subheader("⚪ Gráfico Anterior")
        st.write("*(Aquí aparecerá su gráfico previo para comparar)*")