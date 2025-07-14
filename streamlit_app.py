import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Estrategia S&P 500 vs. Oro",
    page_icon="📈",
    layout="wide"
)

# --- Estilos CSS Personalizados para Mejor Estética Móvil ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stMetric {
        border: 1px solid #2e3440;
        border-radius: 10px;
        padding: 10px;
        background-color: #3b4252;
        color: #eceff4;
    }
    .stMetric .st-bf {
        color: #d8dee9;
    }
    .stMetric .st-c5 {
        color: #a3be8c; /* Verde para valores positivos */
    }
    .stMetric .st-c4 {
        color: #bf616a; /* Rojo para valores negativos */
    }
    h1, h2, h3 {
        color: #eceff4;
    }
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #5e81ac;
        color: #eceff4;
        background-color: #5e81ac;
    }
    .stSlider [data-baseweb="slider"] {
        color: #88c0d0;
    }
</style>
""", unsafe_allow_html=True)


# --- Funciones Principales (Cacheadas para Rendimiento) ---

@st.cache_data
def get_data():
    """Descarga y prepara datos históricos para S&P 500 y Oro."""
    start_date = "2005-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    with st.spinner('Descargando datos de Yahoo Finance...'):
        sp500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)['Close']
        gold = yf.Ticker("GC=F").history(start=start_date, end=end_date)['Close']
    
    data = pd.DataFrame({'SP500': sp500, 'Gold': gold})
    # Asegurarse de que el índice no tenga zona horaria para evitar problemas de alineación
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
        
    return data.ffill().dropna()

def calculate_strategy_returns(data, ma_period, commission_rate):
    """Calcula los retornos de la estrategia aplicando comisiones en cada operación."""
    data['Ratio'] = data['SP500'] / data['Gold']
    # --- CAMBIO: Usar Media Móvil Triangular (TMA) ---
    data['TMA'] = data['Ratio'].rolling(window=ma_period, win_type='triang').mean()

    # Generar señal y determinar operaciones
    data['Signal'] = np.where(data['Ratio'] > data['TMA'], 1, 0)
    data['Trades'] = data['Signal'].diff().abs().fillna(0)

    # Calcular retornos base
    shifted_signal = data['Signal'].shift(1)
    sp500_return = data['SP500'].pct_change()
    gold_return = data['Gold'].pct_change()
    strategy_return = pd.Series(np.where(shifted_signal == 1, sp500_return, gold_return), index=data.index)

    # Aplicar comisiones
    commission_cost = data['Trades'] * commission_rate
    final_returns = strategy_return - commission_cost

    return final_returns, data['Trades'].sum(), data[['Ratio', 'TMA']]

def calculate_metrics(returns):
    """Calcula las métricas de rendimiento clave."""
    returns = returns.dropna()
    if returns.empty:
        return pd.Series(0, index=['Retorno Total', 'Retorno Anualizado', 'Volatilidad Anualizada', 'Ratio de Sharpe', 'Máximo Drawdown'])

    total_return = (1 + returns).cumprod().iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    wealth_index = (1 + returns).cumprod()
    max_drawdown = (wealth_index / wealth_index.cummax() - 1).min()

    return pd.Series({
        'Retorno Total': total_return, 'Retorno Anualizado': annualized_return,
        'Volatilidad Anualizada': annualized_volatility, 'Ratio de Sharpe': sharpe_ratio,
        'Máximo Drawdown': max_drawdown
    })

def plot_ratio_tma(ratio_df):
    """Crea un gráfico interactivo del ratio y su TMA con Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ratio'], mode='lines', name='Ratio (S&P 500 / Oro)', line=dict(color='#81a1c1', width=1.5)))
    fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['TMA'], mode='lines', name='Media Móvil Triangular (TMA)', line=dict(color='#ebcb8b', width=2, dash='dash')))
    
    fig.update_layout(
        title='Ratio S&P 500 / Oro y su Media Móvil Triangular',
        xaxis_title='Fecha',
        yaxis_title='Ratio',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
    )
    return fig

def plot_cumulative_returns(cum_returns_df):
    """Crea un gráfico interactivo de retornos acumulados con Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns_df.index, y=cum_returns_df['SP500_Cum_Return'], mode='lines', name='S&P 500', line=dict(color='#5e81ac')))
    fig.add_trace(go.Scatter(x=cum_returns_df.index, y=cum_returns_df['Gold_Cum_Return'], mode='lines', name='Oro', line=dict(color='#ebcb8b')))
    fig.add_trace(go.Scatter(x=cum_returns_df.index, y=cum_returns_df['Strategy_Cum_Return'], mode='lines', name='Estrategia', line=dict(color='#a3be8c', width=3)))
    
    fig.update_layout(
        title='Comparación de Rendimiento Acumulado',
        xaxis_title='Fecha',
        yaxis_title='Retorno Acumulado',
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        height=700,
    )
    return fig


# --- Construcción de la Interfaz de Usuario (UI) ---

st.title("📊 Backtesting: Estrategia S&P 500 vs. Oro")
st.markdown("Una aplicación para analizar una estrategia de rotación de activos basada en la relación entre el S&P 500 y el Oro.")

# --- Parámetros de la Estrategia ---
ma_period = 220 # Fijo

# --- Barra Lateral con Información y Controles ---
with st.sidebar:
    st.header("⚙️ Parámetros de la Estrategia")
    st.info(f"**Media Móvil Triangular (TMA):** {ma_period} días (Fijo)")
    commission_rate = st.number_input(
        "Tasa de Comisión por Operación (%)",
        min_value=0.00, max_value=1.00, value=0.05, step=0.01,
        help="El costo porcentual aplicado a cada compra o venta."
    ) / 100  # Convertir a decimal para cálculos
    st.markdown("La comisión se aplica en cada operación (al cambiar de un activo a otro), simulando el costo de apertura y cierre.")


# --- Carga de Datos y Ejecución de la Estrategia ---
data = get_data().copy()
data['Strategy_Return'], total_trades, ratio_df = calculate_strategy_returns(data, ma_period, commission_rate)
data['SP500_Return'] = data['SP500'].pct_change()
data['Gold_Return'] = data['Gold'].pct_change()

# Calcular retornos acumulados
for col in ['Strategy', 'SP500', 'Gold']:
    data[f'{col}_Cum_Return'] = (1 + data[f'{col}_Return'].fillna(0)).cumprod()

# --- Visualización Principal ---

# Gráfico de Rendimiento Acumulado (Ahora primero y más alto)
st.header("Curva de Capital (Escala Logarítmica)")
st.plotly_chart(plot_cumulative_returns(data), use_container_width=True)

# Métricas de Rendimiento
st.header("Métricas Clave de Rendimiento")
metrics = pd.DataFrame({
    'Estrategia': calculate_metrics(data['Strategy_Return']),
    'S&P 500': calculate_metrics(data['SP500_Return']),
    'Oro': calculate_metrics(data['Gold_Return'])
}).T

# Mostrar métricas en formato de tarjeta
cols = st.columns(3)
metric_names = ['Estrategia', 'S&P 500', 'Oro']
for i, col in enumerate(cols):
    with col:
        st.subheader(metric_names[i])
        st.metric("Retorno Anualizado", f"{metrics.loc[metric_names[i], 'Retorno Anualizado']:.2%}")
        st.metric("Volatilidad Anualizada", f"{metrics.loc[metric_names[i], 'Volatilidad Anualizada']:.2%}")
        st.metric("Ratio de Sharpe", f"{metrics.loc[metric_names[i], 'Ratio de Sharpe']:.2f}")
        st.metric("Máximo Drawdown", f"{metrics.loc[metric_names[i], 'Máximo Drawdown']:.2%}")

# Gráfico del Ratio y TMA (Ahora segundo)
st.header("Análisis del Ratio y la Señal de Trading")
st.plotly_chart(plot_ratio_tma(ratio_df), use_container_width=True)

# Información Adicional
st.header("Detalles de la Estrategia")
col1, col2 = st.columns(2)

with col1:
    # Posición actual
    current_ratio = data['Ratio'].iloc[-1]
    current_ma = data['TMA'].iloc[-1]
    current_position = "Largo en S&P 500" if current_ratio > current_ma else "Largo en Oro"
    st.metric("Posición Actual Sugerida", current_position)

with col2:
    total_commission_paid = total_trades * commission_rate
    st.metric("Operaciones Totales", f"{total_trades:.0f}")
    st.metric("Costo Total por Comisiones", f"{total_commission_paid:.2%}")

# Tabla de métricas completa
with st.expander("Ver tabla de métricas detallada"):
    st.dataframe(metrics.style.format({
        'Retorno Total': '{:,.2%}',
        'Retorno Anualizado': '{:,.2%}',
        'Volatilidad Anualizada': '{:,.2%}',
        'Ratio de Sharpe': '{:,.2f}',
        'Máximo Drawdown': '{:,.2%}'
    }))

st.info("Aviso: Este es un backtest histórico y no garantiza resultados futuros. Invertir conlleva riesgos.")
