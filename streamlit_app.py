import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="📈 Estrategia S&P 500 vs Oro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para móviles
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        .stMetric {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class="main-header">
    <h1>📈 Estrategia de Trading: S&P 500 vs Oro</h1>
    <p>Análisis interactivo de estrategia basada en media móvil</p>
</div>
""", unsafe_allow_html=True)

# Funciones principales
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_data():
    """Descarga datos históricos para S&P 500 y Oro."""
    try:
        start_date = "2005-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        with st.spinner("Descargando datos del mercado..."):
            sp500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)['Close']
            gold = yf.Ticker("GC=F").history(start=start_date, end=end_date)['Close']
            
        data = pd.DataFrame({'SP500': sp500, 'Gold': gold})
        data.index = data.index.tz_localize(None)
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"Error al descargar datos: {str(e)}")
        return None

def calculate_strategy_returns(data, ma_period, commission_rate):
    """Calcula los retornos de la estrategia aplicando comisiones en cada operación."""
    ratio = data['SP500'] / data['Gold']
    ma = ratio.rolling(window=ma_period).mean()
    
    # Generar señal y determinar operaciones
    signal = pd.Series(np.where(ratio > ma, 1, 0), index=data.index)
    trades = signal.diff().abs().fillna(0)
    
    # Calcular retornos base
    shifted_signal = signal.shift(1)
    sp500_return = data['SP500'].pct_change()
    gold_return = data['Gold'].pct_change()
    strategy_return = pd.Series(np.where(shifted_signal == 1, sp500_return, gold_return), index=data.index)
    
    # Aplicar comisiones
    commission_cost = trades * commission_rate
    final_returns = strategy_return - commission_cost
    
    return final_returns, trades.sum(), ratio, ma, signal

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
        'Retorno Total': total_return, 
        'Retorno Anualizado': annualized_return,
        'Volatilidad Anualizada': annualized_volatility, 
        'Ratio de Sharpe': sharpe_ratio,
        'Máximo Drawdown': max_drawdown
    })

# Sidebar con controles
st.sidebar.header("⚙️ Configuración de la Estrategia")

# Período fijo de media móvil
ma_period = 140

commission_rate = st.sidebar.slider(
    "Comisión por Operación (%)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.05, 
    step=0.01,
    help="Porcentaje de comisión aplicado en cada trade"
) / 100

# Obtener y procesar datos
data = get_data()

if data is not None:
    # Calcular estrategia
    strategy_returns, total_trades, ratio, ma, signal = calculate_strategy_returns(data, ma_period, commission_rate)
    
    # Agregar datos calculados al DataFrame
    data['Strategy_Return'] = strategy_returns
    data['SP500_Return'] = data['SP500'].pct_change()
    data['Gold_Return'] = data['Gold'].pct_change()
    data['Ratio'] = ratio
    data['MA'] = ma
    data['Signal'] = signal
    
    # Calcular retornos acumulados
    for col in ['Strategy', 'SP500', 'Gold']:
        data[f'{col}_Cum_Return'] = (1 + data[f'{col}_Return']).cumprod()
    
    # GRÁFICO 1: Rendimiento Acumulado
    st.subheader("📈 Comparación de Rendimiento Acumulado")
    
    fig_perf = go.Figure()
    
    fig_perf.add_trace(go.Scatter(
        x=data.index, 
        y=data['SP500_Cum_Return'],
        name='S&P 500',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    fig_perf.add_trace(go.Scatter(
        x=data.index, 
        y=data['Gold_Cum_Return'],
        name='Oro',
        line=dict(color='gold', width=2),
        opacity=0.8
    ))
    
    fig_perf.add_trace(go.Scatter(
        x=data.index, 
        y=data['Strategy_Cum_Return'],
        name=f'Estrategia (MA {ma_period})',
        line=dict(color='green', width=3)
    ))
    
    fig_perf.update_layout(
        title='Rendimiento Acumulado (Escala Logarítmica)',
        xaxis_title='Fecha',
        yaxis_title='Retorno Acumulado',
        yaxis_type="log",
        height=650,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # GRÁFICO 2: Ratio S&P 500/Oro y Media Móvil
    st.subheader("📊 Ratio S&P 500/Oro vs Media Móvil")
    
    fig_ratio = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Ratio S&P 500/Oro y Media Móvil', 'Señales de Trading'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Ratio y MA
    fig_ratio.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Ratio'],
            name='Ratio S&P 500/Oro',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig_ratio.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['MA'],
            name=f'Media Móvil ({ma_period}d)',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Señales de trading
    fig_ratio.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Signal'],
            name='Señal (1=S&P500, 0=Oro)',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,128,0,0.2)'
        ),
        row=2, col=1
    )
    
    fig_ratio.update_layout(
        height=600,
        title_text="Análisis del Ratio y Señales de Trading",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig_ratio.update_xaxes(title_text="Fecha", row=2, col=1)
    fig_ratio.update_yaxes(title_text="Ratio", row=1, col=1)
    fig_ratio.update_yaxes(title_text="Señal", row=2, col=1)
    
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Calcular métricas
    metrics = pd.DataFrame({
        'Estrategia': calculate_metrics(data['Strategy_Return']),
        'S&P 500': calculate_metrics(data['SP500_Return']),
        'Oro': calculate_metrics(data['Gold_Return'])
    }).T
    
    # Mostrar métricas en columnas
    st.subheader("📊 Métricas de Rendimiento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 Estrategia")
        st.metric("Retorno Total", f"{metrics.loc['Estrategia', 'Retorno Total']:.2%}")
        st.metric("Retorno Anualizado", f"{metrics.loc['Estrategia', 'Retorno Anualizado']:.2%}")
        st.metric("Volatilidad", f"{metrics.loc['Estrategia', 'Volatilidad Anualizada']:.2%}")
        st.metric("Ratio Sharpe", f"{metrics.loc['Estrategia', 'Ratio de Sharpe']:.2f}")
        st.metric("Máximo Drawdown", f"{metrics.loc['Estrategia', 'Máximo Drawdown']:.2%}")
    
    with col2:
        st.markdown("### 🔵 S&P 500")
        st.metric("Retorno Total", f"{metrics.loc['S&P 500', 'Retorno Total']:.2%}")
        st.metric("Retorno Anualizado", f"{metrics.loc['S&P 500', 'Retorno Anualizado']:.2%}")
        st.metric("Volatilidad", f"{metrics.loc['S&P 500', 'Volatilidad Anualizada']:.2%}")
        st.metric("Ratio Sharpe", f"{metrics.loc['S&P 500', 'Ratio de Sharpe']:.2f}")
        st.metric("Máximo Drawdown", f"{metrics.loc['S&P 500', 'Máximo Drawdown']:.2%}")
    
    with col3:
        st.markdown("### 🟡 Oro")
        st.metric("Retorno Total", f"{metrics.loc['Oro', 'Retorno Total']:.2%}")
        st.metric("Retorno Anualizado", f"{metrics.loc['Oro', 'Retorno Anualizado']:.2%}")
        st.metric("Volatilidad", f"{metrics.loc['Oro', 'Volatilidad Anualizada']:.2%}")
        st.metric("Ratio Sharpe", f"{metrics.loc['Oro', 'Ratio de Sharpe']:.2f}")
        st.metric("Máximo Drawdown", f"{metrics.loc['Oro', 'Máximo Drawdown']:.2%}")
    
    # Información adicional
    st.subheader("ℹ️ Información Adicional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_commission_paid = total_trades * commission_rate
        st.info(f"**Operaciones totales:** {total_trades:.0f}")
        st.info(f"**Comisiones pagadas:** {total_commission_paid:.2%}")
    
    with col2:
        current_ratio = data['SP500'].iloc[-1] / data['Gold'].iloc[-1]
        current_ma = data['MA'].iloc[-1]
        current_position = "🟢 Largo en S&P 500" if current_ratio > current_ma else "🟡 Largo en Oro"
        st.success(f"**Posición actual:** {current_position}")
        st.info(f"**Ratio actual:** {current_ratio:.2f}")
        st.info(f"**MA actual:** {current_ma:.2f}")
    
    # Tabla de métricas detallada
    with st.expander("📋 Ver tabla completa de métricas"):
        st.dataframe(
            metrics.style.format({
                'Retorno Total': '{:.2%}',
                'Retorno Anualizado': '{:.2%}',
                'Volatilidad Anualizada': '{:.2%}',
                'Ratio de Sharpe': '{:.2f}',
                'Máximo Drawdown': '{:.2%}'
            }),
            use_container_width=True
        )
    
    # Información sobre la estrategia
    with st.expander("📖 Acerca de la Estrategia"):
        st.markdown("""
        **Estrategia de Media Móvil S&P 500 vs Oro**
        
        Esta estrategia utiliza el ratio entre el S&P 500 y el precio del oro para determinar cuándo cambiar entre estos dos activos:
        
        - **Señal de compra S&P 500**: Cuando el ratio S&P 500/Oro está por encima de su media móvil
        - **Señal de compra Oro**: Cuando el ratio está por debajo de su media móvil
        - **Comisiones**: Se aplican en cada cambio de posición
        
        La estrategia busca aprovechar los ciclos relativos entre acciones y oro, típicamente:
        - Oro tiende a rendir mejor en períodos de incertidumbre económica
        - S&P 500 tiende a rendir mejor en períodos de crecimiento económico
        """)

else:
    st.error("No se pudieron cargar los datos. Por favor, inténtalo de nuevo más tarde.")

# Footer
st.markdown("---")
st.markdown("*Desarrollado con ❤️ usando Streamlit y Plotly*")
