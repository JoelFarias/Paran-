import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import unicodedata
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Vale do Ribeira - PR",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f8f8fc;
    padding: 2rem;
    font-family: 'Segoe UI', sans-serif;
    color: #333333;
}

.stButton > button {
    background-color: #e6f2ff;
    color: #2c3e50;
    border: 2px solid #bde0ff;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background-color: #cce7ff;
    color: #1a252f;
}

h1, h2, h3 {
    color: #4a4a4a;
}
h1 {
    font-size: 2.2rem;
    border-bottom: 2px solid #e6e6fa;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: #f0f0f8;
    color: #333;
    border-radius: 0.5rem 0.5rem 0 0;
    padding: 0.5rem 1rem;
    margin-right: 0.25rem;
    font-weight: bold;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #e6e6fa;
    color: #111;
}

.stExpander > details {
    background-color: #f5f5ff;
    border: 1px solid #e0e0f0;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

CUSTOM_COLORS = ["#B4C7E7", "#F4B2B0", "#B5E7A0", "#FFD1DC", "#C2A5F5", "#A8DADC", "#F7D794", "#D4A574", "#B2E6CE", "#F9E79F"]

def apply_palette(fig: go.Figure, palette: str = "custom") -> go.Figure:
    seq = CUSTOM_COLORS
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'marker'):
            if hasattr(trace.marker, 'color') and trace.marker.color is None:
                trace.marker.color = seq[i % len(seq)]
            elif isinstance(trace.marker.color, list):
                trace.marker.color = [seq[i % len(seq)] for i in range(len(trace.marker.color))]
        elif hasattr(trace, 'line'):
            if hasattr(trace.line, 'color') and trace.line.color is None:
                trace.line.color = seq[i % len(seq)]
    return fig

def _apply_layout(fig: go.Figure, title: str, title_size: int = 16) -> go.Figure:
    fig = apply_palette(fig)
    fig.update_layout(
        template="plotly_white",
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font_size": title_size
        },
        paper_bgcolor="white",   
        plot_bgcolor="white",     
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCC",
            borderwidth=1,
            font=dict(size=10)
        )
    )
    return fig

def wrap_label(name, width=30):
    if pd.isna(name): 
        return ""
    return "<br>".join(textwrap.wrap(str(name), width))

def truncate_text(text, max_chars=25):
    if pd.isna(text): 
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars-3] + "..."

def normalizar_string(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')
    return s.upper()

@st.cache_data
def carregar_csv(caminho: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar {caminho}: {e}")
        return pd.DataFrame()

def filtrar_alertas_vale_ribeira(df_alertas):
    municipios_vale = [
        "Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", 
        "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"
    ]
    
    if df_alertas.empty or 'MUNICIPIO' not in df_alertas.columns:
        return pd.DataFrame()
    
    return df_alertas[df_alertas['MUNICIPIO'].isin(municipios_vale)].copy()

def filtrar_queimadas_vale_ribeira(df_queimadas):
    """Filtra dados de queimadas para os municípios do Vale do Ribeira"""
    municipios_vale = [
        "ADRIANÓPOLIS", "BOCAIÚVA DO SUL", "CERRO AZUL", 
        "DOUTOR ULYSSES", "ITAPERUÇU", "RIO BRANCO DO SUL", "TUNAS DO PARANÁ"
    ]
    
    if df_queimadas.empty or 'Municipio' not in df_queimadas.columns:
        return pd.DataFrame()
    
    # Normalizar nomes dos municípios
    df_queimadas = df_queimadas.copy()
    df_queimadas['Municipio_norm'] = df_queimadas['Municipio'].str.upper().str.strip()
    
    return df_queimadas[df_queimadas['Municipio_norm'].isin(municipios_vale)].copy()

def criar_graficos_queimadas(df_queimadas):
    """Cria gráficos para análise de queimadas"""
    graficos = {}
    
    if df_queimadas.empty:
        fig_vazio = go.Figure()
        fig_vazio.add_annotation(
            text="Dados não disponíveis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray")
        )
        fig_vazio.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return {
            'temporal': fig_vazio,
            'top_risco': fig_vazio,
            'top_precip': fig_vazio,
            'mapa': fig_vazio
        }
    
    df = df_queimadas.copy()
    
    # Converter DataHora para datetime se necessário
    if 'DataHora' in df.columns:
        df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')
    
    # 1. Evolução Temporal do Risco de Fogo
    if 'DataHora' in df.columns and 'RiscoFogo' in df.columns:
        df_temp = df.dropna(subset=['DataHora', 'RiscoFogo'])
        df_temp['RiscoFogo'] = pd.to_numeric(df_temp['RiscoFogo'], errors='coerce')
        df_temp = df_temp[df_temp['RiscoFogo'].between(0, 1)]
        
        if not df_temp.empty:
            df_temp = df_temp.set_index('DataHora')
            monthly_risco = df_temp['RiscoFogo'].resample('ME').mean().reset_index()
            monthly_risco['DataHora_str'] = monthly_risco['DataHora'].dt.to_period('M').astype(str)
            
            fig_temporal = px.line(
                monthly_risco,
                x='DataHora_str',
                y='RiscoFogo',
                markers=True,
                labels={'DataHora_str': 'Mês/Ano', 'RiscoFogo': 'Risco Médio de Fogo'}
            )
            fig_temporal.update_traces(
                line_color='#F4B2B0',
                marker_color='#F4B2B0',
                mode='lines+markers+text',
                text=monthly_risco['RiscoFogo'].round(3),
                textposition='top center'
            )
            fig_temporal.update_layout(height=400)
            graficos['temporal'] = _apply_layout(fig_temporal, "Evolução Temporal do Risco de Fogo", 16)
        else:
            graficos['temporal'] = go.Figure().update_layout(title="Evolução Temporal do Risco de Fogo - Sem dados")
    else:
        graficos['temporal'] = go.Figure().update_layout(title="Evolução Temporal do Risco de Fogo - Sem dados")
    
    # 2. Top Municípios por Risco de Fogo
    if 'Municipio' in df.columns and 'RiscoFogo' in df.columns:
        df_risco = df.copy()
        df_risco['RiscoFogo'] = pd.to_numeric(df_risco['RiscoFogo'], errors='coerce')
        df_risco = df_risco[df_risco['RiscoFogo'].between(0, 1)]
        
        if not df_risco.empty:
            top_risco = df_risco.groupby('Municipio')['RiscoFogo'].mean().nlargest(7).sort_values()
            
            fig_risco = go.Figure(go.Bar(
                y=top_risco.index,
                x=top_risco.values,
                orientation='h',
                marker_color='#FFD1DC',
                text=top_risco.values.round(3),
                textposition='outside'
            ))
            fig_risco.update_layout(
                height=400,
                xaxis_title='Risco Médio de Fogo',
                yaxis_title='Município'
            )
            graficos['top_risco'] = _apply_layout(fig_risco, "Municípios por Risco Médio de Fogo", 16)
        else:
            graficos['top_risco'] = go.Figure().update_layout(title="Municípios por Risco de Fogo - Sem dados")
    else:
        graficos['top_risco'] = go.Figure().update_layout(title="Municípios por Risco de Fogo - Sem dados")
    
    # 3. Top Municípios por Precipitação
    if 'Municipio' in df.columns and 'Precipitacao' in df.columns:
        df_precip = df.copy()
        df_precip['Precipitacao'] = pd.to_numeric(df_precip['Precipitacao'], errors='coerce')
        df_precip = df_precip[df_precip['Precipitacao'] >= 0]
        
        if not df_precip.empty:
            top_precip = df_precip.groupby('Municipio')['Precipitacao'].mean().nlargest(7).sort_values()
            
            fig_precip = go.Figure(go.Bar(
                y=top_precip.index,
                x=top_precip.values,
                orientation='h',
                marker_color='#B5E7A0',
                text=[f'{x:.1f} mm' for x in top_precip.values],
                textposition='outside'
            ))
            fig_precip.update_layout(
                height=400,
                xaxis_title='Precipitação Média (mm)',
                yaxis_title='Município'
            )
            graficos['top_precip'] = _apply_layout(fig_precip, "Municípios por Precipitação Média", 16)
        else:
            graficos['top_precip'] = go.Figure().update_layout(title="Municípios por Precipitação - Sem dados")
    else:
        graficos['top_precip'] = go.Figure().update_layout(title="Municípios por Precipitação - Sem dados")
    
    # 4. Mapa de Focos de Calor
    map_cols = ['Latitude', 'Longitude', 'RiscoFogo', 'Municipio']
    if all(col in df.columns for col in map_cols):
        df_map = df[map_cols + (['Precipitacao'] if 'Precipitacao' in df.columns else [])].copy()
        df_map = df_map.dropna(subset=['Latitude', 'Longitude', 'RiscoFogo', 'Municipio'])
        df_map['RiscoFogo'] = pd.to_numeric(df_map['RiscoFogo'], errors='coerce')
        df_map = df_map[df_map['RiscoFogo'].between(0, 1)]
        
        if 'Precipitacao' in df_map.columns:
            df_map['Precipitacao'] = pd.to_numeric(df_map['Precipitacao'], errors='coerce')
            df_map = df_map[df_map['Precipitacao'] >= 0]
        else:
            df_map['Precipitacao'] = 0
        
        if not df_map.empty:
            # Limitar a 10000 pontos para performance
            if len(df_map) > 10000:
                df_map = df_map.sample(10000, random_state=42)
            
            fig_mapa = px.scatter_map(
                df_map,
                lat='Latitude',
                lon='Longitude',
                color='RiscoFogo',
                size='Precipitacao',
                hover_name='Municipio',
                hover_data={'Latitude': False, 'Longitude': False, 'RiscoFogo': ':.3f', 'Precipitacao': ':.1f'},
                color_continuous_scale='YlOrRd',
                size_max=15,
                map_style="open-street-map",
                zoom=8,
                center={'lat': df_map['Latitude'].mean(), 'lon': df_map['Longitude'].mean()},
                height=500
            )
            fig_mapa.update_layout(coloraxis_showscale=True)
            graficos['mapa'] = _apply_layout(fig_mapa, "Distribuição dos Focos de Calor", 16)
        else:
            graficos['mapa'] = go.Figure().update_layout(title="Mapa de Focos de Calor - Sem dados")
    else:
        graficos['mapa'] = go.Figure().update_layout(title="Mapa de Focos de Calor - Sem dados")
    
    return graficos

def criar_ranking_queimadas(df_queimadas, indicador):
    """Cria ranking de municípios por indicadores de queimadas"""
    if df_queimadas.empty or 'Municipio' not in df_queimadas.columns:
        return pd.DataFrame()
    
    df = df_queimadas.copy()
    
    if indicador == "Maior Risco de Fogo":
        if 'RiscoFogo' not in df.columns:
            return pd.DataFrame()
        df['RiscoFogo'] = pd.to_numeric(df['RiscoFogo'], errors='coerce')
        df = df[df['RiscoFogo'].between(0, 1)]
        ranking = df.groupby('Municipio')['RiscoFogo'].agg(['mean', 'max', 'count']).round(3)
        ranking.columns = ['Risco Médio', 'Risco Máximo', 'Quantidade de Focos']
        ranking = ranking.sort_values('Risco Médio', ascending=False)
        
    elif indicador == "Maior Precipitação (evento)":
        if 'Precipitacao' not in df.columns:
            return pd.DataFrame()
        df['Precipitacao'] = pd.to_numeric(df['Precipitacao'], errors='coerce')
        df = df[df['Precipitacao'] >= 0]
        ranking = df.groupby('Municipio')['Precipitacao'].agg(['mean', 'max', 'count']).round(1)
        ranking.columns = ['Precipitação Média (mm)', 'Precipitação Máxima (mm)', 'Quantidade de Registros']
        ranking = ranking.sort_values('Precipitação Máxima (mm)', ascending=False)
        
    elif indicador == "Máx. Dias Sem Chuva":
        if 'DiaSemChuva' not in df.columns:
            return pd.DataFrame()
        df['DiaSemChuva'] = pd.to_numeric(df['DiaSemChuva'], errors='coerce')
        df = df[df['DiaSemChuva'] >= 0]
        ranking = df.groupby('Municipio')['DiaSemChuva'].agg(['mean', 'max', 'count']).round(1)
        ranking.columns = ['Dias Médios Sem Chuva', 'Máx. Dias Sem Chuva', 'Quantidade de Registros']
        ranking = ranking.sort_values('Máx. Dias Sem Chuva', ascending=False)
    else:
        return pd.DataFrame()
    
    return ranking.reset_index()

def criar_cards_csv(df_cnuc, df_sigef, df_alertas):
    try:
        municipios_vale = [
            "Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", 
            "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"
        ]
        
        total_municipios = 7
        
        if not df_cnuc.empty and 'ha_total' in df_cnuc.columns:
            area_total_ucs_ha = pd.to_numeric(df_cnuc['ha_total'], errors='coerce').sum()
        else:
            area_total_ucs_ha = 0
        
        area_alertas_ha = 0
        contagem_alerta = 0
        
        if not df_alertas.empty and 'AREAHA' in df_alertas.columns:
            df_alertas_vale = filtrar_alertas_vale_ribeira(df_alertas)
            if not df_alertas_vale.empty:
                contagem_alerta = len(df_alertas_vale)
                area_alertas_ha = pd.to_numeric(df_alertas_vale['AREAHA'], errors='coerce').sum()
        
        contagem_sigef = 0
        if not df_sigef.empty:
            contagem_sigef = len(df_sigef)
        
        return (
            area_alertas_ha,
            contagem_sigef,
            total_municipios,
            contagem_alerta,
            area_total_ucs_ha
        )
        
    except Exception as e:
        st.error(f"Erro ao calcular indicadores: {e}")
        return (0.0, 0, 7, 0, 0.0)

def fig_sobreposicoes_csv(df_cnuc):
    if df_cnuc.empty or 'nome_uc' not in df_cnuc.columns:
        return go.Figure()

    df = df_cnuc.copy()
    
    if 'ha_total' in df.columns:
        df['area_ha'] = pd.to_numeric(df['ha_total'], errors='coerce')
    else:
        df['area_ha'] = 0
        
    df = df.sort_values("area_ha", ascending=False)
    df["uc_short"] = df["nome_uc"].apply(lambda x: wrap_label(x, 15))
    
    fig = px.bar(
        df,
        x="uc_short",
        y="area_ha",
        labels={"area_ha":"Área (ha)","uc_short":"UC"},
        text_auto=True,
    )

    fig.update_traces(
        customdata=np.column_stack([df.nome_uc]),
        hovertemplate="<b>%{customdata[0]}</b><br>Área: %{y:,.0f} ha<extra></extra>"
    )

    fig.update_xaxes(tickangle=0, tickfont=dict(size=9), title_text="")
    fig.update_yaxes(title_text="Área (ha)", tickfont=dict(size=9))
    fig.update_layout(height=400)

    return _apply_layout(fig, title="Área das Unidades de Conservação", title_size=16)

def fig_distribuicao_sigef(df_sigef: pd.DataFrame) -> go.Figure:
    try:
        if df_sigef.empty or 'municipio_' not in df_sigef.columns:
            return go.Figure()
        
        municipios_map = {
            4100103: "Adrianópolis",
            4102703: "Bocaiúva do Sul", 
            4104659: "Cerro Azul",
            4107405: "Doutor Ulysses",
            4111258: "Itaperuçu",
            4122404: "Rio Branco do Sul",
            4127700: "Tunas do Paraná"
        }
        
        df_count = df_sigef['municipio_'].value_counts().reset_index()
        df_count.columns = ['municipio_cod', 'quantidade']
        df_count['municipio'] = df_count['municipio_cod'].map(municipios_map)
        df_count = df_count.dropna(subset=['municipio'])
        df_count = df_count.sort_values('quantidade', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_count['municipio'],
            x=df_count['quantidade'],
            orientation='h',
            marker_color=CUSTOM_COLORS[0],
            text=df_count['quantidade'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Registros SIGEF: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title="Quantidade de Registros SIGEF",
            yaxis_title="",
            showlegend=False
        )
        
        return _apply_layout(fig, title="Distribuição de SIGEF por Município", title_size=14)
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico de SIGEF: {e}")
        return go.Figure()

def mostrar_tabela_unificada_csv(df_alertas, df_sigef, df_cnuc):
    try:
        municipios_vale = [
            "Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", 
            "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"
        ]
        
        municipios_map = {
            4100103: "Adrianópolis",
            4102703: "Bocaiúva do Sul", 
            4104659: "Cerro Azul",
            4107405: "Doutor Ulysses",
            4111258: "Itaperuçu",
            4122404: "Rio Branco do Sul",
            4127700: "Tunas do Paraná"
        }
        
        alertas_data = {}
        if not df_alertas.empty and 'MUNICIPIO' in df_alertas.columns:
            df_alertas_vale = filtrar_alertas_vale_ribeira(df_alertas)
            for municipio in municipios_vale:
                dados_mun = df_alertas_vale[df_alertas_vale['MUNICIPIO'] == municipio]
                if not dados_mun.empty and 'AREAHA' in dados_mun.columns:
                    area_total = pd.to_numeric(dados_mun['AREAHA'], errors='coerce').sum()
                    alertas_data[municipio] = area_total
                else:
                    alertas_data[municipio] = 0
        
        sigef_data = {}
        if not df_sigef.empty and 'municipio_' in df_sigef.columns:
            for municipio in municipios_vale:
                cod_mun = None
                for cod, nome in municipios_map.items():
                    if nome == municipio:
                        cod_mun = cod
                        break
                
                if cod_mun:
                    dados_mun = df_sigef[df_sigef['municipio_'] == cod_mun]
                    sigef_data[municipio] = len(dados_mun)
                else:
                    sigef_data[municipio] = 0
        
        cnuc_data = {}
        if not df_cnuc.empty and 'municipio' in df_cnuc.columns:
            for municipio in municipios_vale:
                municipio_normalizado = municipio.upper()
                dados_mun = df_cnuc[df_cnuc['municipio'].str.upper().str.contains(municipio_normalizado, na=False)]
                if not dados_mun.empty and 'ha_total' in dados_mun.columns:
                    area_total = pd.to_numeric(dados_mun['ha_total'], errors='coerce').sum()
                    cnuc_data[municipio] = area_total
                else:
                    cnuc_data[municipio] = 0
        
        df_unificado = pd.DataFrame(index=municipios_vale)
        df_unificado['Alertas (ha)'] = [alertas_data.get(mun, 0) for mun in municipios_vale]
        df_unificado['SIGEF (registros)'] = [sigef_data.get(mun, 0) for mun in municipios_vale]
        df_unificado['CNUC (ha)'] = [cnuc_data.get(mun, 0) for mun in municipios_vale]
        
        totais = df_unificado.sum()
        df_unificado.loc['TOTAL'] = totais
        
        df_unificado['Alertas (ha)'] = df_unificado['Alertas (ha)'].round(1)
        df_unificado['CNUC (ha)'] = df_unificado['CNUC (ha)'].round(1)
        
        st.dataframe(
            df_unificado,
            use_container_width=True,
            column_config={
                "Alertas (ha)": st.column_config.NumberColumn(
                    "Alertas (ha)",
                    format="%.1f"
                ),
                "SIGEF (registros)": st.column_config.NumberColumn(
                    "SIGEF (registros)",
                    format="%d"
                ),
                "CNUC (ha)": st.column_config.NumberColumn(
                    "CNUC (ha)",
                    format="%.1f"
                ),
            }
        )
        
    except Exception as e:
        st.error(f"Erro ao criar tabela unificada: {e}")

def fig_desmatamento_uc_csv(df_cnuc: pd.DataFrame, df_alertas: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="Análise de sobreposição UC x Alertas requer dados geográficos.<br>Disponível apenas com shapefiles.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=14, color="gray"),
        align="center"
    )
    fig.update_layout(
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return _apply_layout(fig, title="Área de Alertas por UC (Requer Dados Geográficos)", title_size=16)

def fig_desmatamento_temporal_csv(df_alertas: pd.DataFrame) -> go.Figure:
    if df_alertas.empty or 'DATADETEC' not in df_alertas.columns:
        fig = go.Figure()
        fig.update_layout(title="Evolução Temporal de Alertas (Desmatamento)",
                          xaxis_title="Data", yaxis_title="Área (ha)")
        return _apply_layout(fig, title="Evolução Temporal de Alertas (Desmatamento)", title_size=16)

    df_alertas = df_alertas.copy()
    df_alertas['DATADETEC'] = pd.to_datetime(df_alertas['DATADETEC'], errors='coerce')
    df_alertas['AREAHA'] = pd.to_numeric(df_alertas['AREAHA'], errors='coerce')

    df_valid_dates = df_alertas.dropna(subset=['DATADETEC', 'AREAHA'])

    if df_valid_dates.empty:
         fig = go.Figure()
         fig.update_layout(title="Evolução Temporal de Alertas (Desmatamento)",
                          xaxis_title="Data", yaxis_title="Área (ha)")
         return _apply_layout(fig, title="Evolução Temporal de Alertas (Desmatamento)", title_size=16)

    df_monthly = df_valid_dates.set_index('DATADETEC').resample('ME')['AREAHA'].sum().reset_index()
    df_monthly['DATADETEC'] = df_monthly['DATADETEC'].dt.to_period('M').astype(str)

    fig = px.line(
        df_monthly,
        x='DATADETEC',
        y='AREAHA',
        labels={"AREAHA":"Área (ha)","DATADETEC":"Mês/Ano"},
        markers=True,
        text='AREAHA'
    )

    fig.update_traces(
        mode='lines+markers+text',
        textposition='top center',
        texttemplate='%{text:,.0f}',
        hovertemplate=(
            "Mês/Ano: %{x}<br>"
            "Área de Alertas: %{y:,.0f} ha<extra></extra>"
        )
    )

    fig.update_xaxes(title_text="Mês/Ano", tickangle=45)
    fig.update_yaxes(title_text="Área (ha)")
    fig.update_layout(height=400)

    return _apply_layout(fig, title="Evolução Mensal de Alertas (Desmatamento)", title_size=16)

def fig_desmatamento_municipal_csv(df_alertas: pd.DataFrame) -> go.Figure:
    if df_alertas.empty or 'MUNICIPIO' not in df_alertas.columns:
        return go.Figure()
    
    municipios_vale = [
        "Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", 
        "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"
    ]
    
    df_vale = df_alertas[df_alertas['MUNICIPIO'].isin(municipios_vale)]
    
    if df_vale.empty:
        return go.Figure()
        
    df_mun = df_vale.groupby('MUNICIPIO').agg({
        'AREAHA': ['count', 'sum']
    }).round(1)
    
    df_mun.columns = ['Quantidade', 'Área_Total_ha']
    df_mun = df_mun.reset_index()
    
    df_todos_municipios = pd.DataFrame({'MUNICIPIO': municipios_vale})
    df_mun = df_todos_municipios.merge(df_mun, on='MUNICIPIO', how='left').fillna(0)
    
    df_mun = df_mun.sort_values('Área_Total_ha', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_mun['MUNICIPIO'],
        x=df_mun['Área_Total_ha'],
        orientation='h',
        name='Área de Alertas (ha)',
        marker_color=CUSTOM_COLORS[0],
        text=df_mun['Área_Total_ha'].apply(lambda x: f"{x:.0f}" if x > 0 else "0"),
        textposition='outside',
        texttemplate='%{text} ha',
        hovertemplate='<b>%{y}</b><br>Área: %{x:.0f} ha<br>Quantidade: %{customdata}<extra></extra>',
        customdata=df_mun['Quantidade']
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Área de Alertas (hectares)",
        yaxis_title="",
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
    return _apply_layout(fig, title="Desmatamento por Município do Vale do Ribeira", title_size=16)

# --- NOVA FUNÇÃO: Mapa de Alertas de Desmatamento ---
def fig_mapa_alertas_desmatamento(df_alertas: pd.DataFrame) -> go.Figure:
    municipios_vale = [
        "Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", 
        "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"
    ]
    coordenadas_municipios = {
        "Adrianópolis": {"lat": -24.6577, "lon": -48.9933},
        "Bocaiúva do Sul": {"lat": -25.2069, "lon": -49.1172},
        "Cerro Azul": {"lat": -24.8267, "lon": -49.2597},
        "Doutor Ulysses": {"lat": -25.4406, "lon": -49.2775},
        "Itaperuçu": {"lat": -25.2397, "lon": -49.3442},
        "Rio Branco do Sul": {"lat": -25.1858, "lon": -49.3106},
        "Tunas do Paraná": {"lat": -24.9639, "lon": -49.1053}
    }
    
    if df_alertas.empty or 'MUNICIPIO' not in df_alertas.columns or 'AREAHA' not in df_alertas.columns:
        return go.Figure()
    
    df_vale = df_alertas[df_alertas['MUNICIPIO'].isin(municipios_vale)].copy()
    if df_vale.empty:
        return go.Figure()
    
    df_vale['AREAHA'] = pd.to_numeric(df_vale['AREAHA'], errors='coerce')
    df_mun = df_vale.groupby('MUNICIPIO').agg({'AREAHA': 'sum', 'MUNICIPIO': 'count'}).rename(columns={'MUNICIPIO': 'QTD_ALERTAS', 'AREAHA': 'AREA_TOTAL'}).reset_index()
    df_mun['lat'] = df_mun['MUNICIPIO'].map(lambda x: coordenadas_municipios[x]['lat'])
    df_mun['lon'] = df_mun['MUNICIPIO'].map(lambda x: coordenadas_municipios[x]['lon'])
    df_mun['hover'] = df_mun.apply(lambda row: f"<b>{row['MUNICIPIO']}</b><br>Área: {row['AREA_TOTAL']:.1f} ha<br>Alertas: {row['QTD_ALERTAS']}", axis=1)
    
    fig = go.Figure(go.Scattermapbox(
        lat=df_mun['lat'],
        lon=df_mun['lon'],
        mode='markers',
        marker=dict(
            size=12 + (df_mun['AREA_TOTAL'] / max(df_mun['AREA_TOTAL'].max(), 1)) * 30,
            color=df_mun['AREA_TOTAL'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Área de Alertas (ha)"),
            opacity=0.8
        ),
        text=df_mun['MUNICIPIO'],
        hovertemplate=df_mun['hover'] + "<extra></extra>",
        name="Alertas",
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=8,
            center=dict(lat=df_mun['lat'].mean(), lon=df_mun['lon'].mean())
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        title="Mapa de Alertas de Desmatamento por Município",
        showlegend=False,
        dragmode='pan'
    )
    
    # Configurar interatividade com mouse
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=8,
            center=dict(lat=df_mun['lat'].mean(), lon=df_mun['lon'].mean()),
            bearing=0,
            pitch=0
        )
    )
    
    return fig

df_alertas_raw = carregar_csv("Alertas_Vale_Ribeira.csv")
df_cnuc_raw = carregar_csv("cnuc.csv")
df_sigef_raw = carregar_csv("SIGEF_Vale_Ribeira.csv")
df_queimadas_raw = carregar_csv("Risco_Fogo.csv")

st.title("Dashboard Vale do Ribeira - Paraná (Versão CSV)")

debug_mode = False

tabs = st.tabs(["Sobreposições", "Desmatamento", "Queimadas"])

with tabs[0]:
    st.header("Sobreposições")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobreposições territoriais no Vale do Ribeira (PR):
        - Percentuais de alertas e registros SIGEF em relação às Unidades de Conservação
        - Distribuição por municípios
        - Áreas e contagens por Unidade de Conservação
        
        **Municípios analisados:** Adrianópolis, Bocaiúva do Sul, Cerro Azul, Doutor Ulysses, Itaperuçu, Rio Branco do Sul, Tunas do Paraná
        
        **Dados utilizados:**
        - Alertas: Alertas_Vale_Ribeira.csv
        - SIGEF: SIGEF_Vale_Ribeira.csv  
        - UCs: cnuc.csv
        """)

    area_alertas, contagem_sigef, total_unidades, contagem_alerta, area_cnuc = criar_cards_csv(df_cnuc_raw, df_sigef_raw, df_alertas_raw)
    
    cols = st.columns(5, gap="small")
    titulos = [
        ("Área Alertas", f"{area_alertas:.1f} ha", "Área total de alertas"),
        ("Registros SIGEF", f"{contagem_sigef}", "Total de registros SIGEF"),
        ("Municípios", f"{total_unidades}", "Total de municípios na análise"),
        ("Qtd Alertas", f"{contagem_alerta}", "Quantidade de alertas"),
        ("Área UCs", f"{area_cnuc:.1f} ha", "Área total das UCs")
    ]
    
    card_template = """
    <div style="
        background-color:#F9F9FF;
        border:1px solid #E0E0E0;
        padding:1rem;
        border-radius:8px;
        box-shadow:0 2px 4px rgba(0,0,0,0.1);
        text-align:center;
        height:100px;
        display:flex;
        flex-direction:column;
        justify-content:center;">
        <h5 style="margin:0; font-size:0.9rem;">{0}</h5>
        <p style="margin:0; font-size:1.2rem; font-weight:bold; color:#2F5496;">{1}</p>
        <small style="color:#666;">{2}</small>
    </div>
    """
    
    for col, (t, v, d) in zip(cols, titulos):
        col.markdown(card_template.format(t, v, d), unsafe_allow_html=True)

    st.divider()

    row1_charts = st.columns([1], gap="large")[0]
    with row1_charts:
        if not df_sigef_raw.empty:
            st.subheader("Distribuição de SIGEF")
            st.plotly_chart(fig_distribuicao_sigef(df_sigef_raw), use_container_width=True, height=350)
            st.caption("Figura 1.2: Distribuição de registros SIGEF por município do Vale do Ribeira.")
            
            with st.expander("Detalhes e Fonte da Figura 1.2"):
                st.write("""
                **Interpretação:**
                O gráfico apresenta a quantidade de registros no Sistema de Gestão Fundiária (SIGEF) por município.

                **Observações:**
                - Barras horizontais mostram quantidade de registros
                - Inclui apenas os 7 municípios do Vale do Ribeira
                - Ordenado por quantidade de registros

                **Fonte:** INCRA - Sistema de Gestão Fundiária (SIGEF).
                """)
        else:
            st.warning("Dados de SIGEF não disponíveis")

        if not df_cnuc_raw.empty:
            st.subheader("Áreas por UC")
            st.plotly_chart(fig_sobreposicoes_csv(df_cnuc_raw), use_container_width=True, height=350)
            st.caption("Figura 1.3: Distribuição de áreas por unidade de conservação.")
            
            with st.expander("Detalhes e Fonte da Figura 1.3"):
                st.write("""
                **Interpretação:**
                O gráfico mostra as áreas das unidades de conservação em hectares.

                **Observações:**
                - Área total da UC em hectares
                - Ordenado por área total
                - Dados reais do cadastro de UCs

                **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*.
                """)
        else:
            st.warning("Dados de UCs não disponíveis para gráficos")

    st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
        <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Tabela Unificada</h3>
        <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Visualização unificada dos dados de alertas, SIGEF e CNUC por município.</p>
    </div>""", unsafe_allow_html=True)
    
    mostrar_tabela_unificada_csv(df_alertas_raw, df_sigef_raw, df_cnuc_raw)
    
    st.caption("Tabela 1.1: Dados consolidados por município do Vale do Ribeira (PR) - Versão CSV.")
    with st.expander("Detalhes e Fonte da Tabela 1.1"):
        st.write("""
        **Interpretação:**
        A tabela apresenta os dados consolidados por município, incluindo:
        - Área de alertas em hectares
        - Quantidade de registros SIGEF
        - Área do CNUC em hectares

        **Observações:**
        - Alertas em hectares
        - SIGEF em quantidade de registros
        - CNUC em hectares
        - Totais na última linha

        **Fonte:** 
        - Alertas: Alertas_Vale_Ribeira.csv
        - SIGEF: SIGEF_Vale_Ribeira.csv
        - CNUC: cnuc.csv
        """)
    
    st.divider()

with tabs[1]:
    st.header("Desmatamento")

    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre áreas de alerta de desmatamento no Vale do Ribeira (PR):
        - Distribuição por Unidade de Conservação (estimada)
        - Evolução temporal
        - Distribuição por município

        **Municípios analisados:** Adrianópolis, Bocaiúva do Sul, Cerro Azul, Doutor Ulysses, Itaperuçu, Rio Branco do Sul, Tunas do Paraná

        Os dados são provenientes do arquivo Alertas_Vale_Ribeira.csv.
        """)
        st.markdown(
            "**Fonte Geral da Seção:** MapBiomas Alerta. Dados extraídos e compilados.",
            unsafe_allow_html=True
        )

    st.write("**Filtro Global:**")
    df_alertas_vale = filtrar_alertas_vale_ribeira(df_alertas_raw)
    
    if not df_alertas_vale.empty and 'ANODETEC' in df_alertas_vale.columns:
        anos_disponiveis = ['Todos'] + sorted(df_alertas_vale['ANODETEC'].dropna().unique().tolist())
        ano_global_selecionado = st.selectbox('Ano de Detecção:', anos_disponiveis, key="filtro_ano_global")

        if ano_global_selecionado != 'Todos':
            df_alertas_filtrado = df_alertas_vale[df_alertas_vale['ANODETEC'] == ano_global_selecionado].copy()
        else:
            df_alertas_filtrado = df_alertas_vale.copy()
    else:
        df_alertas_filtrado = df_alertas_vale.copy()
        if df_alertas_vale.empty:
            st.info("Nenhum alerta encontrado nos municípios do Vale do Ribeira.")
        else:
            st.info("Coluna de ano não disponível. Exibindo todos os dados dos municípios do Vale do Ribeira.")

    st.divider()

    col_charts1, col_charts2 = st.columns([1, 1], gap="large")

    with col_charts1:
        if not df_cnuc_raw.empty and not df_alertas_filtrado.empty:
            fig_desmat_uc = fig_desmatamento_uc_csv(df_cnuc_raw, df_alertas_filtrado)
            if fig_desmat_uc and fig_desmat_uc.data:
                st.subheader("Área de Alertas por UC")
                st.plotly_chart(fig_desmat_uc, use_container_width=True, height=400, key="desmat_uc_chart")
                st.caption("Figura 2.1: Área estimada de alertas de desmatamento por unidade de conservação.")
                with st.expander("Detalhes e Fonte da Figura 2.1"):
                    st.write("""
                    **Limitação Importante:**
                    Este gráfico requer dados geográficos para análise precisa de sobreposições.

                    **Motivo:**
                    - Calcular sobreposições entre alertas e UCs requer coordenadas geográficas precisas
                    - Dados tabulares não contêm informações espaciais suficientes
                    - Análise apresentada é uma estimativa

                    **Recomendação:**
                    Para análise espacial completa, utilize dados geográficos detalhados.

                    **Fonte:** Limitação técnica - dados geográficos necessários.
                    """)
        else:
            st.warning("Dados de UCs ou Alertas não disponíveis para esta análise.")

        if not df_alertas_filtrado.empty:
            st.subheader("Desmatamento por Município")
            fig_mun = fig_desmatamento_municipal_csv(df_alertas_filtrado)
            if fig_mun and fig_mun.data:
                st.plotly_chart(fig_mun, use_container_width=True, height=400, key="desmat_municipal_chart")
                st.caption("Figura 2.2: Área de alertas por município do Vale do Ribeira.")
                with st.expander("Detalhes e Fonte da Figura 2.2"):
                    st.write("""
                    **Interpretação:**
                    O gráfico mostra a distribuição de alertas de desmatamento por município do Vale do Ribeira.

                    **Observações:**
                    - Barras horizontais representam área total em hectares
                    - Hover mostra quantidade de alertas e área total
                    - Inclui todos os 7 municípios da região
                    - Ordenado do maior para o menor

                    **Fonte:** Alertas_Vale_Ribeira.csv.
                    """)
            else:
                st.info("Nenhum alerta encontrado nos municípios do Vale do Ribeira para o período selecionado.")

    with col_charts2:
        st.subheader("Mapa de Alertas de Desmatamento")
        if not df_alertas_vale.empty:
            fig_mapa = fig_mapa_alertas_desmatamento(df_alertas_vale)
            if fig_mapa and fig_mapa.data:
                st.plotly_chart(fig_mapa, use_container_width=True, height=400, key="mapa_desmatamento")
                st.caption("Figura 2.3: Mapa de alertas de desmatamento por município do Vale do Ribeira.")
                with st.expander("Detalhes e Fonte da Figura 2.3"):
                    st.write("""
                    **Interpretação:**
                    O mapa mostra a distribuição espacial dos alertas de desmatamento por município do Vale do Ribeira.

                    **Observações:**
                    - Tamanho dos marcadores proporcional à área total de alertas
                    - Cor dos marcadores indica intensidade (escala Viridis)
                    - Hover mostra município, área total e quantidade de alertas
                    - Coordenadas aproximadas dos centros municipais

                    **Limitações:**
                    - Posições são centros aproximados dos municípios
                    - Não mostra a localização exata dos alertas individuais
                    - Para análise espacial detalhada, utilize dados geográficos completos

                    **Fonte:** Alertas_Vale_Ribeira.csv com coordenadas aproximadas.
                    """)
            else:
                st.info("Dados insuficientes para gerar o mapa.")
        else:
            st.warning("Dados de alertas não disponíveis para o mapa.")

    st.divider()

    # Gráfico temporal ocupando largura completa
    st.subheader("Evolução Temporal de Alertas")
    if not df_alertas_vale.empty:
        fig_desmat_temp = fig_desmatamento_temporal_csv(df_alertas_vale)
        if fig_desmat_temp and fig_desmat_temp.data:
            st.plotly_chart(fig_desmat_temp, use_container_width=True, height=400, key="desmat_temporal_chart")
            st.caption("Figura 2.4: Evolução mensal dos alertas de desmatamento.")
            with st.expander("Detalhes e Fonte da Figura 2.4"):
                st.write("""
                **Interpretação:**
                O gráfico mostra a evolução temporal dos alertas de desmatamento no Vale do Ribeira.

                **Observações:**
                - Linha conecta os valores mensais
                - Pontos marcam cada mês com dados
                - Valores exibidos sobre os pontos
                - Agregação mensal da área de alertas

                **Fonte:** Alertas_Vale_Ribeira.csv.
                """)
        else:
            st.info("Dados temporais não disponíveis para este gráfico.")
    else:
        st.warning("Dados de Alertas não disponíveis para análise temporal.")

    st.divider()

    st.subheader("Ranking de Municípios por Desmatamento")
    if not df_alertas_filtrado.empty:
        required_ranking_cols = ['MUNICIPIO', 'AREAHA', 'ANODETEC']
        if all(col in df_alertas_filtrado.columns for col in required_ranking_cols):
            df_alertas_filtrado['AREAHA'] = pd.to_numeric(df_alertas_filtrado['AREAHA'], errors='coerce')

            ranking_municipios = df_alertas_filtrado.groupby('MUNICIPIO', observed=False).agg({
                'AREAHA': ['sum', 'count', 'mean'],
                'ANODETEC': ['min', 'max']
            }).round(2)
            
            ranking_municipios.columns = ['Área Total (ha)', 'Qtd Alertas', 'Área Média (ha)', 'Ano Min', 'Ano Max']
            ranking_municipios = ranking_municipios.reset_index()
            ranking_municipios = ranking_municipios.sort_values('Área Total (ha)', ascending=False)
            ranking_municipios.insert(0, 'Posição', range(1, len(ranking_municipios) + 1))

            ranking_municipios['Área Total (ha)'] = ranking_municipios['Área Total (ha)'].apply(lambda x: f"{x:,.2f}")
            ranking_municipios['Área Média (ha)'] = ranking_municipios['Área Média (ha)'].apply(lambda x: f"{x:.2f}")

            st.dataframe(
                ranking_municipios.head(10),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.caption("Tabela 2.1: Ranking de municípios por área de desmatamento.")
            with st.expander("Detalhes e Fonte da Tabela 2.1"):
                st.write("""
                **Interpretação:**
                A tabela apresenta o ranking dos municípios por área total de desmatamento.

                **Colunas:**
                - Posição: Ranking por área total
                - Município: Nome do município
                - Área Total: Soma de todas as áreas de alertas (ha)
                - Qtd Alertas: Número total de alertas
                - Área Média: Área média por alerta (ha)
                - Ano Min/Max: Período de abrangência dos dados

                **Fonte:** Alertas_Vale_Ribeira.csv.
                """)

        else:
            st.info("Colunas necessárias não disponíveis para o ranking.")
    else:
        st.info("Dados não disponíveis para o ranking no período selecionado.")

with tabs[2]:
    st.header("Focos de Calor")

    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre focos de calor detectados por satélite no Vale do Ribeira (PR), incluindo:
        - Risco de fogo
        - Precipitação acumulada
        - Distribuição espacial
        - Evolução temporal

        **Municípios analisados:** Adrianópolis, Bocaiúva do Sul, Cerro Azul, Doutor Ulysses, Itaperuçu, Rio Branco do Sul, Tunas do Paraná

        Os dados são provenientes do arquivo Risco_Fogo.csv.
        """)
        st.markdown(
            "**Fonte Geral da Seção:** INPE – Programa Queimadas, 2025.",
            unsafe_allow_html=True
        )

    st.write("**Filtro Global:**")
    df_queimadas_vale = filtrar_queimadas_vale_ribeira(df_queimadas_raw)
    
    if not df_queimadas_vale.empty and 'DataHora' in df_queimadas_vale.columns:
        df_queimadas_vale['DataHora'] = pd.to_datetime(df_queimadas_vale['DataHora'], errors='coerce')
        df_queimadas_vale['Ano'] = df_queimadas_vale['DataHora'].dt.year
        anos_disponiveis = ['Todos'] + sorted(df_queimadas_vale['Ano'].dropna().unique().tolist())
        ano_global_selecionado = st.selectbox('Ano de Detecção:', anos_disponiveis, key="filtro_ano_global_queimadas")

        if ano_global_selecionado != 'Todos':
            df_queimadas_filtrado = df_queimadas_vale[df_queimadas_vale['Ano'] == ano_global_selecionado].copy()
            display_periodo = f"ano de {ano_global_selecionado}"
        else:
            df_queimadas_filtrado = df_queimadas_vale.copy()
            display_periodo = "todo o período histórico"
    else:
        df_queimadas_filtrado = df_queimadas_vale.copy()
        display_periodo = "todo o período disponível"
        if df_queimadas_vale.empty:
            st.info("Nenhum dado de queimadas encontrado nos municípios do Vale do Ribeira.")
        else:
            st.info("Coluna de data não disponível. Exibindo todos os dados dos municípios do Vale do Ribeira.")

    st.divider()

    # Gráficos de Queimadas
    if not df_queimadas_filtrado.empty:
        graficos_queimadas = criar_graficos_queimadas(df_queimadas_filtrado)

        st.subheader("Evolução Temporal do Risco de Fogo")
        if 'temporal' in graficos_queimadas:
            st.plotly_chart(graficos_queimadas['temporal'], use_container_width=True)
            st.caption(f"Figura 3.1: Evolução mensal do risco médio de fogo para {display_periodo}.")
        else:
            st.info("Dados insuficientes para gerar o gráfico temporal.")

        col_graficos1, col_graficos2 = st.columns(2, gap="large")

        with col_graficos1:
            st.subheader("Top Municípios por Risco Médio de Fogo")
            if 'top_risco' in graficos_queimadas:
                st.plotly_chart(graficos_queimadas['top_risco'], use_container_width=True)
            else:
                st.info("Dados insuficientes para gerar o gráfico de municípios por risco de fogo.")
            
            st.subheader("Top Municípios por Precipitação Acumulada")
            if 'top_precip' in graficos_queimadas:
                st.plotly_chart(graficos_queimadas['top_precip'], use_container_width=True)
            else:
                st.info("Dados insuficientes para gerar o gráfico de municípios por precipitação.")

        with col_graficos2:
            st.subheader("Mapa de Distribuição dos Focos de Calor")
            if 'mapa' in graficos_queimadas:
                st.plotly_chart(graficos_queimadas['mapa'], use_container_width=True, config={'scrollZoom': True})
            else:
                st.info("Dados insuficientes para gerar o mapa de focos de calor.")

        st.divider()

        # Ranking de Municípios por Queimadas
        st.header("Ranking de Municípios por Indicadores de Queimadas")
        st.caption("Classifica municípios pelo maior registro de cada indicador.")
        
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            pass  # Ano já selecionado acima
        with col_rank2:
            indicador_selecionado = st.selectbox(
                "Indicador para ranking:",
                ["Maior Risco de Fogo", "Maior Precipitação (evento)", "Máx. Dias Sem Chuva"],
                key="ranking_indicador"
            )

        periodo_rank = display_periodo.title()
        st.subheader(f"Ranking por {indicador_selecionado} ({periodo_rank})")

        ranking_queimadas = criar_ranking_queimadas(df_queimadas_filtrado, indicador_selecionado)
        
        if not ranking_queimadas.empty:
            st.dataframe(
                ranking_queimadas,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Sem dados válidos para este ranking.")
    else:
        st.error("Não foi possível carregar os dados de queimadas. Verifique se o arquivo Risco_Fogo.csv está disponível.")