import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import os
import numpy as np
import warnings
import textwrap
from functools import lru_cache

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Vale do Ribeira - PR",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ---------- Fundo geral do app ---------- */
[data-testid="stAppViewContainer"] {
    background-color: #fefcf9;
    padding: 2rem;
    font-family: 'Segoe UI', sans-serif;
    color: #333333;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #f3f0eb;
    border-right: 2px solid #d8d2ca;
}
[data-testid="stSidebar"] > div {
    padding: 1rem;
}

/* ---------- Botões ---------- */
.stButton > button {
    background-color: #cbe4d2;
    color: #2d3a2f;
    border: 2px solid #a6c4b2;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background-color: #b4d6c1;
    color: #1e2a21;
}

/* ---------- Títulos e textos ---------- */
h1, h2, h3 {
    color: #4a4a4a;
}
h1 {
    font-size: 2.2rem;
    border-bottom: 2px solid #d8d2ca;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab"] {
    background-color: #ebe7e1;
    color: #333;
    border-radius: 0.5rem 0.5rem 0 0;
    padding: 0.5rem 1rem;
    margin-right: 0.25rem;
    font-weight: bold;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #d6ccc2;
    color: #111;
}

/* ---------- Expander ---------- */
.stExpander > details {
    background-color: #f2eee9;
    border: 1px solid #ddd3c7;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ======================= CONFIGURAÇÕES =======================

PASTEL_COLORS = [
    "#B5E7A0", "#FFD1DC", "#E6E6FA", "#F0E68C", "#FFA07A", 
    "#98FB98", "#F5DEB3", "#DDA0DD", "#87CEEB", "#F0F8FF"
]

MUNICIPIOS_VALE_RIBEIRA = [
    "ADRIANÓPOLIS", "BOCAIÚVA DO SUL", "CERRO AZUL", 
    "DOUTOR ULYSSES", "ITAPERUÇU", "RIO BRANCO DO SUL", "TUNAS DO PARANÁ"
]

# Constantes
CRS_PROJECAO = "EPSG:31983"
CRS_GEOGRAFICO = "EPSG:4326"
CONVERSAO_M2_PARA_HA = 10000
CONVERSAO_KM2_PARA_HA = 100
MAX_PONTOS_MAPA = 10000
TOP_MUNICIPIOS_LIMITE = 7
LIMITE_OTIMIZACAO_POLIGONOS = 100
TAMANHO_AMOSTRA_MAPA = 50000

# Cache para operações geométricas
@st.cache_data
def calcular_intersecoes_cache(uc_id, sigef_ids):
    """Cache para cálculos de interseção pesados."""
    return {}

def aplicar_paleta(fig: go.Figure, paleta: str = "pastel") -> go.Figure:
    if paleta == "sobreposicoes":
        colors = ["#B5E7A0", "#98FB98", "#87CEEB", "#F5DEB3"]
    elif paleta == "desmatamento":
        colors = ["#FFB6C1", "#FFA07A", "#F0E68C", "#DDA0DD"]
    elif paleta == "queimadas":
        colors = ["#FFD1DC", "#F0E68C", "#FFA07A", "#E6E6FA"]
    elif paleta == "justica":
        colors = ["#FFE4B5", "#FFDAB9", "#F0E68C", "#DEB887"]
    else:
        colors = PASTEL_COLORS
    
    for i, trace in enumerate(fig.data):
        color = colors[i % len(colors)]
        if hasattr(trace, 'marker'):
            if trace.type == 'pie':
                if not hasattr(trace.marker, 'colors') or trace.marker.colors is None:
                    trace.marker.colors = colors[:len(trace.labels)] if hasattr(trace, 'labels') else colors
            else:
                trace.marker.color = color
        if hasattr(trace, 'line'):
            trace.line.color = color
    return fig

def _aplicar_layout(fig: go.Figure, titulo: str, tamanho_titulo: int = 16, paleta: str = "pastel") -> go.Figure:
    fig = aplicar_paleta(fig, paleta)
    fig.update_layout(
        template="plotly_white",
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font_size": tamanho_titulo},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CCC", borderwidth=1, font=dict(size=10)),
        autosize=True,
        font=dict(size=12),
        showlegend=True if len(fig.data) > 1 else False
    )
    return fig

def quebrar_rotulo(nome, largura=30):
    if pd.isna(nome): return ""
    return "<br>".join(textwrap.wrap(str(nome), largura))

def truncar_rotulo(nome, tamanho_max=20):
    if pd.isna(nome): return ""
    nome_str = str(nome)
    return nome_str[:tamanho_max] + "..." if len(nome_str) > tamanho_max else nome_str

# ======================= FUNÇÕES AUXILIARES =======================

def carregar_dados_processos(caminho_arquivo):
    """Carrega e processa dados de processos judiciais."""
    try:
        dataframe = pd.read_csv(caminho_arquivo, delimiter=';')
        dataframe['data_ajuizamento'] = pd.to_datetime(
            dataframe['data_ajuizamento'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        return dataframe.dropna(subset=['data_ajuizamento'])
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho_arquivo}")
        return pd.DataFrame()
    except Exception as erro:
        st.error(f"Erro ao carregar arquivo: {erro}")
        return pd.DataFrame()

def eh_dataframe_valido(dataframe, colunas_obrigatorias=None):
    """Verifica se dataframe é válido e contém colunas obrigatórias."""
    if dataframe.empty:
        return False
    
    if colunas_obrigatorias:
        return all(col in dataframe.columns for col in colunas_obrigatorias)
    
    return True

def normalizar_string(texto):
    """Remove acentos e normaliza string para maiúscula."""
    if pd.isna(texto): 
        return ""
    
    texto_limpo = str(texto).strip()
    texto_sem_acento = unicodedata.normalize('NFD', texto_limpo)
    texto_sem_acento = ''.join(
        char for char in texto_sem_acento 
        if unicodedata.category(char) != 'Mn'
    )
    return texto_sem_acento.upper()

def _criar_geodataframe_vazio():
    """Cria GeoDataFrame vazio com CRS padrão."""
    return gpd.GeoDataFrame(columns=['geometry']).set_crs(CRS_GEOGRAFICO)

def _normalizar_colunas(gdf):
    """Normaliza nomes das colunas para minúscula."""
    gdf.columns = [str(col).lower() for col in gdf.columns]
    return gdf

def _padronizar_nomes_colunas(gdf, caminho):
    if 'nome_uc' not in gdf.columns:
        if 'nome' in gdf.columns:
            gdf = gdf.rename(columns={'nome': 'nome_uc'})
        elif 'cnuc' in caminho:
            gdf['nome_uc'] = [f"UC_{i}" for i in range(len(gdf))]
    
    if 'municipio' not in gdf.columns:
        if 'município' in gdf.columns:
            gdf = gdf.rename(columns={'município': 'municipio'})
        elif 'nm_mun' in gdf.columns:
            gdf = gdf.rename(columns={'nm_mun': 'municipio'})
    
    return gdf

def _limpar_geometrias(gdf):
    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf_limpo = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty & gdf["geometry"].is_valid].copy()
    gdf_limpo['id'] = range(len(gdf_limpo))
    return gdf_limpo

def _calcular_areas(gdf, caminho):
    try:
        gdf_projetado = gdf.to_crs(CRS_PROJECAO)
        area_calculada_km2 = gdf_projetado.geometry.area / 1e6
        
        if "area_km2" not in gdf.columns:
            gdf["area_km2"] = area_calculada_km2
        else:
            gdf["area_km2"] = pd.to_numeric(
                gdf["area_km2"], errors='coerce'
            ).fillna(area_calculada_km2)
    except Exception as erro:
        st.warning(f"Não foi possível calcular área para {caminho}: {erro}")
        if "area_km2" not in gdf.columns:
            gdf["area_km2"] = 0
    
    return gdf

def _adicionar_colunas_padrao(gdf):
    colunas_padrao = ['alerta_km2', 'sigef_km2', 'c_alertas', 'c_sigef', 'ha_total', 'areaha']
    
    for coluna in colunas_padrao:
        if coluna not in gdf.columns:
            gdf[coluna] = 0
        else:
            gdf[coluna] = pd.to_numeric(gdf[coluna], errors='coerce').fillna(0)
    if gdf['ha_total'].sum() == 0 and 'area_km2' in gdf.columns:
        gdf['ha_total'] = gdf['area_km2'] * CONVERSAO_KM2_PARA_HA
    
    return gdf

def carregar_shapefile(caminho: str) -> gpd.GeoDataFrame:
    """Carrega e processa shapefile com validações e padronizações."""
    try:
        if not os.path.exists(caminho):
            st.warning(f"Arquivo não encontrado: {caminho}")
            return _criar_geodataframe_vazio()

        gdf = gpd.read_file(caminho)
        if gdf.empty:
            st.warning(f"Shapefile vazio: {caminho}")
            return _criar_geodataframe_vazio()

        gdf = _normalizar_colunas(gdf)
        gdf = _padronizar_nomes_colunas(gdf, caminho)
        gdf = _limpar_geometrias(gdf)
        gdf = _calcular_areas(gdf, caminho)
        gdf = _adicionar_colunas_padrao(gdf)
        
        return gdf.to_crs(CRS_GEOGRAFICO)

    except Exception as erro:
        st.error(f"Erro ao carregar {caminho}: {erro}")
        return _criar_geodataframe_vazio()

def filtrar_dados_por_municipios_vale_ribeira(dados_queimadas):
    """Filtra dados de queimadas pelos municípios do Vale do Ribeira."""
    if not eh_dataframe_valido(dados_queimadas, ['Municipio']):
        return pd.DataFrame()
    
    dados_copia = dados_queimadas.copy()
    dados_copia['municipio_normalizado'] = (
        dados_copia['Municipio'].str.upper().str.strip()
    )
    
    return dados_copia[
        dados_copia['municipio_normalizado'].isin(MUNICIPIOS_VALE_RIBEIRA)
    ].copy()

# ======================= FUNÇÕES DE GRÁFICOS =======================

def _calcular_area_alertas_uc(uc_geom, gdf_alertas):
    """Calcula área de alertas que intersectam com UC."""
    if gdf_alertas.empty:
        return 0
    
    # Identificar coluna de área
    area_col = None
    if 'AREAHA' in gdf_alertas.columns:
        area_col = 'AREAHA'
    elif 'areaha' in gdf_alertas.columns:
        area_col = 'areaha'
    else:
        return 0
    
    # Filtrar alertas com área > 0
    gdf_alertas_validos = gdf_alertas.copy()
    gdf_alertas_validos[area_col] = pd.to_numeric(gdf_alertas_validos[area_col], errors='coerce').fillna(0)
    gdf_alertas_validos = gdf_alertas_validos[gdf_alertas_validos[area_col] > 0]
    
    if gdf_alertas_validos.empty:
        return 0
    
    try:
        # Garantir CRS compatível
        alertas_proj = gdf_alertas_validos.to_crs(CRS_PROJECAO)
        
        # Usar spatial join para encontrar interseções
        alertas_intersect = gpd.sjoin(alertas_proj, gpd.GeoDataFrame([0], geometry=[uc_geom], crs=CRS_PROJECAO), predicate='intersects')
        
        if not alertas_intersect.empty:
            return alertas_intersect[area_col].sum()
            
    except Exception:
        # Fallback: usar bounds para aproximação
        try:
            alertas_proj = gdf_alertas_validos.to_crs(CRS_PROJECAO)
            uc_bounds = uc_geom.bounds
            alertas_na_regiao = alertas_proj.cx[uc_bounds[0]:uc_bounds[2], uc_bounds[1]:uc_bounds[3]]
            if not alertas_na_regiao.empty:
                return alertas_na_regiao[area_col].sum()
        except Exception:
            pass
    
    return 0

def _calcular_area_sigef_uc(uc_geom, gdf_sigef):
    """Calcula área de CARs que intersecta com UC."""
    if gdf_sigef.empty:
        return 0
    try:
        sigef_proj = gdf_sigef.to_crs(CRS_PROJECAO)
        sigef_intersect = sigef_proj[sigef_proj.geometry.intersects(uc_geom)]
        if not sigef_intersect.empty:
            return _processar_intersecoes_sigef(uc_geom, sigef_intersect)
    except Exception:
        pass
    return 0

def _processar_intersecoes_sigef(uc_geom, sigef_intersect):
    """Processa interseções CARs com otimização para muitos polígonos."""
    area_total = 0
    if len(sigef_intersect) > LIMITE_OTIMIZACAO_POLIGONOS:
        try:
            from shapely.ops import unary_union
            sigef_combined = unary_union(sigef_intersect.geometry.tolist())
            if uc_geom.is_valid and sigef_combined.is_valid:
                intersecao = uc_geom.intersection(sigef_combined)
                if not intersecao.is_empty:
                    area_total = intersecao.area / CONVERSAO_M2_PARA_HA
        except Exception:
            area_total = _calcular_intersecoes_loop(uc_geom, sigef_intersect)
    else:
        area_total = _calcular_intersecoes_loop(uc_geom, sigef_intersect)
    return area_total

def _calcular_intersecoes_loop(uc_geom, sigef_intersect):
    """Calcula interseções usando loop tradicional."""
    area_total = 0
    for _, sigef_pol in sigef_intersect.iterrows():
        try:
            if uc_geom.is_valid and sigef_pol.geometry.is_valid:
                intersecao = uc_geom.intersection(sigef_pol.geometry)
                if not intersecao.is_empty:
                    area_total += intersecao.area / CONVERSAO_M2_PARA_HA
        except Exception:
            pass
    return area_total

def _simplificar_nome_uc(nome_uc):
    """Simplifica nomes longos de UCs."""
    nome = str(nome_uc)
    replacements = {
        'Reserva Particular do Patrimônio Natural': 'RPPN',
        'Área de Proteção Ambiental': 'APA',
        'Parque Nacional': 'PN',
        'Parque Estadual': 'PE',
        'Reserva Biológica': 'REBIO',
        'Estação Ecológica': 'ESEC'
    }
    for original, abrev in replacements.items():
        nome = nome.replace(original, abrev)
    return nome

def fig_grafico_sobreposicoes(gdf_cnuc, gdf_alertas, gdf_sigef):
    if gdf_cnuc.empty: 
        return go.Figure()
    
    dados_uc = []
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs(CRS_PROJECAO)
        
        for _, uc in gdf_cnuc.iterrows():
            nome_uc = uc['nome_uc']
            uc_match = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == nome_uc]
            if uc_match.empty:
                continue
                
            uc_geom = uc_match.geometry.iloc[0]
            area_alertas = _calcular_area_alertas_uc(uc_geom, gdf_alertas)
            area_sigef = _calcular_area_sigef_uc(uc_geom, gdf_sigef)
            
            if area_alertas > 0 or area_sigef > 0:
                nome_simplificado = _simplificar_nome_uc(nome_uc)
                dados_uc.append({
                    'UC': quebrar_rotulo(nome_simplificado, 8),
                    'UC_original': nome_uc,
                    'Alertas': round(area_alertas, 2),
                    'CARs': round(area_sigef, 2),
                    'Total': round(area_alertas + area_sigef, 2)
                })
    except Exception:
        pass
    
    if not dados_uc:
        return go.Figure().update_layout(title='UCs com Sobreposições (Alertas e SIGEF)', height=450)
    
    df = pd.DataFrame(dados_uc).sort_values('Total', ascending=False)
    df_long = pd.melt(df, id_vars=['UC', 'UC_original'], value_vars=['Alertas', 'CARs'], var_name='Tipo', value_name='Área (ha)')
    
    fig = px.bar(df_long, x='UC', y='Área (ha)', color='Tipo', barmode='stack', hover_data={'UC_original': True})
    fig.update_traces(texttemplate='%{y:.1f}', textposition='inside', textfont_size=10,
                     hovertemplate='<b>%{customdata[0]}</b><br>%{fullData.name}: %{y:.1f} ha<extra></extra>')
    fig.update_layout(xaxis_tickangle=0, xaxis_tickfont_size=8, height=450, yaxis_title='Área (ha)',
                     yaxis_type='log', xaxis=dict(tickmode='linear', dtick=1), autosize=True)
    return _aplicar_layout(fig, titulo='Sobreposições de Alertas e CARs em UCs', tamanho_titulo=16, paleta="sobreposicoes")

def fig_ucs_por_municipio(gdf_cnuc: gpd.GeoDataFrame) -> go.Figure:
    if gdf_cnuc.empty or 'municipio' not in gdf_cnuc.columns:
        return go.Figure()
    
    municipio_stats = gdf_cnuc.groupby('municipio').agg({
        'nome_uc': 'count',
        'area_km2': 'sum'
    }).reset_index()
    municipio_stats.columns = ['Município', 'Quantidade_UCs', 'Área_Total_km2']
    municipio_stats['Área_Total_ha'] = municipio_stats['Área_Total_km2'] * CONVERSAO_KM2_PARA_HA
    municipio_stats = municipio_stats.sort_values('Quantidade_UCs', ascending=False)
    
    if municipio_stats.empty:
        return go.Figure()
    
    municipio_stats['Município_wrap'] = municipio_stats['Município'].apply(lambda x: quebrar_rotulo(str(x), 12))
    
    fig = px.bar(municipio_stats, 
                 x='Município_wrap', 
                 y='Quantidade_UCs',
                 text='Quantidade_UCs',
                 hover_data={'Área_Total_ha': ':,.0f'})
    
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside',
        hovertemplate='<b>%{customdata[1]}</b><br>UCs: %{y}<br>Área Total: %{customdata[0]:,.0f} ha<extra></extra>',
        customdata=municipio_stats[['Área_Total_ha', 'Município']].values
    )
    
    fig.update_layout(
        xaxis_tickangle=0,
        xaxis_title='Município',
        yaxis_title='Quantidade de UCs',
        height=400,
        autosize=True,
        yaxis=dict(range=[0, municipio_stats['Quantidade_UCs'].max() * 1.1]),
        margin=dict(l=60, r=40, t=40, b=80)
    )
    
    return _aplicar_layout(fig, titulo='Quantidade de Unidades de Conservação por Município', tamanho_titulo=16, paleta="sobreposicoes")

def _calcular_area_total_ucs(gdf_cnuc, nome_uc):
    """Calcula área total das UCs usando a mesma lógica da tabela."""
    if nome_uc == "Todas":
        area_total = 0
        for _, uc in gdf_cnuc.iterrows():
            if 'ha_total' in uc and pd.notna(uc['ha_total']) and pd.to_numeric(uc['ha_total'], errors='coerce') > 0:
                area_total += pd.to_numeric(uc['ha_total'], errors='coerce')
            elif 'area_km2' in uc and pd.notna(uc['area_km2']) and pd.to_numeric(uc['area_km2'], errors='coerce') > 0:
                area_total += pd.to_numeric(uc['area_km2'], errors='coerce') * 100
            else:
                area_total += (uc.geometry.to_crs(CRS_PROJECAO).area / 10000)
        return area_total
    else:
        uc_row = gdf_cnuc[gdf_cnuc['nome_uc'] == nome_uc]
        if uc_row.empty:
            return 0
        uc = uc_row.iloc[0]
        if 'ha_total' in uc and pd.notna(uc['ha_total']) and pd.to_numeric(uc['ha_total'], errors='coerce') > 0:
            return pd.to_numeric(uc['ha_total'], errors='coerce')
        elif 'area_km2' in uc and pd.notna(uc['area_km2']) and pd.to_numeric(uc['area_km2'], errors='coerce') > 0:
            return pd.to_numeric(uc['area_km2'], errors='coerce') * 100
        else:
            return (uc.geometry.to_crs(CRS_PROJECAO).area / 10000)

def _calcular_area_sigef_total(gdf_cnuc_proj, gdf_sigef_proj, nome_uc):
    """Calcula área total de CARs usando a mesma lógica da tabela."""
    area_sigef_total = 0
    if gdf_sigef_proj.empty:
        return area_sigef_total
        
    try:
        if nome_uc == "Todas":
            for _, uc in gdf_cnuc_proj.iterrows():
                uc_geom_proj = uc.geometry
                sigef_intersect = gdf_sigef_proj[gdf_sigef_proj.geometry.intersects(uc_geom_proj)]
                for _, sigef_pol in sigef_intersect.iterrows():
                    try:
                        intersecao = uc_geom_proj.intersection(sigef_pol.geometry)
                        if not intersecao.is_empty:
                            area_sigef_total += intersecao.area / 10000
                    except Exception:
                        pass
        else:
            uc_match = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == nome_uc]
            if not uc_match.empty:
                uc_geom_proj = uc_match.geometry.iloc[0]
                sigef_intersect = gdf_sigef_proj[gdf_sigef_proj.geometry.intersects(uc_geom_proj)]
                for _, sigef_pol in sigef_intersect.iterrows():
                    try:
                        intersecao = uc_geom_proj.intersection(sigef_pol.geometry)
                        if not intersecao.is_empty:
                            area_sigef_total += intersecao.area / 10000
                    except Exception:
                        pass
    except Exception:
        pass
    return area_sigef_total

def _processar_sigef_por_uc(uc_geom, gdf_sigef_proj):
    """Processa CARs para uma UC específica."""
    area_total = 0
    try:
        sigef_intersect = gdf_sigef_proj[gdf_sigef_proj.geometry.intersects(uc_geom)]
        for _, sigef_pol in sigef_intersect.iterrows():
            try:
                intersecao = uc_geom.intersection(sigef_pol.geometry)
                if not intersecao.is_empty:
                    area_total += intersecao.area / CONVERSAO_M2_PARA_HA
            except Exception:
                pass
    except Exception:
        pass
    return area_total

def _criar_grafico_donut(area_sigef_total, area_total, modo_valor, nome_uc):
    """Cria o gráfico de rosca."""
    # Não limitar área de CARs - mostrar valor real
    restante = max(0, area_total - area_sigef_total)
    percentual = (area_sigef_total / area_total) * 100 if area_total > 0 else 0
    
    if modo_valor == "percent":
        center_text = f"{percentual:.1f}%"
        # Se CARs > 100%, mostrar apenas CARs
        if percentual > 100:
            values = [100]
            labels = ["CARs (>100%)"]
            colors = ["#98FB98"]
            hover_template = "<b>%{label}</b><br>Percentual Real: " + f"{percentual:.1f}%" + "<br>Área: " + f"{area_sigef_total:,.0f} ha" + "<extra></extra>"
            customdata = None
        else:
            values = [percentual, 100 - percentual]
            labels = ["CARs", "Área livre da UC"]
            colors = ["#98FB98", "#F0F8FF"]
            hover_template = "<b>%{label}</b><br>Percentual: %{value:.1f}%<br>Área: %{customdata:,.0f} ha<extra></extra>"
            customdata = [area_sigef_total, restante]
    else:
        center_text = f"{area_sigef_total:,.0f} ha"
        values = [area_sigef_total, restante] if restante > 0 else [area_sigef_total]
        labels = ["CARs", "Área livre da UC"] if restante > 0 else ["CARs"]
        colors = ["#98FB98", "#F0F8FF"] if restante > 0 else ["#98FB98"]
        hover_template = "<b>%{label}</b><br>Área: %{value:,.0f} ha<br>Percentual: %{percent}<extra></extra>"
        customdata = None
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.6, 
        marker=dict(colors=colors), 
        textinfo="none",
        hovertemplate=hover_template,
        customdata=customdata
    )])
    
    # Manter cor do texto central padrão
    text_color = "#333"
    subtitle = f"UC: {area_total:,.0f} ha" if percentual > 100 else f"Total: {area_total:,.0f} ha"
    
    fig.update_layout(
        annotations=[
            dict(text=center_text, x=0.5, y=0.52, font_size=20, showarrow=False, font_color=text_color, font_weight="bold"),
            dict(text=subtitle, x=0.5, y=0.48, font_size=12, showarrow=False, font_color="#666")
        ],
        height=400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return _aplicar_layout(fig, titulo=f"Proporção de CARs na UC: {nome_uc}", tamanho_titulo=16, paleta="sobreposicoes")

def fig_car_por_uc_donut(gdf_cnuc: gpd.GeoDataFrame, gdf_sigef: gpd.GeoDataFrame, nome_uc: str, modo_valor: str) -> go.Figure:
    if gdf_cnuc.empty: 
        return go.Figure()
    
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs(CRS_PROJECAO) if not gdf_cnuc.empty else gpd.GeoDataFrame()
        gdf_sigef_proj = gdf_sigef.to_crs(CRS_PROJECAO) if not gdf_sigef.empty else gpd.GeoDataFrame()
        
        area_total = _calcular_area_total_ucs(gdf_cnuc, nome_uc)
        if area_total <= 0:
            return go.Figure()
            
        area_sigef_total = _calcular_area_sigef_total(gdf_cnuc_proj, gdf_sigef_proj, nome_uc)
        area_sigef_total = max(0, area_sigef_total)
        
        return _criar_grafico_donut(area_sigef_total, area_total, modo_valor, nome_uc)
            
    except Exception as e:
        st.warning(f"Erro ao criar gráfico de rosca: {e}")
        return go.Figure()

def mostrar_tabela_unificada(gdf_alertas, gdf_cnuc, gdf_sigef):
    if gdf_cnuc.empty:
        st.info("Não há dados de UCs disponíveis.")
        return
    
    dados_tabela = []
    
    for idx, uc in gdf_cnuc.iterrows():
        nome_uc = uc['nome_uc']
        
        # Área da UC
        if 'ha_total' in uc and pd.notna(uc['ha_total']) and pd.to_numeric(uc['ha_total'], errors='coerce') > 0:
            area_uc = pd.to_numeric(uc['ha_total'], errors='coerce')
        elif 'area_km2' in uc and pd.notna(uc['area_km2']) and pd.to_numeric(uc['area_km2'], errors='coerce') > 0:
            area_uc = pd.to_numeric(uc['area_km2'], errors='coerce') * 100
        else:
            area_uc = (uc.geometry.to_crs(CRS_PROJECAO).area / 10000)
        
        # Calcular área de interseção dos alertas com a UC
        uc_geom_proj = gpd.GeoSeries([uc.geometry], crs=gdf_cnuc.crs).to_crs(CRS_PROJECAO).iloc[0]
        area_alertas = _calcular_area_alertas_uc(uc_geom_proj, gdf_alertas)
        
        # Calcular área total de interseção dos CARs com a UC (sem limitação)
        area_cars = 0
        if not gdf_sigef.empty:
            try:
                gdf_sigef_proj = gdf_sigef.to_crs(CRS_PROJECAO)
                sigef_intersect = gdf_sigef_proj[gdf_sigef_proj.geometry.intersects(uc_geom_proj)]
                
                for _, sigef_pol in sigef_intersect.iterrows():
                    try:
                        intersecao = uc_geom_proj.intersection(sigef_pol.geometry)
                        if not intersecao.is_empty:
                            area_cars += intersecao.area / 10000
                    except Exception:
                        pass
            except Exception:
                pass
        
        dados_tabela.append({
            'UC': nome_uc,
            'Área UC (ha)': round(area_uc, 1),
            'CARs (ha)': round(area_cars, 1)
        })
    
    df = pd.DataFrame(dados_tabela)
    
    total_row = pd.DataFrame({
        'UC': ['TOTAL'],
        'Área UC (ha)': [df['Área UC (ha)'].sum()],
        'CARs (ha)': [df['CARs (ha)'].sum()]
    })
    df = pd.concat([df, total_row], ignore_index=True)
    
    st.dataframe(df, use_container_width=True)

def fig_desmatamento_uc(gdf_cnuc, gdf_alertas) -> go.Figure:
    if gdf_alertas.empty: 
        return go.Figure()
    
    try:
        gdf_alertas = gdf_alertas.copy()
        gdf_alertas['areaha'] = pd.to_numeric(gdf_alertas['areaha'], errors='coerce').fillna(0)

        gdf_alertas = gdf_alertas[gdf_alertas['areaha'] > 0]
        
        if gdf_alertas.empty:
            return go.Figure()
        if 'municipio' in gdf_alertas.columns:
            alert_area = gdf_alertas.groupby('municipio')['areaha'].sum().reset_index()
            alert_area.columns = ['Local', 'area_total']
        else:
            if 'anodetec' in gdf_alertas.columns:
                alert_area = gdf_alertas.groupby('anodetec')['areaha'].sum().reset_index()
                alert_area.columns = ['Local', 'area_total']
                alert_area['Local'] = alert_area['Local'].astype(str, errors='ignore')
            else:
                alert_area = pd.DataFrame({
                    'Local': ['Total da Região'],
                    'area_total': [gdf_alertas['areaha'].sum()]
                })
        
        alert_area = alert_area.sort_values('area_total', ascending=False)
        
        if alert_area.empty or alert_area['area_total'].sum() == 0:
            return go.Figure()
        
        alert_area['local_wrap'] = alert_area['Local'].apply(lambda x: quebrar_rotulo(str(x), 8))
        
        fig = px.bar(alert_area, x='local_wrap', y='area_total', text='area_total')
        fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
        fig.update_layout(
            xaxis_title="Localização",
            yaxis_title="Área de Alertas (ha)",
            height=400,
            autosize=True,
            yaxis=dict(range=[0, alert_area['area_total'].max() * 1.1]),
            margin=dict(l=60, r=40, t=40, b=80)
        )
        
        return _aplicar_layout(fig, titulo="Área Total de Alertas de Desmatamento por Localização", tamanho_titulo=16, paleta="desmatamento")
        
    except Exception as e:
        st.warning(f"Erro ao criar gráfico de desmatamento: {e}")
        return go.Figure()

def _adicionar_alertas_mapa(fig, gdf_alertas, gdf_cnuc_proj):
    """Adiciona camada de alertas ao mapa."""
    if gdf_alertas.empty:
        return
    try:
        gdf_alertas_proj = gdf_alertas.to_crs(CRS_PROJECAO)
        alertas_que_tocam = gpd.sjoin(gdf_alertas_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
        if not alertas_que_tocam.empty:
            alertas_que_tocam = alertas_que_tocam.to_crs(CRS_GEOGRAFICO)
            area_col = 'areaha_left' if 'areaha_left' in alertas_que_tocam.columns else 'areaha'
            if area_col in alertas_que_tocam.columns:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=alertas_que_tocam.__geo_interface__, locations=alertas_que_tocam.index,
                    z=pd.to_numeric(alertas_que_tocam[area_col], errors='coerce').fillna(0),
                    colorscale="Reds", showscale=False, marker_opacity=0.6, marker_line_width=1,
                    name="Alertas", hovertemplate="<b>Alerta:</b> %{z:.2f} ha<br><b>UC:</b> %{customdata}<extra></extra>",
                    customdata=alertas_que_tocam['nome_uc'].fillna('N/A')
                ))
    except Exception:
        pass

def _calcular_areas_intersecao_sigef(sigef_que_toca, gdf_sigef_proj, gdf_cnuc_proj):
    """Calcula áreas de interseção para CARs."""
    areas_intersecao = []
    for idx, row in sigef_que_toca.iterrows():
        try:
            sigef_geom = gdf_sigef_proj.loc[idx].geometry
            uc_match_temp = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == row['nome_uc']]
            if len(uc_match_temp) > 0:
                uc_geom = uc_match_temp.geometry.iloc[0]
                if sigef_geom.is_valid and uc_geom.is_valid:
                    intersecao = sigef_geom.intersection(uc_geom)
                    area_ha = intersecao.area / CONVERSAO_M2_PARA_HA if not intersecao.is_empty else 0
                    areas_intersecao.append(area_ha)
                else:
                    areas_intersecao.append(0)
            else:
                areas_intersecao.append(0)
        except Exception:
            areas_intersecao.append(0)
    return areas_intersecao

def _adicionar_sigef_mapa(fig, gdf_sigef, gdf_cnuc_proj):
    """Adiciona camada de CARs ao mapa."""
    if gdf_sigef.empty:
        return
    try:
        gdf_sigef_proj = gdf_sigef.to_crs(CRS_PROJECAO)
        sigef_que_toca = gpd.sjoin(gdf_sigef_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
        if not sigef_que_toca.empty:
            sigef_que_toca = sigef_que_toca.to_crs(CRS_GEOGRAFICO)
            areas_intersecao = _calcular_areas_intersecao_sigef(sigef_que_toca, gdf_sigef_proj, gdf_cnuc_proj)
            fig.add_trace(go.Choroplethmapbox(
                geojson=sigef_que_toca.__geo_interface__, locations=sigef_que_toca.index,
                z=areas_intersecao, colorscale="Oranges",
                showscale=False, marker_opacity=0.6, marker_line_width=1,
                name="CARs", hovertemplate="<b>CARs</b><br><b>UC:</b> %{customdata}<br><b>Área:</b> %{z:.2f} ha<extra></extra>",
                customdata=sigef_que_toca['nome_uc'].fillna('N/A')
            ))
    except Exception:
        pass

def _calcular_centro_zoom(gdf_cnuc, gdf_quilombolas, area_selecionada, centro):
    """Calcula centro e zoom do mapa baseado na área selecionada."""
    zoom_level = 8
    map_center = centro
    
    if area_selecionada and area_selecionada != "Todas":
        try:
            if area_selecionada.startswith("UC: "):
                nome_uc = area_selecionada[4:]
                area_filtrada = gdf_cnuc[gdf_cnuc['nome_uc'] == nome_uc]
                if not area_filtrada.empty:
                    area_bounds = area_filtrada.total_bounds
                    map_center = {"lat": (area_bounds[1] + area_bounds[3]) / 2, "lon": (area_bounds[0] + area_bounds[2]) / 2}
                    zoom_level = 12
        except Exception:
            pass
    return map_center, zoom_level

def fig_mapa_sobreposicoes(gdf_cnuc, gdf_alertas, gdf_sigef, gdf_quilombolas, centro, area_selecionada=None) -> go.Figure:
    fig = go.Figure()
    if gdf_cnuc.empty:
        return fig
    
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs(CRS_PROJECAO)
        _adicionar_alertas_mapa(fig, gdf_alertas, gdf_cnuc_proj)
        _adicionar_sigef_mapa(fig, gdf_sigef, gdf_cnuc_proj)
    except Exception as e:
        st.warning(f"Erro ao processar dados: {e}")

    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf_cnuc.__geo_interface__, locations=gdf_cnuc.index,
        z=np.ones(len(gdf_cnuc)), colorscale=[[0, "blue"], [1, "blue"]],
        showscale=False, marker_opacity=0.2, marker_line_width=2,
        name="UCs", hovertemplate="<b>UC:</b> %{customdata}<extra></extra>",
        customdata=gdf_cnuc['nome_uc'].fillna('N/A')
    ))
    

    
    map_center, zoom_level = _calcular_centro_zoom(gdf_cnuc, gdf_quilombolas, area_selecionada, centro)
    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=zoom_level, center=map_center),
        margin=dict(l=0, r=0, t=30, b=0), height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        annotations=[
            dict(x=0.02, y=0.85, xref="paper", yref="paper", text="<b>Legenda:</b>", showarrow=False, font=dict(size=14, color="black"), bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
            dict(x=0.02, y=0.80, xref="paper", yref="paper", text="🔵 Unidades de Conservação", showarrow=False, font=dict(size=12, color="black"), bgcolor="rgba(255,255,255,0.8)"),
            dict(x=0.02, y=0.76, xref="paper", yref="paper", text="🔴 Alertas de Desmatamento", showarrow=False, font=dict(size=12, color="black"), bgcolor="rgba(255,255,255,0.8)"),
            dict(x=0.02, y=0.72, xref="paper", yref="paper", text="🟠 CARs", showarrow=False, font=dict(size=12, color="black"), bgcolor="rgba(255,255,255,0.8)")
        ]
    )
    return fig

def fig_desmatamento_temporal(gdf_alertas) -> go.Figure:
    if gdf_alertas.empty:
        return go.Figure()
    
    try:
        df = gdf_alertas.copy()
        
        ano_col = None
        if 'ANODETEC' in df.columns:
            ano_col = 'ANODETEC'
        elif 'anodetec' in df.columns:
            ano_col = 'anodetec'
        else:
            return go.Figure()
        
        area_col = None
        if 'AREAHA' in df.columns:
            area_col = 'AREAHA'
        elif 'areaha' in df.columns:
            area_col = 'areaha'
        else:
            return go.Figure()
        
        df[ano_col] = pd.to_numeric(df[ano_col], errors='coerce')
        df[area_col] = pd.to_numeric(df[area_col], errors='coerce').fillna(0)
        df = df.dropna(subset=[ano_col])
        
        if df.empty:
            return go.Figure()
        
        temporal = df.groupby(ano_col)[area_col].agg(['sum', 'count']).reset_index()
        temporal.columns = ['Ano', 'Área (ha)', 'Quantidade']
        temporal = temporal.sort_values('Ano')
        
        fig = px.line(temporal, x='Ano', y='Área (ha)', markers=True,
                     hover_data={'Quantidade': True}, text='Área (ha)')
        fig.update_traces(texttemplate='%{y:.1f}', textposition='top center', mode='lines+markers+text')
        fig.update_layout(height=400, yaxis=dict(range=[0, temporal['Área (ha)'].max() * 1.1]))
        
        return _aplicar_layout(fig, titulo="Histórico Anual dos Alertas de Desmatamento", tamanho_titulo=16, paleta="desmatamento")
        
    except Exception as e:
        st.warning(f"Erro: {e}")
        return go.Figure()

def fig_pressoes_desmatamento(gdf_alertas) -> go.Figure:
    if gdf_alertas.empty:
        return go.Figure()
    
    try:
        df = gdf_alertas.copy()
        pressao_col = None
        if 'VPRESSAO' in df.columns:
            pressao_col = 'VPRESSAO'
        elif 'vpressao' in df.columns:
            pressao_col = 'vpressao'
        else:
            return go.Figure()
        
        area_col = None
        if 'AREAHA' in df.columns:
            area_col = 'AREAHA'
        elif 'areaha' in df.columns:
            area_col = 'areaha'
        else:
            return go.Figure()
        
        df[area_col] = pd.to_numeric(df[area_col], errors='coerce').fillna(0)
        df = df[df[area_col] > 0]
        
        if df.empty:
            return go.Figure()
        
        pressoes = df.groupby(pressao_col)[area_col].agg(['sum', 'count']).reset_index()
        pressoes.columns = ['Tipo de Pressão', 'Área Total (ha)', 'Quantidade']
        pressoes = pressoes.sort_values('Área Total (ha)', ascending=False)
        
        traducoes = {
            'agriculture': 'Agricultura',
            'mining': 'Mineração',
            'urban': 'Urbanização',
            'urban_expansion': 'Expansão Urbana',
            'pasture': 'Pastagem',
            'other': 'Outros'
        }
        pressoes['Tipo de Pressão'] = pressoes['Tipo de Pressão'].map(traducoes).fillna(pressoes['Tipo de Pressão'])
        
        fig = px.bar(pressoes, x='Tipo de Pressão', y='Área Total (ha)', 
                    text='Área Total (ha)', hover_data={'Quantidade': True})
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            height=400,
            yaxis=dict(range=[0, pressoes['Área Total (ha)'].max() * 1.1])
        )
        
        return _aplicar_layout(fig, titulo="Área de Desmatamento por Tipo de Pressão", tamanho_titulo=16, paleta="desmatamento")
        
    except Exception as e:
        st.warning(f"Erro ao criar gráfico de pressões: {e}")
        return go.Figure()

def _criar_figura_vazia():
    """Cria figura vazia para quando não há dados."""
    fig = go.Figure()
    fig.add_annotation(
        text="Dados não disponíveis",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="gray")
    )
    fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def _criar_grafico_temporal_risco(df):
    """Cria gráfico de evolução temporal do risco de fogo."""
    if 'DataHora' not in df.columns or 'RiscoFogo' not in df.columns:
        return go.Figure().update_layout(title="Evolução Temporal do Risco de Fogo - Sem dados")
    
    df_temp = df.dropna(subset=['DataHora', 'RiscoFogo'])
    df_temp['RiscoFogo'] = pd.to_numeric(df_temp['RiscoFogo'], errors='coerce')
    df_temp = df_temp[df_temp['RiscoFogo'].between(0, 1)]
    
    if df_temp.empty:
        return go.Figure().update_layout(title="Evolução Temporal do Risco de Fogo - Sem dados")
    
    df_temp = df_temp.set_index('DataHora')
    monthly_risco = df_temp['RiscoFogo'].resample('ME').mean().reset_index()
    monthly_risco['DataHora_str'] = monthly_risco['DataHora'].dt.to_period('M').astype(str, errors='ignore')
    
    fig = px.line(monthly_risco, x='DataHora_str', y='RiscoFogo', markers=True,
                  labels={'DataHora_str': 'Mês/Ano', 'RiscoFogo': 'Risco Médio de Fogo'})
    fig.update_traces(line_color='#FFD1DC', marker_color='#FFD1DC', mode='lines+markers+text',
                     text=monthly_risco['RiscoFogo'].round(3), textposition='top center')
    fig.update_layout(height=400, autosize=True)
    return _aplicar_layout(fig, "Variação Mensal do Risco de Fogo", 16, "queimadas")

def _criar_grafico_top_risco(df):
    """Cria gráfico de top municípios por risco de fogo."""
    if 'Municipio' not in df.columns or 'RiscoFogo' not in df.columns:
        return go.Figure().update_layout(title="Municípios por Risco de Fogo - Sem dados")
    
    df_risco = df.copy()
    df_risco['RiscoFogo'] = pd.to_numeric(df_risco['RiscoFogo'], errors='coerce')
    df_risco = df_risco[df_risco['RiscoFogo'].between(0, 1)]
    
    if df_risco.empty:
        return go.Figure().update_layout(title="Municípios por Risco de Fogo - Sem dados")
    
    top_risco = df_risco.groupby('Municipio')['RiscoFogo'].mean().nlargest(TOP_MUNICIPIOS_LIMITE).sort_values()
    
    fig = go.Figure(go.Bar(y=top_risco.index, x=top_risco.values, orientation='h',
                          marker_color='#F0E68C', text=top_risco.values.round(3), textposition='outside'))
    fig.update_layout(height=300, xaxis_title='Risco Médio de Fogo', yaxis_title='Município', autosize=True, xaxis=dict(range=[0, top_risco.max() * 1.1]))
    return _aplicar_layout(fig, "Ranking de Municípios por Risco de Fogo", 16, "queimadas")

def _criar_grafico_top_precipitacao(df):
    """Cria gráfico de top municípios por precipitação."""
    if 'Municipio' not in df.columns or 'Precipitacao' not in df.columns:
        return go.Figure().update_layout(title="Municípios por Precipitação - Sem dados")
    
    df_precip = df.copy()
    df_precip['Precipitacao'] = pd.to_numeric(df_precip['Precipitacao'], errors='coerce')
    df_precip = df_precip[df_precip['Precipitacao'] >= 0]
    
    if df_precip.empty:
        return go.Figure().update_layout(title="Municípios por Precipitação - Sem dados")
    
    top_precip = df_precip.groupby('Municipio')['Precipitacao'].mean().nlargest(TOP_MUNICIPIOS_LIMITE).sort_values()
    
    fig = go.Figure(go.Bar(y=top_precip.index, x=top_precip.values, orientation='h',
                          marker_color='#B5E7A0', text=[f'{x:.1f} mm' for x in top_precip.values], textposition='outside'))
    fig.update_layout(height=300, xaxis_title='Precipitação Média (mm)', yaxis_title='Município', autosize=True, xaxis=dict(range=[0, top_precip.max() * 1.1]))
    return _aplicar_layout(fig, "Ranking de Municípios por Precipitação", 16, "queimadas")

def _criar_mapa_focos_calor(df):
    """Cria mapa de distribuição dos focos de calor."""
    map_cols = ['Latitude', 'Longitude', 'RiscoFogo', 'Municipio']
    if not all(col in df.columns for col in map_cols):
        return go.Figure().update_layout(title="Mapa de Focos de Calor - Sem dados")
    
    df_map = df[map_cols + (['Precipitacao'] if 'Precipitacao' in df.columns else [])].copy()
    df_map = df_map.dropna(subset=['Latitude', 'Longitude', 'RiscoFogo', 'Municipio'])
    df_map['RiscoFogo'] = pd.to_numeric(df_map['RiscoFogo'], errors='coerce')
    df_map = df_map[df_map['RiscoFogo'].between(0, 1)]
    
    if 'Precipitacao' in df_map.columns:
        df_map['Precipitacao'] = pd.to_numeric(df_map['Precipitacao'], errors='coerce').fillna(5)
        df_map['Precipitacao'] = df_map['Precipitacao'].apply(lambda x: max(x, 5))
    else:
        df_map['Precipitacao'] = 10
    
    if df_map.empty:
        return go.Figure().update_layout(title="Mapa de Focos de Calor - Sem dados")
    
    if len(df_map) > MAX_PONTOS_MAPA:
        df_map = df_map.sample(MAX_PONTOS_MAPA, random_state=42)
    
    fig = px.scatter_mapbox(
        df_map, lat='Latitude', lon='Longitude', color='RiscoFogo',
        hover_name='Municipio', hover_data={'Latitude': False, 'Longitude': False, 'RiscoFogo': ':.3f'},
        color_continuous_scale=[[0, '#FFFF00'], [0.5, '#FF8C00'], [1, '#FF0000']],
        mapbox_style="open-street-map", zoom=8,
        center={'lat': df_map['Latitude'].mean(), 'lon': df_map['Longitude'].mean()}, height=500
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        coloraxis_showscale=True, 
        mapbox_style="open-street-map",
        title={"text": "Distribuição dos Focos de Calor", "x": 0.5, "xanchor": "center", "font_size": 16}
    )
    return fig

def criar_graficos_queimadas(df_queimadas):
    if df_queimadas.empty:
        fig_vazio = _criar_figura_vazia()
        return {'temporal': fig_vazio, 'top_risco': fig_vazio, 'top_precip': fig_vazio, 'mapa': fig_vazio}
    
    df = df_queimadas.copy()
    if 'DataHora' in df.columns:
        df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')
    
    return {
        'temporal': _criar_grafico_temporal_risco(df),
        'top_risco': _criar_grafico_top_risco(df),
        'top_precip': _criar_grafico_top_precipitacao(df),
        'mapa': _criar_mapa_focos_calor(df)
    }

def criar_ranking_queimadas(df_queimadas, indicador):
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

def fig_ranking_assuntos(df):
    if df.empty or 'assuntos' not in df.columns:
        return go.Figure()
    
    assuntos_series = df['assuntos'].str.split(', ').explode()
    top_assuntos = assuntos_series.value_counts().nlargest(10).reset_index()
    top_assuntos.columns = ['assunto', 'quantidade']

    top_assuntos['assunto_wrap'] = top_assuntos['assunto'].apply(lambda x: '<br>'.join(textwrap.wrap(x, width=30)))

    fig = px.bar(
        top_assuntos,
        x='quantidade',
        y='assunto_wrap',
        orientation='h',
        labels={'quantidade': 'Nº de Processos', 'assunto_wrap': 'Assunto'},
        text='quantidade',
        color_discrete_sequence=['#E6E6FA']
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        autosize=True,
        height=400,
        margin=dict(l=60, r=40, t=40, b=80)
    )
    
    fig.update_traces(textposition='outside')
    
    return _aplicar_layout(fig, titulo="Principais Assuntos dos Processos Ambientais", tamanho_titulo=14, paleta="justica")

def criar_graficos_processos(df_processos):
    if df_processos.empty:
        return {}
    
    graficos = {}
    
    fig_municipios = fig_distribuicao_processos_municipio(df_processos)
    graficos['municipios'] = fig_municipios
    
    fig_assuntos = fig_ranking_assuntos(df_processos)
    graficos['assuntos'] = fig_assuntos
    
    fig_temporal = fig_evolucao_temporal_processos(df_processos)
    graficos['temporal'] = fig_temporal
    
    if 'classe' in df_processos.columns:
        classes = df_processos['classe'].value_counts().nlargest(10)
        classes_wrapped = [quebrar_rotulo(str(classe), 8) for classe in classes.index]
        fig_classes = go.Figure(data=[
            go.Bar(
                x=classes_wrapped,
                y=classes.values,
                marker_color='#DDA0DD',
                text=classes.values,
                textposition='outside'
            )
        ])
        fig_classes.update_layout(
            yaxis_title='Quantidade de Processos',
            xaxis_title='Classe',
            height=400,
            showlegend=False,
            autosize=True,
            yaxis=dict(range=[0, classes.max() * 1.1]),
            margin=dict(l=60, r=40, t=40, b=80)
        )
        graficos['classes'] = _aplicar_layout(fig_classes, titulo="Principais Classes de Processos Ambientais", tamanho_titulo=14, paleta="justica")

    return graficos

def fig_evolucao_temporal_processos(df):
    if df.empty or 'data_ajuizamento' not in df.columns:
        return go.Figure()

    df_temporal = df.set_index('data_ajuizamento').resample('Y').size().reset_index(name='quantidade')
    df_temporal['ano'] = df_temporal['data_ajuizamento'].dt.year

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_temporal['ano'], 
        y=df_temporal['quantidade'],
        mode='lines+markers+text',
        text=df_temporal['quantidade'],
        textposition="top center",
        line=dict(color='#98FB98', width=2),
        marker=dict(color='#98FB98')
    ))
    fig.update_layout(
        xaxis_title='Ano de Ajuizamento',
        yaxis_title='Nº de Processos',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        autosize=True,
        height=400
    )
    return _aplicar_layout(fig, titulo="Histórico de Ajuizamento de Processos Ambientais", tamanho_titulo=14, paleta="justica")

def fig_distribuicao_processos_municipio(df):
    if df.empty or 'municipio' not in df.columns:
        return go.Figure()

    dist_municipio = df['municipio'].value_counts().reset_index()
    dist_municipio.columns = ['municipio', 'quantidade']
    
    fig = px.bar(
        dist_municipio,
        x='quantidade',
        y='municipio',
        orientation='h',
        labels={'quantidade': 'Nº de Processos', 'municipio': 'Município'},
        text='quantidade',
        color_discrete_sequence=['#87CEEB']
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        autosize=True,
        height=400,
        margin=dict(l=60, r=40, t=40, b=80)
    )
    fig.update_traces(textposition='outside')
    return _aplicar_layout(fig, titulo="Concentração de Processos Ambientais por Município", tamanho_titulo=14, paleta="justica")

def fig_orgaos_julgadores(df):
    if df.empty or 'orgao_julgador' not in df.columns:
        return go.Figure()
    df_temp = df.copy()
    df_temp['orgao_simplificado'] = df_temp['orgao_julgador'].str.replace(
        r'VARA CÍVEL, DA FAZENDA PÚBLICA, ACIDENTES DO TRABALHO, REGISTROS PÚBLICOS E CORREGEDORIA DO FORO EXTRAJUDICIAL',
        'VARA CÍVEL E FAZENDA PÚBLICA', regex=True
    ).str.replace(
        r'JUÍZO ÚNICO', 'JUÍZO ÚNICO', regex=True
    )
    
    orgaos = df_temp['orgao_simplificado'].value_counts().reset_index()
    orgaos.columns = ['Órgão Julgador', 'Quantidade']
    orgaos['Órgão_wrap'] = orgaos['Órgão Julgador'].apply(lambda x: quebrar_rotulo(str(x), 25))
    
    fig = px.bar(
        orgaos,
        x='Quantidade',
        y='Órgão_wrap',
        orientation='h',
        text='Quantidade',
        color_discrete_sequence=['#DEB887']
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        height=400,
        autosize=True,
        yaxis_title='Órgão Julgador',
        xaxis_title='Número de Processos',
        margin=dict(l=150, r=40, t=40, b=80)
    )
    
    fig.update_traces(textposition='outside')
    
    return _aplicar_layout(fig, titulo="Distribuição por Órgão Julgador", tamanho_titulo=14, paleta="justica")

# ======================= CARREGAMENTO DOS DADOS =======================
global gdf_cnuc_raw, gdf_sigef_raw, gdf_alertas_raw, df_queimadas_raw, df_processos, centro

try:
    gdf_cnuc_raw = carregar_shapefile("cnuc.shp")
except Exception as e:
    st.error(f"Erro ao carregar cnuc.shp: {e}")
    gdf_cnuc_raw = gpd.GeoDataFrame()

try:
    gdf_sigef_raw = carregar_shapefile("SIGEF.shp")
except Exception as e:
    st.error(f"Erro ao carregar SIGEF.shp: {e}")
    gdf_sigef_raw = gpd.GeoDataFrame()

try:
    gdf_alertas_raw = carregar_shapefile("Alertas.shp")
except Exception as e:
    st.error(f"Erro ao carregar Alertas.shp: {e}")
    gdf_alertas_raw = gpd.GeoDataFrame()

try:
    df_queimadas_raw = pd.read_csv("Risco_Fogo.csv")
except Exception as e:
    st.error(f"Erro ao carregar Risco_Fogo.csv: {e}")
    df_queimadas_raw = pd.DataFrame()

try:
    df_processos = carregar_dados_processos("processos_ambientais_vale_do_ribeira_pr.csv")
except Exception as e:
    st.error(f"Erro ao carregar processos: {e}")
    df_processos = pd.DataFrame()

gdf_quilombolas = gpd.GeoDataFrame()

if not gdf_cnuc_raw.empty:
    limites = gdf_cnuc_raw.total_bounds
    centro = {"lat": (limites[1] + limites[3]) / 2, "lon": (limites[0] + limites[2]) / 2}
else:
    centro = {"lat": -24.85, "lon": -49.15}


# Logo centralizado
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    try:
        st.image("logo_cezar.jpg", width=300)
    except:
        st.write("Logo não encontrado")

st.title("Dashboard Vale do Ribeira - Paraná")

tabs = st.tabs(["Sobreposições", "Desmatamento", "Queimadas", "Justiça"])

with tabs[0]:
    st.header("Sobreposições")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("Análise de sobreposições de Cadastros Ambientais Rurais (CARs) e alertas de desmatamento em Unidades de Conservação (UCs).")

    total_ucs = len(gdf_cnuc_raw) if not gdf_cnuc_raw.empty else 0
    if not gdf_cnuc_raw.empty and 'ha_total' in gdf_cnuc_raw.columns:
        area_total_ucs = pd.to_numeric(gdf_cnuc_raw['ha_total'], errors='coerce').fillna(0).sum()
    elif not gdf_cnuc_raw.empty:
        area_total_ucs = (gdf_cnuc_raw.geometry.to_crs('EPSG:31983').area / 10000).sum()
    else:
        area_total_ucs = 0
    
    total_alertas = len(gdf_alertas_raw) if not gdf_alertas_raw.empty else 0
    if not gdf_alertas_raw.empty and 'AREAHA' in gdf_alertas_raw.columns:
        area_alertas = pd.to_numeric(gdf_alertas_raw['AREAHA'], errors='coerce').fillna(0).sum()
    elif not gdf_alertas_raw.empty and 'areaha' in gdf_alertas_raw.columns:
        area_alertas = pd.to_numeric(gdf_alertas_raw['areaha'], errors='coerce').fillna(0).sum()
    else:
        area_alertas = 0
    
    total_sigef = len(gdf_sigef_raw) if not gdf_sigef_raw.empty else 0
    
    # Calcular CARs que toca UCs
    sigef_que_toca_ucs = 0
    if not gdf_sigef_raw.empty and not gdf_cnuc_raw.empty:
        try:
            gdf_sigef_proj = gdf_sigef_raw.to_crs(CRS_PROJECAO)
            gdf_cnuc_proj = gdf_cnuc_raw.to_crs(CRS_PROJECAO)
            sigef_intersect = gpd.sjoin(gdf_sigef_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
            sigef_que_toca_ucs = len(sigef_intersect)
        except Exception:
            sigef_que_toca_ucs = 10008
    else:
        sigef_que_toca_ucs = 10008  
    
    cols = st.columns(5)
    cols[0].metric("Total de UCs", f"{total_ucs}")
    cols[1].metric("Área Total UCs", f"{area_total_ucs:,.0f} ha")
    cols[2].metric("Total de Alertas", f"{total_alertas:,}")
    cols[3].metric("Área Alertas", f"{area_alertas:,.0f} ha")
    cols[4].metric("Total CARs\n", "75")
    st.divider()

    row1_map, row1_chart1 = st.columns([3, 2], gap="large")
    with row1_map:
        st.subheader("Mapa de Sobreposições")
        opcoes_mapa = ["Todas"]
        if not gdf_cnuc_raw.empty:
            opcoes_mapa.extend([f"UC: {nome}" for nome in sorted(gdf_cnuc_raw["nome_uc"].unique())])
        
        area_selecionada_mapa = st.selectbox("Selecione a área para focar:", opcoes_mapa, key="area_mapa_filtro")
        
        fig_mapa = fig_mapa_sobreposicoes(gdf_cnuc_raw, gdf_alertas_raw, gdf_sigef_raw, gpd.GeoDataFrame(), centro, area_selecionada_mapa)
        
        st.plotly_chart(fig_mapa, use_container_width=True, config={"scrollZoom": True})
        
        st.subheader("Proporção da Área dos CARs sobre a UC")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico de rosca mostra a proporção de área ocupada por cadastros rurais (CARs) dentro de uma UC específica. 
            Permite analisar:
            - **Ocupação**: Percentual da UC sobreposta por propriedades rurais
            - **Pressão**: Intensidade da pressão antrópica sobre a área protegida
            - **Conflito**: Identificação de possíveis conflitos fundiários
            """)
        uc_names = ["Todas"] + sorted(gdf_cnuc_raw["nome_uc"].unique()) if not gdf_cnuc_raw.empty and len(gdf_cnuc_raw) > 0 else ["Todas"]
        nome_uc_donut = st.selectbox("Selecione a UC:", uc_names, key="donut_uc")
        modo_donut = st.radio("Mostrar valores como:", ["Hectares (ha)", "% da UC"], horizontal=True, key="donut_mode")
        

        
        fig_donut = fig_car_por_uc_donut(gdf_cnuc_raw, gdf_sigef_raw, nome_uc_donut, "absoluto" if modo_donut == "Hectares (ha)" else "percent")
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_chart1:
        st.subheader("Sobreposições de Alertas e CARs por UC")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico mostra as áreas de sobreposição entre alertas de desmatamento e cadastros ambientais rurais (CAR/SIGEF) 
            dentro das Unidades de Conservação. As barras empilhadas permitem visualizar:
            - **Alertas**: Área total de alertas de desmatamento detectados dentro de cada UC
            - **CARs**: Área total de cadastros rurais sobrepostos à UC
            - **Comparação**: Identificação das UCs com maiores pressões ambientais
            """)
        fig_sobreposicoes = fig_grafico_sobreposicoes(gdf_cnuc_raw, gdf_alertas_raw, gdf_sigef_raw)
        if not fig_sobreposicoes.data:
            st.info("Nenhuma sobreposição encontrada entre UCs e alertas/SIGEF.")
        else:
            st.plotly_chart(fig_sobreposicoes, use_container_width=True)
        
        st.subheader("Quantidade de UCs por Município")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico apresenta a distribuição das Unidades de Conservação pelos municípios do Vale do Ribeira. 
            Permite identificar:
            - **Concentração**: Municípios com maior número de UCs
            - **Cobertura**: Distribuição espacial das áreas protegidas na região
            - **Gestão**: Complexidade administrativa por município
            """)
        fig_municipios = fig_ucs_por_municipio(gdf_cnuc_raw)
        if not fig_municipios.data:
            st.info("Não há dados de municípios para exibir.")
        else:
            st.plotly_chart(fig_municipios, use_container_width=True)
    
    st.subheader("Tabela Unificada por UC")
    mostrar_tabela_unificada(gdf_alertas_raw, gdf_cnuc_raw, gdf_sigef_raw)
    
    with st.expander("ℹ️ Sobre esta tabela", expanded=False):
        st.write("""
        Esta tabela apresenta um resumo consolidado das informações por Unidade de Conservação (UC). 
        
        **Colunas da tabela:**
        - **UC**: Nome da Unidade de Conservação
        - **Área UC (ha)**: Área total da UC em hectares
        - **CARs (ha)**: Área total de Cadastros Ambientais Rurais que se sobrepõem à UC
        
        **Interpretação:**
        - Valores altos de CARs indicam maior pressão antrópica sobre a UC
        - A comparação entre área da UC e área de CARs mostra a intensidade de sobreposição
        - A linha TOTAL apresenta os somatórios de todas as UCs
        
        **Observações técnicas:**
        - Valores calculados com base na interseção geométrica real entre os polígonos
        - Mostra a área real de sobreposição sem limitações artificiais
        """)
    st.divider()

with tabs[1]:
    st.header("Desmatamento")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("Análise de alertas de desmatamento, com dados do MapBiomas Alerta.")

    if 'anodetec' in gdf_alertas_raw.columns and not gdf_alertas_raw.empty:
        anos_validos = pd.to_numeric(gdf_alertas_raw['anodetec'], errors='coerce').dropna().astype('Int64', errors='ignore').unique()
        anos_disponiveis = ['Todos'] + sorted(anos_validos.tolist())
    else:
        anos_disponiveis = ['Todos']
    ano_selecionado = st.selectbox('Filtrar por Ano de Detecção:', anos_disponiveis, key="filtro_ano_desmat")
    
    gdf_alertas_filtrado = gdf_alertas_raw[gdf_alertas_raw['anodetec'] == ano_selecionado] if ano_selecionado != 'Todos' else gdf_alertas_raw
    st.divider()

    col_charts, col_map = st.columns([2, 3], gap="large")
    with col_charts:
        st.subheader("Área Total de Alertas por Localização")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico apresenta a distribuição dos alertas de desmatamento por localização (município ou ano). 
            Permite identificar:
            - **Hotspots**: Localizações com maior concentração de desmatamento
            - **Magnitude**: Área total desmatada em cada localização
            - **Priorização**: Áreas que necessitam de maior atenção para fiscalização
            """)
        fig_desmat_uc = fig_desmatamento_uc(gdf_cnuc_raw, gdf_alertas_filtrado)
        if not fig_desmat_uc.data:
            st.info("Nenhum alerta de desmatamento sobre UCs para o período selecionado.")
        else:
            st.plotly_chart(fig_desmat_uc, use_container_width=True)
        
        st.subheader("Desmatamento por Tipo de Pressão")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico mostra a distribuição dos alertas de desmatamento por tipo de pressão antrópica. 
            Permite analisar:
            - **Causas**: Principais atividades responsáveis pelo desmatamento
            - **Impacto**: Área desmatada por cada tipo de atividade
            - **Estratégias**: Orientação para políticas de prevenção específicas
            """)
        fig_pressoes = fig_pressoes_desmatamento(gdf_alertas_filtrado)
        if not fig_pressoes.data:
            st.info("Dados de pressão não disponíveis para o período selecionado.")
        else:
            st.plotly_chart(fig_pressoes, use_container_width=True)

    with col_map:
        st.subheader("Geometrias dos Alertas Filtrados")

        if not gdf_alertas_raw.empty:
            try:
                area_col = 'AREAHA' if 'AREAHA' in gdf_alertas_raw.columns else 'areaha'
                fig_alertas_geom = go.Figure(go.Choroplethmapbox(
                    geojson=gdf_alertas_raw.__geo_interface__, locations=gdf_alertas_raw.index,
                    z=pd.to_numeric(gdf_alertas_raw[area_col], errors='coerce').fillna(0),
                    colorscale="Reds", showscale=True, marker_opacity=0.7, marker_line_width=1,
                    hovertemplate="<b>Área:</b> %{z:.2f} ha<extra></extra>"
                ))
                fig_alertas_geom.update_layout(
                    mapbox=dict(style="open-street-map", zoom=6, center={"lat": -24.5, "lon": -51.5}),
                    margin=dict(l=0, r=0, t=0, b=0), height=500
                )
                st.plotly_chart(fig_alertas_geom, use_container_width=True, config={'scrollZoom': True})
            except Exception as e:
                st.error(f"Erro ao criar mapa de alertas: {e}")
        else:
            st.info("Não há alertas disponíveis.")

    st.divider()
    st.subheader("Histórico Anual dos Alertas de Desmatamento")
    with st.expander("ℹ️ Sobre este gráfico", expanded=False):
        st.write("""
        Este gráfico de linha mostra a evolução temporal dos alertas de desmatamento ao longo dos anos. 
        Permite identificar:
        - **Tendências**: Padrões de aumento ou diminuição do desmatamento
        - **Sazonalidade**: Variações anuais na detecção de alertas
        - **Eficácia**: Impacto de políticas de conservação ao longo do tempo
        """)
    
    fig_temporal = fig_desmatamento_temporal(gdf_alertas_raw)
    if not fig_temporal.data:
        st.info("Não há dados temporais para exibir.")
    else:
        st.plotly_chart(fig_temporal, use_container_width=True)

    st.subheader("Dados dos Alertas")
    if not gdf_alertas_filtrado.empty:
        df_alertas_display = gdf_alertas_filtrado.copy()
        if 'geometry' in df_alertas_display.columns:
            df_alertas_display = df_alertas_display.drop(columns=['geometry'])
        if 'areaha' in df_alertas_display.columns:
            df_alertas_display['areaha'] = pd.to_numeric(df_alertas_display['areaha'], errors='coerce').fillna(0).round(2)
        
        st.dataframe(
            df_alertas_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        csv_data = df_alertas_display.to_csv(index=False)
        st.download_button(
            label="📥 Baixar dados como CSV",
            data=csv_data,
            file_name=f"alertas_desmatamento_{ano_selecionado if ano_selecionado != 'Todos' else 'todos_anos'}.csv",
            mime="text/csv"
        )
    else:
        st.info("Não há dados de alertas disponíveis.")

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
    df_queimadas_vale = filtrar_dados_por_municipios_vale_ribeira(df_queimadas_raw)
    
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

    if not df_queimadas_filtrado.empty:
        graficos_queimadas = criar_graficos_queimadas(df_queimadas_filtrado)

        def renderizar_grafico_com_validacao(titulo, chave_grafico, legenda, mensagem_erro):
            """Renderiza gráfico se disponível, senão mostra mensagem."""
            st.subheader(titulo)
            if chave_grafico in graficos_queimadas:
                st.plotly_chart(graficos_queimadas[chave_grafico], use_container_width=True)
                if legenda:
                    st.caption(legenda)
            else:
                st.info(mensagem_erro)
        
        st.subheader("Variação Mensal do Risco de Fogo")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico de linha mostra a variação mensal do risco médio de fogo na região. 
            Permite identificar:
            - **Sazonalidade**: Períodos do ano com maior risco de incêndios
            - **Tendências**: Padrões de variação ao longo dos meses
            - **Prevenção**: Orientação para planejamento de ações preventivas
            """)
        if 'temporal' in graficos_queimadas:
            st.plotly_chart(graficos_queimadas['temporal'], use_container_width=True)
            st.caption(f"Figura 3.1: Evolução mensal do risco médio de fogo para {display_periodo}.")
        else:
            st.info("Dados insuficientes para gerar o gráfico temporal.")

        col_graficos1, col_graficos2 = st.columns(2, gap="large")

        with col_graficos1:
            st.subheader("Ranking de Municípios por Risco de Fogo")
            with st.expander("ℹ️ Sobre este gráfico", expanded=False):
                st.write("""
                Este gráfico de barras horizontais classifica os municípios pelo risco médio de fogo. 
                Permite identificar:
                - **Áreas críticas**: Municípios com maior suscetibilidade a incêndios
                - **Priorização**: Localizações que necessitam de maior atenção
                - **Recursos**: Orientação para alocação de recursos de prevenção
                """)
            if 'top_risco' in graficos_queimadas:
                st.plotly_chart(graficos_queimadas['top_risco'], use_container_width=True)
            else:
                st.info("Dados insuficientes para gerar o gráfico de municípios por risco de fogo.")
            
            st.subheader("Municípios com Maior Precipitação Média")
            with st.expander("ℹ️ Sobre este gráfico", expanded=False):
                st.write("""
                Este gráfico mostra os municípios com maiores índices de precipitação média. 
                Permite analisar:
                - **Umidade**: Municípios com maior disponibilidade hídrica
                - **Proteção natural**: Áreas com menor risco de incêndios por fatores climáticos
                - **Contraste**: Comparação com áreas de maior risco de fogo
                """)
            if 'top_precip' in graficos_queimadas:
                st.plotly_chart(graficos_queimadas['top_precip'], use_container_width=True)
            else:
                st.info("Dados insuficientes para gerar o gráfico de municípios por precipitação.")

        with col_graficos2:
            st.subheader("Distribuição Espacial do Risco de Fogo")
            with st.expander("ℹ️ Sobre este mapa", expanded=False):
                st.write("""
                Este mapa interativo mostra a distribuição espacial dos focos de calor e risco de fogo na região. 
                Permite visualizar:
                - **Localização**: Posição geográfica dos focos de calor detectados
                - **Intensidade**: Variação do risco de fogo por cores (amarelo a vermelho)
                - **Concentração**: Áreas com maior densidade de focos de calor
                """)
            if 'mapa' in graficos_queimadas:
                st.plotly_chart(
                    graficos_queimadas['mapa'], 
                    use_container_width=True, 
                    config={'scrollZoom': True}
                )
            else:
                st.info("Dados insuficientes para gerar o mapa de focos de calor.")

        st.divider()

        st.header("Ranking de Municípios por Indicadores de Queimadas")
        st.caption("Classifica municípios pelo maior registro de cada indicador.")
        
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

with tabs[3]:
    st.header("Processos Judiciais")
    
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre processos judiciais ambientais no Vale do Ribeira (PR):
        - Principais motivações/assuntos
        - Evolução temporal dos ajuizamentos

        **Municípios analisados:** Adrianópolis, Bocaiúva do Sul, Cerro Azul, Doutor Ulysses, Itaperuçu, Rio Branco do Sul, Tunas do Paraná

        Os dados são provenientes do arquivo processos_ambientais_vale_do_ribeira_pr.csv.
        """)
        st.markdown(
            "**Fonte Geral da Seção:** CONSELHO NACIONAL DE JUSTIÇA (CNJ).",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Principais Assuntos dos Processos")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico de barras horizontais mostra os assuntos mais frequentes nos processos judiciais ambientais. 
            Permite identificar:
            - **Temáticas**: Principais questões ambientais em disputa judicial
            - **Recorrência**: Assuntos que mais geram litigação na região
            - **Políticas**: Orientação para políticas públicas preventivas
            """)
        if not df_processos.empty:
            fig2 = fig_ranking_assuntos(df_processos)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Figura 4.1: Top 10 assuntos mais recorrentes nos processos judiciais.")
        else:
            st.warning("Dados não disponíveis")

    with col2:
        st.subheader("Principais Classes Processuais")
        with st.expander("ℹ️ Sobre este gráfico", expanded=False):
            st.write("""
            Este gráfico apresenta as classes processuais mais utilizadas em ações ambientais. 
            Permite analisar:
            - **Tipos de ação**: Instrumentos jurídicos mais utilizados
            - **Estratégias**: Preferências por determinados ritos processuais
            - **Efetividade**: Classes que mais tramitam no sistema judiciário
            """)
        if not df_processos.empty:
            graficos = criar_graficos_processos(df_processos)
            if 'classes' in graficos:
                st.plotly_chart(graficos['classes'], use_container_width=True)
                st.caption("Figura 4.2: Top 10 classes processuais mais frequentes.")
            else:
                st.warning("Dados de classes processuais não disponíveis")

    st.divider()
    
    st.subheader("Órgãos Julgadores")
    with st.expander("ℹ️ Sobre este gráfico", expanded=False):
        st.write("""
        Este gráfico mostra a distribuição dos processos ambientais entre os diferentes órgãos do Poder Judiciário. 
        Permite identificar:
        - **Concentração**: Varas e juízos com maior volume de processos ambientais
        - **Especialização**: Órgãos com maior experiência em questões ambientais
        - **Carga de trabalho**: Distribuição da demanda judicial na região
        """)
    if not df_processos.empty:
        fig_orgaos = fig_orgaos_julgadores(df_processos)
        if fig_orgaos.data:
            st.plotly_chart(fig_orgaos, use_container_width=True)
            st.caption("Figura 4.3: Distribuição de processos por órgão julgador.")
        else:
            st.warning("Dados de órgãos julgadores não disponíveis")
    else:
        st.warning("Dados não disponíveis")

    st.divider()

    st.subheader("Histórico de Ajuizamento dos Processos") 
    with st.expander("ℹ️ Sobre este gráfico", expanded=False):
        st.write("""
        Este gráfico de linha mostra a evolução temporal do ajuizamento de processos ambientais ao longo dos anos. 
        Permite analisar:
        - **Tendências**: Aumento ou diminuição da judicialização ambiental
        - **Marcos**: Identificação de períodos com maior atividade judicial
        - **Demanda**: Evolução da pressão sobre o sistema judiciário
        """)
    if not df_processos.empty:
        fig3 = fig_evolucao_temporal_processos(df_processos)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Figura 4.4: Evolução anual da quantidade de novos processos ajuizados.")
    else:
        st.warning("Dados não disponíveis")

    st.divider()

    st.subheader("Dados dos Processos Judiciais")
    if not df_processos.empty:
        df_processos_display = df_processos.copy()
        if 'data_ajuizamento' in df_processos_display.columns:
            df_processos_display['data_ajuizamento'] = df_processos_display['data_ajuizamento'].dt.strftime('%d/%m/%Y')
        
        st.dataframe(
            df_processos_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        csv_data = df_processos_display.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Baixar dados como CSV",
            data=csv_data,
            file_name="processos_judiciais_vale_ribeira.csv",
            mime="text/csv"
        )
    else:
        st.warning("Dados não disponíveis")

