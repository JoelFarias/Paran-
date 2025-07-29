import streamlit as st
import geopandas as gpd
import pandas as pd
from typing import List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import unicodedata
import os
import numpy as np
import warnings
import textwrap

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

CUSTOM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def apply_palette(fig: go.Figure, palette: str = "custom") -> go.Figure:
    seq = CUSTOM_COLORS
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'marker'):
            if hasattr(trace.marker, 'color'):
                if trace.marker.color is None:
                    trace.marker.color = seq[i % len(seq)]
        elif hasattr(trace, 'line'):
            if hasattr(trace.line, 'color') and trace.line.color is None:
                trace.line.color = seq[i % len(seq)]
    return fig

def _apply_layout(fig: go.Figure, title: str, title_size: int = 16) -> go.Figure:
    fig = apply_palette(fig)
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.5, "xanchor": "center", "font_size": title_size},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CCC", borderwidth=1, font=dict(size=10))
    )
    return fig

def preparar_hectares(gdf_cnuc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf_cnuc.empty:
        return gpd.GeoDataFrame()
    gdf_ha = gdf_cnuc.copy()
    cols_area = ['area_km2', 'alerta_km2', 'sigef_km2']
    for col in cols_area:
        if col in gdf_ha.columns:
            gdf_ha[f"{col.replace('_km2', '_ha')}"] = pd.to_numeric(gdf_ha[col], errors='coerce').fillna(0) * 100
    return gdf_ha

def wrap_label(name, width=30):
    if pd.isna(name): return ""
    return "<br>".join(textwrap.wrap(str(name), width))

def truncate_label(name, max_length=20):
    if pd.isna(name): return ""
    name_str = str(name)
    return name_str[:max_length] + "..." if len(name_str) > max_length else name_str

# ======================= FUNÇÕES AUXILIARES =======================

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=';')
        df['data_ajuizamento'] = pd.to_datetime(df['data_ajuizamento'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['data_ajuizamento'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {filepath}. Certifique-se de que o arquivo está na mesma pasta que o script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return pd.DataFrame()

def normalizar_string(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')
    return s.upper()

def verificar_e_reprojetar(gdf, target_crs="EPSG:31983"):
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def carregar_shapefile(caminho: str) -> gpd.GeoDataFrame:
    try:
        if not os.path.exists(caminho):
            st.warning(f"Arquivo não encontrado: {caminho}")
            return gpd.GeoDataFrame(columns=['geometry']).set_crs("EPSG:4326")

        gdf = gpd.read_file(caminho)
        if gdf.empty:
            st.warning(f"Shapefile vazio: {caminho}")
            return gpd.GeoDataFrame(columns=['geometry']).set_crs("EPSG:4326")

        gdf.columns = [str(col).lower() for col in gdf.columns]

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

        gdf["geometry"] = gdf["geometry"].make_valid()
        gdf = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty].copy()
        gdf['id'] = range(len(gdf))

        try:
            gdf_proj = gdf.to_crs("EPSG:31983")
            area_calc_km2 = gdf_proj.geometry.area / 1e6
            if "area_km2" not in gdf.columns:
                gdf["area_km2"] = area_calc_km2
            else:
                gdf["area_km2"] = pd.to_numeric(gdf["area_km2"], errors='coerce').fillna(area_calc_km2)
        except Exception as e:
            st.warning(f"Não foi possível calcular a área para {caminho}: {e}")
            if "area_km2" not in gdf.columns:
                gdf["area_km2"] = 0

        for col in ['alerta_km2', 'sigef_km2', 'c_alertas', 'c_sigef', 'ha_total', 'areaha']:
            if col not in gdf.columns:
                gdf[col] = 0
            else:
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
        
        if 'ha_total' in gdf.columns and gdf['ha_total'].sum() == 0 and 'area_km2' in gdf.columns:
            gdf['ha_total'] = gdf['area_km2'] * 100
        
        return gdf.to_crs("EPSG:4326")

    except Exception as e:
        st.error(f"Erro fatal ao carregar {caminho}: {e}")
        return gpd.GeoDataFrame(columns=['geometry']).set_crs("EPSG:4326")

def carregar_kmls_cars():
    municipios_cars = {}
    total_cars = 0
    municipios_vale = ["Adrianópolis", "Bocaiúva do Sul", "Cerro Azul", "Doutor Ulysses", "Itaperuçu", "Rio Branco do Sul", "Tunas do Paraná"]
    
    for municipio in municipios_vale:
        caminho_kml = f"{municipio}.kml"
        try:
            if os.path.exists(caminho_kml):
                gdf_kml = gpd.read_file(caminho_kml, driver='KML')
                if not gdf_kml.empty and 'geometry' in gdf_kml.columns:
                    gdf_proj = gdf_kml.to_crs("EPSG:31983")
                    area_total = gdf_proj.geometry.area.sum() / 10_000  # ha
                    municipios_cars[municipio] = area_total
                    total_cars += len(gdf_kml)
                else:
                    municipios_cars[municipio] = 0
            else:
                municipios_cars[municipio] = 0
        except Exception as e:
            st.warning(f"Erro ao carregar KML de {municipio}: {e}")
            municipios_cars[municipio] = 0
    return municipios_cars, total_cars

def filtrar_queimadas_vale_ribeira(df_queimadas):
    municipios_vale = [
        "ADRIANÓPOLIS", "BOCAIÚVA DO SUL", "CERRO AZUL", 
        "DOUTOR ULYSSES", "ITAPERUÇU", "RIO BRANCO DO SUL", "TUNAS DO PARANÁ"
    ]
    
    if df_queimadas.empty or 'Municipio' not in df_queimadas.columns:
        return pd.DataFrame()
    
    df_queimadas = df_queimadas.copy()
    df_queimadas['Municipio_norm'] = df_queimadas['Municipio'].str.upper().str.strip()
    
    return df_queimadas[df_queimadas['Municipio_norm'].isin(municipios_vale)].copy()

def criar_figura(gdf_cnuc_filtered, centro, ids_selecionados=None):
    try:
        if gdf_cnuc_filtered is None or gdf_cnuc_filtered.empty:
            return go.Figure()

        hover_cols = ['nome_uc', 'municipio', 'area_km2', 'alerta_km2', 'sigef_km2']
        custom_data = gdf_cnuc_filtered[hover_cols].fillna('N/A').values

        fig = go.Figure(go.Choroplethmapbox(
            geojson=gdf_cnuc_filtered.__geo_interface__,
            locations=gdf_cnuc_filtered.index,
            z=np.ones(len(gdf_cnuc_filtered)),
            colorscale=[[0, "#636EFA"], [1, "#636EFA"]], showscale=False,
            marker_opacity=0.5, marker_line_width=1,
            customdata=custom_data,
            hovertemplate="<b>%{customdata[0]}</b><br>Município: %{customdata[1]}<br>Área: %{customdata[2]:.2f} km²<br>Alertas: %{customdata[3]:.2f} km²<br>CAR: %{customdata[4]:.2f} km²<extra></extra>"
        ))

        if ids_selecionados:
             gdf_destaque = gdf_cnuc_filtered[gdf_cnuc_filtered['id'].isin(ids_selecionados)]
             if not gdf_destaque.empty:
                 custom_data_destaque = gdf_destaque[hover_cols].fillna('N/A').values
                 fig.add_trace(go.Choroplethmapbox(
                    geojson=gdf_destaque.__geo_interface__, locations=gdf_destaque.index,
                    z=np.ones(len(gdf_destaque)), colorscale=[[0, "#EF553B"], [1, "#EF553B"]],
                    showscale=False, marker_opacity=0.7, marker_line_width=2,
                    customdata=custom_data_destaque,
                    hovertemplate="<b>%{customdata[0]}</b> (SOBREPOSTA)<br>Município: %{customdata[1]}<br>Área: %{customdata[2]:.2f} km²<extra></extra>"
                 ))

        fig.update_layout(
            mapbox=dict(style="open-street-map", zoom=7, center=centro),
            showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=600,
        )
        return fig
    except Exception as e:
        st.error(f"Erro ao criar mapa: {e}")
        return go.Figure()

# ======================= FUNÇÕES DE GRÁFICOS =======================

def fig_grafico_sobreposicoes(gdf_cnuc, gdf_alertas, gdf_sigef):
    if gdf_cnuc.empty: return go.Figure()
    
    dados_uc = []
    
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs("EPSG:31983")
        
        for _, uc in gdf_cnuc.iterrows():
            nome_uc = uc['nome_uc']
            if 'area_ha' in uc and pd.notna(uc['area_ha']) and uc['area_ha'] > 0:
                area_uc = uc['area_ha']
            elif 'area_km2' in uc:
                area_uc = pd.to_numeric(uc['area_km2'], errors='coerce') * 100 if pd.notna(uc['area_km2']) else 0
            else:
                area_uc = 0
            
            area_alertas = 0
            if not gdf_alertas.empty and 'areaha' in gdf_alertas.columns:
                try:
                    uc_geom = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == nome_uc].geometry.iloc[0]
                    alertas_proj = gdf_alertas.to_crs("EPSG:31983")
                    alertas_intersect = alertas_proj[alertas_proj.geometry.intersects(uc_geom)]
                    if not alertas_intersect.empty:
                        area_alertas = pd.to_numeric(alertas_intersect['areaha'], errors='coerce').fillna(0).sum()
                except Exception:
                    pass
            
            area_sigef = 0
            if not gdf_sigef.empty:
                try:
                    uc_geom = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == nome_uc].geometry.iloc[0]
                    sigef_proj = gdf_sigef.to_crs("EPSG:31983")
                    sigef_intersect = sigef_proj[sigef_proj.geometry.intersects(uc_geom)]
                    if not sigef_intersect.empty:
                        area_sigef = sigef_intersect.geometry.area.sum() / 10000 
                except Exception:
                    pass
            
            if area_alertas > 0 or area_sigef > 0:
                # Simplificar nomes longos
                nome_simplificado = str(nome_uc)
                nome_simplificado = nome_simplificado.replace('Reserva Particular do Patrimônio Natural', 'RPPN')
                nome_simplificado = nome_simplificado.replace('Área de Proteção Ambiental', 'APA')
                nome_simplificado = nome_simplificado.replace('Parque Nacional', 'PN')
                nome_simplificado = nome_simplificado.replace('Parque Estadual', 'PE')
                nome_simplificado = nome_simplificado.replace('Reserva Biológica', 'REBIO')
                nome_simplificado = nome_simplificado.replace('Estação Ecológica', 'ESEC')
                
                dados_uc.append({
                    'UC': wrap_label(nome_simplificado, 8),
                    'UC_original': nome_uc,
                    'Alertas': round(area_alertas, 2),
                    'SIGEF': round(area_sigef, 2),
                    'Total': round(area_alertas + area_sigef, 2)
                })
    
    except Exception:
        pass
    
    if not dados_uc:
        fig = go.Figure()
        fig.update_layout(
            title='UCs com Sobreposições (Alertas e SIGEF)',
            height=450
        )
        return fig
    
    df = pd.DataFrame(dados_uc).sort_values('Total', ascending=False)
    df_long = pd.melt(df, id_vars=['UC', 'UC_original'], value_vars=['Alertas', 'SIGEF'], var_name='Tipo', value_name='Área (ha)')
    
    fig = px.bar(df_long, x='UC', y='Área (ha)', color='Tipo', barmode='stack',
                 hover_data={'UC_original': True})
    fig.update_traces(texttemplate='%{y:.1f}', textposition='inside', textfont_size=10,
                     hovertemplate='<b>%{customdata[0]}</b><br>%{fullData.name}: %{y:.1f} ha<extra></extra>')
    fig.update_layout(
        xaxis_tickangle=0, 
        xaxis_tickfont_size=8, 
        height=450,
        yaxis_title='Área (ha)',
        yaxis_type='log',
        xaxis=dict(tickmode='linear', dtick=1)
    )
    return _apply_layout(fig, title='UCs com Sobreposições (Alertas e SIGEF)', title_size=16)

def fig_ucs_por_municipio(gdf_cnuc: gpd.GeoDataFrame) -> go.Figure:
    if gdf_cnuc.empty or 'municipio' not in gdf_cnuc.columns:
        return go.Figure()
    
    municipio_stats = gdf_cnuc.groupby('municipio').agg({
        'nome_uc': 'count',
        'area_km2': 'sum'
    }).reset_index()
    municipio_stats.columns = ['Município', 'Quantidade_UCs', 'Área_Total_km2']
    municipio_stats['Área_Total_ha'] = municipio_stats['Área_Total_km2'] * 100
    municipio_stats = municipio_stats.sort_values('Quantidade_UCs', ascending=False)
    
    if municipio_stats.empty:
        return go.Figure()
    
    # Quebrar nomes longos em múltiplas linhas
    municipio_stats['Município_wrap'] = municipio_stats['Município'].apply(lambda x: wrap_label(str(x), 12))
    
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
        height=400
    )
    
    return _apply_layout(fig, title='Distribuição de UCs por Município', title_size=16)

def fig_car_por_uc_donut(gdf_cnuc: gpd.GeoDataFrame, gdf_sigef: gpd.GeoDataFrame, nome_uc: str, modo_valor: str) -> go.Figure:
    if gdf_cnuc.empty: 
        return go.Figure()
    
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs("EPSG:31983") if not gdf_cnuc.empty else gpd.GeoDataFrame()
        gdf_sigef_proj = gdf_sigef.to_crs("EPSG:31983") if not gdf_sigef.empty else gpd.GeoDataFrame()
        
        if nome_uc == "Todas":
            area_total = pd.to_numeric(gdf_cnuc['area_km2'], errors='coerce').fillna(0).sum() * 100
            area_sigef_total = 0
            
            if not gdf_sigef_proj.empty:
                try:
                    sigef_in_ucs = gpd.sjoin(gdf_sigef_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
                    if not sigef_in_ucs.empty:
                        area_sigef_total = sigef_in_ucs.geometry.area.sum() / 10000
                except Exception:
                    pass
        else:
            uc_row = gdf_cnuc[gdf_cnuc['nome_uc'] == nome_uc]
            if uc_row.empty: 
                return go.Figure()
            
            area_total = pd.to_numeric(uc_row['area_km2'].iloc[0], errors='coerce')
            area_total = (area_total * 100) if pd.notna(area_total) and area_total > 0 else 0
            area_sigef_total = 0
            
            if not gdf_sigef_proj.empty and area_total > 0:
                try:
                    uc_geom = gdf_cnuc_proj[gdf_cnuc_proj['nome_uc'] == nome_uc].geometry.iloc[0]
                    sigef_intersect = gdf_sigef_proj[gdf_sigef_proj.geometry.intersects(uc_geom)]
                    if not sigef_intersect.empty:
                        area_sigef_total = sigef_intersect.geometry.area.sum() / 10000
                except Exception:
                    pass
        
        if area_total <= 0:
            return go.Figure()
            
        area_sigef_total = max(0, area_sigef_total)
        restante = max(0, area_total - area_sigef_total)
        percentual = (area_sigef_total / area_total) * 100 if area_total > 0 else 0
        
        if modo_valor == "percent":
            center_text = f"{percentual:.1f}%"
            textinfo = "label+percent"
        else:
            center_text = f"{area_sigef_total:,.0f} ha"
            textinfo = "label+value"
        
        if area_total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["SIGEF/CAR", "Área livre da UC"], 
                values=[area_sigef_total, restante],
                hole=0.6, 
                marker_colors=["#2ca02c", "#e8f5e8"], 
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>Área: %{value:,.0f} ha<br>Percentual: %{percent}<extra></extra>"
            )])
            
            fig.update_layout(
                annotations=[
                    dict(
                        text=center_text, 
                        x=0.5, y=0.52, 
                        font_size=20, 
                        showarrow=False,
                        font_color="#333",
                        font_weight="bold"
                    ),
                    dict(
                        text=f"Total: {area_total:,.0f} ha", 
                        x=0.5, y=0.48, 
                        font_size=12, 
                        showarrow=False,
                        font_color="#666"
                    )
                ],
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            return _apply_layout(fig, title=f"SIGEF/CAR em: {nome_uc}", title_size=16)
        else:
            return go.Figure()
            
    except Exception as e:
        st.warning(f"Erro ao criar gráfico de rosca: {e}")
        return go.Figure()

def mostrar_tabela_unificada(gdf_alertas, gdf_cnuc, gdf_sigef):
    municipios_dados = set()
    
    if not gdf_cnuc.empty and 'municipio' in gdf_cnuc.columns:
        municipios_dados.update(gdf_cnuc['municipio'].dropna().unique())
    
    if not gdf_alertas.empty and 'municipio' in gdf_alertas.columns:
        municipios_dados.update(gdf_alertas['municipio'].dropna().unique())
    
    if not municipios_dados:
        st.info("Não há dados de municípios disponíveis.")
        return
    
    municipios_lista = sorted(list(municipios_dados))
    df = pd.DataFrame(index=municipios_lista)

    if not gdf_alertas.empty and 'municipio' in gdf_alertas.columns and 'areaha' in gdf_alertas.columns:
        alertas_data = gdf_alertas.groupby('municipio')['areaha'].sum()
        df['Alertas (ha)'] = df.index.map(alertas_data).fillna(0)
    else:
        df['Alertas (ha)'] = 0

    if not gdf_cnuc.empty and 'municipio' in gdf_cnuc.columns:
        if 'area_km2' in gdf_cnuc.columns:
            cnuc_data = gdf_cnuc.groupby('municipio')['area_km2'].sum() * 100
            df['UCs (ha)'] = df.index.map(cnuc_data).fillna(0)
        else:
            df['UCs (ha)'] = 0
    else:
        df['UCs (ha)'] = 0
    if not gdf_sigef.empty:
        try:
            gdf_sigef_proj = gdf_sigef.to_crs("EPSG:31983")
            area_total_sigef = gdf_sigef_proj.geometry.area.sum() / 10000
            df['SIGEF (ha)'] = area_total_sigef / len(df) if len(df) > 0 else 0
        except Exception:
            df['SIGEF (ha)'] = 0
    else:
        df['SIGEF (ha)'] = 0
    
    df.loc['TOTAL'] = df.sum()
    
    st.dataframe(df.round(1), use_container_width=True)

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
                alert_area['Local'] = alert_area['Local'].astype(str)
            else:
                alert_area = pd.DataFrame({
                    'Local': ['Total da Região'],
                    'area_total': [gdf_alertas['areaha'].sum()]
                })
        
        alert_area = alert_area.sort_values('area_total', ascending=False)
        
        if alert_area.empty or alert_area['area_total'].sum() == 0:
            return go.Figure()
        
        alert_area['local_wrap'] = alert_area['Local'].apply(lambda x: truncate_label(str(x), 15))
        
        fig = px.bar(alert_area, x='local_wrap', y='area_total', text='area_total')
        fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
        fig.update_layout(
            xaxis_title="Localização",
            yaxis_title="Área de Alertas (ha)",
            xaxis_tickangle=-45,
            height=400
        )
        
        return _apply_layout(fig, title="Área de Alertas por Localização", title_size=16)
        
    except Exception as e:
        st.warning(f"Erro ao criar gráfico de desmatamento: {e}")
        return go.Figure()

def fig_mapa_sobreposicoes(gdf_cnuc, gdf_alertas, gdf_sigef, centro, uc_selecionada=None) -> go.Figure:
    fig = go.Figure()
    
    if gdf_cnuc.empty:
        return fig
    
    try:
        gdf_cnuc_proj = gdf_cnuc.to_crs("EPSG:31983")
        
        if not gdf_alertas.empty:
            gdf_alertas_proj = gdf_alertas.to_crs("EPSG:31983")
            alertas_que_tocam = gpd.sjoin(gdf_alertas_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
            if not alertas_que_tocam.empty:
                alertas_que_tocam = alertas_que_tocam.to_crs("EPSG:4326")
                area_col = 'areaha_left' if 'areaha_left' in alertas_que_tocam.columns else 'areaha'
                if area_col in alertas_que_tocam.columns:
                    fig.add_trace(go.Choroplethmapbox(
                        geojson=alertas_que_tocam.__geo_interface__, locations=alertas_que_tocam.index,
                        z=pd.to_numeric(alertas_que_tocam[area_col], errors='coerce').fillna(0),
                        colorscale="Reds", showscale=False, marker_opacity=0.6, marker_line_width=1,
                        name="Alertas", hovertemplate="<b>Alerta:</b> %{z:.2f} ha<br><b>UC:</b> %{customdata}<extra></extra>",
                        customdata=alertas_que_tocam['nome_uc'].fillna('N/A')
                    ))
        
        # SIGEF que toca UCs
        if not gdf_sigef.empty:
            gdf_sigef_proj = gdf_sigef.to_crs("EPSG:31983")
            sigef_que_toca = gpd.sjoin(gdf_sigef_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
            if not sigef_que_toca.empty:
                sigef_que_toca = sigef_que_toca.to_crs("EPSG:4326")
                fig.add_trace(go.Choroplethmapbox(
                    geojson=sigef_que_toca.__geo_interface__, locations=sigef_que_toca.index,
                    z=np.ones(len(sigef_que_toca)), colorscale=[[0, "green"], [1, "green"]],
                    showscale=False, marker_opacity=0.2, marker_line_width=1,
                    name="SIGEF", hovertemplate="<b>SIGEF</b><br><b>UC:</b> %{customdata}<extra></extra>",
                    customdata=sigef_que_toca['nome_uc'].fillna('N/A')
                ))
    
    except Exception as e:
        st.warning(f"Erro ao processar dados: {e}")
    
    # UCs por cima (sempre mostrar)
    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf_cnuc.__geo_interface__, locations=gdf_cnuc.index,
        z=np.ones(len(gdf_cnuc)), colorscale=[[0, "blue"], [1, "blue"]],
        showscale=False, marker_opacity=0.2, marker_line_width=2,
        name="UCs", hovertemplate="<b>UC:</b> %{customdata}<extra></extra>",
        customdata=gdf_cnuc['nome_uc'].fillna('N/A')
    ))
    
    # Ajustar zoom e centro baseado na UC selecionada
    zoom_level = 8
    map_center = centro
    
    # Se uma UC específica foi selecionada, focar nela
    if uc_selecionada and uc_selecionada != "Todas":
        try:
            uc_filtrada = gdf_cnuc[gdf_cnuc['nome_uc'] == uc_selecionada]
            if not uc_filtrada.empty:
                uc_bounds = uc_filtrada.total_bounds
                map_center = {"lat": (uc_bounds[1] + uc_bounds[3]) / 2, "lon": (uc_bounds[0] + uc_bounds[2]) / 2}
                zoom_level = 12
        except Exception:
            pass
    
    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=zoom_level, center=map_center),
        margin=dict(l=0, r=0, t=30, b=0), height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig

def fig_desmatamento_mapa_pontos(gdf_alertas) -> go.Figure:
    if gdf_alertas.empty: return go.Figure()
    
    gdf_map = gdf_alertas.copy()
    gdf_map['areaha'] = pd.to_numeric(gdf_map['areaha'], errors='coerce').fillna(0)
    
    fig = go.Figure(go.Choroplethmapbox(
        geojson=gdf_map.__geo_interface__, locations=gdf_map.index,
        z=gdf_map['areaha'], colorscale="Reds", showscale=True, 
        marker_opacity=0.7, marker_line_width=1,
        hovertemplate="<b>Área:</b> %{z:.2f} ha<extra></extra>",
        colorbar=dict(title="Área (ha)")
    ))
    
    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=7),
        margin=dict(l=0, r=0, t=30, b=0), height=400
    )
    return fig

def fig_desmatamento_temporal(gdf_alertas) -> go.Figure:
    if gdf_alertas.empty or 'anodetec' not in gdf_alertas.columns: 
        return go.Figure()
    
    try:
        df = gdf_alertas.copy()
        df['anodetec'] = pd.to_numeric(df['anodetec'], errors='coerce')
        df['areaha'] = pd.to_numeric(df['areaha'], errors='coerce').fillna(0)
        
        # Filtrar anos válidos
        df = df.dropna(subset=['anodetec'])
        
        if df.empty:
            return go.Figure()
        
        # Agrupar por ano
        temporal = df.groupby('anodetec')['areaha'].agg(['sum', 'count']).reset_index()
        temporal.columns = ['Ano', 'Área (ha)', 'Quantidade']
        temporal = temporal.sort_values('Ano')
        
        fig = px.line(temporal, x='Ano', y='Área (ha)', markers=True,
                     hover_data={'Quantidade': True})
        fig.update_traces(texttemplate='%{y:.1f}', textposition='top center')
        fig.update_layout(height=400)
        
        return _apply_layout(fig, title="Evolução Temporal dos Alertas de Desmatamento", title_size=16)
        
    except Exception as e:
        st.warning(f"Erro: {e}")
        return go.Figure()

def criar_graficos_queimadas(df_queimadas):
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
        color='quantidade',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False  
    )
    
    fig.update_traces(textposition='outside')
    
    return _apply_layout(fig, title="Top 10 Assuntos mais Frequentes", title_size=14)

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
        fig_classes = go.Figure(data=[
            go.Bar(
                x=classes.index,
                y=classes.values,
                marker_color='#1f77b4'
            )
        ])
        fig_classes.update_layout(
            yaxis_title='Quantidade de Processos',
            xaxis_title='Classe',
            height=400,
            showlegend=False
        )
        graficos['classes'] = _apply_layout(fig_classes, title="Top 10 Classes Processuais", title_size=14)

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
        line=dict(color='#2ca02c', width=2)
    ))
    fig.update_layout(
        xaxis_title='Ano de Ajuizamento',
        yaxis_title='Nº de Processos',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return _apply_layout(fig, title="Evolução Anual de Novos Processos", title_size=14)

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
        color='quantidade',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(textposition='outside')
    return _apply_layout(fig, title="Distribuição de Processos por Município", title_size=14)

# ======================= CARREGAMENTO DOS DADOS =======================
gdf_cnuc_raw = carregar_shapefile("cnuc.shp")
gdf_sigef_raw = carregar_shapefile("SIGEF.shp")
gdf_alertas_raw = carregar_shapefile("Alertas.shp")
gdf_cnuc_ha_raw = preparar_hectares(gdf_cnuc_raw)
df_queimadas_raw = pd.read_csv("Risco_Fogo.csv")
df_processos = load_data("processos_ambientais_vale_do_ribeira_pr.csv")

if not gdf_cnuc_raw.empty:
    limites = gdf_cnuc_raw.total_bounds
    centro = {"lat": (limites[1] + limites[3]) / 2, "lon": (limites[0] + limites[2]) / 2}
else:
    centro = {"lat": -24.85, "lon": -49.15}


st.title("Dashboard Vale do Ribeira - Paraná")

tabs = st.tabs(["Sobreposições", "Desmatamento", "Queimadas", "Justiça"])

with tabs[0]:
    st.header("Sobreposições")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("Análise de sobreposições de Cadastros Ambientais Rurais (CAR) e alertas de desmatamento em Unidades de Conservação (UCs).")

    total_ucs = len(gdf_cnuc_raw) if not gdf_cnuc_raw.empty else 0
    area_total_ucs = gdf_cnuc_raw['area_km2'].sum() if not gdf_cnuc_raw.empty and 'area_km2' in gdf_cnuc_raw.columns else 0
    total_alertas = len(gdf_alertas_raw) if not gdf_alertas_raw.empty else 0
    area_alertas = pd.to_numeric(gdf_alertas_raw['areaha'], errors='coerce').fillna(0).sum() if not gdf_alertas_raw.empty and 'areaha' in gdf_alertas_raw.columns else 0
    total_sigef = len(gdf_sigef_raw) if not gdf_sigef_raw.empty else 0
    

    
    cols = st.columns(5)
    cols[0].metric("Total de UCs", f"{total_ucs}")
    cols[1].metric("Área Total UCs", f"{area_total_ucs:,.0f} km²")
    cols[2].metric("Total de Alertas", f"{total_alertas:,}")
    cols[3].metric("Área Alertas", f"{area_alertas:,.0f} ha")
    cols[4].metric("Total SIGEF", f"{total_sigef:,}")
    st.divider()

    row1_map, row1_chart1 = st.columns([3, 2], gap="large")
    with row1_map:
        st.subheader("Mapa de Sobreposições")
        uc_names_mapa = ["Todas"] + sorted(gdf_cnuc_raw["nome_uc"].unique()) if not gdf_cnuc_raw.empty else ["Todas"]
        uc_selecionada_mapa = st.selectbox("Selecione a UC para focar:", uc_names_mapa, key="uc_mapa_filtro")
        
        fig_mapa = fig_mapa_sobreposicoes(gdf_cnuc_raw, gdf_alertas_raw, gdf_sigef_raw, centro, uc_selecionada_mapa)
        
        st.plotly_chart(fig_mapa, use_container_width=True, config={"scrollZoom": True})
        
        st.subheader("Proporção da Área do CAR sobre a UC")
        uc_names = ["Todas"] + sorted(gdf_cnuc_ha_raw["nome_uc"].unique()) if not gdf_cnuc_ha_raw.empty else ["Todas"]
        nome_uc_donut = st.selectbox("Selecione a UC:", uc_names, key="donut_uc")
        modo_donut = st.radio("Mostrar valores como:", ["Hectares (ha)", "% da UC"], horizontal=True, key="donut_mode")
        fig_donut = fig_car_por_uc_donut(gdf_cnuc_raw, gdf_sigef_raw, nome_uc_donut, "absoluto" if modo_donut == "Hectares (ha)" else "percent")
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_chart1:
        st.subheader("Áreas de Sobreposição por UC")
        fig_sobreposicoes = fig_grafico_sobreposicoes(gdf_cnuc_raw, gdf_alertas_raw, gdf_sigef_raw)
        if not fig_sobreposicoes.data:
            st.info("Nenhuma sobreposição encontrada entre UCs e alertas/SIGEF.")
        else:
            st.plotly_chart(fig_sobreposicoes, use_container_width=True)
        
        st.subheader("Distribuição de UCs por Município")
        fig_municipios = fig_ucs_por_municipio(gdf_cnuc_raw)
        if not fig_municipios.data:
            st.info("Não há dados de municípios para exibir.")
        else:
            st.plotly_chart(fig_municipios, use_container_width=True)
    
    st.subheader("Áreas das Unidades de Conservação")
    if not gdf_cnuc_raw.empty:
        df_ucs = gdf_cnuc_raw[['nome_uc', 'area_km2']].copy()
        df_ucs['area_ha'] = pd.to_numeric(df_ucs['area_km2'], errors='coerce').fillna(0) * 100
        df_ucs = df_ucs.sort_values('area_km2', ascending=False)
        df_ucs.columns = ['Nome da UC', 'Área (km²)', 'Área (ha)']
        st.dataframe(df_ucs, use_container_width=True, hide_index=True)
    else:
        st.info("Não há dados de UCs disponíveis.")
    
    st.subheader("Tabela Unificada por Município")
    mostrar_tabela_unificada(gdf_alertas_raw, gdf_cnuc_raw, gdf_sigef_raw)
    st.divider()

with tabs[1]:
    st.header("Desmatamento")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("Análise de alertas de desmatamento, com dados do MapBiomas Alerta.")

    anos_disponiveis = ['Todos'] + sorted(gdf_alertas_raw['anodetec'].dropna().unique().astype(int).tolist()) if 'anodetec' in gdf_alertas_raw.columns and not gdf_alertas_raw.empty else ['Todos']
    ano_selecionado = st.selectbox('Filtrar por Ano de Detecção:', anos_disponiveis, key="filtro_ano_desmat")
    
    gdf_alertas_filtrado = gdf_alertas_raw[gdf_alertas_raw['anodetec'] == ano_selecionado] if ano_selecionado != 'Todos' else gdf_alertas_raw
    st.divider()

    col_charts, col_map = st.columns([2, 3], gap="large")
    with col_charts:
        st.subheader("Desmatamento por Localização")
        fig_desmat_uc = fig_desmatamento_uc(gdf_cnuc_raw, gdf_alertas_filtrado)
        if not fig_desmat_uc.data:
            st.info("Nenhum alerta de desmatamento sobre UCs para o período selecionado.")
        else:
            st.plotly_chart(fig_desmat_uc, use_container_width=True)

    with col_map:
        st.subheader("Geometrias dos Alertas Filtrados")

        if not gdf_alertas_filtrado.empty:
            try:
                fig_alertas_geom = go.Figure(go.Choroplethmapbox(
                    geojson=gdf_alertas_filtrado.__geo_interface__, locations=gdf_alertas_filtrado.index,
                    z=pd.to_numeric(gdf_alertas_filtrado['areaha'], errors='coerce').fillna(0),
                    colorscale="Reds", showscale=True, marker_opacity=0.7, marker_line_width=1,
                    hovertemplate="<b>Área:</b> %{z:.2f} ha<extra></extra>"
                ))
                fig_alertas_geom.update_layout(
                    mapbox=dict(style="open-street-map", zoom=8, center=centro),
                    margin=dict(l=0, r=0, t=0, b=0), height=500
                )
                st.plotly_chart(fig_alertas_geom, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao criar mapa de alertas: {e}")
        else:
            st.info("Não há alertas para o período selecionado.")

    st.divider()
    st.subheader("Evolução Temporal do Desmatamento (Geral)")
    
    fig_temporal = fig_desmatamento_temporal(gdf_alertas_raw)
    if not fig_temporal.data:
        st.info("Não há dados temporais para exibir.")
    else:
        st.plotly_chart(fig_temporal, use_container_width=True)

    st.subheader("Resumo dos Alertas")
    if not gdf_alertas_filtrado.empty and 'areaha' in gdf_alertas_filtrado.columns:
        gdf_temp = gdf_alertas_filtrado.copy()
        gdf_temp['areaha'] = pd.to_numeric(gdf_temp['areaha'], errors='coerce').fillna(0)
        
        total_area = gdf_temp['areaha'].sum()
        total_registros = len(gdf_temp[gdf_temp['areaha'] > 0])
        
        if total_area > 0:
            resumo = pd.DataFrame({
                'Métrica': ['Total de Registros', 'Área Total (ha)', 'Área Média por Alerta (ha)'],
                'Valor': [total_registros, f"{total_area:,.2f}", f"{total_area/total_registros:,.2f}" if total_registros > 0 else "0"]
            })
            st.dataframe(resumo, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum alerta com área válida encontrado.")
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

        st.header("Ranking de Municípios por Indicadores de Queimadas")
        st.caption("Classifica municípios pelo maior registro de cada indicador.")
        
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            pass  
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
        st.subheader("Principais Motivações dos Processos")
        if not df_processos.empty:
            fig2 = fig_ranking_assuntos(df_processos)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Figura 4.1: Top 10 assuntos mais recorrentes nos processos judiciais.")
        else:
            st.warning("Dados não disponíveis")

    with col2:
        st.subheader("Classes Processuais")
        if not df_processos.empty:
            graficos = criar_graficos_processos(df_processos)
            if 'classes' in graficos:
                st.plotly_chart(graficos['classes'], use_container_width=True)
                st.caption("Figura 4.2: Top 10 classes processuais mais frequentes.")
            else:
                st.warning("Dados de classes processuais não disponíveis")

    st.divider()
                
    st.subheader("Histórico de Ajuizamento de Ações") 
    if not df_processos.empty:
        fig3 = fig_evolucao_temporal_processos(df_processos)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Figura 4.3: Evolução anual da quantidade de novos processos ajuizados.")
    else:
        st.warning("Dados não disponíveis")

    st.divider()

    st.subheader("Ranking de Processos por Município")
    if not df_processos.empty:
        df_ranking = df_processos.groupby('municipio').agg({
            'data_ajuizamento': ['count', 'min', 'max']
        }).round(2)

df_ranking.columns = ['Total Processos', 'Data Inicial', 'Data Final']
df_ranking = df_ranking.reset_index()
df_ranking = df_ranking.sort_values('Total Processos', ascending=False)
df_ranking.insert(0, 'Posição', range(1, len(df_ranking) + 1))

df_ranking['Data Inicial'] = pd.to_datetime(df_ranking['Data Inicial']).dt.strftime('%d/%m/%Y')
df_ranking['Data Final'] = pd.to_datetime(df_ranking['Data Final']).dt.strftime('%d/%m/%Y')

st.dataframe(
    df_ranking,
    use_container_width=True,
    hide_index=True,
    height=400
)

st.caption("Tabela 4.1: Ranking de municípios por quantidade de processos judiciais.")
