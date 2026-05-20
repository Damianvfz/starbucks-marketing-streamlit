import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# CONFIGURACIÓN GENERAL DE LA APP
# ============================================================

st.set_page_config(
    page_title="Starbucks America | Segmentación",
    page_icon="☕",
    layout="wide"
)

# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    df = pd.read_csv("s_order.csv")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


# ============================================================
# CONSTRUCCIÓN DE BASE A NIVEL CLIENTE
# ============================================================

@st.cache_data
def crear_base_clientes(df):
    fecha_referencia = df["order_date"].max() + pd.Timedelta(days=1)

    clientes = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (fecha_referencia - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("total_spend", "sum"),
        gasto_promedio=("total_spend", "mean"),
        cart_size_promedio=("cart_size", "mean"),
        customizaciones_promedio=("num_customizations", "mean"),
        satisfaccion_promedio=("customer_satisfaction", "mean"),
        tiempo_promedio=("fulfillment_time_min", "mean"),
        porcentaje_food=("has_food_item", "mean"),
        porcentaje_order_ahead=("order_ahead", "mean"),
        rewards_member=("is_rewards_member", "max"),
        region_principal=("region", lambda x: x.mode()[0]),
        tipo_tienda_principal=("store_location_type", lambda x: x.mode()[0]),
        canal_principal=("order_channel", lambda x: x.mode()[0]),
        edad_principal=("customer_age_group", lambda x: x.mode()[0]),
        genero_principal=("customer_gender", lambda x: x.mode()[0])
    ).reset_index()

    return clientes

df = cargar_datos()
clientes = crear_base_clientes(df)
# ============================================================
# TÍTULO GENERAL
# ============================================================

st.title("☕ Starbucks America")
st.subheader("Segmentación, mercado meta y posicionamiento")

# ============================================================
# MENÚ LATERAL
# ============================================================

seccion = st.sidebar.radio(
    "Secciones",
    [
        "1. Contexto",
        "2. Diagnóstico de datos",
        "3. Análisis descriptivo",
        "4. Segmentación",
        "5. Mercado meta y recomendación"
    ]
)

# ============================================================
# SECCIÓN 1: CONTEXTO
# ============================================================

if seccion == "1. Contexto":
    st.header("1. Contexto del análisis")

    st.markdown("""
    Este análisis se centra en el mercado **Starbucks America**, utilizando la base de transacciones `s_order.csv`.

    El objetivo es identificar segmentos de clientes relevantes para un inversionista interesado en evaluar la apertura de nuevas franquicias en Estados Unidos.

    La pregunta central del análisis es:

    **¿Qué tipos de clientes son más atractivos para Starbucks y dónde conviene enfocar una estrategia de expansión?**
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Órdenes", f"{df['order_id'].nunique():,}")
    col2.metric("Clientes únicos", f"{df['customer_id'].nunique():,}")
    col3.metric("Tiendas", f"{df['store_id'].nunique():,}")

    st.info(
        "La estrategia del análisis consiste en combinar una segmentación RFM, enfocada en valor comercial, "
        "con una segmentación de perfil de cliente para identificar mercados meta relevantes."
    )

# ============================================================
# SECCIÓN 2: DIAGNÓSTICO DE DATOS
# ============================================================

elif seccion == "2. Diagnóstico de datos":
    st.header("2. Diagnóstico de datos")

    st.markdown("""
    Antes de segmentar, se revisó la calidad general de la base de datos.
    Esta etapa permite confirmar si existen valores nulos, duplicados o problemas estructurales.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filas", f"{df.shape[0]:,}")
    col2.metric("Columnas", f"{df.shape[1]:,}")
    col3.metric("Valores nulos", f"{df.isnull().sum().sum():,}")
    col4.metric("Órdenes duplicadas", f"{df['order_id'].duplicated().sum():,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fecha mínima", df["order_date"].min().strftime("%Y-%m-%d"))

    with col2:
        st.metric("Fecha máxima", df["order_date"].max().strftime("%Y-%m-%d"))

    st.subheader("Vista inicial de la base")
    st.dataframe(df.head(10), use_container_width=True)

    st.success(
        "La base contiene 100.000 órdenes, no presenta valores nulos ni órdenes duplicadas, "
        "y cubre el periodo 2024–2025. Por lo tanto, se considera adecuada para construir las segmentaciones."
    )

# ============================================================
# SECCIÓN 3: ANÁLISIS DESCRIPTIVO
# ============================================================

elif seccion == "3. Análisis descriptivo":
    st.header("3. Análisis descriptivo del mercado")

    st.markdown("""
    En esta etapa se analizan patrones generales de consumo antes de construir los segmentos.
    El foco está en identificar diferencias por **canal de compra**, **región** y **tipo de tienda**.
    """)

    # ========================================================
    # INDICADORES GENERALES
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gasto total", f"US$ {df['total_spend'].sum():,.0f}")
    col2.metric("Gasto promedio", f"US$ {df['total_spend'].mean():.2f}")
    col3.metric("Satisfacción promedio", f"{df['customer_satisfaction'].mean():.2f}/5")
    col4.metric("Tiempo promedio", f"{df['fulfillment_time_min'].mean():.2f} min")

    st.divider()

    # ========================================================
    # GRÁFICOS SIMPLES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:
        ordenes_canal = df["order_channel"].value_counts().reset_index()
        ordenes_canal.columns = ["Canal", "Órdenes"]

        fig_canal = px.bar(
            ordenes_canal,
            x="Canal",
            y="Órdenes",
            text="Órdenes",
            title="Órdenes por canal de compra"
        )

        st.plotly_chart(fig_canal, use_container_width=True)

    with col2:
        ordenes_region = df["region"].value_counts().reset_index()
        ordenes_region.columns = ["Región", "Órdenes"]

        fig_region = px.bar(
            ordenes_region,
            x="Región",
            y="Órdenes",
            text="Órdenes",
            title="Órdenes por región"
        )

        st.plotly_chart(fig_region, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        ordenes_tienda = df["store_location_type"].value_counts().reset_index()
        ordenes_tienda.columns = ["Tipo de tienda", "Órdenes"]

        fig_tienda = px.bar(
            ordenes_tienda,
            x="Tipo de tienda",
            y="Órdenes",
            text="Órdenes",
            title="Órdenes por tipo de tienda"
        )

        st.plotly_chart(fig_tienda, use_container_width=True)

    with col2:
        ordenes_edad = df["customer_age_group"].value_counts().reset_index()
        ordenes_edad.columns = ["Grupo etario", "Órdenes"]

        fig_edad = px.bar(
            ordenes_edad,
            x="Grupo etario",
            y="Órdenes",
            text="Órdenes",
            title="Órdenes por grupo etario"
        )

        st.plotly_chart(fig_edad, use_container_width=True)

    st.divider()

    # ========================================================
    # TABLA CLAVE POR CANAL
    # ========================================================

    st.subheader("Desempeño por canal de compra")

    analisis_canal = df.groupby("order_channel").agg(
        ordenes=("order_id", "count"),
        clientes_unicos=("customer_id", "nunique"),
        gasto_total=("total_spend", "sum"),
        gasto_promedio=("total_spend", "mean"),
        satisfaccion_promedio=("customer_satisfaction", "mean"),
        tiempo_promedio=("fulfillment_time_min", "mean")
    ).reset_index().sort_values("gasto_total", ascending=False)

    analisis_canal = analisis_canal.round(2)

    st.dataframe(analisis_canal, use_container_width=True)

    st.success(
        "El canal Mobile App destaca como el canal más relevante: concentra la mayor cantidad de órdenes, "
        "el mayor gasto total, el mayor gasto promedio y una satisfacción promedio superior al resto. "
        "Este hallazgo será clave para definir el posicionamiento."
    )

# ============================================================
# SECCIÓN 4: SEGMENTACIÓN
# ============================================================

elif seccion == "4. Segmentación":
    st.header("4. Segmentación de clientes")

    st.markdown("""
    Para segmentar el mercado, primero se transformó la base desde nivel de **orden** a nivel de **cliente**.
    Luego se aplicó una segmentación **RFM**, que clasifica a los clientes según:
    
    - **Recency:** días desde la última compra.
    - **Frequency:** cantidad de compras realizadas.
    - **Monetary:** gasto total acumulado.
    """)

    st.subheader("Base construida a nivel cliente")

    col1, col2, col3 = st.columns(3)

    col1.metric("Clientes segmentados", f"{clientes.shape[0]:,}")
    col2.metric("Frecuencia promedio", f"{clientes['frequency'].mean():.2f}")
    col3.metric("Gasto promedio acumulado", f"US$ {clientes['monetary'].mean():.2f}")

    st.divider()

    # ========================================================
    # VARIABLES RFM
    # ========================================================

    variables_rfm = ["recency", "frequency", "monetary"]

    X_rfm = clientes[variables_rfm]

    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(X_rfm)

    # ========================================================
    # EVALUACIÓN DE K
    # ========================================================

    st.subheader("Evaluación del número de segmentos")

    resultados_k = []

    for k in range(2, 7):
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas_temp = modelo_temp.fit_predict(X_rfm_scaled)

        silhouette = silhouette_score(
            X_rfm_scaled,
            etiquetas_temp,
            sample_size=min(5000, len(clientes)),
            random_state=42
        )

        resultados_k.append({
            "K": k,
            "Inercia": modelo_temp.inertia_,
            "Silhouette": silhouette
        })

    evaluacion_k = pd.DataFrame(resultados_k)

    col1, col2 = st.columns(2)

    with col1:
        fig_codo = px.line(
            evaluacion_k,
            x="K",
            y="Inercia",
            markers=True,
            title="Método del codo"
        )
        st.plotly_chart(fig_codo, use_container_width=True)

    with col2:
        fig_silhouette = px.line(
            evaluacion_k,
            x="K",
            y="Silhouette",
            markers=True,
            title="Silhouette Score"
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)

    mejor_k = evaluacion_k.loc[evaluacion_k["Silhouette"].idxmax(), "K"]

    st.info(
        f"Según Silhouette, el mejor K es {int(mejor_k)}. "
        "Sin embargo, se utiliza K=3 porque permite una segmentación comercial más específica: "
        "clientes de alto valor, valor medio y baja actividad."
    )

    st.divider()

    # ========================================================
    # MODELO FINAL RFM CON K=3
    # ========================================================

    st.subheader("Segmentación RFM final")

    k_final = 3

    modelo_rfm = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    clientes["segmento_rfm"] = modelo_rfm.fit_predict(X_rfm_scaled)

    resumen_rfm = clientes.groupby("segmento_rfm").agg(
        clientes=("customer_id", "count"),
        recency_promedio=("recency", "mean"),
        frequency_promedio=("frequency", "mean"),
        monetary_promedio=("monetary", "mean"),
        gasto_promedio=("gasto_promedio", "mean"),
        satisfaccion_promedio=("satisfaccion_promedio", "mean"),
        order_ahead_promedio=("porcentaje_order_ahead", "mean"),
        food_promedio=("porcentaje_food", "mean")
    ).reset_index()

    # Orden comercial: más monetary, más frequency y menor recency
    resumen_rfm["score_valor"] = (
        resumen_rfm["monetary_promedio"].rank(ascending=True) +
        resumen_rfm["frequency_promedio"].rank(ascending=True) +
        resumen_rfm["recency_promedio"].rank(ascending=False)
    )

    resumen_rfm = resumen_rfm.sort_values("score_valor", ascending=False).reset_index(drop=True)

    nombres_segmentos = [
        "Clientes de alto valor",
        "Clientes de valor medio",
        "Clientes de baja actividad"
    ]

    mapa_segmentos = {}

    for i, fila in resumen_rfm.iterrows():
        mapa_segmentos[fila["segmento_rfm"]] = nombres_segmentos[i]

    clientes["nombre_segmento_rfm"] = clientes["segmento_rfm"].map(mapa_segmentos)
    resumen_rfm["nombre_segmento_rfm"] = resumen_rfm["segmento_rfm"].map(mapa_segmentos)

    resumen_rfm = resumen_rfm[
        [
            "nombre_segmento_rfm",
            "clientes",
            "recency_promedio",
            "frequency_promedio",
            "monetary_promedio",
            "gasto_promedio",
            "satisfaccion_promedio",
            "order_ahead_promedio",
            "food_promedio"
        ]
    ].round(2)

    st.dataframe(resumen_rfm, use_container_width=True)

    fig_segmentos = px.bar(
        resumen_rfm,
        x="nombre_segmento_rfm",
        y="clientes",
        text="clientes",
        title="Cantidad de clientes por segmento RFM"
    )

    st.plotly_chart(fig_segmentos, use_container_width=True)

    st.success(
        "La segmentación RFM permite identificar un grupo de clientes de alto valor, "
        "caracterizado por mayor frecuencia de compra, mayor gasto acumulado y menor recencia. "
        "Este grupo será la base para definir el mercado meta principal."
    )

# ============================================================
# SECCIÓN 5: MERCADO META Y RECOMENDACIÓN
# ============================================================

elif seccion == "5. Mercado meta y recomendación":
    st.header("5. Mercado meta y recomendación")

    st.markdown("""
    A partir de la segmentación RFM, se identifica el segmento de mayor valor comercial.
    Luego, este grupo se analiza según canal principal, región, tipo de tienda y grupo etario,
    para construir una recomendación de expansión y posicionamiento.
    """)

    # ========================================================
    # RECONSTRUIR SEGMENTACIÓN RFM
    # ========================================================

    variables_rfm = ["recency", "frequency", "monetary"]

    X_rfm = clientes[variables_rfm]

    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(X_rfm)

    k_final = 3

    modelo_rfm = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    clientes_final = clientes.copy()
    clientes_final["segmento_rfm"] = modelo_rfm.fit_predict(X_rfm_scaled)

    resumen_rfm = clientes_final.groupby("segmento_rfm").agg(
        clientes=("customer_id", "count"),
        recency_promedio=("recency", "mean"),
        frequency_promedio=("frequency", "mean"),
        monetary_promedio=("monetary", "mean"),
        gasto_promedio=("gasto_promedio", "mean"),
        satisfaccion_promedio=("satisfaccion_promedio", "mean")
    ).reset_index()

    resumen_rfm["score_valor"] = (
        resumen_rfm["monetary_promedio"].rank(ascending=True) +
        resumen_rfm["frequency_promedio"].rank(ascending=True) +
        resumen_rfm["recency_promedio"].rank(ascending=False)
    )

    resumen_rfm = resumen_rfm.sort_values("score_valor", ascending=False).reset_index(drop=True)

    nombres_segmentos = [
        "Clientes de alto valor",
        "Clientes de valor medio",
        "Clientes de baja actividad"
    ]

    mapa_segmentos = {}

    for i, fila in resumen_rfm.iterrows():
        mapa_segmentos[fila["segmento_rfm"]] = nombres_segmentos[i]

    clientes_final["nombre_segmento_rfm"] = clientes_final["segmento_rfm"].map(mapa_segmentos)

    # ========================================================
    # DEFINICIÓN DEL MERCADO META
    # ========================================================

    mercado_meta = clientes_final[
        clientes_final["nombre_segmento_rfm"] == "Clientes de alto valor"
    ].copy()

    porcentaje_mercado_meta = len(mercado_meta) / len(clientes_final) * 100

    st.subheader("Mercado meta principal")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Clientes objetivo", f"{len(mercado_meta):,}")
    col2.metric("% del total", f"{porcentaje_mercado_meta:.2f}%")
    col3.metric("Gasto acumulado promedio", f"US$ {mercado_meta['monetary'].mean():.2f}")
    col4.metric("Frecuencia promedio", f"{mercado_meta['frequency'].mean():.2f}")

    st.success(
        "El mercado meta principal corresponde a los clientes de alto valor RFM: "
        "clientes con mayor frecuencia de compra, mayor gasto acumulado y compras más recientes."
    )

    st.divider()

    # ========================================================
    # PERFIL DEL MERCADO META
    # ========================================================

    st.subheader("Perfil del mercado meta")

    col1, col2 = st.columns(2)

    with col1:
        canal_meta = mercado_meta["canal_principal"].value_counts().reset_index()
        canal_meta.columns = ["Canal principal", "Clientes"]

        fig_canal_meta = px.bar(
            canal_meta,
            x="Canal principal",
            y="Clientes",
            text="Clientes",
            title="Canal principal del mercado meta"
        )

        st.plotly_chart(fig_canal_meta, use_container_width=True)

    with col2:
        region_meta = mercado_meta["region_principal"].value_counts().reset_index()
        region_meta.columns = ["Región principal", "Clientes"]

        fig_region_meta = px.bar(
            region_meta,
            x="Región principal",
            y="Clientes",
            text="Clientes",
            title="Región principal del mercado meta"
        )

        st.plotly_chart(fig_region_meta, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        tienda_meta = mercado_meta["tipo_tienda_principal"].value_counts().reset_index()
        tienda_meta.columns = ["Tipo de tienda", "Clientes"]

        fig_tienda_meta = px.bar(
            tienda_meta,
            x="Tipo de tienda",
            y="Clientes",
            text="Clientes",
            title="Tipo de tienda principal del mercado meta"
        )

        st.plotly_chart(fig_tienda_meta, use_container_width=True)

    with col2:
        edad_meta = mercado_meta["edad_principal"].value_counts().reset_index()
        edad_meta.columns = ["Grupo etario", "Clientes"]

        fig_edad_meta = px.bar(
            edad_meta,
            x="Grupo etario",
            y="Clientes",
            text="Clientes",
            title="Grupo etario del mercado meta"
        )

        st.plotly_chart(fig_edad_meta, use_container_width=True)

    st.divider()

    # ========================================================
    # TABLA DE UBICACIÓN: REGIÓN + TIPO DE TIENDA
    # ========================================================

    st.subheader("Oportunidades por región y tipo de tienda")

    oportunidad = mercado_meta.groupby(
        ["region_principal", "tipo_tienda_principal"]
    ).agg(
        clientes_objetivo=("customer_id", "count"),
        gasto_promedio_acumulado=("monetary", "mean"),
        frecuencia_promedio=("frequency", "mean"),
        satisfaccion_promedio=("satisfaccion_promedio", "mean")
    ).reset_index()

    oportunidad["porcentaje_clientes"] = (
        oportunidad["clientes_objetivo"] / oportunidad["clientes_objetivo"].sum() * 100
    )

    oportunidad = oportunidad.sort_values(
        by=["clientes_objetivo", "gasto_promedio_acumulado"],
        ascending=False
    ).round(2)

    st.dataframe(oportunidad, use_container_width=True)

    top_oportunidad = oportunidad.iloc[0]

    st.info(
        f"La combinación con mayor concentración de clientes objetivo es "
        f"{top_oportunidad['region_principal']} - {top_oportunidad['tipo_tienda_principal']}. "
        "Sin embargo, la decisión de expansión debe considerar también otras combinaciones cercanas del ranking, "
        "ya que las diferencias pueden no ser extremadamente amplias."
    )

    st.divider()

    # ========================================================
    # RECOMENDACIONES FINALES
    # ========================================================

    st.subheader("Recomendación estratégica")

    canal_prioritario = mercado_meta["canal_principal"].mode()[0]
    region_prioritaria = mercado_meta["region_principal"].mode()[0]
    tienda_prioritaria = mercado_meta["tipo_tienda_principal"].mode()[0]
    edad_prioritaria = mercado_meta["edad_principal"].mode()[0]

    recomendaciones = pd.DataFrame({
        "Dimensión": [
            "Mercado meta",
            "Canal prioritario",
            "Ubicación sugerida",
            "Perfil de cliente",
            "Posicionamiento",
            "Acción comercial"
        ],
        "Recomendación": [
            "Clientes de alto valor RFM: frecuentes, recientes y con mayor gasto acumulado.",
            f"Fortalecer {canal_prioritario}, ya que aparece como canal dominante dentro del mercado meta.",
            f"Priorizar zonas {tienda_prioritaria} en la región {region_prioritaria}, evaluando también alternativas cercanas del ranking.",
            f"Enfocar la comunicación en clientes del grupo {edad_prioritaria}, sin excluir otros grupos relevantes.",
            "Posicionar Starbucks como una opción rápida, conveniente, digital y personalizable para consumidores frecuentes.",
            "Potenciar Mobile App, Rewards, pedidos anticipados y promociones combinadas de bebida + comida."
        ]
    })

    st.dataframe(recomendaciones, use_container_width=True)

    st.success(
        "Conclusión: la expansión no debería enfocarse solo en zonas con muchas órdenes, "
        "sino en lugares donde se concentran clientes frecuentes, rentables y con afinidad digital."
    )