
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Starbucks America | Segmentación",
    page_icon="☕",
    layout="wide"
)

# ============================================================
# FUNCIONES
# ============================================================

def moda_principal(x):
    moda = x.mode()
    return moda.iloc[0] if not moda.empty else np.nan


@st.cache_data
def cargar_datos():
    df = pd.read_csv("s_order.csv")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    df["order_hour"] = pd.to_datetime(
        df["order_time"],
        format="%H:%M",
        errors="coerce"
    ).dt.hour

    df["time_slot"] = pd.cut(
        df["order_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Madrugada", "Mañana", "Tarde", "Noche"],
        right=False
    )

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day

    return df


@st.cache_data
def crear_base_clientes(df):
    fecha_referencia = df["order_date"].max() + pd.Timedelta(days=1)

    customer_features = df.groupby("customer_id").agg(
        # Variables RFM
        recency=("order_date", lambda x: (fecha_referencia - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("total_spend", "sum"),

        # Variables de comportamiento
        avg_ticket=("total_spend", "mean"),
        avg_cart_size=("cart_size", "mean"),
        avg_customizations=("num_customizations", "mean"),
        avg_fulfillment_time=("fulfillment_time_min", "mean"),
        avg_satisfaction=("customer_satisfaction", "mean"),

        # Proporciones de comportamiento
        food_order_rate=("has_food_item", "mean"),
        order_ahead_rate=("order_ahead", "mean"),

        # Variables principales por cliente
        main_region=("region", moda_principal),
        main_store_location_type=("store_location_type", moda_principal),
        main_order_channel=("order_channel", moda_principal),
        main_age_group=("customer_age_group", moda_principal),
        main_gender=("customer_gender", moda_principal),
        rewards_member=("is_rewards_member", moda_principal),
        main_drink_category=("drink_category", moda_principal),
        main_time_slot=("time_slot", moda_principal)
    ).reset_index()

    return customer_features


@st.cache_data
def segmentar_clientes(customer_features):
    customer_features = customer_features.copy()

    # ========================================================
    # SEGMENTACIÓN RFM
    # ========================================================

    rfm_vars = ["recency", "frequency", "monetary"]
    X_rfm = customer_features[rfm_vars]

    scaler_rfm = StandardScaler()
    X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

    # Evaluación de K para RFM
    resultados_k = []

    for k in range(2, 7):
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_rfm_scaled)

        score = silhouette_score(
            X_rfm_scaled,
            labels_temp,
            sample_size=min(5000, X_rfm_scaled.shape[0]),
            random_state=42
        )

        resultados_k.append({
            "K": k,
            "Inercia": modelo_temp.inertia_,
            "Silhouette": score
        })

    evaluacion_k = pd.DataFrame(resultados_k)

    # Modelo RFM final
    k_rfm = 3
    kmeans_rfm = KMeans(n_clusters=k_rfm, random_state=42, n_init=10)
    customer_features["segmento_rfm"] = kmeans_rfm.fit_predict(X_rfm_scaled)

    resumen_rfm = customer_features.groupby("segmento_rfm").agg(
        clientes=("customer_id", "count"),
        recency_promedio=("recency", "mean"),
        frequency_promedio=("frequency", "mean"),
        monetary_promedio=("monetary", "mean"),
        ticket_promedio=("avg_ticket", "mean"),
        satisfaccion_promedio=("avg_satisfaction", "mean"),
        tasa_pedido_anticipado=("order_ahead_rate", "mean"),
        tasa_pedido_con_comida=("food_order_rate", "mean")
    ).reset_index().round(2)

    # Diccionario definido en el notebook final del grupo
    nombres_segmentos_rfm = {
        2: "Clientes de alto valor",
        0: "Clientes de valor medio",
        1: "Clientes de baja actividad"
    }

    customer_features["nombre_segmento_rfm"] = customer_features["segmento_rfm"].map(nombres_segmentos_rfm)
    resumen_rfm["nombre_segmento_rfm"] = resumen_rfm["segmento_rfm"].map(nombres_segmentos_rfm)

    resumen_rfm_ordenado = resumen_rfm.sort_values(
        by=["monetary_promedio", "frequency_promedio"],
        ascending=False
    )

    # ========================================================
    # SEGMENTACIÓN SOCIO-DEMOGRÁFICA / CONDUCTUAL
    # ========================================================

    socio_cat_vars = [
        "main_region",
        "main_store_location_type",
        "main_order_channel",
        "main_age_group",
        "main_gender",
        "rewards_member",
        "main_time_slot"
    ]

    socio_num_vars = [
        "avg_ticket",
        "avg_cart_size",
        "avg_customizations",
        "avg_satisfaction",
        "order_ahead_rate",
        "food_order_rate"
    ]

    X_socio = customer_features[socio_cat_vars + socio_num_vars].copy()
    X_socio_processed = pd.get_dummies(X_socio, columns=socio_cat_vars, drop_first=True)

    scaler_socio = StandardScaler()
    X_socio_scaled = scaler_socio.fit_transform(X_socio_processed)

    k_socio = 2
    kmeans_socio = KMeans(n_clusters=k_socio, random_state=42, n_init=10)
    customer_features["segmento_socio"] = kmeans_socio.fit_predict(X_socio_scaled)

    nombres_segmentos_socio = {
        0: "Clientes digitales de consumo ampliado",
        1: "Clientes tradicionales de compra rápida"
    }

    customer_features["nombre_segmento_socio"] = customer_features["segmento_socio"].map(nombres_segmentos_socio)

    resumen_socio = customer_features.groupby("segmento_socio").agg(
        clientes=("customer_id", "count"),
        ticket_promedio=("avg_ticket", "mean"),
        carrito_promedio=("avg_cart_size", "mean"),
        personalizaciones_promedio=("avg_customizations", "mean"),
        satisfaccion_promedio=("avg_satisfaction", "mean"),
        tasa_pedido_anticipado=("order_ahead_rate", "mean"),
        tasa_pedido_con_comida=("food_order_rate", "mean")
    ).reset_index().round(2)

    resumen_socio["nombre_segmento_socio"] = resumen_socio["segmento_socio"].map(nombres_segmentos_socio)

    # ========================================================
    # SEGMENTO COMBINADO Y MERCADO META
    # ========================================================

    customer_features["segmento_combinado"] = (
        customer_features["nombre_segmento_rfm"] +
        " | " +
        customer_features["nombre_segmento_socio"]
    )

    resumen_segmentos_combinados = customer_features.groupby("segmento_combinado").agg(
        clientes=("customer_id", "count"),
        recency_promedio=("recency", "mean"),
        frequency_promedio=("frequency", "mean"),
        monetary_promedio=("monetary", "mean"),
        ticket_promedio=("avg_ticket", "mean"),
        carrito_promedio=("avg_cart_size", "mean"),
        satisfaccion_promedio=("avg_satisfaction", "mean"),
        tasa_pedido_anticipado=("order_ahead_rate", "mean"),
        tasa_pedido_con_comida=("food_order_rate", "mean")
    ).reset_index()

    resumen_segmentos_combinados["porcentaje_mercado_total"] = (
        resumen_segmentos_combinados["clientes"] / customer_features.shape[0] * 100
    )

    resumen_segmentos_combinados = resumen_segmentos_combinados[
        [
            "segmento_combinado",
            "clientes",
            "porcentaje_mercado_total",
            "recency_promedio",
            "frequency_promedio",
            "monetary_promedio",
            "ticket_promedio",
            "carrito_promedio",
            "satisfaccion_promedio",
            "tasa_pedido_anticipado",
            "tasa_pedido_con_comida"
        ]
    ].sort_values(
        by=["monetary_promedio", "frequency_promedio"],
        ascending=False
    ).round(2)

    customer_features["mercado_meta_principal"] = np.where(
        (customer_features["nombre_segmento_rfm"] == "Clientes de alto valor") &
        (customer_features["segmento_socio"] == 0),
        "Mercado meta principal",
        "Otros clientes"
    )

    return (
        customer_features,
        evaluacion_k.round(4),
        resumen_rfm_ordenado,
        resumen_socio,
        resumen_segmentos_combinados
    )


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

df = cargar_datos()
clientes = crear_base_clientes(df)
(
    clientes_segmentados,
    evaluacion_k,
    resumen_rfm,
    resumen_socio,
    resumen_segmentos_combinados
) = segmentar_clientes(clientes)

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
        "con una segmentación socio-demográfica/conductual para identificar mercados meta relevantes."
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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gasto total", f"US$ {df['total_spend'].sum():,.0f}")
    col2.metric("Gasto promedio", f"US$ {df['total_spend'].mean():.2f}")
    col3.metric("Satisfacción promedio", f"{df['customer_satisfaction'].mean():.2f}/5")
    col4.metric("Tiempo promedio", f"{df['fulfillment_time_min'].mean():.2f} min")

    st.divider()

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

    st.subheader("Desempeño por canal de compra")

    analisis_canal = df.groupby("order_channel").agg(
        ordenes=("order_id", "count"),
        clientes_unicos=("customer_id", "nunique"),
        gasto_total=("total_spend", "sum"),
        gasto_promedio=("total_spend", "mean"),
        satisfaccion_promedio=("customer_satisfaction", "mean"),
        tiempo_promedio=("fulfillment_time_min", "mean")
    ).reset_index().sort_values("gasto_total", ascending=False).round(2)

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
    La segmentación se construye a nivel de **cliente**, no a nivel de orden.
    Se utilizan dos enfoques complementarios:

    1. **RFM:** identifica valor comercial según recencia, frecuencia y gasto monetario.  
    2. **Socio-demográfico/conductual:** identifica perfiles de consumo según región, tipo de tienda, canal, edad, género, Rewards, horario y variables de comportamiento.
    """)

    st.subheader("Base construida a nivel cliente")

    col1, col2, col3 = st.columns(3)

    col1.metric("Clientes segmentados", f"{clientes.shape[0]:,}")
    col2.metric("Frecuencia promedio", f"{clientes['frequency'].mean():.2f}")
    col3.metric("Gasto promedio acumulado", f"US$ {clientes['monetary'].mean():.2f}")

    st.divider()

    st.subheader("Evaluación de K para RFM")

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

    st.subheader("Segmentación RFM final")

    tabla_rfm = resumen_rfm[
        [
            "nombre_segmento_rfm",
            "clientes",
            "recency_promedio",
            "frequency_promedio",
            "monetary_promedio",
            "ticket_promedio",
            "satisfaccion_promedio",
            "tasa_pedido_anticipado",
            "tasa_pedido_con_comida"
        ]
    ]

    st.dataframe(tabla_rfm, use_container_width=True)

    fig_rfm = px.bar(
        tabla_rfm,
        x="nombre_segmento_rfm",
        y="clientes",
        text="clientes",
        title="Cantidad de clientes por segmento RFM"
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

    st.divider()

    st.subheader("Segmentación socio-demográfica/conductual")

    tabla_socio = resumen_socio[
        [
            "nombre_segmento_socio",
            "clientes",
            "ticket_promedio",
            "carrito_promedio",
            "personalizaciones_promedio",
            "satisfaccion_promedio",
            "tasa_pedido_anticipado",
            "tasa_pedido_con_comida"
        ]
    ]

    st.dataframe(tabla_socio, use_container_width=True)

    fig_socio = px.bar(
        tabla_socio,
        x="nombre_segmento_socio",
        y="clientes",
        text="clientes",
        title="Cantidad de clientes por segmento socio-demográfico/conductual"
    )
    st.plotly_chart(fig_socio, use_container_width=True)

    st.divider()

    st.subheader("Cruce entre ambos modelos")

    st.dataframe(resumen_segmentos_combinados, use_container_width=True)

    st.success(
        "El mercado meta no se define solo por alto valor RFM. "
        "En el modelo final se cruza alto valor RFM con el segmento de clientes digitales de consumo ampliado."
    )

# ============================================================
# SECCIÓN 5: MERCADO META Y RECOMENDACIÓN
# ============================================================

elif seccion == "5. Mercado meta y recomendación":
    st.header("5. Mercado meta y recomendación")

    st.markdown("""
    El mercado meta principal se define como el cruce entre:

    **Clientes de alto valor RFM**  
    y  
    **Clientes digitales de consumo ampliado**.
    """)

    mercado_meta = clientes_segmentados[
        clientes_segmentados["mercado_meta_principal"] == "Mercado meta principal"
    ].copy()

    porcentaje_mercado_meta = mercado_meta.shape[0] / clientes_segmentados.shape[0] * 100

    st.subheader("Mercado meta principal")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Clientes objetivo", f"{mercado_meta.shape[0]:,}")
    col2.metric("% del total", f"{porcentaje_mercado_meta:.2f}%")
    col3.metric("Monetary promedio", f"US$ {mercado_meta['monetary'].mean():.2f}")
    col4.metric("Frequency promedio", f"{mercado_meta['frequency'].mean():.2f}")

    st.success(
        "Este grupo concentra clientes frecuentes, recientes, rentables y con perfil digital, "
        "por lo que representa el mercado meta más atractivo para la estrategia de expansión."
    )

    st.divider()

    st.subheader("Perfil categórico del mercado meta")

    perfil_vars = [
        "main_region",
        "main_store_location_type",
        "main_order_channel",
        "main_age_group",
        "main_gender",
        "rewards_member",
        "main_time_slot",
        "main_drink_category"
    ]

    perfil_categorico = []

    for col in perfil_vars:
        categoria_principal = mercado_meta[col].mode()[0]
        porcentaje_categoria = mercado_meta[col].value_counts(normalize=True).iloc[0] * 100

        perfil_categorico.append({
            "Variable": col,
            "Categoría principal": categoria_principal,
            "Porcentaje": round(porcentaje_categoria, 2)
        })

    perfil_categorico = pd.DataFrame(perfil_categorico)

    st.dataframe(perfil_categorico, use_container_width=True)

    top_region = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_region",
        "Categoría principal"
    ].values[0]

    top_store_type = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_store_location_type",
        "Categoría principal"
    ].values[0]

    top_channel = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_order_channel",
        "Categoría principal"
    ].values[0]

    top_age_group = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_age_group",
        "Categoría principal"
    ].values[0]

    top_drink = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_drink_category",
        "Categoría principal"
    ].values[0]

    top_time_slot = perfil_categorico.loc[
        perfil_categorico["Variable"] == "main_time_slot",
        "Categoría principal"
    ].values[0]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        mercado_meta_region = mercado_meta["main_region"].value_counts().reset_index()
        mercado_meta_region.columns = ["Región", "Cantidad de clientes"]

        fig_region_meta = px.bar(
            mercado_meta_region,
            x="Región",
            y="Cantidad de clientes",
            text="Cantidad de clientes",
            title="Distribución del mercado meta por región"
        )
        st.plotly_chart(fig_region_meta, use_container_width=True)

    with col2:
        mercado_meta_canal = mercado_meta["main_order_channel"].value_counts().reset_index()
        mercado_meta_canal.columns = ["Canal", "Cantidad de clientes"]

        fig_canal_meta = px.bar(
            mercado_meta_canal,
            x="Canal",
            y="Cantidad de clientes",
            text="Cantidad de clientes",
            title="Distribución del mercado meta por canal"
        )
        st.plotly_chart(fig_canal_meta, use_container_width=True)

    st.divider()

    st.subheader("Ranking de oportunidades de expansión")

    mercado_meta_region_tienda = mercado_meta.groupby(
        ["main_region", "main_store_location_type"]
    ).agg(
        clientes=("customer_id", "count"),
        ticket_promedio=("avg_ticket", "mean"),
        monetary_promedio=("monetary", "mean"),
        satisfaccion_promedio=("avg_satisfaction", "mean")
    ).reset_index()

    mercado_meta_region_tienda["Porcentaje_clientes"] = (
        mercado_meta_region_tienda["clientes"] / mercado_meta_region_tienda["clientes"].sum() * 100
    ).round(2)

    mercado_meta_region_tienda = mercado_meta_region_tienda.sort_values(
        by=["clientes", "monetary_promedio"],
        ascending=False
    ).round(2)

    ranking_expansion = mercado_meta_region_tienda.copy()

    ranking_expansion["ranking_oportunidad"] = ranking_expansion["clientes"].rank(
        method="dense",
        ascending=False
    ).astype(int)

    ranking_expansion = ranking_expansion.sort_values("ranking_oportunidad")

    ranking_expansion = ranking_expansion[
        [
            "ranking_oportunidad",
            "main_region",
            "main_store_location_type",
            "clientes",
            "Porcentaje_clientes",
            "ticket_promedio",
            "monetary_promedio",
            "satisfaccion_promedio"
        ]
    ]

    st.dataframe(ranking_expansion, use_container_width=True)

    top_oportunidad = ranking_expansion.iloc[0]

    st.info(
        f"La combinación con mayor concentración de clientes objetivo es "
        f"{top_oportunidad['main_region']} - {top_oportunidad['main_store_location_type']}. "
        "Aun así, conviene mirar también las alternativas cercanas del ranking antes de tomar una decisión definitiva."
    )

    st.divider()

    st.subheader("Resumen estratégico")

    resumen_estrategico = pd.DataFrame({
        "Elemento estratégico": [
            "Mercado meta principal",
            "Perfil dominante",
            "Región prioritaria",
            "Tipo de tienda prioritario",
            "Canal prioritario",
            "Grupo etario principal",
            "Producto principal",
            "Horario de mayor consumo"
        ],
        "Resultado del análisis": [
            "Clientes de alto valor RFM pertenecientes al segmento socio-demográfico 0",
            "Clientes digitales de consumo ampliado",
            top_region,
            top_store_type,
            top_channel,
            top_age_group,
            top_drink,
            top_time_slot
        ]
    })

    st.dataframe(resumen_estrategico, use_container_width=True)

    st.subheader("Recomendaciones de posicionamiento")

    recomendaciones_posicionamiento = pd.DataFrame({
        "Dimensión": [
            "Segmento objetivo",
            "Localización",
            "Canal de atención",
            "Propuesta de valor",
            "Promociones",
            "Producto",
            "Experiencia de compra",
            "Fidelización"
        ],
        "Recomendación": [
            "Priorizar clientes de alto valor con perfil digital y mayor consumo por pedido.",
            f"Evaluar expansión principalmente en zonas {top_store_type} de la región {top_region}.",
            f"Fortalecer el canal {top_channel}, especialmente para pedidos anticipados.",
            "Posicionar Starbucks como una opción rápida, conveniente y personalizable para consumo diario.",
            "Diseñar promociones combinadas de café y comida para aumentar ticket y carrito promedio.",
            f"Utilizar {top_drink} como producto base para campañas, complementado con alimentos.",
            f"Optimizar la operación durante el horario de {top_time_slot}, donde se concentra mayor demanda.",
            "Reforzar el programa Rewards para aumentar recurrencia y retención de clientes valiosos."
        ]
    })

    st.dataframe(recomendaciones_posicionamiento, use_container_width=True)

    st.success(
        "Conclusión: la expansión no debería enfocarse solo en zonas con muchas órdenes, "
        "sino en lugares donde se concentren clientes frecuentes, rentables y con afinidad digital."
    )
