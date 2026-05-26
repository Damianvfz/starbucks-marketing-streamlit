
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


def normalizar_columna(columna):
    minimo = columna.min()
    maximo = columna.max()

    if maximo == minimo:
        return columna * 0 + 1
    return (columna - minimo) / (maximo - minimo)


def titulo_seccion(numero, titulo):
    st.markdown(f"""
    <div style="
        background-color:#F5F5F5;
        padding:18px 22px;
        border-radius:16px;
        border-left:7px solid #0B3D2E;
        margin-top:36px;
        margin-bottom:18px;
    ">
        <h2 style="margin:0; color:#12372A;">{numero}. {titulo}</h2>
    </div>
    """, unsafe_allow_html=True)


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

        # Variables de comportamiento usadas para caracterización
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
    # SEGMENTACIÓN RFM CON K-MEANS
    # ========================================================

    rfm_vars = ["recency", "frequency", "monetary"]
    X_rfm = customer_features[rfm_vars]

    scaler_rfm = StandardScaler()
    X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

    resultados_k_rfm = []

    for k in range(2, 7):
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_rfm_scaled)

        score = silhouette_score(
            X_rfm_scaled,
            labels_temp,
            sample_size=min(5000, X_rfm_scaled.shape[0]),
            random_state=42
        )

        resultados_k_rfm.append({
            "K": k,
            "Inercia": modelo_temp.inertia_,
            "Silhouette": score
        })

    evaluacion_k_rfm = pd.DataFrame(resultados_k_rfm)

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
    # SEGMENTACIÓN SOCIO-DEMOGRÁFICA CON K-MEANS
    # ========================================================
    # Nota: según el notebook final, este modelo usa solo variables
    # socio-demográficas/categóricas. Las variables de comportamiento
    # se usan después para caracterizar los segmentos, no para crearlos.

    socio_cat_vars = [
        "main_region",
        "main_store_location_type",
        "main_age_group",
        "main_gender"
    ]

    X_socio = customer_features[socio_cat_vars].copy()
    X_socio = X_socio.fillna("No informado")

    for col in socio_cat_vars:
        X_socio[col] = X_socio[col].astype(str)

    X_socio_processed = pd.get_dummies(X_socio, columns=socio_cat_vars, drop_first=False)

    resultados_k_socio = []

    for k in range(2, 11):
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = modelo_temp.fit_predict(X_socio_processed)

        score = silhouette_score(
            X_socio_processed,
            labels_temp,
            sample_size=min(5000, X_socio_processed.shape[0]),
            random_state=42
        )

        resultados_k_socio.append({
            "K": k,
            "Inercia": modelo_temp.inertia_,
            "Silhouette": score
        })

    evaluacion_k_socio = pd.DataFrame(resultados_k_socio)

    k_socio = 5
    kmeans_socio = KMeans(n_clusters=k_socio, random_state=42, n_init=10)
    customer_features["segmento_socio"] = kmeans_socio.fit_predict(X_socio_processed)

    nombres_segmentos_socio = {
        0: "Consumidores suburbanos personalizados",
        1: "Consumidores rurales de alto ticket",
        2: "Consumidores tradicionales",
        3: "Consumidores suburbanos de carrito alto",
        4: "Consumidores urbanos de consumo moderado"
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
    # CRUCE Y MERCADO META
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
        by=["monetary_promedio", "frequency_promedio", "ticket_promedio"],
        ascending=False
    ).round(2)

    segmento_socio_objetivo = 0
    nombre_socio_objetivo = nombres_segmentos_socio[segmento_socio_objetivo]

    customer_features["mercado_meta_principal"] = np.where(
        (customer_features["nombre_segmento_rfm"] == "Clientes de alto valor") &
        (customer_features["segmento_socio"] == segmento_socio_objetivo),
        "Mercado meta principal",
        "Otros clientes"
    )

    return (
        customer_features,
        evaluacion_k_rfm.round(4),
        evaluacion_k_socio.round(4),
        resumen_rfm_ordenado,
        resumen_socio,
        resumen_segmentos_combinados,
        nombre_socio_objetivo
    )


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

df = cargar_datos()
clientes = crear_base_clientes(df)

with st.spinner("Preparando segmentaciones..."):
    (
        clientes_segmentados,
        evaluacion_k_rfm,
        evaluacion_k_socio,
        resumen_rfm,
        resumen_socio,
        resumen_segmentos_combinados,
        nombre_socio_objetivo
    ) = segmentar_clientes(clientes)

# ============================================================
# PORTADA
# ============================================================

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0B3D2E, #12372A);
    padding: 36px;
    border-radius: 22px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.12);
">
    <h1 style="color:white; margin-bottom: 5px;">☕ Starbucks America</h1>
    <h3 style="color:#D8F3DC; font-weight:400; margin-top:0;">
        Segmentación, mercado meta y posicionamiento
    </h3>
    <p style="color:#F1F1F1; font-size:18px; max-width:950px;">
        Presentación interactiva para identificar segmentos de clientes de alto valor y orientar decisiones de expansión de franquicias.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("Desplázate hacia abajo para recorrer el análisis completo.")

# ============================================================
# SECCIÓN 1: CONTEXTO
# ============================================================

titulo_seccion("1", "Contexto del análisis")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Órdenes", f"{df['order_id'].nunique():,}")
col2.metric("Clientes únicos", f"{df['customer_id'].nunique():,}")
col3.metric("Tiendas", f"{df['store_id'].nunique():,}")
col4.metric("Periodo", f"{df['order_date'].min().year}–{df['order_date'].max().year}")

st.markdown("""
El objetivo es identificar segmentos de clientes relevantes para un inversionista interesado en evaluar
la apertura de nuevas franquicias de Starbucks en Estados Unidos.

La pregunta central es:

**¿Qué tipos de clientes son más atractivos para Starbucks y dónde conviene enfocar una estrategia de expansión?**
""")

st.info(
    "La estrategia combina una segmentación RFM, enfocada en valor comercial, "
    "con una segmentación socio-demográfica para identificar mercados meta accionables."
)

# ============================================================
# SECCIÓN 2: DIAGNÓSTICO
# ============================================================

titulo_seccion("2", "Diagnóstico de datos")

st.success(
    "La base se encuentra limpia y es adecuada para el análisis. "
    "No presenta valores nulos ni órdenes duplicadas, por lo que no fue necesario eliminar registros "
    "antes de construir las segmentaciones."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas", f"{df.shape[0]:,}")
col2.metric("Columnas", f"{df.shape[1]:,}")
col3.metric("Valores nulos", f"{df.isnull().sum().sum():,}")
col4.metric("Órdenes duplicadas", f"{df['order_id'].duplicated().sum():,}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Órdenes únicas", f"{df['order_id'].nunique():,}")
col2.metric("Clientes únicos", f"{df['customer_id'].nunique():,}")
col3.metric("Tiendas únicas", f"{df['store_id'].nunique():,}")
col4.metric(
    "Periodo",
    f"{df['order_date'].min().strftime('%Y-%m-%d')} a {df['order_date'].max().strftime('%Y-%m-%d')}"
)

st.subheader("Resumen de variables principales")

variables_numericas = [
    "cart_size",
    "num_customizations",
    "total_spend",
    "fulfillment_time_min",
    "customer_satisfaction"
]

resumen_numerico = df[variables_numericas].describe().round(2).rename(columns={
    "cart_size": "Tamaño carrito",
    "num_customizations": "Personalizaciones",
    "total_spend": "Gasto total",
    "fulfillment_time_min": "Tiempo preparación",
    "customer_satisfaction": "Satisfacción"
})

st.dataframe(resumen_numerico, use_container_width=True)

with st.expander("Ver primeras filas de la base"):
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# SECCIÓN 3: DESCRIPTIVO
# ============================================================

titulo_seccion("3", "Análisis descriptivo del mercado")

st.success(
    "El canal Mobile App aparece como el principal motor del mercado, "
    "ya que concentra más órdenes, mayor gasto total, mayor gasto promedio y una satisfacción superior al resto."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gasto total", f"US$ {df['total_spend'].sum():,.0f}")
col2.metric("Gasto promedio", f"US$ {df['total_spend'].mean():.2f}")
col3.metric("Satisfacción promedio", f"{df['customer_satisfaction'].mean():.2f}/5")
col4.metric("Tiempo promedio", f"{df['fulfillment_time_min'].mean():.2f} min")

st.subheader("Canal de compra: principal hallazgo descriptivo")

analisis_canal = df.groupby("order_channel").agg(
    ordenes=("order_id", "count"),
    clientes_unicos=("customer_id", "nunique"),
    gasto_total=("total_spend", "sum"),
    gasto_promedio=("total_spend", "mean"),
    satisfaccion_promedio=("customer_satisfaction", "mean"),
    tiempo_promedio=("fulfillment_time_min", "mean")
).reset_index().sort_values("gasto_total", ascending=False).round(2)

analisis_canal_presentacion = analisis_canal.rename(columns={
    "order_channel": "Canal",
    "ordenes": "Órdenes",
    "clientes_unicos": "Clientes únicos",
    "gasto_total": "Gasto total",
    "gasto_promedio": "Gasto promedio",
    "satisfaccion_promedio": "Satisfacción promedio",
    "tiempo_promedio": "Tiempo promedio"
})

col1, col2 = st.columns(2)

with col1:
    fig_canal_ordenes = px.bar(
        analisis_canal_presentacion,
        x="Canal",
        y="Órdenes",
        text="Órdenes",
        title="Órdenes por canal de compra"
    )
    st.plotly_chart(fig_canal_ordenes, use_container_width=True)

with col2:
    fig_canal_gasto = px.bar(
        analisis_canal_presentacion,
        x="Canal",
        y="Gasto promedio",
        text="Gasto promedio",
        title="Gasto promedio por canal"
    )
    st.plotly_chart(fig_canal_gasto, use_container_width=True)

st.dataframe(analisis_canal_presentacion, use_container_width=True)

st.info(
    "Mobile App no solo concentra el mayor volumen de órdenes, sino también el mayor gasto promedio. "
    "Esto indica que el canal digital no es solo masivo, sino también más valioso comercialmente."
)

st.subheader("Actividad por región y tipo de tienda")

analisis_region = df.groupby("region").agg(
    ordenes=("order_id", "count"),
    clientes_unicos=("customer_id", "nunique"),
    gasto_total=("total_spend", "sum"),
    gasto_promedio=("total_spend", "mean"),
    satisfaccion_promedio=("customer_satisfaction", "mean")
).reset_index().sort_values("gasto_total", ascending=False).round(2)

analisis_tienda = df.groupby("store_location_type").agg(
    ordenes=("order_id", "count"),
    clientes_unicos=("customer_id", "nunique"),
    gasto_total=("total_spend", "sum"),
    gasto_promedio=("total_spend", "mean"),
    satisfaccion_promedio=("customer_satisfaction", "mean")
).reset_index().sort_values("gasto_total", ascending=False).round(2)

col1, col2 = st.columns(2)

with col1:
    fig_region = px.bar(
        analisis_region,
        x="region",
        y="ordenes",
        text="ordenes",
        title="Órdenes por región"
    )
    fig_region.update_layout(xaxis_title="Región", yaxis_title="Órdenes")
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    fig_tienda = px.bar(
        analisis_tienda,
        x="store_location_type",
        y="ordenes",
        text="ordenes",
        title="Órdenes por tipo de tienda"
    )
    fig_tienda.update_layout(xaxis_title="Tipo de tienda", yaxis_title="Órdenes")
    st.plotly_chart(fig_tienda, use_container_width=True)

st.info(
    "La región West y las tiendas Suburban concentran mayor actividad total. "
    "Sin embargo, la decisión final debe complementarse con la segmentación de clientes."
)

st.subheader("Perfil general de consumo")

col1, col2 = st.columns(2)

with col1:
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

with col2:
    ordenes_bebida = df["drink_category"].value_counts().reset_index()
    ordenes_bebida.columns = ["Categoría de bebida", "Órdenes"]

    fig_bebida = px.bar(
        ordenes_bebida,
        x="Categoría de bebida",
        y="Órdenes",
        text="Órdenes",
        title="Órdenes por categoría de bebida"
    )
    st.plotly_chart(fig_bebida, use_container_width=True)

# ============================================================
# SECCIÓN 4: SEGMENTACIÓN
# ============================================================

titulo_seccion("4", "Segmentación de clientes")

st.success(
    "El modelo RFM identifica el valor comercial del cliente, "
    "mientras que el modelo socio-demográfico permite entender su perfil según ubicación, tipo de tienda, edad y género. "
    "El mercado meta final se define cruzando ambos enfoques."
)

st.subheader("Base construida a nivel cliente")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes segmentados", f"{clientes.shape[0]:,}")
col2.metric("Frecuencia promedio", f"{clientes['frequency'].mean():.2f}")
col3.metric("Gasto acumulado promedio", f"US$ {clientes['monetary'].mean():.2f}")
col4.metric("Ticket promedio", f"US$ {clientes['avg_ticket'].mean():.2f}")

# -----------------------------
# RFM
# -----------------------------

st.subheader("Segmentación RFM con K-Means")

st.markdown("""
El modelo **RFM** clasifica clientes según tres dimensiones: **recency**, **frequency** y **monetary**.
Se utiliza para identificar el valor comercial del cliente.
""")

with st.expander("Ver evaluación del número de segmentos RFM"):
    col1, col2 = st.columns(2)

    with col1:
        fig_codo = px.line(
            evaluacion_k_rfm,
            x="K",
            y="Inercia",
            markers=True,
            title="Método del codo: RFM"
        )
        st.plotly_chart(fig_codo, use_container_width=True)

    with col2:
        fig_silhouette = px.line(
            evaluacion_k_rfm,
            x="K",
            y="Silhouette",
            markers=True,
            title="Silhouette Score: RFM"
        )
        st.plotly_chart(fig_silhouette, use_container_width=True)

    mejor_k_rfm = evaluacion_k_rfm.loc[evaluacion_k_rfm["Silhouette"].idxmax(), "K"]

    col1, col2, col3 = st.columns(3)
    col1.metric("K sugerido por Silhouette", int(mejor_k_rfm))
    col2.metric("StepMix evaluado", "5 clases RFM")
    col3.metric("K elegido", 3)

    st.info(
        "También se evaluó StepMix como método alternativo en el notebook. "
        "StepMix entregó una segmentación más granular, pero se eligió K-Means con K=3 "
        "porque permite explicar mejor el mercado en clientes de alto valor, valor medio y baja actividad."
    )

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
].copy()

tabla_rfm = tabla_rfm.rename(columns={
    "nombre_segmento_rfm": "Segmento RFM",
    "clientes": "Clientes",
    "recency_promedio": "Recency promedio",
    "frequency_promedio": "Frequency promedio",
    "monetary_promedio": "Monetary promedio",
    "ticket_promedio": "Ticket promedio",
    "satisfaccion_promedio": "Satisfacción promedio",
    "tasa_pedido_anticipado": "Tasa pedido anticipado",
    "tasa_pedido_con_comida": "Tasa pedido con comida"
})

st.dataframe(tabla_rfm, use_container_width=True)

fig_rfm = px.bar(
    tabla_rfm,
    x="Segmento RFM",
    y="Clientes",
    text="Clientes",
    title="Clientes por segmento RFM"
)
st.plotly_chart(fig_rfm, use_container_width=True)

# -----------------------------
# SOCIO-DEMOGRÁFICO
# -----------------------------

st.subheader("Segmentación socio-demográfica con K-Means")

st.markdown("""
Esta segmentación utiliza solo variables socio-demográficas/categóricas:
**región principal**, **tipo de tienda principal**, **grupo etario** y **género**.

Las variables de comportamiento, como ticket, personalización o pedido con comida, se usan después para caracterizar los segmentos,
pero no para construir este modelo.
""")

with st.expander("Ver evaluación del número de segmentos socio-demográficos"):
    col1, col2 = st.columns(2)

    with col1:
        fig_codo_socio = px.line(
            evaluacion_k_socio,
            x="K",
            y="Inercia",
            markers=True,
            title="Método del codo: socio-demográfico"
        )
        st.plotly_chart(fig_codo_socio, use_container_width=True)

    with col2:
        fig_silhouette_socio = px.line(
            evaluacion_k_socio,
            x="K",
            y="Silhouette",
            markers=True,
            title="Silhouette Score: socio-demográfico"
        )
        st.plotly_chart(fig_silhouette_socio, use_container_width=True)

    mejor_k_socio = evaluacion_k_socio.loc[evaluacion_k_socio["Silhouette"].idxmax(), "K"]

    col1, col2, col3 = st.columns(3)
    col1.metric("K sugerido por Silhouette", int(mejor_k_socio))
    col2.metric("StepMix evaluado", "5 clases")
    col3.metric("K elegido", 5)

    st.info(
        "Aunque K=7 presenta el mayor Silhouette Score, se selecciona K=5 porque entrega un equilibrio más adecuado "
        "entre separación estadística, simplicidad e interpretabilidad comercial."
    )

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
].copy()

tabla_socio = tabla_socio.rename(columns={
    "nombre_segmento_socio": "Segmento socio-demográfico",
    "clientes": "Clientes",
    "ticket_promedio": "Ticket promedio",
    "carrito_promedio": "Carrito promedio",
    "personalizaciones_promedio": "Personalizaciones promedio",
    "satisfaccion_promedio": "Satisfacción promedio",
    "tasa_pedido_anticipado": "Tasa pedido anticipado",
    "tasa_pedido_con_comida": "Tasa pedido con comida"
})

st.dataframe(tabla_socio, use_container_width=True)

# -----------------------------
# COMPARACIÓN METODOLÓGICA
# -----------------------------

st.subheader("Comparación metodológica: K-Means vs StepMix")

comparacion_metodos = pd.DataFrame({
    "Análisis": [
        "RFM",
        "Socio-demográfico"
    ],
    "K-Means usado en la presentación": [
        "3 segmentos",
        "5 segmentos"
    ],
    "StepMix evaluado en el notebook": [
        "5 clases",
        "5 clases"
    ],
    "Decisión final": [
        "Se mantiene K-Means",
        "Se mantiene K-Means"
    ],
    "Justificación": [
        "Permite separar clientes de alto valor, valor medio y baja actividad.",
        "Permite construir perfiles accionables según ubicación, tipo de tienda, edad y género."
    ]
})

st.dataframe(comparacion_metodos, use_container_width=True)

st.info(
    "StepMix se utiliza como comparación metodológica, pero no se recalcula dentro de Streamlit para evitar que la app quede lenta. "
    "El respaldo del cálculo queda en el notebook final del trabajo."
)

# -----------------------------
# MAPA DE CALOR
# -----------------------------

st.subheader("Mapa de calor: cruce entre segmentos RFM y socio-demográficos")

orden_rfm = [
    "Clientes de alto valor",
    "Clientes de valor medio",
    "Clientes de baja actividad"
]

orden_socio = [
    "Consumidores suburbanos personalizados",
    "Consumidores rurales de alto ticket",
    "Consumidores tradicionales",
    "Consumidores suburbanos de carrito alto",
    "Consumidores urbanos de consumo moderado"
]

matriz_segmentos = pd.crosstab(
    clientes_segmentados["nombre_segmento_rfm"],
    clientes_segmentados["nombre_segmento_socio"]
)

matriz_segmentos = (
    matriz_segmentos
    .reindex(index=orden_rfm, columns=orden_socio, fill_value=0)
    .fillna(0)
    .astype(int)
)

total_clientes_matriz = matriz_segmentos.values.sum()
matriz_pct_total = (matriz_segmentos / total_clientes_matriz * 100).round(1)

texto_heatmap = matriz_segmentos.astype(str) + "<br>(" + matriz_pct_total.astype(str) + "%)"

fig_heatmap = px.imshow(
    matriz_segmentos,
    labels=dict(x="Segmento socio-demográfico", y="Segmento RFM", color="Cantidad de clientes"),
    x=matriz_segmentos.columns,
    y=matriz_segmentos.index,
    color_continuous_scale="YlGnBu",
    aspect="auto",
    title="Cruce K1 (RFM) × K2 (Socio-demográfico)"
)

fig_heatmap.update_traces(
    text=texto_heatmap.values,
    texttemplate="%{text}",
    textfont=dict(size=13)
)

fig_heatmap.update_layout(
    height=520,
    xaxis_tickangle=-20
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# CRUCE DE MODELOS
# -----------------------------

st.subheader("Cruce entre ambos modelos")

col1, col2 = st.columns(2)

with col1:
    clientes_alto_valor = clientes_segmentados[
        clientes_segmentados["nombre_segmento_rfm"] == "Clientes de alto valor"
    ].shape[0]

    st.metric("Clientes de alto valor RFM", f"{clientes_alto_valor:,}")

with col2:
    mercado_meta_n = clientes_segmentados[
        clientes_segmentados["mercado_meta_principal"] == "Mercado meta principal"
    ].shape[0]

    st.metric("Mercado meta principal", f"{mercado_meta_n:,}")

with st.expander("Ver cruce completo entre segmentos"):
    tabla_cruce = resumen_segmentos_combinados.rename(columns={
        "segmento_combinado": "Segmento combinado",
        "clientes": "Clientes",
        "porcentaje_mercado_total": "% del mercado total",
        "recency_promedio": "Recency promedio",
        "frequency_promedio": "Frequency promedio",
        "monetary_promedio": "Monetary promedio",
        "ticket_promedio": "Ticket promedio",
        "carrito_promedio": "Carrito promedio",
        "satisfaccion_promedio": "Satisfacción promedio",
        "tasa_pedido_anticipado": "Tasa pedido anticipado",
        "tasa_pedido_con_comida": "Tasa pedido con comida"
    })

    st.dataframe(tabla_cruce, use_container_width=True)

# ============================================================
# SECCIÓN 5: MERCADO META Y RECOMENDACIÓN
# ============================================================

titulo_seccion("5", "Mercado meta y recomendación")

st.success(
    f"El mercado meta principal corresponde a clientes de alto valor RFM que pertenecen al segmento socio-demográfico: "
    f"{nombre_socio_objetivo}. Este grupo combina alto valor comercial, mayor personalización y consumo complementario."
)

mercado_meta = clientes_segmentados[
    clientes_segmentados["mercado_meta_principal"] == "Mercado meta principal"
].copy()

porcentaje_mercado_meta = mercado_meta.shape[0] / clientes_segmentados.shape[0] * 100

st.subheader("Mercado meta principal")

st.markdown(f"""
El mercado meta se define como:

**Clientes de alto valor RFM**  
+  
**{nombre_socio_objetivo}**
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes objetivo", f"{mercado_meta.shape[0]:,}")
col2.metric("% del total", f"{porcentaje_mercado_meta:.2f}%")
col3.metric("Gasto acumulado promedio", f"US$ {mercado_meta['monetary'].mean():.2f}")
col4.metric("Frecuencia promedio", f"{mercado_meta['frequency'].mean():.2f}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticket promedio", f"US$ {mercado_meta['avg_ticket'].mean():.2f}")
col2.metric("Satisfacción promedio", f"{mercado_meta['avg_satisfaction'].mean():.2f}/5")
col3.metric("Pedido anticipado", f"{mercado_meta['order_ahead_rate'].mean() * 100:.1f}%")
col4.metric("Pedido con comida", f"{mercado_meta['food_order_rate'].mean() * 100:.1f}%")

st.subheader("Perfil dominante del mercado meta")

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

nombres_variables = {
    "main_region": "Región principal",
    "main_store_location_type": "Tipo de tienda principal",
    "main_order_channel": "Canal principal",
    "main_age_group": "Grupo etario principal",
    "main_gender": "Género principal",
    "rewards_member": "Miembro Rewards",
    "main_time_slot": "Horario principal",
    "main_drink_category": "Categoría de bebida principal"
}

perfil_categorico = []

for col in perfil_vars:
    categoria_principal = mercado_meta[col].mode()[0]
    porcentaje_categoria = mercado_meta[col].value_counts(normalize=True).iloc[0] * 100

    perfil_categorico.append({
        "Variable": nombres_variables[col],
        "Categoría principal": categoria_principal,
        "Porcentaje del mercado meta": round(porcentaje_categoria, 2)
    })

perfil_categorico = pd.DataFrame(perfil_categorico)

st.dataframe(perfil_categorico, use_container_width=True)

def obtener_top(variable):
    resultado = perfil_categorico.loc[
        perfil_categorico["Variable"] == variable,
        "Categoría principal"
    ]
    return resultado.values[0] if len(resultado) > 0 else "No disponible"

top_region = obtener_top("Región principal")
top_store_type = obtener_top("Tipo de tienda principal")
top_channel = obtener_top("Canal principal")
top_age_group = obtener_top("Grupo etario principal")
top_gender = obtener_top("Género principal")
top_rewards = obtener_top("Miembro Rewards")
top_drink = obtener_top("Categoría de bebida principal")
top_time_slot = obtener_top("Horario principal")

st.subheader("Distribución del mercado meta")

col1, col2 = st.columns(2)

with col1:
    mercado_meta_region = mercado_meta["main_region"].value_counts().reset_index()
    mercado_meta_region.columns = ["Región", "Clientes"]

    fig_region_meta = px.bar(
        mercado_meta_region,
        x="Región",
        y="Clientes",
        text="Clientes",
        title="Mercado meta por región"
    )
    st.plotly_chart(fig_region_meta, use_container_width=True)

with col2:
    mercado_meta_canal = mercado_meta["main_order_channel"].value_counts().reset_index()
    mercado_meta_canal.columns = ["Canal", "Clientes"]

    fig_canal_meta = px.bar(
        mercado_meta_canal,
        x="Canal",
        y="Clientes",
        text="Clientes",
        title="Mercado meta por canal"
    )
    st.plotly_chart(fig_canal_meta, use_container_width=True)

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

ranking_expansion = mercado_meta_region_tienda.copy()

ranking_expansion["score_clientes"] = normalizar_columna(ranking_expansion["clientes"])
ranking_expansion["score_monetary"] = normalizar_columna(ranking_expansion["monetary_promedio"])
ranking_expansion["score_ticket"] = normalizar_columna(ranking_expansion["ticket_promedio"])
ranking_expansion["score_satisfaccion"] = normalizar_columna(ranking_expansion["satisfaccion_promedio"])

ranking_expansion["puntaje_oportunidad"] = (
    ranking_expansion["score_clientes"] * 0.50 +
    ranking_expansion["score_monetary"] * 0.20 +
    ranking_expansion["score_ticket"] * 0.15 +
    ranking_expansion["score_satisfaccion"] * 0.15
)

ranking_expansion["ranking_oportunidad"] = ranking_expansion["puntaje_oportunidad"].rank(
    method="dense",
    ascending=False
).astype(int)

ranking_expansion = ranking_expansion.sort_values("ranking_oportunidad")

ranking_expansion = ranking_expansion.rename(columns={
    "ranking_oportunidad": "Ranking",
    "main_region": "Región",
    "main_store_location_type": "Tipo de tienda",
    "clientes": "Clientes objetivo",
    "Porcentaje_clientes": "% del mercado meta",
    "ticket_promedio": "Ticket promedio",
    "monetary_promedio": "Gasto acumulado promedio",
    "satisfaccion_promedio": "Satisfacción promedio",
    "puntaje_oportunidad": "Puntaje de oportunidad"
})

ranking_expansion = ranking_expansion[
    [
        "Ranking",
        "Región",
        "Tipo de tienda",
        "Clientes objetivo",
        "% del mercado meta",
        "Ticket promedio",
        "Gasto acumulado promedio",
        "Satisfacción promedio",
        "Puntaje de oportunidad"
    ]
].round(3)

st.dataframe(ranking_expansion, use_container_width=True)

top_oportunidad = ranking_expansion.iloc[0]

# Combinación con mayor cantidad de clientes, aunque no necesariamente mayor puntaje ponderado
mayor_clientes = ranking_expansion.sort_values("Clientes objetivo", ascending=False).iloc[0]

st.info(
    f"Según el puntaje ponderado, la principal oportunidad es "
    f"{top_oportunidad['Región']} + {top_oportunidad['Tipo de tienda']}. "
    f"Por cantidad de clientes objetivo, destaca {mayor_clientes['Región']} + {mayor_clientes['Tipo de tienda']}."
)

st.subheader("Resumen estratégico")

resumen_estrategico = pd.DataFrame({
    "Elemento estratégico": [
        "Mercado meta principal",
        "Perfil dominante",
        "Región prioritaria",
        "Tipo de tienda prioritario",
        "Canal prioritario",
        "Grupo etario principal",
        "Género predominante",
        "Relación con Rewards",
        "Producto principal",
        "Horario de mayor consumo"
    ],
    "Resultado del análisis": [
        f"Clientes de alto valor RFM + {nombre_socio_objetivo}",
        "Consumidores de alto valor con mayor personalización y consumo complementario",
        top_region,
        top_store_type,
        top_channel,
        top_age_group,
        top_gender,
        top_rewards,
        top_drink,
        top_time_slot
    ]
})

st.dataframe(resumen_estrategico, use_container_width=True)

st.subheader("Recomendaciones principales")

recomendaciones_principales = pd.DataFrame({
    "Recomendación": [
        "Priorizar oportunidades de expansión según el ranking ponderado.",
        "Orientar la estrategia hacia clientes de alto valor con mayor personalización y consumo complementario.",
        "Potenciar fidelización y aumento de ticket mediante campañas personalizadas, combos y beneficios segmentados."
    ],
    "Justificación": [
        "La decisión considera cantidad de clientes, gasto acumulado, ticket promedio y satisfacción, no solo volumen de órdenes.",
        f"El mercado meta corresponde a clientes de alto valor RFM pertenecientes al segmento: {nombre_socio_objetivo}.",
        "Estas acciones calzan con clientes frecuentes, rentables y con potencial de mayor consumo por pedido."
    ]
})

st.dataframe(recomendaciones_principales, use_container_width=True)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0B3D2E, #1B5E3C);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-top: 20px;
">
    <h3 style="color:white; margin-top:0;">Conclusión final</h3>
    <p style="color:#F1F1F1; font-size:18px;">
        La expansión no debería enfocarse solo donde existen más órdenes, sino donde se concentran
        clientes de alto valor, con mayor personalización, consumo complementario y potencial de fidelización.
    </p>
</div>
""", unsafe_allow_html=True)
