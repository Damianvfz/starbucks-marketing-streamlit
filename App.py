
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from stepmix.stepmix import StepMix

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
    # SEGMENTACIÓN RFM CON K-MEANS
    # ========================================================

    rfm_vars = ["recency", "frequency", "monetary"]
    X_rfm = customer_features[rfm_vars]

    scaler_rfm = StandardScaler()
    X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

    # Evaluación de K para RFM con K-Means
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

    # Modelo RFM final con K-Means
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
    # SEGMENTACIÓN SOCIO-DEMOGRÁFICA / CONDUCTUAL CON K-MEANS
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
    # COMPARACIÓN METODOLÓGICA CON STEPMIX
    # ========================================================

    def calcular_entropia_relativa(probabilidades):
        """
        Calcula una medida de entropía relativa a partir de las probabilidades
        de pertenencia a cada clase. Valores más cercanos a 1 indican una
        clasificación más clara.
        """
        eps = 1e-12
        probs = np.clip(probabilidades, eps, 1)
        n = probs.shape[0]
        k = probs.shape[1]

        entropia = -np.sum(probs * np.log(probs))
        entropia_relativa = 1 - (entropia / (n * np.log(k)))

        return entropia_relativa

    # StepMix para RFM
    resultados_stepmix_rfm = []

    for k in range(2, 7):
        modelo_stepmix_rfm = StepMix(
            n_components=k,
            measurement="continuous",
            random_state=42,
            n_init=5,
            max_iter=500
        )

        modelo_stepmix_rfm.fit(X_rfm_scaled)

        probabilidades_rfm = modelo_stepmix_rfm.predict_proba(X_rfm_scaled)
        entropia_rfm = calcular_entropia_relativa(probabilidades_rfm)

        resultados_stepmix_rfm.append({
            "Número de clases": k,
            "BIC": modelo_stepmix_rfm.bic(X_rfm_scaled),
            "Entropía relativa": entropia_rfm
        })

    evaluacion_stepmix_rfm = pd.DataFrame(resultados_stepmix_rfm).round(4)

    # StepMix para socio-demográfico/conductual
    resultados_stepmix_socio = []

    for k in range(2, 7):
        modelo_stepmix_socio = StepMix(
            n_components=k,
            measurement="continuous",
            random_state=42,
            n_init=5,
            max_iter=500
        )

        modelo_stepmix_socio.fit(X_socio_scaled)

        probabilidades_socio = modelo_stepmix_socio.predict_proba(X_socio_scaled)
        entropia_socio = calcular_entropia_relativa(probabilidades_socio)

        resultados_stepmix_socio.append({
            "Número de clases": k,
            "BIC": modelo_stepmix_socio.bic(X_socio_scaled),
            "Entropía relativa": entropia_socio
        })

    evaluacion_stepmix_socio = pd.DataFrame(resultados_stepmix_socio).round(4)

    # Comparación metodológica resumida según la decisión del notebook final
    comparacion_metodos = pd.DataFrame({
        "Análisis": [
            "RFM",
            "Socio-demográfico/conductual"
        ],
        "Método K-Means": [
            "K-Means con 3 segmentos",
            "K-Means con 2 segmentos"
        ],
        "Método StepMix": [
            "StepMix con 5 clases",
            "StepMix con 6 clases"
        ],
        "Método elegido para análisis final": [
            "K-Means",
            "K-Means"
        ],
        "Justificación": [
            "K-Means permite separar clientes de alto valor, valor medio y baja actividad de forma clara.",
            "K-Means entrega dos perfiles accionables y más simples de interpretar comercialmente."
        ]
    })

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
        resumen_segmentos_combinados,
        evaluacion_stepmix_rfm,
        evaluacion_stepmix_socio,
        comparacion_metodos
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
    resumen_segmentos_combinados,
    evaluacion_stepmix_rfm,
    evaluacion_stepmix_socio,
    comparacion_metodos
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

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0B3D2E, #12372A);
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.12);
    ">
        <h3 style="color:#D8F3DC; font-weight:400; margin-top:0;">
            Segmentación de clientes para expansión de franquicias
        </h3>
        <p style="color:#F1F1F1; font-size:18px; max-width:900px;">
            Este análisis busca identificar mercados meta de alto valor para orientar decisiones de expansión,
            posicionamiento y captación de clientes en el mercado Starbucks America.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Órdenes", f"{df['order_id'].nunique():,}")
    col2.metric("Clientes únicos", f"{df['customer_id'].nunique():,}")
    col3.metric("Tiendas", f"{df['store_id'].nunique():,}")
    col4.metric(
        "Periodo",
        f"{df['order_date'].min().year}–{df['order_date'].max().year}"
    )

    st.divider()

    st.subheader("Objetivo del análisis")

    st.markdown("""
    El objetivo es identificar segmentos de clientes relevantes para un inversionista interesado en evaluar
    la apertura de nuevas franquicias de Starbucks en Estados Unidos.

    La pregunta central es:

    **¿Qué tipos de clientes son más atractivos para Starbucks y dónde conviene enfocar una estrategia de expansión?**
    """)

    st.info(
        "La estrategia combina una segmentación RFM, enfocada en valor comercial, "
        "con una segmentación socio-demográfica/conductual para identificar mercados meta accionables."
    )

    st.subheader("Ruta del análisis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1. Diagnóstico de datos**  
        Revisión de calidad, duplicados, valores nulos y periodo de análisis.
        """)

    with col2:
        st.markdown("""
        **2. Segmentación**  
        Aplicación de modelos RFM y socio-demográfico/conductual.
        """)

    with col3:
        st.markdown("""
        **3. Recomendación**  
        Definición de mercado meta, ubicación prioritaria y posicionamiento.
        """)

# ============================================================
# SECCIÓN 2: DIAGNÓSTICO DE DATOS
# ============================================================

elif seccion == "2. Diagnóstico de datos":
    st.header("2. Diagnóstico de datos")

    st.markdown("""
    Antes de construir los modelos de segmentación, se revisó la estructura general de la base,
    la presencia de valores nulos, duplicados y el periodo cubierto por las transacciones.
    """)

    st.success(
        "La base se encuentra limpia y es adecuada para el análisis. "
        "No presenta valores nulos ni órdenes duplicadas, por lo que no fue necesario eliminar registros "
        "antes de construir las segmentaciones."
    )

    st.divider()

    # ========================================================
    # MÉTRICAS PRINCIPALES DE CALIDAD Y COBERTURA
    # ========================================================

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

    st.divider()

    # ========================================================
    # RESUMEN DE VARIABLES NUMÉRICAS
    # ========================================================

    st.subheader("Resumen de variables principales")

    variables_numericas = [
        "cart_size",
        "num_customizations",
        "total_spend",
        "fulfillment_time_min",
        "customer_satisfaction"
    ]

    resumen_numerico = df[variables_numericas].describe().round(2)

    resumen_numerico = resumen_numerico.rename(columns={
        "cart_size": "Tamaño carrito",
        "num_customizations": "Personalizaciones",
        "total_spend": "Gasto total",
        "fulfillment_time_min": "Tiempo preparación",
        "customer_satisfaction": "Satisfacción"
    })

    st.dataframe(resumen_numerico, use_container_width=True)

    st.info(
        "Esta revisión permite identificar rangos razonables en gasto, tamaño de pedido, tiempo de preparación "
        "y satisfacción. Estas variables luego ayudan a caracterizar los segmentos de clientes."
    )

    st.divider()

    # ========================================================
    # VISTA DE LA BASE
    # ========================================================

    with st.expander("Ver primeras filas de la base"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Ver valores nulos por columna"):
        nulos = df.isnull().sum().reset_index()
        nulos.columns = ["Variable", "Valores nulos"]
        st.dataframe(nulos, use_container_width=True)

# ============================================================
# SECCIÓN 3: ANÁLISIS DESCRIPTIVO
# ============================================================

elif seccion == "3. Análisis descriptivo":
    st.header("3. Análisis descriptivo del mercado")

    st.markdown("""
    En esta etapa se analizan patrones generales de consumo antes de construir los segmentos.
    El objetivo es identificar qué canales, regiones y tipos de tienda concentran mayor actividad comercial.
    """)

    st.success(
        "El canal Mobile App aparece como el principal motor del mercado, "
        "ya que concentra más órdenes, mayor gasto total, mayor gasto promedio y una satisfacción superior al resto."
    )

    st.divider()

    # ========================================================
    # INDICADORES GENERALES DEL MERCADO
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gasto total", f"US$ {df['total_spend'].sum():,.0f}")
    col2.metric("Gasto promedio", f"US$ {df['total_spend'].mean():.2f}")
    col3.metric("Satisfacción promedio", f"{df['customer_satisfaction'].mean():.2f}/5")
    col4.metric("Tiempo promedio", f"{df['fulfillment_time_min'].mean():.2f} min")

    st.divider()

    # ========================================================
    # ANÁLISIS POR CANAL
    # ========================================================

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

    st.divider()

    # ========================================================
    # ANÁLISIS GEOGRÁFICO Y FORMATO DE TIENDA
    # ========================================================

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
        fig_region.update_layout(
            xaxis_title="Región",
            yaxis_title="Órdenes"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        fig_tienda = px.bar(
            analisis_tienda,
            x="store_location_type",
            y="ordenes",
            text="ordenes",
            title="Órdenes por tipo de tienda"
        )
        fig_tienda.update_layout(
            xaxis_title="Tipo de tienda",
            yaxis_title="Órdenes"
        )
        st.plotly_chart(fig_tienda, use_container_width=True)

    with st.expander("Ver tabla por región"):
        tabla_region = analisis_region.rename(columns={
            "region": "Región",
            "ordenes": "Órdenes",
            "clientes_unicos": "Clientes únicos",
            "gasto_total": "Gasto total",
            "gasto_promedio": "Gasto promedio",
            "satisfaccion_promedio": "Satisfacción promedio"
        })
        st.dataframe(tabla_region, use_container_width=True)

    with st.expander("Ver tabla por tipo de tienda"):
        tabla_tienda = analisis_tienda.rename(columns={
            "store_location_type": "Tipo de tienda",
            "ordenes": "Órdenes",
            "clientes_unicos": "Clientes únicos",
            "gasto_total": "Gasto total",
            "gasto_promedio": "Gasto promedio",
            "satisfaccion_promedio": "Satisfacción promedio"
        })
        st.dataframe(tabla_tienda, use_container_width=True)

    st.info(
        "La región West y las tiendas Suburban concentran mayor actividad total. "
        "Sin embargo, las diferencias de gasto promedio entre regiones y tipos de tienda son moderadas, "
        "por lo que la decisión final debe complementarse con la segmentación de clientes."
    )

    st.divider()

    # ========================================================
    # PERFIL ETARIO Y CATEGORÍA DE BEBIDA
    # ========================================================

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

    st.info(
        "Los grupos etarios de 25–34 y 35–44 años concentran una parte importante de la actividad. "
        "En bebidas, la distribución es relativamente equilibrada, por lo que la categoría de producto por sí sola "
        "no parece explicar completamente las diferencias del mercado."
    )
# ============================================================
# SECCIÓN 4: SEGMENTACIÓN
# ============================================================

elif seccion == "4. Segmentación":
    st.header("4. Segmentación de clientes")

    st.markdown("""
    La segmentación se realiza a nivel de **cliente**, no a nivel de orden. 
    Para esto, primero se agregaron las transacciones por `customer_id` y luego se aplicaron dos modelos complementarios.
    """)

    st.success(
        "El modelo RFM identifica el valor comercial del cliente, "
        "mientras que el modelo socio-demográfico/conductual permite entender su perfil de consumo. "
        "El mercado meta final se define cruzando ambos enfoques."
    )

    st.divider()

    # ========================================================
    # BASE A NIVEL CLIENTE
    # ========================================================

    st.subheader("Base construida a nivel cliente")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Clientes segmentados", f"{clientes.shape[0]:,}")
    col2.metric("Frecuencia promedio", f"{clientes['frequency'].mean():.2f}")
    col3.metric("Gasto acumulado promedio", f"US$ {clientes['monetary'].mean():.2f}")
    col4.metric("Ticket promedio", f"US$ {clientes['avg_ticket'].mean():.2f}")

    st.info(
        "Cada fila de esta base representa un cliente. Para cada uno se calcularon variables como recencia, "
        "frecuencia, gasto acumulado, ticket promedio, satisfacción, canal principal, región principal y tipo de tienda principal."
    )

    st.divider()

    # ========================================================
    # MODELO 1: RFM
    # ========================================================

    st.subheader("Modelo 1: Segmentación RFM")

    st.markdown("""
    El modelo **RFM** clasifica clientes según tres dimensiones:

    - **Recency:** qué tan reciente fue su última compra.
    - **Frequency:** cuántas compras realizó.
    - **Monetary:** cuánto gastó en total.

    Este modelo permite identificar clientes de mayor valor comercial.
    """)

    with st.expander("Ver evaluación del número de segmentos RFM"):
        st.markdown("""
        Para definir el número de segmentos se revisaron dos criterios:

        1. **Criterio estadístico:** método del codo y Silhouette Score en K-Means.  
        2. **Criterio comercial:** utilidad de los segmentos para explicar el mercado y proponer acciones.
        """)

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

        mejor_k = int(evaluacion_k.loc[evaluacion_k["Silhouette"].idxmax(), "K"])
        mejor_score = evaluacion_k["Silhouette"].max()

        st.dataframe(evaluacion_k, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("K sugerido por Silhouette", mejor_k)
            st.caption(f"Silhouette máximo: {mejor_score:.4f}")

        with col2:
            st.metric("K sugerido por codo", "≈ 5")
            st.caption("Interpretación visual del método del codo.")

        with col3:
            st.metric("K utilizado finalmente", 3)
            st.caption("Seleccionado por interpretación comercial.")

        st.info(
            "Aunque los criterios estadísticos permiten considerar diferentes valores de K, "
            "se utiliza K=3 en RFM porque permite explicar el mercado de forma más clara: "
            "clientes de alto valor, valor medio y baja actividad. Esta decisión equilibra "
            "desempeño estadístico e interpretabilidad estratégica."
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

    st.info(
        "El segmento de alto valor concentra clientes con mayor frecuencia y mayor gasto acumulado. "
        "Este grupo es relevante porque representa a los consumidores más atractivos desde el punto de vista comercial."
    )

    st.divider()

    # ========================================================
    # MODELO 2: SOCIO-DEMOGRÁFICO / CONDUCTUAL
    # ========================================================

    st.subheader("Modelo 2: Segmentación socio-demográfica/conductual")

    st.markdown("""
    El segundo modelo complementa la segmentación RFM incorporando variables de perfil y comportamiento,
    como región, tipo de tienda, canal principal, grupo etario, género, Rewards, horario, ticket promedio,
    personalización, satisfacción, pedidos anticipados y compra de comida.
    """)

    st.caption(
        "Nota: esta segmentación combina variables socio-demográficas con variables conductuales para obtener perfiles más accionables comercialmente."
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
        "nombre_segmento_socio": "Segmento socio-conductual",
        "clientes": "Clientes",
        "ticket_promedio": "Ticket promedio",
        "carrito_promedio": "Carrito promedio",
        "personalizaciones_promedio": "Personalizaciones promedio",
        "satisfaccion_promedio": "Satisfacción promedio",
        "tasa_pedido_anticipado": "Tasa pedido anticipado",
        "tasa_pedido_con_comida": "Tasa pedido con comida"
    })

    st.dataframe(tabla_socio, use_container_width=True)

    fig_socio = px.bar(
        tabla_socio,
        x="Segmento socio-conductual",
        y="Clientes",
        text="Clientes",
        title="Clientes por segmento socio-demográfico/conductual"
    )
    st.plotly_chart(fig_socio, use_container_width=True)

    st.info(
        "El segmento digital de consumo ampliado permite identificar clientes con mayor afinidad hacia canales digitales, "
        "mayor consumo por pedido y mayor potencial para estrategias de fidelización."
    )

    st.divider()

    # ========================================================
    # COMPARACIÓN K-MEANS VS STEPMIX
    # ========================================================

    st.subheader("Comparación metodológica: K-Means vs StepMix")

    st.markdown("""
    Además de K-Means, se estimó **StepMix** como método alternativo de contraste. 
    StepMix identifica clases latentes mediante un enfoque probabilístico, por lo que puede generar una segmentación más granular.
    """)

    st.dataframe(comparacion_metodos, use_container_width=True)

    with st.expander("Ver evaluación StepMix RFM"):
        st.dataframe(evaluacion_stepmix_rfm, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_stepmix_rfm_bic = px.line(
                evaluacion_stepmix_rfm,
                x="Número de clases",
                y="BIC",
                markers=True,
                title="StepMix RFM: BIC por número de clases"
            )
            st.plotly_chart(fig_stepmix_rfm_bic, use_container_width=True)

        with col2:
            fig_stepmix_rfm_entropia = px.line(
                evaluacion_stepmix_rfm,
                x="Número de clases",
                y="Entropía relativa",
                markers=True,
                title="StepMix RFM: Entropía relativa"
            )
            st.plotly_chart(fig_stepmix_rfm_entropia, use_container_width=True)

    with st.expander("Ver evaluación StepMix socio-demográfico/conductual"):
        st.dataframe(evaluacion_stepmix_socio, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_stepmix_socio_bic = px.line(
                evaluacion_stepmix_socio,
                x="Número de clases",
                y="BIC",
                markers=True,
                title="StepMix socio-conductual: BIC por número de clases"
            )
            st.plotly_chart(fig_stepmix_socio_bic, use_container_width=True)

        with col2:
            fig_stepmix_socio_entropia = px.line(
                evaluacion_stepmix_socio,
                x="Número de clases",
                y="Entropía relativa",
                markers=True,
                title="StepMix socio-conductual: Entropía relativa"
            )
            st.plotly_chart(fig_stepmix_socio_entropia, use_container_width=True)

    st.info(
        "StepMix entrega una segmentación más granular: 5 clases para RFM y 6 clases para el modelo socio-demográfico/conductual. "
        "Sin embargo, el análisis final utiliza K-Means porque permite explicar segmentos más simples, accionables y coherentes "
        "con una presentación ejecutiva de marketing."
    )

    st.divider()

    # ========================================================
    # CRUCE DE MODELOS
    # ========================================================

    st.subheader("Cruce entre ambos modelos")

    st.markdown("""
    La decisión final no se basa solo en el valor comercial ni solo en el perfil del cliente.
    El mercado meta principal surge del cruce entre:

    **Clientes de alto valor RFM**  
    +  
    **Clientes digitales de consumo ampliado**
    """)

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

    st.success(
        "Finalmente el mercado meta se define como un grupo más específico que los clientes de alto valor, "
        "porque además incorpora el perfil digital de consumo ampliado. Esto permite una recomendación más precisa para expansión y posicionamiento."
    )


# ============================================================
# SECCIÓN 5: MERCADO META Y RECOMENDACIÓN
# ============================================================

elif seccion == "5. Mercado meta y recomendación":
    st.header("5. Mercado meta y recomendación")

    st.markdown("""
    La recomendación final se construye a partir del cruce entre el segmento de mayor valor comercial
    y el segmento con perfil digital de consumo ampliado.
    """)

    st.success(
        "El mercado meta principal corresponde a clientes frecuentes, rentables, recientes "
        "y con mayor afinidad digital. Por eso, la estrategia debe priorizar ubicaciones donde se concentre este grupo "
        "y reforzar canales como Mobile App, Rewards y pedidos anticipados."
    )

    st.divider()

    # ========================================================
    # MERCADO META PRINCIPAL
    # ========================================================

    mercado_meta = clientes_segmentados[
        clientes_segmentados["mercado_meta_principal"] == "Mercado meta principal"
    ].copy()

    porcentaje_mercado_meta = mercado_meta.shape[0] / clientes_segmentados.shape[0] * 100

    st.subheader("Mercado meta principal")

    st.markdown("""
    El mercado meta se define como:

    **Clientes de alto valor RFM**  
    +  
    **Clientes digitales de consumo ampliado**
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

    st.info(
        "Este grupo es más específico que el segmento de alto valor RFM, porque además incorpora comportamiento digital "
        "y mayor potencial de consumo ampliado."
    )

    st.divider()

    # ========================================================
    # PERFIL DOMINANTE DEL MERCADO META
    # ========================================================

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

    top_region = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Región principal",
        "Categoría principal"
    ].values[0]

    top_store_type = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Tipo de tienda principal",
        "Categoría principal"
    ].values[0]

    top_channel = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Canal principal",
        "Categoría principal"
    ].values[0]

    top_age_group = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Grupo etario principal",
        "Categoría principal"
    ].values[0]

    top_drink = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Categoría de bebida principal",
        "Categoría principal"
    ].values[0]

    top_time_slot = perfil_categorico.loc[
        perfil_categorico["Variable"] == "Horario principal",
        "Categoría principal"
    ].values[0]

    st.info(
        f"El perfil dominante del mercado meta se concentra principalmente en la región {top_region}, "
        f"en tiendas de tipo {top_store_type}, con uso relevante del canal {top_channel}."
    )

    st.divider()

    # ========================================================
    # GRÁFICOS DEL PERFIL DEL MERCADO META
    # ========================================================

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

    col1, col2 = st.columns(2)

    with col1:
        mercado_meta_tienda = mercado_meta["main_store_location_type"].value_counts().reset_index()
        mercado_meta_tienda.columns = ["Tipo de tienda", "Clientes"]

        fig_tienda_meta = px.bar(
            mercado_meta_tienda,
            x="Tipo de tienda",
            y="Clientes",
            text="Clientes",
            title="Mercado meta por tipo de tienda"
        )
        st.plotly_chart(fig_tienda_meta, use_container_width=True)

    with col2:
        mercado_meta_edad = mercado_meta["main_age_group"].value_counts().reset_index()
        mercado_meta_edad.columns = ["Grupo etario", "Clientes"]

        fig_edad_meta = px.bar(
            mercado_meta_edad,
            x="Grupo etario",
            y="Clientes",
            text="Clientes",
            title="Mercado meta por grupo etario"
        )
        st.plotly_chart(fig_edad_meta, use_container_width=True)

    st.divider()

    # ========================================================
    # RANKING DE OPORTUNIDADES DE EXPANSIÓN
    # ========================================================

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

    ranking_expansion = ranking_expansion.rename(columns={
        "ranking_oportunidad": "Ranking",
        "main_region": "Región",
        "main_store_location_type": "Tipo de tienda",
        "clientes": "Clientes objetivo",
        "Porcentaje_clientes": "% del mercado meta",
        "ticket_promedio": "Ticket promedio",
        "monetary_promedio": "Gasto acumulado promedio",
        "satisfaccion_promedio": "Satisfacción promedio"
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
            "Satisfacción promedio"
        ]
    ]

    st.dataframe(ranking_expansion, use_container_width=True)

    top_oportunidad = ranking_expansion.iloc[0]

    st.info(
        f"La combinación con mayor concentración de clientes objetivo es "
        f"{top_oportunidad['Región']} - {top_oportunidad['Tipo de tienda']}. "
        "Sin embargo, la recomendación debe interpretarse como una prioridad inicial, no como única alternativa, "
        "porque otras combinaciones cercanas también pueden ser atractivas."
    )

    st.divider()

    # ========================================================
    # RESUMEN ESTRATÉGICO
    # ========================================================

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
            "Clientes de alto valor RFM + clientes digitales de consumo ampliado",
            "Consumidores frecuentes, rentables y con afinidad digital",
            top_region,
            top_store_type,
            top_channel,
            top_age_group,
            top_drink,
            top_time_slot
        ]
    })

    st.dataframe(resumen_estrategico, use_container_width=True)

    st.divider()

    # ========================================================
    # RECOMENDACIONES DE POSICIONAMIENTO
    # ========================================================

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
            f"Evaluar expansión principalmente en zonas {top_store_type} de la región {top_region}, considerando alternativas cercanas del ranking.",
            f"Fortalecer el canal {top_channel}, especialmente para pedidos anticipados y experiencia digital.",
            "Posicionar Starbucks como una opción rápida, conveniente y personalizable para consumidores frecuentes.",
            "Diseñar promociones combinadas de bebida y comida para aumentar ticket y tamaño de carrito.",
            f"Utilizar {top_drink} como categoría base para campañas, sin descuidar la variedad del portafolio.",
            f"Optimizar la operación durante el horario de {top_time_slot}, donde se concentra mayor demanda del mercado meta.",
            "Reforzar el programa Rewards para aumentar recurrencia, retención y valor de vida del cliente."
        ]
    })

    st.dataframe(recomendaciones_posicionamiento, use_container_width=True)

    st.divider()

    # ========================================================
    # CIERRE FINAL
    # ========================================================

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
            clientes de alto valor: frecuentes, rentables, digitales y fidelizables.
        </p>
    </div>
    """, unsafe_allow_html=True)