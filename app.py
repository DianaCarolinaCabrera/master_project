"""
Streamlit web UI — Agente Recomendador de Arquitecturas ML en AWS.

Ejecutar localmente:
  streamlit run app.py

Desplegar en Streamlit Cloud:
  1. git push al repositorio GitHub
  2. share.streamlit.io → New app → seleccionar repo → Main file: app.py
  3. Deploy  →  URL https automática
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
import streamlit as st

from ml_arch_recommender.scoring.engine import (
    CaseInput,
    RecommendationResult,
    ServiceAnalysisResult,
    recommend,
    analyze_services_for_arch,
    get_relevant_services_for_arch,
    _ARCH_MIN_COST_USD,
    _build_configuracion,
)
from ml_arch_recommender.demo import CASOS, IMPL_GUIDES

# ─────────────────────────────────────────────
# PAGE CONFIG  (debe ir antes de cualquier st.*)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ML Arch Recommender · AWS",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PALETA — cohesiva y contrastante
# ─────────────────────────────────────────────

C = {
    "bg":         "#0B0F1A",   # fondo principal — azul-negro espacial
    "surface":    "#131929",   # superficie elevada
    "card":       "#1C2438",   # cards y paneles
    "border":     "#2A3450",   # bordes suaves
    "primary":    "#FF9900",   # AWS orange — CTAs
    "emerald":    "#10B981",   # éxito / viable
    "amber":      "#F59E0B",   # advertencia / sustitución
    "red":        "#EF4444",   # error / no viable
    "blue":       "#3B82F6",   # info
    "text":       "#F1F5F9",   # texto principal
    "muted":      "#94A3B8",   # texto secundario
    "dim":        "#475569",   # texto atenuado
}

ARCH_COLORS: dict[str, str] = {
    "serverless": "#10B981",   # esmeralda
    "batch":      "#3B82F6",   # zafiro
    "streaming":  "#A855F7",   # violeta
    "containers": "#F59E0B",   # ámbar dorado
    "sagemaker":  "#06B6D4",   # cian eléctrico
}

ARCH_GLOW: dict[str, str] = {
    "serverless": "rgba(16,185,129,0.18)",
    "batch":      "rgba(59,130,246,0.18)",
    "streaming":  "rgba(168,85,247,0.18)",
    "containers": "rgba(245,158,11,0.18)",
    "sagemaker":  "rgba(6,182,212,0.18)",
}

ARCH_ICONS: dict[str, str] = {
    "serverless": "⚡",
    "batch":      "📦",
    "streaming":  "🌊",
    "containers": "🐳",
    "sagemaker":  "🧠",
}

ALL_SERVICES = [
    "AWS Lambda", "Amazon API Gateway", "Amazon S3", "AWS Glue",
    "Amazon ECS", "AWS Fargate", "Amazon Kinesis", "Amazon DynamoDB",
    "Amazon SageMaker AI", "Amazon EventBridge", "AWS Step Functions",
    "Amazon CloudWatch", "Amazon RDS", "Amazon EMR", "AWS App Runner",
    "Amazon EC2", "Amazon CloudFront", "Amazon MSK", "Amazon Redshift",
]

AWS_SERVICE_INFO: dict[str, dict] = {
    "AWS Lambda": {
        "icon": "λ",
        "categoria": "Compute",
        "descripcion": (
            "Función serverless que ejecuta código bajo demanda sin gestionar servidores. "
            "Escala automáticamente de 0 a miles de ejecuciones en paralelo. "
            "Factura por invocación + tiempo de ejecución (ms), con 1 M de invocaciones gratuitas al mes."
        ),
        "uso_ml": "Inferencia puntual en tiempo real, latencia < 1 s, bajo volumen o tráfico muy variable.",
    },
    "Amazon API Gateway": {
        "icon": "🔗",
        "categoria": "Networking / API",
        "descripcion": (
            "Puerta de entrada HTTP/REST o WebSocket para exponer funciones Lambda o microservicios. "
            "Gestiona autenticación (IAM, Cognito, JWT), throttling, CORS y caché de respuestas."
        ),
        "uso_ml": "Endpoint público del modelo: recibe la solicitud del cliente y la reenvía a Lambda o ECS.",
    },
    "Amazon S3": {
        "icon": "🪣",
        "categoria": "Storage",
        "descripcion": (
            "Almacenamiento de objetos de alta durabilidad (99,999999999 %). "
            "Sin límite de capacidad, facturación por GB almacenado y transferencia. "
            "Soporta versionado, lifecycle policies y cifrado en reposo."
        ),
        "uso_ml": "Persistir artefactos del modelo (pickle, joblib, .pt), datasets de entrada/salida de batch y logs.",
    },
    "AWS Glue": {
        "icon": "🔧",
        "categoria": "ETL / Data Integration",
        "descripcion": (
            "Servicio ETL serverless basado en Apache Spark. "
            "Ejecuta jobs de transformación y scoring masivo sobre datos en S3 o bases de datos, "
            "sin necesidad de provisionar ni administrar clústeres."
        ),
        "uso_ml": "Scoring batch de millones de registros, reprocesos nocturnos, pipelines de preprocesado.",
    },
    "Amazon ECS": {
        "icon": "📦",
        "categoria": "Containers",
        "descripcion": (
            "Orquestador de contenedores Docker totalmente gestionado. "
            "Define tasks con CPU, RAM, variables de entorno y health checks precisos. "
            "Se integra con ALB para balanceo de carga y con ECR para el registro de imágenes."
        ),
        "uso_ml": "APIs de inferencia persistentes con estado, deploys blue/green, control total del entorno.",
    },
    "AWS Fargate": {
        "icon": "🚀",
        "categoria": "Serverless Containers",
        "descripcion": (
            "Motor de cómputo serverless para ECS y EKS: elimina la necesidad de gestionar instancias EC2. "
            "Solo se paga por los recursos (vCPU + GB RAM) consumidos mientras el contenedor corre."
        ),
        "uso_ml": "Reducir la carga operativa de ECS; escalar contenedores sin administrar nodos del clúster.",
    },
    "Amazon Kinesis": {
        "icon": "🌊",
        "categoria": "Streaming",
        "descripcion": (
            "Plataforma de ingesta y procesamiento de datos en tiempo real. "
            "Kinesis Data Streams captura millones de eventos/s con retención configurable (1–365 días). "
            "Kinesis Data Analytics permite SQL o Apache Flink directamente sobre el stream."
        ),
        "uso_ml": "Detección de fraude transaccional, scoring de eventos en tiempo real, monitoreo continuo.",
    },
    "Amazon DynamoDB": {
        "icon": "⚡",
        "categoria": "Database NoSQL",
        "descripcion": (
            "Base de datos key-value/document con latencia de milisegundo de un solo dígito a cualquier escala. "
            "Modo on-demand: sin provisionar capacidad. "
            "Cifrado en reposo, backups point-in-time y replicación global."
        ),
        "uso_ml": "Caché de predicciones recientes, feature store en tiempo real, registro auditable de solicitudes.",
    },
    "Amazon SageMaker AI": {
        "icon": "🧠",
        "categoria": "ML Platform",
        "descripcion": (
            "Plataforma MLOps completa: entrenamiento distribuido, tuning de hiperparámetros, "
            "hosting de endpoints gestionados, monitoreo de data drift y model quality, "
            "pipelines CI/CD para modelos y A/B testing nativo entre variantes."
        ),
        "uso_ml": "Modelos complejos que requieren ciclo de vida gestionado, múltiples versiones en producción simultánea.",
    },
    "Amazon EventBridge": {
        "icon": "🔔",
        "categoria": "Event Bus",
        "descripcion": (
            "Bus de eventos serverless para desacoplar productores y consumidores. "
            "Soporta reglas basadas en cron (horario fijo) o patrones de eventos de más de 200 servicios AWS. "
            "Enruta eventos a Lambdas, Glue jobs, Step Functions u otros targets."
        ),
        "uso_ml": "Disparar el job batch a medianoche, orquestar pipelines de ML basados en eventos de S3.",
    },
    "AWS Step Functions": {
        "icon": "🔄",
        "categoria": "Orchestration",
        "descripcion": (
            "Orquestador de workflows visual (máquinas de estado) para secuenciar Lambdas, Glue jobs y tasks ECS. "
            "Maneja reintentos, paralelismo, bifurcaciones condicionales y compensación de errores de forma declarativa."
        ),
        "uso_ml": "Pipelines complejos: preprocesado → inferencia → validación → escritura de resultados.",
    },
    "Amazon CloudWatch": {
        "icon": "📊",
        "categoria": "Observability",
        "descripcion": (
            "Suite de observabilidad: métricas de infraestructura y aplicación, logs centralizados, "
            "alarmas automáticas y dashboards personalizables. "
            "Integrado con todos los servicios AWS sin configuración adicional."
        ),
        "uso_ml": "Alertas de latencia p99, logs de errores del modelo, dashboards de throughput y tasa de error.",
    },
    "Amazon RDS": {
        "icon": "🗄️",
        "categoria": "Database Relacional",
        "descripcion": (
            "Base de datos relacional gestionada: PostgreSQL, MySQL, MariaDB, Oracle o SQL Server. "
            "Backups automatizados, Multi-AZ para alta disponibilidad y read replicas para escalar lecturas."
        ),
        "uso_ml": "Histórico de scores con metadatos estructurados, auditoría de predicciones, feature store relacional.",
    },
    "Amazon EMR": {
        "icon": "💡",
        "categoria": "Big Data",
        "descripcion": (
            "Clúster Hadoop/Spark/Hive gestionado para procesamiento a gran escala. "
            "Más potente que Glue cuando se necesita lógica Spark muy personalizada "
            "o se procesan decenas de terabytes."
        ),
        "uso_ml": "Scoring batch de cientos de millones de registros, reentrenamiento distribuido.",
    },
    "AWS App Runner": {
        "icon": "🏃",
        "categoria": "Containers Simplificados",
        "descripcion": (
            "Despliegue fully-managed de aplicaciones containerizadas desde código fuente o imagen Docker, "
            "sin configurar load balancers, ECS tasks ni grupos de Auto Scaling. "
            "Escala automáticamente a cero cuando no hay tráfico."
        ),
        "uso_ml": "APIs de inferencia sencillas con mínimo overhead operativo; prototipado rápido.",
    },
    "Amazon EC2": {
        "icon": "🖥️",
        "categoria": "Compute",
        "descripcion": (
            "Máquinas virtuales en la nube con control total de SO, red y almacenamiento. "
            "Familias GPU (p3, p4, g4) para deep learning. "
            "Spot instances ofrecen hasta 90 % de descuento para cargas tolerantes a interrupciones."
        ),
        "uso_ml": "Inferencia con GPU, cargas con requisitos muy específicos de hardware o de configuración de SO.",
    },
    "Amazon CloudFront": {
        "icon": "🌐",
        "categoria": "CDN",
        "descripcion": (
            "Red de distribución de contenido con más de 450 puntos de presencia globales. "
            "Reduce latencia para usuarios geográficamente dispersos, "
            "cachea respuestas y protege contra DDoS mediante AWS Shield Standard."
        ),
        "uso_ml": "Caché de predicciones estáticas o de baja volatilidad, distribución global del endpoint.",
    },
    "Amazon MSK": {
        "icon": "📨",
        "categoria": "Streaming (Kafka)",
        "descripcion": (
            "Apache Kafka gestionado por AWS. "
            "Alternativa a Kinesis cuando el equipo ya usa Kafka o necesita compatibilidad "
            "con el ecosistema Kafka (Kafka Connect, Kafka Streams, Schema Registry)."
        ),
        "uso_ml": "Pipelines de eventos con infraestructura Kafka existente, integración con herramientas del ecosistema Kafka.",
    },
    "Amazon Redshift": {
        "icon": "🔴",
        "categoria": "Data Warehouse",
        "descripcion": (
            "Data warehouse columnar para analítica a escala de petabytes con SQL estándar. "
            "Redshift ML permite entrenar y ejecutar modelos directamente en el warehouse con CREATE MODEL."
        ),
        "uso_ml": "Scoring SQL-based sobre datos que ya residen en el DWH; evita mover datos fuera del warehouse.",
    },
}


# ─────────────────────────────────────────────
# CSS — diseño profesional nivel producto
# ─────────────────────────────────────────────

st.markdown(f"""
<style>
/* ════════════════════════════════════════════
   1. FUENTES + VARIABLES CSS
   ════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg:        {C['bg']};
    --surface:   {C['surface']};
    --card:      {C['card']};
    --border:    {C['border']};
    --primary:   {C['primary']};
    --emerald:   {C['emerald']};
    --amber:     {C['amber']};
    --red:       {C['red']};
    --blue:      {C['blue']};
    --text:      {C['text']};
    --muted:     {C['muted']};
    --dim:       {C['dim']};
    --r-sm:  8px;
    --r:    12px;
    --r-lg: 16px;
    --r-xl: 20px;
    --sh-sm: 0 2px 8px  rgba(0,0,0,0.28);
    --sh:    0 4px 20px rgba(0,0,0,0.38);
    --sh-lg: 0 8px 40px rgba(0,0,0,0.50);
    --sh-xl: 0 16px 60px rgba(0,0,0,0.60);
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}}

/* ════════════════════════════════════════════
   2. KEYFRAMES
   ════════════════════════════════════════════ */

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0);    }}
}}

@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}

@keyframes barScale {{
    from {{ transform: scaleX(0); }}
    to   {{ transform: scaleX(1); }}
}}

@keyframes shimmerSlide {{
    0%   {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(250%);  }}
}}

@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 0  0px rgba(255,153,0,0.25); }}
    50%       {{ box-shadow: 0 0 0 10px rgba(255,153,0,0.00); }}
}}

@keyframes spinRing {{
    from {{ stroke-dashoffset: 176; }}
    to   {{ stroke-dashoffset: 0;   }}
}}

@keyframes gradientShift {{
    0%   {{ background-position: 0%   50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0%   50%; }}
}}

/* ════════════════════════════════════════════
   3. BASE / RESET GLOBAL
   ════════════════════════════════════════════ */

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* Fondo del área principal con degradado sutil */
.stApp {{
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,153,0,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(6,182,212,0.04) 0%, transparent 60%),
        var(--bg);
}}

/* Área de contenido principal */
.main .block-container {{
    padding-top: 1.8rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1440px !important;
}}

/* Ocultar footer "Made with Streamlit" */
footer {{ visibility: hidden !important; height: 0 !important; }}

/* Header de Streamlit con glassmorphism */
header[data-testid="stHeader"] {{
    background: rgba(11, 15, 26, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(42, 52, 80, 0.6) !important;
}}

/* ════════════════════════════════════════════
   4. SIDEBAR
   ════════════════════════════════════════════ */

[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, #090E1A 0%, #0D1220 35%, #111827 100%) !important;
    border-right: 1px solid rgba(42, 52, 80, 0.8) !important;
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
}}

/* Logo — centrado con padding elegante */
[data-testid="stSidebar"] [data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
    padding: 22px 28px 14px !important;
}}
[data-testid="stSidebar"] [data-testid="stImage"] img {{
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
}}

/* Labels del formulario */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput  label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox  label {{
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

[data-testid="stSidebar"] .stMarkdown p {{
    font-size: 0.87rem;
    color: var(--muted);
    line-height: 1.6;
}}

/* ════════════════════════════════════════════
   5. INPUTS / SELECTBOX / TEXTAREA
   ════════════════════════════════════════════ */

.stTextInput  input,
.stNumberInput input {{
    background:    rgba(20, 28, 46, 0.9) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color:         var(--text) !important;
    font-size:     0.9rem !important;
    padding:       10px 14px !important;
    transition:    border-color 0.2s, box-shadow 0.2s !important;
}}
.stTextInput  input:focus,
.stNumberInput input:focus {{
    border-color: var(--primary) !important;
    box-shadow:   0 0 0 3px rgba(255,153,0,0.18) !important;
    outline:      none !important;
}}
/* Ocultar spinners numéricos feos */
.stNumberInput input::-webkit-inner-spin-button,
.stNumberInput input::-webkit-outer-spin-button {{ opacity: 0.3; }}

/* Selectbox */
.stSelectbox [data-baseweb="select"] > div:first-child {{
    background:    rgba(20, 28, 46, 0.9) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color:         var(--text) !important;
    transition:    border-color 0.2s !important;
}}
.stSelectbox [data-baseweb="select"]:focus-within > div:first-child {{
    border-color: var(--primary) !important;
    box-shadow:   0 0 0 3px rgba(255,153,0,0.18) !important;
}}

/* Menú desplegable */
[data-baseweb="popover"] [data-baseweb="menu"] {{
    background:    var(--card) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r) !important;
    box-shadow:    var(--sh-xl) !important;
    padding:       4px !important;
}}
[data-baseweb="popover"] [role="option"] {{
    color:         var(--text) !important;
    font-size:     0.9rem !important;
    border-radius: 6px !important;
    margin:        1px 4px !important;
    padding:       8px 12px !important;
    transition:    background 0.15s !important;
}}
[data-baseweb="popover"] [role="option"]:hover {{
    background: rgba(255,153,0,0.10) !important;
}}
[data-baseweb="popover"] [aria-selected="true"] {{
    background: rgba(255,153,0,0.16) !important;
    color:      var(--primary) !important;
    font-weight: 600 !important;
}}

/* Checkboxes */
.stCheckbox {{ margin: 4px 0 !important; }}
.stCheckbox label {{
    font-size:  0.87rem !important;
    color:      var(--muted) !important;
    cursor:     pointer;
    transition: color 0.15s;
    gap:        8px !important;
}}
.stCheckbox label:hover {{ color: var(--text) !important; }}
.stCheckbox [data-baseweb="checkbox"] {{
    border-radius: 5px !important;
    border-color:  var(--border) !important;
}}
.stCheckbox [data-baseweb="checkbox"][data-checked="true"] {{
    background:   var(--primary) !important;
    border-color: var(--primary) !important;
}}

/* ════════════════════════════════════════════
   6. TABS
   ════════════════════════════════════════════ */

.stTabs [data-baseweb="tab-list"] {{
    background:    var(--surface);
    border-radius: var(--r-lg);
    padding:       5px;
    gap:           3px;
    border:        1px solid var(--border);
    box-shadow:    inset 0 2px 6px rgba(0,0,0,0.35);
    margin-bottom: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: var(--r) !important;
    font-weight:   500 !important;
    font-size:     0.91rem !important;
    color:         var(--muted) !important;
    padding:       10px 24px !important;
    letter-spacing: 0.01em;
    transition:    all 0.2s ease !important;
    border:        none !important;
    outline:       none !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color:      var(--text) !important;
    background: rgba(255,255,255,0.04) !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #1E2D4A 0%, #1C2A45 100%) !important;
    color:      var(--text) !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.08),
                0 0 0 1px rgba(255,153,0,0.2) !important;
}}

/* ════════════════════════════════════════════
   7. EXPANDERS
   ════════════════════════════════════════════ */

[data-testid="stExpander"] {{
    background:    var(--card) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--r) !important;
    margin:        8px 0 !important;
    overflow:      hidden;
    transition:    border-color 0.2s, box-shadow 0.2s;
}}
[data-testid="stExpander"]:hover {{
    border-color: rgba(255,153,0,0.28) !important;
    box-shadow:   var(--sh-sm);
}}
[data-testid="stExpander"] > details > summary {{
    font-weight:   600 !important;
    font-size:     0.93rem !important;
    padding:       15px 20px !important;
    color:         var(--text) !important;
    cursor:        pointer;
    border-radius: var(--r) !important;
    transition:    background 0.15s;
    list-style:    none;
    display:       flex;
    align-items:   center;
}}
[data-testid="stExpander"] > details > summary::-webkit-details-marker,
[data-testid="stExpander"] > details > summary::marker {{ display: none; }}
[data-testid="stExpander"] > details > summary::after {{
    content: '›';
    margin-left: auto;
    font-size: 1.3em;
    color: var(--dim);
    transition: transform 0.2s;
    line-height: 1;
}}
[data-testid="stExpander"] > details[open] > summary::after {{
    transform: rotate(90deg);
    color: var(--primary);
}}
[data-testid="stExpander"] > details > summary:hover {{
    background: rgba(255,255,255,0.025);
}}
[data-testid="stExpander"] > details[open] > summary {{
    border-bottom: 1px solid var(--border);
}}
[data-testid="stExpander"] > details > div {{
    padding: 18px 20px !important;
    animation: fadeUp 0.22s ease;
}}

/* ════════════════════════════════════════════
   8. BOTONES
   ════════════════════════════════════════════ */

.stButton > button {{
    font-family:    'Inter', sans-serif !important;
    font-weight:    600 !important;
    letter-spacing: 0.02em;
    border-radius:  var(--r) !important;
    transition:     all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

/* CTA principal */
.stButton > button[kind="primary"] {{
    background:  linear-gradient(135deg, #FF9900 0%, #E07B00 100%) !important;
    border:      none !important;
    color:       #000 !important;
    font-weight: 800 !important;
    font-size:   0.93rem !important;
    padding:     11px 26px !important;
    box-shadow:  0 4px 18px rgba(255,153,0,0.40),
                 inset 0 1px 0 rgba(255,255,255,0.28) !important;
    animation:   pulseGlow 3s ease-in-out infinite;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow:  0 7px 26px rgba(255,153,0,0.60),
                 inset 0 1px 0 rgba(255,255,255,0.28) !important;
    transform:   translateY(-2px) !important;
}}
.stButton > button[kind="primary"]:active {{
    transform:   translateY(0) !important;
    box-shadow:  0 2px 10px rgba(255,153,0,0.35) !important;
}}

/* Secundarios */
.stButton > button:not([kind="primary"]) {{
    background: rgba(20, 28, 46, 0.95) !important;
    border:     1px solid var(--border) !important;
    color:      var(--text) !important;
    font-size:  0.86rem !important;
    padding:    8px 16px !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    background:   rgba(255,153,0,0.07) !important;
    border-color: rgba(255,153,0,0.45) !important;
    color:        var(--primary) !important;
}}

/* ════════════════════════════════════════════
   9. STREAMLIT ALERTS (st.info / warning / success / error)
   ════════════════════════════════════════════ */

[data-testid="stAlert"] {{
    border-radius: var(--r) !important;
    border-width:  1px !important;
    font-size:     0.9rem !important;
    animation:     fadeUp 0.3s ease;
    backdrop-filter: blur(6px) !important;
}}

/* ════════════════════════════════════════════
   10. CODE BLOCKS
   ════════════════════════════════════════════ */

.stCode, [data-testid="stCode"] {{
    border-radius: var(--r-sm) !important;
}}
.stCode code,
[data-testid="stCode"] pre,
[data-testid="stCode"] code {{
    font-family:   var(--font-mono) !important;
    font-size:     0.83rem !important;
    line-height:   1.65 !important;
    background:    #0D1117 !important;
    border:        1px solid #21262D !important;
    border-radius: var(--r-sm) !important;
}}

/* Botón "copy" del code block */
[data-testid="stCode"] button {{
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
}}
[data-testid="stCode"] button:hover {{
    color: var(--primary) !important;
}}

/* ════════════════════════════════════════════
   11. MÉTRICAS
   ════════════════════════════════════════════ */

[data-testid="metric-container"] {{
    background:    var(--card);
    border:        1px solid var(--border);
    border-radius: var(--r);
    padding:       18px 22px;
    transition:    border-color 0.2s, box-shadow 0.2s;
}}
[data-testid="metric-container"]:hover {{
    border-color: rgba(255,153,0,0.3);
    box-shadow:   var(--sh-sm);
}}
[data-testid="stMetricValue"] {{
    color:       var(--primary) !important;
    font-weight: 800 !important;
    font-family: var(--font-mono) !important;
}}
[data-testid="stMetricLabel"] {{
    color:          var(--muted) !important;
    font-size:      0.72rem !important;
    font-weight:    700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}

/* ════════════════════════════════════════════
   12. CAPTIONS
   ════════════════════════════════════════════ */

.stCaption, [data-testid="stCaption"] {{
    color:       var(--dim) !important;
    font-size:   0.77rem !important;
    line-height: 1.55;
}}

/* ════════════════════════════════════════════
   13. SCROLLBAR
   ════════════════════════════════════════════ */

::-webkit-scrollbar              {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track        {{ background: transparent; }}
::-webkit-scrollbar-thumb        {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover  {{ background: var(--muted); }}
::-webkit-scrollbar-corner       {{ background: transparent; }}

/* ════════════════════════════════════════════
   14. HR / SEPARADORES
   ════════════════════════════════════════════ */

hr {{
    border:     none !important;
    height:     1px !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        var(--border) 20%,
        var(--border) 80%,
        transparent 100%
    ) !important;
    margin: 22px 0 !important;
}}

/* ════════════════════════════════════════════
   15. CLASES DE UTILIDAD (componentes custom)
   ════════════════════════════════════════════ */

/* ── Glass card ── */
.glass-card {{
    background:             rgba(28, 36, 56, 0.72);
    backdrop-filter:        blur(14px);
    -webkit-backdrop-filter:blur(14px);
    border:                 1px solid rgba(255,255,255,0.07);
    border-radius:          var(--r-lg);
    padding:                20px 26px;
    margin:                 10px 0;
    box-shadow:             var(--sh), inset 0 1px 0 rgba(255,255,255,0.05);
    animation:              fadeUp 0.35s ease;
}}

/* ── Winner card ── */
.winner-card {{
    border-radius: var(--r-xl);
    padding:       26px 30px;
    margin:        12px 0;
    position:      relative;
    overflow:      hidden;
    animation:     fadeUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}}
/* Resplandor sutil en esquina superior derecha */
.winner-card::after {{
    content:         '';
    position:        absolute;
    top:    -30%;
    right:  -5%;
    width:  280px;
    height: 280px;
    background:      radial-gradient(circle, rgba(255,255,255,0.035) 0%, transparent 65%);
    pointer-events:  none;
}}

/* ── Score pill / badge ── */
.score-pill {{
    display:        inline-flex;
    align-items:    center;
    padding:        4px 12px;
    border-radius:  20px;
    font-size:      0.76em;
    font-weight:    700;
    letter-spacing: 0.04em;
    white-space:    nowrap;
    vertical-align: middle;
}}

/* ── Criterion bars ── */
.criterion-bar-wrap  {{ margin: 7px 0; }}
.criterion-bar-label {{
    display:         flex;
    justify-content: space-between;
    align-items:     center;
    font-size:       0.79em;
    margin-bottom:   4px;
}}
.criterion-bar-track {{
    background:    rgba(42, 52, 80, 0.9);
    border-radius: 6px;
    height:        6px;
    overflow:      hidden;
}}
.criterion-bar-fill {{
    height:           100%;
    border-radius:    6px;
    transform-origin: left center;
    transform:        scaleX(0);
    animation:        barScale 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    position:         relative;
    overflow:         hidden;
}}
/* Shimmer de brillo que pasa por la barra al entrar */
.criterion-bar-fill::after {{
    content:  '';
    position: absolute;
    top: 0; left: 0;
    width:      40%;
    height:     100%;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(255,255,255,0.35) 50%,
        transparent 100%
    );
    animation: shimmerSlide 0.9s ease 0.65s;
}}
/* Delays escalonados para las 5 barras */
.criterion-bar-wrap:nth-child(1) .criterion-bar-fill {{ animation-delay: 0.05s; }}
.criterion-bar-wrap:nth-child(2) .criterion-bar-fill {{ animation-delay: 0.15s; }}
.criterion-bar-wrap:nth-child(3) .criterion-bar-fill {{ animation-delay: 0.25s; }}
.criterion-bar-wrap:nth-child(4) .criterion-bar-fill {{ animation-delay: 0.35s; }}
.criterion-bar-wrap:nth-child(5) .criterion-bar-fill {{ animation-delay: 0.45s; }}

/* ── Avisos / strips ── */
.warning-strip {{
    border-left:   4px solid var(--red);
    background:    rgba(239,68,68,0.07);
    padding:       12px 18px;
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    margin:        8px 0;
    font-size:     0.89em;
    line-height:   1.55;
    animation:     fadeUp 0.3s ease;
}}
.tension-strip {{
    border-left:   4px solid var(--amber);
    background:    rgba(245,158,11,0.07);
    padding:       12px 18px;
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    margin:        8px 0;
    font-size:     0.89em;
    line-height:   1.55;
    animation:     fadeUp 0.3s ease;
}}

/* ── Banners de viabilidad ── */
.viable-banner {{
    background:    linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(6,182,212,0.06) 100%);
    border:        1px solid rgba(16,185,129,0.35);
    border-radius: var(--r);
    padding:       20px 24px;
    margin:        14px 0;
    animation:     fadeUp 0.35s ease;
}}
.not-viable-banner {{
    background:    linear-gradient(135deg, rgba(239,68,68,0.09) 0%, rgba(168,85,247,0.05) 100%);
    border:        1px solid rgba(239,68,68,0.35);
    border-radius: var(--r);
    padding:       20px 24px;
    margin:        14px 0;
    animation:     fadeUp 0.35s ease;
}}

/* ── Tarjeta de sustitución ── */
.sub-card {{
    background:    rgba(245,158,11,0.06);
    border-left:   3px solid var(--amber);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding:       13px 18px;
    margin:        8px 0;
    transition:    background 0.2s;
}}
.sub-card:hover {{ background: rgba(245,158,11,0.11); }}

/* ── Alternativa económica ── */
.alt-price-card {{
    background:    rgba(245,158,11,0.05);
    border:        1px solid rgba(245,158,11,0.28);
    border-radius: var(--r);
    padding:       20px 24px;
    margin:        14px 0;
    animation:     fadeUp 0.4s ease;
    transition:    box-shadow 0.2s;
}}
.alt-price-card:hover {{ box-shadow: var(--sh); }}

/* ── Pasos de despliegue ── */
.step-card {{
    background:    rgba(255,153,0,0.06);
    border-left:   3px solid var(--primary);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding:       13px 18px;
    margin:        8px 0;
    transition:    background 0.2s;
}}
.step-card:hover {{ background: rgba(255,153,0,0.10); }}

/* ── IAM / permisos ── */
.iam-card {{
    background:    rgba(59,130,246,0.06);
    border:        1px solid rgba(59,130,246,0.18);
    border-radius: var(--r-sm);
    padding:       15px 20px;
    margin:        8px 0;
    font-size:     0.86em;
    font-family:   var(--font-mono);
    white-space:   pre-line;
    line-height:   1.7;
}}
.role-card {{
    background:    rgba(6,182,212,0.05);
    border:        1px solid rgba(6,182,212,0.18);
    border-radius: var(--r-sm);
    padding:       15px 20px;
    margin:        8px 0;
}}

/* ── Buena práctica ── */
.tip-card {{
    background:    linear-gradient(135deg, rgba(16,185,129,0.09) 0%, rgba(6,182,212,0.05) 100%);
    border:        1px solid rgba(16,185,129,0.28);
    border-radius: var(--r);
    padding:       22px 26px;
    margin:        10px 0;
    animation:     fadeUp 0.3s ease;
}}

/* ── Service chips ── */
.service-chip {{
    display:        inline-block;
    padding:        4px 11px;
    border-radius:  20px;
    font-size:      0.76em;
    font-weight:    600;
    margin:         3px 3px;
    letter-spacing: 0.01em;
    transition:     opacity 0.2s, transform 0.2s;
}}
.service-chip:hover {{ opacity: 0.82; transform: translateY(-1px); }}

/* ── Landing — feature cards ── */
.hero-feature {{
    background:             rgba(28, 36, 56, 0.75);
    backdrop-filter:        blur(10px);
    -webkit-backdrop-filter:blur(10px);
    border:                 1px solid var(--border);
    border-radius:          var(--r-lg);
    padding:                26px 22px;
    height:                 100%;
    text-align:             center;
    transition:             all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
    animation:              fadeUp 0.4s ease;
}}
.hero-feature:hover {{
    transform:    translateY(-4px);
    border-color: rgba(255,153,0,0.35);
    box-shadow:   0 12px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,153,0,0.12);
}}
.hero-feature .icon {{
    font-size:     2.5em;
    margin-bottom: 14px;
    display:       block;
}}

/* ── Section labels con barra izquierda ── */
.section-label {{
    font-size:      0.70em;
    font-weight:    700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color:          var(--muted);
    margin-bottom:  10px;
    display:        flex;
    align-items:    center;
    gap:            8px;
}}
.section-label::before {{
    content:       '';
    display:       inline-block;
    width:         3px;
    height:        12px;
    background:    linear-gradient(180deg, var(--primary), rgba(255,153,0,0.4));
    border-radius: 2px;
    flex-shrink:   0;
}}

/* ── Divider ── */
.divider {{
    height:     1px;
    background: linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
    margin:     20px 0;
}}

/* ── Tipo mono inline ── */
.mono {{
    font-family: var(--font-mono);
    font-size:   0.88em;
}}

/* ════════════════════════════════════════════
   16. ANIMACIONES DE ENTRADA POR PROFUNDIDAD
   ════════════════════════════════════════════ */

/* Cada tarjeta del ranking entra con delay acumulado */
div[style*="margin:8px 0"]:nth-child(1) {{ animation: fadeUp 0.30s ease both; }}
div[style*="margin:8px 0"]:nth-child(2) {{ animation: fadeUp 0.30s ease 0.07s both; }}
div[style*="margin:8px 0"]:nth-child(3) {{ animation: fadeUp 0.30s ease 0.14s both; }}
div[style*="margin:8px 0"]:nth-child(4) {{ animation: fadeUp 0.30s ease 0.21s both; }}
div[style*="margin:8px 0"]:nth-child(5) {{ animation: fadeUp 0.30s ease 0.28s both; }}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS VISUALES
# ─────────────────────────────────────────────

def _score_ring_svg(score: float, color: str, size: int = 70) -> str:
    """SVG circular score ring — más visual que un número suelto."""
    pct = score / 10.0
    r = 28
    circ = 2 * 3.14159 * r
    dash = pct * circ
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 70 70" style="display:block;margin:0 auto">'
        f'<circle cx="35" cy="35" r="{r}" fill="none" stroke="{C["border"]}" stroke-width="6"/>'
        f'<circle cx="35" cy="35" r="{r}" fill="none" stroke="{color}" stroke-width="6"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"'
        f' transform="rotate(-90 35 35)" style="filter:drop-shadow(0 0 5px {color})"/>'
        f'<text x="35" y="40" text-anchor="middle" font-family="Inter,sans-serif"'
        f' font-size="13" font-weight="700" fill="{C["text"]}">{score:.1f}</text>'
        f'</svg>'
    )


def _criterion_bar(label: str, score: float, color: str) -> str:
    pct = score / 10.0 * 100
    if pct >= 75:
        fill_color = C["emerald"]
    elif pct >= 45:
        fill_color = C["amber"]
    else:
        fill_color = C["red"]
    return (
        f'<div class="criterion-bar-wrap">'
        f'<div class="criterion-bar-label">'
        f'<span style="color:{C["muted"]}">{label}</span>'
        f'<span style="color:{C["text"]};font-weight:600">{score:.1f}</span>'
        f'</div>'
        f'<div class="criterion-bar-track">'
        f'<div class="criterion-bar-fill" style="width:{pct:.0f}%;background:linear-gradient(90deg,{fill_color}CC,{fill_color})"></div>'
        f'</div>'
        f'</div>'
    )


def _service_chips(services: list[str], color: str) -> str:
    chips = "".join(
        f'<span class="service-chip" style="background:{color}22;color:{color};border:1px solid {color}55">{s}</span>'
        for s in services
    )
    return f'<div style="margin-top:8px">{chips}</div>'


# ─────────────────────────────────────────────
# YAML HELPERS
# ─────────────────────────────────────────────

_TIPO_OPTIONS  = ["clasificacion_binaria", "propensity_scoring", "regression",
                  "clustering", "deteccion_fraude", "scoring_credito", "churn", "aml"]
_FREC_OPTIONS  = ["baja", "media", "alta", "continua"]
_ESCAL_OPTIONS = ["baja", "media", "alta"]
_EXP_OPTIONS   = ["baja", "media", "alta"]


def _parse_yaml_upload(content: bytes) -> tuple[bool, str]:
    """Parse YAML bytes and write f_* session state keys. Returns (ok, message)."""
    try:
        data = yaml.safe_load(content)
    except Exception as exc:
        return False, f"Error al leer el YAML: {exc}"

    if not isinstance(data, dict):
        return False, "Formato no válido — se esperaba un diccionario."

    modelo     = data.get("modelo",       {}) or {}
    despliegue = data.get("despliegue",   {}) or {}
    org        = data.get("organizacion", {}) or {}

    n = 0

    if modelo.get("descripcion"):
        st.session_state.f_descripcion = str(modelo["descripcion"])
        n += 1

    if modelo.get("tipo") in _TIPO_OPTIONS:
        st.session_state.f_tipo_modelo = modelo["tipo"]
        n += 1

    if modelo.get("num_features"):
        try:
            st.session_state.f_num_features = max(1, min(10_000, int(modelo["num_features"])))
            n += 1
        except (ValueError, TypeError):
            pass

    if despliegue.get("latencia_requerida_ms"):
        try:
            st.session_state.f_latencia_ms = max(50, min(86_400_000, int(despliegue["latencia_requerida_ms"])))
            n += 1
        except (ValueError, TypeError):
            pass

    if despliegue.get("frecuencia_inferencia") in _FREC_OPTIONS:
        st.session_state.f_frecuencia = despliegue["frecuencia_inferencia"]
        n += 1

    if despliegue.get("presupuesto_mensual_usd"):
        try:
            st.session_state.f_presupuesto = max(5, min(10_000, int(despliegue["presupuesto_mensual_usd"])))
            n += 1
        except (ValueError, TypeError):
            pass

    if despliegue.get("escalabilidad_requerida") in _ESCAL_OPTIONS:
        st.session_state.f_escalabilidad = despliegue["escalabilidad_requerida"]
        n += 1

    if org.get("experiencia_tecnica") in _EXP_OPTIONS:
        st.session_state.f_experiencia = org["experiencia_tecnica"]
        n += 1

    if n == 0:
        return False, "No se encontraron campos reconocidos en el YAML."

    return True, f"{n} campo(s) cargado(s) correctamente."


def _apply_preset_to_state(case: CaseInput) -> None:
    """Copy a CaseInput preset into f_* session state keys."""
    st.session_state.f_descripcion   = case.descripcion
    st.session_state.f_latencia_ms   = int(case.latencia_requerida_ms)
    st.session_state.f_frecuencia    = case.frecuencia_inferencia
    st.session_state.f_presupuesto   = int(case.presupuesto_mensual_usd)
    st.session_state.f_escalabilidad = case.escalabilidad_requerida
    st.session_state.f_experiencia   = case.experiencia_tecnica


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _render_sidebar() -> None:
    # ── LOGO ──────────────────────────────────────────────────────────
    logo_path = Path(__file__).parent / "assets" / "logosimbolo_ucentral_4.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=220)

    st.sidebar.markdown(f"""
    <div style="padding:10px 0 6px 0;text-align:center">
        <div style="font-size:1.35em;font-weight:800;
                    background:linear-gradient(135deg,#FF9900,#FFD166);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            🏗️ ML Arch Recommender
        </div>
        <div style="font-size:0.72em;color:{C['muted']};margin-top:3px;letter-spacing:0.06em">
            AWS DEPLOYMENT ADVISOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(
        f'<div style="height:1px;background:linear-gradient(90deg,transparent,{C["border"]},transparent);margin:8px 0 16px"></div>',
        unsafe_allow_html=True,
    )

    # ── CASOS DE EJEMPLO ──────────────────────────────────────────────
    st.sidebar.markdown(f'<div class="section-label">Casos de ejemplo</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("💳 Fintech", use_container_width=True):
        case = CASOS.get("serverless")
        if case:
            _apply_preset_to_state(case)
        st.session_state.result = None
        st.session_state.analysis = None
        st.rerun()
    if col2.button("🏦 Banco", use_container_width=True):
        case = CASOS.get("batch")
        if case:
            _apply_preset_to_state(case)
        st.session_state.result = None
        st.session_state.analysis = None
        st.rerun()
    if col3.button("🚨 Fraude", use_container_width=True):
        case = CASOS.get("streaming")
        if case:
            _apply_preset_to_state(case)
        st.session_state.result = None
        st.session_state.analysis = None
        st.rerun()

    st.sidebar.markdown(
        f'<div style="height:1px;background:{C["border"]};margin:16px 0 12px"></div>',
        unsafe_allow_html=True,
    )

    # ── CARGA DE YAML ─────────────────────────────────────────────────
    st.sidebar.markdown(f'<div class="section-label">Cargar configuración YAML</div>', unsafe_allow_html=True)

    template_path = Path(__file__).parent / "configs" / "deployment_request_template.yaml"
    if template_path.exists():
        st.sidebar.download_button(
            "⬇️  Descargar plantilla",
            data=template_path.read_bytes(),
            file_name="deployment_request_template.yaml",
            mime="text/yaml",
            use_container_width=True,
            help="Descarga la plantilla, complétala y súbela aquí.",
        )

    uploaded = st.sidebar.file_uploader(
        "Subir YAML",
        type=["yaml", "yml"],
        label_visibility="collapsed",
        help="Completa la plantilla con tus parámetros y súbela para rellenar el formulario automáticamente.",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_yaml_file_id") != file_id:
            ok, msg = _parse_yaml_upload(uploaded.read())
            st.session_state._yaml_file_id = file_id
            st.session_state._yaml_msg     = (ok, msg)
            st.session_state.result        = None
            st.session_state.analysis      = None
            st.rerun()
    else:
        st.session_state._yaml_file_id = None

    if st.session_state.get("_yaml_msg"):
        ok, msg = st.session_state._yaml_msg
        if ok:
            st.sidebar.success(f"✅ {msg}")
        else:
            st.sidebar.error(f"❌ {msg}")
        st.session_state._yaml_msg = None

    st.sidebar.markdown(
        f'<div style="height:1px;background:{C["border"]};margin:16px 0 12px"></div>',
        unsafe_allow_html=True,
    )

    # ── PARÁMETROS ────────────────────────────────────────────────────
    st.sidebar.markdown(f'<div class="section-label">Parámetros del modelo</div>', unsafe_allow_html=True)

    st.sidebar.text_input("Descripción del caso", key="f_descripcion")

    st.sidebar.selectbox(
        "Tipo de modelo",
        _TIPO_OPTIONS,
        key="f_tipo_modelo",
    )

    st.sidebar.number_input(
        "Latencia requerida (ms)",
        min_value=50, max_value=86_400_000,
        step=100,
        key="f_latencia_ms",
        help="< 1 000 ms = tiempo real  ·  86 400 000 ms = batch diario",
    )

    st.sidebar.selectbox(
        "Frecuencia de inferencia",
        _FREC_OPTIONS,
        key="f_frecuencia",
        help="baja < 1K req/día  ·  continua = 24/7",
    )

    st.sidebar.number_input(
        "Presupuesto mensual (USD)",
        min_value=5, max_value=10_000,
        step=10,
        key="f_presupuesto",
    )

    st.sidebar.selectbox(
        "Escalabilidad requerida",
        _ESCAL_OPTIONS,
        key="f_escalabilidad",
    )

    st.sidebar.selectbox(
        "Experiencia técnica del equipo",
        _EXP_OPTIONS,
        key="f_experiencia",
    )

    st.sidebar.number_input(
        "Número de features",
        min_value=1, max_value=10_000,
        step=5,
        key="f_num_features",
        help="Estimado: num_features × 8 bytes = KB por request",
    )

    st.sidebar.markdown(
        f'<div style="height:1px;background:{C["border"]};margin:16px 0 12px"></div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔍  Analizar arquitectura", type="primary", use_container_width=True):
        num_f = st.session_state.f_num_features
        volumen_kb = round(num_f * 8 / 1024, 3) or 0.1
        case = CaseInput(
            descripcion=st.session_state.f_descripcion,
            tipo_modelo=st.session_state.f_tipo_modelo,
            latencia_requerida_ms=float(st.session_state.f_latencia_ms),
            frecuencia_inferencia=st.session_state.f_frecuencia,
            volumen_datos_kb=volumen_kb,
            presupuesto_mensual_usd=float(st.session_state.f_presupuesto),
            escalabilidad_requerida=st.session_state.f_escalabilidad,
            experiencia_tecnica=st.session_state.f_experiencia,
        )
        st.session_state.result   = recommend(case)
        st.session_state.analysis = None
        st.session_state.preset   = None
        st.session_state.sel_arch = None  # reset to winner on next render

    st.sidebar.markdown(f"""
    <div style="margin-top:32px;text-align:center;font-size:0.72em;color:{C['dim']};line-height:1.7">
        Motor: engine.py<br>
        Latencia 25% · Frecuencia 25%<br>
        Presupuesto 20% · Escalabilidad 15% · Experiencia 15%
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LANDING PAGE (sin resultado)
# ─────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown(f"""
    <div style="text-align:center;padding:48px 20px 32px">
        <div style="font-size:3.2em;margin-bottom:10px">🏗️</div>
        <h1 style="font-size:2.4em;font-weight:800;margin:0;
                   background:linear-gradient(135deg,#FF9900 0%,#FFD166 50%,#06B6D4 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            ML Architecture Recommender
        </h1>
        <p style="font-size:1.1em;color:{C['muted']};margin:12px auto 0;max-width:560px;line-height:1.6">
            Recibe en segundos la recomendación de arquitectura AWS óptima
            para desplegar tu modelo de Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("📊", "Recomendación inteligente",
         "Evalúa 5 patrones de despliegue — Serverless, Batch, Streaming, Containers y SageMaker — usando 5 criterios ponderados con base en conocimiento real de AWS."),
        ("🔍", "Análisis de viabilidad",
         "Verifica si puedes desplegar la arquitectura con los servicios que ya tiene tu empresa. Propone sustitutos cuando es posible y busca la alternativa más económica."),
        ("📘", "Guía lista para producción",
         "Genera estructura de proyecto, código base, comandos de despliegue, variables de entorno y roles IAM con principio de mínimo privilegio."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class="hero-feature">
                <div class="icon">{icon}</div>
                <div style="font-weight:700;font-size:1.05em;margin-bottom:8px">{title}</div>
                <div style="color:{C['muted']};font-size:0.88em;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:36px;padding:20px;
                background:{C['surface']};border-radius:14px;
                border:1px dashed {C['border']}">
        <span style="font-size:1.3em">👈</span>
        <span style="color:{C['muted']};margin-left:10px">
            Selecciona un caso de ejemplo o configura los parámetros en el panel lateral para comenzar
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture reference cards ────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:40px">
        <div class="section-label" style="text-align:center;margin-bottom:16px">
            Arquitecturas disponibles
        </div>
    </div>
    """, unsafe_allow_html=True)

    arch_info = [
        ("serverless", "⚡ Serverless",   "Lambda + API GW",       "< $5/mes",   "Bajo volumen, respuesta rápida"),
        ("batch",      "📦 Batch",        "Glue + S3 + EventBridge","< $50/mes",  "Scoring masivo, sin tiempo real"),
        ("streaming",  "🌊 Streaming",    "Kinesis + Lambda",       "~$100/mes",  "Fraude, eventos en tiempo real"),
        ("containers", "🐳 Containers",   "ECS Fargate + ALB",      "~$80/mes",   "APIs persistentes, control total"),
        ("sagemaker",  "🧠 SageMaker",    "SageMaker Endpoint",     "~$150/mes",  "MLOps maduro, A/B testing"),
    ]
    cols = st.columns(5)
    for col, (arch_id, title, stack, cost, use_case) in zip(cols, arch_info):
        color = ARCH_COLORS[arch_id]
        with col:
            st.markdown(f"""
            <div style="background:{C['card']};border:1px solid {color}40;border-top:3px solid {color};
                        border-radius:12px;padding:14px 12px;text-align:center;height:100%">
                <div style="font-weight:700;font-size:0.95em;color:{color};margin-bottom:6px">{title}</div>
                <div style="font-size:0.78em;color:{C['muted']};margin-bottom:6px">{stack}</div>
                <div style="font-size:0.8em;color:{C['primary']};font-weight:600;margin-bottom:4px">{cost}</div>
                <div style="font-size:0.75em;color:{C['dim']}">{use_case}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 1 — RECOMENDACIÓN
# ─────────────────────────────────────────────

def _render_tab_recommendation(result: RecommendationResult) -> None:
    # ── Resolve selected architecture (defaults to winner) ───────────
    sel_id     = st.session_state.get("sel_arch") or result.recomendacion.id
    valid_ids  = {a.id for a in result.ranking}
    if sel_id not in valid_ids:
        sel_id = result.recomendacion.id

    sel  = next(a for a in result.ranking if a.id == sel_id)
    is_winner = sel.posicion == 1
    color = ARCH_COLORS.get(sel.id, C["primary"])
    glow  = ARCH_GLOW.get(sel.id, "rgba(255,153,0,0.1)")
    icon  = ARCH_ICONS.get(sel.id, "🏗️")

    # ── Detail card label ────────────────────────────────────────────
    if is_winner:
        st.markdown(f'<div class="section-label">Arquitectura recomendada</div>', unsafe_allow_html=True)
    else:
        winner_name = result.recomendacion.nombre
        st.markdown(
            f'<div class="section-label">'
            f'Vista de alternativa'
            f'<span style="font-weight:400;color:{C["dim"]};margin-left:8px">·  Recomendada: {winner_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Detail card ──────────────────────────────────────────────────
    ring_svg = _score_ring_svg(sel.score_total, color, size=80)

    criteria_bars = "".join(
        _criterion_bar(lbl, sel.desglose.get(key, 0), color)
        for key, lbl in [
            ("latencia", "Latencia"), ("frecuencia", "Frecuencia"),
            ("presupuesto", "Presupuesto"), ("escalabilidad", "Escalabilidad"),
            ("experiencia", "Experiencia"),
        ]
    )

    services_html = _service_chips(sel.servicios_aws, color)

    if is_winner:
        badge_text = f'#{sel.posicion} &nbsp;·&nbsp; {result.confianza:.0%} confianza'
        body_html = (
            f'<div style="color:{C["muted"]};font-size:0.89em;margin-bottom:10px;line-height:1.65">{result.justificacion}</div>'
            f'<div style="font-size:0.82em;color:{C["dim"]}">'
            f'<b style="color:{C["muted"]}">Criterio decisivo:</b> {result.criterio_decisivo}'
            f'</div>'
        )
    else:
        badge_text = f'#{sel.posicion} en el ranking'
        descartada = sel.descartada_por or "No es la primera opción del motor para los parámetros actuales."
        body_html = (
            f'<div style="background:rgba(245,158,11,0.09);border-left:3px solid {C["amber"]};'
            f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:8px;'
            f'font-size:0.86em;color:{C["muted"]};line-height:1.55">'
            f'<b style="color:{C["amber"]}">¿Por qué no es la primera opción?</b><br>{descartada}'
            f'</div>'
        )

    st.markdown(
        f'<div class="winner-card" style="background:linear-gradient(135deg,{glow},rgba(28,36,56,0.9));'
        f'border:2px solid {color}55;box-shadow:0 8px 36px {glow},inset 0 1px 0 {color}25">'
        f'<div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap">'
        f'<div style="text-align:center;flex-shrink:0">'
        f'{ring_svg}'
        f'<div style="font-size:0.65em;color:{C["muted"]};margin-top:3px;letter-spacing:0.1em;text-transform:uppercase">Score</div>'
        f'</div>'
        f'<div style="flex:1;min-width:200px">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">'
        f'<span style="font-size:1.55em">{icon}</span>'
        f'<span style="font-size:1.45em;font-weight:800;color:{color}">{sel.nombre}</span>'
        f'<span class="score-pill" style="background:{color}22;color:{color};border:1px solid {color}55">'
        f'{badge_text}</span>'
        f'</div>'
        f'{body_html}'
        f'{services_html}'
        f'</div>'
        f'<div style="min-width:190px;flex:0 0 210px">'
        f'<div class="section-label">Desglose de criterios</div>'
        f'{criteria_bars}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Advertencias ────────────────────────────────────────────────
    if result.advertencias:
        st.markdown(f'<div class="section-label" style="margin-top:20px">Advertencias detectadas</div>', unsafe_allow_html=True)
        for adv in result.advertencias:
            if adv.tipo == "restriccion_dura":
                st.markdown(f'<div class="warning-strip">🚨 <b>Restricción dura:</b> {adv.mensaje}</div>', unsafe_allow_html=True)
            elif adv.tipo in ("tension", "restriccion_presupuestal"):
                st.markdown(f'<div class="tension-strip">💡 <b>Tensión:</b> {adv.mensaje}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tension-strip">⚠️ {adv.mensaje}</div>', unsafe_allow_html=True)

    # ── Ranking — clickable rows ─────────────────────────────────────
    st.markdown(
        f'<div class="section-label" style="margin-top:24px">Ranking &nbsp;—&nbsp; haz clic en una fila para explorar su detalle</div>',
        unsafe_allow_html=True,
    )

    for arch_r in result.ranking:
        c  = ARCH_COLORS.get(arch_r.id, C["muted"])
        ic = ARCH_ICONS.get(arch_r.id, "🏗️")
        is_row_winner   = arch_r.posicion == 1
        is_row_selected = arch_r.id == sel_id
        d  = arch_r.desglose

        criteria_mini = "".join(
            f'<div style="text-align:center;padding:0 4px">'
            f'<div style="font-size:0.7em;color:{C["dim"]};margin-bottom:2px">{lbl}</div>'
            f'<div style="font-size:0.92em;font-weight:700;color:{"#10B981" if d.get(k,0)>=7 else "#F59E0B" if d.get(k,0)>=4.5 else "#EF4444"}">{d.get(k,0):.1f}</div>'
            f'</div>'
            for k, lbl in [("latencia","Lat"),("frecuencia","Frec"),("presupuesto","Pres"),("escalabilidad","Esc"),("experiencia","Exp")]
        )

        if is_row_selected:
            border_style = f"border:2px solid {c};box-shadow:0 4px 20px {ARCH_GLOW.get(arch_r.id,'rgba(0,0,0,0.2)')}"
        elif is_row_winner:
            border_style = f"border:2px solid {c}55;box-shadow:0 4px 20px {ARCH_GLOW.get(arch_r.id,'rgba(0,0,0,0.2)')}"
        else:
            border_style = f"border:1px solid {C['border']}"

        winner_badge = (
            f'<span class="score-pill" style="background:{c}22;color:{c};border:1px solid {c}44;margin-left:8px">★ RECOMENDADO</span>'
            if is_row_winner else ""
        )
        descartada_html = (
            f'<div style="font-size:0.79em;color:{C["dim"]};margin-top:5px;line-height:1.4">{arch_r.descartada_por}</div>'
            if arch_r.descartada_por else ""
        )

        col_card, col_btn = st.columns([11, 1])
        with col_card:
            st.markdown(
                f'<div style="background:{C["card"]};{border_style};border-radius:13px;padding:16px 20px;margin:8px 0">'
                f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
                f'<div style="flex:1;min-width:160px">'
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
                f'<span style="font-size:1.4em;color:{C["border"]};font-weight:800">#{arch_r.posicion}</span>'
                f'<span style="font-size:1.2em">{ic}</span>'
                f'<span style="font-weight:700;font-size:1.0em;color:{c}">{arch_r.nombre}</span>'
                f'{winner_badge}'
                f'</div>'
                f'{descartada_html}'
                f'</div>'
                f'<div style="display:flex;gap:4px;flex-wrap:wrap">{criteria_mini}</div>'
                f'<div style="flex-shrink:0">{_score_ring_svg(arch_r.score_total, c, size=58)}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            if is_row_selected:
                st.markdown(
                    f'<div style="padding-top:22px;text-align:center;font-size:0.78em;font-weight:700;color:{c}">✓ Viendo</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div style="padding-top:16px">', unsafe_allow_html=True)
                if st.button("Ver →", key=f"sel_btn_{arch_r.id}", use_container_width=True):
                    st.session_state.sel_arch = arch_r.id
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2 — ANÁLISIS DE SERVICIOS
# ─────────────────────────────────────────────

def _render_tab_services(result: RecommendationResult) -> None:
    st.markdown(f'<div class="section-label">Selecciona la arquitectura que te interesa</div>', unsafe_allow_html=True)

    arch_options = {
        f"{ARCH_ICONS.get(a.id,'🏗️')}  {a.posicion}. {a.nombre}": a.id
        for a in result.ranking
    }
    labels = list(arch_options.keys())
    ids    = list(arch_options.values())
    sel_id = st.session_state.get("sel_arch") or result.recomendacion.id
    default_idx = ids.index(sel_id) if sel_id in ids else 0

    chosen_label = st.selectbox(
        "Arquitectura de interés",
        labels,
        index=default_idx,
        label_visibility="collapsed",
    )
    chosen_id = arch_options[chosen_label]
    arch_color = ARCH_COLORS.get(chosen_id, C["primary"])

    servicios_relevantes = get_relevant_services_for_arch(chosen_id)
    requeridos_set = {s["service"] for s in servicios_relevantes if s["tipo"] == "requerido"}
    sustitutos_map = {s["service"]: s["sustituye"] for s in servicios_relevantes if s["tipo"] == "sustituto"}

    # ── Leyenda de servicios ─────────────────────────────────────────
    req_chips = _service_chips(list(requeridos_set), arch_color)
    sub_chips  = _service_chips(list(sustitutos_map.keys()), C["amber"]) if sustitutos_map else f'<span style="color:{C["dim"]};font-size:0.85em">Ninguno</span>'

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:16px">
      <div style="display:flex;gap:32px;flex-wrap:wrap">
        <div>
          <div class="section-label">Servicios requeridos</div>
          {req_chips}
        </div>
        <div>
          <div class="section-label">Sustitutos posibles</div>
          {sub_chips}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Glosario de servicios relevantes ────────────────────────────
    glosario_services = list(requeridos_set) + list(sustitutos_map.keys())
    with st.expander(f"📖  Glosario — ¿qué hace cada servicio de esta arquitectura? ({len(glosario_services)} servicios)", expanded=False):
        cols_g = st.columns(2)
        for i, svc_name in enumerate(glosario_services):
            info = AWS_SERVICE_INFO.get(svc_name)
            if not info:
                continue
            is_req = svc_name in requeridos_set
            badge_color = arch_color if is_req else C["amber"]
            badge_label = "Requerido" if is_req else "Sustituto"
            with cols_g[i % 2]:
                st.markdown(
                    f'<div style="background:{C["surface"]};border:1px solid {badge_color}40;'
                    f'border-left:3px solid {badge_color};border-radius:0 10px 10px 0;'
                    f'padding:14px 16px;margin:6px 0">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap">'
                    f'<span style="font-size:1.25em">{info["icon"]}</span>'
                    f'<span style="font-weight:700;color:{badge_color};font-size:0.97em">{svc_name}</span>'
                    f'<span style="background:{badge_color}18;color:{badge_color};border:1px solid {badge_color}40;'
                    f'border-radius:20px;font-size:0.68em;font-weight:700;padding:2px 9px;letter-spacing:0.04em">'
                    f'{badge_label}</span>'
                    f'<span style="background:{C["card"]};color:{C["dim"]};border:1px solid {C["border"]};'
                    f'border-radius:20px;font-size:0.68em;padding:2px 9px">{info["categoria"]}</span>'
                    f'</div>'
                    f'<div style="font-size:0.86em;color:{C["muted"]};line-height:1.65;margin-bottom:6px">{info["descripcion"]}</div>'
                    f'<div style="font-size:0.80em;color:{C["dim"]};line-height:1.5">'
                    f'<b style="color:{C["muted"]}">En ML:</b> {info["uso_ml"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Selector de servicios disponibles ───────────────────────────
    st.markdown(f'<div class="section-label">¿Qué servicios tiene tu empresa?</div>', unsafe_allow_html=True)
    st.caption("Marca todos los que tu organización tiene contratados o autorizados en AWS.")

    all_relevant_names = [s["service"] for s in servicios_relevantes]
    extra_services = [s for s in ALL_SERVICES if s not in all_relevant_names]
    display_services = all_relevant_names + extra_services

    selected_services: list[str] = []
    cols = st.columns(3)
    for i, svc in enumerate(display_services):
        if svc in requeridos_set:
            label = f"🔵 {svc}"
        elif svc in sustitutos_map:
            label = f"🟡 {svc}"
        else:
            label = f"⚪ {svc}"
        if cols[i % 3].checkbox(label, key=f"svc_{chosen_id}_{svc}"):
            selected_services.append(svc)

    st.markdown(f"""
    <div style="font-size:0.78em;color:{C['dim']};margin:8px 0 16px">
        🔵 Requerido  ·  🟡 Sustituto posible  ·  ⚪ Otro servicio AWS
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍  Analizar viabilidad", type="primary"):
        analysis = analyze_services_for_arch(chosen_id, selected_services, result.ranking)
        st.session_state.analysis = analysis
        st.session_state.analysis_arch_id = chosen_id

    if st.session_state.get("analysis") and st.session_state.get("analysis_arch_id") == chosen_id:
        _render_analysis_result(st.session_state.analysis)


def _render_analysis_result(analysis: ServiceAnalysisResult) -> None:
    arch = analysis.arch
    icon = ARCH_ICONS.get(arch.id, "🏗️")
    color = ARCH_COLORS.get(arch.id, C["primary"])

    # ── Resultado principal ──────────────────────────────────────────
    if analysis.es_viable:
        st.markdown(f"""
        <div class="viable-banner">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <span style="font-size:1.5em">✅</span>
            <span style="font-size:1.15em;font-weight:700;color:{C['emerald']}">
              {icon} {arch.nombre} es viable con tus servicios
            </span>
          </div>
          <div style="color:{C['muted']};font-size:0.9em">
            Puedes desplegar esta arquitectura directamente con los servicios disponibles en tu empresa.
            Pasa a la pestaña <b>Guía de implementación</b> para ver los pasos.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        faltantes_chips = "".join(
            f'<span class="score-pill" style="background:rgba(239,68,68,0.12);color:{C["red"]};border:1px solid {C["red"]}40;margin:2px">{f}</span>'
            for f in analysis.faltantes
        )
        st.markdown(f"""
        <div class="not-viable-banner">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span style="font-size:1.5em">❌</span>
            <span style="font-size:1.15em;font-weight:700;color:{C['red']}">
              {icon} {arch.nombre} no es viable directamente
            </span>
          </div>
          <div style="color:{C['muted']};font-size:0.9em;margin-bottom:8px">
            Servicios faltantes sin sustituto disponible:
          </div>
          <div>{faltantes_chips}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Servicios OK ─────────────────────────────────────────────────
    if analysis.servicios_ok:
        with st.expander(f"✅  Servicios confirmados ({len(analysis.servicios_ok)})", expanded=False):
            chips = _service_chips(analysis.servicios_ok, C["emerald"])
            st.markdown(chips, unsafe_allow_html=True)

    # ── Sustituciones ────────────────────────────────────────────────
    if analysis.sustituciones:
        with st.expander(f"🔄  {len(analysis.sustituciones)} sustitución(es) aplicable(s)", expanded=True):
            for sub in analysis.sustituciones:
                st.markdown(f"""
                <div class="sub-card">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <span style="color:{C['muted']};font-weight:600">{sub.servicio_requerido}</span>
                    <span style="color:{C['dim']}">→</span>
                    <span style="color:{C['amber']};font-weight:700">{sub.servicio_sustituto}</span>
                  </div>
                  <div style="display:flex;gap:20px;font-size:0.82em;color:{C['dim']};margin-bottom:6px">
                    <span>💰 {sub.impacto_costo}</span>
                    <span>⏱ {sub.impacto_latencia}</span>
                  </div>
                  <div style="font-size:0.85em;color:{C['muted']};line-height:1.5">{sub.nota}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Alternativa económica ────────────────────────────────────────
    if not analysis.es_viable and analysis.alternativa_precio:
        alt = analysis.alternativa_precio
        alt_icon  = ARCH_ICONS.get(alt.id, "🏗️")
        alt_color = ARCH_COLORS.get(alt.id, C["primary"])
        min_cost  = _ARCH_MIN_COST_USD.get(alt.id, "?")
        ring = _score_ring_svg(alt.score_total, alt_color, size=64)

        st.markdown(f"""
        <div class="alt-price-card">
          <div class="section-label">Alternativa más económica disponible</div>
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <div>{ring}</div>
            <div>
              <div style="font-size:1.15em;font-weight:700;color:{alt_color};margin-bottom:4px">
                {alt_icon} {alt.nombre}
              </div>
              <div style="font-size:0.88em;color:{C['muted']}">
                Desde <b style="color:{C['primary']}">${min_cost}/mes</b>
                &nbsp;·&nbsp; Score: {alt.score_total:.2f}/10
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if analysis.alternativa_precio_sustituciones:
            with st.expander("🔄  Sustituciones para la alternativa"):
                for sub in analysis.alternativa_precio_sustituciones:
                    st.markdown(f"- **{sub.servicio_requerido}** → `{sub.servicio_sustituto}`: {sub.nota}")

    elif not analysis.es_viable and analysis.alternativa_precio is None:
        st.markdown(f"""
        <div class="tension-strip" style="margin-top:12px">
            ⚠️ No se encontró arquitectura viable con los servicios indicados.
            Intenta agregar más servicios de la lista o consulta la pestaña de Recomendación
            para ver qué arquitecturas requieren qué servicios.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3 — GUÍA DE IMPLEMENTACIÓN
# ─────────────────────────────────────────────

def _render_tab_guide(result: RecommendationResult) -> None:
    analysis: ServiceAnalysisResult | None = st.session_state.get("analysis")

    arch_options = {
        f"{ARCH_ICONS.get(a.id,'🏗️')}  {a.posicion}. {a.nombre}": a.id
        for a in result.ranking
    }
    default_idx = 0
    if analysis:
        eff_id = (
            analysis.alternativa_precio.id
            if (not analysis.es_viable and analysis.alternativa_precio)
            else analysis.arch.id
        )
        ids = list(arch_options.values())
        if eff_id in ids:
            default_idx = ids.index(eff_id)

    chosen_label = st.selectbox(
        "Arquitectura para la guía",
        list(arch_options.keys()),
        index=default_idx,
        key="guide_arch_sel",
        label_visibility="collapsed",
    )
    chosen_id = arch_options[chosen_label]

    guide = IMPL_GUIDES.get(chosen_id)
    if not guide:
        st.info("Guía de implementación no disponible para esta arquitectura.")
        return

    arch_obj  = next(a for a in result.ranking if a.id == chosen_id)
    icon      = ARCH_ICONS.get(chosen_id, "🏗️")
    color     = ARCH_COLORS.get(chosen_id, C["primary"])
    glow      = ARCH_GLOW.get(chosen_id, "rgba(255,153,0,0.1)")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{glow},transparent);
                border:1px solid {color}40;border-radius:14px;
                padding:18px 24px;margin-bottom:20px">
      <div style="font-size:1.5em;font-weight:800;color:{color}">
          {icon} Guía de implementación — {arch_obj.nombre}
      </div>
      <div style="font-size:0.85em;color:{C['muted']};margin-top:4px">
          Mínimo viable para llevar tu modelo a producción en AWS
      </div>
    </div>
    """, unsafe_allow_html=True)

    if analysis and analysis.sustituciones and analysis.arch.id == chosen_id:
        subs_txt = "  ·  ".join(
            f"{s.servicio_requerido} → {s.servicio_sustituto}"
            for s in analysis.sustituciones
        )
        st.markdown(f"""
        <div class="tension-strip" style="margin-bottom:16px">
            🔄 <b>Sustituciones activas:</b> {subs_txt}
        </div>
        """, unsafe_allow_html=True)

    # ── 1. Estructura ────────────────────────────────────────────────
    with st.expander("📁  1. Estructura del proyecto", expanded=True):
        st.code("\n".join(guide["estructura"]), language="text")

    # ── 2. Código ────────────────────────────────────────────────────
    with st.expander("📄  2. Archivos y código base", expanded=True):
        for archivo in guide["archivos"]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:12px 0 6px">
                <span style="background:{color}22;color:{color};padding:3px 10px;
                             border-radius:6px;font-family:monospace;font-size:0.85em;font-weight:600">
                    {archivo['nombre']}
                </span>
                <span style="color:{C['muted']};font-size:0.85em">{archivo['descripcion']}</span>
            </div>
            """, unsafe_allow_html=True)
            lang = "yaml" if archivo["nombre"].endswith((".yaml", ".yml")) else "python"
            st.code(archivo["codigo"], language=lang)

    # ── 3. Despliegue ────────────────────────────────────────────────
    with st.expander("🚀  3. Comandos de despliegue", expanded=True):
        config_steps = _build_configuracion(chosen_id)
        for step in config_steps:
            st.markdown(f"""
            <div class="step-card">
                <div style="font-weight:700;color:{C['primary']}">Paso {step.numero} — {step.titulo}</div>
                <div style="color:{C['muted']};font-size:0.87em;margin-top:4px">{step.descripcion}</div>
            </div>
            """, unsafe_allow_html=True)
            if step.comando:
                st.code(step.comando, language="bash")

    # ── 4. Env vars ──────────────────────────────────────────────────
    with st.expander("🔑  4. Variables de entorno"):
        for var in guide["env_vars"]:
            secret_icon = " 🔒" if var["es_secreto"] else ""
            st.markdown(f"""
            <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;
                        padding:12px 16px;margin:8px 0">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-family:monospace;font-size:0.9em;font-weight:700;color:{color}">{var['nombre']}{secret_icon}</span>
                <span style="font-size:0.78em;color:{C['dim']}">{var['descripcion']}</span>
              </div>
              <div style="font-family:monospace;font-size:0.82em;color:{C['muted']};margin-top:6px">{var['nombre']}={var['ejemplo']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.78em;color:{C["dim"]};margin-top:8px">🔒 Las marcadas con candado van en AWS Secrets Manager o SSM Parameter Store (SecureString) — nunca en texto plano ni en el repositorio.</div>', unsafe_allow_html=True)

    # ── 5. IAM ───────────────────────────────────────────────────────
    with st.expander("🔐  5. Permisos de acceso y roles IAM"):
        p = guide["permisos"]
        vis = p["visibilidad"]
        vis_color = {
            "PÚBLICO":         C["red"],
            "RESTRINGIDO":     C["amber"],
            "PRIVADO":         C["emerald"],
            "PRIVADO/INTERNO": C["emerald"],
        }.get(vis, C["muted"])

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{vis_color}18,transparent);
                    border:2px solid {vis_color}50;border-radius:14px;
                    padding:16px 20px;margin-bottom:16px">
            <div style="font-size:1.1em;font-weight:700;color:{vis_color};margin-bottom:6px">
                🌐 Visibilidad: {vis}
            </div>
            <div style="font-size:0.87em;white-space:pre-line;color:{C['muted']};line-height:1.6">
                {p['visibilidad_detalle']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="section-label">Quién edita configuración</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="iam-card">{p["quien_edita_config"]}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="section-label">Quién accede a logs</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="iam-card">{p["quien_accede_logs"]}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-label" style="margin-top:12px">Quién modifica secretos</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="iam-card">{p["quien_modifica_secrets"]}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-label" style="margin-top:12px">Permisos del repositorio / CI-CD</div>', unsafe_allow_html=True)
        for item in p["permisos_repo"]:
            if item.startswith("✓"):
                bg, ic, text_color = f"{C['emerald']}18", "✓", C["emerald"]
                st.markdown(f'<div style="background:{bg};border-left:3px solid {text_color};border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0;font-size:0.87em"><span style="color:{text_color};font-weight:700">{ic}</span> {item[1:].strip()}</div>', unsafe_allow_html=True)
            elif item.startswith("✗"):
                bg, ic, text_color = f"{C['red']}18", "✗", C["red"]
                st.markdown(f'<div style="background:{bg};border-left:3px solid {text_color};border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0;font-size:0.87em"><span style="color:{text_color};font-weight:700">{ic}</span> {item[1:].strip()}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:4px 12px;font-size:0.87em;color:{C["muted"]}">• {item}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-label" style="margin-top:14px">Roles mínimos recomendados</div>', unsafe_allow_html=True)
        for rol in p["roles"]:
            with st.expander(f"👤  {rol['nombre']}  —  {rol['uso']}"):
                perms_html = "".join(
                    f'<div style="font-size:0.85em;padding:4px 0;color:{C["muted"]}">✓ <code style="font-size:0.92em">{perm}</code></div>'
                    for perm in rol["permisos"]
                )
                st.markdown(f"""
                <div class="role-card">
                    {perms_html}
                    <div style="margin-top:10px;padding:8px 12px;background:rgba(239,68,68,0.10);
                                border-radius:8px;font-size:0.83em;color:{C['red']}">
                        <b>NO incluir:</b> {rol['no_incluir']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                    border-radius:12px;padding:14px 18px;margin-top:12px">
            <div style="font-size:0.85em;font-weight:600;color:{C['amber']};margin-bottom:6px">
                ⚖️ Principio de mínimo privilegio
            </div>
            <div style="font-size:0.87em;color:{C['muted']};line-height:1.6">{p['minimo_privilegio']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 6. Buena práctica ────────────────────────────────────────────
    with st.expander("💡  6. Buena práctica final", expanded=True):
        st.markdown(f"""
        <div class="tip-card">
            <div style="font-size:1.2em;margin-bottom:8px">💡</div>
            <div style="font-size:0.95em;color:{C['text']};line-height:1.7">{guide['buena_practica']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    for key in ("result", "analysis", "preset", "analysis_arch_id", "_yaml_file_id", "_yaml_msg", "sel_arch"):
        if key not in st.session_state:
            st.session_state[key] = None

    _field_defaults: dict = {
        "f_descripcion":   "Modelo de clasificación crediticia",
        "f_tipo_modelo":   "clasificacion_binaria",
        "f_latencia_ms":   800,
        "f_frecuencia":    "baja",
        "f_presupuesto":   50,
        "f_escalabilidad": "media",
        "f_experiencia":   "media",
        "f_num_features":  10,
    }
    for k, v in _field_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _render_sidebar()

    result: RecommendationResult | None = st.session_state.result

    if result is None:
        _render_landing()
        return

    # ── Input recibido (header compacto) ────────────────────────────
    ci = result.input_recibido
    st.markdown(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:12px;
                padding:12px 20px;margin-bottom:16px;display:flex;align-items:center;
                flex-wrap:wrap;gap:16px">
        <div style="font-weight:600;color:{C['text']};flex:1;min-width:200px">{ci.descripcion}</div>
        <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:0.82em;color:{C['muted']}">
            <span>⏱ {ci.latencia_requerida_ms:,.0f} ms</span>
            <span>🔄 Frec: {ci.frecuencia_inferencia}</span>
            <span>💰 ${ci.presupuesto_mensual_usd}/mes</span>
            <span>📈 Escal: {ci.escalabilidad_requerida}</span>
            <span>🎓 Exp: {ci.experiencia_tecnica}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  📊  Recomendación  ",
        "  🔍  Análisis de servicios  ",
        "  📘  Guía de implementación  ",
    ])

    with tab1:
        _render_tab_recommendation(result)

    with tab2:
        _render_tab_services(result)

    with tab3:
        _render_tab_guide(result)


if __name__ == "__main__":
    main()
