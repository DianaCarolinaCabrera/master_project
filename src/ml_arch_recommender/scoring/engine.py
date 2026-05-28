"""
Multicriterion scoring engine for ML deployment architecture recommendation.

Design principle: every score has a documented reason. The engine produces not
only a ranking but a structured explanation of WHY each architecture was scored
the way it was — covering the winner's advantages, the runner-up's gap, and
why each discarded option fails the specific case.

Mirrors the logic implemented in the n8n Code Node (n8n/workflow_export.json).
"""

from __future__ import annotations

import csv
import yaml  # pyyaml — installed via poetry (pyproject.toml)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ArchitectureId = Literal["serverless", "batch", "streaming", "containers", "sagemaker"]
FrequencyLevel  = Literal["baja", "media", "alta", "continua"]
ScaleLevel      = Literal["baja", "media", "alta"]
ExperienceLevel = Literal["baja", "media", "alta"]


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CaseInput:
    descripcion: str
    tipo_modelo: str
    latencia_requerida_ms: float
    frecuencia_inferencia: FrequencyLevel
    volumen_datos_kb: float
    presupuesto_mensual_usd: float
    escalabilidad_requerida: ScaleLevel
    experiencia_tecnica: ExperienceLevel
    disponibilidad_requerida: str = "alta"
    # Servicios disponibles en la organización (vacío = sin restricción)
    # Arquitecturas que requieran servicios no listados serán marcadas como bloqueadas.
    # Ejemplo: ["lambda", "s3", "ecs"] excluye sagemaker, kinesis, glue.
    servicios_disponibles: list[str] = field(default_factory=list)


@dataclass
class CriterionDetail:
    """Score and documented reason for a single criterion of a single architecture."""
    score: float
    razon: str           # Why this architecture gets this score for this criterion
    es_restriccion: bool = False  # True = hard constraint triggered (near-disqualifying)
    nota_catalogo: str = ""       # Constraint text from the service catalog (auto-populated for restricted criteria)


@dataclass
class CatalogServiceDetail:
    """Service card from the AWS catalog — enriches the agent output with official references."""
    service: str
    category: str
    description: str
    benefits: str
    constraints: str
    use_cases: str
    official_url: str
    vectores: dict[str, int]   # keys: cost_efficiency, simplicity, scalability, low_latency, managed_level, beginner_friendly

    @property
    def beginner_friendly(self) -> bool:
        return self.vectores.get("beginner_friendly", 0) >= 4


@dataclass
class CatalogValidationNote:
    """Cross-validation note between engine score and catalog vector for a criterion."""
    criterio: str
    servicio_principal: str
    score_motor: float      # Score 0–10 assigned by the engine
    score_catalogo: float   # Score 0–10 derived from catalog vector (raw × 2)
    divergencia: float      # |score_motor - score_catalogo|
    nota: str               # Explanation of why they differ (or confirmation they align)


@dataclass
class SustitucionServicio:
    """
    Proposed service substitution when a required service is unavailable.
    The architecture remains viable using the substitute instead.
    """
    servicio_requerido: str    # Original service the architecture normally needs
    servicio_sustituto: str    # Available service that can replace it
    impacto_costo: str         # Cost difference vs original service
    impacto_latencia: str      # Latency impact vs original service
    nota: str                  # What is gained/lost with this substitution


@dataclass
class ArchitectureScore:
    id: ArchitectureId
    nombre: str
    servicios_aws: list[str]
    score_total: float
    desglose: dict[str, float]
    razonamiento: dict[str, CriterionDetail]  # per-criterion documented reason
    descartada_por: str = ""   # Main reason this arch lost (empty = winner)
    posicion: int = 0
    bloqueada_por_servicio: bool = False      # True = requiere servicio no disponible y sin sustituto
    viable_con_sustitucion: bool = False      # True = bloqueada pero con sustituto disponible
    sustituciones_propuestas: list["SustitucionServicio"] = field(default_factory=list)


@dataclass
class Advertencia:
    tipo: str     # "restriccion_dura" | "tension" | "coherencia"
    mensaje: str


@dataclass
class PasoConfiguracion:
    """Un paso concreto para empezar a desplegar la arquitectura recomendada."""
    numero: int
    titulo: str
    descripcion: str
    comando: str     # CLI/código de ejemplo ejecutable


@dataclass
class AlternativaViable:
    """
    Arquitectura no óptima pero viable si la organización tiene restricciones
    que impiden usar la arquitectura recomendada (costo, políticas, servicios bloqueados).
    """
    id: ArchitectureId
    nombre: str
    score_total: float
    cuando_elegirla: str      # Condición organizacional que justifica esta elección
    trade_off: str            # Qué se gana y qué se pierde frente a la recomendada
    costo_estimado_usd: str   # Rango de costo mensual referencial


@dataclass
class RecommendationResult:
    recomendacion: ArchitectureScore
    ranking: list[ArchitectureScore]
    confianza: float
    justificacion: str
    criterio_decisivo: str              # Criterio que más diferenció al ganador
    razonamiento_completo: str          # Explicación párrafo completo
    advertencias: list[Advertencia]     # Restricciones o tensiones detectadas
    configuracion_inicial: list[PasoConfiguracion]  # Primeros pasos para arrancar
    alternativas_viables: list[AlternativaViable]   # Opciones si hay restricciones
    input_recibido: CaseInput
    # Catalog enrichment (populated from aws_service_catalog_complete.csv)
    servicios_detalle: list[CatalogServiceDetail] = field(default_factory=list)
    notas_validacion_catalogo: list[CatalogValidationNote] = field(default_factory=list)


# ─────────────────────────────────────────────
# KNOWLEDGE BASE — scores + documented reasons
# ─────────────────────────────────────────────
#
# Each function returns a CriterionDetail with the score AND the reason behind it.
# This is the knowledge base of the agent: every number is traceable.
#
# Score scale (0-10):
#   9-10  Excellent match — architecture is purpose-built for this need
#   7-8   Good match — works well with minor trade-offs
#   5-6   Acceptable — works but not ideal
#   3-4   Poor match — significant limitations for this need
#   1-2   Near-disqualifying — architecture fundamentally misaligned

def _kb_latencia(arch: ArchitectureId, ms: float) -> CriterionDetail:
    """
    Latency knowledge base.

    AWS reference latencies (warm):
      Lambda:           ~50–200ms warm, 300–1500ms cold start
      API Gateway:      adds ~10ms overhead
      Batch (Glue/EMR): minutes to hours (job scheduling)
      Kinesis:          ~200ms end-to-end (stream processing)
      ECS Fargate:      ~50–300ms (depends on service config)
      SageMaker EP:     ~30–150ms warm
    """
    baja = ms <= 1_000

    rules: dict[ArchitectureId, CriterionDetail] = {
        "serverless": CriterionDetail(
            score=9 if baja else 4,
            razon=(
                "Lambda warm: ~50–200ms; cold start mitigable con Provisioned Concurrency. "
                "Adecuado para SLA ≤ 1s."
                if baja else
                "Lambda puede responder en <200ms pero no se justifica su costo/complejidad "
                "si la latencia tolerada es alta (>1s). Batch sería más eficiente."
            ),
        ),
        "batch": CriterionDetail(
            score=1 if baja else 9,
            razon=(
                "Batch es inherentemente asíncrono (minutos a horas por job). "
                "Incompatible con SLA de tiempo real. ← RESTRICCIÓN DURA"
                if baja else
                "Diseñado para procesamiento diferido. Alta eficiencia cuando "
                "la latencia tolerada es de horas o días."
            ),
            es_restriccion=baja,
        ),
        "streaming": CriterionDetail(
            score=8 if baja else 6,
            razon=(
                "Kinesis + Lambda: ~200ms típico. Viable para tiempo real, "
                "aunque agrega overhead de stream frente a Lambda directa."
                if baja else
                "Adecuado para flujos continuos pero no optimizado para "
                "latencia baja sin necesidad de streaming."
            ),
        ),
        "containers": CriterionDetail(
            score=7,
            razon=(
                "ECS Fargate: ~50–300ms según configuración. Estable pero "
                "requiere warmup y configuración de auto-scaling."
            ),
        ),
        "sagemaker": CriterionDetail(
            score=8 if baja else 7,
            razon=(
                "SageMaker Real-time Endpoint: ~30–150ms warm. Buena latencia "
                "pero costo fijo elevado independiente del tráfico."
                if baja else
                "Latencia aceptable; para alta tolerancia, Async Inference "
                "reduce costos frente al endpoint en tiempo real."
            ),
        ),
    }
    return rules[arch]


def _kb_frecuencia(arch: ArchitectureId, freq: FrequencyLevel) -> CriterionDetail:
    """
    Inference frequency knowledge base.

    Patterns:
      baja (esporádica):   Lambda ideal — pay-per-request, $0 sin tráfico
      media (regular):     ECS o SageMaker — contenedores warm reducen latencia
      alta (frecuente):    Streaming o contenedores — evitan cold starts
      continua (24/7):     Kinesis + Lambda — diseñado para flujo sin pausa
    """
    rules: dict[FrequencyLevel, dict[ArchitectureId, CriterionDetail]] = {
        "baja": {
            "serverless": CriterionDetail(10, "Pay-per-request: $0 cuando no hay tráfico. "
                "Free tier AWS Lambda: 1M requests/mes. Ideal para uso esporádico."),
            "batch":      CriterionDetail(8,  "Baja frecuencia = alto intervalo entre jobs. "
                "Válido si la latencia no importa (ej: scoring diario de cartera)."),
            "streaming":  CriterionDetail(2,  "Montar infraestructura Kinesis para uso esporádico "
                "es costoso e innecesario. ← INEFICIENTE", es_restriccion=True),
            "containers": CriterionDetail(4,  "ECS Fargate cobra por hora aunque no haya tráfico. "
                "Con baja frecuencia el costo/request es muy alto."),
            "sagemaker":  CriterionDetail(5,  "SageMaker endpoint cobra ~$0.20/hr en idle. "
                "Para baja frecuencia, costo desproporcionado."),
        },
        "media": {
            "serverless": CriterionDetail(8,  "Funciona bien. Cold starts poco frecuentes. "
                "Provisioned Concurrency elimina el problema si se necesita."),
            "batch":      CriterionDetail(5,  "Media frecuencia con latencia media puede funcionar "
                "en batch si las solicitudes se agrupan."),
            "streaming":  CriterionDetail(7,  "Aceptable; el overhead de Kinesis se amortiza "
                "con volumen medio-alto de eventos."),
            "containers": CriterionDetail(8,  "ECS Fargate con auto-scaling: eficiente para "
                "tráfico regular y predecible."),
            "sagemaker":  CriterionDetail(8,  "Endpoint en tiempo real justificado con "
                "tráfico medio. Auto-scaling disponible."),
        },
        "alta": {
            "serverless": CriterionDetail(5,  "Lambda escala bien pero cold starts frecuentes "
                "si hay picos. Provisioned Concurrency costoso a escala."),
            "batch":      CriterionDetail(2,  "Alta frecuencia con expectativa de respuesta "
                "inmediata es incompatible con batch. ← RESTRICCIÓN DURA", es_restriccion=True),
            "streaming":  CriterionDetail(10, "Kinesis diseñado para alto volumen. "
                "Shards escalan horizontalmente sin límite práctico."),
            "containers": CriterionDetail(9,  "ECS con auto-scaling maneja picos predecibles. "
                "Latencia estable. Modelo en memoria siempre cargado."),
            "sagemaker":  CriterionDetail(9,  "Auto-scaling de instancias ML. Monitoreo nativo "
                "con CloudWatch. Ideal para alta frecuencia con SLA estricto."),
        },
        "continua": {
            "serverless": CriterionDetail(3,  "Lambda no está diseñada para procesamiento continuo. "
                "Timeout máx 15min. Costo por invocación acumula a escala."),
            "batch":      CriterionDetail(1,  "Batch procesa lotes, no flujo continuo. "
                "Fundamentalmente incompatible. ← RESTRICCIÓN DURA", es_restriccion=True),
            "streaming":  CriterionDetail(10, "Kinesis Data Streams: diseñado exactamente para "
                "procesamiento continuo de eventos en tiempo real."),
            "containers": CriterionDetail(9,  "Servicio siempre activo, modelo en memoria, "
                "sin límites de timeout. Apto para flujos continuos."),
            "sagemaker":  CriterionDetail(8,  "Real-time endpoint soporta tráfico continuo "
                "con auto-scaling, aunque a mayor costo que Kinesis."),
        },
    }
    return rules.get(freq, rules["media"])[arch]


def _kb_presupuesto(arch: ArchitectureId, usd: float) -> CriterionDetail:
    """
    Budget knowledge base — three tiers.

    muy_bajo (≤ $50):   startup / MVP — serverless is the only realistic option
    bajo ($51–$200):    pyme / growth — all options viable with trade-offs
    alto (> $200):      corporate — full platform options justified

    Reference minimum monthly costs (us-east-1):
      Lambda + API GW:   ~$0–1 (free tier covers most MVPs)
      Glue:              ~$0.50 (pay-per-DPU-hr)
      Kinesis:           ~$11/shard (always-on)
      ECS Fargate:       ~$15 (0.5 vCPU, always-on task)
      SageMaker ml.t3m:  ~$38 (always-on endpoint)
    """
    muy_bajo = usd <= 50
    bajo     = usd <= 200

    rules: dict[ArchitectureId, CriterionDetail] = {
        "serverless": CriterionDetail(
            score=10 if bajo else 8,
            razon=(
                f"Costo estimado: <$1/mes para cargas ligeras. "
                f"Free tier AWS cubre 1M requests/mes — ideal para presupuesto ${usd:.0f}/mes."
                if muy_bajo else
                "Costo estimado para 10K req/mes: <$1 USD. "
                "Free tier cubre demos y prototipos. Escala sin costo fijo."
                if bajo else
                "A alto volumen sigue siendo económico; "
                "API Gateway cobra $3.50/M requests. Considerar HTTP API ($1/M) como alternativa."
            ),
        ),
        "batch": CriterionDetail(
            score=9 if muy_bajo else (8 if bajo else 7),
            razon=(
                f"Glue cobra por DPU-hora solo cuando corre el job. "
                f"Para presupuesto ${usd:.0f}/mes es viable si los jobs son poco frecuentes."
                if muy_bajo else
                "Glue cobra por DPU-hora solo cuando corre el job. "
                "Con baja frecuencia el costo mensual es mínimo ($2–$5)."
                if bajo else
                "Jobs frecuentes o grandes aumentan costo de DPUs. "
                "Evaluar EMR para volúmenes muy altos."
            ),
        ),
        "streaming": CriterionDetail(
            score=2 if muy_bajo else (5 if bajo else 7),
            razon=(
                f"Kinesis: ~$11/mes/shard siempre activo. "
                f"Representa {round(11/usd*100):.0f}% de tu presupuesto de ${usd:.0f}/mes. ← EXCEDE BUDGET"
                if muy_bajo else
                "Kinesis: $0.015/shard-hr = ~$11/mes/shard siempre activo. "
                "Costoso para presupuesto bajo si el tráfico es esporádico."
                if bajo else
                "Kinesis amortiza su costo fijo con alto volumen de eventos. "
                "Aceptable con presupuesto medio-alto."
            ),
            es_restriccion=muy_bajo,
        ),
        "containers": CriterionDetail(
            score=2 if muy_bajo else (4 if bajo else 7),
            razon=(
                f"ECS Fargate mínimo: ~$15/mes. "
                f"Representa {round(15/usd*100):.0f}% de tu presupuesto de ${usd:.0f}/mes. ← EXCEDE BUDGET"
                if muy_bajo else
                "ECS Fargate mínimo: ~$15/mes por tarea siempre activa. "
                "Con presupuesto bajo representa fracción alta del budget."
                if bajo else
                "ECS Fargate con spot instances y auto-scaling. "
                "Eficiente a escala media-alta."
            ),
            es_restriccion=muy_bajo,
        ),
        "sagemaker": CriterionDetail(
            score=1 if muy_bajo else (2 if bajo else 6),
            razon=(
                f"SageMaker ml.t3.medium: ~$38/mes siempre activo. "
                f"Representa {round(38/usd*100):.0f}% de tu presupuesto de ${usd:.0f}/mes. ← INVIABLE"
                if muy_bajo else
                "SageMaker ml.t3.medium: ~$38/mes siempre activo. "
                "Con presupuesto <$200 representa fracción alta del budget. ← COSTOSO"
                if bajo else
                "Justificado con presupuesto alto por las capacidades MLOps "
                "incluidas (monitoreo, A/B testing, logging)."
            ),
            es_restriccion=(usd <= 200),
        ),
    }
    return rules[arch]


def _kb_escalabilidad(arch: ArchitectureId, escala: ScaleLevel) -> CriterionDetail:
    """
    Scalability knowledge base.

    baja:  tráfico predecible y controlado, no se esperan picos
    media: picos ocasionales, auto-scaling deseable
    alta:  picos imprevisibles o crecimiento acelerado esperado
    """
    rules: dict[ScaleLevel, dict[ArchitectureId, CriterionDetail]] = {
        "baja": {
            "serverless": CriterionDetail(9,  "Lambda escala automáticamente. Con baja escala "
                "no es necesaria configuración adicional."),
            "batch":      CriterionDetail(8,  "Jobs batch son naturalmente escalables "
                "al aumentar DPUs o nodos EMR."),
            "streaming":  CriterionDetail(5,  "Kinesis requiere gestión de shards. "
                "Innecesariamente complejo para baja escala."),
            "containers": CriterionDetail(6,  "ECS funciona pero el auto-scaling agrega "
                "complejidad innecesaria para baja escala."),
            "sagemaker":  CriterionDetail(6,  "Endpoint single-instance suficiente. "
                "Auto-scaling disponible pero no necesario."),
        },
        "media": {
            "serverless": CriterionDetail(9,  "Lambda escala de 0 a miles de instancias en "
                "segundos. Reserva concurrencia si hay SLA estricto."),
            "batch":      CriterionDetail(6,  "Escalable verticalmente (más DPUs) pero "
                "no diseñado para picos de demanda instantáneos."),
            "streaming":  CriterionDetail(8,  "Kinesis escala via shards. Resharding manual "
                "pero Enhanced Fan-Out automatiza distribución."),
            "containers": CriterionDetail(8,  "ECS + Application Auto Scaling gestiona "
                "picos previsibles eficientemente."),
            "sagemaker":  CriterionDetail(8,  "Auto-scaling por métrica (InvocationsPerInstance). "
                "Escala en 1-3 minutos."),
        },
        "alta": {
            "serverless": CriterionDetail(7,  "Lambda tiene límite de concurrencia por cuenta "
                "(default 1000). Suficiente para mayoría pero pedir aumento si es necesario."),
            "batch":      CriterionDetail(4,  "Escalabilidad batch es horizontal pero "
                "no responde a picos instantáneos de demanda."),
            "streaming":  CriterionDetail(9,  "Kinesis: escala a millones de eventos/s "
                "con múltiples shards. Diseñado para alta escala."),
            "containers": CriterionDetail(9,  "ECS Fargate + Spot + ALB: escala "
                "horizontal casi ilimitada con el patrón correcto."),
            "sagemaker":  CriterionDetail(9,  "Multi-model endpoints, auto-scaling avanzado, "
                "y soporte de GPU. Plataforma enterprise para alta escala."),
        },
    }
    return rules.get(escala, rules["media"])[arch]


def _kb_experiencia(arch: ArchitectureId, exp: ExperienceLevel) -> CriterionDetail:
    """
    Technical experience knowledge base.

    baja:  equipo sin experiencia en cloud/infraestructura
    media: equipo con experiencia básica en AWS, sin DevOps dedicado
    alta:  equipo con experiencia en cloud-native, IaC, CI/CD
    """
    rules: dict[ExperienceLevel, dict[ArchitectureId, CriterionDetail]] = {
        "baja": {
            "serverless": CriterionDetail(7,  "ZIP + boto3: despliegue sencillo desde consola "
                "o CLI. Sin gestión de servidores. Curva baja."),
            "batch":      CriterionDetail(5,  "AWS Glue Studio tiene UI visual pero "
                "requiere entender PySpark/ETL. Curva media."),
            "streaming":  CriterionDetail(3,  "Kinesis + Lambda + DynamoDB: arquitectura "
                "multi-servicio con conceptos de particiones y ordering. Complejo."),
            "containers": CriterionDetail(2,  "Docker + ECS + ALB + ECR + IAM roles: "
                "curva empinada sin experiencia en contenedores. ← NO RECOMENDADO"),
            "sagemaker":  CriterionDetail(6,  "SageMaker Studio tiene UI amigable para "
                "despliegue pero requiere entender instancias ML y configuración."),
        },
        "media": {
            "serverless": CriterionDetail(9,  "Patrón bien documentado con cientos de ejemplos. "
                "CLI + CloudFormation/SAM lo automatizan completamente."),
            "batch":      CriterionDetail(7,  "Glue o Step Functions son manejables con "
                "experiencia básica en AWS."),
            "streaming":  CriterionDetail(6,  "Kinesis Data Streams manejable si se conoce "
                "el patrón productor-consumidor. Documentación abundante."),
            "containers": CriterionDetail(6,  "ECS Fargate abstrae la gestión de EC2 pero "
                "requiere conocer Docker y networking en VPC."),
            "sagemaker":  CriterionDetail(7,  "SageMaker SDK + boto3 bien documentados. "
                "Despliegue viable con experiencia media en Python y AWS."),
        },
        "alta": {
            "serverless": CriterionDetail(8,  "Equipo puede optimizar con IaC (CDK/Terraform), "
                "capas Lambda, y Provisioned Concurrency."),
            "batch":      CriterionDetail(8,  "Equipo puede usar Step Functions + Glue + EMR "
                "con orquestación compleja y optimización de costos."),
            "streaming":  CriterionDetail(9,  "Equipo puede implementar exactly-once semantics, "
                "DLQ, y monitoreo avanzado en Kinesis."),
            "containers": CriterionDetail(9,  "Docker + ECS + blue/green deployments + "
                "Terraform. Máximo control sobre la infraestructura."),
            "sagemaker":  CriterionDetail(9,  "MLOps completo: pipelines, model registry, "
                "A/B testing, data drift monitoring. Plataforma enterprise."),
        },
    }
    return rules.get(exp, rules["media"])[arch]


# ─────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "latencia":      0.25,
    "frecuencia":    0.25,
    "presupuesto":   0.20,
    "escalabilidad": 0.15,
    "experiencia":   0.15,
}

# Known minimum monthly cost per architecture (us-east-1, minimal configuration).
# Used to detect hard budget violations before scoring.
_ARCH_MIN_COST_USD: dict[ArchitectureId, float] = {
    "serverless": 0.0,    # Pay-per-request — free tier covers demos and MVPs
    "batch":      0.5,    # Glue: ~$0.44/DPU-hr, minimal for small jobs
    "streaming":  11.0,   # Kinesis: ~$11/month per shard (always-on cost)
    "containers": 15.0,   # Fargate: ~$15/month for 0.5 vCPU task
    "sagemaker":  38.0,   # ml.t3.medium: ~$38/month always-on endpoint
}


def _adaptive_weights(case: "CaseInput") -> dict[str, float]:
    """
    Returns scoring weights tuned to the company's budget and experience profile.

    Base:           latencia 25% | frecuencia 25% | presupuesto 20% | escala 15% | exp 15%
    Budget ≤ $50:   presupuesto rises to 35% — cost becomes the dominant criterion
    Exp = baja:     experiencia rises to 25% — simplicity matters for new cloud teams
    Both:           combined adjustment, always normalized to sum = 1.0

    The adjustment is intentional: a company spending $30/month should not score
    the same as one spending $500/month when evaluating SageMaker ($38 min cost).
    """
    w = dict(WEIGHTS)

    if case.presupuesto_mensual_usd <= 50:
        w["presupuesto"]   = 0.35
        w["latencia"]      = 0.20
        w["frecuencia"]    = 0.20
        w["escalabilidad"] = 0.10
        # experiencia unchanged at 0.15

    if case.experiencia_tecnica == "baja":
        w["experiencia"]   = w["experiencia"] + 0.10
        w["latencia"]      = max(w["latencia"] - 0.05, 0.10)
        w["escalabilidad"] = max(w["escalabilidad"] - 0.05, 0.05)

    # Normalize so weights always sum to exactly 1.0
    total = sum(w.values())
    return {k: round(v / total, 4) for k, v in w.items()}


# ─────────────────────────────────────────────
# ARCHITECTURE METADATA
# ─────────────────────────────────────────────

ARCHITECTURES_META: dict[ArchitectureId, dict] = {
    "serverless": {
        "nombre": "Serverless (Lambda + API Gateway)",
        "servicios_aws": ["AWS Lambda", "API Gateway", "Amazon S3"],
        "descripcion": "Función sin servidor invocada por HTTP. Paga solo por cada request.",
    },
    "batch": {
        "nombre": "Batch Processing (S3 + Glue)",
        "servicios_aws": ["Amazon S3", "AWS Glue", "Amazon EventBridge"],
        "descripcion": "Procesamiento diferido de grandes volúmenes. Jobs programados o disparados.",
    },
    "streaming": {
        "nombre": "Streaming (Kinesis + Lambda)",
        "servicios_aws": ["Amazon Kinesis", "AWS Lambda", "Amazon DynamoDB"],
        "descripcion": "Procesamiento continuo de eventos en tiempo real sobre flujos de datos.",
    },
    "containers": {
        "nombre": "Containers (ECS Fargate + ALB)",
        "servicios_aws": ["Amazon ECS Fargate", "Application Load Balancer", "Amazon ECR"],
        "descripcion": "Microservicio contenerizado con control total sobre runtime y dependencias.",
    },
    "sagemaker": {
        "nombre": "SageMaker Managed Endpoint",
        "servicios_aws": ["Amazon SageMaker", "API Gateway", "Amazon CloudWatch"],
        "descripcion": "Plataforma MLOps administrada con endpoint de inferencia, monitoreo y A/B testing.",
    },
}


# ─────────────────────────────────────────────
# SERVICE AVAILABILITY FILTER
# ─────────────────────────────────────────────
#
# Maps each architecture to the minimum set of service keys it requires.
# Keys are normalized (lowercase, underscores). These are matched against
# CaseInput.servicios_disponibles after normalization.

_ARCH_REQUIRED_SERVICES: dict[ArchitectureId, list[str]] = {
    "serverless": ["AWS Lambda", "Amazon API Gateway", "Amazon S3"],
    "batch":      ["Amazon S3", "AWS Glue"],
    "streaming":  ["Amazon Kinesis", "AWS Lambda"],
    "containers": ["Amazon ECS"],
    "sagemaker":  ["Amazon SageMaker AI"],
}


def _normalize_service(s: str) -> str:
    """Normalizes a service name for case-insensitive comparison."""
    return s.strip().lower()


# ─────────────────────────────────────────────
# SERVICE SUBSTITUTIONS
# ─────────────────────────────────────────────
#
# When a required service is unavailable, the agent tries substitutes in order.
# First substitute whose name appears in servicios_disponibles wins.
# Each entry: sustituto, impacto_costo, impacto_latencia, nota.

_SERVICE_SUBSTITUTIONS: dict[str, list[dict]] = {
    "Amazon Kinesis": [
        {
            "sustituto": "Amazon MSK",
            "impacto_costo": "Mayor (~$0.21/hr por broker vs $0.015/shard-hr Kinesis)",
            "impacto_latencia": "Equivalente (~200ms extremo a extremo)",
            "nota": (
                "Apache Kafka administrado. Misma semántica de streaming con mayor "
                "ecosistema de conectores. Recomendado si la org ya usa Kafka."
            ),
        },
        {
            "sustituto": "Amazon EventBridge",
            "impacto_costo": "Menor ($1/M eventos, sin costo fijo por shard)",
            "impacto_latencia": "Similar para eventos individuales; no optimizado para alto volumen continuo",
            "nota": (
                "Válido para patrones event-driven con volumen bajo-medio (<10K eventos/min). "
                "No reemplaza shards de Kinesis para flujos de alta frecuencia."
            ),
        },
        {
            "sustituto": "AWS Step Functions",
            "impacto_costo": "Menor para flujos orquestados ($0.025/1K transiciones)",
            "impacto_latencia": "Mayor latencia por step (~100ms overhead por estado)",
            "nota": (
                "Alternativa para pipelines de inferencia secuenciales, no para "
                "streaming puro. Útil cuando se necesita orquestación con reintentos."
            ),
        },
    ],
    "Amazon SageMaker AI": [
        {
            "sustituto": "Amazon ECS",
            "impacto_costo": "Menor (~$15–80/mes Fargate vs $38+/mes SageMaker ml.t3.medium)",
            "impacto_latencia": "Equivalente (~50–300ms con contenedor warm)",
            "nota": (
                "Contenedor propio con FastAPI/Flask sirviendo el modelo. "
                "Pierde: model registry, data drift monitoring y A/B testing nativos de SageMaker."
            ),
        },
        {
            "sustituto": "AWS Fargate",
            "impacto_costo": "Menor (~$15–50/mes sin gestionar clusters)",
            "impacto_latencia": "Equivalente",
            "nota": (
                "Contenedor serverless sin infraestructura EC2. Buena alternativa de bajo costo "
                "para servir modelos sklearn/XGBoost con una API REST custom."
            ),
        },
        {
            "sustituto": "AWS Lambda",
            "impacto_costo": "Muy bajo (<$5/mes para cargas ligeras, free tier generoso)",
            "impacto_latencia": "Aceptable warm (50–200ms); cold start 300–1500ms mitigable",
            "nota": (
                "Válido si el modelo cabe en 512MB RAM y la inferencia tarda <15min. "
                "Ideal para modelos sklearn/XGBoost ligeros con pocas dependencias."
            ),
        },
    ],
    "Amazon ECS": [
        {
            "sustituto": "AWS Fargate",
            "impacto_costo": "Equivalente — Fargate es el motor de cómputo de ECS",
            "impacto_latencia": "Equivalente",
            "nota": (
                "Fargate abstrae la gestión de instancias EC2 subyacentes de ECS. "
                "Para despliegue de modelos ML es prácticamente intercambiable con ECS."
            ),
        },
        {
            "sustituto": "AWS App Runner",
            "impacto_costo": "Similar (~$0.064/vCPU-hr, sin configuración de VPC)",
            "impacto_latencia": "Equivalente",
            "nota": (
                "Despliegue de contenedores más simple que ECS: sin clusters, VPCs ni "
                "task definitions. Ideal para equipos sin experiencia en orquestación."
            ),
        },
        {
            "sustituto": "AWS Lambda",
            "impacto_costo": "Menor (pay-per-request, $0 sin tráfico)",
            "impacto_latencia": "Aceptable para modelos ligeros; cold start si hay inactividad",
            "nota": (
                "Alternativa válida si el modelo cabe en Lambda (<512MB descomprimido). "
                "Pierde: modelo siempre en memoria, control total del runtime."
            ),
        },
    ],
    "Amazon API Gateway": [
        {
            "sustituto": "AWS App Runner",
            "impacto_costo": "Similar; incluye endpoint HTTPS sin costo adicional",
            "impacto_latencia": "Equivalente",
            "nota": (
                "App Runner expone automáticamente un endpoint HTTPS para el contenedor, "
                "eliminando la necesidad de API Gateway en arquitecturas de contenedores."
            ),
        },
        {
            "sustituto": "Amazon CloudFront",
            "impacto_costo": "Menor para alto volumen ($0.0085/10K requests)",
            "impacto_latencia": "Menor (CDN con PoP globales)",
            "nota": (
                "CloudFront puede enrutar al Lambda o ECS origin directamente. "
                "Agrega caché y protección DDoS; requiere más configuración que API Gateway."
            ),
        },
    ],
    "AWS Glue": [
        {
            "sustituto": "AWS Lambda",
            "impacto_costo": "Menor para jobs cortos (<15min); free tier 1M requests/mes",
            "impacto_latencia": "Equivalente para ETL ligero sin PySpark",
            "nota": (
                "Lambda reemplaza Glue para transformaciones simples en Python puro. "
                "Límite: 15min de ejecución, 10GB /tmp. Sin soporte nativo de PySpark."
            ),
        },
        {
            "sustituto": "Amazon EMR",
            "impacto_costo": "Mayor para volúmenes bajos; más eficiente para datasets grandes",
            "impacto_latencia": "Similar o mejor para datasets > 100GB",
            "nota": (
                "Cluster Spark/Hadoop administrado. Más potente que Glue para grandes volúmenes "
                "pero requiere más configuración y experiencia técnica."
            ),
        },
        {
            "sustituto": "AWS Step Functions",
            "impacto_costo": "Menor para ETL orquestado ($0.025/1K transiciones de estado)",
            "impacto_latencia": "Overhead de orquestación (~100ms por estado)",
            "nota": (
                "Step Functions orquesta Lambdas en pipelines ETL complejos con reintentos, "
                "paralelismo y manejo de errores. Alternativa liviana a Glue para ETL sin Spark."
            ),
        },
    ],
}


def _find_substitution(
    required_service: str, available: set[str]
) -> dict | None:
    """Returns the first available substitute for required_service, or None."""
    for sub in _SERVICE_SUBSTITUTIONS.get(required_service, []):
        if _normalize_service(sub["sustituto"]) in available:
            return sub
    return None


# ─────────────────────────────────────────────
# INTERACTIVE SERVICE ANALYSIS
# ─────────────────────────────────────────────


@dataclass
class ServiceAnalysisResult:
    """Result of analyzing whether an architecture can be deployed with the company's services."""
    arch: ArchitectureScore
    servicios_ok: list[str]                                 # Required services the company has
    sustituciones: list[SustitucionServicio]                # Missing but substitutable
    faltantes: list[str]                                    # Missing with no substitute
    es_viable: bool                                         # True when faltantes is empty
    configuracion_inicial: list[PasoConfiguracion]          # Quickstart for chosen arch
    alternativa_precio: ArchitectureScore | None            # Cheapest viable if not viable
    alternativa_precio_sustituciones: list[SustitucionServicio] = field(default_factory=list)
    alternativa_precio_config: list[PasoConfiguracion] = field(default_factory=list)


def get_relevant_services_for_arch(arch_id: ArchitectureId) -> list[dict]:
    """
    Returns required services + their substitutes for the given architecture.
    Each entry has keys: service (str), tipo ("requerido"|"sustituto"), sustituye (str|None).
    Used in interactive mode so the user can indicate which services their company has.
    """
    required = _ARCH_REQUIRED_SERVICES.get(arch_id, [])
    result: list[dict] = []
    seen: set[str] = set()
    for svc in required:
        if svc not in seen:
            result.append({"service": svc, "tipo": "requerido", "sustituye": None})
            seen.add(svc)
        for sub in _SERVICE_SUBSTITUTIONS.get(svc, []):
            sub_name = sub["sustituto"]
            if sub_name not in seen:
                result.append({"service": sub_name, "tipo": "sustituto", "sustituye": svc})
                seen.add(sub_name)
    return result


def analyze_services_for_arch(
    arch_id: ArchitectureId,
    servicios_disponibles: list[str],
    ranking: list[ArchitectureScore],
) -> ServiceAnalysisResult:
    """
    Given a chosen architecture and the company's available services, determines:
    - which required services are present (servicios_ok)
    - which missing services have a viable substitute (sustituciones)
    - which missing services have no substitute (faltantes)

    If the architecture is not viable (faltantes not empty), finds the cheapest
    architecture from the ranking that the company can actually deploy.
    Cheapest = lowest _ARCH_MIN_COST_USD; ties broken by highest score.
    """
    available = {_normalize_service(s) for s in servicios_disponibles}
    required = _ARCH_REQUIRED_SERVICES.get(arch_id, [])
    arch = next(a for a in ranking if a.id == arch_id)

    servicios_ok: list[str] = []
    sustituciones: list[SustitucionServicio] = []
    faltantes: list[str] = []

    for svc in required:
        if _normalize_service(svc) in available:
            servicios_ok.append(svc)
        else:
            sub = _find_substitution(svc, available)
            if sub:
                sustituciones.append(SustitucionServicio(
                    servicio_requerido=svc,
                    servicio_sustituto=sub["sustituto"],
                    impacto_costo=sub["impacto_costo"],
                    impacto_latencia=sub["impacto_latencia"],
                    nota=sub["nota"],
                ))
            else:
                faltantes.append(svc)

    es_viable = len(faltantes) == 0
    configuracion_inicial = _build_configuracion(arch_id)

    alternativa_precio: ArchitectureScore | None = None
    alt_sustituciones: list[SustitucionServicio] = []
    alt_config: list[PasoConfiguracion] = []

    if not es_viable:
        viables: list[ArchitectureScore] = []
        for a in ranking:
            if a.id == arch_id:
                continue
            req_a = _ARCH_REQUIRED_SERVICES.get(a.id, [])
            missing_a = [s for s in req_a if _normalize_service(s) not in available]
            unresolved_a = [s for s in missing_a if _find_substitution(s, available) is None]
            if not unresolved_a:
                viables.append(a)

        if viables:
            alternativa_precio = min(
                viables,
                key=lambda a: (_ARCH_MIN_COST_USD.get(a.id, 999), -a.score_total),
            )
            req_alt = _ARCH_REQUIRED_SERVICES.get(alternativa_precio.id, [])
            for svc in req_alt:
                if _normalize_service(svc) not in available:
                    sub = _find_substitution(svc, available)
                    if sub:
                        alt_sustituciones.append(SustitucionServicio(
                            servicio_requerido=svc,
                            servicio_sustituto=sub["sustituto"],
                            impacto_costo=sub["impacto_costo"],
                            impacto_latencia=sub["impacto_latencia"],
                            nota=sub["nota"],
                        ))
            alt_config = _build_configuracion(alternativa_precio.id)

    return ServiceAnalysisResult(
        arch=arch,
        servicios_ok=servicios_ok,
        sustituciones=sustituciones,
        faltantes=faltantes,
        es_viable=es_viable,
        configuracion_inicial=configuracion_inicial,
        alternativa_precio=alternativa_precio,
        alternativa_precio_sustituciones=alt_sustituciones,
        alternativa_precio_config=alt_config,
    )


# ─────────────────────────────────────────────
# AWS SERVICE CATALOG — loaded once at import
# ─────────────────────────────────────────────
#
# Maps each architecture pattern to the specific services in the catalog.
# Service names must match the "service" column in aws_service_catalog_complete.csv.

_ARCH_CATALOG_SERVICES: dict[ArchitectureId, list[str]] = {
    "serverless": ["AWS Lambda", "Amazon API Gateway", "Amazon S3"],
    "batch":      ["Amazon S3", "AWS Glue", "Amazon EventBridge"],
    "streaming":  ["Amazon Kinesis", "AWS Lambda", "Amazon DynamoDB"],
    "containers": ["Amazon ECS", "AWS Fargate", "Amazon CloudWatch"],
    "sagemaker":  ["Amazon SageMaker AI", "Amazon CloudWatch", "Amazon S3"],
}

# Primary service for each architecture — used for cross-validation
_ARCH_PRIMARY_SERVICE: dict[ArchitectureId, str] = {
    "serverless": "AWS Lambda",
    "batch":      "AWS Glue",
    "streaming":  "Amazon Kinesis",
    "containers": "Amazon ECS",
    "sagemaker":  "Amazon SageMaker AI",
}

# Engine criterion → catalog vector name (for cross-validation, scale 0–5)
# "frecuencia" has no direct catalog vector — excluded intentionally
_CRITERION_TO_VECTOR: dict[str, str] = {
    "latencia":      "low_latency",
    "presupuesto":   "cost_efficiency",
    "escalabilidad": "scalability",
    "experiencia":   "beginner_friendly",
}

# Most relevant service per criterion per architecture
# (used to pick which catalog entry's constraints text applies)
_CRITERION_SERVICE_MAP: dict[str, dict[ArchitectureId, str]] = {
    "latencia":      {k: _ARCH_PRIMARY_SERVICE[k] for k in _ARCH_PRIMARY_SERVICE},
    "frecuencia":    {k: _ARCH_PRIMARY_SERVICE[k] for k in _ARCH_PRIMARY_SERVICE},
    "presupuesto":   {
        "serverless": "AWS Lambda",
        "batch":      "AWS Glue",
        "streaming":  "Amazon Kinesis",
        "containers": "AWS Fargate",
        "sagemaker":  "Amazon SageMaker AI",
    },
    "escalabilidad": {k: _ARCH_PRIMARY_SERVICE[k] for k in _ARCH_PRIMARY_SERVICE},
    "experiencia":   {
        "serverless": "AWS Lambda",
        "batch":      "AWS Glue",
        "streaming":  "Amazon Kinesis",
        "containers": "AWS Fargate",
        "sagemaker":  "Amazon SageMaker AI",
    },
}


def _load_catalog() -> dict[str, "CatalogServiceDetail"]:
    """Loads aws_service_catalog_complete.csv from the same directory as this module."""
    catalog_path = Path(__file__).parent / "aws_service_catalog_complete.csv"
    if not catalog_path.exists():
        return {}
    result: dict[str, CatalogServiceDetail] = {}
    with open(catalog_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vectores = {
                    "cost_efficiency":   int(row["cost_efficiency"]),
                    "simplicity":        int(row["simplicity"]),
                    "scalability":       int(row["scalability"]),
                    "low_latency":       int(row["low_latency"]),
                    "managed_level":     int(row["managed_level"]),
                    "beginner_friendly": int(row["beginner_friendly"]),
                }
            except (KeyError, ValueError):
                continue
            name = row["service"].strip()
            result[name] = CatalogServiceDetail(
                service=name,
                category=row.get("category", "").strip(),
                description=row.get("description", "").strip(),
                benefits=row.get("benefits", "").strip(),
                constraints=row.get("constraints", "").strip(),
                use_cases=row.get("use_cases", "").strip(),
                official_url=row.get("official_url", "").strip(),
                vectores=vectores,
            )
    return result


_CATALOG: dict[str, CatalogServiceDetail] = _load_catalog()


# ─────────────────────────────────────────────
# HARD CONSTRAINT DETECTION
# ─────────────────────────────────────────────

def _detect_advertencias(case: CaseInput, ranking: list[ArchitectureScore]) -> list[Advertencia]:
    """
    Detects tensions and hard constraints in the input that the agent
    should surface to the user, regardless of which architecture wins.
    """
    warns: list[Advertencia] = []

    # Hard constraint: batch + real-time SLA
    if case.latencia_requerida_ms <= 1_000 and case.frecuencia_inferencia in ("alta", "continua"):
        warns.append(Advertencia(
            tipo="restriccion_dura",
            mensaje=(
                f"Latencia ≤{case.latencia_requerida_ms}ms + frecuencia '{case.frecuencia_inferencia}': "
                "batch queda descartado. La arquitectura debe ser reactiva (serverless o streaming)."
            ),
        ))

    # Hard constraint: continuous frequency + low budget
    if case.frecuencia_inferencia == "continua" and case.presupuesto_mensual_usd < 50:
        warns.append(Advertencia(
            tipo="tension",
            mensaje=(
                f"Frecuencia 'continua' con presupuesto ${case.presupuesto_mensual_usd}/mes "
                "es una tensión real: Kinesis cobra ~$11/mes/shard siempre activo. "
                "Valida si el volumen justifica el costo."
            ),
        ))

    # Tension: high scalability + low budget
    if case.escalabilidad_requerida == "alta" and case.presupuesto_mensual_usd < 50:
        warns.append(Advertencia(
            tipo="tension",
            mensaje=(
                f"Escalabilidad 'alta' + presupuesto ${case.presupuesto_mensual_usd}/mes: "
                "las arquitecturas que mejor escalan (ECS, SageMaker) tienen costo fijo alto. "
                "Lambda es la opción más económica con escala automática."
            ),
        ))

    # Coherence: low experience + streaming/containers recommended
    top_id = ranking[0].id if ranking else None
    if case.experiencia_tecnica == "baja" and top_id in ("streaming", "containers"):
        warns.append(Advertencia(
            tipo="coherencia",
            mensaje=(
                f"La arquitectura recomendada ('{top_id}') requiere experiencia técnica media-alta. "
                "Con experiencia 'baja', considera aumentar el score de 'experiencia' como criterio "
                "prioritario, o planifica capacitación del equipo."
            ),
        ))

    # Tension: very small payload + streaming overhead
    if case.volumen_datos_kb < 1 and case.frecuencia_inferencia in ("alta", "continua"):
        warns.append(Advertencia(
            tipo="coherencia",
            mensaje=(
                f"Payloads muy pequeños ({case.volumen_datos_kb} KB) con alta frecuencia: "
                "el overhead de Kinesis por evento puede superar el payload útil. "
                "Considera batching de micro-solicitudes."
            ),
        ))

    # Hard budget constraint — flag every architecture whose minimum cost exceeds the budget
    if case.presupuesto_mensual_usd > 0:
        for arch_id, min_cost in _ARCH_MIN_COST_USD.items():
            if min_cost > case.presupuesto_mensual_usd:
                arch_name = ARCHITECTURES_META[arch_id]["nombre"]
                pct = round(min_cost / case.presupuesto_mensual_usd * 100)
                warns.append(Advertencia(
                    tipo="restriccion_presupuestal",
                    mensaje=(
                        f"'{arch_name}' costo mínimo ~${min_cost}/mes "
                        f"({pct}% de tu presupuesto de ${case.presupuesto_mensual_usd:.0f}/mes). "
                        "Requiere presupuesto mayor para ser viable."
                    ),
                ))

    # Budget profile advisory for very low budgets
    if case.presupuesto_mensual_usd <= 50:
        warns.append(Advertencia(
            tipo="perfil_presupuestal",
            mensaje=(
                f"Presupuesto ${case.presupuesto_mensual_usd:.0f}/mes — perfil startup/MVP detectado. "
                "El motor priorizará arquitecturas de bajo costo: Serverless (<$1/mes) y Batch (~$0.50/mes). "
                "SageMaker ($38/mes) y ECS ($15/mes) quedan fuera del rango presupuestal."
            ),
        ))

    return warns


# ─────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────

def _build_razonamiento(
    top: ArchitectureScore,
    second: ArchitectureScore,
    ranking: list[ArchitectureScore],
    case: CaseInput,
    weights: dict[str, float],
) -> tuple[str, str]:
    """
    Returns (criterio_decisivo, razonamiento_completo) — human-readable explanation.
    Receives the effective weights used for scoring (may differ from WEIGHTS when
    adaptive budget/experience adjustments were applied).
    """
    # Find which criterion contributed most to the gap between 1st and 2nd
    max_gap_criterion = max(
        weights.keys(),
        key=lambda k: (top.desglose[k] - second.desglose[k]) * weights[k],
    )

    weight_pct = round(weights[max_gap_criterion] * 100)
    top_detail = top.razonamiento[max_gap_criterion]

    criterio_decisivo = (
        f"{max_gap_criterion.capitalize()} (peso {weight_pct}%): "
        f"{top.nombre} obtuvo {top.desglose[max_gap_criterion]}/10 "
        f"vs {second.nombre} con {second.desglose[max_gap_criterion]}/10."
    )

    # Find criteria where each discarded arch is particularly weak
    descartadas_razones = []
    for arch in ranking[1:]:
        criterio_debil = min(arch.desglose, key=lambda k: arch.desglose[k] * weights[k])
        descartadas_razones.append(
            f"• {arch.nombre} (#{arch.posicion}, score {arch.score_total}): "
            f"{arch.razonamiento[criterio_debil].razon}"
        )

    # Note when adaptive weights were applied
    using_base = all(abs(weights[k] - WEIGHTS[k]) < 0.001 for k in WEIGHTS)
    pesos_nota = "" if using_base else (
        "\n[Pesos ajustados por perfil presupuestal/experiencia: "
        + ", ".join(f"{k} {round(v*100)}%" for k, v in weights.items()) + "]"
    )

    razonamiento = (
        f"RECOMENDACIÓN: {top.nombre} (score {top.score_total}/10, confianza sobre el 2° lugar).\n\n"
        f"CRITERIO DECISIVO — {criterio_decisivo}\n"
        f"  Razón: {top_detail.razon}\n\n"
        f"FORTALEZAS DEL GANADOR:\n"
        + "\n".join(
            f"  • {k.capitalize()} ({round(weights[k]*100)}%): "
            f"{top.razonamiento[k].score}/10 — {top.razonamiento[k].razon}"
            for k in weights
        )
        + "\n\nPOR QUÉ SE DESCARTARON LAS ALTERNATIVAS:\n"
        + "\n".join(descartadas_razones)
        + f"\n\nCASO DE USO: {case.descripcion}"
        + pesos_nota
    )

    return criterio_decisivo, razonamiento


# ─────────────────────────────────────────────
# CONFIGURATION QUICKSTART GENERATOR
# ─────────────────────────────────────────────

_CONFIG_STEPS: dict[ArchitectureId, list[dict]] = {
    "serverless": [
        {
            "titulo": "Subir modelo a S3",
            "descripcion": "Crea el bucket y sube el artefacto .joblib serializado con joblib.dump().",
            "comando": (
                "aws s3 mb s3://mi-modelo-bucket --region us-east-1\n"
                "aws s3 cp data/models/credit_model.joblib "
                "s3://mi-modelo-bucket/models/credit_model.joblib"
            ),
        },
        {
            "titulo": "Empaquetar y crear la función Lambda",
            "descripcion": "Instala dependencias en el zip y crea la función con el rol IAM.",
            "comando": (
                "bash deploy/build_lambda.sh   # genera build/credit_inference.zip\n"
                "aws lambda create-function \\\n"
                "  --function-name credit-risk-inference \\\n"
                "  --runtime python3.11 \\\n"
                "  --handler lambda_function.lambda_handler \\\n"
                "  --role arn:aws:iam::ACCOUNT_ID:role/LambdaMLRole \\\n"
                "  --zip-file fileb://build/credit_inference.zip \\\n"
                "  --environment Variables={MODEL_BUCKET=mi-modelo-bucket,"
                "MODEL_KEY=models/credit_model.joblib}"
            ),
        },
        {
            "titulo": "Crear API Gateway y exponer el endpoint",
            "descripcion": "Crea la API REST, el recurso /predict y lo conecta al Lambda.",
            "comando": (
                "bash deploy/deploy.sh\n"
                "# Al terminar muestra el endpoint público:\n"
                "# https://XXXXXX.execute-api.us-east-1.amazonaws.com/prod/predict"
            ),
        },
        {
            "titulo": "Probar el endpoint",
            "descripcion": "Verifica la inferencia con un ejemplo de bajo y alto riesgo.",
            "comando": (
                'curl -X POST https://TU_ENDPOINT/prod/predict \\\n'
                '  -H "Content-Type: application/json" \\\n'
                '  -d \'{"edad":35,"ingresos_mensuales":3500000,'
                '"ratio_deuda":0.35,"meses_historial_crediticio":24,"num_cuentas_activas":3}\''
            ),
        },
    ],
    "batch": [
        {
            "titulo": "Subir datos de entrada a S3",
            "descripcion": "Organiza los datos a procesar en un CSV o Parquet en S3.",
            "comando": (
                "aws s3 mb s3://mi-batch-bucket\n"
                "aws s3 cp datos_solicitudes.csv s3://mi-batch-bucket/input/solicitudes.csv"
            ),
        },
        {
            "titulo": "Crear Glue Job con el script de inferencia",
            "descripcion": "El job carga el modelo de S3, procesa el CSV y guarda resultados.",
            "comando": (
                "# En AWS Glue Console:\n"
                "# Glue → Jobs → Add Job → Python Shell\n"
                "# Script: carga model.joblib, lee CSV con pandas,\n"
                "#         ejecuta predict_proba(), guarda resultados en S3/output/"
            ),
        },
        {
            "titulo": "Programar ejecución con EventBridge",
            "descripcion": "Dispara el job automáticamente cada noche o en el horario requerido.",
            "comando": (
                "# EventBridge Console → Rules → Create Rule\n"
                "# Schedule: cron(0 2 * * ? *)   ← todos los días a las 2am UTC\n"
                "# Target: Glue Job → nombre del job creado en paso 2"
            ),
        },
        {
            "titulo": "Consultar resultados",
            "descripcion": "Los resultados quedan en S3 con la clasificación de cada solicitud.",
            "comando": (
                "aws s3 ls s3://mi-batch-bucket/output/\n"
                "aws s3 cp s3://mi-batch-bucket/output/resultados.csv ."
            ),
        },
    ],
    "streaming": [
        {
            "titulo": "Crear Kinesis Data Stream",
            "descripcion": "Crea el stream donde llegarán los eventos en tiempo real.",
            "comando": (
                "aws kinesis create-stream \\\n"
                "  --stream-name modelo-inference-stream \\\n"
                "  --shard-count 1 \\\n"
                "  --region us-east-1"
            ),
        },
        {
            "titulo": "Crear Lambda consumidor del stream",
            "descripcion": "Lambda se activa automáticamente con cada registro en Kinesis.",
            "comando": (
                "aws lambda create-event-source-mapping \\\n"
                "  --function-name credit-risk-inference \\\n"
                "  --event-source-arn arn:aws:kinesis:us-east-1:ACCOUNT:stream/modelo-inference-stream \\\n"
                "  --starting-position LATEST \\\n"
                "  --batch-size 10"
            ),
        },
        {
            "titulo": "Enviar eventos de inferencia al stream",
            "descripcion": "El productor (app/servicio) publica eventos en el stream.",
            "comando": (
                "aws kinesis put-record \\\n"
                '  --stream-name modelo-inference-stream \\\n'
                '  --data \'{"edad":35,"ingresos_mensuales":3500000,'
                '"ratio_deuda":0.35,"meses_historial_crediticio":24,"num_cuentas_activas":3}\' \\\n'
                "  --partition-key user-session-001"
            ),
        },
        {
            "titulo": "Monitorear métricas del stream",
            "descripcion": "Verifica latencia y throughput desde CloudWatch.",
            "comando": (
                "# CloudWatch → Metrics → Kinesis → GetRecords.IteratorAgeMilliseconds\n"
                "# Alerta si iterator age > 1000ms indica que Lambda no procesa a tiempo"
            ),
        },
    ],
    "containers": [
        {
            "titulo": "Crear imagen Docker con el modelo",
            "descripcion": "Empaqueta el modelo y la API de inferencia en un contenedor.",
            "comando": (
                "# Dockerfile:\n"
                "# FROM python:3.11-slim\n"
                "# RUN pip install scikit-learn joblib boto3 fastapi uvicorn\n"
                "# COPY lambda_function/handler.py app.py\n"
                "# CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]\n\n"
                "docker build -t credit-inference .\n"
                "docker run -p 8080:8080 credit-inference   # prueba local"
            ),
        },
        {
            "titulo": "Subir imagen a Amazon ECR",
            "descripcion": "Registra la imagen en el repositorio privado de AWS.",
            "comando": (
                "aws ecr create-repository --repository-name credit-inference\n"
                "aws ecr get-login-password | docker login --username AWS \\\n"
                "  --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com\n"
                "docker tag credit-inference ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/credit-inference\n"
                "docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/credit-inference"
            ),
        },
        {
            "titulo": "Crear servicio ECS Fargate + ALB",
            "descripcion": "Despliega el contenedor con balanceador de carga público.",
            "comando": (
                "# ECS Console → Create Cluster → Networking only (Fargate)\n"
                "# Task Definition → contenedor: imagen ECR, puerto 8080, 512MB RAM\n"
                "# Service → ALB → Target Group → puerto 8080\n"
                "# Auto Scaling: CPU > 70% → agregar tarea"
            ),
        },
        {
            "titulo": "Probar el endpoint del ALB",
            "descripcion": "El ALB expone una URL pública con health check automático.",
            "comando": (
                'curl -X POST http://MI-ALB.us-east-1.elb.amazonaws.com/predict \\\n'
                '  -H "Content-Type: application/json" \\\n'
                '  -d \'{"edad":35,"ingresos_mensuales":3500000,'
                '"ratio_deuda":0.35,"meses_historial_crediticio":24,"num_cuentas_activas":3}\''
            ),
        },
    ],
    "sagemaker": [
        {
            "titulo": "Empaquetar modelo en formato SageMaker (tar.gz)",
            "descripcion": "SageMaker requiere el artefacto comprimido con estructura específica.",
            "comando": (
                "mkdir model_pkg && cp data/models/credit_model.joblib model_pkg/model.joblib\n"
                "tar -czf model.tar.gz -C model_pkg .\n"
                "aws s3 cp model.tar.gz s3://mi-sagemaker-bucket/models/"
            ),
        },
        {
            "titulo": "Registrar modelo en SageMaker",
            "descripcion": "Crea el objeto Model apuntando al artefacto en S3.",
            "comando": (
                "import boto3\n"
                "sm = boto3.client('sagemaker')\n"
                "sm.create_model(\n"
                "  ModelName='credit-risk-v1',\n"
                "  PrimaryContainer={\n"
                "    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1',\n"
                "    'ModelDataUrl': 's3://mi-sagemaker-bucket/models/model.tar.gz'\n"
                "  },\n"
                "  ExecutionRoleArn='arn:aws:iam::ACCOUNT:role/SageMakerRole'\n"
                ")"
            ),
        },
        {
            "titulo": "Crear endpoint de inferencia",
            "descripcion": "Despliega el modelo en una instancia administrada por SageMaker.",
            "comando": (
                "sm.create_endpoint_config(\n"
                "  EndpointConfigName='credit-config',\n"
                "  ProductionVariants=[{\n"
                "    'ModelName': 'credit-risk-v1',\n"
                "    'InstanceType': 'ml.t3.medium',\n"
                "    'InitialInstanceCount': 1\n"
                "  }]\n"
                ")\n"
                "sm.create_endpoint(EndpointName='credit-endpoint', EndpointConfigName='credit-config')"
            ),
        },
        {
            "titulo": "Invocar el endpoint",
            "descripcion": "Llama al endpoint via SDK o API Gateway (SageMaker Runtime).",
            "comando": (
                "import json\n"
                "runtime = boto3.client('sagemaker-runtime')\n"
                "payload = json.dumps([35, 3500000, 0.35, 24, 3])   # mismo orden que FEATURE_ORDER\n"
                "resp = runtime.invoke_endpoint(\n"
                "  EndpointName='credit-endpoint',\n"
                "  ContentType='application/json',\n"
                "  Body=payload\n"
                ")"
            ),
        },
    ],
}


def _build_configuracion(arch_id: ArchitectureId) -> list[PasoConfiguracion]:
    """Returns the quickstart configuration steps for the given architecture."""
    steps = _CONFIG_STEPS.get(arch_id, [])
    return [
        PasoConfiguracion(
            numero=i + 1,
            titulo=s["titulo"],
            descripcion=s["descripcion"],
            comando=s["comando"],
        )
        for i, s in enumerate(steps)
    ]


# ─────────────────────────────────────────────
# VIABLE ALTERNATIVES GENERATOR
# ─────────────────────────────────────────────

_ALTERNATIVAS_META: dict[ArchitectureId, dict] = {
    "serverless": {
        "cuando_elegirla": (
            "Si tu organización ya tiene experiencia con Lambda y prefiere evitar la "
            "complejidad de contenedores. También cuando el presupuesto es muy bajo "
            "y el tráfico es esporádico (< 1M requests/mes)."
        ),
        "trade_off": "Ganas: $0 sin tráfico, despliegue simple. Pierdes: cold start 300-1500ms, "
                     "límite 15min de ejecución, dependencias pesadas requieren Lambda Layer.",
        "costo_estimado_usd": "< $5 / mes para cargas ligeras",
    },
    "batch": {
        "cuando_elegirla": (
            "Si la empresa no necesita respuesta en tiempo real (scoring diario/semanal), "
            "ya tiene datos en S3, o tiene restricciones para exponer APIs públicas "
            "(entornos regulados, finanzas, salud)."
        ),
        "trade_off": "Ganas: muy bajo costo, procesa millones de registros fácilmente. "
                     "Pierdes: no hay respuesta inmediata (horas de latencia).",
        "costo_estimado_usd": "$2–$20 / mes según volumen de datos",
    },
    "streaming": {
        "cuando_elegirla": (
            "Si la empresa ya usa Kafka o Kinesis para otros flujos de datos y quiere "
            "integrar el modelo ML en el mismo pipeline. O cuando el volumen de eventos "
            "es continuo y muy alto (> 10K eventos/min)."
        ),
        "trade_off": "Ganas: procesamiento en tiempo real a escala casi ilimitada. "
                     "Pierdes: costo fijo de Kinesis (~$11/mes/shard), mayor complejidad operativa.",
        "costo_estimado_usd": "$15–$100 / mes según shards y volumen",
    },
    "containers": {
        "cuando_elegirla": (
            "Si la organización tiene política de usar Docker en toda su infraestructura, "
            "ya tiene un equipo DevOps con experiencia en ECS/K8s, o el modelo es demasiado "
            "grande para Lambda (> 512MB descomprimido con dependencias)."
        ),
        "trade_off": "Ganas: control total del runtime, sin límites de tamaño ni timeout, "
                     "modelo siempre en memoria (sin cold start). "
                     "Pierdes: costo mínimo de ~$15/mes aunque no haya tráfico.",
        "costo_estimado_usd": "$15–$80 / mes (Fargate 0.5 vCPU)",
    },
    "sagemaker": {
        "cuando_elegirla": (
            "Si la empresa es grande con equipo MLOps dedicado que necesita model registry, "
            "A/B testing, data drift monitoring y pipelines automatizados de reentrenamiento. "
            "O si ya usan SageMaker Studio y quieren centralizar operaciones de ML."
        ),
        "trade_off": "Ganas: plataforma MLOps completa, monitoreo nativo, auto-scaling avanzado. "
                     "Pierdes: costo mínimo ~$38/mes aunque no haya tráfico.",
        "costo_estimado_usd": "$38–$200 / mes según tipo de instancia",
    },
}


def _build_alternativas(
    ranking: list[ArchitectureScore], winner_id: ArchitectureId
) -> list[AlternativaViable]:
    """
    Returns the non-winner architectures as viable alternatives with organizational
    context for when each one makes sense despite not being the optimal choice.

    Only includes architectures with score >= 4.0 to avoid proposing truly
    incompatible options.
    """
    alternativas = []
    for arch in ranking:
        if arch.id == winner_id:
            continue
        if arch.score_total < 4.0:
            continue
        meta = _ALTERNATIVAS_META.get(arch.id, {})
        alternativas.append(AlternativaViable(
            id=arch.id,
            nombre=arch.nombre,
            score_total=arch.score_total,
            cuando_elegirla=meta.get("cuando_elegirla", ""),
            trade_off=meta.get("trade_off", ""),
            costo_estimado_usd=meta.get("costo_estimado_usd", ""),
        ))
    return alternativas


# ─────────────────────────────────────────────
# CATALOG ENRICHMENT FUNCTIONS
# ─────────────────────────────────────────────

def _get_servicios_detalle(arch_id: ArchitectureId) -> list[CatalogServiceDetail]:
    """Feature 2: returns catalog service cards for all services in the architecture."""
    return [
        _CATALOG[name]
        for name in _ARCH_CATALOG_SERVICES.get(arch_id, [])
        if name in _CATALOG
    ]


def _enrich_restrictions_with_catalog(
    arch_id: ArchitectureId,
    razonamiento: dict[str, CriterionDetail],
) -> None:
    """
    Feature 1: populates CriterionDetail.nota_catalogo with the catalog's constraint
    text for any criterion where es_restriccion=True OR score <= 3.

    Mutates razonamiento in-place. Safe to call on all architectures — only adds
    information, never changes scores or razon text.
    """
    for criterion, detail in razonamiento.items():
        if detail.score > 3 and not detail.es_restriccion:
            continue
        service_name = _CRITERION_SERVICE_MAP.get(criterion, {}).get(arch_id)
        if not service_name:
            continue
        entry = _CATALOG.get(service_name)
        if not entry or not entry.constraints:
            continue
        detail.nota_catalogo = f"[{service_name}] {entry.constraints}"


def _catalog_cross_validate(
    arch_id: ArchitectureId,
    razonamiento: dict[str, CriterionDetail],
) -> list[CatalogValidationNote]:
    """
    Feature 3: cross-validates engine scores against catalog vectors for the winner.

    The catalog uses a 0–5 scale per service; this function scales it to 0–10
    to compare directly with engine scores. Notes are generated only when the
    divergence is >= 2 points — smaller gaps are considered natural variance
    between service-level properties and architecture-level contextual scoring.

    "frecuencia" is excluded because no catalog vector maps to inference frequency.
    """
    primary = _ARCH_PRIMARY_SERVICE.get(arch_id)
    if not primary or primary not in _CATALOG:
        return []

    entry = _CATALOG[primary]
    notes: list[CatalogValidationNote] = []

    for criterion, vector_name in _CRITERION_TO_VECTOR.items():
        if criterion not in razonamiento:
            continue
        catalog_raw = entry.vectores.get(vector_name)
        if catalog_raw is None:
            continue

        catalog_scaled = round(catalog_raw * 2.0, 1)   # 0–5 → 0–10
        engine_score   = razonamiento[criterion].score
        divergencia    = round(abs(engine_score - catalog_scaled), 1)

        if divergencia < 2.0:
            continue

        # Build a human-readable explanation of the divergence
        if criterion == "latencia":
            explanation = (
                "El motor captura el comportamiento warm (~50ms); "
                "el catálogo incluye el cold start risk en su vector."
            )
        elif criterion == "presupuesto":
            explanation = (
                "El motor pondera el presupuesto del caso específico; "
                "el catálogo mide la eficiencia de costo del servicio de forma genérica."
            )
        elif criterion == "escalabilidad":
            explanation = (
                "El motor evalúa si la escala requerida por el caso justifica la arquitectura; "
                "el catálogo mide la capacidad de escala máxima del servicio."
            )
        else:  # experiencia
            explanation = (
                "El motor evalúa el stack completo (servicio + integraciones + IaC); "
                "el catálogo evalúa la curva de aprendizaje del servicio de forma aislada."
            )

        notes.append(CatalogValidationNote(
            criterio=criterion,
            servicio_principal=primary,
            score_motor=engine_score,
            score_catalogo=catalog_scaled,
            divergencia=divergencia,
            nota=(
                f"Motor: {engine_score}/10 · Catálogo {primary} "
                f"({vector_name}={catalog_raw}/5 → {catalog_scaled}/10 escalado). "
                f"Δ={divergencia} pts — {explanation}"
            ),
        ))

    return notes


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def recommend_from_yaml(source: "str | dict | Path") -> RecommendationResult:
    """
    Loads a deployment request YAML and runs the recommender.

    The YAML describes the model AND the deployment context (requirements +
    available services). This is the primary entry point when the user has a
    trained model and wants to know which architecture to use for deployment.

    Args:
        source: one of —
          • local path  : "/path/to/deployment_request.yaml"
          • S3 URI      : "s3://bucket/path/config.yaml"
          • Path object : Path("configs/deployment_request.yaml")
          • dict        : already-parsed YAML (for testing / n8n integration)

    YAML structure expected (see configs/deployment_request.yaml for full template):
      modelo.tipo           → tipo_modelo
      modelo.descripcion    → descripcion
      modelo.num_features   → volumen_datos_kb (estimated if not set explicitly)
      despliegue.*          → latencia, frecuencia, presupuesto, escalabilidad
      organizacion.*        → experiencia_tecnica, servicios_disponibles
    """
    import io as _io

    if isinstance(source, dict):
        cfg = source
    elif str(source).startswith("s3://"):
        # s3://bucket/key/path.yaml
        parts = str(source)[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        import boto3 as _boto3
        s3 = _boto3.client("s3")
        buf = _io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        cfg = yaml.safe_load(buf)
    else:
        with open(source, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    modelo    = cfg.get("modelo", {})
    despliegue = cfg.get("despliegue", {})
    org        = cfg.get("organizacion", {})

    # Estimate volumen_datos_kb from feature count if not given explicitly
    n_features = int(modelo.get("num_features", 10))
    volumen_kb = float(despliegue.get("volumen_datos_kb", round(n_features * 8 / 1024, 2)))

    case = CaseInput(
        descripcion=modelo.get("descripcion", modelo.get("nombre", "Modelo ML")),
        tipo_modelo=modelo.get("tipo", "clasificacion_binaria"),
        latencia_requerida_ms=float(despliegue["latencia_requerida_ms"]),
        frecuencia_inferencia=despliegue["frecuencia_inferencia"],
        volumen_datos_kb=volumen_kb,
        presupuesto_mensual_usd=float(despliegue["presupuesto_mensual_usd"]),
        escalabilidad_requerida=despliegue["escalabilidad_requerida"],
        experiencia_tecnica=org.get("experiencia_tecnica", "media"),
        servicios_disponibles=org.get("servicios_disponibles", []),
    )
    return recommend(case)


def recommend(case: CaseInput) -> RecommendationResult:
    """
    Evaluates all 5 architectures against the 5 weighted criteria,
    returns the best match with full documented reasoning.
    """
    weights = _adaptive_weights(case)
    scores: list[ArchitectureScore] = []

    for arch_id, meta in ARCHITECTURES_META.items():
        lat  = _kb_latencia(arch_id, case.latencia_requerida_ms)
        freq = _kb_frecuencia(arch_id, case.frecuencia_inferencia)
        pres = _kb_presupuesto(arch_id, case.presupuesto_mensual_usd)
        esc  = _kb_escalabilidad(arch_id, case.escalabilidad_requerida)
        exp  = _kb_experiencia(arch_id, case.experiencia_tecnica)

        razonamiento = {
            "latencia":      lat,
            "frecuencia":    freq,
            "presupuesto":   pres,
            "escalabilidad": esc,
            "experiencia":   exp,
        }
        desglose = {k: v.score for k, v in razonamiento.items()}
        total = sum(desglose[k] * weights[k] for k in desglose)

        scores.append(ArchitectureScore(
            id=arch_id,
            nombre=meta["nombre"],
            servicios_aws=meta["servicios_aws"],
            score_total=round(total, 3),
            desglose=desglose,
            razonamiento=razonamiento,
        ))

    ranking = sorted(scores, key=lambda s: s.score_total, reverse=True)
    for i, item in enumerate(ranking):
        item.posicion = i + 1

    top    = ranking[0]
    second = ranking[1]

    # Tag each discarded architecture with its main failure reason
    for arch in ranking[1:]:
        worst_k = min(arch.desglose, key=lambda k: arch.desglose[k] * weights[k])
        arch.descartada_por = arch.razonamiento[worst_k].razon

    confianza = min((top.score_total - second.score_total) / 10 * 5, 1.0)

    advertencias = _detect_advertencias(case, ranking)

    # Service availability filter — block or substitute missing services
    if case.servicios_disponibles:
        available = {_normalize_service(s) for s in case.servicios_disponibles}
        for arch in ranking:
            required = _ARCH_REQUIRED_SERVICES.get(arch.id, [])
            missing  = [s for s in required if _normalize_service(s) not in available]
            if not missing:
                continue

            # For each missing service, try to find an available substitute
            unresolved: list[str] = []
            for svc in missing:
                sub = _find_substitution(svc, available)
                if sub:
                    arch.viable_con_sustitucion = True
                    arch.sustituciones_propuestas.append(SustitucionServicio(
                        servicio_requerido=svc,
                        servicio_sustituto=sub["sustituto"],
                        impacto_costo=sub["impacto_costo"],
                        impacto_latencia=sub["impacto_latencia"],
                        nota=sub["nota"],
                    ))
                    advertencias.append(Advertencia(
                        tipo="sustitucion_servicio",
                        mensaje=(
                            f"'{arch.nombre}': {svc} no disponible → "
                            f"se sugiere {sub['sustituto']} como alternativa equivalente. "
                            f"Impacto costo: {sub['impacto_costo']}."
                        ),
                    ))
                else:
                    unresolved.append(svc)

            if unresolved:
                arch.bloqueada_por_servicio = True
                arch.descartada_por = (
                    f"Requiere servicios sin sustituto disponible: {', '.join(unresolved)}"
                )
                advertencias.append(Advertencia(
                    tipo="servicio_no_disponible",
                    mensaje=(
                        f"'{arch.nombre}' requiere {', '.join(unresolved)} "
                        "y no hay sustituto habilitado en la organización."
                    ),
                ))

    # If the top scorer is blocked, use first non-blocked (includes viable_con_sustitucion)
    effective_top = next((a for a in ranking if not a.bloqueada_por_servicio), top)
    if effective_top is not top:
        top = effective_top
        second = next((a for a in ranking if a is not top and not a.bloqueada_por_servicio), ranking[1])

    criterio_decisivo, razonamiento_completo = _build_razonamiento(top, second, ranking, case, weights)

    # Feature 1 — enrich restricted/low-score criteria with catalog constraints
    for arch in ranking:
        _enrich_restrictions_with_catalog(arch.id, arch.razonamiento)

    # Short justification (for the API response / n8n output)
    justificacion = (
        f"Arquitectura recomendada: '{top.nombre}' con score {top.score_total}/10. "
        f"Criterio decisivo: {criterio_decisivo} "
        f"Segunda opción descartada: '{second.nombre}' ({second.score_total}/10) — "
        f"{second.descartada_por[:120]}."
    )

    configuracion_inicial = _build_configuracion(top.id)
    alternativas_viables  = _build_alternativas(ranking, top.id)

    # Feature 2 — catalog service cards for the winner
    servicios_detalle = _get_servicios_detalle(top.id)

    # Feature 3 — cross-validate winner scores against catalog vectors
    notas_validacion_catalogo = _catalog_cross_validate(top.id, top.razonamiento)

    return RecommendationResult(
        recomendacion=top,
        ranking=ranking,
        confianza=round(confianza, 2),
        justificacion=justificacion,
        criterio_decisivo=criterio_decisivo,
        razonamiento_completo=razonamiento_completo,
        advertencias=advertencias,
        configuracion_inicial=configuracion_inicial,
        alternativas_viables=alternativas_viables,
        input_recibido=case,
        servicios_detalle=servicios_detalle,
        notas_validacion_catalogo=notas_validacion_catalogo,
    )
