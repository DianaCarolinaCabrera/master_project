"""
Terminal demo del motor de recomendación — sin n8n ni AWS corriendo.

Uso:
  poetry run demo                                           # caso serverless (proyecto de tesis)
  poetry run demo --caso batch                             # banco con scoring nocturno
  poetry run demo --caso streaming                         # fraude en tiempo real
  poetry run demo --yaml configs/deployment_request.yaml  # desde YAML local
  poetry run demo --yaml s3://mi-bucket/config.yaml       # desde YAML en S3
  poetry run demo --interactivo                            # ingresa los valores manualmente
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ml_arch_recommender.scoring.engine import (
    CaseInput,
    RecommendationResult,
    ServiceAnalysisResult,
    recommend,
    recommend_from_yaml,
    analyze_services_for_arch,
    get_relevant_services_for_arch,
    _ARCH_MIN_COST_USD,
)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CASOS PRE-DEFINIDOS
# ─────────────────────────────────────────────────────────────────────────────

CASOS: dict[str, CaseInput] = {
    "serverless": CaseInput(
        descripcion="Modelo de riesgo crediticio — fintech de microcréditos",
        tipo_modelo="clasificacion_binaria",
        latencia_requerida_ms=800,
        frecuencia_inferencia="baja",
        volumen_datos_kb=5,
        presupuesto_mensual_usd=30,
        escalabilidad_requerida="media",
        experiencia_tecnica="media",
    ),
    "batch": CaseInput(
        descripcion="Rescoring nocturno del portafolio completo — banco tradicional",
        tipo_modelo="clasificacion_binaria",
        latencia_requerida_ms=86_400_000,
        frecuencia_inferencia="baja",
        volumen_datos_kb=500_000,
        presupuesto_mensual_usd=300,
        escalabilidad_requerida="alta",
        experiencia_tecnica="alta",
    ),
    "streaming": CaseInput(
        descripcion="Detección de fraude en tiempo real — plataforma de pagos",
        tipo_modelo="clasificacion_binaria",
        latencia_requerida_ms=200,
        frecuencia_inferencia="continua",
        volumen_datos_kb=2,
        presupuesto_mensual_usd=500,
        escalabilidad_requerida="alta",
        experiencia_tecnica="alta",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION GUIDES — estructura, archivos, env vars, permisos, best practice
# ─────────────────────────────────────────────────────────────────────────────

IMPL_GUIDES: dict[str, dict] = {
    "serverless": {
        "estructura": [
            "credit-inference/",
            "├── lambda_function/",
            "│   ├── handler.py        ← inferencia + carga del modelo desde S3",
            "│   └── requirements.txt  ← scikit-learn, joblib, boto3, numpy",
            "├── deploy/",
            "│   ├── template.yaml     ← SAM / CloudFormation (función + API + rol IAM)",
            "│   └── build.sh          ← pip install + zip + aws s3 cp",
            "├── .env.example",
            "└── README.md",
        ],
        "archivos": [
            {
                "nombre": "lambda_function/handler.py",
                "descripcion": "Entry point Lambda — carga el modelo en cold start, ejecuta predict_proba en cada request.",
                "codigo": (
                    "import os, json, boto3, joblib\n"
                    "from io import BytesIO\n\n"
                    "_model = None\n\n"
                    "def _load_model():\n"
                    "    global _model\n"
                    "    if _model is None:\n"
                    "        buf = BytesIO()\n"
                    "        boto3.client('s3').download_fileobj(\n"
                    "            os.environ['MODEL_BUCKET'], os.environ['MODEL_KEY'], buf)\n"
                    "        buf.seek(0)\n"
                    "        _model = joblib.load(buf)\n"
                    "    return _model\n\n"
                    "def lambda_handler(event, context):\n"
                    "    body  = json.loads(event.get('body', '{}'))\n"
                    "    order = os.environ['FEATURE_ORDER'].split(',')\n"
                    "    prob  = float(_load_model().predict_proba([[body[f] for f in order]])[0][1])\n"
                    "    return {'statusCode': 200,\n"
                    "            'body': json.dumps({'probabilidad': prob, 'clase': int(prob >= 0.5)})}"
                ),
            },
            {
                "nombre": "deploy/template.yaml",
                "descripcion": "SAM template — define función, API HTTP y rol IAM con mínimo privilegio.",
                "codigo": (
                    "AWSTemplateFormatVersion: '2010-09-09'\n"
                    "Transform: AWS::Serverless-2016-10-31\n"
                    "Resources:\n"
                    "  InferenceFunction:\n"
                    "    Type: AWS::Serverless::Function\n"
                    "    Properties:\n"
                    "      Runtime: python3.11\n"
                    "      Handler: handler.lambda_handler\n"
                    "      MemorySize: 512\n"
                    "      Timeout: 30\n"
                    "      Environment:\n"
                    "        Variables:\n"
                    "          MODEL_BUCKET: !Ref ModelBucket\n"
                    "          MODEL_KEY: models/v1/credit_model.joblib\n"
                    "          FEATURE_ORDER: edad,ingresos_mensuales,ratio_deuda\n"
                    "      Policies:\n"
                    "        - S3ReadPolicy:\n"
                    "            BucketName: !Ref ModelBucket\n"
                    "      Events:\n"
                    "        Predict:\n"
                    "          Type: HttpApi\n"
                    "          Properties:\n"
                    "            Path: /predict\n"
                    "            Method: POST"
                ),
            },
        ],
        "env_vars": [
            {"nombre": "MODEL_BUCKET",  "descripcion": "Bucket S3 con el modelo",        "ejemplo": "mi-modelo-bucket",             "es_secreto": False},
            {"nombre": "MODEL_KEY",     "descripcion": "Ruta del joblib en S3",           "ejemplo": "models/v1/credit_model.joblib", "es_secreto": False},
            {"nombre": "FEATURE_ORDER", "descripcion": "Features en orden, separadas por coma", "ejemplo": "edad,ingresos,ratio_deuda", "es_secreto": False},
            {"nombre": "LOG_LEVEL",     "descripcion": "Nivel de logging",                "ejemplo": "INFO",                         "es_secreto": False},
        ],
        "permisos": {
            "visibilidad": "RESTRINGIDO",
            "visibilidad_detalle": (
                "API Gateway puede ser público (apps externas) pero SIEMPRE con autenticación:\n"
                "  • API Key   — para integraciones B2B simples\n"
                "  • Cognito   — para usuarios finales con login\n"
                "  • Lambda Authorizer — para JWT custom\n"
                "  Nunca dejar el endpoint abierto sin auth en producción."
            ),
            "quien_edita_config": (
                "Rol  ml-deploy-admin:\n"
                "  • lambda:UpdateFunctionConfiguration\n"
                "  • lambda:UpdateFunctionCode\n"
                "  • ssm:PutParameter en /ml/inference/*\n"
                "  NO: IAM admin, lambda:CreateFunction, lambda:DeleteFunction"
            ),
            "quien_accede_logs": (
                "Rol  ml-ops-readonly:\n"
                "  • logs:FilterLogEvents  en  /aws/lambda/credit-risk-inference  (solo ese grupo)\n"
                "  • cloudwatch:GetMetricStatistics  en dimensión FunctionName específica\n"
                "  NO acceso a otros grupos de logs ni métricas de otros servicios"
            ),
            "quien_modifica_secrets": (
                "Rol  ml-secrets-admin  (solo security lead o arquitecto):\n"
                "  • secretsmanager:UpdateSecret  en ARN específico\n"
                "  • ssm:PutParameter  con SecureString  en /ml/inference/\n"
                "  Los desarrolladores NO deben tener acceso a secretos de producción"
            ),
            "permisos_repo": [
                "✓  lambda:UpdateFunctionCode  (en ARN de la función específica)",
                "✓  s3:PutObject / s3:GetObject  (en bucket de deploy, NO el de datos de producción)",
                "✓  lambda:GetFunctionConfiguration  (verificar que el deploy fue exitoso)",
                "✗  NO: lambda:CreateFunction, lambda:DeleteFunction, IAM:*",
            ],
            "roles": [
                {
                    "nombre": "LambdaExecutionRole",
                    "uso": "Attached a la función Lambda (runtime)",
                    "permisos": [
                        "AWSLambdaBasicExecutionRole  (logs básicos a CloudWatch)",
                        "s3:GetObject  en  arn:aws:s3:::MODEL_BUCKET/models/*",
                    ],
                    "no_incluir": "s3:*, s3:DeleteObject, iam:*, ec2:*",
                },
                {
                    "nombre": "MLDeployRole",
                    "uso": "CI/CD pipeline (GitHub Actions, CodePipeline)",
                    "permisos": [
                        "lambda:UpdateFunctionCode  en la función específica",
                        "s3:PutObject  en bucket-deploy/builds/*",
                        "lambda:GetFunction  (verificación post-deploy)",
                    ],
                    "no_incluir": "lambda:CreateFunction, lambda:DeleteFunction, IAM:*",
                },
                {
                    "nombre": "MLOpsReadonlyRole",
                    "uso": "Monitoreo (equipo MLOps, dashboards, alertas)",
                    "permisos": [
                        "logs:FilterLogEvents  en  /aws/lambda/credit-risk-inference",
                        "cloudwatch:GetMetricStatistics  en función específica",
                    ],
                    "no_incluir": "lambda:InvokeFunction, ssm:GetParameter, s3:*",
                },
            ],
            "minimo_privilegio": (
                "1. Usa ARNs específicos en Resource — NUNCA  \"Resource\": \"*\"\n"
                "2. Separa roles: ejecución, deploy y lectura son 3 roles distintos\n"
                "3. Revisa permisos cada 90 días con IAM Access Analyzer\n"
                "4. Usa condition key  aws:SourceAccount  para bloquear invocaciones cross-account\n"
                "5. CloudTrail muestra qué permisos no se usan — elimínalos"
            ),
        },
        "buena_practica": (
            "Versiona los modelos en S3 (models/v1/, models/v2/). La función Lambda apunta a una versión "
            "específica via MODEL_KEY. Un rollback es cambiar MODEL_KEY en SSM Parameter Store "
            "— sin redespliegue de código, en menos de 30 segundos."
        ),
    },

    "batch": {
        "estructura": [
            "batch-scoring/",
            "├── scripts/",
            "│   └── inference_job.py   ← Python shell: lee S3, predice, escribe resultados",
            "├── configs/",
            "│   └── job_params.json    ← rutas S3, nombre del modelo, umbrales",
            "├── deploy/",
            "│   └── create_job.sh      ← aws glue create-job",
            "├── .env.example",
            "└── README.md",
        ],
        "archivos": [
            {
                "nombre": "scripts/inference_job.py",
                "descripcion": "Script Glue — lee CSV/Parquet de S3, carga joblib, genera scores y escribe resultados.",
                "codigo": (
                    "import os, sys, boto3, joblib, pandas as pd\n"
                    "from io import BytesIO\n"
                    "from awsglue.utils import getResolvedOptions\n\n"
                    "args = getResolvedOptions(sys.argv, ['MODEL_KEY', 'INPUT_PATH', 'OUTPUT_PATH'])\n"
                    "FEATURE_ORDER = os.environ.get('FEATURE_ORDER', 'edad,ingresos,ratio_deuda').split(',')\n\n"
                    "# Carga modelo desde S3\n"
                    "buf = BytesIO()\n"
                    "boto3.client('s3').download_fileobj(os.environ['MODEL_BUCKET'], args['MODEL_KEY'], buf)\n"
                    "buf.seek(0)\n"
                    "model = joblib.load(buf)\n\n"
                    "# Procesa datos\n"
                    "df = pd.read_csv(args['INPUT_PATH'])\n"
                    "df['score'] = model.predict_proba(df[FEATURE_ORDER])[:, 1]\n"
                    "df['clase'] = (df['score'] >= 0.5).astype(int)\n\n"
                    "# Guarda con fecha para trazabilidad\n"
                    "from datetime import date\n"
                    "out = f\"{args['OUTPUT_PATH']}{date.today()}/resultados.csv\"\n"
                    "df[['id', 'score', 'clase']].to_csv(out, index=False)"
                ),
            },
            {
                "nombre": "deploy/create_job.sh",
                "descripcion": "Crea el Glue Job con bookmark activado para procesar solo archivos nuevos.",
                "codigo": (
                    "aws glue create-job \\\n"
                    "  --name credit-batch-scoring \\\n"
                    "  --role GlueJobRole \\\n"
                    "  --command Name=pythonshell,\\\n"
                    "ScriptLocation=s3://mi-scripts-bucket/scripts/inference_job.py,\\\n"
                    "PythonVersion=3 \\\n"
                    "  --default-arguments '{\n"
                    "    \"--job-bookmark-enable\": \"job-bookmark-enable\",\n"
                    "    \"--MODEL_BUCKET\": \"mi-modelo-bucket\",\n"
                    "    \"--MODEL_KEY\":    \"models/v1/credit_model.joblib\"\n"
                    "  }'"
                ),
            },
        ],
        "env_vars": [
            {"nombre": "MODEL_BUCKET",  "descripcion": "Bucket del modelo",            "ejemplo": "mi-modelo-bucket",         "es_secreto": False},
            {"nombre": "MODEL_KEY",     "descripcion": "Ruta del joblib en S3",         "ejemplo": "models/v1/model.joblib",   "es_secreto": False},
            {"nombre": "INPUT_PATH",    "descripcion": "Ruta S3 de datos de entrada",  "ejemplo": "s3://data-bucket/input/",  "es_secreto": False},
            {"nombre": "OUTPUT_PATH",   "descripcion": "Ruta S3 de resultados",         "ejemplo": "s3://data-bucket/output/", "es_secreto": False},
            {"nombre": "FEATURE_ORDER", "descripcion": "Features en orden",             "ejemplo": "edad,ingresos,ratio_deuda","es_secreto": False},
        ],
        "permisos": {
            "visibilidad": "PRIVADO",
            "visibilidad_detalle": (
                "No hay endpoint HTTP expuesto. El proceso corre completamente en AWS.\n"
                "  • Los buckets S3 DEBEN tener  BlockPublicAcls: true  y  BlockPublicPolicy: true\n"
                "  • Los resultados son accesibles solo con IAM explícito\n"
                "  • Si se comparten resultados externamente: usar URLs pre-firmadas (expiración ≤ 24h)"
            ),
            "quien_edita_config": (
                "Rol  ml-glue-admin:\n"
                "  • glue:UpdateJob  (en el job específico)\n"
                "  • s3:PutObject  en bucket de scripts\n"
                "  NO: glue:CreateJob, glue:DeleteJob sin aprobación"
            ),
            "quien_accede_logs": (
                "Rol  ml-ops-readonly:\n"
                "  • logs:GetLogEvents  en  /aws-glue/jobs/output  (solo ese grupo)\n"
                "  • glue:GetJobRun + glue:GetJobRuns  en job específico\n"
                "  • s3:GetObject  en  data-bucket/output/*  (para verificar resultados)"
            ),
            "quien_modifica_secrets": (
                "Batch típicamente no usa credenciales externas.\n"
                "  Si el job accede a una BD externa:\n"
                "  • Guardar en  AWS Secrets Manager\n"
                "  • GlueJobRole: secretsmanager:GetSecretValue  solo en ese ARN de secret específico"
            ),
            "permisos_repo": [
                "✓  glue:UpdateJob  (en el job específico)",
                "✓  s3:PutObject  (en bucket de scripts, para actualizar código)",
                "✓  glue:StartJobRun  (para deploy + smoke test automático)",
                "✗  NO: glue:CreateJob, glue:DeleteJob  (operaciones de infraestructura, no de deploy)",
            ],
            "roles": [
                {
                    "nombre": "GlueJobRole",
                    "uso": "Execution role del Glue Job (runtime)",
                    "permisos": [
                        "s3:GetObject  en  arn:aws:s3:::mi-modelo-bucket/models/*",
                        "s3:GetObject  en  arn:aws:s3:::data-bucket/input/*",
                        "s3:PutObject  en  arn:aws:s3:::data-bucket/output/*",
                        "cloudwatch:PutMetricData  (métricas del job)",
                    ],
                    "no_incluir": "s3:DeleteBucket, s3:DeleteObject, s3:*, iam:*, glue:DeleteJob",
                },
                {
                    "nombre": "EventBridgeTriggerRole",
                    "uso": "Dispara el job según el schedule (cron)",
                    "permisos": [
                        "glue:StartJobRun  en el job específico",
                    ],
                    "no_incluir": "glue:DeleteJob, glue:CreateJob, s3:*",
                },
                {
                    "nombre": "MLOpsReadonlyRole",
                    "uso": "Monitoreo del proceso batch",
                    "permisos": [
                        "glue:GetJobRun, glue:GetJobRuns  en job específico",
                        "logs:GetLogEvents  en  /aws-glue/jobs/output",
                        "s3:GetObject  en  data-bucket/output/*",
                    ],
                    "no_incluir": "glue:StartJobRun, s3:PutObject, glue:UpdateJob",
                },
            ],
            "minimo_privilegio": (
                "1. GlueJobRole: limitar s3:GetObject/PutObject a prefijos exactos (models/, input/, output/)\n"
                "2. NO dar s3:* ni s3:DeleteObject al job — si falla, no debe poder borrar datos\n"
                "3. EventBridge solo necesita StartJobRun — sin acceso directo a S3\n"
                "4. Activa S3 Bucket Versioning en el bucket de output para recovery ante sobreescritura"
            ),
        },
        "buena_practica": (
            "Activa --job-bookmark-enable para que Glue procese solo archivos nuevos en cada ejecución. "
            "Guarda resultados con fecha: output/2024-01-15/resultados.csv — "
            "nunca sobreescribas el output anterior. Esto permite auditoría completa y rollback inmediato."
        ),
    },

    "streaming": {
        "estructura": [
            "streaming-inference/",
            "├── lambda_function/",
            "│   ├── handler.py          ← consumer Kinesis: decodifica, predice, guarda",
            "│   └── requirements.txt",
            "├── producer/",
            "│   └── example_producer.py ← cómo enviar eventos al stream",
            "├── deploy/",
            "│   ├── create_stream.sh    ← aws kinesis create-stream",
            "│   └── create_mapping.sh   ← Lambda ← Kinesis + configura DLQ",
            "├── .env.example",
            "└── README.md",
        ],
        "archivos": [
            {
                "nombre": "lambda_function/handler.py",
                "descripcion": "Consumer Kinesis — decodifica base64, predice y guarda en DynamoDB.",
                "codigo": (
                    "import os, json, base64, boto3, joblib\n"
                    "from io import BytesIO\n\n"
                    "_model = None\n"
                    "def _load_model():\n"
                    "    global _model\n"
                    "    if _model is None:\n"
                    "        buf = BytesIO()\n"
                    "        boto3.client('s3').download_fileobj(\n"
                    "            os.environ['MODEL_BUCKET'], os.environ['MODEL_KEY'], buf)\n"
                    "        buf.seek(0); _model = joblib.load(buf)\n"
                    "    return _model\n\n"
                    "def lambda_handler(event, context):\n"
                    "    ddb   = boto3.resource('dynamodb').Table(os.environ['RESULTS_TABLE'])\n"
                    "    order = os.environ['FEATURE_ORDER'].split(',')\n"
                    "    for record in event['Records']:\n"
                    "        data = json.loads(base64.b64decode(record['kinesis']['data']))\n"
                    "        prob = float(_load_model().predict_proba([[data[f] for f in order]])[0][1])\n"
                    "        ddb.put_item(Item={\n"
                    "            'id':    data.get('id', record['kinesis']['sequenceNumber']),\n"
                    "            'score': str(prob),\n"
                    "            'clase': int(prob >= 0.5),\n"
                    "        })"
                ),
            },
            {
                "nombre": "deploy/create_mapping.sh",
                "descripcion": "Conecta Kinesis con Lambda y configura DLQ para capturar errores.",
                "codigo": (
                    "# 1. Crear la DLQ primero\n"
                    "aws sqs create-queue --queue-name inference-dlq\n\n"
                    "# 2. Event source mapping con DLQ y bisect-on-error\n"
                    "aws lambda create-event-source-mapping \\\n"
                    "  --function-name streaming-inference \\\n"
                    "  --event-source-arn arn:aws:kinesis:us-east-1:ACCOUNT:stream/inference-stream \\\n"
                    "  --starting-position LATEST \\\n"
                    "  --batch-size 10 \\\n"
                    "  --bisect-batch-on-function-error \\\n"
                    "  --destination-config \\\n"
                    "    '{\"OnFailure\":{\"Destination\":\"arn:aws:sqs:us-east-1:ACCOUNT:inference-dlq\"}}'"
                ),
            },
        ],
        "env_vars": [
            {"nombre": "MODEL_BUCKET",  "descripcion": "Bucket del modelo",                "ejemplo": "mi-modelo-bucket",         "es_secreto": False},
            {"nombre": "MODEL_KEY",     "descripcion": "Ruta del joblib",                  "ejemplo": "models/v1/model.joblib",   "es_secreto": False},
            {"nombre": "FEATURE_ORDER", "descripcion": "Features esperadas en cada evento","ejemplo": "edad,ingresos,ratio_deuda","es_secreto": False},
            {"nombre": "RESULTS_TABLE", "descripcion": "Tabla DynamoDB para resultados",  "ejemplo": "inference-results-prod",   "es_secreto": False},
            {"nombre": "BATCH_SIZE",    "descripcion": "Records por invocación Lambda",    "ejemplo": "10",                       "es_secreto": False},
        ],
        "permisos": {
            "visibilidad": "PRIVADO/INTERNO",
            "visibilidad_detalle": (
                "El stream Kinesis es un recurso privado de AWS — no tiene endpoint HTTP.\n"
                "  • Solo accesible desde servicios con IAM explícito (kinesis:PutRecord)\n"
                "  • Si el productor es externo: API Gateway → Lambda → Kinesis como proxy seguro\n"
                "  • Nunca compartir credenciales AWS para que un tercero escriba directo al stream"
            ),
            "quien_edita_config": (
                "Rol  ml-streaming-admin:\n"
                "  • lambda:UpdateFunctionCode + UpdateFunctionConfiguration\n"
                "  • kinesis:UpdateShardCount  (para escalar shards)\n"
                "  NO: kinesis:DeleteStream, kinesis:MergeShards sin aprobación"
            ),
            "quien_accede_logs": (
                "Rol  ml-ops-readonly:\n"
                "  • logs:FilterLogEvents  en  /aws/lambda/streaming-inference\n"
                "  • kinesis:GetShardIterator + kinesis:GetRecords  (debug del stream)\n"
                "  • cloudwatch: GetRecords.IteratorAgeMilliseconds  (métrica clave de latencia)"
            ),
            "quien_modifica_secrets": (
                "No hay secretos en el flujo básico.\n"
                "  Si Lambda guarda en BD externa:\n"
                "  • Credenciales en  Secrets Manager\n"
                "  • LambdaKinesisRole: secretsmanager:GetSecretValue  en ARN específico"
            ),
            "permisos_repo": [
                "✓  lambda:UpdateFunctionCode  (en función específica)",
                "✓  s3:PutObject  (en bucket de deploy para el zip)",
                "✓  lambda:GetEventSourceMapping  (verificación del mapping)",
                "✗  NO: kinesis:CreateStream, kinesis:DeleteStream, lambda:CreateEventSourceMapping",
            ],
            "roles": [
                {
                    "nombre": "LambdaKinesisRole",
                    "uso": "Execution role de la función Lambda consumer",
                    "permisos": [
                        "kinesis:GetRecords, GetShardIterator, DescribeStream, ListShards  (en stream específico)",
                        "s3:GetObject  en  arn:aws:s3:::mi-modelo-bucket/models/*",
                        "dynamodb:PutItem  en  arn:...:table/inference-results-prod",
                        "AWSLambdaBasicExecutionRole  (logs CloudWatch)",
                        "sqs:SendMessage  en  la DLQ  (para reenvío de errores)",
                    ],
                    "no_incluir": "kinesis:DeleteStream, kinesis:MergeShards, dynamodb:DeleteTable",
                },
                {
                    "nombre": "ProducerRole",
                    "uso": "Aplicación que envía eventos al stream",
                    "permisos": [
                        "kinesis:PutRecord, kinesis:PutRecords  en stream específico",
                    ],
                    "no_incluir": "kinesis:GetRecords, kinesis:DeleteStream — solo escritura",
                },
                {
                    "nombre": "MLOpsReadonlyRole",
                    "uso": "Monitoreo del pipeline en tiempo real",
                    "permisos": [
                        "kinesis:GetShardIterator, kinesis:GetRecords  (debug del stream)",
                        "cloudwatch:GetMetricStatistics  en stream específico",
                        "logs:FilterLogEvents  en  /aws/lambda/streaming-inference",
                        "sqs:ReceiveMessage  en DLQ  (para inspeccionar errores)",
                    ],
                    "no_incluir": "kinesis:PutRecord, lambda:InvokeFunction, sqs:DeleteMessage",
                },
            ],
            "minimo_privilegio": (
                "1. Lambda solo lee del stream (GET) — nunca escribe (PUT), evita loops infinitos\n"
                "2. ProducerRole solo escribe (PUT) — separación clara de responsabilidades\n"
                "3. Configura DLQ en el event source mapping para capturar eventos fallidos\n"
                "4. Usa condition key  kinesis:StreamARN  para limitar permisos al stream específico"
            ),
        },
        "buena_practica": (
            "Configura Dead Letter Queue (DLQ) en el event source mapping "
            "(--destination-config OnFailure → SQS). "
            "Sin DLQ, los errores silenciosos son el mayor riesgo en streaming: "
            "puedes perder predicciones sin ninguna alerta. "
            "La DLQ captura los records fallidos para diagnóstico posterior sin perder datos."
        ),
    },

    "containers": {
        "estructura": [
            "ml-inference-service/",
            "├── app/",
            "│   ├── main.py            ← FastAPI: POST /predict + GET /health",
            "│   └── model_loader.py    ← carga joblib de S3 al arrancar el contenedor",
            "├── Dockerfile",
            "├── requirements.txt",
            "├── task-definition.json   ← ECS task: cpu, memory, imagen ECR, env vars",
            "├── .env.example",
            "└── README.md",
        ],
        "archivos": [
            {
                "nombre": "app/main.py",
                "descripcion": "API FastAPI — carga modelo al inicio, expone /predict y /health.",
                "codigo": (
                    "import os\n"
                    "from contextlib import asynccontextmanager\n"
                    "from fastapi import FastAPI\n"
                    "from pydantic import BaseModel\n"
                    "from model_loader import load_model\n\n"
                    "_model = None\n\n"
                    "@asynccontextmanager\n"
                    "async def lifespan(app: FastAPI):\n"
                    "    global _model\n"
                    "    _model = load_model()  # carga desde S3 al arrancar\n"
                    "    yield\n\n"
                    "app = FastAPI(lifespan=lifespan)\n\n"
                    "class Features(BaseModel):\n"
                    "    edad: int; ingresos_mensuales: float; ratio_deuda: float\n\n"
                    "@app.get('/health')\n"
                    "def health(): return {'status': 'ok', 'model': _model is not None}\n\n"
                    "@app.post('/predict')\n"
                    "def predict(f: Features):\n"
                    "    prob = float(_model.predict_proba([[f.edad, f.ingresos_mensuales, f.ratio_deuda]])[0][1])\n"
                    "    return {'probabilidad': prob, 'clase': int(prob >= 0.5)}"
                ),
            },
            {
                "nombre": "Dockerfile",
                "descripcion": "Imagen mínima con usuario no-root (principio de mínimo privilegio en el contenedor).",
                "codigo": (
                    "FROM python:3.11-slim\n"
                    "WORKDIR /app\n"
                    "COPY requirements.txt .\n"
                    "RUN pip install --no-cache-dir -r requirements.txt\n"
                    "COPY app/ .\n"
                    "EXPOSE 8080\n"
                    "# Usuario no-root — reduce superficie de ataque\n"
                    "RUN useradd -m appuser && chown -R appuser /app\n"
                    "USER appuser\n"
                    "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\", \"--workers\", \"2\"]"
                ),
            },
        ],
        "env_vars": [
            {"nombre": "MODEL_BUCKET", "descripcion": "Bucket S3 con el modelo",      "ejemplo": "mi-modelo-bucket",       "es_secreto": False},
            {"nombre": "MODEL_KEY",    "descripcion": "Ruta del joblib en S3",         "ejemplo": "models/v1/model.joblib", "es_secreto": False},
            {"nombre": "PORT",         "descripcion": "Puerto del servidor",            "ejemplo": "8080",                   "es_secreto": False},
            {"nombre": "LOG_LEVEL",    "descripcion": "Nivel de logging uvicorn",       "ejemplo": "info",                   "es_secreto": False},
        ],
        "permisos": {
            "visibilidad": "RESTRINGIDO",
            "visibilidad_detalle": (
                "El contenedor NUNCA es accesible directo desde internet.\n"
                "  • ALB recibe tráfico externo en puerto 443 (HTTPS, certificado ACM)\n"
                "  • Security Group del contenedor: solo permite ingreso desde SG del ALB\n"
                "  • Security Group del ALB: permite 443 desde los rangos IP requeridos\n"
                "  • HTTPS obligatorio en producción — no aceptar HTTP en el ALB listener"
            ),
            "quien_edita_config": (
                "Rol  ml-ecs-admin:\n"
                "  • ecs:UpdateService + ecs:RegisterTaskDefinition\n"
                "  • ecr:PutImage  (actualizar imagen del contenedor)\n"
                "  NO: ecs:DeleteService, ecs:DeleteCluster sin aprobación"
            ),
            "quien_accede_logs": (
                "Rol  ml-ops-readonly:\n"
                "  • logs:FilterLogEvents  en  /ecs/ml-inference-service\n"
                "  • ecs:DescribeServices  (ver estado del servicio)\n"
                "  • cloudwatch:GetMetricStatistics  en ServiceName específico"
            ),
            "quien_modifica_secrets": (
                "Rol  ml-secrets-admin:\n"
                "  • secretsmanager:UpdateSecret  en ARN específico\n"
                "  • ssm:PutParameter  en  /ml/containers/*\n"
                "  Usar  envFrom  en task-definition.json para leer desde SSM — nunca hardcodear env vars"
            ),
            "permisos_repo": [
                "✓  ecr:BatchCheckLayerAvailability, PutImage, InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload",
                "✓  ecr:GetAuthorizationToken",
                "✓  ecs:UpdateService + ecs:RegisterTaskDefinition + ecs:DescribeServices",
                "✓  iam:PassRole  (solo para pasar ECSTaskRole/ExecutionRole al task definition)",
                "✗  NO: ecs:DeleteService, ecs:DeleteCluster, iam:CreateRole",
            ],
            "roles": [
                {
                    "nombre": "ECSTaskExecutionRole",
                    "uso": "ECS agent — pull de imagen ECR y escritura de logs (infraestructura)",
                    "permisos": [
                        "AmazonECSTaskExecutionRolePolicy  (policy managed por AWS)",
                        "ssm:GetParameters  en  /ml/containers/*  (para secrets en task def)",
                    ],
                    "no_incluir": "s3:*, iam:*, ec2:* — este rol es para ECS agent, NO para tu código",
                },
                {
                    "nombre": "ECSTaskRole",
                    "uso": "Runtime del contenedor (acceso a AWS desde el código Python)",
                    "permisos": [
                        "s3:GetObject  en  arn:aws:s3:::mi-modelo-bucket/models/*",
                        "cloudwatch:PutMetricData  (métricas custom del modelo)",
                    ],
                    "no_incluir": "ecr:*, ecs:*, iam:* — este rol es para el código, NO para ECS agent",
                },
                {
                    "nombre": "ECRPushRole",
                    "uso": "CI/CD — build y push de la imagen Docker",
                    "permisos": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability, PutImage, InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload",
                    ],
                    "no_incluir": "ecr:DeleteRepository, ecs:*, iam:*",
                },
                {
                    "nombre": "MLOpsReadonlyRole",
                    "uso": "Monitoreo del servicio",
                    "permisos": [
                        "ecs:DescribeServices, ecs:ListTasks",
                        "logs:FilterLogEvents  en  /ecs/ml-inference-service",
                        "cloudwatch:GetMetricStatistics",
                    ],
                    "no_incluir": "ecs:UpdateService, ecr:PutImage, s3:PutObject",
                },
            ],
            "minimo_privilegio": (
                "1. ECSTaskExecutionRole y ECSTaskRole son roles DISTINTOS — nunca mezclarlos:\n"
                "   • ExecutionRole: para ECS agent (pull imagen, logs)\n"
                "   • TaskRole:      para tu código (S3, DynamoDB, etc.)\n"
                "2. Correr el contenedor como usuario no-root (USER appuser en Dockerfile)\n"
                "3. Security Groups: solo whitelist — permitir exclusivamente lo necesario\n"
                "4. Nunca pasar credenciales AWS como variables de entorno — usar IAM Task Role"
            ),
        },
        "buena_practica": (
            "El endpoint GET /health debe responder 200 solo cuando el modelo ya está cargado desde S3. "
            "ECS envía tráfico al task únicamente después de que el health check pase. "
            "Sin /health correcto, ECS puede enrutar requests antes de que el modelo esté listo "
            "y causar errores 500 en los primeros segundos de cada deploy."
        ),
    },

    "sagemaker": {
        "estructura": [
            "sagemaker-endpoint/",
            "├── inference/",
            "│   ├── inference.py       ← model_fn() + input_fn() + predict_fn() + output_fn()",
            "│   └── requirements.txt",
            "├── build/",
            "│   └── package_model.sh   ← tar.gz del artefacto + inference.py → S3",
            "├── deploy/",
            "│   └── create_endpoint.py ← boto3: create_model + endpoint_config + endpoint",
            "├── .env.example",
            "└── README.md",
        ],
        "archivos": [
            {
                "nombre": "inference/inference.py",
                "descripcion": "Script de inferencia SageMaker con las 4 funciones del contrato de la plataforma.",
                "codigo": (
                    "import os, joblib, json, numpy as np\n\n"
                    "def model_fn(model_dir):\n"
                    "    # SageMaker descomprime model.tar.gz en model_dir automáticamente\n"
                    "    return joblib.load(os.path.join(model_dir, 'model.joblib'))\n\n"
                    "def input_fn(request_body, content_type='application/json'):\n"
                    "    data  = json.loads(request_body)\n"
                    "    order = os.environ.get('FEATURE_ORDER', 'edad,ingresos,ratio_deuda').split(',')\n"
                    "    return np.array([[data[f] for f in order]])\n\n"
                    "def predict_fn(input_data, model):\n"
                    "    return model.predict_proba(input_data)[0].tolist()\n\n"
                    "def output_fn(prediction, accept='application/json'):\n"
                    "    prob = prediction[1]\n"
                    "    return json.dumps({'probabilidad': prob, 'clase': int(prob >= 0.5)})"
                ),
            },
            {
                "nombre": "deploy/create_endpoint.py",
                "descripcion": "Crea o actualiza el endpoint con Data Capture habilitado (10% de muestreo).",
                "codigo": (
                    "import boto3, os\n\n"
                    "sm      = boto3.client('sagemaker')\n"
                    "ROLE    = os.environ['EXECUTION_ROLE_ARN']\n"
                    "BUCKET  = os.environ['SAGEMAKER_BUCKET']\n"
                    "VERSION = os.environ.get('MODEL_VERSION', 'v1')\n\n"
                    "sm.create_model(\n"
                    "    ModelName=f'credit-model-{VERSION}',\n"
                    "    PrimaryContainer={\n"
                    "        'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1',\n"
                    "        'ModelDataUrl': f's3://{BUCKET}/models/{VERSION}/model.tar.gz',\n"
                    "        'Environment': {'FEATURE_ORDER': 'edad,ingresos_mensuales,ratio_deuda'},\n"
                    "    },\n"
                    "    ExecutionRoleArn=ROLE,\n"
                    ")\n"
                    "sm.create_endpoint_config(\n"
                    "    EndpointConfigName=f'credit-config-{VERSION}',\n"
                    "    ProductionVariants=[{'ModelName': f'credit-model-{VERSION}',\n"
                    "                         'InstanceType': 'ml.t3.medium', 'InitialInstanceCount': 1}],\n"
                    "    DataCaptureConfig={\n"
                    "        'EnableCapture': True, 'SamplingPercentage': 10,\n"
                    "        'DestinationS3Uri': f's3://{BUCKET}/data-capture/',\n"
                    "        'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],\n"
                    "    },\n"
                    ")"
                ),
            },
        ],
        "env_vars": [
            {"nombre": "SAGEMAKER_BUCKET",   "descripcion": "Bucket para artefactos del modelo",      "ejemplo": "mi-sagemaker-bucket",                        "es_secreto": False},
            {"nombre": "EXECUTION_ROLE_ARN", "descripcion": "ARN del SageMaker execution role",        "ejemplo": "arn:aws:iam::123456789:role/SageMakerRole",  "es_secreto": False},
            {"nombre": "ENDPOINT_NAME",      "descripcion": "Nombre del endpoint",                     "ejemplo": "credit-endpoint-prod",                       "es_secreto": False},
            {"nombre": "MODEL_VERSION",      "descripcion": "Versión del modelo (versionado en S3)",   "ejemplo": "v1",                                         "es_secreto": False},
            {"nombre": "FEATURE_ORDER",      "descripcion": "Orden de features en el request",         "ejemplo": "edad,ingresos_mensuales,ratio_deuda",        "es_secreto": False},
            {"nombre": "INSTANCE_TYPE",      "descripcion": "Tipo de instancia del endpoint",          "ejemplo": "ml.t3.medium",                               "es_secreto": False},
        ],
        "permisos": {
            "visibilidad": "PRIVADO",
            "visibilidad_detalle": (
                "SageMaker endpoint se invoca solo desde aplicaciones internas via boto3 con IAM.\n"
                "  • NO exponer el endpoint directo a internet\n"
                "  • Si se necesita acceso externo: API Gateway → Lambda → sagemaker-runtime:InvokeEndpoint\n"
                "  • Habilitar VPC endpoint para SageMaker Runtime si el tráfico es sensible (fintech, salud)"
            ),
            "quien_edita_config": (
                "Rol  ml-sagemaker-admin:\n"
                "  • sagemaker:UpdateEndpoint\n"
                "  • sagemaker:UpdateEndpointWeightsAndCapacities  (tráfico A/B)\n"
                "  • sagemaker:CreateEndpointConfig  (para nuevas versiones)\n"
                "  NO: sagemaker:DeleteEndpoint sin aprobación explícita"
            ),
            "quien_accede_logs": (
                "Rol  ml-ops-readonly:\n"
                "  • cloudwatch:GetMetricStatistics  en  /aws/sagemaker/Endpoints/<nombre>\n"
                "  • logs:FilterLogEvents  en  /aws/sagemaker/Endpoints/<nombre>\n"
                "  • sagemaker:DescribeEndpoint  (estado del endpoint)\n"
                "  • s3:GetObject  en  BUCKET/data-capture/*  (análisis de drift)"
            ),
            "quien_modifica_secrets": (
                "SageMaker no usa secretos externos en el flujo básico.\n"
                "  Si el modelo consume APIs externas:\n"
                "  • Guardar en  SSM Parameter Store  como SecureString\n"
                "  • SageMakerExecutionRole: ssm:GetParameter  en ARN específico\n"
                "  EXECUTION_ROLE_ARN viene desde SSM o CI/CD — nunca hardcodeado en código"
            ),
            "permisos_repo": [
                "✓  sagemaker:CreateModel + sagemaker:CreateEndpointConfig  (deploy de nueva versión)",
                "✓  sagemaker:UpdateEndpoint  (actualizar endpoint a nueva config)",
                "✓  s3:PutObject  en  BUCKET/models/*  (subir artefacto)",
                "✓  iam:PassRole  (solo para pasar SageMakerExecutionRole en create_model)",
                "✗  NO: sagemaker:DeleteEndpoint, sagemaker:DeleteModel  (requieren aprobación manual)",
            ],
            "roles": [
                {
                    "nombre": "SageMakerExecutionRole",
                    "uso": "Rol del endpoint en runtime (NO usar AmazonSageMakerFullAccess)",
                    "permisos": [
                        "s3:GetObject  en  arn:aws:s3:::BUCKET/models/*",
                        "s3:PutObject  en  arn:aws:s3:::BUCKET/data-capture/*",
                        "ecr:GetAuthorizationToken  (pull imagen de inferencia)",
                        "cloudwatch:PutMetricData, logs:CreateLogGroup/Stream/PutLogEvents",
                    ],
                    "no_incluir": "AmazonSageMakerFullAccess (permite borrar cualquier endpoint), iam:*, s3:DeleteObject",
                },
                {
                    "nombre": "InvokerRole",
                    "uso": "App o servicio que llama al endpoint (mínimo privilegio extremo)",
                    "permisos": [
                        "sagemaker:InvokeEndpoint  en  arn:aws:sagemaker:<region>:<account>:endpoint/credit-endpoint-prod",
                    ],
                    "no_incluir": "sagemaker:CreateEndpoint, sagemaker:DeleteEndpoint, sagemaker:UpdateEndpoint",
                },
                {
                    "nombre": "MLOpsRole",
                    "uso": "Monitoreo y análisis del endpoint",
                    "permisos": [
                        "sagemaker:DescribeEndpoint, sagemaker:ListEndpoints",
                        "cloudwatch:GetMetricStatistics  en endpoint específico",
                        "s3:GetObject  en  BUCKET/data-capture/*  (análisis de drift)",
                    ],
                    "no_incluir": "sagemaker:UpdateEndpoint, sagemaker:InvokeEndpoint, s3:PutObject",
                },
            ],
            "minimo_privilegio": (
                "1. NUNCA usar AmazonSageMakerFullAccess — permite borrar cualquier endpoint de la cuenta\n"
                "2. InvokerRole: solo sagemaker:InvokeEndpoint en el ARN exacto del endpoint\n"
                "3. SageMakerExecutionRole: limitar s3 a prefijos exactos (models/, data-capture/)\n"
                "4. Habilitar SageMaker Model Monitor para detectar drift sin acceso manual a datos\n"
                "5. IAM Access Analyzer te alerta si el rol tiene permisos no usados"
            ),
        },
        "buena_practica": (
            "Habilita Data Capture en el endpoint config (ya incluido en create_endpoint.py, SamplingPercentage=10). "
            "Captura el 10% de requests y responses en S3 para análisis de data drift. "
            "Con SageMaker Model Monitor puedes detectar cuándo el modelo empieza a degradarse "
            "antes de que el negocio lo reporte."
        ),
    },
}

COLORES = {
    "serverless": "green",
    "batch":      "blue",
    "streaming":  "magenta",
    "containers": "yellow",
    "sagemaker":  "cyan",
}


def _print_implementation_guide(
    arch_id: str,
    arch_nombre: str,
    config_steps: list,
    sustituciones: list,
) -> None:
    guide = IMPL_GUIDES.get(arch_id)
    if not guide:
        console.print(f"  [yellow]Guía no disponible para {arch_nombre}.[/yellow]")
        return

    color = COLORES.get(arch_id, "white")
    console.print()
    console.print(Rule(f"[bold {color}]Guía de implementación — {arch_nombre}[/bold {color}]"))

    if sustituciones:
        subs_note = "  ".join(
            f"[dim]{s.servicio_requerido}[/dim] → [bold green]{s.servicio_sustituto}[/bold green]"
            for s in sustituciones
        )
        console.print(Panel(
            f"[bold yellow]Sustituciones activas en esta guía:[/bold yellow]  {subs_note}\n"
            "[dim]Los pasos de despliegue usan el servicio original; reemplaza por el sustituto indicado.[/dim]",
            border_style="yellow", padding=(0, 1),
        ))

    # ── 1. Estructura del proyecto ────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]1. Estructura mínima del proyecto[/bold]", style="dim"))
    console.print(Panel("\n".join(guide["estructura"]), border_style="dim"))

    # ── 2. Archivos necesarios y código básico ────────────────────────────────
    console.print()
    console.print(Rule("[bold]2. Archivos necesarios y código básico[/bold]", style="dim"))
    for archivo in guide["archivos"]:
        console.print(f"\n  [bold cyan]{archivo['nombre']}[/bold cyan]")
        console.print(f"  [dim]{archivo['descripcion']}[/dim]")
        console.print(Panel(archivo["codigo"], border_style="dim", padding=(0, 1)))

    # ── 3. Comandos de despliegue ─────────────────────────────────────────────
    if config_steps:
        console.print()
        console.print(Rule("[bold]3. Comandos de despliegue[/bold]", style="dim"))
        for paso in config_steps:
            console.print(
                f"\n  [bold cyan]Paso {paso.numero}[/bold cyan] — [bold]{paso.titulo}[/bold]\n"
                f"  [dim]{paso.descripcion}[/dim]"
            )
            if paso.comando:
                console.print(Panel(paso.comando, border_style="dim", padding=(0, 1)))

    # ── 4. Variables de entorno ───────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]4. Variables de entorno[/bold]", style="dim"))
    env_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    env_table.add_column("Variable",    style="bold cyan", no_wrap=True)
    env_table.add_column("Descripción")
    env_table.add_column("Ejemplo",     style="dim")
    env_table.add_column("¿Secreto?",   justify="center")
    for ev in guide["env_vars"]:
        secreto = "[bold red]🔒 Sí[/bold red]" if ev["es_secreto"] else "[green]No[/green]"
        env_table.add_row(ev["nombre"], ev["descripcion"], ev["ejemplo"], secreto)
    console.print()
    console.print(env_table)
    console.print(
        "\n  [dim]Las marcadas 🔒 deben ir en AWS Secrets Manager o SSM Parameter Store "
        "(SecureString) — nunca en texto plano ni en el repositorio.[/dim]"
    )

    # ── 5. Permisos de acceso y roles en AWS ─────────────────────────────────
    p = guide["permisos"]
    console.print()
    console.print(Rule("[bold]5. Permisos de acceso y roles en AWS[/bold]", style="dim"))

    vis_color = {"PÚBLICO": "red", "RESTRINGIDO": "yellow", "PRIVADO": "green",
                 "PRIVADO/INTERNO": "green"}.get(p["visibilidad"], "white")
    console.print()
    console.print(Panel(
        f"[bold {vis_color}]{p['visibilidad']}[/bold {vis_color}]\n\n{p['visibilidad_detalle']}",
        title="[bold]Visibilidad del despliegue[/bold]",
        border_style=vis_color,
    ))
    console.print()
    console.print(Panel(p["quien_edita_config"],
                        title="[bold]Quién puede editar configuración[/bold]", border_style="dim"))
    console.print()
    console.print(Panel(p["quien_accede_logs"],
                        title="[bold]Quién puede acceder a logs, métricas y errores[/bold]", border_style="dim"))
    console.print()
    console.print(Panel(p["quien_modifica_secrets"],
                        title="[bold]Quién puede modificar variables de entorno o secretos[/bold]", border_style="dim"))
    console.print()
    console.print(Panel(
        "\n".join(
            f"  [green]✓[/green] {item}" if item.startswith("✓") else
            f"  [red]✗[/red] {item[2:]}" if item.startswith("✗") else
            f"  {item}"
            for item in p["permisos_repo"]
        ),
        title="[bold]Permisos del repositorio de código / CI-CD[/bold]",
        border_style="dim",
    ))

    console.print()
    console.print(Rule("[dim]Roles mínimos recomendados[/dim]", style="dim"))
    for rol in p["roles"]:
        permisos_str = "\n".join(f"  [green]✓[/green] {perm}" for perm in rol["permisos"])
        console.print(Panel(
            f"[dim]Uso:[/dim]  {rol['uso']}\n\n"
            f"[bold]Permisos:[/bold]\n{permisos_str}\n\n"
            f"[dim]NO incluir:[/dim]  [red]{rol['no_incluir']}[/red]",
            title=f"[bold cyan]{rol['nombre']}[/bold cyan]",
            border_style="cyan",
        ))

    console.print()
    console.print(Panel(
        p["minimo_privilegio"],
        title="[bold yellow]Cómo aplicar el principio de mínimo privilegio[/bold yellow]",
        border_style="yellow",
    ))

    # ── 6. Buena práctica final ───────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]💡[/bold green]  {guide['buena_practica']}",
        title="[bold green]6. Buena práctica final[/bold green]",
        border_style="green",
    ))
    console.print()
    console.print(Rule(style="dim"))


def _print_header(case_input: CaseInput) -> None:
    console.print()
    console.print(Rule("[bold]Agente Recomendador de Arquitecturas ML[/bold]", style="dim"))
    console.print(Panel(
        f"[bold]{case_input.descripcion}[/bold]\n"
        f"[dim]Tipo: {case_input.tipo_modelo} · "
        f"Latencia: {case_input.latencia_requerida_ms:,.0f} ms · "
        f"Frecuencia: {case_input.frecuencia_inferencia} · "
        f"Presupuesto: ${case_input.presupuesto_mensual_usd}/mes[/dim]",
        title="[bold cyan]Caso de Uso[/bold cyan]",
        border_style="cyan",
    ))


def _print_winner(result) -> None:
    arch = result.recomendacion
    color = COLORES.get(arch.id, "white")
    servicios = ", ".join(arch.servicios_aws)
    console.print(Panel(
        f"[bold {color}]{arch.nombre}[/bold {color}]  "
        f"[dim]({arch.id})[/dim]\n\n"
        f"[white]{result.justificacion}[/white]\n\n"
        f"[dim]Servicios AWS:[/dim]  {servicios}\n"
        f"[dim]Confianza:[/dim]  {result.confianza:.0%}",
        title=f"[bold green]★  RECOMENDACIÓN  —  {arch.score_total:.2f}/10[/bold green]",
        border_style="green",
    ))


def _print_criterio(result) -> None:
    console.print()
    console.print(f"[bold yellow]Criterio decisivo:[/bold yellow]  {result.criterio_decisivo}")


def _print_ranking(result) -> None:
    console.print()
    table = Table(title="Ranking completo", show_header=True, header_style="bold dim")
    table.add_column("#",        width=4, justify="right")
    table.add_column("Arquitectura",   width=26)
    table.add_column("Score",   width=9, justify="right")
    table.add_column("Latencia", width=9, justify="right")
    table.add_column("Frec.",    width=9, justify="right")
    table.add_column("Presup.",  width=9, justify="right")
    table.add_column("Escal.",   width=9, justify="right")
    table.add_column("Expér.",   width=9, justify="right")

    for arch in result.ranking:
        color = COLORES.get(arch.id, "white")
        winner_marker = " ★" if arch.posicion == 1 else ""
        d = arch.desglose
        table.add_row(
            str(arch.posicion),
            Text(arch.nombre + winner_marker, style=f"bold {color}" if arch.posicion == 1 else color),
            Text(f"{arch.score_total:.2f}", style="bold green" if arch.posicion == 1 else ""),
            f"{d['latencia']:.1f}",
            f"{d['frecuencia']:.1f}",
            f"{d['presupuesto']:.1f}",
            f"{d['escalabilidad']:.1f}",
            f"{d['experiencia']:.1f}",
        )

    console.print(table)


def _print_advertencias(result) -> None:
    if not result.advertencias:
        return
    console.print()
    for adv in result.advertencias:
        if adv.tipo == "restriccion_dura":
            icon, style = "🚨", "bold red"
        elif adv.tipo == "restriccion_presupuestal":
            icon, style = "💸", "bold red"
        elif adv.tipo == "perfil_presupuestal":
            icon, style = "💡", "bold yellow"
        elif adv.tipo == "sustitucion_servicio":
            icon, style = "🔄", "bold cyan"
        elif adv.tipo == "servicio_no_disponible":
            icon, style = "🚫", "bold red"
        else:
            icon, style = "⚠️", "yellow"
        console.print(f"  {icon}  [{style}]{adv.mensaje}[/{style}]")


def _print_sustituciones(result) -> None:
    """Shows substitution proposals for architectures that are viable with a substitute service."""
    arches_con_sub = [a for a in result.ranking if a.sustituciones_propuestas]
    if not arches_con_sub:
        return
    console.print()
    console.print(Rule("[bold]Sustituciones de servicios sugeridas[/bold]", style="dim"))
    for arch in arches_con_sub:
        color = COLORES.get(arch.id, "white")
        console.print(f"\n  [{color}]{arch.nombre}[/{color}]")
        for sub in arch.sustituciones_propuestas:
            console.print(Panel(
                f"[dim]Servicio requerido:[/dim]  [red]{sub.servicio_requerido}[/red]  [dim](no disponible)[/dim]\n"
                f"[dim]Sustituto sugerido:[/dim]  [bold green]{sub.servicio_sustituto}[/bold green]  [dim](disponible en la org)[/dim]\n\n"
                f"[dim]Impacto en costo:[/dim]    {sub.impacto_costo}\n"
                f"[dim]Impacto en latencia:[/dim] {sub.impacto_latencia}\n\n"
                f"{sub.nota}",
                border_style="cyan",
                padding=(0, 1),
            ))


def _print_config(result) -> None:
    console.print()
    console.print(Rule("[bold]Quickstart de configuración[/bold]", style="dim"))
    for paso in result.configuracion_inicial:
        console.print(
            f"\n  [bold cyan]Paso {paso.numero}[/bold cyan] — [bold]{paso.titulo}[/bold]\n"
            f"  [dim]{paso.descripcion}[/dim]"
        )
        if paso.comando:
            console.print(Panel(paso.comando, border_style="dim", padding=(0, 1)))


def _print_alternativas(result) -> None:
    if not result.alternativas_viables:
        return
    console.print()
    console.print(Rule("[bold]Alternativas viables[/bold]", style="dim"))
    for alt in result.alternativas_viables:
        color = COLORES.get(alt.id, "white")
        console.print(
            f"\n  [{color}]{alt.nombre}[/{color}]  [dim]({alt.score_total:.2f}/10)[/dim]\n"
            f"  [bold]Cuándo elegirla:[/bold] {alt.cuando_elegirla}\n"
            f"  [dim]Trade-off vs. {result.recomendacion.nombre}:[/dim] {alt.trade_off}"
        )


def _print_servicios_detalle(result) -> None:
    if not result.servicios_detalle:
        return
    console.print()
    console.print(Rule("[bold]Servicios AWS recomendados — ficha del catálogo[/bold]", style="dim"))
    for svc in result.servicios_detalle:
        v = svc.vectores
        console.print(
            f"\n  [bold cyan]{svc.service}[/bold cyan]  [dim]{svc.category}[/dim]\n"
            f"  {svc.description}\n"
            f"  [dim]Vectores:[/dim]"
            f"  costo={v['cost_efficiency']}/5"
            f"  simplicidad={v['simplicity']}/5"
            f"  escala={v['scalability']}/5"
            f"  latencia={v['low_latency']}/5"
            f"  gestionado={v['managed_level']}/5"
            f"  principiante={'✓' if svc.beginner_friendly else '✗'}"
        )
        if svc.constraints:
            console.print(f"  [yellow]Restricción:[/yellow] [dim]{svc.constraints}[/dim]")
        console.print(f"  [dim]{svc.official_url}[/dim]")


def _print_validacion_catalogo(result) -> None:
    if not result.notas_validacion_catalogo:
        return
    console.print()
    console.print(Rule("[bold]Cross-validación con catálogo[/bold]", style="dim"))
    for note in result.notas_validacion_catalogo:
        console.print(f"\n  [bold yellow]Δ {note.criterio}[/bold yellow]  {note.nota}")


def _print_service_analysis(analysis: ServiceAnalysisResult) -> None:
    console.print()
    console.print(Rule("[bold]Resultado del análisis de servicios[/bold]", style="dim"))

    if analysis.es_viable:
        status_lines = []
        for svc in analysis.servicios_ok:
            status_lines.append(f"  [green]✓[/green] {svc}  [dim](disponible)[/dim]")
        for sub in analysis.sustituciones:
            status_lines.append(
                f"  [yellow]↻[/yellow] {sub.servicio_requerido}  "
                f"[dim]→ sustituido por[/dim]  [bold green]{sub.servicio_sustituto}[/bold green]"
            )
        console.print(Panel(
            "[bold green]✓ Arquitectura viable con tus servicios[/bold green]\n\n"
            + "\n".join(status_lines),
            border_style="green",
        ))
        if analysis.sustituciones:
            console.print()
            console.print("[bold yellow]Detalle de sustituciones:[/bold yellow]")
            for sub in analysis.sustituciones:
                console.print(Panel(
                    f"[dim]Requería:[/dim]   [red]{sub.servicio_requerido}[/red]\n"
                    f"[dim]Sustituto:[/dim]  [bold green]{sub.servicio_sustituto}[/bold green]\n\n"
                    f"[dim]Impacto costo:[/dim]    {sub.impacto_costo}\n"
                    f"[dim]Impacto latencia:[/dim] {sub.impacto_latencia}\n\n"
                    f"{sub.nota}",
                    border_style="yellow",
                    padding=(0, 1),
                ))
        if analysis.configuracion_inicial:
            console.print()
            console.print(Rule("[bold]Quickstart de configuración[/bold]", style="dim"))
            for paso in analysis.configuracion_inicial:
                console.print(
                    f"\n  [bold cyan]Paso {paso.numero}[/bold cyan] — [bold]{paso.titulo}[/bold]\n"
                    f"  [dim]{paso.descripcion}[/dim]"
                )
                if paso.comando:
                    console.print(Panel(paso.comando, border_style="dim", padding=(0, 1)))
    else:
        faltantes_str = "\n".join(f"  [red]✗[/red] {s}" for s in analysis.faltantes)
        console.print(Panel(
            "[bold red]✗ Arquitectura no viable — servicios sin sustituto disponible[/bold red]\n\n"
            + faltantes_str,
            border_style="red",
        ))
        if analysis.alternativa_precio:
            alt = analysis.alternativa_precio
            color = COLORES.get(alt.id, "white")
            min_cost = _ARCH_MIN_COST_USD.get(alt.id, 0)
            console.print()
            console.print(Panel(
                f"[bold {color}]{alt.nombre}[/bold {color}]  [dim]({alt.id})[/dim]\n\n"
                f"Arquitectura más económica que puedes desplegar con tus servicios actuales.\n"
                f"[dim]Score original:[/dim]  {alt.score_total}/10\n"
                f"[dim]Costo mínimo:[/dim]    ~${min_cost}/mes",
                title="[bold yellow]💡 Recomendación por precio[/bold yellow]",
                border_style="yellow",
            ))
            if analysis.alternativa_precio_sustituciones:
                console.print()
                console.print("[bold cyan]Sustituciones aplicadas:[/bold cyan]")
                for sub in analysis.alternativa_precio_sustituciones:
                    console.print(Panel(
                        f"[dim]Requería:[/dim]   [red]{sub.servicio_requerido}[/red]\n"
                        f"[dim]Sustituto:[/dim]  [bold green]{sub.servicio_sustituto}[/bold green]\n\n"
                        f"[dim]Impacto costo:[/dim]    {sub.impacto_costo}\n"
                        f"[dim]Impacto latencia:[/dim] {sub.impacto_latencia}\n\n"
                        f"{sub.nota}",
                        border_style="cyan",
                        padding=(0, 1),
                    ))
            if analysis.alternativa_precio_config:
                console.print()
                console.print(Rule(f"[bold]Quickstart — {alt.nombre}[/bold]", style="dim"))
                for paso in analysis.alternativa_precio_config:
                    console.print(
                        f"\n  [bold cyan]Paso {paso.numero}[/bold cyan] — [bold]{paso.titulo}[/bold]\n"
                        f"  [dim]{paso.descripcion}[/dim]"
                    )
                    if paso.comando:
                        console.print(Panel(paso.comando, border_style="dim", padding=(0, 1)))
        else:
            console.print()
            console.print(
                "  [yellow]No se encontró arquitectura viable con los servicios indicados.\n"
                "  Considera habilitar al menos: AWS Lambda, Amazon S3, Amazon ECS o AWS Fargate.[/yellow]"
            )

    console.print()
    console.print(Rule(style="dim"))


def _interactive_servicios_mode() -> None:
    """
    Two-step interactive mode:
      1. Show the full scored ranking (no service filter).
      2. User picks an architecture; agent asks which services the company has.
         — If viable (all required services present or substitutable): shows substitutions + quickstart.
         — If not viable: recommends the cheapest architecture the company can actually deploy.
    """
    console.print()
    console.print(Rule("[bold]Agente Recomendador — Análisis de Servicios[/bold]", style="dim"))

    # Step 1: pick or enter case
    console.print()
    casos_str = " / ".join(CASOS.keys())
    opcion = console.input(
        f"  ¿Caso predefinido? [[bold]{casos_str}[/bold]] o ENTER para ingresar manualmente: "
    ).strip().lower()
    case = CASOS.get(opcion) or _interactive_input()

    # Full recommendation — no service filter
    result = recommend(case)
    _print_header(result.input_recibido)
    _print_ranking(result)
    _print_winner(result)
    _print_criterio(result)
    _print_advertencias(result)

    # Step 2: pick architecture
    console.print()
    console.print(Rule("[bold]Paso 2 — ¿Qué servicios tiene tu empresa?[/bold]", style="dim"))
    arch_by_pos = {str(a.posicion): a.id for a in result.ranking}

    while True:
        sel = console.input(
            "\n  [bold cyan]¿Cuál arquitectura te interesa?[/bold cyan] "
            "[dim](número 1-5)[/dim]: "
        ).strip()
        chosen_id = arch_by_pos.get(sel)
        if chosen_id:
            break
        console.print("  [red]Ingresa un número del 1 al 5.[/red]")

    chosen_arch = next(a for a in result.ranking if a.id == chosen_id)
    servicios_relevantes = get_relevant_services_for_arch(chosen_id)

    # Show required + substitutes so user can check which they have
    console.print()
    lines = []
    for i, s in enumerate(servicios_relevantes, 1):
        if s["tipo"] == "requerido":
            tipo_label = "[dim](requerido)[/dim]"
        else:
            tipo_label = f"[dim](sustituto de {s['sustituye']})[/dim]"
        lines.append(f"  [bold]{i}.[/bold] {s['service']}  {tipo_label}")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold cyan]Servicios relevantes — {chosen_arch.nombre}[/bold cyan]",
        border_style="cyan",
    ))

    svcs_input = console.input(
        "\n  Ingresa los [bold]números[/bold] que tu empresa tiene "
        "[dim](ej: 1,3) — ENTER si ninguno[/dim]: "
    ).strip()

    if svcs_input:
        indices = [
            int(x.strip()) - 1
            for x in svcs_input.split(",")
            if x.strip().isdigit() and 1 <= int(x.strip()) <= len(servicios_relevantes)
        ]
        servicios_usuario = [servicios_relevantes[i]["service"] for i in indices]
    else:
        servicios_usuario = []

    analysis = analyze_services_for_arch(chosen_id, servicios_usuario, result.ranking)  # type: ignore[arg-type]

    # If not viable and no alternative found, the user may have services for other
    # architectures that we haven't asked about yet.  Do a second round.
    if not analysis.es_viable and analysis.alternativa_precio is None:
        already_asked = {s["service"] for s in servicios_relevantes}
        extra: list[dict] = []
        seen_extra: set[str] = set()
        for aid in ["serverless", "batch", "streaming", "containers", "sagemaker"]:
            if aid == chosen_id:
                continue
            for s in get_relevant_services_for_arch(aid):
                if s["service"] not in already_asked and s["service"] not in seen_extra:
                    extra.append(s)
                    seen_extra.add(s["service"])

        if extra:
            console.print()
            extra_lines = [
                f"  [bold]{i+1}.[/bold] {s['service']}"
                + (f"  [dim](sustituto de {s['sustituye']})[/dim]" if s["tipo"] == "sustituto" else "")
                for i, s in enumerate(extra)
            ]
            console.print(Panel(
                "\n".join(extra_lines),
                title="[bold yellow]Para buscar alternativa — ¿tiene tu empresa alguno de estos?[/bold yellow]",
                border_style="yellow",
            ))
            extra_input = console.input(
                "\n  Números separados por coma [dim](ENTER si ninguno)[/dim]: "
            ).strip()
            if extra_input:
                extra_idx = [
                    int(x.strip()) - 1
                    for x in extra_input.split(",")
                    if x.strip().isdigit() and 1 <= int(x.strip()) <= len(extra)
                ]
                servicios_usuario = servicios_usuario + [extra[i]["service"] for i in extra_idx]
                analysis = analyze_services_for_arch(chosen_id, servicios_usuario, result.ranking)

    _print_service_analysis(analysis)

    # Confirmation → full implementation guide
    guide_arch_id    = None
    guide_arch_nombre = None
    guide_steps: list = []
    guide_subs:  list = []

    if analysis.es_viable:
        guide_arch_id     = analysis.arch.id
        guide_arch_nombre = analysis.arch.nombre
        guide_steps       = analysis.configuracion_inicial
        guide_subs        = analysis.sustituciones
    elif analysis.alternativa_precio:
        guide_arch_id     = analysis.alternativa_precio.id
        guide_arch_nombre = analysis.alternativa_precio.nombre
        guide_steps       = analysis.alternativa_precio_config
        guide_subs        = analysis.alternativa_precio_sustituciones

    if guide_arch_id:
        console.print()
        confirm = console.input(
            f"  [bold cyan]¿Quieres ver la guía completa de implementación "
            f"para {guide_arch_nombre}?[/bold cyan] [dim][s/n][/dim]: "
        ).strip().lower()
        if confirm in ("s", "si", "sí", "y", "yes"):
            _print_implementation_guide(guide_arch_id, guide_arch_nombre, guide_steps, guide_subs)


def _interactive_input() -> CaseInput:
    console.print("\n[bold cyan]Ingrese las características del caso de uso:[/bold cyan]\n")

    def ask(label: str, default: str) -> str:
        val = console.input(f"  [dim]{label}[/dim] [[bold]{default}[/bold]]: ").strip()
        return val if val else default

    desc     = ask("Descripción", "Mi modelo de ML")
    tipo     = ask("Tipo de modelo", "clasificacion_binaria")
    lat      = float(ask("Latencia requerida (ms)", "800"))
    freq     = ask("Frecuencia inferencia [baja/media/alta/continua]", "baja")
    vol      = float(ask("Volumen de datos por request (KB)", "5"))
    presup   = float(ask("Presupuesto mensual (USD)", "50"))
    escala   = ask("Escalabilidad requerida [baja/media/alta]", "media")
    experta  = ask("Experiencia técnica del equipo [baja/media/alta]", "media")

    return CaseInput(
        descripcion=desc,
        tipo_modelo=tipo,
        latencia_requerida_ms=lat,
        frecuencia_inferencia=freq,  # type: ignore[arg-type]
        volumen_datos_kb=vol,
        presupuesto_mensual_usd=presup,
        escalabilidad_requerida=escala,  # type: ignore[arg-type]
        experiencia_tecnica=experta,     # type: ignore[arg-type]
    )


def run_demo(result: RecommendationResult) -> None:
    _print_header(result.input_recibido)
    _print_winner(result)
    _print_criterio(result)
    _print_advertencias(result)
    _print_sustituciones(result)
    _print_ranking(result)
    _print_config(result)
    _print_alternativas(result)
    _print_servicios_detalle(result)
    _print_validacion_catalogo(result)
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        "[dim]Motor: engine.py · Pesos: latencia 25% | frecuencia 25% | "
        "presupuesto 20% | escalabilidad 15% | experiencia 15%[/dim]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo terminal del agente recomendador")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--caso",
        choices=list(CASOS.keys()),
        default="serverless",
        help="Caso pre-definido a ejecutar (default: serverless)",
    )
    group.add_argument(
        "--yaml",
        metavar="PATH",
        help="YAML con el caso de uso (ruta local o s3://bucket/key)",
    )
    group.add_argument(
        "--interactivo",
        action="store_true",
        help="Ingresar los valores del caso manualmente",
    )
    group.add_argument(
        "--interactivo-servicios",
        action="store_true",
        help="Muestra el ranking completo y luego analiza si puedes desplegar con tus servicios",
    )
    args = parser.parse_args()

    try:
        if args.yaml:
            console.print(f"\n[dim]Cargando YAML:[/dim] {args.yaml}")
            result = recommend_from_yaml(args.yaml)
            run_demo(result)
        elif args.interactivo:
            result = recommend(_interactive_input())
            run_demo(result)
        elif args.interactivo_servicios:
            _interactive_servicios_mode()
            return
        else:
            result = recommend(CASOS[args.caso])
            run_demo(result)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Archivo no encontrado — {e}")
        sys.exit(1)
    except KeyError as e:
        console.print(f"[bold red]Error:[/bold red] Campo requerido faltante en el YAML — {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Demo cancelada.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
