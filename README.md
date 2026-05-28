# Agente Recomendador de Arquitecturas ML

**Tesis de Maestría en Analítica de Datos · 2026**
Diana Cabrera · Andrea Mercado · Leonardo Lasso

---

## ¿Qué hace este proyecto?

Este sistema es un **agente inteligente que recomienda la arquitectura AWS óptima para desplegar un modelo de Machine Learning**, considerando criterios de latencia, frecuencia de inferencia, presupuesto, escalabilidad y experiencia técnica del equipo.

El agente **no entrena modelos** — su función es decidir *cómo y dónde* desplegar un modelo que ya fue entrenado. El caso de estudio es un modelo de riesgo crediticio, pero el agente funciona con cualquier modelo sklearn Pipeline.

**Resultado de una consulta:**
- Arquitectura recomendada con score ponderado (0–10)
- Justificación basada en criterios documentados
- Detección de restricciones duras (ej: batch no es viable si latencia ≤ 1s)
- Pasos de configuración listos para ejecutar (AWS CLI)
- Alternativas viables con sus trade-offs

---

## Estructura de carpetas

```
week_apr_23/
│
├── src/ml_arch_recommender/          # Código fuente principal
│   ├── scoring/
│   │   └── engine.py                 # Motor multicriterio — el corazón del agente
│   ├── model/
│   │   └── train.py                  # Genera el artefacto del caso de estudio
│   ├── lambda_function/
│   │   └── handler.py                # Función AWS Lambda para inferencia
│   ├── validation/
│   │   ├── validate_agent.py         # Valida el agente contra n8n (3 casos)
│   │   └── test_endpoint.py          # Prueba el endpoint Lambda desplegado
│   └── demo.py                       # Demo terminal con Rich (sin n8n ni AWS)
│
├── tests/
│   ├── test_scoring_engine.py        # 23 tests del motor (3 capas de validación)
│   └── test_model_train.py           # 5 tests del pipeline de entrenamiento
│
├── n8n/
│   └── workflow_export.json          # Workflow importable directamente en n8n
│
├── deploy/
│   ├── build_lambda.sh               # Empaqueta Lambda como .zip
│   └── deploy.sh                     # Despliega todo en AWS (S3 + Lambda + API GW)
│
├── data/models/                      # Generado por train-model (gitignored)
│   ├── credit_model.joblib           # Artefacto sklearn Pipeline serializado
│   └── model_metadata.json           # Métricas del modelo entrenado
│
├── docs/
│   └── guide.html                    # Guía visual completa del proyecto (abrir en browser)
│
├── .env.example                      # Variables de entorno requeridas
└── pyproject.toml                    # Configuración Poetry + dependencias
```

### Qué hace cada carpeta clave

| Carpeta | Propósito |
|---------|-----------|
| `scoring/` | Motor de recomendación multicriterio. Aquí vive toda la lógica de puntuación, razonamiento y detección de restricciones. |
| `model/` | Prepara el artefacto del caso de estudio. Puedes reemplazar `train.py` con tu propio modelo — ver sección abajo. |
| `lambda_function/` | Código que corre en AWS Lambda. Lee el modelo de S3, recibe el JSON de features, devuelve la predicción. |
| `validation/` | Scripts para verificar que el agente y el endpoint funcionan correctamente antes de presentar. |
| `n8n/` | El workflow del agente listo para importar. Contiene el motor de scoring en JavaScript (espejo de `engine.py`). |
| `deploy/` | Scripts bash que crean todos los recursos AWS necesarios con un solo comando. |

---

## Qué espera n8n como entrada y qué entrega

### Entrada (POST al webhook de n8n)

```json
{
  "descripcion": "Modelo de riesgo crediticio para fintech",
  "tipo_modelo": "clasificacion_binaria",
  "latencia_requerida_ms": 800,
  "frecuencia_inferencia": "baja",
  "volumen_datos_kb": 5,
  "presupuesto_mensual_usd": 30,
  "escalabilidad_requerida": "media",
  "experiencia_tecnica": "media"
}
```

| Campo | Tipo | Valores válidos | Descripción |
|-------|------|-----------------|-------------|
| `latencia_requerida_ms` | float | cualquier número | Tiempo máximo tolerable de respuesta |
| `frecuencia_inferencia` | string | `baja`, `media`, `alta`, `continua` | Con qué frecuencia se hacen predicciones |
| `presupuesto_mensual_usd` | float | cualquier número | Presupuesto mensual en dólares |
| `escalabilidad_requerida` | string | `baja`, `media`, `alta` | Necesidad de crecer horizontalmente |
| `experiencia_tecnica` | string | `baja`, `media`, `alta` | Capacidad del equipo para gestionar infraestructura |

### Salida (respuesta del webhook)

```json
{
  "arquitectura_id": "serverless",
  "arquitectura_nombre": "Serverless (AWS Lambda + API Gateway)",
  "score_total": 9.45,
  "confianza": 0.87,
  "justificacion": "Serverless es la arquitectura óptima para este caso...",
  "criterio_decisivo": "Frecuencia baja e inferencia esporádica...",
  "servicios_aws": ["Lambda", "API Gateway", "S3"],
  "advertencias": [],
  "configuracion_inicial": [
    {
      "numero": 1,
      "titulo": "Crear bucket S3 para el modelo",
      "descripcion": "...",
      "comando": "aws s3 mb s3://microcredit-demo-models --region us-east-1"
    }
  ],
  "alternativas_viables": [
    {
      "id": "containers",
      "nombre": "Containers (ECS Fargate)",
      "score_total": 6.8,
      "cuando_elegirla": "Empresa con política Docker obligatoria",
      "trade_off": "Mayor control pero $50–80/mes mínimo vs. $0 en reposo",
      "costo_estimado_usd": "$50–150/mes"
    }
  ],
  "ranking": [...]
}
```

---

## Cómo conectar tu propio modelo entrenado

El `train.py` del proyecto genera un modelo **sintético** con fines demostrativos. Si tienes un modelo real, puedes reemplazarlo en 3 pasos:

### Paso 1 — Tu modelo debe ser un sklearn Pipeline con `predict_proba()`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipeline.fit(X_train, y_train)
```

Cualquier clasificador funciona (XGBoost, RandomForest, etc.) siempre que el Pipeline tenga `predict_proba()`.

### Paso 2 — Serializar con joblib

```python
import joblib
joblib.dump(pipeline, "data/models/credit_model.joblib")
```

### Paso 3 — Actualizar `FEATURE_ORDER` en el handler Lambda

Abre `src/ml_arch_recommender/lambda_function/handler.py` y actualiza la lista `FEATURE_ORDER` con los nombres de columna de tu modelo, en el **mismo orden** que usaste al entrenar:

```python
FEATURE_ORDER = [
    "tu_feature_1",
    "tu_feature_2",
    "tu_feature_3",
]
```

Luego sube el nuevo `.joblib` a S3 y redespliega:

```bash
aws s3 cp data/models/credit_model.joblib s3://microcredit-demo-models/models/
bash deploy/deploy.sh
```

> **Lo que NO cambia:** el agente de n8n, el motor de scoring, los tests, y toda la infraestructura AWS. Solo cambia el artefacto `.joblib` y la lista de features.

---

## Instalación y primeros pasos

```bash
# 1. Clonar e instalar dependencias
git clone <repo>
cd week_apr_23
pip install poetry
poetry install

# 2. Copiar variables de entorno
cp .env.example .env
# Editar .env con tus valores

# 3. Entrenar el modelo del caso de estudio
poetry run train-model

# 4. Ejecutar la demo del agente en terminal (sin n8n ni AWS)
poetry run demo                   # caso serverless
poetry run demo --caso batch      # banco con scoring nocturno
poetry run demo --caso streaming  # detección de fraude
poetry run demo --interactivo     # ingresar valores manualmente

# 5. Correr los tests
poetry run pytest
```

---

## Cómo ejecutar n8n

El agente está implementado como un workflow n8n. Hay 3 formas de ejecutarlo:

### Opción 1 — n8n Cloud (recomendado para la demo)

1. Crear cuenta gratuita en **app.n8n.cloud**
2. Ir a **Workflows → Import from file** → seleccionar `n8n/workflow_export.json`
3. En **Settings → Variables**, agregar:
   ```
   API_ENDPOINT = https://tu-api-id.execute-api.us-east-1.amazonaws.com/prod/predict
   ```
4. Hacer clic en **Activate** (botón verde)
5. Copiar la URL del webhook del nodo "Recepción de Caso"

**Ventajas:** URL pública HTTPS desde el primer minuto, sin configuración de red, accesible desde cualquier dispositivo. Ideal para presentar al jurado.

### Opción 2 — Docker local

```bash
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# Acceder en: http://localhost:5678
```

Para exponer al jurado con URL pública:
```bash
ngrok http 5678
# ngrok genera una URL tipo: https://a1b2c3d4.ngrok.io
```

### Opción 3 — npx (sin instalación)

```bash
npx n8n
# Disponible en: http://localhost:5678
```

---

## Despliegue completo en AWS

```bash
# Configurar credenciales AWS
aws configure

# Empaquetar y desplegar todo
bash deploy/build_lambda.sh
bash deploy/deploy.sh

# El script crea automáticamente:
# - Bucket S3 para el modelo
# - Rol IAM para Lambda
# - Función Lambda con el handler
# - API Gateway REST con endpoint /predict
```

---

## Correr los tests

```bash
# Todos los tests (23 en scoring + 5 en modelo)
poetry run pytest

# Solo el motor de recomendación
poetry run pytest tests/test_scoring_engine.py -v

# Con cobertura
poetry run pytest --cov=ml_arch_recommender --cov-report=html
```

Los tests validan 3 capas:
1. **Correctitud** — que serverless, batch y streaming se recomienden en los escenarios correctos
2. **Calidad del razonamiento** — que cada criterio tenga una razón documentada, scores entre 0-10, suma ponderada exacta
3. **Detección de restricciones** — que batch sea marcado como restricción dura cuando latencia ≤ 1s

---

## Próximos pasos — extensión a GCP

El motor de scoring está diseñado para ser agnóstico a la nube. Agregar soporte GCP requiere:

1. **Extender `ArchitectureId`** en `engine.py`:
   ```python
   ArchitectureId = Literal[
       "serverless", "batch", "streaming", "containers", "sagemaker",
       "gcp_cloud_run", "gcp_dataflow", "gcp_pubsub", "gcp_gke", "gcp_vertex_ai"
   ]
   ```

2. **Agregar equivalentes GCP** en las funciones `_kb_*`:

   | Arquitectura AWS | Equivalente GCP |
   |-----------------|-----------------|
   | Lambda + API GW | Cloud Run |
   | Glue + Step Functions | Dataflow + Cloud Workflows |
   | Kinesis + Lambda | Pub/Sub + Cloud Run |
   | ECS Fargate | GKE Autopilot |
   | SageMaker Endpoint | Vertex AI Endpoint |

3. **Agregar un parámetro `cloud_provider`** al `CaseInput`:
   ```python
   cloud_provider: Literal["aws", "gcp", "azure", "agnostico"] = "aws"
   ```

4. **Replicar el cambio en el nodo JS de n8n** (`workflow_export.json`, Nodo 3) para mantener el motor Python y el workflow sincronizados.

Esta extensión permitiría que el agente recomiende también arquitecturas GCP y compare entre nubes — caso de uso directo para empresas que ya operan en GCP o tienen restricciones contractuales con AWS.

---

## Guía visual completa

Abre `docs/guide.html` en tu navegador para ver la documentación completa con diagramas, ejemplos de I/O, el script de demo para el jurado, y las instrucciones paso a paso de despliegue.
