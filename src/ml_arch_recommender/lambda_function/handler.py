"""
AWS Lambda handler — inferencia genérica dirigida por config.yaml.

CÓMO FUNCIONA:
  Este handler no sabe qué modelo va a ejecutar hasta que lee config.yaml de S3.
  El YAML es el contrato: define las features, el framework, los umbrales y las
  etiquetas de salida. Cambiar el modelo no requiere redesplegar este código.

FLUJO DE COLD START:
  1. Descarga config.yaml de S3  (MODEL_BUCKET + CONFIG_KEY)
  2. Parsea el YAML → obtiene ruta del modelo, lista de features, umbrales
  3. Descarga el artefacto del modelo usando la ruta del config
  4. Descarga el scaler por separado si config.paths.scaler_file no es null
  5. Todo queda cacheado en memoria para las invocaciones warm

FLUJO POR REQUEST:
  1. Parsea el JSON de entrada (acepta API Gateway proxy o dict directo)
  2. Valida que todos los campos de config.features están presentes
  3. Construye numpy array en el orden exacto de config.features
  4. Aplica scaler si preprocessing.normalize = true
  5. Ejecuta predict_proba → clasifica con config.inference.threshold
  6. Responde con las etiquetas de config.inference.output_labels

PARA CAMBIAR EL MODELO (sin redesplegar Lambda):
  1. Actualiza configs/model_config.yaml con el nuevo archivo y features
  2. aws s3 cp configs/model_config.yaml s3://TU-BUCKET/models/config.yaml
  3. aws s3 cp nuevo_modelo.joblib s3://TU-BUCKET/models/
  4. El próximo cold start de Lambda carga el nuevo modelo automáticamente

VARIABLES DE ENTORNO:
  MODEL_BUCKET  — nombre del bucket S3 (ej: "mi-bucket-modelos")
  CONFIG_KEY    — ruta al config.yaml dentro del bucket (default: "models/config.yaml")

MODELOS COMPATIBLES:
  - sklearn Pipeline con predict_proba() — recomendado, incluye preprocesamiento
  - sklearn estimador + scaler separado  — configurar scaler_file en el YAML
  - XGBoost / LightGBM serializado con joblib — deben tener predict_proba()
"""

from __future__ import annotations

import io
import json
import logging
import os

import boto3
import joblib
import numpy as np
import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ DE ARTEFACTOS
# El módulo persiste entre invocaciones Lambda del mismo contenedor (warm).
# Los tres objetos se cargan UNA SOLA VEZ en el cold start.
# ─────────────────────────────────────────────────────────────────────────────
_model = None
_scaler = None
_config: dict | None = None


def _load_artifacts() -> tuple[object, object | None, dict]:
    """
    Cold start: descarga config.yaml + artefactos del modelo desde S3.
    Invocaciones warm reutilizan los objetos ya cargados en memoria.
    Retorna (model, scaler_o_None, config_dict).
    """
    global _model, _scaler, _config

    if _model is not None:
        return _model, _scaler, _config

    bucket     = os.environ["MODEL_BUCKET"]
    config_key = os.environ.get("CONFIG_KEY", "models/config.yaml")
    s3         = boto3.client("s3")

    # 1. Cargar y parsear config.yaml
    buf = io.BytesIO()
    s3.download_fileobj(bucket, config_key, buf)
    buf.seek(0)
    _config = yaml.safe_load(buf)

    n_features = len(_config.get("features", []))
    logger.info(
        "Config cargada: %s v%s | framework=%s | features=%d",
        _config["model"]["name"],
        _config["model"]["version"],
        _config["model"]["framework"],
        n_features,
    )

    # 2. Cargar modelo usando la ruta del config
    model_key = _config["paths"]["model_file"]
    buf2 = io.BytesIO()
    s3.download_fileobj(bucket, model_key, buf2)
    buf2.seek(0)
    _model = joblib.load(buf2)
    logger.info("Modelo cargado: tipo=%s | key=%s", type(_model).__name__, model_key)

    # 3. Cargar scaler separado si está configurado
    scaler_key = (_config.get("paths") or {}).get("scaler_file")
    if scaler_key:
        buf3 = io.BytesIO()
        s3.download_fileobj(bucket, scaler_key, buf3)
        buf3.seek(0)
        _scaler = joblib.load(buf3)
        logger.info("Scaler cargado: tipo=%s | key=%s", type(_scaler).__name__, scaler_key)

    return _model, _scaler, _config


def _build_response(status_code: int, body: dict) -> dict:
    """Respuesta HTTP compatible con API Gateway (Lambda Proxy Integration)."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


def _parse_body(event: dict) -> dict:
    """
    Extrae el body del evento Lambda.
    API Gateway envía el body como string JSON; las pruebas locales lo envían como dict.
    """
    raw = event.get("body", event)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw if isinstance(raw, dict) else event


def lambda_handler(event: dict, context) -> dict:
    """
    Punto de entrada de AWS Lambda.

    INPUT — JSON con las features definidas en config.features:
      El orden de los campos en el JSON no importa.
      El handler construye el array siempre en el orden del config.

    OUTPUT exitoso (200):
      {
        "prediccion": "ALTO",                    ← etiqueta de config.inference.output_labels
        "probabilidades": {"BAJO": 0.18, "ALTO": 0.82},
        "features_recibidas": { ...echo del input... },
        "modelo_nombre": "credit-risk-model",
        "modelo_version": "1.0.0",
        "modelo_tipo": "clasificacion_binaria",
        "arquitectura_despliegue": "serverless-lambda"
      }

    ERRORES:
      400 — campos faltantes en el JSON o valores no numéricos
      500 — error interno (config corrupto, S3 inaccesible, modelo incompatible)
    """
    logger.info("Evento recibido: %s", json.dumps(event))

    # CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return _build_response(200, {"message": "OK"})

    try:
        model, scaler, config = _load_artifacts()
        body = _parse_body(event)

        # ── Parámetros del config ────────────────────────────────────────────
        feature_names: list[str] = config["features"]
        threshold: float         = config.get("inference", {}).get("threshold", 0.5)
        output_labels: dict      = config.get("inference", {}).get("output_labels", {0: "0", 1: "1"})
        return_proba: bool       = config.get("inference", {}).get("return_proba", True)
        normalize: bool          = config.get("preprocessing", {}).get("normalize", False)

        # ── Validación de campos ─────────────────────────────────────────────
        missing = [f for f in feature_names if f not in body]
        if missing:
            return _build_response(400, {
                "error": "Campos requeridos faltantes",
                "campos_faltantes": missing,
                "campos_requeridos": feature_names,
                "modelo": config["model"]["name"],
                "total_features_requeridas": len(feature_names),
            })

        # ── Construcción del vector de features ─────────────────────────────
        # El orden de feature_names garantiza el contrato con el modelo entrenado
        try:
            features = np.array([[float(body[f]) for f in feature_names]])
        except (ValueError, TypeError) as e:
            return _build_response(400, {
                "error": "Valor no numérico en algún campo",
                "detalle": str(e),
                "campos_requeridos": feature_names,
            })

        # ── Preprocesamiento opcional ────────────────────────────────────────
        # Solo se aplica si normalize=true Y hay scaler disponible.
        # Con sklearn Pipeline el escalado ya va integrado en el objeto del modelo.
        if normalize and scaler is not None:
            features = scaler.transform(features)

        # ── Inferencia ───────────────────────────────────────────────────────
        probas         = model.predict_proba(features)[0]
        prob_positiva  = float(probas[1]) if len(probas) > 1 else float(probas[0])
        clase_idx      = 1 if prob_positiva >= threshold else 0
        clasificacion  = str(output_labels.get(clase_idx, clase_idx))

        logger.info(
            "Predicción: %s | prob_positiva=%.4f | modelo=%s v%s | n_features=%d",
            clasificacion, prob_positiva,
            config["model"]["name"], config["model"]["version"],
            len(feature_names),
        )

        # ── Construcción de la respuesta ─────────────────────────────────────
        response: dict = {
            "prediccion": clasificacion,
            "modelo_nombre":  config["model"]["name"],
            "modelo_version": config["model"]["version"],
            "modelo_tipo":    config["model"].get("tipo", ""),
            "arquitectura_despliegue": "serverless-lambda",
        }

        if return_proba:
            response["probabilidades"] = {
                str(output_labels.get(i, i)): round(float(p), 4)
                for i, p in enumerate(probas)
            }
            if len(probas) == 2:
                response["umbral_clasificacion"] = threshold

        response["features_recibidas"] = {f: body[f] for f in feature_names}

        return _build_response(200, response)

    except Exception as exc:
        logger.exception("Error no controlado en lambda_handler")
        return _build_response(500, {
            "error": "Error interno del servidor",
            "detail": str(exc),
        })
