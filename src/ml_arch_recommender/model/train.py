"""
Entrenamiento y serialización del modelo de riesgo crediticio.

ROL EN EL PROYECTO:
  Este script prepara el ARTEFACTO del caso de estudio para el demo de tesis.
  El modelo en sí NO es el objetivo del proyecto — el objetivo es el AGENTE
  RECOMENDADOR que decide cómo desplegarlo.

CÓMO USAR TU PROPIO MODELO (sin ejecutar este script):
  1. Tu modelo debe ser un sklearn Pipeline con predict_proba()
  2. Serializa: joblib.dump(pipeline, "data/models/mi_modelo.joblib")
  3. Edita configs/model_config.yaml:
       - paths.model_file → nombre de tu .joblib
       - features         → lista de features EN EL MISMO ORDEN de entrenamiento
       - inference.output_labels → las etiquetas de tus clases
  4. Sube ambos archivos a S3 y redespliega: bash deploy/deploy.sh
  NO necesitas tocar handler.py — el YAML es el único contrato.

DATOS SINTÉTICOS:
  Se usan datos generados con reglas de negocio coherentes porque el objetivo
  es demostrar el agente, no construir el mejor modelo de riesgo crediticio.
  Un evaluador puede reemplazar estos datos por datos reales sin cambiar nada
  más en el proyecto.

SALIDA:
  data/models/credit_model.joblib  — artefacto para subir a S3
  data/models/config.yaml          — contrato YAML que lee el Lambda handler
  data/models/model_metadata.json  — métricas del modelo entrenado
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CONTRATO DE FEATURES
# Debe coincidir EXACTAMENTE con FEATURE_ORDER en lambda_function/handler.py
# y con el orden de columnas usado durante el entrenamiento.
# Si agregas o quitas features aquí, actualiza también handler.py.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "edad",                       # Años del solicitante (20–65)
    "ingresos_mensuales",         # Ingresos en COP (800K–10M)
    "ratio_deuda",                # Proporción deuda/ingreso (0.05–0.95)
    "meses_historial_crediticio", # Antigüedad crediticia en meses (1–120)
    "num_cuentas_activas",        # Número de cuentas vigentes (1–10)
]

FEATURE_RANGES = {
    "edad":                       (20, 65),
    "ingresos_mensuales":         (800_000, 10_000_000),
    "ratio_deuda":                (0.05, 0.95),
    "meses_historial_crediticio": (1, 120),
    "num_cuentas_activas":        (1, 10),
}


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos con reglas de negocio coherentes con el sector financiero.
    Las reglas de etiquetado son intencionalmente interpretables para que un jurado
    pueda verificar que el modelo tiene sentido económico, no solo estadístico.

    Si tienes datos reales, reemplaza esta función y llama directamente a train().
    """
    rng = np.random.default_rng(seed)

    edad            = rng.integers(*FEATURE_RANGES["edad"], size=n_samples).astype(float)
    ingresos        = rng.uniform(*FEATURE_RANGES["ingresos_mensuales"], size=n_samples)
    ratio_deuda     = rng.uniform(*FEATURE_RANGES["ratio_deuda"], size=n_samples)
    meses_historial = rng.integers(*FEATURE_RANGES["meses_historial_crediticio"], size=n_samples).astype(float)
    num_cuentas     = rng.integers(*FEATURE_RANGES["num_cuentas_activas"], size=n_samples).astype(float)

    X = np.column_stack([edad, ingresos, ratio_deuda, meses_historial, num_cuentas])

    # Reglas de negocio para etiquetado (1 = ALTO riesgo):
    #   - ratio_deuda > 0.65: persona muy endeudada
    #   - ingresos < 1.5M COP: capacidad de pago insuficiente
    #   - historial < 6 meses: sin historial para evaluar
    #   - ratio > 0.50 Y historial < 18 meses: combinación riesgosa
    riesgo_alto = (
        (ratio_deuda > 0.65)
        | (ingresos < 1_500_000)
        | (meses_historial < 6)
        | ((ratio_deuda > 0.50) & (meses_historial < 18))
    )
    y = riesgo_alto.astype(int)

    return X, y


def build_pipeline() -> Pipeline:
    """
    Construye el Pipeline sklearn que será serializado como artefacto.

    Estructura:
      StandardScaler      → normaliza features al rango de entrenamiento
      LogisticRegression  → clasificador binario (0=BAJO, 1=ALTO riesgo)

    El Pipeline garantiza que el preprocesamiento y la inferencia van siempre
    juntos en el mismo objeto. Lambda solo necesita cargar este Pipeline y
    llamar predict_proba() — sin pasos manuales de preprocesamiento.
    """
    return Pipeline(
        steps=[
            ("scaler",     StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=500, C=1.0)),
        ]
    )


def train(output_dir: Path) -> dict:
    console.rule("[bold blue]Entrenamiento del Modelo de Riesgo Crediticio")

    X, y = generate_synthetic_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    console.print(f"[green]Datos generados:[/green] {len(X_train)} train / {len(X_test)} test")
    console.print(f"[green]Distribución clases:[/green] ALTO={y.sum()} ({y.mean():.1%}) | BAJO={len(y)-y.sum()}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["BAJO", "ALTO"], output_dict=True)

    # Display metrics table
    table = Table(title="Métricas del Modelo")
    table.add_column("Clase", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")

    for label in ["BAJO", "ALTO"]:
        m = report[label]
        table.add_row(label, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1-score']:.3f}")

    console.print(table)
    console.print(f"[bold green]AUC-ROC:[/bold green] {auc:.4f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "credit_model.joblib"
    joblib.dump(pipeline, model_path)

    # Save config.yaml — contrato YAML que usa el Lambda handler
    # Editar este archivo para adaptar el handler a un modelo diferente
    model_config = {
        "model": {
            "name": "credit-risk-model",
            "framework": "sklearn",
            "version": "1.0.0",
            "tipo": "clasificacion_binaria",
            "sector": "financiero",
            "description": "Modelo de riesgo crediticio para microcréditos — caso de estudio tesis",
        },
        "paths": {
            "model_file": "models/credit_model.joblib",
            "scaler_file": None,  # scaler está dentro del sklearn Pipeline
        },
        "features": FEATURE_NAMES,
        "inference": {
            "threshold": 0.5,
            "return_proba": True,
            "output_labels": {0: "BAJO", 1: "ALTO"},
        },
        "preprocessing": {
            "normalize": False,  # StandardScaler ya está dentro del Pipeline
        },
    }
    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(model_config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    )

    # Save metadata
    metadata = {
        "version": "1.0.0",
        "algorithm": "LogisticRegression",
        "features": FEATURE_NAMES,
        "feature_ranges": FEATURE_RANGES,
        "metrics": {
            "auc_roc": round(auc, 4),
            "precision_alto": round(report["ALTO"]["precision"], 4),
            "recall_alto": round(report["ALTO"]["recall"], 4),
            "f1_alto": round(report["ALTO"]["f1-score"], 4),
        },
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
    meta_path = output_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    console.print(f"\n[bold]Modelo guardado en:[/bold] {model_path}")
    console.print(f"[bold]Config YAML guardado en:[/bold] {config_path}")
    console.print(f"[bold]Metadata guardada en:[/bold] {meta_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de riesgo crediticio")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Directorio de salida para el modelo serializado",
    )
    args = parser.parse_args()
    train(args.output_dir)


if __name__ == "__main__":
    main()
