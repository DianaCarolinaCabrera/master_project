"""
Validates the n8n agent's recommendation coherence.
Sends predefined cases to the n8n webhook and checks that the
recommended architecture matches the expected one for each case.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

VALIDATION_CASES = [
    {
        "descripcion": "Fintech tiempo real, bajo presupuesto → SERVERLESS",
        "input": {
            "caso_negocio": {
                "descripcion": "Predicción de riesgo crediticio en tiempo real",
                "tipo_modelo": "clasificacion_binaria",
                "latencia_requerida_ms": 800,
                "frecuencia_inferencia": "baja",
                "volumen_datos_kb": 5,
                "presupuesto_mensual_usd": 30,
                "escalabilidad_requerida": "media",
                "experiencia_tecnica": "media",
                "disponibilidad_requerida": "alta",
            },
            "datos_prueba": {
                "edad": 35,
                "ingresos_mensuales": 3_500_000,
                "ratio_deuda": 0.35,
                "meses_historial_crediticio": 24,
                "num_cuentas_activas": 3,
            },
        },
        "arquitectura_esperada": "serverless",
    },
    {
        "descripcion": "Procesamiento masivo nocturno, sin restricción latencia → BATCH",
        "input": {
            "caso_negocio": {
                "descripcion": "Scoring masivo de cartera crediticia cada noche",
                "tipo_modelo": "clasificacion_binaria",
                "latencia_requerida_ms": 86_400_000,
                "frecuencia_inferencia": "baja",
                "volumen_datos_kb": 500_000,
                "presupuesto_mensual_usd": 300,
                "escalabilidad_requerida": "alta",
                "experiencia_tecnica": "alta",
                "disponibilidad_requerida": "media",
            },
            "datos_prueba": {
                "edad": 28,
                "ingresos_mensuales": 1_200_000,
                "ratio_deuda": 0.70,
                "meses_historial_crediticio": 12,
                "num_cuentas_activas": 2,
            },
        },
        "arquitectura_esperada": "batch",
    },
    {
        "descripcion": "Fraude en transacciones continuas → STREAMING",
        "input": {
            "caso_negocio": {
                "descripcion": "Detección de fraude en flujo continuo de transacciones",
                "tipo_modelo": "clasificacion_binaria",
                "latencia_requerida_ms": 200,
                "frecuencia_inferencia": "continua",
                "volumen_datos_kb": 1,
                "presupuesto_mensual_usd": 500,
                "escalabilidad_requerida": "alta",
                "experiencia_tecnica": "alta",
                "disponibilidad_requerida": "alta",
            },
            "datos_prueba": {
                "edad": 30,
                "ingresos_mensuales": 2_000_000,
                "ratio_deuda": 0.40,
                "meses_historial_crediticio": 48,
                "num_cuentas_activas": 3,
            },
        },
        "arquitectura_esperada": "streaming",
    },
]


def run_validation(webhook_url: str, output_file: str | None = None) -> None:
    console.rule("[bold blue]Validación de Coherencia del Agente n8n")
    console.print(f"[dim]Webhook:[/dim] {webhook_url}\n")

    table = Table(title="Resultados de Validación del Agente")
    table.add_column("Caso", style="cyan", max_width=45)
    table.add_column("Esperado", justify="center")
    table.add_column("Recomendado", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Confianza", justify="right")
    table.add_column("Latencia (ms)", justify="right")
    table.add_column("Estado", justify="center")

    results = []
    passed = 0

    for case in VALIDATION_CASES:
        try:
            t0 = time.perf_counter()
            resp = requests.post(webhook_url, json=case["input"], timeout=20)
            latencia_ms = (time.perf_counter() - t0) * 1000

            data = resp.json()
            rec = data.get("recomendacion", {})
            arch_id = rec.get("arquitectura_id", "unknown")
            score = rec.get("score", 0)
            confianza = rec.get("confianza", 0)

            correcto = arch_id == case["arquitectura_esperada"]
            if correcto:
                passed += 1

            estado = "[green]PASS[/green]" if correcto else "[red]FAIL[/red]"
            table.add_row(
                case["descripcion"],
                case["arquitectura_esperada"],
                arch_id,
                f"{score:.2f}",
                f"{confianza:.2f}",
                f"{latencia_ms:.0f}",
                estado,
            )
            results.append({
                "caso": case["descripcion"],
                "esperado": case["arquitectura_esperada"],
                "obtenido": arch_id,
                "correcto": correcto,
                "score": score,
                "confianza": confianza,
                "latencia_ms": round(latencia_ms, 2),
                "respuesta_completa": data,
            })

        except Exception as exc:
            console.print(f"[red]Error en caso '{case['descripcion']}': {exc}[/red]")
            results.append({"caso": case["descripcion"], "error": str(exc)})

    console.print(table)
    precision = passed / len(VALIDATION_CASES)
    color = "green" if precision == 1.0 else "yellow" if precision >= 0.67 else "red"
    console.print(f"\n[bold {color}]Coherencia del agente: {passed}/{len(VALIDATION_CASES)} ({precision:.0%})[/bold {color}]")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({"precision": precision, "casos": results}, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Resultados guardados en: {output_file}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida coherencia del agente n8n")
    parser.add_argument(
        "--webhook",
        default=os.getenv("N8N_WEBHOOK_URL", ""),
        help="URL del webhook n8n (o N8N_WEBHOOK_URL en .env)",
    )
    parser.add_argument("--output", default=None, help="Archivo JSON para guardar resultados")
    args = parser.parse_args()

    if not args.webhook:
        console.print("[red]Error: especifica --webhook o define N8N_WEBHOOK_URL en .env[/red]")
        raise SystemExit(1)

    run_validation(args.webhook, args.output)


if __name__ == "__main__":
    main()
