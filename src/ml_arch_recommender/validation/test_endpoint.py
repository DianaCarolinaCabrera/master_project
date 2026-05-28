"""
Direct endpoint test for the deployed AWS Lambda + API Gateway.
Run after deployment to verify the live endpoint works correctly.
"""

from __future__ import annotations

import argparse
import os
import time

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

TEST_CASES = [
    {
        "nombre": "Perfil BAJO RIESGO — empleado estable con buen historial",
        "esperado": "BAJO",
        "payload": {
            "edad": 40,
            "ingresos_mensuales": 5_000_000,
            "ratio_deuda": 0.25,
            "meses_historial_crediticio": 60,
            "num_cuentas_activas": 4,
        },
    },
    {
        "nombre": "Perfil ALTO RIESGO — joven, bajos ingresos, historial corto",
        "esperado": "ALTO",
        "payload": {
            "edad": 22,
            "ingresos_mensuales": 900_000,
            "ratio_deuda": 0.78,
            "meses_historial_crediticio": 3,
            "num_cuentas_activas": 1,
        },
    },
    {
        "nombre": "Perfil ALTO RIESGO — alto ratio de deuda",
        "esperado": "ALTO",
        "payload": {
            "edad": 35,
            "ingresos_mensuales": 2_000_000,
            "ratio_deuda": 0.82,
            "meses_historial_crediticio": 36,
            "num_cuentas_activas": 5,
        },
    },
    {
        "nombre": "Perfil BAJO RIESGO — senior con alta capacidad",
        "esperado": "BAJO",
        "payload": {
            "edad": 52,
            "ingresos_mensuales": 8_500_000,
            "ratio_deuda": 0.15,
            "meses_historial_crediticio": 96,
            "num_cuentas_activas": 6,
        },
    },
]


def run_tests(endpoint: str) -> None:
    console.rule("[bold blue]Pruebas del Endpoint AWS Lambda")
    console.print(f"[dim]Endpoint:[/dim] {endpoint}\n")

    table = Table(title="Resultados de Prueba")
    table.add_column("Caso", style="cyan", max_width=40)
    table.add_column("Esperado", justify="center")
    table.add_column("Obtenido", justify="center")
    table.add_column("Prob. Alto", justify="right")
    table.add_column("Latencia (ms)", justify="right")
    table.add_column("Estado", justify="center")

    passed = 0
    for case in TEST_CASES:
        try:
            t0 = time.perf_counter()
            resp = requests.post(endpoint, json=case["payload"], timeout=15)
            latencia_ms = (time.perf_counter() - t0) * 1000

            data = resp.json()
            obtenido = data.get("riesgo", "ERROR")
            prob = data.get("probabilidad_riesgo_alto", 0)
            correcto = obtenido == case["esperado"]
            if correcto:
                passed += 1

            estado = "[green]PASS[/green]" if correcto else "[red]FAIL[/red]"
            table.add_row(
                case["nombre"],
                case["esperado"],
                obtenido,
                f"{prob:.4f}",
                f"{latencia_ms:.0f}",
                estado,
            )
        except Exception as exc:
            table.add_row(case["nombre"], case["esperado"], "ERROR", "-", "-", f"[red]{exc}[/red]")

    console.print(table)
    precision = passed / len(TEST_CASES)
    color = "green" if precision == 1.0 else "yellow" if precision >= 0.75 else "red"
    console.print(f"\n[bold {color}]Precisión: {passed}/{len(TEST_CASES)} ({precision:.0%})[/bold {color}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test del endpoint AWS desplegado")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("API_ENDPOINT", ""),
        help="URL del API Gateway (o variable API_ENDPOINT en .env)",
    )
    args = parser.parse_args()

    if not args.endpoint:
        console.print("[red]Error: especifica --endpoint o define API_ENDPOINT en .env[/red]")
        raise SystemExit(1)

    run_tests(args.endpoint)


if __name__ == "__main__":
    main()
