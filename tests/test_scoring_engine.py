"""
Tests for the multicriterion scoring engine.

Validates three layers:
  1. Recommendation correctness  — right architecture for each scenario
  2. Reasoning quality           — justification has content and structure
  3. Validation / warnings       — hard constraints and tensions are detected
"""

import pytest
from ml_arch_recommender.scoring.engine import CaseInput, recommend, WEIGHTS, _adaptive_weights


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _case(**overrides) -> CaseInput:
    defaults = dict(
        descripcion="Test case",
        tipo_modelo="clasificacion_binaria",
        latencia_requerida_ms=800,
        frecuencia_inferencia="baja",
        volumen_datos_kb=5,
        presupuesto_mensual_usd=30,
        escalabilidad_requerida="media",
        experiencia_tecnica="media",
    )
    defaults.update(overrides)
    return CaseInput(**defaults)


# ─────────────────────────────────────────────
# 1. RECOMMENDATION CORRECTNESS
# ─────────────────────────────────────────────

def test_serverless_for_low_frequency_low_budget_realtime():
    r = recommend(_case(
        latencia_requerida_ms=800,
        frecuencia_inferencia="baja",
        presupuesto_mensual_usd=30,
    ))
    assert r.recomendacion.id == "serverless"


def test_batch_for_high_latency_tolerance_high_experience():
    r = recommend(_case(
        latencia_requerida_ms=86_400_000,
        frecuencia_inferencia="baja",
        presupuesto_mensual_usd=300,
        escalabilidad_requerida="alta",
        experiencia_tecnica="alta",
    ))
    assert r.recomendacion.id == "batch"


def test_streaming_for_continuous_high_frequency():
    r = recommend(_case(
        latencia_requerida_ms=200,
        frecuencia_inferencia="continua",
        presupuesto_mensual_usd=500,
        escalabilidad_requerida="alta",
        experiencia_tecnica="alta",
    ))
    assert r.recomendacion.id == "streaming"


def test_ranking_contains_all_five_architectures():
    r = recommend(_case())
    ids = {a.id for a in r.ranking}
    assert ids == {"serverless", "batch", "streaming", "containers", "sagemaker"}


def test_ranking_is_sorted_descending():
    r = recommend(_case())
    scores = [a.score_total for a in r.ranking]
    assert scores == sorted(scores, reverse=True)


def test_positions_are_one_indexed_and_sequential():
    r = recommend(_case())
    for i, arch in enumerate(r.ranking):
        assert arch.posicion == i + 1


def test_winner_has_highest_score():
    r = recommend(_case())
    assert r.recomendacion.score_total == max(a.score_total for a in r.ranking)


# ─────────────────────────────────────────────
# 2. REASONING QUALITY
# ─────────────────────────────────────────────

def test_justificacion_mentions_winner_name():
    r = recommend(_case())
    assert r.recomendacion.nombre in r.justificacion


def test_razonamiento_completo_has_key_sections():
    r = recommend(_case())
    text = r.razonamiento_completo
    assert "RECOMENDACIÓN" in text
    assert "CRITERIO DECISIVO" in text
    assert "FORTALEZAS" in text
    assert "DESCARTARON" in text


def test_each_architecture_has_five_criterion_details():
    r = recommend(_case())
    for arch in r.ranking:
        assert set(arch.razonamiento.keys()) == set(WEIGHTS.keys())


def test_each_criterion_detail_has_non_empty_razon():
    r = recommend(_case())
    for arch in r.ranking:
        for k, detail in arch.razonamiento.items():
            assert len(detail.razon) > 10, (
                f"arch={arch.id}, criterion={k}: razon is too short: '{detail.razon}'"
            )


def test_criterio_decisivo_is_non_empty():
    r = recommend(_case())
    assert len(r.criterio_decisivo) > 15


def test_discarded_architectures_have_descartada_por():
    r = recommend(_case())
    for arch in r.ranking[1:]:
        assert len(arch.descartada_por) > 10, (
            f"arch={arch.id} missing descartada_por"
        )


def test_confidence_is_between_0_and_1():
    r = recommend(_case())
    assert 0.0 <= r.confianza <= 1.0


def test_score_total_matches_weighted_sum():
    case = _case()
    weights = _adaptive_weights(case)  # adaptive weights may differ from WEIGHTS
    r = recommend(case)
    for arch in r.ranking:
        expected = round(sum(arch.desglose[k] * weights[k] for k in weights), 3)
        assert abs(arch.score_total - expected) < 0.001, (
            f"arch={arch.id}: score_total={arch.score_total} != weighted_sum={expected}"
        )


def test_all_scores_within_0_to_10():
    r = recommend(_case())
    for arch in r.ranking:
        for k, v in arch.desglose.items():
            assert 0 <= v <= 10, f"arch={arch.id}, criterion={k}: score {v} out of range"


# ─────────────────────────────────────────────
# 3. VALIDATION — HARD CONSTRAINTS & WARNINGS
# ─────────────────────────────────────────────

def test_warns_batch_incompatible_with_realtime_high_frequency():
    """Batch must be flagged when latency ≤1s AND frequency is alta/continua."""
    r = recommend(_case(
        latencia_requerida_ms=500,
        frecuencia_inferencia="continua",
    ))
    tipos = [w.tipo for w in r.advertencias]
    assert "restriccion_dura" in tipos


def test_warns_tension_continuous_frequency_low_budget():
    """Kinesis always-on cost is incompatible with very low budget."""
    r = recommend(_case(
        frecuencia_inferencia="continua",
        presupuesto_mensual_usd=20,
    ))
    mensajes = " ".join(w.mensaje for w in r.advertencias)
    assert "continua" in mensajes.lower() or "kinesis" in mensajes.lower()


def test_warns_tension_high_scale_low_budget():
    r = recommend(_case(
        escalabilidad_requerida="alta",
        presupuesto_mensual_usd=30,
    ))
    tipos = [w.tipo for w in r.advertencias]
    assert "tension" in tipos


def test_no_spurious_warnings_for_clean_serverless_case():
    """A well-formed serverless case with adequate budget should generate no warnings."""
    r = recommend(_case(
        latencia_requerida_ms=800,
        frecuencia_inferencia="baja",
        presupuesto_mensual_usd=200,  # above all architecture minimum costs
        escalabilidad_requerida="media",
        experiencia_tecnica="media",
        volumen_datos_kb=5,
    ))
    assert r.advertencias == []


def test_batch_has_hard_constraint_flag_when_realtime():
    """batch must have es_restriccion=True on latencia when latency ≤1000ms."""
    r = recommend(_case(latencia_requerida_ms=800))
    batch = next(a for a in r.ranking if a.id == "batch")
    assert batch.razonamiento["latencia"].es_restriccion is True


def test_streaming_has_hard_constraint_flag_for_low_frequency():
    """Streaming is near-disqualifying for low/baja frequency (high fixed cost)."""
    r = recommend(_case(frecuencia_inferencia="baja"))
    streaming = next(a for a in r.ranking if a.id == "streaming")
    assert streaming.razonamiento["frecuencia"].es_restriccion is True


def test_sagemaker_has_cost_constraint_for_tiny_budget():
    """SageMaker always-on cost should be flagged for budget < $50."""
    r = recommend(_case(presupuesto_mensual_usd=20))
    sm = next(a for a in r.ranking if a.id == "sagemaker")
    assert sm.razonamiento["presupuesto"].es_restriccion is True
