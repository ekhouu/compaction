"""
Ceteris paribus 2x2 evaluation config.

Compares four conditions that differ ONLY in the treatment applied to the
KV cache, while keeping model, dataset, prompts, and metrics identical:

    | Condition  | Quantize? | Compact? | Description                    |
    |------------|-----------|----------|--------------------------------|
    | original   | No        | No       | Baseline (handled separately)  |
    | AM         | No        | Yes      | Compaction only                |
    | TQ+        | Yes       | No       | Quantization only              |
    | TQ+AM      | Yes       | Yes      | Quantization + Compaction      |

Run with:
    python -m evaluation.run_qa_evaluation \\
        --algorithm-config ceteris_paribus \\
        --methods original AM TQ_int8 TQ_int8_AM \\
        --dataset-name quality --n-articles 10

Multiple quantization bit-widths are provided (int8, int4, int2) so you
can sweep across precision levels.  The AM configs share matched
hyperparameters (NNLS-2, bounds exp(-3)..exp(3), on-policy) for a fair
comparison.
"""
import math

exp = math.exp

# Shared AM hyperparameters -- identical to key-selection.py's AM alias
_AM_BASE = {
    'algorithm': 'highest_attention_keys',
    'score_method': 'rms',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
    'c2_method': 'lsq',
    'on_policy': True,
}

config = {
    # ── Compaction only (no quantization) ────────────────────────────
    'AM': {**_AM_BASE},

    # ── Quantization only (no compaction) ────────────────────────────
    'TQ_int8': {
        'algorithm': 'quantize_only',
        'quantize': 'int8',
    },
    'TQ_int4': {
        'algorithm': 'quantize_only',
        'quantize': 'int4',
    },
    'TQ_int2': {
        'algorithm': 'quantize_only',
        'quantize': 'int2',
    },

    # ── Quantization + Compaction ────────────────────────────────────
    'TQ_int8_AM': {
        **_AM_BASE,
        'quantize': 'int8',
    },
    'TQ_int4_AM': {
        **_AM_BASE,
        'quantize': 'int4',
    },
    'TQ_int2_AM': {
        **_AM_BASE,
        'quantize': 'int2',
    },
}
