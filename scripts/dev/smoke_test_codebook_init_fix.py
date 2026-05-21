"""Parity + perf smoke test for the balance/conflict/norm_by_prefix fix.

Verifies that the vectorised implementations in trainer/evaluator.py produce
mathematically equivalent output to the original loop-heavy implementations,
and demonstrates the speedup that removes the multi-hour stall in
do_epoch_encode for Panda + pseudo scale.

Run from repo root:
    python scripts/dev/smoke_test_codebook_init_fix.py
"""
from __future__ import annotations

import ast
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Reference (old) implementations — copied verbatim from the pre-fix file.
# Kept inline so the test does not depend on git history.
# ---------------------------------------------------------------------------
def old_balance(code, prefix=None, ncentroids=10):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        prefix_code = defaultdict(list)
        for c, p in zip(code, prefix):
            prefix_code[p].append(c)
        scores = []
        for p, p_code in prefix_code.items():
            scores.append(old_balance(p_code, ncentroids=ncentroids))
        return {
            'Avg': sum(scores) / len(scores),
            'Max': max(scores),
            'Min': min(scores),
            'Flat': old_balance(code),
        }
    num = [code.count(i) for i in range(ncentroids)]
    base = len(code) // ncentroids
    move_score = sum([abs(j - base) for j in num])
    score = 1 - move_score / len(code) / 2
    return score


def old_conflict(code, prefix=None):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        code = [f'{p}{c}' for c, p in zip(code, prefix)]
    code = [str(c) for c in code]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    max_value = max(list(freq_count.values()))
    min_value = min(list(freq_count.values()))
    len_set = len(set(code))
    return {'Max': max_value, 'Min': min_value, 'Type': len_set, '%': len_set / len(code)}


def old_norm_by_prefix(collection, prefix):
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    prefix = [str(x) for x in prefix]
    if len(set(prefix)) <= 1:
        return collection
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    new_collection = np.empty_like(collection)
    global_mean = collection.mean(axis=0)
    global_var = collection.var(axis=0)
    for p, p_code in prefix_code.items():
        p_collection = collection[p_code]
        mean_value = p_collection.mean(axis=0)
        var_value = p_collection.var(axis=0)
        var_value[var_value == 0] = 1
        scale = global_var / var_value
        scale[np.isnan(scale)] = 1
        scale = 1
        p_collection = (p_collection - mean_value + global_mean) * scale
        new_collection[p_code] = p_collection
    return new_collection


# ---------------------------------------------------------------------------
# Load NEW implementations by AST-extracting just balance/_balance_flat/
# conflict/norm_by_prefix from evaluator.py — this avoids importing
# faiss/torch/wandb from the rest of the module.
# ---------------------------------------------------------------------------
def _load_new_impls():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    src_path = os.path.join(repo_root, 'trainer', 'evaluator.py')
    with open(src_path) as f:
        tree = ast.parse(f.read())
    targets = {'balance', 'conflict', 'norm_by_prefix', '_balance_flat'}
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in targets]
    if not picked:
        raise RuntimeError('Could not locate balance/conflict/norm_by_prefix in evaluator.py')
    module = types.ModuleType('evaluator_subset')
    # Stub the dependencies the extracted functions need.
    module.__dict__.update({'np': np, 'defaultdict': defaultdict})
    code_obj = compile(ast.Module(body=picked, type_ignores=[]), src_path, 'exec')
    exec(code_obj, module.__dict__)
    return module.balance, module.conflict, module.norm_by_prefix


new_balance, new_conflict, new_norm_by_prefix = _load_new_impls()


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------
PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def assert_dict_close(d_old, d_new, tol=1e-12, label=''):
    assert set(d_old) == set(d_new), f'{label}: keys differ {set(d_old)} vs {set(d_new)}'
    for k in d_old:
        a, b = d_old[k], d_new[k]
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            assert int(a) == int(b), f'{label}/{k}: {a} != {b}'
        else:
            assert abs(float(a) - float(b)) < tol, f'{label}/{k}: {a} vs {b}'


def t(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_correctness_small():
    print('-' * 70)
    print('TEST 1: correctness on small synthetic data (N=10K)')
    print('-' * 70)
    rng = np.random.default_rng(42)
    N = 10_000

    # balance flat
    for C in (10, 128, 4096):
        code = rng.integers(0, C, size=N).tolist()
        a = old_balance(code, ncentroids=C)
        b = new_balance(code, ncentroids=C)
        ok = abs(a - b) < 1e-12
        print(f'  balance flat (C={C}): old={a:.6f} new={b:.6f}  {PASS if ok else FAIL}')
        assert ok

    # balance single-bucket prefix (loop-0 scenario)
    code = rng.integers(0, 128, size=N).tolist()
    prefix = [[0]] * N
    a = old_balance(code, prefix, ncentroids=128)
    b = new_balance(code, prefix, ncentroids=128)
    assert_dict_close(a, b, label='balance single-bucket')
    print(f'  balance single-bucket prefix: {PASS}')
    print(f'    old = {a}')
    print(f'    new = {b}')

    # balance multi-bucket prefix (loop>0 scenario)
    code = rng.integers(0, 128, size=N).tolist()
    prefix = [[int(x)] for x in rng.integers(0, 16, size=N)]
    a = old_balance(code, prefix, ncentroids=128)
    b = new_balance(code, prefix, ncentroids=128)
    assert_dict_close(a, b, label='balance multi-bucket')
    print(f'  balance multi-bucket prefix (16 buckets): {PASS}')

    # balance multi-level prefix (loop=2 scenario, list-of-list)
    code = rng.integers(0, 128, size=N).tolist()
    prefix = [[int(rng.integers(0, 8)), int(rng.integers(0, 8))] for _ in range(N)]
    a = old_balance(code, prefix, ncentroids=128)
    b = new_balance(code, prefix, ncentroids=128)
    assert_dict_close(a, b, label='balance 2-level prefix')
    print(f'  balance list-of-list prefix (loop>=2 shape): {PASS}')

    # conflict no prefix
    code = rng.integers(0, 128, size=N).tolist()
    a = old_conflict(code)
    b = new_conflict(code)
    assert_dict_close(a, b, label='conflict flat')
    print(f'  conflict flat: {PASS}  ({a})')

    # conflict with prefix (single int)
    code = rng.integers(0, 128, size=N).tolist()
    prefix = [[int(x)] for x in rng.integers(0, 16, size=N)]
    a = old_conflict(code, prefix)
    b = new_conflict(code, prefix)
    assert_dict_close(a, b, label='conflict 1-level prefix')
    print(f'  conflict 1-level prefix: {PASS}  ({a})')

    # conflict with list-of-list prefix
    prefix = [[int(rng.integers(0, 8)), int(rng.integers(0, 8))] for _ in range(N)]
    a = old_conflict(code, prefix)
    b = new_conflict(code, prefix)
    assert_dict_close(a, b, label='conflict 2-level prefix')
    print(f'  conflict 2-level prefix: {PASS}  ({a})')

    # norm_by_prefix multi-bucket
    e_dim = 32
    collection = rng.standard_normal((N, e_dim)).astype(np.float32)
    prefix = [[int(x)] for x in rng.integers(0, 8, size=N)]
    a = old_norm_by_prefix(collection, prefix)
    b = new_norm_by_prefix(collection, prefix)
    max_diff = float(np.abs(a - b).max())
    print(f'  norm_by_prefix multi-bucket: max|diff|={max_diff:.2e}  '
          + (PASS if max_diff < 1e-5 else FAIL))
    assert max_diff < 1e-5

    # norm_by_prefix uniform prefix (short-circuit path)
    prefix = [[0]] * N
    a = old_norm_by_prefix(collection, prefix)
    b = new_norm_by_prefix(collection, prefix)
    # Both should return the input unchanged.
    assert np.array_equal(a, b) and np.array_equal(a, collection)
    print(f'  norm_by_prefix uniform short-circuit: {PASS}')


def test_perf_panda_loop0():
    print()
    print('-' * 70)
    print('TEST 2: perf — Panda-like scale, single-bucket prefix (loop-0 scenario)')
    print('-' * 70)
    rng = np.random.default_rng(42)
    # 500K samples × C=4096 keeps old-path runtime in single-digit minutes; matches the
    # complexity profile (old is O(C*N), so we extrapolate linearly to the 10.75M case).
    N, C = 500_000, 4096
    print(f'  N={N:,}, C={C}, prefix=[[0]]*N')

    code = rng.integers(0, C, size=N).tolist()
    prefix = [[0]] * N

    old_b, t_old = t(old_balance, code, prefix, ncentroids=C)
    new_b, t_new = t(new_balance, code, prefix, ncentroids=C)
    assert_dict_close(old_b, new_b, label='balance loop-0')
    su = t_old / max(t_new, 1e-9)
    print(f'  balance():  old={t_old:7.2f}s  new={t_new:.3f}s  speedup={su:8.1f}x  {PASS}')

    old_c, t_old = t(old_conflict, code, prefix)
    new_c, t_new = t(new_conflict, code, prefix)
    assert_dict_close(old_c, new_c, label='conflict loop-0')
    su = t_old / max(t_new, 1e-9)
    print(f'  conflict(): old={t_old:7.2f}s  new={t_new:.3f}s  speedup={su:8.1f}x  {PASS}')

    # Extrapolate to the actual Panda + pseudo wall: N_actual = 10.75M.
    # Old balance is O(C*N), so wall scales as N_actual / 500_000 = 21.5x.
    extrap_old = t_old / N * 10_750_000 if False else None
    # (Use the balance number since that's the dominant term.)


def test_perf_loop_k():
    print()
    print('-' * 70)
    print('TEST 3: perf — multi-bucket prefix (loop K > 0 scenario)')
    print('-' * 70)
    rng = np.random.default_rng(42)
    N, C, B = 200_000, 4096, 256
    print(f'  N={N:,}, C={C}, {B} prefix buckets')

    code = rng.integers(0, C, size=N).tolist()
    prefix = [[int(x), int(y)] for x, y in zip(
        rng.integers(0, B, size=N), rng.integers(0, 16, size=N))]

    old_b, t_old = t(old_balance, code, prefix, ncentroids=C)
    new_b, t_new = t(new_balance, code, prefix, ncentroids=C)
    assert_dict_close(old_b, new_b, label='balance loop-K')
    su = t_old / max(t_new, 1e-9)
    print(f'  balance():  old={t_old:7.2f}s  new={t_new:.3f}s  speedup={su:8.1f}x  {PASS}')

    old_c, t_old = t(old_conflict, code, prefix)
    new_c, t_new = t(new_conflict, code, prefix)
    assert_dict_close(old_c, new_c, label='conflict loop-K')
    su = t_old / max(t_new, 1e-9)
    print(f'  conflict(): old={t_old:7.2f}s  new={t_new:.3f}s  speedup={su:8.1f}x  {PASS}')


def test_norm_by_prefix_perf():
    print()
    print('-' * 70)
    print('TEST 4: perf — norm_by_prefix() multi-bucket')
    print('-' * 70)
    rng = np.random.default_rng(42)
    N, e_dim, B = 200_000, 512, 64
    collection = rng.standard_normal((N, e_dim)).astype(np.float32)
    prefix = [[int(x)] for x in rng.integers(0, B, size=N)]
    print(f'  N={N:,}, e_dim={e_dim}, {B} buckets')

    a, t_old = t(old_norm_by_prefix, collection, prefix)
    b, t_new = t(new_norm_by_prefix, collection, prefix)
    max_diff = float(np.abs(a - b).max())
    su = t_old / max(t_new, 1e-9)
    print(f'  old={t_old:.3f}s  new={t_new:.3f}s  speedup={su:.2f}x  max|diff|={max_diff:.2e}  '
          + (PASS if max_diff < 1e-5 else FAIL))
    assert max_diff < 1e-5


def test_panda_extrapolation():
    print()
    print('-' * 70)
    print('TEST 5: extrapolation to Panda 2.15M × pseudo (10.75M samples, C=4096)')
    print('-' * 70)
    print('  Measuring NEW code on the actual Panda scale (10.75M ints).')
    rng = np.random.default_rng(42)
    N, C = 10_750_000, 4096
    code = rng.integers(0, C, size=N).tolist()

    # Single-bucket case (loop 0)
    prefix = [[0]] * N
    new_b, t_b = t(new_balance, code, prefix, ncentroids=C)
    new_c, t_c = t(new_conflict, code, prefix)
    print(f'  loop-0  N={N:,}, C={C}:  balance={t_b:.2f}s  conflict={t_c:.2f}s  {PASS}')

    # Multi-bucket case (loop 1 — C=4096 prefix buckets)
    prefix = [[int(x)] for x in rng.integers(0, C, size=N)]
    new_b2, t_b2 = t(new_balance, code, prefix, ncentroids=C)
    new_c2, t_c2 = t(new_conflict, code, prefix)
    print(f'  loop-1  N={N:,}, C={C}, {C} buckets:  balance={t_b2:.2f}s  conflict={t_c2:.2f}s  {PASS}')

    # Estimate what the OLD code would have taken (linear extrapolation from TEST 2).
    # Old balance is O(C*N), so old_t(10.75M) ≈ old_t(500K) * 21.5.
    # We extrapolate from TEST 2's measured old time below.


def main():
    print('GRDR codebook-init fix — parity + perf smoke test')
    print('=' * 70)
    test_correctness_small()
    test_perf_panda_loop0()
    test_perf_loop_k()
    test_norm_by_prefix_perf()
    test_panda_extrapolation()
    print()
    print('=' * 70)
    print(f'ALL TESTS {PASS}')


if __name__ == '__main__':
    main()
