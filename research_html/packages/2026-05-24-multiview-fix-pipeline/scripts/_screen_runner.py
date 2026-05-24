#!/usr/bin/env python
"""S screening sweeper: ablate F2 mechanisms (a, b, c) one at a time, code_length=1."""
import argparse
import itertools
import json
import os
import subprocess
import sys


VARIANTS = [
    # name, high_w, low_w, ortho_w, per_slot_init
    ('full_F2',     0.7, 0.1, 0.1, True),
    ('no_soft_a',   0.0, 0.0, 0.1, True),
    ('no_ortho_b',  0.7, 0.1, 0.0, True),
    ('no_init_c',   0.7, 0.1, 0.1, False),
]


def run_variant(args, variant_name, high_w, low_w, ortho_w, per_slot_init):
    """Launch run.py for a single screening variant and parse the best metric."""
    exp_name = f'S_{variant_name}_s{args.seed}'
    save_path = os.path.join(args.runtime_root, 'output/GRDR/S_screen', exp_name)
    cache_dir = os.path.join(args.runtime_root, 'cache', f'S_{variant_name}')
    log_path = os.path.join(args.screen_dir, f'{exp_name}.log')

    cmd = [
        sys.executable, 'run.py',
        '--device', str(args.device),
        '--model_name', 't5-small',
        '--dataset', 'panda',
        '--features_root', args.features_root,
        '--cache_dir', cache_dir,
        '--code_num', '4096',
        '--max_length', '1',  # code_length=1 single loop only
        '--batch_size', '512',
        '--eval_batch_size', '32',
        '--num_candidates', '100',
        '--setting', '1',
        '--num_latent_tokens', '4',
        '--use_pseudo_queries',
        '--multiview_all_slot_ce',
        '--view_div_high_weight', str(high_w),
        '--view_div_low_weight', str(low_w),
        '--slot_orthogonality_weight', str(ortho_w),
        '--pretrain_lr', '2e-4', '--main_lr', '5e-5', '--fit_lr', '2e-5',
        '--pretrain_epochs', '1', '--main_epochs', '1', '--fit_epochs', '1',
        '--save_path', save_path,
        '--exp_name', exp_name,
        '--seed', str(args.seed),
        '--w2_cl_loss', '0.2', '--w2_ce_loss', '0.5', '--w2_code_loss', '0.8',
        '--w2_cl_dd_loss', '0.1', '--w2_rq_loss', '0.3',
        '--w3_ce_loss', '1', '--w3_code_loss', '1', '--w3_rq_loss', '0',
        '--w3_bucket_route_loss', '0.10',
        '--enable_fit',
        '--wandb_project', '2026-05-24-multiview-fix-pipeline',
        '--wandb_run_name', f'S_{variant_name}_s{args.seed}',
    ]
    if per_slot_init:
        cmd.append('--per_slot_init')

    print(f'[S] launching {exp_name}: high={high_w} low={low_w} ortho={ortho_w} per_slot_init={per_slot_init}')
    with open(log_path, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f'[S] {exp_name} rc={rc} log={log_path}')

    # Best-effort metric scrape: look for 'Current Eval Metric' or similar in the log.
    metric = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if 'CanHit@100' in line or 'best_metric' in line.lower():
                    metric = line.strip()
    return {'variant': variant_name, 'rc': rc, 'log': log_path,
            'config': {'high_w': high_w, 'low_w': low_w, 'ortho_w': ortho_w,
                       'per_slot_init': per_slot_init},
            'metric_line': metric}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--screen-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.screen_dir, exist_ok=True)
    results = []
    for name, h, l, o, p in VARIANTS:
        r = run_variant(args, name, h, l, o, p)
        results.append(r)
        with open(os.path.join(args.screen_dir, 'mech_ablation.json'), 'w') as f:
            json.dump(results, f, indent=2)

    out = os.path.join(args.screen_dir, 'mech_ablation.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[S] wrote {out}')


if __name__ == '__main__':
    main()
