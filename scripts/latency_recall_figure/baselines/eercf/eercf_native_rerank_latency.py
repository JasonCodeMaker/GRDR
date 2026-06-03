#!/usr/bin/env python
"""P4 EERCF native rerank latency (Pass-B Stage-2).

Times EERCF's *native* rerank step per query — text encode + cold-load top-K
multi-level features (visual_output + visual_patches + video_mask) + 5 logits
via model.get_similarity_logits + fusion (mirrors main_eercf.eval_epoch's
weighted-combine at lines 1116-1121). Reads candidates from the Stage-1
candidate JSON written by P4_eercf_perquery_latency.py; does NOT re-run the
recall step.

This is the EERCF analogue of X-Pool's test_perquery.py: per-(query, candidate)
disk-cold-load + per-query CUDA-synced rerank fusion. Required when the
shared-X-Pool stage-2 (test_perquery.py) is not a faithful representation of
EERCF's rerank cost because EERCF's rerank consumes patch-level features that
X-Pool doesn't touch.

Output schema matches the shape that summarize_pass_b_latency.py expects under
`rerank_latency_ms`, so the file drops into results_b/eercf/<ds>/setting<n>/
perquery_summary.json and is picked up by the aggregator without change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

EERCF_DIR_DEFAULT = '/home/uqzzha35/Project/SemanticID/EERCF'
HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))


def _import_eercf(eercf_dir: str):
    if eercf_dir not in sys.path:
        sys.path.insert(0, eercf_dir)
    from modules.modeling_fineclip_patches_cdcl import XCLIP  # noqa: E402
    from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE  # noqa: E402
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402
    return XCLIP, ClipTokenizer, PYTORCH_PRETRAINED_BERT_CACHE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eercf_dir', default=EERCF_DIR_DEFAULT)
    p.add_argument('--init_model', required=True, help='P3d EERCF ckpt (.bin)')
    p.add_argument('--datatype', required=True,
                   choices=['msrvtt', 'activity', 'didemo', 'lsmdc'])
    p.add_argument('--candidate_file', required=True,
                   help='Stage-1 candidate JSON written by P4_eercf_perquery_latency.py')
    p.add_argument('--frame_cache_dir', required=True,
                   help='P3.6 multi-level cache root (.npz with visual_output/'
                        'visual_patches/video_mask). The script appends args.datatype.')
    p.add_argument('--subset_manifest', required=True)
    p.add_argument('--warmup_n_used', type=int, default=10)
    p.add_argument('--wall_time_cap_s', type=float, default=300.0)
    p.add_argument('--summary_json', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    # EERCF model/inference knobs (mirror main_eercf.py defaults)
    p.add_argument('--max_words', type=int, default=32)
    p.add_argument('--max_frames', type=int, default=12)
    p.add_argument('--feature_framerate', type=float, default=1.0)
    p.add_argument('--image_resolution', type=int, default=224)
    p.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2])
    p.add_argument('--frame_order', type=int, default=0, choices=[0, 1, 2])
    p.add_argument('--cross_model', type=str, default='cross-base')
    p.add_argument('--cross_num_hidden_layers', type=int, default=4)
    p.add_argument('--loose_type', action='store_true', default=True)
    p.add_argument('--linear_patch', type=str, default='2d')
    p.add_argument('--sim_header', type=str, default='seqTransf')
    p.add_argument('--freeze_layer_num', type=int, default=0)
    p.add_argument('--pretrained_clip_name', type=str, default='ViT-B/32')
    # Fusion weights (mirror main_eercf.py defaults).
    p.add_argument('--video_weight', type=float, default=0.5)
    p.add_argument('--weakframe_weight', type=float, default=0.8)
    p.add_argument('--weakpatch_weight', type=float, default=0.2)
    p.add_argument('--strongframe_weight', type=float, default=0.2)
    p.add_argument('--strongpatch_weight', type=float, default=0.0)
    p.add_argument('--weak_loss_weight', type=float, default=1.0)
    p.add_argument('--strong_loss_weight', type=float, default=1.0)
    return p.parse_args()


def _load_model(args, device):
    XCLIP, _, PYTORCH_PRETRAINED_BERT_CACHE = _import_eercf(args.eercf_dir)
    state = torch.load(args.init_model, map_location='cpu')
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = XCLIP.from_pretrained(
        args.cross_model, cache_dir=cache_dir, state_dict=state, task_config=args,
    )
    model = model.to(device).eval()
    return model


SPECIAL_TOKEN = {
    "CLS_TOKEN": "<|startoftext|>",
    "SEP_TOKEN": "<|endoftext|>",
}


def _tokenize(tokenizer, text: str, max_words: int):
    """Mirror EERCF dataloaders' _get_text (single sentence)."""
    words = tokenizer.tokenize(text)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return (
        np.asarray(input_ids, dtype=np.int64)[None, :],
        np.asarray(input_mask, dtype=np.int64)[None, :],
        np.asarray(segment_ids, dtype=np.int64)[None, :],
    )


def _cold_load_candidate(frame_cache_root: str, vid: str):
    path = os.path.join(frame_cache_root, f'{vid}.npz')
    with np.load(path) as z:
        vo = torch.from_numpy(z['visual_output'].astype(np.float32))
        vp = torch.from_numpy(z['visual_patches'].astype(np.float32))
        vm = torch.from_numpy(z['video_mask']).long()
    return vo, vp, vm


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if HELPERS_DIR not in sys.path:
        sys.path.insert(0, HELPERS_DIR)
    from latency_helpers import host_fingerprint, load_subset_manifest  # noqa: E402

    # ---------- setup (untimed) ----------
    print('[setup] loading EERCF model + tokenizer ...', flush=True)
    _, ClipTokenizer, _ = _import_eercf(args.eercf_dir)
    tokenizer = ClipTokenizer()
    model = _load_model(args, device)

    print(f'[setup] reading candidate file {args.candidate_file}', flush=True)
    with open(args.candidate_file) as f:
        cand_payload = json.load(f)
    # query_to_cands: map qid -> (query_text, [candidate ids])
    query_to_cands: dict[str, tuple[str, list[str]]] = {}
    for entry in cand_payload.get('results', []):
        qid = entry.get('ground_truth_video_id', '')
        if not qid:
            continue
        query_to_cands[qid] = (entry.get('query_text', ''),
                                list(entry.get('candidates', [])))
    print(f'[setup] candidate map size = {len(query_to_cands)}', flush=True)

    frame_cache_root = os.path.join(args.frame_cache_dir, args.datatype)
    print(f'[setup] frame cache root = {frame_cache_root}', flush=True)

    manifest = load_subset_manifest(args.subset_manifest)
    warmup_ids = list(manifest['warmup_query_ids'][: args.warmup_n_used])
    timed_ids = list(manifest['timed_query_ids'])

    def _run_one(qid: str):
        if qid not in query_to_cands:
            return None
        sent, cand_ids = query_to_cands[qid]
        if not cand_ids:
            return None
        ids = list(cand_ids)

        # Tokenize once on CPU (deterministic), to mirror what test_perquery
        # does inside its timed window.
        pairs_text, pairs_mask, pairs_segment = _tokenize(tokenizer, sent, args.max_words)

        # CUDA-sync + timer start.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            input_ids = torch.from_numpy(pairs_text).to(device)
            input_mask = torch.from_numpy(pairs_mask).to(device)
            segment_ids = torch.from_numpy(pairs_segment).to(device)
            sequence_output, seq_features = model.get_sequence_output(
                input_ids, segment_ids, input_mask,
            )

            vo_list, vp_list, vm_list, used = [], [], [], []
            for vid in ids:
                p = os.path.join(frame_cache_root, f'{vid}.npz')
                if not os.path.exists(p):
                    continue
                vo, vp, vm = _cold_load_candidate(frame_cache_root, vid)
                vo_list.append(vo)
                vp_list.append(vp)
                vm_list.append(vm)
                used.append(vid)

            if not used:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                return {
                    'qid': qid, 'n_used': 0, 'n_cands_in': len(ids),
                    'ms_total': (t1 - t0) * 1000.0,
                    'topk': [], 'scores': [],
                }

            K = len(used)
            visual_output = torch.stack(vo_list, dim=0).to(device)        # (K, F, 512)
            visual_patches = torch.stack(vp_list, dim=0).to(device)       # (K, F, P, 512)
            video_mask = torch.stack(vm_list, dim=0).to(device)           # (K, F)

            seq_K = sequence_output.expand(K, -1, -1).contiguous()
            seqf_K = (seq_features.expand(K, -1, -1).contiguous()
                      if seq_features.dim() == 3 else seq_features)
            mask_K = input_mask.expand(K, -1).contiguous()

            start_patch = args.strongpatch_weight > 0
            TIB_output, _ = model.get_similarity_logits(
                seq_K, seqf_K, visual_output, visual_patches, mask_K, video_mask,
                loose_type=model.loose_type, start_patch_eval=start_patch,
            )
            sv = TIB_output[0].diagonal().cpu().numpy()
            swf = TIB_output[1].diagonal().cpu().numpy()
            ssf = TIB_output[2].diagonal().cpu().numpy()
            swp = TIB_output[3].diagonal().cpu().numpy()
            # main_eercf.eval_epoch line 1121 fusion: 0.8*swf + 0.2*ssf + 0.2*swp + 0.5*sv
            ii, jj, kk = 0.8, 0.2, 0.5
            fused = swf * ii + (1 - ii) * ssf + jj * swp + kk * sv

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        order = np.argsort(-fused)
        top100 = order[: min(100, K)]
        return {
            'qid': qid,
            'n_used': K,
            'n_cands_in': len(ids),
            'ms_total': (t1 - t0) * 1000.0,
            'topk': [used[i] for i in top100],
            'scores': [float(fused[i]) for i in top100],
        }

    # ---------- warmup ----------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'[warmup] {len(warmup_ids)} queries ...', flush=True)
    for qid in warmup_ids:
        try:
            _ = _run_one(qid)
        except Exception as exc:
            print(f'[warmup] {qid} skipped: {exc}', flush=True)

    # ---------- timed phase ----------
    print(f'[timed] up to {len(timed_ids)} queries (cap {args.wall_time_cap_s:.0f}s) ...',
          flush=True)
    timings: list[dict] = []
    cap_t0 = time.perf_counter()
    cap_hit = False
    for qid in timed_ids:
        if (time.perf_counter() - cap_t0) >= args.wall_time_cap_s:
            cap_hit = True
            break
        try:
            row = _run_one(qid)
        except Exception as exc:
            print(f'[timed] {qid} skipped: {exc}', flush=True)
            continue
        if row is not None:
            timings.append(row)

    wall_seconds = time.perf_counter() - cap_t0
    n_processed = len(timings)
    n_target = len(timed_ids)

    ms_arr = np.asarray([t['ms_total'] for t in timings], dtype=np.float64) \
        if timings else np.zeros(1)
    cand_arr = np.asarray([t['n_used'] for t in timings], dtype=np.int64) \
        if timings else np.zeros(1, dtype=np.int64)

    rerank_block = {
        'online_total_mean': float(ms_arr.mean()) if timings else 0.0,
        'online_total_p95': float(np.percentile(ms_arr, 95)) if timings else 0.0,
        'online_total_std': float(ms_arr.std()) if timings else 0.0,
        'component_breakdown_mean_ms': {
            'query_encode': 0.0,
            'video_load': 0.0,
            'frame_pooling': 0.0,
            'similarity_compute': 0.0,
        },
        'n_processed': int(n_processed),
        'n_target': int(n_target),
        'warmup_n': len(manifest.get('warmup_query_ids', [])),
        'warmup_n_used': int(args.warmup_n_used),
        'validity': 'full_subset' if (n_processed >= n_target and not cap_hit) else 'truncated_subset',
        'wall_seconds': float(wall_seconds),
        'cap_hit': bool(cap_hit),
        'strict_latency_contract': 'eercf_native_rerank_cold_load_topk',
        'latency_batch_size': 1,
    }

    payload = {
        'method': 'eercf',
        'dataset': args.datatype,
        'setting': cand_payload.get('metadata', {}).get('setting', 1),
        'subset_manifest': args.subset_manifest,
        'subset_manifest_sha256': manifest['metadata'].get('content_sha256', ''),
        'host_fingerprint': host_fingerprint(),
        'rerank_latency_ms': rerank_block,
        'summary': {
            'retrieval_mode': 'candidates',
            'candidate_file': args.candidate_file,
            'candidate_count_mean': float(cand_arr.mean()) if timings else 0.0,
            'candidate_count_min': int(cand_arr.min()) if timings else 0,
            'candidate_count_max': int(cand_arr.max()) if timings else 0,
        },
    }
    os.makedirs(os.path.dirname(args.summary_json) or '.', exist_ok=True)
    with open(args.summary_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[done] processed={n_processed}/{n_target} '
          f'mean={rerank_block["online_total_mean"]:.2f} ms '
          f'mean_cands={cand_arr.mean():.1f} '
          f'validity={rerank_block["validity"]}', flush=True)
    print(f'[write] {args.summary_json}', flush=True)


if __name__ == '__main__':
    main()
