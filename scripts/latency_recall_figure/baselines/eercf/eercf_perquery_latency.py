#!/usr/bin/env python
"""P4 EERCF per-query latency simulation.

Faithful simulator of EERCF's two-stage Stage-1 (initial dense recall +
multi-level top-K rerank) that emits per-query latency under the same
contract as X-Pool Stage-2 and ANN-HNSW. Imports the EERCF XCLIP model and
its tokenizer from the EERCF repo, but lives in the GRDR research package so
EERCF/main_eercf.py's Pass-A path stays bit-identical.

What the per-query timer wraps (symmetric with X-Pool / ANN per-query):
  1. CLIP text encode (sequence_output, seq_features).
  2. Initial dense scan: text @ pool_pooled.T (pool preloaded at setup).
  3. Top-K (= rerantopk) selection on initial scores.
  4. Cold-load top-K candidates' P3.6 multi-level features from .npz (one
     np.load per candidate; this is the EERCF symmetric I/O cost vs X-Pool's
     per-candidate frame_embed load).
  5. model.get_similarity_logits over the batched top-K candidates.
  6. EERCF rerank fusion (mirrors main_eercf.py eval_epoch lines 1116-1121).
  7. Final top-100 ranking.

Outside the timer (setup, untimed): model load, tokenizer init, dataset
caption table, pool pooled-feature preload. This matches ANN's "build index
once, search per query" contract.

Output: canonical Pass-B candidate JSON with metadata.stage1_latency_ms and a
real results[] (200 entries) so P4B_stage2_latency.sh consumes it directly.

See docs/eval-efficiency.html for the contract.
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
    """Make the EERCF repo importable; return (XCLIP, ClipTokenizer, modeling_helpers)."""
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
    p.add_argument('--setting', type=int, default=1, choices=[1, 2])
    # Cache roots
    p.add_argument('--pool_cache_dir', required=True,
                   help='P3.5 pooled-feature cache root (.npz with "feature")')
    p.add_argument('--frame_cache_dir', required=True,
                   help='P3.6 multi-level cache root (.npz with visual_output/'
                        'visual_patches/video_mask)')
    # Dataset paths (forwarded to the EERCF dataloader class). Required keys
    # match what each dataset's __init__ expects.
    p.add_argument('--data_path', default='')
    p.add_argument('--train_csv', default='')
    p.add_argument('--test_csv', default='')
    p.add_argument('--train_json', default='')
    p.add_argument('--test_json', default='')
    p.add_argument('--features_path', default='')
    # Pass-B contract
    p.add_argument('--subset_manifest', required=True)
    p.add_argument('--warmup_n_used', type=int, default=10,
                   help='Per-query EERCF can honor full warmup (vs matrix-formula '
                        'override of 1); default keeps GR/ANN parity')
    p.add_argument('--wall_time_cap_s', type=float, default=300.0)
    p.add_argument('--output_json', required=True)
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
    p.add_argument('--rerantopk', type=int, default=50)
    # EERCF fusion weights (mirror main_eercf.py defaults; consumed by model)
    p.add_argument('--video_weight', type=float, default=0.5)
    p.add_argument('--weakframe_weight', type=float, default=0.8)
    p.add_argument('--weakpatch_weight', type=float, default=0.2)
    p.add_argument('--strongframe_weight', type=float, default=0.2)
    p.add_argument('--strongpatch_weight', type=float, default=0.0)
    p.add_argument('--weak_loss_weight', type=float, default=1.0)
    p.add_argument('--strong_loss_weight', type=float, default=1.0)
    return p.parse_args()


def _build_test_dataset(args, tokenizer):
    """Instantiate the existing EERCF test-dataset class (no DataLoader).

    Signatures verified against EERCF/dataloaders/* on 2026-05-27:
      - MSRVTT_DataLoader(subset, csv_path, features_path, tokenizer, ...)
      - ActivityNet_DataLoader(json_path, features_path, tokenizer, ...)  # no subset
      - DiDeMo_DataLoader(json_path, features_path, tokenizer, ...)       # no subset
      - LSMDC_DataLoader(json_path, features_path, tokenizer, ...)        # no subset

    Per-dataset paths mirror P3.5_eercf_recache.sh's canonical mapping.
    """
    dt = args.datatype
    if dt == 'msrvtt':
        from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
        ds = MSRVTT_DataLoader(
            subset='test',
            csv_path=args.test_csv,
            features_path=args.features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
        )
    elif dt == 'activity':
        from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
        ds = ActivityNet_DataLoader(
            json_path=args.test_json,
            features_path=args.features_path,
            tokenizer=tokenizer,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
        )
    elif dt == 'didemo':
        from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader
        ds = DiDeMo_DataLoader(
            json_path=args.test_json,
            features_path=args.features_path,
            tokenizer=tokenizer,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
        )
    elif dt == 'lsmdc':
        from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
        ds = LSMDC_DataLoader(
            json_path=args.test_json,
            features_path=args.features_path,
            tokenizer=tokenizer,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
        )
    else:
        raise ValueError(f'unknown datatype: {dt}')
    return ds


def _qid_to_caption(dataset, datatype: str):
    """Extract {manifest_qid -> (caption, raw_sentences_dict_key)} maps.

    Manifest qids (from build_latency_subset.py via X-Pool's load_test_queries)
    are the X-Pool short clip-id form, which:
      - matches sentences_dict keys directly for ACTNET / DIDEMO / MSRVTT,
      - drops the "<prefix>/" dir and ".avi" suffix for LSMDC.
    Returns:
      qid_to_caption[manifest_qid] = caption
      qid_to_raw[manifest_qid] = raw sentences_dict[i][0] (used by _get_text)
    """
    qid_to_caption: dict[str, str] = {}
    qid_to_raw: dict[str, str] = {}
    if datatype == 'msrvtt':
        for vid, sent in zip(dataset.data['video_id'].astype(str).tolist(),
                              dataset.data['sentence'].astype(str).tolist()):
            if vid not in qid_to_caption:
                qid_to_caption[vid] = sent
                qid_to_raw[vid] = vid
    else:
        for idx in range(len(dataset.sentences_dict)):
            item = dataset.sentences_dict[idx]
            raw = str(item[0])
            sent = str(item[1])
            if datatype == 'lsmdc':
                # raw is "<prefix>/<clip>.avi" — manifest qid is just <clip>
                short = raw.split('/')[-1].replace('.avi', '').replace('.mp4', '')
            else:
                short = raw  # activity/didemo already store the short clip-id
            if short not in qid_to_caption:
                qid_to_caption[short] = sent
                qid_to_raw[short] = raw
    return qid_to_caption, qid_to_raw


def _load_pool_pooled(pool_cache_dir: str, video_ids: list[str], dataset_subdir: str):
    """Pool-side preload (untimed): one np.load per video, stack to (N, 512).

    Per the convention in cached_video_features_p3d/<DS>/<dataset_lower>/<vid>.npz,
    each file has a single 'feature' (512,) float32 RAW (not L2-normalized).
    Normalizes after stack.
    """
    feats: list[np.ndarray] = []
    kept_ids: list[str] = []
    missing = 0
    cache_root = os.path.join(pool_cache_dir, dataset_subdir)
    for vid in video_ids:
        path = os.path.join(cache_root, f'{vid}.npz')
        if not os.path.exists(path):
            missing += 1
            continue
        with np.load(path) as z:
            feats.append(z['feature'].astype(np.float32))
        kept_ids.append(vid)
    if not feats:
        raise FileNotFoundError(
            f'No pool pooled features under {cache_root}; '
            f'verify P3.5 cache (--pool_cache_dir)'
        )
    arr = np.stack(feats, axis=0)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    if missing:
        print(f'WARN: pool preload skipped {missing} videos (cache miss)')
    return arr, kept_ids


def _cold_load_candidate(frame_cache_root: str, vid: str):
    """Per-query cold-load: one np.load returns (visual_output, visual_patches, video_mask).
    Matches the P3.6 cache schema: float16 features, int64 mask.
    """
    path = os.path.join(frame_cache_root, f'{vid}.npz')
    with np.load(path) as z:
        vo = torch.from_numpy(z['visual_output'].astype(np.float32))
        vp = torch.from_numpy(z['visual_patches'].astype(np.float32))
        vm = torch.from_numpy(z['video_mask']).long()
    return vo, vp, vm


def _load_model(args, device):
    XCLIP, _, PYTORCH_PRETRAINED_BERT_CACHE = _import_eercf(args.eercf_dir)
    state = torch.load(args.init_model, map_location='cpu')
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = XCLIP.from_pretrained(
        args.cross_model, cache_dir=cache_dir, state_dict=state, task_config=args,
    )
    model = model.to(device).eval()
    return model


def _resolve_train_video_ids(args) -> list[str]:
    """Setting 2 expanded pool. Inlined to avoid importing main_eercf (which has
    a module-level torch.distributed.init_process_group that requires RANK env).
    Mirrors main_eercf.get_train_video_ids_from_data_file at 2026-05-27.
    """
    train_path_map = {
        'msrvtt': args.train_csv,
        'activity': args.train_json,
        'didemo': args.train_json,
        'lsmdc': args.train_json,
    }
    train_path = train_path_map[args.datatype]
    if not train_path or not os.path.exists(train_path):
        print(f'WARN: Setting 2 train data file not found: {train_path}; '
              f'falling back to test-only pool')
        return []
    ids: set[str] = set()
    if args.datatype == 'msrvtt':
        import pandas as pd
        df = pd.read_csv(train_path)
        ids = set(df['video_id'].astype(str).unique())
    else:
        with open(train_path) as f:
            data = json.load(f)
        for item in data:
            raw = item.get('video', '')
            if args.datatype == 'activity':
                vid = raw.replace('.mp4', '')
            elif args.datatype == 'didemo':
                vid = raw.replace('.avi', '').replace('.mp4', '')
            elif args.datatype == 'lsmdc':
                vid = raw.replace('/', '_').replace('.avi', '').replace('.mp4', '')
            else:
                vid = raw
            if vid:
                ids.add(vid)
    return sorted(ids)


def _normalize_qid_for_cache(raw_or_short_qid: str, datatype: str) -> str:
    """Map a dataset-side qid into the cache .npz stem.

    For LSMDC, the cache stem is the doubled '<prefix>_<clip>' form produced by
    P3.6's extract path-normalization. Both the raw "<prefix>/<clip>.avi" form
    AND the short "<clip>" form should resolve correctly here; if a short clip
    id is passed in, this function cannot recover the prefix and will return
    it unchanged (callers should pass the raw form for LSMDC).
    """
    if datatype == 'lsmdc':
        return raw_or_short_qid.replace('/', '_').replace('.avi', '').replace('.mp4', '')
    if datatype == 'msrvtt':
        return raw_or_short_qid
    return raw_or_short_qid.replace('.mp4', '').replace('.avi', '')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Import helpers from this package (after PYTHONPATH side effects).
    if HELPERS_DIR not in sys.path:
        sys.path.insert(0, HELPERS_DIR)
    from latency_helpers import (  # noqa: E402
        WallCappedTimer, build_latency_meta, host_fingerprint, load_subset_manifest,
    )

    # ---------- setup (untimed) ----------
    print('[setup] importing EERCF + loading model ...', flush=True)
    _, ClipTokenizer, _ = _import_eercf(args.eercf_dir)
    tokenizer = ClipTokenizer()
    model = _load_model(args, device)

    print(f'[setup] building test dataset ({args.datatype}) ...', flush=True)
    dataset = _build_test_dataset(args, tokenizer)
    qid_to_caption, qid_to_raw = _qid_to_caption(dataset, args.datatype)
    print(f'[setup] qid->caption map size = {len(qid_to_caption)}', flush=True)

    print('[setup] resolving pool video ids ...', flush=True)
    dataset_subdir = args.datatype  # cache layout: <DS>/<dataset_lower>/<vid>.npz
    # Manifest qids are X-Pool short clip-ids. Cache filenames need the raw
    # dataset-side qid normalized to cache form. For LSMDC, only the raw form
    # (sentences_dict[i][0]) carries the <prefix>/ needed to reconstruct the
    # P3.6 cache stem. For ACTNET/DIDEMO/MSRVTT, raw == short.
    test_ids_norm = [
        _normalize_qid_for_cache(qid_to_raw[q], args.datatype)
        for q in qid_to_caption
    ]
    if args.setting == 2:
        train_ids_norm = _resolve_train_video_ids(args)  # already normalized
        seen = set(test_ids_norm)
        train_ids_norm = [v for v in train_ids_norm if v not in seen]
        pool_ids_norm = test_ids_norm + train_ids_norm
        print(f'[setup] Setting 2 pool: {len(test_ids_norm)} test + {len(train_ids_norm)} train '
              f'= {len(pool_ids_norm)} videos', flush=True)
    else:
        pool_ids_norm = test_ids_norm
        print(f'[setup] Setting 1 pool: {len(pool_ids_norm)} test videos', flush=True)

    print('[setup] preloading pool pooled features (untimed) ...', flush=True)
    pool_pooled_np, pool_ids_norm = _load_pool_pooled(
        args.pool_cache_dir, pool_ids_norm, dataset_subdir,
    )
    pool_pooled = torch.from_numpy(pool_pooled_np).to(device)  # (N_pool, 512)
    print(f'[setup] pool tensor on device: shape={tuple(pool_pooled.shape)} '
          f'dtype={pool_pooled.dtype}', flush=True)

    # The rerank-side frame cache root.
    frame_cache_root = os.path.join(args.frame_cache_dir, dataset_subdir)

    # Manifest + timer
    manifest = load_subset_manifest(args.subset_manifest)
    warmup_ids = list(manifest['warmup_query_ids'])
    timed_ids = list(manifest['timed_query_ids'])
    missing_w = [q for q in warmup_ids[: args.warmup_n_used] if q not in qid_to_caption]
    missing_t = [q for q in timed_ids if q not in qid_to_caption]
    if missing_w or missing_t:
        print(f'WARN: manifest qids missing from EERCF loader — '
              f'warmup={len(missing_w)} timed={len(missing_t)}')

    timer = WallCappedTimer(
        warmup_ids=warmup_ids, timed_ids=timed_ids,
        wall_cap_s=args.wall_time_cap_s,
        warmup_n_used=args.warmup_n_used,
    )

    # ---------- per-query loop ----------
    def _run_one(qid: str):
        sent = qid_to_caption[qid]
        raw_qid = qid_to_raw.get(qid, qid)  # _get_text accepts raw form
        # MSRVTT/ACTNET/DIDEMO _get_text returns 4 values (incl. choice_video_ids);
        # LSMDC returns 3. Unpack defensively.
        _tup = dataset._get_text(raw_qid, sent)
        pairs_text, pairs_mask, pairs_segment = _tup[0], _tup[1], _tup[2]
        input_ids = torch.from_numpy(pairs_text).long().to(device)         # (1, max_words)
        input_mask = torch.from_numpy(pairs_mask).long().to(device)
        segment_ids = torch.from_numpy(pairs_segment).long().to(device)
        with torch.no_grad():
            sequence_output, seq_features = model.get_sequence_output(
                input_ids, segment_ids, input_mask,
            )
            # Initial recall: text @ global_mat_weight @ pool.T (matches
            # eval_epoch_expanded_pool initial_sim formula).
            text_feat = sequence_output.squeeze(1)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
            text_feat = torch.matmul(text_feat, model.global_mat_weight)   # (1, 512)
            logit_scale = model.clip.logit_scale.exp()
            initial_scores = logit_scale * torch.matmul(text_feat, pool_pooled.t())
            initial_scores = initial_scores.squeeze(0)                     # (N_pool,)

            topk_scores, topk_idx = torch.topk(initial_scores, k=min(args.rerantopk, initial_scores.numel()))
            topk_idx_cpu = topk_idx.detach().cpu().numpy().tolist()
            topk_vids = [pool_ids_norm[i] for i in topk_idx_cpu]

            # Cold-load top-K rerank features from .npz (this is the symmetric
            # per-candidate disk I/O cost vs X-Pool's frame_embed cold-load).
            vo_list, vp_list, vm_list = [], [], []
            for vid in topk_vids:
                vo, vp, vm = _cold_load_candidate(frame_cache_root, vid)
                vo_list.append(vo); vp_list.append(vp); vm_list.append(vm)
            visual_output = torch.stack(vo_list, dim=0).to(device)         # (K, F, 512)
            visual_patches = torch.stack(vp_list, dim=0).to(device)        # (K, F, P, 512)
            video_mask = torch.stack(vm_list, dim=0).to(device)            # (K, F)

            # Broadcast text features to K so get_similarity_logits sees the
            # same row repeated against K candidate videos (per-query single text).
            seq_K = sequence_output.expand(args.rerantopk, -1, -1).contiguous()
            seqf_K = seq_features.expand(args.rerantopk, -1, -1).contiguous() if seq_features.dim() == 3 else seq_features
            mask_K = input_mask.expand(args.rerantopk, -1).contiguous()

            start_patch = args.strongpatch_weight > 0
            TIB_output, _ = model.get_similarity_logits(
                seq_K, seqf_K, visual_output, visual_patches, mask_K, video_mask,
                loose_type=model.loose_type, start_patch_eval=start_patch,
            )
            sv = TIB_output[0].diagonal().cpu().numpy()
            swf = TIB_output[1].diagonal().cpu().numpy()
            ssf = TIB_output[2].diagonal().cpu().numpy()
            swp = TIB_output[3].diagonal().cpu().numpy()
            ssp = TIB_output[4].diagonal().cpu().numpy()

        # Rerank fusion (mirrors main_eercf.py eval_epoch lines 1116-1121,
        # collapsed for single-query single-row case).
        ii, jj, kk = 0.8, 0.2, 0.5
        rerank_logits = sv * args.video_weight + swf * args.weakframe_weight + swp * args.weakpatch_weight
        rerank_logits = (
            rerank_logits * args.weak_loss_weight
            + (ssf * args.strongframe_weight + ssp * args.strongpatch_weight) * args.strong_loss_weight
        )
        # Apply the second-stage rerank boost: re-score top-K with the
        # multi-level fusion (boost is 100 + weighted sum); per-query, the
        # +100 constant cancels in ranking, so simulate just the weighted sum.
        fused = swf * ii + (1 - ii) * ssf + jj * swp + kk * sv

        order = np.argsort(-fused)
        top100 = order[: min(100, len(order))]
        top100_vids = [topk_vids[i] for i in top100]
        top100_scores = [float(fused[i]) for i in top100]
        return top100_vids, top100_scores

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Warmup (untimed).
    print(f'[warmup] {args.warmup_n_used} queries ...', flush=True)
    for qid in timer.warmup_iter():
        if qid in qid_to_caption:
            try:
                _ = _run_one(qid)
            except Exception as exc:
                print(f'[warmup] {qid} skipped: {exc}', flush=True)

    # Timed phase.
    print(f'[timed] up to {len(timed_ids)} queries (cap {args.wall_time_cap_s:.0f}s) ...',
          flush=True)
    results = []
    for qid in timer.timed_iter():
        if qid not in qid_to_caption:
            continue
        try:
            with timer.measure():
                top100_vids, top100_scores = _run_one(qid)
        except Exception as exc:
            print(f'[timed] {qid} skipped: {exc}', flush=True)
            continue
        results.append({
            'query_idx': len(results),
            'query_text': qid_to_caption[qid],
            'ground_truth_video_id': qid,
            'candidates': top100_vids,
            'scores': top100_scores,
            'num_candidates': len(top100_vids),
        })

    summary = timer.summarize()
    print(f'[done] processed={summary["n_processed"]}/{summary["n_target"]} '
          f'mean={summary["mean_ms"]:.2f} ms validity={summary["validity"]}',
          flush=True)

    # Compose canonical Pass-B metadata + results.
    lat_meta = build_latency_meta(
        'stage1_latency_ms', summary, manifest, args.warmup_n_used,
        extra={
            'strict_latency_contract': 'eercf_perquery_cold_load_topk',
            'rerantopk': int(args.rerantopk),
        },
    )['stage1_latency_ms']

    avg_cands = sum(r['num_candidates'] for r in results) / max(1, len(results))
    metadata = {
        'method': 'eercf',
        'dataset': args.datatype,
        'setting': args.setting,
        'rerantopk': int(args.rerantopk),
        'timestamp': time.strftime('%m%d%H%M'),
        'subset_manifest': args.subset_manifest,
        'subset_manifest_sha256': manifest['metadata'].get('content_sha256', ''),
        'host_fingerprint': host_fingerprint(),
        'stage1_latency_ms': lat_meta,
    }
    metrics = {
        'avg_candidates_per_query': float(avg_cands),
        'total_queries': int(len(results)),
        'pool_size': int(len(pool_ids_norm)),
    }

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({'metadata': metadata, 'metrics': metrics, 'results': results}, f, indent=2)
    print(f'[write] {args.output_json}', flush=True)


if __name__ == '__main__':
    main()
