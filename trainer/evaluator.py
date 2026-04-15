import json
import os
import time
from collections import defaultdict, Counter

import faiss
import numpy as np
import torch
import wandb
from tqdm import tqdm

from models.grdr import GRDR, QuantizeOutput, VideoOutput
from utils.model_utils import create_videorqvae, sinkhorn_raw
from utils.training_utils import safe_load
from utils.data_utils import load_shared_features
from data.video_dataset import VideoTextDataset, collate_fn


def eval_all(predict, label):
    """Evaluate all recall metrics at k=1, 5, 10."""
    log_dict = {}
    log_dict.update(eval_recall(predict, label, at=1))
    log_dict.update(eval_recall(predict, label, at=5))
    log_dict.update(eval_recall(predict, label, at=10))

    # Convert to percentage
    for k, v in log_dict.items():
        log_dict[k] = v * 100

    return log_dict


def base_it(predict, label, at, score_func):
    """Base iteration function for evaluation."""
    assert len(predict) == len(label)
    scores = []
    for pred, lbs in zip(predict, label):
        pred = pred.tolist() if not isinstance(pred, list) else pred
        best_score = 0.
        if not isinstance(lbs, list):
            lbs = [lbs]
        for lb in lbs:
            if isinstance(lb, list):
                lb = lb[0]
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            best_score = max(best_score, cur_score)
        scores.append(best_score)
    return scores


def eval_recall(predict, label, at=10):
    """Evaluate recall at a specific k value."""
    scores = base_it(predict, label, at, lambda rank: int(rank != 0))
    return {f'R@{at}': sum(scores) / len(scores)}


def compute_sid_collision_stats(sample_codes_dict, num_latent_tokens):
    """Compute token frequency stats and collision rate for sIDs."""
    flat_codes = []
    for token_codes in sample_codes_dict.values():
        for code in token_codes:
            flat_codes.append(str(code))

    total_slots = len(sample_codes_dict) * num_latent_tokens
    freq = Counter(flat_codes)
    unique_count = len(freq)
    max_frequency = max(freq.values()) if freq else 0
    min_frequency = min(freq.values()) if freq else 0
    utility = unique_count / total_slots if total_slots > 0 else 0.0

    return {
        "frequency": freq,
        "max_frequency": max_frequency,
        "min_frequency": min_frequency,
        "unique_count": unique_count,
        "utility": utility,
    }


def compute_train_test_collision(train_code_path: str, test_codes_dict: dict) -> dict:
    """Compute collision between train and test sIDs."""
    with open(train_code_path) as f:
        train_codes = json.load(f)
    train_sids = {tuple(code) for code in train_codes.values()}

    test_sids = {tuple(code) for codes in test_codes_dict.values() for code in codes}

    collision = train_sids & test_sids
    collision_rate = len(collision) / len(test_sids) if test_sids else 0.0

    return {
        "collision_count": len(collision),
        "train_unique": len(train_sids),
        "test_unique": len(test_sids),
        "collision_rate": collision_rate
    }


@torch.no_grad()
def our_encode_dual(data_loader, model: GRDR, type='both', residual_layer=None, return_all=False):
    """Encode videos and/or queries from a data loader."""
    if type not in ('both', 'video', 'query'):
        raise ValueError(f"Invalid type: {type}. Must be 'both', 'video', or 'query'")

    encode_video = type in ('both', 'video')
    encode_query = type in ('both', 'query')

    video_embeddings_list, video_code_list = [], []
    query_embeddings_list, query_code_list = [], []
    sample_keys_ordered = []

    for batch in tqdm(data_loader):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items() if v is not None}

        if encode_video:
            video_output: VideoOutput = model(
                video_features=batch['video_features'],
                token_idx=batch['token_idx'],
                return_code=False,
                return_quantized_embedding=False,
                use_constraint=False,
                return_residual_layer=residual_layer,
                return_all=return_all
            )
            video_emb = video_output.continuous_embeds.cpu().numpy()
            if video_output.probability is not None:
                video_codes = video_output.probability.argmax(-1).cpu().tolist()
            else:
                video_codes = [None] * video_emb.shape[0]

        if encode_query:
            query_output: QuantizeOutput = model(
                input_ids=batch['caption_tokens'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['ids'],
                aux_ids=batch.get('aux_ids'),
                return_code=False,
                return_quantized_embedding=False,
                use_constraint=False
            )
            query_emb = query_output.total_embeds.cpu().numpy()
            query_codes = query_output.probability.argmax(-1).cpu().tolist()

        for i in range(len(batch['video_ids'])):
            sample_key = batch['video_ids'][i]
            sample_keys_ordered.append(sample_key)
            if encode_video:
                video_embeddings_list.append(video_emb[i])
                video_code_list.append(video_codes[i])
            if encode_query:
                query_embeddings_list.append(query_emb[i])
                query_code_list.append(query_codes[i])

    if encode_video:
        video_embeddings_array = np.array(video_embeddings_list, dtype=np.float32)
        video_code_dict = dict(zip(sample_keys_ordered, video_code_list))
    if encode_query:
        query_embeddings_array = np.array(query_embeddings_list, dtype=np.float32)
        query_code_dict = dict(zip(sample_keys_ordered, query_code_list))

    if type == 'both':
        return (video_embeddings_array, video_code_dict, sample_keys_ordered,
                query_embeddings_array, query_code_dict, sample_keys_ordered)
    elif type == 'video':
        return (video_embeddings_array, video_code_dict, sample_keys_ordered)
    else:
        return (query_embeddings_array, query_code_dict, sample_keys_ordered)


def build_index(collection, shard=True, dim=None, gpu=True):
    """Build FAISS index for retrieval."""
    t = time.time()
    dim = collection.shape[1] if dim is None else dim
    cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if gpu:
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        index = gpu_index
    else:
        index = cpu_index

    index.add(collection)
    print(f'build index of {len(collection)} instances, time cost ={time.time() - t}')
    return index


def do_retrieval(xq, index, k=1):
    """Perform retrieval using FAISS index."""
    t = time.time()
    distance, rank = index.search(xq, k)
    print(f'search {len(xq)} queries, time cost ={time.time() - t}')
    return rank, distance


def do_maxsim_retrieval(query_emb, video_emb, k=100):
    """Perform MaxSim retrieval for multi-token embeddings."""
    t = time.time()
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    video_norm = video_emb / (np.linalg.norm(video_emb, axis=2, keepdims=True) + 1e-8)

    N_q, dim = query_norm.shape
    N_v, num_tokens, _ = video_norm.shape

    sim = np.einsum('qd,vtd->qvt', query_norm, video_norm)
    max_sim = sim.max(axis=2)

    rank = np.argsort(-max_sim, axis=1)[:, :k]
    distance = np.take_along_axis(max_sim, rank, axis=1)

    print(f'MaxSim search {N_q} queries over {N_v} videos ({num_tokens} tokens), time={time.time() - t:.2f}s')
    return rank, distance


def summarize_recall(rank, ks=(1, 5, 10)):
    """Summarize recall metrics at various k values."""
    if not isinstance(rank, np.ndarray):
        rank = np.array(rank)
    num_q = rank.shape[0]
    gt = np.arange(num_q).reshape(num_q, 1)
    results = {}
    for k in ks:
        hits = (rank[:, :k] == gt).any(axis=1)
        recall = float(hits.mean()) if num_q > 0 else 0.0
        results[k] = recall
    summary = " | ".join([f"R@{k}:{results[k] * 100:.2f}%" for k in ks])
    print(f"[Recall] {summary}")
    return results


def build_sid_to_videos_mapping(sample_codes_dict):
    """Build reverse mapping from semantic ID strings to video IDs."""
    sid_to_videos = defaultdict(list)

    for video_id, token_codes in sample_codes_dict.items():
        for code in token_codes:
            sid_str = str([0, *code])
            sid_to_videos[sid_str].append(video_id)

    return dict(sid_to_videos)


@torch.no_grad()
def save_code(model, train_dataset, video_codes, tokenizer, batch_size, save_path):
    """Save hierarchical codes for training samples."""
    model.eval()
    collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_wrapper,
        batch_size=batch_size, shuffle=False, num_workers=8
    )

    (_, video_code_dict, video_keys) = our_encode_dual(train_data_loader, model, type='video')

    if video_codes is None:
        video_codes = train_dataset.video_codes

    all_sample_codes = {}
    for sample_key in video_keys:
        prev_code = video_codes.get(sample_key, [0])
        new_code = video_code_dict[sample_key]
        hierarchical_code = prev_code[1:] + [new_code]
        all_sample_codes[sample_key] = hierarchical_code

    json.dump(all_sample_codes, open(f'{save_path}/best_model.pt.code', 'w'))
    print(f'Saved {len(all_sample_codes)} hierarchical codes to {save_path}/best_model.pt.code')


@torch.no_grad()
def eval_retrieval(model, train_dataset, val_dataset, test_dataset, tokenizer, batch_size, accelerator, global_step=0,
                   is_pretrain=False, code_length=4, drift_monitor=None):
    """Evaluate Dense Retrieval on test set. Also evaluates sID Retrieval when not in pretrain phase."""
    # Import Tree here to avoid circular import
    from utils.model_utils import Tree

    model.eval()
    collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=collate_wrapper,
        batch_size=batch_size, shuffle=False, num_workers=8
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_wrapper,
        batch_size=batch_size, shuffle=False, num_workers=8
    )

    accelerator.print('Evaluating Dense Retrieval on test set...')
    (test_video_emb, _, test_video_keys,
     test_query_emb, _, test_query_keys) = our_encode_dual(test_data_loader, model, type='both', return_all=True)

    accelerator.print(f'Test video embeddings shape: {test_video_emb.shape}')
    accelerator.print(f'Test query embeddings shape: {test_query_emb.shape}')

    if len(test_video_emb.shape) == 3:
        rank, distance = do_maxsim_retrieval(test_query_emb, test_video_emb, k=100)
    else:
        index = build_index(test_video_emb, gpu=False)
        rank, distance = do_retrieval(test_query_emb, index, k=100)
    results = summarize_recall(rank, ks=(1, 5, 10))

    unwrap_model = accelerator.unwrap_model(model)

    train_sample_codes_dict = unwrap_model.gen_sid(train_data_loader)

    if drift_monitor is not None:
        drift_metrics = drift_monitor.compute_drift(train_sample_codes_dict)
        accelerator.print(
            f'[Drift Monitor] Total drift: {drift_metrics["drift_rate_total"]:.2f}% '
            f'({drift_metrics["drifted_count"]}/{drift_metrics["total_samples"]} samples)'
        )
        for layer_idx, rate in enumerate(drift_metrics['drift_rate_per_layer']):
            accelerator.print(f'  Layer {layer_idx}: {rate:.2f}% drift')
        if accelerator.is_main_process:
            wandb.log({
                'drift/total_rate': drift_metrics['drift_rate_total'],
                **{f'drift/layer_{i}_rate': r for i, r in enumerate(drift_metrics['drift_rate_per_layer'])}
            }, step=global_step)

    if not is_pretrain:
        accelerator.print('Evaluating sID Retrieval on test set...')

        unique_train_sids = {str(code) for token_codes in train_sample_codes_dict.values() for code in token_codes}
        accelerator.print(f'Unique train sID count: {len(unique_train_sids)}')

        sample_codes_dict = unwrap_model.gen_sid(test_data_loader)
        num_latent_tokens = getattr(getattr(unwrap_model, "video_rqvae", None), "num_latent_tokens", None)
        if num_latent_tokens is None and sample_codes_dict:
            num_latent_tokens = len(next(iter(sample_codes_dict.values())))
        sid_stats = compute_sid_collision_stats(sample_codes_dict, num_latent_tokens or 0)
        accelerator.print(
            f'Test sID token frequency - max: {sid_stats["max_frequency"]}, '
            f'min: {sid_stats["min_frequency"]}, '
            f'unique: {sid_stats["unique_count"]}, '
            f'utility: {sid_stats["utility"]:.6f} '
        )

        if accelerator.is_main_process:
            wandb.log({
                "eval/test_sid_utility": sid_stats["utility"],
            }, step=global_step)

        train_sids = {tuple(code) for codes in train_sample_codes_dict.values() for code in codes}
        test_sids = {tuple(code) for codes in sample_codes_dict.values() for code in codes}
        collision = train_sids & test_sids
        collision_rate = len(collision) / len(test_sids) if test_sids else 0.0
        accelerator.print(
            f'Train-Test Collision - train_unique: {len(train_sids)}, '
            f'test_unique: {len(test_sids)}, collision: {len(collision)}, rate: {collision_rate:.4f}'
        )
        if accelerator.is_main_process:
            wandb.log({"eval/train_test_collision_rate": collision_rate}, step=global_step)

        # Validation Set Evaluation
        accelerator.print('Evaluating sID Retrieval on validation set...')

        val_collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, collate_fn=val_collate_wrapper, batch_size=batch_size,
            shuffle=False, num_workers=8
        )

        val_sample_codes_dict = unwrap_model.gen_sid(val_data_loader)

        val_corpus_ids = []
        for token_codes in val_sample_codes_dict.values():
            for code in token_codes:
                val_corpus_ids.append([0, *code])
        val_tree = Tree()
        val_tree.set_all(val_corpus_ids)

        val_query_labels = []
        for sample in val_dataset.samples:
            video_id = sample['video_id']
            gt_sids = val_sample_codes_dict[video_id]
            gt_sid_strs = [str([0, *code]) for code in gt_sids]
            val_query_labels.append(gt_sid_strs)

        val_tk0 = tqdm(val_data_loader, total=len(val_data_loader), desc='Val sID')
        val_output_all = []
        top_k = 10
        with torch.no_grad():
            for batch in val_tk0:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                output = unwrap_model.generate(
                    input_ids=batch['caption_tokens'],
                    attention_mask=batch['attention_mask'],
                    max_length=code_length + 1,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=val_tree
                )
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line.cpu().tolist())
                new_output.append(beam)
                val_output_all.extend(new_output)

        val_predictions = []
        for generated_codes in val_output_all:
            seen_sids = set()
            sid_list = []
            for code in generated_codes:
                code_str = str(code)
                if code_str not in seen_sids:
                    seen_sids.add(code_str)
                    sid_list.append(code_str)
            val_predictions.append(sid_list)

        val_sid_results = eval_all(val_predictions, val_query_labels)
        accelerator.print('Validation sID Retrieval:', val_sid_results)

        if accelerator.is_main_process:
            wandb.log({f"eval/val_sID_R@{k}": v for k, v in val_sid_results.items()}, step=global_step)

        # Test Set Evaluation
        accelerator.print('Evaluating sID Retrieval on test set...')

        query_labels = []
        for sample in test_dataset.samples:
            video_id = sample['video_id']
            gt_sids = sample_codes_dict[video_id]
            gt_sid_strs = [str([0, *code]) for code in gt_sids]
            query_labels.append(gt_sid_strs)

        corpus_ids = []
        for token_codes in sample_codes_dict.values():
            for code in token_codes:
                corpus_ids.append([0, *code])
        tree = Tree()
        tree.set_all(corpus_ids)

        tk0 = tqdm(test_data_loader, total=len(test_data_loader), desc='sID Retrieval')
        output_all = []
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                top_k = 10
                output = unwrap_model.generate(
                    input_ids=batch['caption_tokens'],
                    attention_mask=batch['attention_mask'],
                    max_length=code_length + 1,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree
                )
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line.cpu().tolist())
                new_output.append(beam)
                output_all.extend(new_output)

        predictions = []
        for generated_codes in output_all:
            seen_sids = set()
            sid_list = []

            for code in generated_codes:
                code_str = str(code)
                if code_str not in seen_sids:
                    seen_sids.add(code_str)
                    sid_list.append(code_str)

            predictions.append(sid_list)

        sid_results = eval_all(predictions, query_labels)
        accelerator.print('sID Retrieval:', sid_results)

        if accelerator.is_main_process:
            wandb.log({f"eval/sID_R@{k}": v for k, v in sid_results.items()}, step=global_step)

        results.update({f"sID_{k}": v for k, v in sid_results.items()})

    if is_pretrain:
        overall_metric = sum(results.values())
    else:
        overall_metric = sum(sid_results.values())

    return results, overall_metric


def compute_detailed_metrics(results, predictions, labels, total_time, num_queries, num_candidates):
    """Compute detailed evaluation metrics."""
    metrics = {}
    for k in [1, 5, 10, num_candidates]:
        recall_result = eval_recall(predictions, labels, at=k)
        metrics[f'Recall@{k}'] = recall_result[f'R@{k}']

    metrics['seconds_per_query'] = total_time / num_queries if num_queries > 0 else 0
    metrics['total_queries'] = num_queries
    metrics['batch_size'] = 1

    for k in [1, 5, 10, num_candidates]:
        correct_count = 0
        for pred, lbs in zip(predictions, labels):
            if any(lb in pred[:k] for lb in lbs):
                correct_count += 1
        metrics[f'correct_retrievals_at_{k}'] = correct_count

    metrics['correct_retrievals'] = metrics.get(f'correct_retrievals_at_{num_candidates}', 0)

    avg_candidates = sum(r['num_candidates'] for r in results) / len(results) if results else 0
    metrics['avg_candidates_per_query'] = round(avg_candidates, 2)

    return metrics


def save_candidates_json(results, metrics, config, output_dir, timestamp):
    """Save candidates results to JSON file."""
    dataset = config.get('dataset', 'unknown')
    code_num = config.get('code_num', 0)
    code_length = config.get('max_length', 0)
    num_candidates = max(1, config.get('num_candidates', 20))
    num_latent_tokens = config.get('num_latent_tokens', 4)
    model_name = config.get('prev_model', 'unknown')
    setting = config.get('setting', 1)

    metadata = {
        "dataset": dataset,
        "model_name": model_name,
        "num_candidates": num_candidates,
        "index_type": "videorqvae",
        "code_book_size": code_num,
        "code_book_num": num_latent_tokens,
        "timestamp": timestamp
    }

    output_data = {
        "metadata": metadata,
        "metrics": metrics,
        "results": results
    }

    filename = f"{dataset}_c{code_num}l{code_length}_{num_candidates}_candidates_t{setting}.json"

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Candidates JSON saved to: {filepath}")
    print(f"{'='*80}\n")

    return filepath


def test(config):
    """Main test/evaluation function for GRDR model."""
    from utils.data_utils import write_pkl
    from utils.model_utils import Tree

    model_name = config.get('model_name', 't5-small')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 3)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)
    batch_size = config.get('batch_size')
    epochs = config.get('epochs', 1)

    dataset_name = config.get('dataset', 'msrvtt')
    features_root = config.get('features_root', './data_process/datasets/features')
    num_latent_tokens = config.get('num_latent_tokens', 4)
    cache_dir = config.get('cache_dir', './cache')
    use_pseudo_queries = config.get('use_pseudo_queries', False)
    detailed_generation = config.get('detailed_generation', False)

    print(f'Loading features for {dataset_name}...')
    feature_cache = load_shared_features(
        dataset_name=dataset_name,
        features_root=features_root,
        logger=print,
        use_pseudo_queries=use_pseudo_queries
    )

    from transformers import T5Config, AutoTokenizer
    from models.t5 import T5ForConditionalGeneration

    t5_config = T5Config.from_pretrained(model_name)
    t5_config.dropout_rate = config.get('dropout_rate', t5_config.dropout_rate)

    if config.get('float16', False):
        torch_dtype = torch.float16
    elif config.get('bf16', False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    videorqvae = create_videorqvae(
        code_num=code_num,
        code_length=code_length,
        num_latent_tokens=num_latent_tokens,
        e_dim=t5_config.d_model,
        in_dim=config.get('in_dim', 512),
        device='cuda'
    )

    t5 = T5ForConditionalGeneration.from_pretrained(model_name,
                                                    torch_dtype=torch_dtype,
                                                    config=t5_config)
    model = GRDR(model=t5, use_constraint=False, code_length=code_length, zero_inp=False,
                 code_number=code_num, videorqvae=videorqvae)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if prev_id is not None:
        prev_codes_dict = json.load(open(prev_id))
        video_codes = {k: [0, *v] for k, v in prev_codes_dict.items()}
    else:
        video_codes = None

    dataset = VideoTextDataset(
        dataset_name=dataset_name,
        video_features=feature_cache['test_video'],
        text_features=feature_cache['test_text'],
        tokenizer=tokenizer,
        split='test',
        max_text_len=128,
        num_latent_tokens=num_latent_tokens,
        cache_dir=cache_dir,
        ids=video_codes
    )

    collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_wrapper, batch_size=batch_size,
        shuffle=False, num_workers=16
    )

    setting = config.get('setting', 1)
    print(f"Setting: {setting} ({'train+test combined pool' if setting == 2 else 'test only pool'})")

    train_data_loader = None
    if setting == 2:
        print("Creating train dataset for combined pool...")
        train_dataset = VideoTextDataset(
            dataset_name=dataset_name,
            video_features=feature_cache['train_video'],
            text_features=feature_cache['train_text'],
            tokenizer=tokenizer,
            split='train',
            max_text_len=128,
            num_latent_tokens=num_latent_tokens,
            cache_dir=cache_dir,
            ids=video_codes
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=collate_wrapper, batch_size=batch_size,
            shuffle=False, num_workers=16
        )
        print(f"Train dataset: {len(train_dataset)} samples")

    model = model.cuda()
    model.eval()

    best_model_path = config.get('eval_checkpoint', None)
    if not os.path.exists(best_model_path):
        print(f'Best model not found: {best_model_path}')
        return

    print(f'Test {best_model_path}')
    safe_load(model, best_model_path)

    # Dense Retrieval Evaluation
    print('Evaluating Dense Retrieval on test set...')
    (test_video_emb, _, test_video_keys,
     test_query_emb, _, test_query_keys) = our_encode_dual(data_loader, model, type='both', return_all=True)

    print(f'Test video embeddings shape: {test_video_emb.shape}')
    print(f'Test query embeddings shape: {test_query_emb.shape}')

    # Generate semantic IDs
    if detailed_generation and setting == 1:
        sample_codes_dict, sid_to_features = model.gen_sid(data_loader, return_quantized_features=True)

        candidates_dir = "candidates"
        sid_features_filename = f"{dataset_name}_sid_quantized_features_c{code_num}l{code_length}_t{setting}.pkl"
        sid_features_path = os.path.join(candidates_dir, sid_features_filename)
        os.makedirs(candidates_dir, exist_ok=True)
        write_pkl(sid_to_features, sid_features_path)
        print(f"Saved {len(sid_to_features)} unique sID quantized features to: {sid_features_path}")
    else:
        sample_codes_dict = model.gen_sid(data_loader)

    num_latent_tokens = getattr(getattr(model, "video_rqvae", None), "num_latent_tokens", None)
    if num_latent_tokens is None and sample_codes_dict:
        num_latent_tokens = len(next(iter(sample_codes_dict.values())))
    sid_stats = compute_sid_collision_stats(sample_codes_dict, num_latent_tokens or 0)
    print(
        f'Test sID token frequency - max: {sid_stats["max_frequency"]}, '
        f'min: {sid_stats["min_frequency"]}, '
        f'unique: {sid_stats["unique_count"]}, '
        f'utility: {sid_stats["utility"]:.6f} '
    )

    # Generate train sIDs and combine when setting=2
    if setting == 2 and train_data_loader is not None:
        print("Generating semantic IDs for train set...")
        raw_train_codes = model.gen_sid(train_data_loader)

        train_sample_codes_dict = {}
        for video_id, codes in raw_train_codes.items():
            parts = video_id.rsplit('_', 1)
            base_video_id = parts[0] if (len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2) else video_id
            if base_video_id not in train_sample_codes_dict:
                train_sample_codes_dict[base_video_id] = codes

        print(f"Train videos: {len(raw_train_codes)} samples -> {len(train_sample_codes_dict)} unique videos")

        train_sid_stats = compute_sid_collision_stats(train_sample_codes_dict, num_latent_tokens or 0)
        print(
            f'Train sID token frequency - max: {train_sid_stats["max_frequency"]}, '
            f'min: {train_sid_stats["min_frequency"]}, '
            f'unique: {train_sid_stats["unique_count"]}, '
            f'utility: {train_sid_stats["utility"]:.6f} '
        )

        test_sample_codes_dict = {}
        for video_id, codes in sample_codes_dict.items():
            parts = video_id.rsplit('_', 1)
            base_video_id = parts[0] if (len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2) else video_id
            if base_video_id not in test_sample_codes_dict:
                test_sample_codes_dict[base_video_id] = codes

        combined_sample_codes_dict = {**train_sample_codes_dict, **test_sample_codes_dict}
        print(f"Combined pool: {len(train_sample_codes_dict)} train + {len(test_sample_codes_dict)} test = {len(combined_sample_codes_dict)} total videos")
    else:
        combined_sample_codes_dict = {}
        for video_id, codes in sample_codes_dict.items():
            parts = video_id.rsplit('_', 1)
            base_video_id = parts[0] if (len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2) else video_id
            if base_video_id not in combined_sample_codes_dict:
                combined_sample_codes_dict[base_video_id] = codes

    train_code_path = f"{best_model_path}.code"
    if os.path.exists(train_code_path):
        collision_stats = compute_train_test_collision(train_code_path, sample_codes_dict)
        print(
            f'Train-Test Collision - '
            f'train_unique: {collision_stats["train_unique"]}, '
            f'test_unique: {collision_stats["test_unique"]}, '
            f'collision: {collision_stats["collision_count"]}, '
            f'rate: {collision_stats["collision_rate"]:.4f}'
        )

    print("Building sID-to-videos mapping...")
    sid_to_videos = build_sid_to_videos_mapping(combined_sample_codes_dict)
    print(f"Mapping complete: {len(sid_to_videos)} unique sIDs")

    query_labels = []
    for sample in dataset.samples:
        video_id = sample['video_id']
        gt_sids = sample_codes_dict[video_id]
        gt_sid_strs = [str([0, *code]) for code in gt_sids]
        query_labels.append(gt_sid_strs)

    corpus_ids = []
    for token_codes in combined_sample_codes_dict.values():
        for code in token_codes:
            corpus_ids.append([0, *code])
    tree = Tree()
    tree.set_all(corpus_ids)

    results = []
    num_candidates = config.get('num_candidates', 20)
    start_time = time.time()

    tk0 = tqdm(data_loader, total=len(data_loader))
    output_all = []
    scores_all = []
    with torch.no_grad():
        for batch in tk0:
            batch_tensor = {k: v.cuda() for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            gen_output = model.generate(
                input_ids=batch_tensor['caption_tokens'],
                attention_mask=batch_tensor['attention_mask'],
                max_length=code_length + 1,
                num_beams=num_candidates,
                num_return_sequences=num_candidates,
                prefix_allowed_tokens_fn=tree,
                return_dict_in_generate=True,
                output_scores=True
            )
            output = gen_output.sequences
            batch_scores = gen_output.sequences_scores.cpu().tolist()

            beam = []
            beam_scores = []
            new_output = []
            new_scores = []
            for idx, line in enumerate(output):
                if len(beam) >= num_candidates:
                    new_output.append(beam)
                    new_scores.append(beam_scores)
                    beam = []
                    beam_scores = []
                beam.append(line.cpu().tolist())
                beam_scores.append(batch_scores[idx])
            new_output.append(beam)
            new_scores.append(beam_scores)
            output_all.extend(new_output)
            scores_all.extend(new_scores)

    end_time = time.time()
    total_time = end_time - start_time

    predictions = []
    for idx, (generated_codes, beam_scores) in enumerate(zip(output_all, scores_all)):
        sample = dataset.samples[idx]
        query_text = sample['caption']
        gt_video_id = sample['video_id']

        seen_sids = set()
        sid_list = []
        ranked_videos = []
        ranked_videos_with_sid = []
        ranked_scores = []
        seen_videos = set()

        for code, score in zip(generated_codes, beam_scores):
            code_str = str(code)

            if code_str not in seen_sids:
                seen_sids.add(code_str)
                sid_list.append(code_str)

            if code_str in sid_to_videos:
                for video_id in sid_to_videos[code_str]:
                    if video_id not in seen_videos:
                        seen_videos.add(video_id)
                        ranked_videos.append(video_id)
                        ranked_videos_with_sid.append([code_str, video_id])
                        ranked_scores.append(score)

        def strip_video_suffix(video_id):
            if '_' in video_id:
                parts = video_id.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
                    return parts[0]
            return video_id

        cleaned_gt_video_id = strip_video_suffix(gt_video_id)
        cleaned_ranked_videos = ranked_videos

        result = {
            "query_text": query_text,
            "ground_truth_video_id": cleaned_gt_video_id,
        }

        if detailed_generation:
            result["ground_truth_sID"] = query_labels[idx]

        if detailed_generation:
            result["candidates"] = ranked_videos_with_sid
        else:
            result["candidates"] = cleaned_ranked_videos

        result["scores"] = ranked_scores
        result["num_candidates"] = len(cleaned_ranked_videos)
        results.append(result)

        predictions.append(sid_list)

    eval_results = eval_all(predictions, query_labels)
    print('Test', eval_results)

    num_queries = len(dataset.samples)
    metrics = compute_detailed_metrics(results, predictions, query_labels, total_time, num_queries, num_candidates)

    timestamp = time.strftime('%m%d%H%M')
    candidates_dir = "candidates"
    save_candidates_json(results, metrics, config, candidates_dir, timestamp)


def kmeans(x, ncentroids=10, niter=100, seed=42):
    """Run FAISS k-means clustering."""
    verbose = True
    x = np.array(x, dtype=np.float32)
    d = x.shape[1]
    n = x.shape[0] // 10
    model = faiss.Kmeans(d, ncentroids, niter=niter, max_points_per_centroid=n, verbose=verbose, seed=seed)
    model.train(x)
    D, I = model.index.search(x, 1)
    code = [i[0] for i in I.tolist()]
    return model.centroids, code


def skl_kmeans(x, ncentroids=10, niter=300, n_init=10, mini=False, reassign=0.01):
    """Run scikit-learn k-means clustering."""
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if x.shape[0] > 1000 or mini:
        model = MiniBatchKMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3,
                                batch_size=4096, reassignment_ratio=reassign, max_no_improvement=20, tol=1e-7,
                                verbose=1)
    else:
        model = KMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3, tol=1e-7,
                       verbose=1)
    model.fit(x)
    return model.cluster_centers_, model.labels_.tolist()


def constrained_km(data, n_clusters=512):
    """Run constrained k-means clustering."""
    from k_means_constrained import KMeansConstrained
    size_min = min(len(data) // (n_clusters * 2), n_clusters // 4)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 2, max_iter=10, n_init=10,
                            n_jobs=10, verbose=True)
    clf.fit(data)
    return clf.cluster_centers_, clf.labels_.tolist()


def balance(code, prefix=None, ncentroids=10):
    """Compute balance score for code distribution."""
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        prefix_code = defaultdict(list)
        for c, p in zip(code, prefix):
            prefix_code[p].append(c)
        scores = []
        for p, p_code in prefix_code.items():
            scores.append(balance(p_code, ncentroids=ncentroids))
        return {'Avg': sum(scores) / len(scores), 'Max': max(scores), 'Min': min(scores), 'Flat': balance(code)}
    num = [code.count(i) for i in range(ncentroids)]
    base = len(code) // ncentroids
    move_score = sum([abs(j - base) for j in num])
    score = 1 - move_score / len(code) / 2
    return score


def conflict(code, prefix=None):
    """Compute conflict statistics for code distribution."""
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


def norm_by_prefix(collection, prefix):
    """Normalize collection by prefix groups."""
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_collection = deepcopy(collection)
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


def center_pq(m, prefix):
    """Center by prefix."""
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    from copy import deepcopy
    new_m = deepcopy(m)
    for p, p_code in prefix_code.items():
        sub_m = m[p_code]
        new_m[p_code] = sub_m.mean(axis=0)
    return new_m


def norm_code_by_prefix(collection, centroids, prefix, epsilon=1):
    """Normalize codes by prefix using Sinkhorn."""
    if prefix is None:
        prefix = [0 for _ in range(len(collection))]
    attention = np.matmul(collection, centroids.T)
    prefix = [str(x) for x in prefix]
    prefix_code = defaultdict(list)
    for c, p in zip(range(len(prefix)), prefix):
        prefix_code[p].append(c)
    code = [None for _ in range(len(collection))]
    for p, p_code in prefix_code.items():
        p_collection = attention[p_code]
        distances = p_collection
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        distances = torch.tensor(centered_distances)
        Q = sinkhorn_raw(
            distances,
            epsilon,
            100,
            use_distrib_train=False
        )
        codes = torch.argmax(Q, dim=-1).tolist()
        for i, c in zip(p_code, codes):
            code[i] = c
    return code


def do_epoch_encode(model: GRDR, train_dataset: VideoTextDataset,
                    video_codes: dict, tokenizer, batch_size, save_path, epoch, n_code,
                    code_length=1,
                    dataset_name='msrvtt', features_root='./data_process/datasets/features'):
    """Encode video-text samples for an epoch and run k-means."""
    from utils.data_utils import write_pkl

    print(f'Encoding video-text samples for epoch {epoch}...')

    residual_layer = code_length - 1
    print(f'Using residual_layer={residual_layer} for K-Means (code_length={code_length})')

    collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    print('Encoding train video features and text captions for k-means...')
    (video_embeddings_array, video_code_dict, video_keys) = our_encode_dual(
        train_data_loader, model, type='video', residual_layer=residual_layer
    )

    print(f'Train video embeddings shape: {video_embeddings_array.shape}')

    write_pkl(video_embeddings_array, f'{save_path}/{epoch}.pt.collection')

    video_code_list = [video_code_dict[key] for key in video_keys]
    prev_code_list = [video_codes.get(key, [0]) for key in video_keys]

    print('Video_code balance', balance(video_code_list, prev_code_list, ncentroids=n_code))
    print('Video_code conflict', conflict(video_code_list, prev_code_list))

    normed_collection = norm_by_prefix(video_embeddings_array, prev_code_list)
    nc = n_code
    centroids, code = kmeans(normed_collection, ncentroids=nc, niter=100)
    print('Kmeans balance', balance(code, prev_code_list))
    print('Kmeans conflict', conflict(code, prev_code_list))
    write_pkl(centroids, f'{save_path}/{epoch}.pt.kmeans.{nc}')
    json.dump(code, open(f'{save_path}/{epoch}.pt.kmeans_code.{nc}', 'w'))

    print(f'Epoch {epoch} encoding complete!')


def test_dr(config, checkpoint):
    """Test Dense Retrieval for VideoTextDataset with dict-format codes."""
    from utils.data_utils import write_pkl

    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size')
    if batch_size is None:
        batch_size = 128

    dataset_name = config.get('dataset', 'msrvtt')
    features_root = config.get('features_root', './data_process/datasets/features')

    print('DR evaluation for VideoTextDataset', f'{save_path}')

    from transformers import T5Config, AutoTokenizer
    from models.t5 import T5ForConditionalGeneration

    t5_config = T5Config.from_pretrained(model_name)
    t5_config.dropout_rate = config.get('dropout_rate', t5_config.dropout_rate)

    if config.get('float16', False):
        torch_dtype = torch.float16
    elif config.get('bf16', False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    videorqvae = create_videorqvae(
        code_num=code_num,
        code_length=code_length,
        num_latent_tokens=config.get('num_latent_tokens', 4),
        e_dim=t5_config.d_model,
        in_dim=config.get('in_dim', 512),
        device='cuda'
    )

    t5 = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype, config=t5_config)
    model = GRDR(model=t5, use_constraint=False, code_length=code_length, zero_inp=False,
                 code_number=code_num, videorqvae=videorqvae)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.cuda()
    model.eval()

    use_pseudo_queries = config.get('use_pseudo_queries', False)
    print(f'Loading features for {dataset_name}...')
    feature_cache = load_shared_features(
        dataset_name=dataset_name,
        features_root=features_root,
        logger=print,
        use_pseudo_queries=use_pseudo_queries
    )

    best_model_path = checkpoint if checkpoint is not None else f'{save_path}/best_model.pt'
    if not os.path.exists(best_model_path):
        print(f'Best model not found: {best_model_path}')
        return

    print('#' * 20)
    print(f'DR evaluation {best_model_path}')

    safe_load(model, best_model_path)

    if prev_id is not None and os.path.exists(prev_id):
        prev_codes_dict = json.load(open(prev_id))
        video_codes = {k: [0, *v] for k, v in prev_codes_dict.items()}
    else:
        use_pseudo_queries = config.get('use_pseudo_queries', False)
        temp_dataset = VideoTextDataset(
            dataset_name=dataset_name,
            video_features=feature_cache['train_video'],
            text_features=feature_cache['train_text'],
            tokenizer=tokenizer,
            split='train',
            max_text_len=128,
            num_latent_tokens=config.get('num_latent_tokens', 4),
            cache_dir=config.get('cache_dir', './cache'),
            ids=None,
            use_pseudo_queries=use_pseudo_queries
        )
        video_codes = {s['video_id']: [0] for s in temp_dataset.samples}

    use_pseudo_queries = config.get('use_pseudo_queries', False)
    train_dataset = VideoTextDataset(
        dataset_name=dataset_name,
        video_features=feature_cache['train_video'],
        text_features=feature_cache['train_text'],
        tokenizer=tokenizer,
        split='train',
        max_text_len=128,
        num_latent_tokens=config.get('num_latent_tokens', 4),
        cache_dir=config.get('cache_dir', './cache'),
        ids=video_codes,
        use_pseudo_queries=use_pseudo_queries
    )

    do_epoch_encode(
        model=model,
        train_dataset=train_dataset,
        video_codes=video_codes,
        tokenizer=tokenizer,
        batch_size=batch_size,
        save_path=save_path,
        epoch='best_model',
        n_code=code_num,
        code_length=code_length,
        dataset_name=dataset_name,
        features_root=features_root
    )


def eval_recall_wrapper(predictions, labels, subset=None):
    """Evaluate recall metrics (wrapper function)."""
    if subset is not None:
        predictions = [predictions[j] for j in subset]
        labels = [labels[j] for j in subset]
    labels = [[x] for x in labels]
    return eval_all(predictions, labels)
