#!/usr/bin/env python
"""Smoke-test for the 6 multiview-fix-pipeline flags.

Validates without invoking the full panda data pipeline:
  (1) VectorQuantizer.init_emb runs the per-slot k-means branch
  (2) GRDR.forward(return_all_slots=True) produces [B, N, ...] outputs with
      per-slot discrete_codes / probability / code_logits
  (3) OurTrainer.train_step with multiview_all_slot_ce=True returns finite
      ce_loss / code_loss / cl_dd_loss + soft routing + orthogonality reg
  (4) build_per_video_loader yields one row per unique video_id
  (5) build_loss_weights threads the multi-view flags
  (6) do_epoch_encode(codebook_seed_all_slots=True) flattens
      [N_vid, N_slots, D] -> [N_vid * N_slots, D] before kmeans (K1 contract)
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('DEVICE', '0'))

from transformers import T5Config, AutoTokenizer
from models.t5 import T5ForConditionalGeneration
from models.grdr import GRDR
from utils.model_utils import create_videorqvae
from trainer.trainer import OurTrainer


def _check_finite(name, t):
    """Assert tensor is finite and not NaN."""
    if isinstance(t, (int, float)):
        assert torch.isfinite(torch.tensor(float(t))).item(), f'{name} is non-finite: {t}'
    else:
        assert torch.isfinite(t).all().item(), f'{name} contains non-finite values'


def main():
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        print('CUDA not available, smoke aborted')
        sys.exit(2)
    device = torch.device('cuda:0')

    # ---- (1) per_slot_init -------------------------------------------------
    print('[1] VectorQuantizer.init_emb with per_slot_init=True ...')
    code_num, num_latent_tokens, code_length = 64, 4, 1  # 64 % 4 = 0
    videorqvae = create_videorqvae(
        code_num=code_num, code_length=code_length,
        num_latent_tokens=num_latent_tokens, e_dim=512, in_dim=512,
        device=device, per_slot_init=True,
    )
    videorqvae.train()
    assert videorqvae.rq.vq_layers[0].per_slot_init is True
    assert videorqvae.rq.vq_layers[0].num_latent_tokens == num_latent_tokens

    # Synthetic features that will reach init_emb on the first VQ forward call.
    B = 8
    x = torch.randn(B, num_latent_tokens, 512, device=device)
    x_q, _, indices, _, _ = videorqvae.rq(x, use_sk=False, return_probs=True)
    assert videorqvae.rq.vq_layers[0].initted, 'init_emb did not run'
    print(f'    OK  codebook shape={videorqvae.rq.vq_layers[0].embedding.weight.shape}')

    # ---- (2) GRDR.forward(return_all_slots=True) ---------------------------
    print('[2] GRDR.forward(return_all_slots=True) ...')
    t5_config = T5Config.from_pretrained('t5-small')
    t5 = T5ForConditionalGeneration.from_pretrained('t5-small', config=t5_config)
    model = GRDR(model=t5, code_length=code_length, use_constraint=False,
                 sk_epsilon=1, zero_inp=False, code_number=code_num,
                 videorqvae=videorqvae).to(device)
    model.train()

    video_features = torch.randn(B, 512, device=device)
    token_idx = torch.randint(0, num_latent_tokens, (B,), device=device)
    out = model(video_features=video_features, token_idx=token_idx, return_all_slots=True)
    assert out.continuous_embeds.shape == (B, num_latent_tokens, 512), out.continuous_embeds.shape
    assert out.discrete_codes.shape == (B, num_latent_tokens), out.discrete_codes.shape
    assert out.probability.shape == (B, num_latent_tokens, code_num), out.probability.shape
    # code_length=1 means n_pred=L-1=0, so code_logits is None
    assert out.code_logits is None, f'code_logits should be None at code_length=1, got {out.code_logits}'
    print(f'    OK  continuous_embeds={tuple(out.continuous_embeds.shape)} discrete_codes={tuple(out.discrete_codes.shape)}')

    # ---- (3) train_step with all_slot_ce -----------------------------------
    print('[3] OurTrainer.train_step with multiview_all_slot_ce ...')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    caps = ['a small dog runs in the field', 'a person walks', 'cars on the road',
            'birds fly', 'water flowing', 'lights at night', 'a cup of tea', 'mountains and clouds']
    enc = tokenizer(caps, padding=True, truncation=True, max_length=32, return_tensors='pt')
    batch = {
        'caption_tokens': enc['input_ids'].to(device),
        'attention_mask': enc['attention_mask'].to(device),
        'video_features': video_features,
        'token_idx': token_idx,
        'video_ids': [f'vid{i}_0' for i in range(B)],
        'ids': torch.zeros(B, code_length + 1, dtype=torch.long, device=device),
        'aux_ids': torch.zeros(B, code_length, dtype=torch.long, device=device),
    }
    loss_weights = {
        'cl_loss': 0.2, 'ce_loss': 0.5, 'code_loss': 0.8, 'cl_dd_loss': 0.1, 'rq_loss': 0.3,
        'multiview_all_slot_ce': True,
        'view_div_high_weight': 0.7, 'view_div_low_weight': 0.1,
        'slot_orthogonality_weight': 0.1,
        'route_agree_loss': 0, 'bucket_route_loss': 0,
        'video_rank_loss': 0, 'expanded_size_loss': 0,
    }
    losses = OurTrainer.train_step(model, batch, current_layer=0, loss_weights=loss_weights, gathered=False)
    for k in ('cl_loss', 'ce_loss', 'code_loss', 'cl_dd_loss', 'rq_loss'):
        v = losses[k]
        _check_finite(k, v)
        val = v.item() if hasattr(v, 'item') else v
        print(f'    {k}={val:.4f}')
    total = sum([losses[k] * loss_weights.get(k, 0) for k in losses if loss_weights.get(k, 0) != 0])
    _check_finite('total', total)
    total.backward()
    print(f'    OK  total_loss={total.item():.4f} backward-pass succeeded')

    # ---- (4) build_per_video_loader ----------------------------------------
    print('[4] build_per_video_loader ...')

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self):
            # 5 unique videos, 3 captions each (text suffixes _0, _1, _2)
            self.samples = [
                {'video_id': f'v{i}_{j}', 'caption': f'cap{i}_{j}'}
                for i in range(5) for j in range(3)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {
                'video_features': torch.randn(512),
                'caption_tokens': torch.zeros(8, dtype=torch.long),
                'attention_mask': torch.ones(8, dtype=torch.long),
                'token_idx': torch.tensor(0, dtype=torch.long),
                'video_id': self.samples[idx]['video_id'],
                'ids': torch.zeros(code_length + 1, dtype=torch.long),
                'aux_ids': torch.zeros(code_length, dtype=torch.long),
            }

    from trainer.evaluator import build_per_video_loader
    fake = FakeDataset()
    loader = build_per_video_loader(fake, batch_size=2, tokenizer=tokenizer)
    assert len(loader.sampler) == 5, f'expected 5 unique videos, got {len(loader.sampler)}'
    print(f'    OK  per-video loader emits {len(loader.sampler)} unique videos')

    # ---- (5) all-slot flags propagate through build_loss_weights -----------
    print('[5] build_loss_weights propagation ...')
    from trainer.trainer import build_loss_weights
    config = {
        'w2_cl_loss': 0.2, 'w2_ce_loss': 0.5, 'w2_code_loss': 0.8, 'w2_cl_dd_loss': 0.1, 'w2_rq_loss': 0.3,
        'multiview_all_slot_ce': True, 'view_div_high_weight': 0.7, 'view_div_low_weight': 0.1,
        'slot_orthogonality_weight': 0.1,
    }
    lw = build_loss_weights(config, 2)
    assert lw['multiview_all_slot_ce'] is True
    assert lw['view_div_high_weight'] == 0.7
    assert lw['slot_orthogonality_weight'] == 0.1
    print('    OK  loss_weights threaded')

    # ---- (6) do_epoch_encode K1 branch reshapes [N_vid,N_slots,D]->flat ----
    print('[6] do_epoch_encode(codebook_seed_all_slots=True) flatten contract ...')
    import numpy as np
    from trainer import evaluator as ev_mod

    N_vid_synth, N_slots_synth, D_synth = 6, 4, 16
    video_keys_synth = [f'vid{i}_0' for i in range(N_vid_synth)]
    fake_video_embs = np.random.randn(N_vid_synth, N_slots_synth, D_synth).astype(np.float32)
    fake_code_dict = {k: [0] * N_slots_synth for k in video_keys_synth}

    def fake_encode_dual(*a, **kw):
        return (fake_video_embs, fake_code_dict, video_keys_synth)

    seen_kmeans_inputs = []
    def fake_kmeans(x, ncentroids, niter=100, seed=42):
        seen_kmeans_inputs.append(tuple(np.asarray(x).shape))
        centers = np.random.randn(ncentroids, np.asarray(x).shape[1]).astype(np.float32)
        return centers, [i % ncentroids for i in range(np.asarray(x).shape[0])]

    orig_encode = ev_mod.our_encode_dual
    orig_kmeans = ev_mod.kmeans
    orig_loader = ev_mod.build_per_video_loader
    orig_write_pkl = None

    # Stub build_per_video_loader so do_epoch_encode does not touch the real dataset.
    class StubSampler:
        def __init__(self, n): self._n = n
        def __len__(self): return self._n
    class StubLoader:
        def __init__(self, n): self.sampler = StubSampler(n)
    ev_mod.our_encode_dual = fake_encode_dual
    ev_mod.kmeans = fake_kmeans
    ev_mod.build_per_video_loader = lambda *a, **kw: StubLoader(N_vid_synth)

    # write_pkl + json.dump should not write real files; redirect save_path to a temp dir.
    save_dir = tempfile.mkdtemp(prefix='smoke_k1_')
    epoch_tag = 1
    n_code = 8
    try:
        ev_mod.do_epoch_encode(
            model=None,  # unused in stubbed K1 branch
            train_dataset=None,
            video_codes={k: [0] for k in video_keys_synth},
            tokenizer=None,
            batch_size=2,
            save_path=save_dir,
            epoch=epoch_tag,
            n_code=n_code,
            code_length=1,
            codebook_seed_all_slots=True,
        )
    finally:
        ev_mod.our_encode_dual = orig_encode
        ev_mod.kmeans = orig_kmeans
        ev_mod.build_per_video_loader = orig_loader

    assert len(seen_kmeans_inputs) == 1, f'expected exactly one kmeans call, saw {seen_kmeans_inputs}'
    saw_shape = seen_kmeans_inputs[0]
    expected_rows = N_vid_synth * N_slots_synth
    assert saw_shape == (expected_rows, D_synth), \
        f'expected kmeans input shape ({expected_rows},{D_synth}), got {saw_shape}'
    print(f'    OK  kmeans saw shape={saw_shape} (= [N_vid*N_slots, D])')

    print('\nALL SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
