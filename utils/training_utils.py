import os
from copy import deepcopy

import torch


def safe_load(model, file):
    """Load model weights with flexible key matching."""
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    model.load_state_dict(state_dict, strict=False)


def safe_load_embedding(model, file):
    """Load embedding weights with size adaptation."""
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)

    matched_state_dict = deepcopy(model.state_dict())
    for key in model_state_dict_keys:
        if key in state_dict:
            file_size = state_dict[key].size(0)
            model_embedding = matched_state_dict[key].clone()
            model_size = model_embedding.size(0)
            model_embedding[:file_size, :] = state_dict[key][:model_size, :]
            matched_state_dict[key] = model_embedding
            print(f'Copy {key} {matched_state_dict[key].size()} from {state_dict[key].size()}')
    model.load_state_dict(matched_state_dict, strict=False)


def safe_save(accelerator, model, save_path, best_metric, current_metric, last_checkpoint=None):
    """Save model checkpoint if current metric improves."""
    os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()

    # Only save if current metric is better
    if current_metric > best_metric and accelerator.is_local_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        accelerator.save(unwrap_model.state_dict(), f'{save_path}/best_model.pt')
        accelerator.save(unwrap_model.model.state_dict(), f'{save_path}/best_model.pt.model')
        accelerator.save(unwrap_model.centroids.state_dict(), f'{save_path}/best_model.pt.centroids')
        accelerator.save(unwrap_model.code_embedding.state_dict(), f'{save_path}/best_model.pt.embedding')
        # Save complete VideoRQVAE state (encoder, decoder, quantizer)
        accelerator.save(unwrap_model.video_rqvae.state_dict(), f'{save_path}/best_model.pt.videorqvae')
        # Save start token embedding (separate from codebook to avoid collision with code=0)
        accelerator.save(unwrap_model.start_token_embedding, f'{save_path}/best_model.pt.start_token')
        accelerator.print(f'Save best model {save_path}/best_model.pt (metric: {current_metric:.4f})')
        return current_metric, f'{save_path}/best_model.pt', True  # is_new_best=True

    return best_metric, last_checkpoint, False  # is_new_best=False

