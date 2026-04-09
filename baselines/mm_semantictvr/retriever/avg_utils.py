import os
import pickle
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    GenerationConfig,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from typing import Dict, List, Optional
import torch.nn.functional as F
import sys
import json


def write_pkl(obj, filename):
    """Write object to pickle file, creating directories as needed."""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def preprocess_function(source_text, target_text, tokenizer):
    prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
    response_text = f"{prefix_text}{target_text}</s>"

    input = tokenizer(
        response_text,
        return_tensors=None,
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    input_ids = input["input_ids"]

    labels = input_ids.copy()

    output_start_index = (
        len(
            tokenizer.encode(
                prefix_text, max_length=128, truncation=True, padding="max_length"
            )
        )
        - 9
    )
    labels[:output_start_index] = [-100] * output_start_index

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def llama_prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # 13291 is the token id for vokens, so the function can return the correct allowed tokens
        # you may need to change this value in your case
        index = sentence.index(13291)
        sentence = sentence[index:]
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def load_codes(target_file):
    """Load semantic ID codes from various data formats."""
    with open(target_file, "r", encoding="utf-8") as tgt_file:
        data = json.load(tgt_file)

    res = []

    # Detect format
    if isinstance(data, dict):
        # Dict format: old RQVAE or new VideoRQVAE test
        for video_id, tokens in data.items():
            if isinstance(tokens[0], list):
                # New VideoRQVAE test: list of lists
                for token_list in tokens:
                    token_sequence = " ".join(token_list).replace("<", "").replace(">", "")
                    if token_sequence not in res:
                        res.append(token_sequence)
            else:
                # Old RQVAE: single list
                token_sequence = " ".join(tokens).replace("<", "").replace(">", "")
                if token_sequence not in res:
                    res.append(token_sequence)

    elif isinstance(data, list):
        # List format: new VideoRQVAE train
        for item in data:
            if isinstance(item, dict) and 'SemanticID' in item:
                semantic_ids = item['SemanticID']
                token_sequence = " ".join(semantic_ids).replace("<", "").replace(">", "")
                if token_sequence not in res:
                    res.append(token_sequence)

    return res


def load_prompt(source_file, target_file, sub_size=None):
    target_lines = open(target_file, encoding="utf-8").read().splitlines()
    source_lines = open(source_file, encoding="utf-8").read().splitlines()

    if sub_size is not None and sub_size < len(source_lines):
        indeces = np.random.choice(len(source_lines), sub_size, replace=False)
        source_lines = [source_lines[i] for i in indeces]
        target_lines = [target_lines[i] for i in indeces]

    source = []
    target = []
    for i in range(len(source_lines)):
        source_text = source_lines[i]
        target_text = target_lines[i]
        prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
        source.append(prefix_text)
        target.append(target_text)

    return source, target


def load_response(source_file, target_file):
    target_lines = open(target_file, encoding="utf-8").read().splitlines()
    source_lines = open(source_file, encoding="utf-8").read().splitlines()

    res = []
    for i in range(len(source_lines)):
        prefix_text = f"Response:{target_lines[i]}</s>"
        if prefix_text not in res:
            res.append(prefix_text)

    return res


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)





class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, source_file, target_file, subset_size=None):
        self.tokenizer = tokenizer
        self.source_texts = open(
            source_file, encoding="utf-8").read().splitlines()
        self.target_texts = open(
            target_file, encoding="utf-8").read().splitlines()
        self.subset_size = subset_size

        if self.subset_size is not None:
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        return preprocess_function(source_text, target_text, self.tokenizer)


class QueryEvalCallback(TrainerCallback):
    def __init__(
        self,
        local_rank,
        sub_train_dataset,
        test_dataset,
        tgt_file,
        logger,
        batch_size,
        collator,
        tokenizer,
        train_index_file=None,
        wandb=None,
        log_freq=3,
        gen_len=20,
        use_train_addition=False,
        include_train_in_search=False,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.logger = logger
        self.sub_train_dataset = sub_train_dataset
        self.test_dataset = test_dataset
        self.sub_train_dataloader = DataLoader(
            sub_train_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )
        # Configure search pool based on experimental conditions
        self.use_train_addition = use_train_addition
        self.include_train_in_search = include_train_in_search
        
        # Load train codes 
        self.train_code_list = self.sub_train_dataset.target_texts
        train_trie = Trie([[0] + self.tokenizer.encode(x) for x in self.train_code_list])
        self.train_prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(train_trie)

        # Load test codes
        self.test_code_list = load_codes(tgt_file)
        test_trie = Trie([[0] + self.tokenizer.encode(x) for x in self.test_code_list])
        self.test_prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(test_trie)
        
        # Configure combined search pool for Track 4 if needed
        if include_train_in_search and train_index_file:
            train_codes = load_codes(train_index_file)

            # Check the collision of train and test codes
            collision_codes = list(set(self.train_code_list) & set(self.test_code_list))
            print(f"Collision of train and test codes: {collision_codes}")

            # Combine and deduplicate codes
            combined_codes = list(set(self.test_code_list + train_codes))
            combined_trie = Trie([[0] + self.tokenizer.encode(x) for x in combined_codes])
            self.combined_prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(combined_trie)
            print(f"Track 4: Combined search pool size = {len(combined_codes)} (test: {len(self.test_code_list)}, train: {len(train_codes)})")
        else:
            self.combined_prefix_allowed_tokens_fn = None
        
        self.wandb = wandb if wandb else None
        self.log_freq = log_freq
        self.gen_len = gen_len
        self.local_rank = local_rank
        
        # Best model tracking for Recall@1
        self.best_recall_at_1 = 0.0
        self.best_model_path = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Evaluate the model before training begins to establish baseline"""
        if self.local_rank != 0:
            return
        self._perform_evaluation(args, state, control, epoch=0, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.local_rank != 0:
            return
        current_epoch = state.epoch
        if int(current_epoch) % self.log_freq != 0:
            return
        self._perform_evaluation(args, state, control, epoch=current_epoch, **kwargs)

    def _perform_evaluation(self, args, state, control, epoch, **kwargs):
        """Perform evaluation logic for both pre-training and epoch-end evaluation"""
        recall_count_at_1_1 = 0
        recall_count_at_5_1 = 0
        recall_count_at_10_1 = 0
        model = kwargs["model"].eval()

        # Eval on T1 - Only train queries
        for batch_1 in tqdm(self.sub_train_dataloader, desc="Track 1: Train queries (sub-train-only pool)"):
            inputs_1 = batch_1
            with torch.no_grad():
                generation_config = GenerationConfig(
                    num_beams=10,
                    max_new_tokens=self.gen_len,
                    num_return_sequences=10,
                    early_stopping=True,
                    use_cache=True,
                )
                batch_beams_1 = model.generate(
                    inputs_1["input_ids"].to(model.device),
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=self.train_prefix_allowed_tokens_fn,
                ).reshape(inputs_1["input_ids"].shape[0], 10, -1)

                for beams, label in zip(batch_beams_1, inputs_1["labels"]):
                    rank_list = self.tokenizer.batch_decode(
                        beams, skip_special_tokens=True
                    )
                    rank_list = [x.split(" ") for x in rank_list]
                    label[label == -100] = self.tokenizer.pad_token_id
                    label = self.tokenizer.decode(
                        label, skip_special_tokens=True
                    ).strip()
                    label = label.split(" ")
                    hits = [i for i, x in enumerate(rank_list) if x == label]
                    hits = [x for x in hits if x < 10]
                    if len(hits) != 0:
                        recall_count_at_10_1 += 1
                        if hits[0] < 5:
                            recall_count_at_5_1 += 1
                        if hits[0] == 0:
                            recall_count_at_1_1 += 1

        hits_at_1_data_1 = recall_count_at_1_1 / len(self.sub_train_dataset)
        hits_at_5_data_1 = recall_count_at_5_1 / len(self.sub_train_dataset)
        hits_at_10_data_1 = recall_count_at_10_1 / len(self.sub_train_dataset)

        log_msg = f"Epoch {epoch} training set: Recall@1: {hits_at_1_data_1}, Recall@5: {hits_at_5_data_1}, Recall@10: {hits_at_10_data_1}"
        self.logger.info(log_msg)
        if self.wandb:
            self.wandb.log(
                {
                    "T1/Train Recall@1": hits_at_1_data_1,
                    "T1/Train Recall@5": hits_at_5_data_1,
                    "T1/Train Recall@10": hits_at_10_data_1,
                }
            )

        # Eval on T2/T3 - Standard test evaluation with test-only search pool
        recall_count_at_1_standard = 0
        recall_count_at_5_standard = 0
        recall_count_at_10_standard = 0
        
        for batch_2 in tqdm(self.test_dataloader, desc="Track 2/3: Test queries (test-only pool)"):
            inputs_2 = batch_2
            with torch.no_grad():
                generation_config = GenerationConfig(
                    num_beams=10,
                    max_new_tokens=self.gen_len,
                    num_return_sequences=10,
                    early_stopping=True,
                    use_cache=True,
                )
                batch_beams_2 = model.generate(
                    inputs_2["input_ids"].to(model.device),
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=self.test_prefix_allowed_tokens_fn,
                ).reshape(inputs_2["input_ids"].shape[0], 10, -1)

                # Check if batch has multiple target_texts (new VideoRQVAE format)
                has_multi_targets = "target_texts" in inputs_2

                batch_idx = 0
                for beams, label in zip(batch_beams_2, inputs_2["labels"]):
                    rank_list = self.tokenizer.batch_decode(
                        beams, skip_special_tokens=True
                    )
                    rank_list = [x.split(" ") for x in rank_list]

                    if has_multi_targets:
                        valid_targets = [t.split(" ") for t in inputs_2["target_texts"][batch_idx]]
                        hits = []
                        for i, generated in enumerate(rank_list):
                            if any(generated == valid_target for valid_target in valid_targets):
                                hits.append(i)
                                break
                    else:
                        label[label == -100] = self.tokenizer.pad_token_id
                        label = self.tokenizer.decode(
                            label, skip_special_tokens=True
                        ).strip()
                        label = label.split(" ")
                        hits = [i for i, x in enumerate(rank_list) if x == label]

                    hits = [x for x in hits if x < 10]
                    if len(hits) != 0:
                        recall_count_at_10_standard += 1
                        if hits[0] < 5:
                            recall_count_at_5_standard += 1
                        if hits[0] == 0:
                            recall_count_at_1_standard += 1

                    batch_idx += 1

        # Calculate and log Track 2/3 results
        hits_at_1_standard = recall_count_at_1_standard / len(self.test_dataset)
        hits_at_5_standard = recall_count_at_5_standard / len(self.test_dataset)
        hits_at_10_standard = recall_count_at_10_standard / len(self.test_dataset)

        # Determine track and category for logging
        if self.use_train_addition:
            track_category = "T3"
            track_description = "Test queries (train+addition data, test-only pool)"
            metric_prefix = "Train Addition"
        else:
            track_category = "T2"
            track_description = "Test queries (standard training, test-only pool)"
            metric_prefix = "Standard"
        
        log_msg = f"Epoch {epoch} {metric_prefix} test: Recall@1: {hits_at_1_standard}, Recall@5: {hits_at_5_standard}, Recall@10: {hits_at_10_standard}"
        self.logger.info(log_msg)
        if self.wandb:
            self.wandb.log({
                f"{track_category}/Test Recall@1": hits_at_1_standard,
                f"{track_category}/Test Recall@5": hits_at_5_standard,
                f"{track_category}/Test Recall@10": hits_at_10_standard
            })

        # TRACK 4: Combined search pool evaluation (if enabled)
        if self.include_train_in_search and self.combined_prefix_allowed_tokens_fn:
            recall_count_at_1_combined = 0
            recall_count_at_5_combined = 0
            recall_count_at_10_combined = 0
            
            # Check if batch has multiple target_texts (new VideoRQVAE format)
            has_multi_targets_t4 = None

            for batch_2 in tqdm(self.test_dataloader, desc="Track 4: Test queries (combined pool)"):
                inputs_2 = batch_2

                # Detect format once per epoch
                if has_multi_targets_t4 is None:
                    has_multi_targets_t4 = "target_texts" in inputs_2

                with torch.no_grad():
                    generation_config = GenerationConfig(
                        num_beams=10,
                        max_new_tokens=self.gen_len,
                        num_return_sequences=10,
                        early_stopping=True,
                        use_cache=True,
                    )
                    batch_beams_2 = model.generate(
                        inputs_2["input_ids"].to(model.device),
                        generation_config=generation_config,
                        prefix_allowed_tokens_fn=self.combined_prefix_allowed_tokens_fn,
                    ).reshape(inputs_2["input_ids"].shape[0], 10, -1)

                    batch_idx = 0
                    for beams, label in zip(batch_beams_2, inputs_2["labels"]):
                        rank_list = self.tokenizer.batch_decode(
                            beams, skip_special_tokens=True
                        )
                        rank_list = [x.split(" ") for x in rank_list]

                        if has_multi_targets_t4:
                            # New format: check if generated matches ANY valid target
                            valid_targets = [t.split(" ") for t in inputs_2["target_texts"][batch_idx]]
                            hits = []
                            for i, generated in enumerate(rank_list):
                                if any(generated == valid_target for valid_target in valid_targets):
                                    hits.append(i)
                                    break  # Only count first match
                        else:
                            # Old format: single label comparison
                            label[label == -100] = self.tokenizer.pad_token_id
                            label = self.tokenizer.decode(
                                label, skip_special_tokens=True
                            ).strip()
                            label = label.split(" ")
                            hits = [i for i, x in enumerate(rank_list) if x == label]

                        hits = [x for x in hits if x < 10]
                        if len(hits) != 0:
                            recall_count_at_10_combined += 1
                            if hits[0] < 5:
                                recall_count_at_5_combined += 1
                            if hits[0] == 0:
                                recall_count_at_1_combined += 1

                        batch_idx += 1

            # Calculate and log Track 4 results
            hits_at_1_combined = recall_count_at_1_combined / len(self.test_dataset)
            hits_at_5_combined = recall_count_at_5_combined / len(self.test_dataset)
            hits_at_10_combined = recall_count_at_10_combined / len(self.test_dataset)

            log_msg = f"Epoch {epoch} Track 4: Test queries (combined pool): Recall@1: {hits_at_1_combined}, Recall@5: {hits_at_5_combined}, Recall@10: {hits_at_10_combined}\n"
            self.logger.info(log_msg)
            if self.wandb:
                self.wandb.log({
                    "T4/Test Recall@1": hits_at_1_combined,
                    "T4/Test Recall@5": hits_at_5_combined,
                    "T4/Test Recall@10": hits_at_10_combined,
                })

        # Save best model based on Recall@1 from T2/T3 (standard test evaluation)
        if self.local_rank == 0:
            if self.include_train_in_search and self.combined_prefix_allowed_tokens_fn:
                current_recall_at_1 = hits_at_1_combined
            else:
                current_recall_at_1 = hits_at_1_standard
                
            if current_recall_at_1 >= self.best_recall_at_1:
                previous_best = self.best_recall_at_1
                self.best_recall_at_1 = current_recall_at_1
                
                # Save the best model with same structure as regular checkpoints
                best_model_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                
                # Get model and tokenizer
                model = kwargs["model"]
                tokenizer = kwargs.get("tokenizer", self.tokenizer)
                
                # Save model using save_pretrained to match checkpoint structure
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(best_model_dir)
                else:
                    # Handle PEFT models or wrapped models
                    if hasattr(model, 'module'):
                        model.module.save_pretrained(best_model_dir)
                    else:
                        # Last resort: save state dict
                        torch.save(model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
                        # Also save config if available
                        if hasattr(model, 'config'):
                            model.config.save_pretrained(best_model_dir)
                
                # Save tokenizer with all its files 
                if tokenizer and hasattr(tokenizer, 'save_pretrained'):
                    tokenizer.save_pretrained(best_model_dir)
                
                # Store the recall results
                recall_results = {
                    "epoch": epoch,
                    "recall_at_1": current_recall_at_1,
                    "recall_at_5": hits_at_5_standard,
                    "recall_at_10": hits_at_10_standard
                }

                recall_results_dir = os.path.join(best_model_dir, "recall_results")
                os.makedirs(recall_results_dir, exist_ok=True)
                with open(os.path.join(recall_results_dir, "recall_results.json"), "w") as f:
                    json.dump(recall_results, f)

                self.best_model_path = best_model_dir
                log_msg = f"New best model saved at epoch {epoch} with Recall@1: {current_recall_at_1:.4f} (previous best: {previous_best:.4f})"
                self.logger.info(log_msg)
                print(f"*** {log_msg} ***")

    def on_train_end(self, args, state, control, **kwargs):
        """Log the final best model information"""
        if self.local_rank == 0 and self.best_model_path:
            final_msg = f"Training completed. Best model (Recall@1: {self.best_recall_at_1:.4f}) saved at: {self.best_model_path}"
            self.logger.info(final_msg)
            print(f"*** {final_msg} ***")

    def get_best_model_info(self):
        """Return information about the best saved model"""
        return {
            "best_recall_at_1": self.best_recall_at_1,
            "best_model_path": self.best_model_path
        }


class TrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits = logits / self.temperature
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # print(logits.shape, labels.shape)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class LlaMaTrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, vocab_size=32000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits / self.temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss


class LTRTrainer(Trainer):
    def __init__(
        self,
        temperature=1.0,
        ltr_loss_factor=1.0,
        train_allowed_tokens=None,
        margin=1.0,
        seed=42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.train_allowed_tokens = train_allowed_tokens
        self.ltr_loss_factor = ltr_loss_factor
        self.margin = margin
        self.seed = seed

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels") # (seq_length)
        outputs = model(**inputs) # (seq_length, vocab_size)
        logits = outputs.logits

        logits = logits / self.temperature

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        tem_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        ltr_loss = self.multi_ltr_loss(model, inputs, logits, labels)

        loss = self.ltr_loss_factor * ltr_loss + tem_loss

        return (loss, outputs) if return_outputs else loss

    def ltr_loss(self, model, inputs, logits, labels, margin):

        if isinstance(
            model, (torch.nn.DataParallel,
                    torch.nn.parallel.DistributedDataParallel)
        ):
            model = model.module

        num_beams = 10
        generation_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=6,
            num_return_sequences=num_beams,
            early_stopping=True,
            use_cache=False,
        )

        # Set random seed for reproducible beam search
        torch.manual_seed(self.seed)
        
        beams = model.generate(
            inputs.get("input_ids"),
            generation_config=generation_config,
            prefix_allowed_tokens_fn=self.train_allowed_tokens
        ).reshape(inputs.get("input_ids").shape[0], 10, -1)

        # Find the index of the first token that is not a padding token
        first_non_padding_token_index = torch.argmax(labels != self.tokenizer.pad_token_id, dim=1)
        first_non_padding_token_index = first_non_padding_token_index.unsqueeze(1)
        beams = beams[:, :first_non_padding_token_index, :]
        labels = labels[:, :first_non_padding_token_index]
        logits = logits[:, :first_non_padding_token_index, :]

        # positive scores
        positive_logits = logits
        positive_probs = F.log_softmax(positive_logits, dim=-1)
        positive_selected_probs = torch.gather(
            positive_probs, 2, labels[:, :5].unsqueeze(-1)
        ).squeeze(-1)
        positive_scores = positive_selected_probs.sum(dim=1)

        # negative scores
        batch_size, num_beams, seq_length = beams.size()
        negative_indices = torch.empty(
            batch_size, seq_length - 1, dtype=torch.long, device=labels.device
        )

        for i in range(batch_size):
            positive_seq = labels[i, :5]
            filtered_beams = [
                beam[1:] for beam in beams[i] if not torch.equal(beam[1:], positive_seq)
            ]
            negative_seq = random.choice(filtered_beams)
            negative_indices[i] = negative_seq

        # print(negative_indices)
        negative_logits = model(inputs.get("input_ids"),
                                labels=negative_indices).logits
        negative_probs = F.log_softmax(negative_logits, dim=-1)
        negative_selected_probs = torch.gather(
            negative_probs, 2, negative_indices.unsqueeze(-1)
        ).squeeze(-1)
        negative_scores = negative_selected_probs.sum(dim=1)

        losses = F.relu(negative_scores - positive_scores + margin)
        loss = losses.mean()

        return loss

    def prefix_ltr_loss(self, model, inputs, logits, labels, temperature, margin):

        if isinstance(
            model, (torch.nn.DataParallel,
                    torch.nn.parallel.DistributedDataParallel)
        ):
            model = model.module

        prefix_loss = []
        for prefix_length in range(1, 5):
            num_beams = 10
            generation_config = GenerationConfig(
                num_beams=num_beams,
                max_new_tokens=prefix_length,
                num_return_sequences=num_beams,
                early_stopping=True,
                use_cache=False,
            )
            beams = model.generate(
                inputs.get("input_ids"),
                generation_config=generation_config,
                prefix_allowed_tokens_fn=self.train_allowed_tokens
            ).reshape(inputs.get("input_ids").shape[0], 10, -1)

            # positive scores
            positive_logits = logits[:, :prefix_length, :]
            positive_probs = F.log_softmax(positive_logits, dim=-1)
            positive_selected_probs = torch.gather(
                positive_probs, 2, labels[:, :prefix_length].unsqueeze(-1)
            ).squeeze(-1)
            positive_scores = positive_selected_probs.sum(dim=1)

            # print(prefix_length)
            # print(labels[0,:prefix_length])
            # print(positive_scores[0])
            # print(beams[0])
            # negative scores
            batch_size, num_beams, seq_length = beams.size()
            negative_indices = torch.empty(
                batch_size, seq_length - 1, dtype=torch.long, device=labels.device
            )

            for i in range(batch_size):
                positive_seq = labels[i, :prefix_length]
                filtered_beams = [
                    beam[1:]
                    for beam in beams[i]
                    if not torch.equal(beam[1:], positive_seq)
                ]
                negative_seq = random.choice(filtered_beams)
                negative_indices[i] = negative_seq

            # print(negative_indices[0])
            negative_logits = (
                model(inputs.get("input_ids"), labels=negative_indices).logits
                / temperature
            )
            negative_probs = F.log_softmax(negative_logits, dim=-1)
            negative_selected_probs = torch.gather(
                negative_probs, 2, negative_indices.unsqueeze(-1)
            ).squeeze(-1)
            negative_scores = negative_selected_probs.sum(dim=1)
            # print(negative_scores[0])

            losses = F.relu(negative_scores - positive_scores + margin)
            loss = losses.mean()
            prefix_loss.append(loss)

        return sum(prefix_loss) / len(prefix_loss)

    def multi_ltr_loss(self, model, inputs, logits, labels):

        if isinstance(
            model, (torch.nn.DataParallel,
                    torch.nn.parallel.DistributedDataParallel)
        ):
            model = model.module

        num_beams = 10
        generation_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=6,
            num_return_sequences=num_beams,
            early_stopping=True,
            use_cache=False,
        )
        beams = model.generate(
            inputs.get("input_ids"),
            generation_config=generation_config,
            prefix_allowed_tokens_fn=self.train_allowed_tokens
        ).reshape(inputs.get("input_ids").shape[0], num_beams, -1)

        # Filter out padding tokens from labels for positive score calculation
        batch_size, seq_len = labels.shape
        vocab_size = logits.size(-1)
        
        # Create mask for valid (non-padding) tokens
        valid_mask = (labels != -100)
        
        # Clone labels and replace -100 with 0 to avoid index errors
        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0
        
        # positive scores
        positive_logits = logits
        positive_probs = F.log_softmax(positive_logits, dim=-1)
        positive_selected_probs = torch.gather(
            positive_probs, 2, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding positions - set their probabilities to 0
        positive_selected_probs = positive_selected_probs * valid_mask.float()
        positive_scores = positive_selected_probs.sum(dim=1)

        # negative scores
        batch_size, num_beams, seq_length = beams.size()

        losses = []
        for i in range(batch_size):
            positive_seq = safe_labels[i]  # Use safe_labels instead of original labels
            filtered_beams = [
                beam[1:] for beam in beams[i] if not torch.equal(beam[1:], positive_seq)
            ]
            
            # Skip if no negative examples available
            if len(filtered_beams) == 0:
                continue
                
            input_i = (
                inputs.get("input_ids")[i].unsqueeze(
                    0).repeat(len(filtered_beams), 1)
            )
            filtered_beams = torch.stack(filtered_beams, dim=0)
            
            # Validate negative beam indices
            if torch.any(filtered_beams >= vocab_size) or torch.any(filtered_beams < 0):
                print(f"Warning: Invalid token IDs in filtered beams for sample {i}, skipping")
                continue
            
            negative_logits = model(input_i, labels=filtered_beams).logits
            negative_probs = F.log_softmax(negative_logits, dim=-1)
            negative_selected_probs = torch.gather(
                negative_probs, 2, filtered_beams.unsqueeze(-1)
            ).squeeze(-1)
            negative_scores = negative_selected_probs.sum(dim=1)
            # print(negative_scores)
            total_scores = torch.cat(
                (positive_scores[i].unsqueeze(0), negative_scores), dim=0
            )
            total_prob = F.softmax(total_scores, dim=0)
            target = torch.zeros_like(total_scores)
            target[0] = 1
            losses.append(F.cross_entropy(total_prob, target))

        # Handle case where no valid losses were computed
        if len(losses) == 0:
            # Return a zero loss if no valid negative examples
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        loss = torch.stack(losses).mean()

        return loss

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_video_id(video_id: str) -> str:
    # Strip common video file extensions
    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
        if video_id.endswith(ext):
            video_id = video_id[:-len(ext)]
            break
    
    return video_id


def build_sid_to_video_mapping(index_file: str, index_type: str) -> Dict[str, List[str]]:
    """
    Build a mapping from semantic ID string to list of video IDs.

    Args:
        index_file: Path to the index JSON file
        index_type: 'standard', 'text_guided', or 'videorqvae'

    Returns:
        Dict mapping sID string (space-separated tokens) -> list of video_ids

    Format handling:
        - videorqvae: video_id -> [[tokens], [tokens], ...] (list of lists per video)
        - standard/text_guided: video_id -> [tokens] (single list per video)
    """
    from collections import defaultdict

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sid_to_videos = defaultdict(list)

    if isinstance(data, dict):
        # Dict format: {video_id: tokens}
        for video_id, tokens in data.items():
            # Clean video ID: remove caption suffix and file extension
            base_video_id = clean_video_id(video_id)

            if index_type == 'videorqvae' and tokens and isinstance(tokens[0], list):
                # VideoRQVAE format: list of lists (multiple sIDs per video)
                for token_list in tokens:
                    sid_str = " ".join(token_list).replace("<", "").replace(">", "")
                    if base_video_id not in sid_to_videos[sid_str]:
                        sid_to_videos[sid_str].append(base_video_id)
            else:
                # Standard/text_guided format: single list
                sid_str = " ".join(tokens).replace("<", "").replace(">", "")
                if base_video_id not in sid_to_videos[sid_str]:
                    sid_to_videos[sid_str].append(base_video_id)

    elif isinstance(data, list):
        # List format: VideoRQVAE train format with SemanticID field
        for item in data:
            if isinstance(item, dict):
                video_id = item.get('video_id', item.get('video', ''))
                # Clean video ID: remove caption suffix and file extension
                base_video_id = clean_video_id(video_id)

                if 'SemanticID' in item:
                    semantic_ids = item['SemanticID']
                    sid_str = " ".join(semantic_ids).replace("<", "").replace(">", "")
                    if base_video_id not in sid_to_videos[sid_str]:
                        sid_to_videos[sid_str].append(base_video_id)

    return dict(sid_to_videos)


def compute_detailed_metrics(results: List[Dict], predictions: List[List[str]],
                             labels: List[List[str]], total_time: float,
                             num_queries: int, num_candidates: int) -> Dict:
    """
    Compute detailed retrieval metrics based on video ID matching.

    Args:
        results: Per-query result dicts with candidates (video IDs)
        predictions: List of sID predictions per query (not used for recall calculation)
        labels: List of ground-truth sIDs per query (not used for recall calculation)
        total_time: Total generation time in seconds
        num_queries: Total number of queries
        num_candidates: Number of candidates per query

    Returns:
        Dict with R@1, R@5, R@10, R@K, timing stats, etc.

    Note:
        Recall is computed based on whether the ground truth VIDEO ID appears
        in the candidate VIDEO IDs, NOT based on semantic ID token matching.
    """
    # Count correct retrievals at each K by checking video ID matching
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    correct_at_k = 0

    for result in results:
        gt_video = result.get("ground_truth_video_id", "")
        candidates = result.get("candidates", [])

        if candidates and isinstance(candidates[0], (list, tuple)):
            candidates = [
                item[1] for item in candidates
                if isinstance(item, (list, tuple)) and len(item) > 1
            ]

        if gt_video in candidates[:1]:
            correct_at_1 += 1
        if gt_video in candidates[:5]:
            correct_at_5 += 1
        if gt_video in candidates[:10]:
            correct_at_10 += 1
        if gt_video in candidates[:num_candidates]:
            correct_at_k += 1

    # Compute recall metrics as percentages based on video ID matching
    metrics = {
        "Recall@1": (correct_at_1 / num_queries * 100) if num_queries > 0 else 0,
        "Recall@5": (correct_at_5 / num_queries * 100) if num_queries > 0 else 0,
        "Recall@10": (correct_at_10 / num_queries * 100) if num_queries > 0 else 0,
        f"Recall@{num_candidates}": (correct_at_k / num_queries * 100) if num_queries > 0 else 0,
        "seconds_per_query": total_time / num_queries if num_queries > 0 else 0,
        "total_queries": num_queries,
        "batch_size": 1,  # True per-query latency measurement
        "correct_retrievals_at_1": correct_at_1,
        "correct_retrievals_at_5": correct_at_5,
        "correct_retrievals_at_10": correct_at_10,
        "correct_retrievals": correct_at_k,
    }

    # Average candidates per query
    avg_candidates = sum(r.get('num_candidates', 0) for r in results) / len(results) if results else 0
    metrics['avg_candidates_per_query'] = round(avg_candidates, 2)

    return metrics


def save_candidates_json(results: List[Dict], metrics: Dict, config: Dict,
                         output_dir: str) -> str:
    """
    Save candidates to JSON file.

    Args:
        results: Per-query result dicts
        metrics: Computed metrics dict
        config: Configuration dict with metadata
        output_dir: Output directory path

    Returns:
        Path to saved JSON file
    """
    import time

    # Extract configuration parameters
    dataset = config.get('dataset', 'unknown')
    model_name = config.get('model_name', 'unknown')
    num_candidates = config.get('num_candidates', 20)
    index_type = config.get('index_type', 'unknown')
    code_book_size = config.get('code_book_size', 0)
    code_book_num = config.get('code_book_num', 0)
    setting = config.get('setting', 1)
    checkpoint = config.get('eval_checkpoint', 'unknown')
    timestamp = time.strftime('%m%d%H%M')

    # Create metadata
    metadata = {
        "dataset": dataset,
        "model_name": model_name.split('/')[-1] if '/' in model_name else model_name,
        "num_candidates": num_candidates,
        "index_type": index_type,
        "code_book_size": code_book_size,
        "code_book_num": code_book_num,
        "setting": setting,
        "checkpoint": checkpoint,
        "timestamp": timestamp
    }

    # Build output structure
    output_data = {
        "metadata": metadata,
        "metrics": metrics,
        "results": results
    }

    # Create filename: {dataset}_{index_type}_c{size}l{num}_{K}_candidates_t{setting}.json
    filename = f"{dataset}_{index_type}_c{code_book_size}l{code_book_num}_{num_candidates}_candidates_t{setting}.json"

    # Create output directory (./candidates/{index_type}/)
    full_output_dir = os.path.join(output_dir, index_type)
    os.makedirs(full_output_dir, exist_ok=True)

    # Save JSON file
    filepath = os.path.join(full_output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Candidates JSON saved to: {filepath}")
    print(f"{'='*80}\n")

    return filepath


def run_evaluation(train_args, tokenizer, model, test_dataset, test_index_file: str,
                   train_index_file: str, data_collator, detailed_generation: bool = False,
                   codebook_embedding: Optional[torch.Tensor] = None,
                   code_book_size: int = 256,
                   code_book_num: int = 4,
                   candidate_output_dir: str = "./candidates") -> None:
    """
    Run evaluation to generate video candidates.

    Args:
        train_args: Parsed command-line arguments
        tokenizer: T5 tokenizer
        model: T5 model (will be loaded from checkpoint)
        test_dataset: Test dataset instance
        test_index_file: Path to test semantic ID index file
        train_index_file: Path to train semantic ID index file (for setting=2)
        data_collator: Data collator for batching
        detailed_generation: If True, save detailed pkl files with embeddings and decoder hidden states
        codebook_embedding: Loaded codebook embeddings tensor [code_book_num * code_book_size, e_dim]
        code_book_size: Size of each codebook layer
        code_book_num: Number of codebook layers (RQ layers)
    """
    import time

    print(f"\n{'='*80}")
    print("EVALUATION MODE: Generating Video Candidates")
    print(f"Setting: {train_args.setting} ({'train+test combined pool' if train_args.setting == 2 else 'test only pool'})")
    print(f"{'='*80}\n")

    # Validate checkpoint path
    if train_args.eval_checkpoint is None:
        raise ValueError("--eval_checkpoint is required in evaluation mode")

    if not os.path.exists(train_args.eval_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {train_args.eval_checkpoint}")

    # Load checkpoint weights
    checkpoint_path = train_args.eval_checkpoint
    model_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(model_bin):
        print(f"Loading model from: {model_bin}")
        state_dict = torch.load(model_bin, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        # Try loading as a directory with model files
        print(f"Loading model from directory: {checkpoint_path}")
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

    model = model.cuda()
    model.eval()

    # Build sID-to-video mapping
    print("Building sID-to-video mapping...")
    sid_to_video = build_sid_to_video_mapping(test_index_file, train_args.index_type)
    print(f"Test pool: {len(sid_to_video)} unique sIDs")

    if train_args.setting == 2:
        # Setting 2: Combined train+test pool
        print("Adding train pool for setting 2...")
        train_sid_to_video = build_sid_to_video_mapping(train_index_file, train_args.index_type)
        print(f"Train pool: {len(train_sid_to_video)} unique sIDs")

        # Merge mappings
        for sid, videos in train_sid_to_video.items():
            if sid in sid_to_video:
                for vid in videos:
                    if vid not in sid_to_video[sid]:
                        sid_to_video[sid].append(vid)
            else:
                sid_to_video[sid] = videos
        print(f"Combined pool: {len(sid_to_video)} unique sIDs")

    # Extract codebook embeddings for videos when detailed_generation=True
    codebook_emb_dict = {}  # video_id -> [emb per latent token]
    query_decoder_hidden_dict = {}  # video_id -> first generated token hidden state

    if detailed_generation and codebook_embedding is not None:
        print("Extracting codebook embeddings for videos...")
        # codebook_embedding shape: [code_book_num * code_book_size, e_dim]
        # Reshape to [code_book_num, code_book_size, e_dim]
        e_dim = codebook_embedding.shape[1]
        codebook_reshaped = codebook_embedding.view(code_book_num, code_book_size, e_dim)
        first_layer_codebook = codebook_reshaped[0]  # [code_book_size, e_dim]

        # Load raw index data to get video_id -> sID mapping
        with open(test_index_file, 'r') as f:
            test_index_data = json.load(f)

        # Extract embeddings for each video in test index
        for video_id, semantic_ids in test_index_data.items():
            first_layer_indices = []
            # semantic_ids format varies - handle both cases
            if isinstance(semantic_ids, list) and semantic_ids:
                if isinstance(semantic_ids[0], list):
                    # Multiple sIDs per video [[t1, t2, ...], [t1', t2', ...]]
                    for sid_list in semantic_ids:
                        for token in sid_list:
                            if isinstance(token, str) and token.startswith('A_'):
                                parts = token.split('_', 1)
                                if len(parts) == 2 and parts[1].isdigit():
                                    first_layer_indices.append(int(parts[1]))
                                break
                else:
                    # Single sID [t1, t2, ...]
                    for token in semantic_ids:
                        if isinstance(token, str) and token.startswith('A_'):
                            parts = token.split('_', 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                first_layer_indices.append(int(parts[1]))

            if first_layer_indices:
                indices_tensor = torch.tensor(first_layer_indices, device=codebook_embedding.device)
                embs = first_layer_codebook[indices_tensor].cpu().tolist()
                codebook_emb_dict[video_id] = embs

        print(f"Extracted codebook embeddings for {len(codebook_emb_dict)} videos")

    # Build Trie for constrained generation
    print("Building Trie for constrained generation...")
    test_codes = load_codes(test_index_file)
    if train_args.setting == 2:
        train_codes = load_codes(train_index_file)
        all_codes = list(set(test_codes + train_codes))
    else:
        all_codes = test_codes

    trie_sequences = [[0] + tokenizer.encode(code) for code in all_codes]
    candidate_trie = Trie(trie_sequences)
    allowed_tokens_fn = prefix_allowed_tokens_fn(candidate_trie)
    print(f"Trie built with {len(all_codes)} unique code sequences")

    # Create test dataloader
    # Use batch_size=1 for true per-query latency measurement
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
        num_workers=10,
    )

    # Generate candidates
    print(f"\nGenerating candidates (num_candidates={train_args.num_candidates})...")
    results = []
    predictions = []
    labels = []

    # Start timing (only measure model.generate, exclude post-processing)
    start_time = time.time()

    generation_config = GenerationConfig(
        num_beams=train_args.num_candidates,
        max_new_tokens=train_args.gen_len,
        num_return_sequences=train_args.num_candidates,
        early_stopping=True,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    # Store generation outputs for post-processing
    all_sequences = []
    all_beam_scores = []
    all_sample_indices = []

    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating candidates")):
        with torch.no_grad():
            input_ids = batch["input_ids"].cuda()
            batch_size = input_ids.shape[0]

            gen_output = model.generate(
                input_ids,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=allowed_tokens_fn
            )

            # Reshape generated sequences: [batch*num_candidates, seq_len] -> [batch, num_candidates, seq_len]
            sequences = gen_output.sequences
            sequences = sequences.reshape(batch_size, train_args.num_candidates, -1)

            # Get beam scores
            if hasattr(gen_output, 'sequences_scores') and gen_output.sequences_scores is not None:
                beam_scores = gen_output.sequences_scores.reshape(batch_size, train_args.num_candidates).cpu().tolist()
            else:
                beam_scores = [[0.0] * train_args.num_candidates for _ in range(batch_size)]

            first_token_hidden = None
            if detailed_generation:
                attention_mask = batch["attention_mask"].cuda()
                encoder_outputs = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                best_sequences = sequences[:, 0, :]  # top-1 beam
                if best_sequences.size(1) > 1:
                    decoder_input_ids = best_sequences[:, :2]
                    decoder_outputs = model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=attention_mask,
                        return_dict=True
                    )
                    # Skip start token and keep the first generated token feature
                    first_token_hidden = decoder_outputs.last_hidden_state[:, 1, :]

            # Store generation results (minimal processing)
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= len(test_dataset):
                    break
                
                all_sequences.append(sequences[i].cpu())
                all_beam_scores.append(beam_scores[i])
                all_sample_indices.append(sample_idx)
                
                if detailed_generation and first_token_hidden is not None:
                    raw_sample = test_dataset.samples[sample_idx]
                    gt_video_id = raw_sample.get('video_id', '')
                    query_decoder_hidden_dict[gt_video_id] = first_token_hidden[i].cpu().tolist()

    end_time = time.time()
    total_time = end_time - start_time

    # POST-PROCESSING (outside timing window)
    print("\nPost-processing results...")
    for idx, (generated_codes, beam_scores_list, sample_idx) in enumerate(zip(all_sequences, all_beam_scores, all_sample_indices)):
        # Get query info from dataset
        raw_sample = test_dataset.samples[sample_idx]
        query_text = raw_sample.get('caption', '')
        gt_video_id = raw_sample.get('video_id', '')

        # Clean ground truth video ID
        cleaned_gt_video_id = clean_video_id(gt_video_id)

        # Get ground truth sIDs from index_data
        gt_sids = []
        if hasattr(test_dataset, 'index_data') and gt_video_id in test_dataset.index_data:
            semantic_id_data = test_dataset.index_data[gt_video_id]
            if semantic_id_data and isinstance(semantic_id_data[0], list):
                # VideoRQVAE format: list of lists
                for sid_list in semantic_id_data:
                    sid_str = " ".join(sid_list).replace("<", "").replace(">", "")
                    gt_sids.append(sid_str)
            else:
                # Standard format: single list
                sid_str = " ".join(semantic_id_data).replace("<", "").replace(">", "")
                gt_sids.append(sid_str)
        labels.append([sid.split() for sid in gt_sids] if gt_sids else [[]])

        # Decode generated sequences
        decoded_sids = tokenizer.batch_decode(generated_codes, skip_special_tokens=True)
        decoded_sids = [sid.strip() for sid in decoded_sids]

        # Track predictions for evaluation
        pred_tokens = [sid.split() for sid in decoded_sids]
        predictions.append(pred_tokens)

        # Map sIDs to video candidates
        seen_videos = set()
        ranked_videos = []
        ranked_videos_with_sid = []  # (sID, video_id) pairs for detailed mode
        ranked_scores = []

        for sid, score in zip(decoded_sids, beam_scores_list):
            if sid in sid_to_video:
                for video_id in sid_to_video[sid]:
                    if video_id not in seen_videos:
                        seen_videos.add(video_id)
                        ranked_videos.append(video_id)
                        ranked_videos_with_sid.append([sid, video_id])
                        ranked_scores.append(score)

        # Store result
        result = {
            "query_text": query_text,
            "ground_truth_video_id": cleaned_gt_video_id,
        }

        # Add ground_truth_sID in detailed mode
        if detailed_generation:
            result["ground_truth_sID"] = gt_sids

        # Add candidates (detailed or standard format)
        if detailed_generation:
            result["candidates"] = ranked_videos_with_sid  # [[sID, video_id], ...]
        else:
            result["candidates"] = ranked_videos  # [video_id, ...]

        result["scores"] = ranked_scores
        result["num_candidates"] = len(ranked_videos)
        results.append(result)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_detailed_metrics(
        results, predictions, labels,
        total_time, len(test_dataset), train_args.num_candidates
    )

    # Print metrics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Recall@1:  {metrics['Recall@1']:.2f}%")
    print(f"Recall@5:  {metrics['Recall@5']:.2f}%")
    print(f"Recall@10: {metrics['Recall@10']:.2f}%")
    print(f"Recall@{train_args.num_candidates}: {metrics.get(f'Recall@{train_args.num_candidates}', metrics['Recall@10']):.2f}%")
    print(f"Seconds/query: {metrics['seconds_per_query']:.4f}")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"{'='*60}\n")

    # Save results
    config = {
        'dataset': train_args.dataset,
        'model_name': train_args.model_name,
        'num_candidates': train_args.num_candidates,
        'index_type': train_args.index_type,
        'code_book_size': train_args.code_book_size,
        'code_book_num': train_args.code_book_num,
        'setting': train_args.setting,
        'eval_checkpoint': train_args.eval_checkpoint
    }

    save_candidates_json(results, metrics, config, candidate_output_dir)

    if detailed_generation:
        base_name = (
            f"{train_args.dataset}_{train_args.index_type}_c{train_args.code_book_size}"
            f"l{train_args.code_book_num}_{train_args.num_candidates}_candidates_t{train_args.setting}"
        )
        detailed_dir = os.path.join(candidate_output_dir, train_args.index_type)
        os.makedirs(detailed_dir, exist_ok=True)

        codebook_emb_path = os.path.join(detailed_dir, f"{base_name}_codebook_emb.pkl")
        write_pkl(codebook_emb_dict, codebook_emb_path)
        print(f"Saved codebook embeddings to {codebook_emb_path}")

        decoder_hidden_path = os.path.join(detailed_dir, f"{base_name}_decoder_hidden.pkl")
        write_pkl(query_decoder_hidden_dict, decoder_hidden_path)
        print(f"Saved decoder hidden states to {decoder_hidden_path}")
