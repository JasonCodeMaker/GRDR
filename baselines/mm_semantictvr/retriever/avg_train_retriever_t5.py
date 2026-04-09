import os
from torch.utils.data import Dataset
from models.t5 import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import logging
import wandb
import argparse
from pathlib import Path
from .avg_utils import QueryEvalCallback, TrainerwithTemperature, seed_everything
from .data.dataset import T5Dataset, CombinedT5Dataset, T5TestDataset, DataCollator
from .paths import build_retriever_paths
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model with RQ-VAE codes")

    # Dataset Parameters
    parser.add_argument('--dataset', type=str, default='msrvtt', choices=['msrvtt', 'didemo', 'actnet', 'lsmdc'], help='dataset name')
    parser.add_argument('--mode', type=str, default='none', choices=['joint', 'separate', 'nn', 'none'], help='mode')
    parser.add_argument('--index_type', type=str, default='text_guided', choices=['standard', 'text_guided', 'videorqvae'], help='index file type')
    parser.add_argument('--code_book_size', type=int, default=256, help='code book size')
    parser.add_argument('--code_book_num', type=int, default=4, help='number of code books')
    parser.add_argument('--e_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--version', type=str, default='2.0', help='version of the model')

    # Output and Path Parameters
    parser.add_argument('--output_dir', type=str, default='output/baseline', help='output directory root')
    parser.add_argument('--candidate_output_dir', type=str, default='candidates/baseline', help='candidate output directory root')

    # Model Configuration Parameters
    parser.add_argument('--model_name', type=str, default='t5-small', choices=['google/t5-efficient-tiny', 't5-small', 't5-base', 't5-large', 't5-3b'], help='model name')
    parser.add_argument('--add_embedding', action='store_true', default=True, help='add rq_embedding to tokenizer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--lora', action='store_true', help='use lora')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')

    # Training Hyperparameters
    parser.add_argument('--train_epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')

    # Sequence Length Parameters
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--target_length', type=int, default=8, help='target length')
    parser.add_argument('--gen_len', type=int, default=8, help='generation length')

    # Training Strategy Parameters
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=1, help='save total limit')
    parser.add_argument('--add_prefix', action='store_true', help='add prefix to inputs')

    # Logging and Monitoring Parameters
    parser.add_argument('--log_freq', type=int, default=1, help='eval log frequency')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')

    # Distributed Training Parameters
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--gpu_id', type=int, default=1, help='specific GPU ID to use (None for all available GPUs)')

    # Experimental Evaluation Parameters
    parser.add_argument('--use_train_addition', action='store_true', help='include additional training data for Track 3 evaluation')
    parser.add_argument('--include_train_in_search', action='store_true', help='include training videos in search pool for Track 4 evaluation')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    # Evaluation Mode Parameters
    parser.add_argument('--eval', action='store_true', default=False, help='Run evaluation mode to generate video candidates')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='Path to checkpoint for evaluation (e.g., output/.../best_model)')
    parser.add_argument('--num_candidates', type=int, default=20, help='Number of candidates to generate per query (beam size)')
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help='Search pool setting: 1=test-only, 2=train+test combined')
    parser.add_argument('--detailed_generation', action='store_true', default=False,
                       help='Include (sID, video_id) pairs in candidates and ground_truth_sID in output')

    return parser.parse_args()


def validate_eval_checkpoint(eval_checkpoint: str) -> None:
    if not eval_checkpoint:
        raise ValueError("--eval_checkpoint is required when --eval is set")

    checkpoint_dir = Path(eval_checkpoint).resolve()
    if checkpoint_dir.name != 'best_model':
        raise ValueError(
            "Baseline inference only supports checkpoints saved as a 'best_model' directory. "
            f"Got: {checkpoint_dir.name}"
        )

    model_bin = checkpoint_dir / 'pytorch_model.bin'
    if not model_bin.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint weights: {model_bin}")


if __name__ == '__main__':
    train_args = parse_args()
    
    # Set specific GPU if requested (must be done before any CUDA operations)
    if train_args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(train_args.gpu_id)
        print(f'Using GPU {train_args.gpu_id} only')
    
    seed_everything(train_args.seed)
    dataset_name = train_args.dataset
    resolved_paths = build_retriever_paths(
        dataset_name=dataset_name,
        mode=train_args.mode,
        index_type=train_args.index_type,
        code_book_size=train_args.code_book_size,
        code_book_num=train_args.code_book_num,
        version=train_args.version,
        output_root=train_args.output_dir,
        candidate_output_root=train_args.candidate_output_dir,
    )

    print('training on dataset: ', dataset_name)
    print('using baseline indices from: ', resolved_paths.dataset_dir)

    model_name = train_args.model_name
    dataset_dir = str(resolved_paths.dataset_dir)
    train_caption_file = str(resolved_paths.train_caption_file) if resolved_paths.train_caption_file else None
    train_addition_caption_file = (
        str(resolved_paths.train_addition_caption_file)
        if resolved_paths.train_addition_caption_file
        else None
    )
    test_caption_file = str(resolved_paths.test_caption_file)
    train_data_file = str(resolved_paths.train_data_file) if resolved_paths.train_data_file else None
    train_index_file = str(resolved_paths.train_index_file)
    test_index_file = str(resolved_paths.test_index_file)
    codebook_embedding_path = str(resolved_paths.codebook_embedding_file)

    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    code_book_size = train_args.code_book_size
    code_book_num = train_args.code_book_num
    add_embedding = train_args.add_embedding
    dropout_rate = train_args.dropout_rate
    log_freq = train_args.log_freq
    source_length = train_args.source_length
    target_length = train_args.target_length
    gen_len = train_args.gen_len
    
    # Detect active evaluation tracks
    tracks = ['T1']  # Track 1 always active
    if train_args.use_train_addition:
        tracks.append('T3')  # Track 3: train+addition data
    else:
        tracks.append('T2')  # Track 2: standard training data
    
    if train_args.include_train_in_search:
        tracks.append('T4')  # Track 4: combined search pool
    
    track_string = ''.join(sorted(tracks))
    
    # Extract and clean model name
    model_short = train_args.model_name.split('/')[-1].replace('-', '').replace('_', '')
    
    # Detect precision
    if train_args.float16:
        precision = 'fp16'
    elif train_args.bf16:
        precision = 'bf16'
    else:
        precision = 'fp32'
    
    # Generate cleaner naming components
    rq_signal = 'emb' if add_embedding else 'noemb'
    lora_signal = 'lora' if train_args.lora else ''
    prefix_signal = 'pfx' if train_args.add_prefix else ''
    
    # Combine extras
    extras = '_'.join(filter(None, [rq_signal, lora_signal, prefix_signal]))
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    # New output directory format
    output_dir = f"{track_string}_{model_short}_{train_args.mode}_c{code_book_size}l{code_book_num}_v{train_args.version}"

    # When using single GPU mode, ensure local_rank is 0
    if train_args.gpu_id is not None:
        local_rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    
    if train_args.eval:
        validate_eval_checkpoint(train_args.eval_checkpoint)

    if local_rank == 0 and not train_args.eval:
        # New WANDB project naming
        project_name = f'SemanticID_{dataset_name}_{model_short}_{track_string}'
        exp_name = f'{train_args.index_type}_v{train_args.version}_{train_args.mode}_c{code_book_size}l{code_book_num}'
        wandb.login()
        wandb.init(project=project_name, name=exp_name)

    output_dir_name = str(
        resolved_paths.output_root
        / dataset_name
        / train_args.index_type
        / f"{train_args.model_name.split('/')[-1]}_{output_dir}"
        / current_time
    )

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.dropout_rate = dropout_rate
    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torch_dtype=torch_dtype,
                                                       config=config)

    # add extra tokens
    prefix = ['A_', 'B_', 'C_', 'D_', 'E_', 'F_', 'G_', 'H_', 'I_', 'J_', 'K_', 'L_', 'M_', 'N_', 'O_', 'P_', 'Q_', 'R_', 'S_', 'T_', 'U_', 'V_', 'W_', 'X_', 'Y_', 'Z_']
    extra_tokens = []
    if code_book_num == 1:
        for count in range(code_book_size):
            extra_tokens.append('C_'+str(count))
    else:
        for code_book in range(code_book_num):
            for count in range(code_book_size):
                extra_tokens.append(prefix[code_book]+str(count))
    print('number of extra tokens: ', len(extra_tokens))
    tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    codebook_embedding = None
    if add_embedding or train_args.detailed_generation:
        codebook_embedding = torch.load(codebook_embedding_path)
        if add_embedding:
            token_embeddings = model.get_input_embeddings()
            assert codebook_embedding.size(0) == code_book_size * code_book_num
            original_vocab_size = len(tokenizer) - code_book_size * code_book_num
            token_embeddings.weight.data[
                original_vocab_size:original_vocab_size + code_book_size * code_book_num
            ] = codebook_embedding
            print('codebook_embedding added')

    if train_args.lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            modules_to_save=['embed_tokens', 'lm_head'],
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Print total trainable parameters in millions
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M total')

    reporter = ['wandb'] if local_rank == 0 else "none"
    # reporter = "none"
    training_args = TrainingArguments(
        output_dir=output_dir_name,
        num_train_epochs=train_epoch,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        dataloader_num_workers=10,

        optim="adamw_torch",
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        evaluation_strategy=train_args.eval_strategy,
        # eval_steps=1000,

        save_strategy="no",
        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,

        # load_best_model_at_end=True,  # Disabled - using custom Recall@1 based saving
        # metric_for_best_model="eval_loss",  # Disabled - using custom Recall@1 based saving  
        save_only_model=True,

        fp16=train_args.float16,
        bf16=train_args.bf16,
        seed=train_args.seed,
    )
    model.config.use_cache = False

    # Create datasets based on format
    if train_args.index_type == 'videorqvae':
        # VideoRQVAE format
        if train_args.use_train_addition:
            raise NotImplementedError("train_addition not yet supported with videorqvae format")
        if train_args.include_train_in_search:
            raise NotImplementedError("include_train_in_search (Track 4) not yet supported with videorqvae format")

        # Training dataset (combined format)
        train_dataset = T5Dataset(
            dataset_name, tokenizer, 
            data_file=train_data_file,
            max_source_len=source_length, max_target_len=target_length,
            add_prefix=train_args.add_prefix
        )
        sub_train_dataset = T5Dataset(
            dataset_name, tokenizer,
            data_file=train_data_file,
            max_source_len=source_length, max_target_len=target_length,
            add_prefix=train_args.add_prefix, subset_size=1000
        )
        # Test dataset (separate caption + index files)
        test_dataset = T5TestDataset(
            dataset_name, tokenizer, test_caption_file, test_index_file,
            max_source_len=source_length, max_target_len=target_length,
            add_prefix=train_args.add_prefix
        )
        # For callback compatibility
        train_index_file = train_data_file

        # Use custom collator for videorqvae (handles non-tensor fields)
        data_collator = DataCollator(tokenizer=tokenizer, padding='max_length', max_length=source_length)
    else:
        # Standard/text_guided format (separate caption + index files)
        if train_args.use_train_addition:
            if train_addition_caption_file and os.path.exists(train_addition_caption_file):
                train_dataset = CombinedT5Dataset(
                    dataset_name, tokenizer,
                    [train_caption_file, train_addition_caption_file],
                    [train_index_file, test_index_file],
                    max_source_len=source_length,
                    max_target_len=target_length,
                    add_prefix=train_args.add_prefix
                )
                print(f"Using combined training data: {train_caption_file} + {train_addition_caption_file}")
            else:
                raise FileNotFoundError(f"Train addition caption file not found: {train_addition_caption_file}")
        else:
            train_dataset = T5Dataset(
                dataset_name, tokenizer,
                caption_file=train_caption_file, index_file=train_index_file,
                max_source_len=source_length, max_target_len=target_length,
                add_prefix=train_args.add_prefix
            )

        test_dataset = T5TestDataset(
            dataset_name, tokenizer, test_caption_file, test_index_file,
            max_source_len=source_length, max_target_len=target_length,
            add_prefix=train_args.add_prefix
        )
        sub_train_dataset = T5Dataset(
            dataset_name, tokenizer,
            caption_file=train_caption_file, index_file=train_index_file,
            max_source_len=source_length, max_target_len=target_length,
            add_prefix=train_args.add_prefix, subset_size=1000
        )

        # Use custom collator for consistency
        data_collator = DataCollator(tokenizer=tokenizer, padding='max_length', max_length=source_length)

    # EVALUATION MODE: Generate video candidates and exit
    if train_args.eval:
        from .avg_utils import run_evaluation
        run_evaluation(
            train_args, tokenizer, model, test_dataset,
            test_index_file, train_index_file, data_collator,
            detailed_generation=train_args.detailed_generation,
            codebook_embedding=codebook_embedding,
            code_book_size=code_book_size,
            code_book_num=code_book_num,
            candidate_output_dir=str(resolved_paths.candidate_output_root),
        )
        import sys
        sys.exit(0)

    os.makedirs(output_dir_name, exist_ok=True)
    logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    if local_rank == 0:
        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transfomers training_args: '+str(training_args))

    trainer = Trainer(
        # temperature=train_args.temperature,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[QueryEvalCallback(local_rank=local_rank,
                                     sub_train_dataset=sub_train_dataset,
                                     test_dataset=test_dataset,
                                     tgt_file=test_index_file,
                                     train_index_file=train_index_file,  # For Track 4 combined search pool
                                     logger=logger,
                                     batch_size=128,
                                     collator=data_collator,
                                     tokenizer=tokenizer,
                                     wandb=wandb,
                                     log_freq=log_freq,
                                     gen_len=gen_len,
                                     use_train_addition=train_args.use_train_addition,  # Track 3 flag
                                     include_train_in_search=train_args.include_train_in_search,
                                     seed=train_args.seed)],  # Track 4 flag
    )

    trainer.train()
