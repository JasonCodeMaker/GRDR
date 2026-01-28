import os
import csv
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    # Verify if the candidate file exists
    if config.candidate_file is not None:
        print(f"Candidate file: {config.candidate_file}")
    else:
        print("No candidate file provided")

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    test_dataset = test_data_loader.dataset

    # Create expanded pool loader if flag enabled
    expanded_pool_loader = None
    extra_vid_ids = None
    if hasattr(config, 'expanded_pool') and config.expanded_pool:
        extra_vid_ids, videos_dir, video_ext, path_fn = DataFactory.get_train_video_ids(config)
        expanded_pool_loader = DataFactory.get_video_only_loader(
            config, extra_vid_ids, videos_dir, video_ext, path_fn)
        print(f"Expanded pool enabled: {len(extra_vid_ids)} train videos will be added to search pool")

    # Generate candidate mask ONCE after expanded_pool decision is made
    if hasattr(config, 'rerank_mode') and config.rerank_mode and config.candidate_file:
        print(f"Generating candidate mask from {config.candidate_file}")
        if hasattr(test_dataset, '_generate_candidate_mask'):
            test_dataset.candidate_mask = test_dataset._generate_candidate_mask(
                config.candidate_file, extra_vid_ids=extra_vid_ids)
            print(f"Generated candidate mask with shape: {test_dataset.candidate_mask.shape}")
        else:
            print(f"Warning: Dataset {type(test_dataset).__name__} does not support candidate mask generation")

    model = ModelFactory.get_model(config)
    
    # Print total model parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters: {total_params / 1e6:.2f}M')
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer,
                      expanded_pool_loader=expanded_pool_loader)

    # if config.load_epoch is not None:
    #     if config.load_epoch > 0:
    #         trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
    #     else:
    #         trainer.load_checkpoint("model_best.pth")    

    trainer.load_checkpoint(config.eval_checkpoint)

    result = trainer.validate()

    # Save results to CSV
    output_dir = "output/reranker"
    csv_path = os.path.join(output_dir, config.result_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['R@1', 'R@5', 'R@10', 'MedR', 'MeanR'])
        # Write values
        csv_writer.writerow([
            result['R1'],
            result['R5'],
            result['R10'],
            result['MedR'],
            result['MeanR']
        ])

    print(f"\nResults saved to: {csv_path}")
    print(f"R@1: {result['R1']:.2f}, R@5: {result['R5']:.2f}, R@10: {result['R10']:.2f}, MedR: {result['MedR']:.2f}, MeanR: {result['MeanR']:.2f}")


if __name__ == '__main__':
    main()

