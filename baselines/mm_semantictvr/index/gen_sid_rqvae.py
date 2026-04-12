import collections
import json

import numpy as np
import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm
import argparse
from collections import defaultdict

from torch.utils.data import DataLoader

from .datasets import VideoTextGuidedDataset, load_internvideo2_features
from .models.rqvae import RQVAE

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Index Generation for MSRVTT Video")
    
    parser.add_argument('--ckpt_path', type=str, default="index/log/msrvtt/standard/code_num_256_codebook_layers_4/best_collision_model.pth", help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default="../data/msrvtt", help='Output directory for generated indices')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for inference')
    
    # Dataset parameters - updated to match current structure
    parser.add_argument("--dataset", type=str, default="msrvtt", 
                        choices=['msrvtt', 'didemo', 'actnet', 'activitynet', 'lsmdc'],
                        help="Dataset name")
    parser.add_argument("--features_root", type=str,
                        default="./dataset/features",
                        help="Path to features directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=['train', 'test'],
                        help="Dataset split")
    parser.add_argument("--type", type=str, default="standard",
                        choices=['standard', 'text_guided'],
                        help="Type of model")
    parser.add_argument("--mode", type=str, default="none",
                        choices=['joint', 'separate', 'nn', 'none'],
                        help="Processing mode: 'separate' avoids collisions within current split only, 'joint' ensures no collisions between train and test sets, 'nn' assigns test videos the semantic IDs of their nearest neighbor train videos, 'none' generates raw semantic IDs without collision handling and provides detailed collision analysis")

    # Codebook embedding extraction parameters
    parser.add_argument("--extract_codebook", action='store_true', default=True,
                        help="Extract and save codebook embeddings as codebook_embedding.pt")
    parser.add_argument("--codebook_output_path", type=str, default=None,
                        help="Output path for codebook_embedding.pt (default: same as output_dir)")

    # Model architecture parameters (must match training configuration)
    parser.add_argument('--code_num', type=int, default=256, help='number of codes per quantization layer')
    parser.add_argument('--codebook_layers', type=int, default=4, help='number of quantization layers in RQ-VAE')
    parser.add_argument('--e_dim', type=int, default=512, help='vq codebook embedding size')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512], help='hidden sizes of encoder/decoder layers')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--bn', type=bool, default=False, help='use batch normalization')
    parser.add_argument('--loss_type', type=str, default="mse", help='reconstruction loss type')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantization loss weight')
    parser.add_argument('--kmeans_init', type=bool, default=True, help='use kmeans initialization')
    parser.add_argument('--kmeans_iters', type=int, default=100, help='max kmeans iterations')
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0], help='sinkhorn epsilons')
    parser.add_argument('--sk_iters', type=int, default=50, help='max sinkhorn iterations')

    return parser.parse_args()

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def get_collision_video_ids(collision_item_groups, data):
    """Convert collision item indices to actual video IDs for better analysis"""
    collision_video_groups = []
    for collision_items in collision_item_groups:
        video_ids = [data.pairs[item] for item in collision_items]
        collision_video_groups.append({
            'video_ids': video_ids,
            'indices': collision_items,
            'count': len(video_ids)
        })
    return collision_video_groups

def print_collision_analysis(collision_video_groups):
    """Print detailed collision analysis with video IDs"""
    print(f"\n=== Collision Analysis ===")    
    # Sort by collision group size for better analysis
    sorted_groups = sorted(collision_video_groups, key=lambda x: x['count'], reverse=True)
    
    for i, group in enumerate(sorted_groups):
        print(f"  Collision Group {i+1}: {group['count']} videos")
        print(f"    Video IDs: {group['video_ids']}")
        print(f"    Item indices: {group['indices']}")

    print(f"Total collision groups: {len(collision_video_groups)}")
    total_colliding_videos = sum(group['count'] for group in collision_video_groups)
    print(f"Total videos involved in collisions: {total_colliding_videos}")


def extract_codebook_embeddings(model, code_book_num, code_book_size, e_dim):
    """Extract and concatenate codebook embeddings from all quantization layers
    
    Args:
        model: Trained RQVAE model
        code_book_num: Number of codebook layers
        code_book_size: Size of each codebook
        e_dim: Embedding dimension
        
    Returns:
        torch.Tensor: Concatenated codebook embeddings with shape (code_book_num * code_book_size, e_dim)
    """
    print(f"\n=== Extracting Codebook Embeddings ===")
    print(f"Code book configuration: {code_book_num} layers × {code_book_size} codes = {code_book_num * code_book_size} total codes")
    print(f"Embedding dimension: {e_dim}")
    
    # Extract codebook embeddings from the residual quantizer
    all_codebooks = model.rq.get_codebook()  # Shape: (num_layers, code_book_size, e_dim)
    
    # Concatenate all codebook layers into a single tensor
    codebook_embeddings = all_codebooks.view(-1, e_dim)  # Shape: (code_book_num * code_book_size, e_dim)
    
    print(f"Extracted codebook shape: {codebook_embeddings.shape}")
    print(f"Expected shape: ({code_book_num * code_book_size}, {e_dim})")
    
    # Validation
    assert codebook_embeddings.shape[0] == code_book_num * code_book_size, \
        f"Expected {code_book_num * code_book_size} total codes, got {codebook_embeddings.shape[0]}"
    assert codebook_embeddings.shape[1] == e_dim, \
        f"Expected embedding dimension {e_dim}, got {codebook_embeddings.shape[1]}"
    
    print(" Codebook embedding extraction successful")
    return codebook_embeddings

def main():
    args = parse_args()

    ckpt_path = args.ckpt_path
    output_dir = args.output_dir

    device = torch.device(args.device)

    print(f"Checkpoint path: {ckpt_path}")
    print(f"Using device: {device}")
    print(f"Processing mode: {args.mode}")
    print(f"Split: {args.split}")
    print(f"Dataset: {args.dataset}")

    # video prefix
    prefix = ["A_{}","B_{}","C_{}","D_{}","E_{}","F_{}","G_{}","H_{}","I_{}","J_{}","K_{}","L_{}","M_{}","N_{}","O_{}","P_{}","Q_{}","R_{}","S_{}","T_{}","U_{}","V_{}","W_{}","X_{}","Y_{}","Z_{}"]

    # Load checkpoint (without args - use command-line args for model architecture)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = ckpt["state_dict"]

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best collision rate from checkpoint: {ckpt.get('best_collision_rate', 'N/A')}")

    # Generate num_emb_list from command-line args
    num_emb_list = [args.code_num] * args.codebook_layers
    print(f"Using num_emb_list from args: code_num={args.code_num}, codebook_layers={args.codebook_layers} -> {num_emb_list}")

    # Output file naming using InternVideo2 style
    output_file = f"{args.dataset}_index_internvideo2_emb_{args.split}.json"
    # Ensure output directory exists
    output_dir = os.path.join(args.output_dir, f"{args.mode}/{args.type}_c{args.code_num}_l{args.codebook_layers}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)

    print(f"Output will be saved to: {output_file}")

    # Pre-load InternVideo2 features for efficiency
    print(f"\nLoading InternVideo2 features for {args.dataset.upper()}...")
    train_vid, train_txt, test_vid, test_txt = load_internvideo2_features(
        args.dataset, args.features_root
    )
    print(f"Loaded train video features: {len(train_vid)} samples")
    if test_vid:
        print(f"Loaded test video features: {len(test_vid)} samples")

    # Load dataset(s) based on processing mode
    if args.mode == "joint":
        print(f"\nJoint mode: Loading both train and test datasets for {args.dataset.upper()}")
        train_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="train", text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=train_vid, text_features=train_txt
        )
        test_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="test", text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=test_vid, text_features=test_txt
        )
        print(f"Training dataset: {len(train_data)} samples, Test dataset: {len(test_data)} samples")
        print(f"Embedding dimension: {train_data.dim}")

        # Create combined dataset for joint processing
        class CombinedDataset:
            def __init__(self, train_data, test_data):
                self.train_data = train_data
                self.test_data = test_data
                self.train_size = len(train_data)
                self.test_size = len(test_data)
                self.dim = train_data.dim
                # Combine pairs with split labels
                self.pairs = []
                self.split_labels = []
                for pair in train_data.pairs:
                    self.pairs.append(pair)
                    self.split_labels.append('train')
                for pair in test_data.pairs:
                    self.pairs.append(pair)
                    self.split_labels.append('test')

            def __len__(self):
                return self.train_size + self.test_size

            def __getitem__(self, idx):
                if idx < self.train_size:
                    return self.train_data[idx]
                else:
                    return self.test_data[idx - self.train_size]

        data = CombinedDataset(train_data, test_data)
        print(f"Combined dataset: {len(data)} total samples ({data.train_size} train + {data.test_size} test)")
    elif args.mode == "nn":
        print(f"\nNN mode: Loading both train and test datasets for nearest neighbor processing")
        train_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="train", text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=train_vid, text_features=train_txt
        )
        test_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="test", text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=test_vid, text_features=test_txt
        )
        print(f"Training dataset: {len(train_data)} samples, Test dataset: {len(test_data)} samples")
        print(f"Embedding dimension: {train_data.dim}")

        # For NN mode, we need to keep train and test data separate for different processing
        data = train_data  # Use train_data for initial model processing
        print(f"Will process train data through RQVAE, then use cosine similarity for test data")
    elif args.mode == "none":
        print(f"\nNone mode: Loading {args.dataset.upper()} {args.split} dataset for collision analysis and raw output generation...")
        video_features = train_vid if args.split == "train" else test_vid
        text_features = train_txt if args.split == "train" else test_txt
        data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split=args.split, text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=video_features, text_features=text_features
        )
        print(f"Dataset loaded: {len(data)} video samples, embedding dimension: {data.dim}")
        print(f"Note: No collision handling will be applied - will generate raw semantic IDs with collision analysis")
    else:
        print(f"\nSeparate mode: Loading {args.dataset.upper()} {args.split} dataset...")
        video_features = train_vid if args.split == "train" else test_vid
        text_features = train_txt if args.split == "train" else test_txt
        data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split=args.split, text_guided=False,
            model_type='rqvae', feature_extractor='InternVideo2',
            video_features=video_features, text_features=text_features
        )
        print(f"Dataset loaded: {len(data)} video samples, embedding dimension: {data.dim}")

    model = RQVAE(in_dim=data.dim,
                    num_emb_list=num_emb_list,
                    e_dim=args.e_dim,
                    layers=args.layers,
                    dropout_prob=args.dropout_prob,
                    bn=args.bn,
                    loss_type=args.loss_type,
                    quant_loss_weight=args.quant_loss_weight,
                    kmeans_init=args.kmeans_init,
                    kmeans_iters=args.kmeans_iters,
                    sk_epsilons=args.sk_epsilons,
                    sk_iters=args.sk_iters,
                    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully with {len(num_emb_list)} quantization layers")
    print(f"Model architecture summary: {data.dim}D input -> {num_emb_list} codebooks -> {args.e_dim}D quantized")
    # print(model)

    # Extract codebook embeddings if requested
    if args.extract_codebook:
        codebook_output_path = args.codebook_output_path or output_dir
        codebook_file = os.path.join(codebook_output_path, "codebook_embedding.pt")
        
        # Extract codebook embeddings
        codebook_embeddings = extract_codebook_embeddings(
            model=model,
            code_book_num=len(num_emb_list),
            code_book_size=num_emb_list[0],  # Assuming all layers have same code size
            e_dim=args.e_dim
        )

        # Save codebook embeddings
        os.makedirs(codebook_output_path, exist_ok=True)
        torch.save(codebook_embeddings.cpu(), codebook_file)
        print(f"  Codebook embeddings saved to: {codebook_file}")
        print(f"  Shape: {codebook_embeddings.shape}")
        print(f"  Compatible with T5 training (e_dim={args.e_dim})")
        
        # Early return if only extracting codebook (no need to generate semantic IDs)
        if not any([args.split]):  # If no specific processing requested, just extract codebook
            return

    # Handle nn mode processing separately
    if args.mode == "nn":
        print(f"\n=== NN Mode Processing ===")
        # Phase 1: Process train data through RQVAE to get semantic IDs
        print(f"Phase 1: Processing {len(train_data)} train videos through RQVAE...")
        train_loader = DataLoader(train_data, num_workers=4,
                                 batch_size=64, shuffle=False, pin_memory=True)
        
        train_indices = []
        train_indices_str = []
        train_distances = []
        train_embeddings = []  # Store raw embeddings for similarity computation
        
        for batch in tqdm(train_loader, desc="Processing train data"):
            # Handle both dict (test split) and tensor (train split) returns
            if isinstance(batch, dict):
                d = batch['video_patches'].to(device)
            else:
                d = batch.to(device)
            indices, distances = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            distances = distances.cpu().tolist()

            # Store raw embeddings before quantization
            train_embeddings.extend(d.cpu().numpy())
            
            for index in indices:
                code = [int(ind) for ind in index]
                train_indices.append(code)
                train_indices_str.append(str(code))
            train_distances.extend(distances)
        
        print(f"Generated {len(train_indices)} train semantic IDs")
        
        # Resolve collisions in train data using existing collision resolution
        print("Resolving collisions in train semantic IDs...")
        all_indices = train_indices
        all_indices_str = train_indices_str
        all_distances = np.array(train_distances)
        all_indices_str_set = set(train_indices_str)
        
        # Apply existing collision resolution to train data
        print("Processing train data collision resolution...")
        
    else:
        # Original processing for joint/separate/none modes
        data_loader = DataLoader(data, num_workers=4,
                                    batch_size=64, shuffle=False,
                                    pin_memory=True)

        all_indices = []
        all_indices_str = []
        all_distances = []
        all_indices_str_set = set()

    if args.mode != "nn":
        for batch in tqdm(data_loader):
            # Handle both dict (test split) and tensor (train split) returns
            if isinstance(batch, dict):
                d = batch['video_patches'].to(device)
            else:
                d = batch.to(device)
            indices, distances = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            distances = distances.cpu().tolist()
            for index in indices:
                code = []
                for i, ind in enumerate(index):
                    # code.append(prefix[i].format(int(ind)))
                    code.append(int(ind))

                all_indices.append(code)
                all_indices_str.append(str(code))
                all_indices_str_set.add(str(code))
                # print(str(code))
            # break
            all_distances.extend(distances)


    # NN mode: Handle test data processing using cosine similarity
    if args.mode == "nn":
        # Apply collision resolution to train data first
        print(f"Train data generated {len(all_indices)} semantic IDs, applying collision resolution...")
        
    all_distances = np.array(all_distances)

    # print(all_distances)
    print(all_distances.shape) ## (num, 4, 256)

    sort_distances_index = np.argsort(all_distances, axis=2)

    item_min_dis = defaultdict(list)

    for item, distances in tqdm(enumerate(all_distances), desc='cal distances'):
        for dis in distances:
            item_min_dis[item].append(np.min(dis))

    # Initial collision detection
    collision_item_groups = get_collision_item(all_indices_str)
    all_collision_items = set()
    for collision_items in collision_item_groups:
        for item in collision_items:
            all_collision_items.add(item)
    
    # Collision analysis with video IDs
    collision_video_groups = get_collision_video_ids(collision_item_groups, data)
    # print_collision_analysis(collision_video_groups)
    
    # None mode: print collision info but continue to generate output files
    if args.mode == "none":
        print(f"\n=== None Mode: Collision Analysis Complete ===")
        print(f"Total items processed: {len(all_indices_str)}")
        print(f"Unique codes generated: {len(set(all_indices_str))}")
        print(f"Overall collision rate: {(len(all_indices_str)-len(set(all_indices_str)))/len(all_indices_str):.6f}")
        print(f"Max conflicts per code: {max(get_indices_count(all_indices_str).values())}")
        print(f"Total collision groups: {len(collision_item_groups)}")
        total_colliding_videos = sum(group['count'] for group in collision_video_groups)
        print(f"Total videos involved in collisions: {total_colliding_videos}")

        # Skip collision resolution and jump directly to output generation
        # Set variables needed for final output generation
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        
        # Generate output file for none mode (same as separate mode)
        all_indices_dict = {}
        for item, indices in enumerate(all_indices):
            video_id = data.pairs[item]
            code = []
            for i, ind in enumerate(indices):
                code.append(prefix[i].format(int(ind)))
            all_indices_dict[video_id] = code
        
        with open(output_file, 'w') as fp:
            json.dump(all_indices_dict, fp, indent=4)
            
        print(f"\n=== None Mode Output Generation Complete ===")
        print(f"Semantic indices successfully generated and saved to: {output_file}")
        print(f"Total videos processed: {len(all_indices_dict)}")
        print(f"Final collision rate: {(tot_item-tot_indice)/tot_item:.6f}")
        print(f"Unique semantic codes generated: {tot_indice}/{tot_item}")
        print(f"WARNING: Output contains unresolved collisions - use only for analysis purposes")
        return
    
    # Joint mode: analyze train-test collisions
    if args.mode == "joint":
        train_indices_str = all_indices_str[:data.train_size]
        test_indices_str = all_indices_str[data.train_size:]
        train_codes_set = set(train_indices_str)
        test_train_collisions = sum(1 for code in test_indices_str if code in train_codes_set)
        
        print(f"\n=== Joint Mode Collision Analysis ===")
        print(f"Training samples: {len(train_indices_str)}")
        print(f"Test samples: {len(test_indices_str)}")
        print(f"Test codes colliding with training: {test_train_collisions}")
        print(f"Cross-split collision rate: {test_train_collisions/len(test_indices_str):.6f}")
    else:
        train_codes_set = set()  # Empty for separate mode

    # new_indices_set = set()

    tt = 0
    level = len(num_emb_list) - 1
    max_num = num_emb_list[0]

    while True:
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        
        # Calculate collision rates based on mode
        if args.mode == "joint":
            train_indices_str = all_indices_str[:data.train_size]
            test_indices_str = all_indices_str[data.train_size:]
            train_codes_set = set(train_indices_str)
            
            # Cross-split collisions: test codes that appear in training set
            test_train_collisions = sum(1 for code_str in test_indices_str if code_str in train_codes_set)
            
            # Internal collisions within each split
            train_internal_collisions = len(train_indices_str) - len(set(train_indices_str))
            test_internal_collisions = len(test_indices_str) - len(set(test_indices_str))
            
            print(f'Iteration {tt+1}: Total items: {tot_item}, Unique codes: {tot_indice}')
            print(f'  Train internal collisions: {train_internal_collisions}')
            print(f'  Test internal collisions: {test_internal_collisions}')
            print(f'  Test-Train cross collisions: {test_train_collisions}')
            print(f'  Overall collision rate: {(tot_item-tot_indice)/tot_item:.6f}')
        else:
            print(f'Iteration {tt+1}: tot_item: {tot_item}, tot_indice: {tot_indice}')
            print(f"  Collision Rate: {(tot_item-tot_indice)/tot_item:.6f}")
        
        # Check termination condition
        has_internal_collisions = not check_collision(all_indices_str)
        has_cross_collisions = False
        
        if args.mode == "joint":
            # Check for cross-split collisions
            train_indices_str = all_indices_str[:data.train_size]
            test_indices_str = all_indices_str[data.train_size:]
            train_codes_set = set(train_indices_str)
            has_cross_collisions = any(code_str in train_codes_set for code_str in test_indices_str)
        
        if (not has_internal_collisions and not has_cross_collisions) or tt >= 5:
            print(f'Termination: tt={tt}, internal_collisions={has_internal_collisions}, cross_collisions={has_cross_collisions}')
            break

        collision_item_groups = get_collision_item(all_indices_str)
        
        # Joint mode: add test items that collide with training set
        if args.mode == "joint":
            train_indices_str = all_indices_str[:data.train_size]
            test_indices_str = all_indices_str[data.train_size:]
            train_codes_set = set(train_indices_str)
            
            # Find test items that have codes colliding with training set
            test_train_collision_items = []
            for i, code_str in enumerate(test_indices_str):
                if code_str in train_codes_set:
                    # Add data.train_size to get actual index in combined dataset
                    test_train_collision_items.append(i + data.train_size)
            
            print(f"  Iteration {tt+1}: Found {len(collision_item_groups)} internal collision groups, {len(test_train_collision_items)} test-train collisions")
            
            # Add test-train collision items as individual "collision groups"
            for item in test_train_collision_items:
                collision_item_groups.append([item])
        else:
            print(f"  Iteration {tt+1}: Found {len(collision_item_groups)} internal collision groups")
        
        for collision_items in collision_item_groups:
            min_distances = []
            for i, item in enumerate(collision_items):
                min_distances.append(item_min_dis[item][level])

            min_index = np.argsort(np.array(min_distances))
            
            for i, m_index in enumerate(min_index):
                
                if i == 0:
                    continue
                
                item = collision_items[m_index]
                # print(item)
                
                ori_code = copy.deepcopy(all_indices[item])
                # print(ori_code)
                
                num = i
                # Use all current codes as collision check set (includes both train and test in joint mode)
                collision_check_set = all_indices_str_set
                
                while str(ori_code) in collision_check_set and num < max_num:
                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1
                    # print(sort_distances_index[item][level])
                    # print(ori_code)
                    # print(num)
                
                for i in range(1, max_num):
                    if str(ori_code) in collision_check_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level-1] = sort_distances_index[item][level-1][i]
                        
                    num = 0
                    while str(ori_code) in collision_check_set and num < max_num:
                        ori_code[level] = sort_distances_index[item][level][num]
                        num += 1
                        
                    if str(ori_code) not in collision_check_set:
                        break
                    
                all_indices[item] = ori_code
                all_indices_str[item] = str(ori_code)

                all_indices_str_set.add(str(ori_code))

                # print(str(ori_code))
            
            
        # if level == 2:
        #     break
        tt += 1

    # NN mode: Process test data using cosine similarity
    if args.mode == "nn":
        print(f"\n=== NN Mode: Test Data Processing ===")
        print(f"Phase 2: Processing {len(test_data)} test videos using cosine similarity...")
        
        # Convert train embeddings to tensor for similarity computation (keep on CPU to save memory)
        train_embeddings_tensor = torch.tensor(np.array(train_embeddings))
        print(f"Train embeddings shape: {train_embeddings_tensor.shape}")
        
        # Store final train semantic IDs after collision resolution
        final_train_indices = copy.deepcopy(all_indices)
        final_train_indices_str = copy.deepcopy(all_indices_str)
        
        # Process test data in batches
        test_loader = DataLoader(test_data, num_workers=4,
                                batch_size=64, shuffle=False, pin_memory=True)
        
        test_embeddings = []
        test_similarity_rankings = []
        test_video_ids = []
        
        # Collect all test embeddings
        print("Loading test embeddings...")
        for batch in tqdm(test_loader, desc="Loading test embeddings"):
            # Handle both dict (test split) and tensor (train split) returns
            if isinstance(batch, dict):
                d = batch['video_patches']
            else:
                d = batch
            test_embeddings.extend(d.cpu().numpy())
        
        # Convert test embeddings to tensor (keep on CPU)
        test_embeddings_tensor = torch.tensor(np.array(test_embeddings))
        print(f"Test embeddings shape: {test_embeddings_tensor.shape}")
        
        # Compute cosine similarity in batches to avoid memory issues
        print("Computing cosine similarity matrix in batches...")
        batch_size = 100  # Process 100 test videos at a time
        similarity_rankings = []
        
        for batch_start in tqdm(range(0, len(test_embeddings_tensor), batch_size), desc="Computing similarities"):
            batch_end = min(batch_start + batch_size, len(test_embeddings_tensor))
            test_batch = test_embeddings_tensor[batch_start:batch_end]  # (batch_size, 768)
            
            # Compute similarity between batch and all train embeddings
            batch_similarity = F.cosine_similarity(
                test_batch.unsqueeze(1),  # (batch_size, 1, 768)
                train_embeddings_tensor.unsqueeze(0),  # (1, N_train, 768)
                dim=2  # (batch_size, N_train)
            )
            
            # Get rankings for this batch
            batch_rankings = torch.argsort(batch_similarity, dim=1, descending=True)
            similarity_rankings.append(batch_rankings)
        
        # Concatenate all batch rankings
        similarity_rankings = torch.cat(similarity_rankings, dim=0)
        print(f"Similarity rankings shape: {similarity_rankings.shape}")
        print("Phase 3: Assigning test semantic IDs using nearest neighbors...")
        
        # Initialize test results
        test_indices = []
        test_indices_str = []
        test_assigned_ids = {}
        test_similarity_scores = []
        test_rank_used = []
        
        # Assign semantic IDs to test videos with collision resolution
        for test_idx in tqdm(range(len(test_data)), desc="Assigning test IDs"):
            video_id = test_data.pairs[test_idx]
            test_video_ids.append(video_id)
            
            rank = 0
            assigned = False
            
            while rank < len(final_train_indices) and not assigned:
                # Get the train video with rank-th highest similarity
                best_train_idx = similarity_rankings[test_idx, rank].item()
                candidate_semantic_id = final_train_indices[best_train_idx]
                candidate_semantic_id_str = str(candidate_semantic_id)
                
                # Calculate similarity score on demand to save memory
                test_emb = test_embeddings_tensor[test_idx].unsqueeze(0)  # (1, 768)
                train_emb = train_embeddings_tensor[best_train_idx].unsqueeze(0)  # (1, 768)
                similarity_score = F.cosine_similarity(test_emb, train_emb, dim=1).item()
                
                # Check if this semantic ID is already used by another test video
                if candidate_semantic_id_str not in test_assigned_ids.values():
                    # Assign this semantic ID to the test video
                    test_indices.append(candidate_semantic_id)
                    test_indices_str.append(candidate_semantic_id_str)
                    test_assigned_ids[test_idx] = candidate_semantic_id_str
                    test_similarity_scores.append(similarity_score)
                    test_rank_used.append(rank + 1)  # 1-indexed for reporting
                    assigned = True
                else:
                    rank += 1
            
            if not assigned:
                raise ValueError(f"Could not find unique semantic ID for test video {test_idx} (video_id: {video_id})")
        
        # Extend all_indices and related lists with test data
        all_indices.extend(test_indices)
        all_indices_str.extend(test_indices_str)
        
        # Print collision resolution statistics
        print(f"\n=== Test Collision Resolution Statistics ===")
        rank_distribution = {}
        for rank in test_rank_used:
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        for rank in sorted(rank_distribution.keys()):
            count = rank_distribution[rank]
            percentage = (count / len(test_data)) * 100
            print(f"  Rank {rank} (choice #{rank}): {count} videos ({percentage:.1f}%)")
        
        print(f"  Average similarity score: {np.mean(test_similarity_scores):.4f}")
        print(f"  Min similarity score: {np.min(test_similarity_scores):.4f}")
        print(f"  Max similarity score: {np.max(test_similarity_scores):.4f}")
        print(f"  Test videos requiring 2+ choices: {sum(1 for r in test_rank_used if r > 1)} ({sum(1 for r in test_rank_used if r > 1)/len(test_data)*100:.1f}%)")
        
        # Store additional data for output generation
        nn_mode_data = {
            'train_indices': final_train_indices,
            'train_indices_str': final_train_indices_str,
            'test_indices': test_indices,
            'test_indices_str': test_indices_str,
            'test_similarity_scores': test_similarity_scores,
            'test_rank_used': test_rank_used,
            'train_video_ids': [train_data.pairs[i] for i in range(len(train_data))],
            'test_video_ids': test_video_ids
        }
    else:
        # Initialize empty nn_mode_data for other modes to avoid undefined variable errors
        nn_mode_data = {}

    print("All indices number: ",len(all_indices))
    all_indices_str = [str(indice) for indice in all_indices]
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    
    print(f"Final Processing Summary:")
    print(f"  Total items processed: {tot_item}")
    print(f"  Unique codes generated: {tot_indice}")
    print(f"  Overall collision rate: {(tot_item-tot_indice)/tot_item:.6f}")
    print(f"  Max conflicts per code: {max(get_indices_count(all_indices_str).values())}")

    # Final collision analysis with video IDs  
    collision_item_groups = get_collision_item(all_indices_str)
    
    # Handle collision analysis differently for nn mode
    if args.mode == "nn":
        # For nn mode, we need to create a temporary combined data structure for collision analysis
        class TempNNData:
            def __init__(self, train_video_ids, test_video_ids):
                self.pairs = train_video_ids + test_video_ids
                
        temp_data = TempNNData(nn_mode_data['train_video_ids'], nn_mode_data['test_video_ids'])
        collision_video_groups = get_collision_video_ids(collision_item_groups, temp_data)
    else:
        collision_video_groups = get_collision_video_ids(collision_item_groups, data)
        
    if len(collision_video_groups) > 0:
        print_collision_analysis(collision_video_groups)
    else:
        print("No remaining internal collisions found.")

    # Create final output with video IDs mapped to generated codes
    if args.mode == "joint":
        # Split results into train and test dictionaries
        train_indices_dict = {}
        test_indices_dict = {}
        
        for item, indices in enumerate(all_indices):
            video_id = data.pairs[item]
            code = []
            for i, ind in enumerate(indices):
                code.append(prefix[i].format(int(ind)))
            
            if item < data.train_size:
                train_indices_dict[video_id] = code
            else:
                test_indices_dict[video_id] = code
        
        # Save train results
        train_output_file = os.path.join(output_dir, "msrvtt_index_cliplargel14_emb_train.json")
        with open(train_output_file, 'w') as fp:
            json.dump(train_indices_dict, fp, indent=4)
        
        # Save test results  
        test_output_file = os.path.join(output_dir, "msrvtt_index_cliplargel14_emb_test.json")
        with open(test_output_file, 'w') as fp:
            json.dump(test_indices_dict, fp, indent=4)
        
        print(f"\nJoint processing results:")
        print(f"Training indices saved to: {train_output_file} ({len(train_indices_dict)} videos)")
        print(f"Test indices saved to: {test_output_file} ({len(test_indices_dict)} videos)")
        
        # Verify uniqueness
        train_codes = set(str(code) for code in train_indices_dict.values())
        test_codes = set(str(code) for code in test_indices_dict.values())
        all_codes = train_codes.union(test_codes)
        
        train_internal_collisions = len(train_indices_dict) - len(train_codes)
        test_internal_collisions = len(test_indices_dict) - len(test_codes)
        cross_collisions = len(train_codes.intersection(test_codes))
        
        print(f"\nFinal Collision Analysis:")
        print(f"  Train internal collisions: {train_internal_collisions}")
        print(f"  Test internal collisions: {test_internal_collisions}")
        print(f"  Train-Test cross collisions: {cross_collisions}")
        print(f"  Total unique codes: {len(all_codes)}/{len(all_indices)}")
        print(f"  Overall collision rate: {(len(all_indices) - len(all_codes))/len(all_indices):.6f}")
        print(f"  Cross-set uniqueness status: {'GUARANTEED' if cross_collisions == 0 else 'VIOLATED'}")
        
    elif args.mode == "nn":
        # NN mode: save separate train and test results with additional metadata
        train_indices_dict = {}
        test_indices_dict = {}
        
        # Process train indices
        for item, indices in enumerate(nn_mode_data['train_indices']):
            video_id = nn_mode_data['train_video_ids'][item]
            code = []
            for i, ind in enumerate(indices):
                code.append(prefix[i].format(int(ind)))
            train_indices_dict[video_id] = code
        
        # Process test indices
        for item, indices in enumerate(nn_mode_data['test_indices']):
            video_id = nn_mode_data['test_video_ids'][item]
            code = []
            for i, ind in enumerate(indices):
                code.append(prefix[i].format(int(ind)))
            test_indices_dict[video_id] = code
        
        # Save train results
        train_output_file = os.path.join(output_dir, "msrvtt_index_cliplargel14_emb_train.json")
        with open(train_output_file, 'w') as fp:
            json.dump(train_indices_dict, fp, indent=4)
        
        # Save test results  
        test_output_file = os.path.join(output_dir, "msrvtt_index_cliplargel14_emb_test.json")
        with open(test_output_file, 'w') as fp:
            json.dump(test_indices_dict, fp, indent=4)
        
        # Save additional metadata for nn mode
        metadata = {
            'mode': 'nn',
            'train_videos': len(train_indices_dict),
            'test_videos': len(test_indices_dict),
            'test_similarity_stats': {
                'mean': float(np.mean(nn_mode_data['test_similarity_scores'])),
                'min': float(np.min(nn_mode_data['test_similarity_scores'])),
                'max': float(np.max(nn_mode_data['test_similarity_scores'])),
                'std': float(np.std(nn_mode_data['test_similarity_scores']))
            },
            'test_rank_distribution': {
                str(rank): nn_mode_data['test_rank_used'].count(rank) 
                for rank in sorted(set(nn_mode_data['test_rank_used']))
            },
            'videos_needing_fallback': sum(1 for r in nn_mode_data['test_rank_used'] if r > 1),
            'fallback_percentage': (sum(1 for r in nn_mode_data['test_rank_used'] if r > 1) / len(nn_mode_data['test_rank_used'])) * 100
        }
        
        metadata_file = os.path.join(output_dir, "nn_mode_metadata.json")
        with open(metadata_file, 'w') as fp:
            json.dump(metadata, fp, indent=4)
        
        print(f"\n=== NN Mode Results ===")
        print(f"Training indices saved to: {train_output_file} ({len(train_indices_dict)} videos)")
        print(f"Test indices saved to: {test_output_file} ({len(test_indices_dict)} videos)")
        print(f"Metadata saved to: {metadata_file}")
        
        # Verify uniqueness
        train_codes = set(str(code) for code in train_indices_dict.values())
        test_codes = set(str(code) for code in test_indices_dict.values())
        
        print(f"\nFinal NN Mode Analysis:")
        print(f"  Train unique codes: {len(train_codes)}/{len(train_indices_dict)}")
        print(f"  Test unique codes: {len(test_codes)}/{len(test_indices_dict)}")
        print(f"  Test codes are subset of train codes: {test_codes.issubset(train_codes)}")
        print(f"  Test reuses {len(test_codes)} unique train IDs out of {len(train_codes)} available")
        print(f"  Train ID reuse rate: {len(test_codes)/len(train_codes)*100:.1f}%")
        
    else:
        # Separate mode: save single output as before
        all_indices_dict = {}
        for item, indices in enumerate(all_indices):
            video_id = data.pairs[item]
            code = []
            for i, ind in enumerate(indices):
                code.append(prefix[i].format(int(ind)))
            all_indices_dict[video_id] = code
        
        with open(output_file, 'w') as fp:
            json.dump(all_indices_dict, fp, indent=4)
            
        print(f"\nSemantic indices successfully generated and saved to: {output_file}")
        print(f"Total videos processed: {len(all_indices_dict)}")
        print(f"Final collision rate: {(tot_item-tot_indice)/tot_item:.6f}")
        print(f"Unique semantic codes generated: {tot_indice}/{tot_item}") 

if __name__ == "__main__":
    main()
