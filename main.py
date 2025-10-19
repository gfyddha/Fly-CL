import argparse
import time

import torch
import timm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from datasets.load_dataset import load_dataset
from models.load_model import load_model
from utils import random_initialization, feature_extract, target2onehot


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Input hyperparameters for the experiment.")

    # Continual Learning Task Setting
    parser.add_argument('--dataset', default='CIFAR-100', help='Choose dataset')
    parser.add_argument('--root', default='../data', help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=100, help='Total number of classes')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')

    # model Architecture
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help='model name')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension of pre-trained model')
    parser.add_argument('--expand_dim', type=int, default=10000, help='Expansion dimension of FlyModel')
    parser.add_argument('--synaptic_degree', type=int, default=100, help='Number of connections')
    parser.add_argument('--coding_level', type=float, default=0.01, help='Top-k number')

    # Training Configuration
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--ridge_lower', type=float, default=4, help='lower bound for ridge coefficient (log10)')
    parser.add_argument('--ridge_upper', type=float, default=10, help='lower bound for ridge coefficient (log10)')
    parser.add_argument('--data_augmentation', default=None, help='choose which normalization or not')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu')
    
    return parser


def select_ridge_parameter(Features, Y, ridge_lower, ridge_upper):
    X = Features
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_sq = S**2
    UTY = U.T @ Y
    ridges = torch.tensor(10.0 ** np.arange(ridge_lower, ridge_upper))
    n_samples = X.shape[0]
    
    gcv_scores = []
    for ridge in ridges:
        diag = S_sq / (S_sq + ridge)
        df = diag.sum()
        Y_hat = U @ (diag[:, None] * UTY)
        residual = torch.norm(Y - Y_hat)**2
        gcv = (residual / n_samples) / (1 - df / n_samples)**2
        gcv_scores.append(gcv.item())

    optimal_idx = np.argmin(gcv_scores)
    return ridges[optimal_idx]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda_available else "cpu")
    random_initialization(args.seed)

    if args.dataset == "CIFAR-100" or args.dataset == "CUB-200-2011" or args.dataset == "VTAB":
        print("Load and Split CIL Dataset...")
        train_loader, test_loader = load_dataset(args)
        print("Load and Split CIL Dataset Done")

    pretrained_model = load_model(args.model_name)
    pretrained_model.out_dim = args.embedding_dim
    pretrained_model.eval()
    pretrained_model.to(device)
    
    non_zero_per_col = args.synaptic_degree
    projection_matrix = torch.zeros(args.expand_dim, args.embedding_dim)
    for row in range(args.expand_dim):
        selected_cols = torch.randperm(args.embedding_dim)[:non_zero_per_col]
        projection_matrix[row, selected_cols] = torch.randn(non_zero_per_col)
    projection_matrix = projection_matrix.to(device).to_sparse_csc()

    acc = {}
    training_time = []
    feature_extract_time = []
    Q = torch.zeros(args.expand_dim, args.num_classes).to(device)
    G = torch.zeros(args.expand_dim, args.expand_dim).to(device)
    last_ridge = None
    print("Start Continual Learning")
    for task in range(args.num_tasks):
        acc[task] = []
        training_start = time.time()
        feature_extract_start = time.time()
        train_embeddings, train_labels = feature_extract(pretrained_model, train_loader[task], device)
        feature_extract_end = time.time()
        feature_extract_time.append(feature_extract_end - feature_extract_start)

        train_embeddings = torch.sparse.mm(projection_matrix, train_embeddings.T) # 10000, N
        values, indices = train_embeddings.topk(int(args.expand_dim * args.coding_level), dim=0, largest=True)
        output = torch.zeros_like(train_embeddings)
        output.scatter_(0, indices, values)
        train_embeddings = output

        Y = target2onehot(train_labels, args.num_classes)
        Q = Q + train_embeddings @ Y
        G = G + train_embeddings @ train_embeddings.T
        ridge = select_ridge_parameter(train_embeddings.T, Y, args.ridge_lower, args.ridge_upper)
        L = torch.linalg.cholesky(G + ridge * torch.eye(G.size(dim=0)).to(device)) # 40% faster
        Wo = torch.cholesky_solve(Q, L)
        training_end = time.time()
        training_time.append(training_end - training_start)

        for sub_task in range(task + 1):
            test_embeddings, test_labels = feature_extract(pretrained_model, test_loader[sub_task], device)
            test_embeddings = torch.sparse.mm(projection_matrix, test_embeddings.T)
            values, indices = test_embeddings.topk(int(args.expand_dim * args.coding_level), dim=0, largest=True)
            output = torch.zeros_like(test_embeddings)
            output.scatter_(0, indices, values)
            test_embeddings = output.T.to_sparse_csc()
            output = torch.sparse.mm(test_embeddings, Wo)        
            predicts = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].squeeze()
            test_accuracy = np.mean(predicts.cpu().numpy() == test_labels.cpu().numpy()) * 100
            acc[sub_task].append(test_accuracy)

    # display acc_matrix
    acc_matrix = [["{:.2f}".format(0.00) for _ in range(args.num_tasks)] for _ in range(len(acc))]
    for i, (task, values) in enumerate(acc.items()):
        for j, value in enumerate(values):
            acc_matrix[i][i + j] = round(value, 2)
    
    print("Accuracy Matrix")
    for row in acc_matrix:
        print(row)
    print()

    print("Average Accuracy")
    A_t = []
    for j in range(args.num_tasks):
        cnt = 0.0
        for i in range(j + 1):
            cnt += acc_matrix[i][j]
        cnt /= (j + 1)
        A_t.append(cnt)
        print(round(cnt, 2), end=", ")
    print("\n")

    print("Accumulated Accuracy")
    print(round(np.mean(A_t), 2))
    print()

    print("Training Time")
    for task_time in training_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Training Time")
    print(round(np.mean(training_time), 2))
    print()

    print("Feature Extract Time")
    for task_time in feature_extract_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Feature Extract Time")
    print(round(np.mean(feature_extract_time), 2))
    print()