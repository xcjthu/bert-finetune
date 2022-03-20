import time
import argparse
import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertModel as Bert
from transformers import BertTokenizer
from HFBertDataset import DATASET
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def add_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    parser.add_argument('--model-version', type=str, default=None,
                       help='Model Version')
    parser.add_argument('--base-path', type=str, default=None,
                       help='Path to the project base directory.')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Name of the dataset')
    parser.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    parser.add_argument('--save-name', type=str, default=None,
                       help='Output filename to save checkpoints to.')
    parser.add_argument('--save-iters', type=int, default=1000,
                       help='number of iterations between saves')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    parser.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    parser.add_argument('--max-length', type=int, default=512,
                       help='max length of input')
    parser.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=1,
                       help='total number of epochs to train over all training runs')
    parser.add_argument('--grad-accumulation', type=int, default=1,
                       help='grad accumulation')

    # Learning rate.
    parser.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')

    parser.add_argument('--warmup-iters', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    parser.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    parser.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args

class BertModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert : Bert = Bert.from_pretrained(args.model_version)
        dim_model = self.bert.config.hidden_size
        self.dense = nn.Linear(dim_model, 2)

    def forward(self, input_ids, attention_mask):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_version)
    return tokenizer

def get_model(args):
    model = BertModel(args).cuda()
    return nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()])

def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    # return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters, num_training_steps=args.train_iters)
    return None

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_length)
    return dataset

def prepare_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

def sum_loss(loss):
    global_loss = loss.clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    return global_loss / dist.get_world_size()

def print_rank(s):
    if dist.get_rank() == 0:
        print(s)

def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(10):
        dataloader = {
            "train": prepare_dataloader(dataset['train'], batch_size=args.batch_size),
            "dev": prepare_dataloader(dataset['dev'], batch_size=args.batch_size),
        }

        model.train()
        for it, data in enumerate(dataloader['train']):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            torch.cuda.synchronize()
            st_time = time.time()

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            pd = logits.argmax(dim=-1).cpu().tolist()
            gt = labels.cpu().tolist()

            global_loss = sum_loss(loss).item()

            # loss = optimizer.loss_scale(loss)
            loss = loss / args.grad_accumulation
            loss.backward()
            if (it + 1) % args.grad_accumulation == 0:
                optimizer.step()
                # lr_scheduler.step()

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e} | time: {:.3f} | acc: {:.4f}".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_loss,
                    args.lr,
                    elapsed_time,
                    accuracy_score(gt, pd),
                )
            )

        model.eval()
        with torch.no_grad():
            for split in ['dev']:
                pd = []
                gt = []
                for it, data in enumerate(dataloader[split]):
                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]
                    labels = data["labels"]

                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

                    logits = logits.argmax(dim=-1)
                    pd.extend(logits.cpu().tolist())
                    gt.extend(labels.cpu().tolist())

                    print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f}".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                            loss,
                        )
                    )

                print_rank(f"{split} epoch {epoch}:")
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    print_rank(f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    f1 = f1_score(gt, pd, average="macro")
                    print_rank(f"Average F1: {f1*100:.2f}")


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/superglue/",
        args.dataset_name,
        dist.get_rank(),dist.get_world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
