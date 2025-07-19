import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from model import ModelConfig, MyLLaMA
from typing import Tuple
from functools import partial
import wandb
from drive_managment import authorize_gdrive, empty_gdrive_trash


def save_checkpoint(
    total_tokens_processed: int,
    step_count: int,
    model: MyLLaMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    checkpoint_path: Path,
    task_type: str,
    drive
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    for file in checkpoint_path.glob(f"{task_type}_*.pt"):
        file.unlink()

    checkpoint = {
        'total_tokens_processed': total_tokens_processed,
        'step_count': step_count,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }

    million_tokens = total_tokens_processed // 1_000_000
    filename = checkpoint_path / f"{task_type}_{million_tokens}M.pt"
    torch.save(checkpoint, filename)

    empty_gdrive_trash(drive)

def load_checkpoint(
    model: MyLLaMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    checkpoint_path: Path,
    task_type: str
) -> Tuple[int, int]:
    checkpoint_files = sorted(checkpoint_path.glob(f"{task_type}_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError("checkpoint doesn't exist")
    filename = checkpoint_files[-1]
    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    return (
        checkpoint['total_tokens_processed'],
        checkpoint['step_count']
    )


@dataclass
class TrainConfig:
    total_target_tokens: int = 250_000_000
    bs_target_tokens: int = 250_000
    bs_sentence: int = 2
    max_batch_size: int = 20
    checkpoint_interval: int = 5_000_000
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eta_min: float = 1e-5


def compute_grad_norm(model: MyLLaMA) -> float:
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    return float(np.mean(norms)) if norms else 0.0


def get_warmup_lr(current_step: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return 0.1 + 0.9 * float(current_step / warmup_steps)
    else:
        return 1.0


def train_model(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    model: MyLLaMA,
    loader: DataLoader,
    train_name: str,
    checkpoint_path: Path,
    task_type: str
) -> None:
    wandb.init(
        project=train_name,
        config={
            "context_len": model_cfg.context_len,
            "d_model": model_cfg.d_model,
            "n_heads": model_cfg.n_heads,
            "n_layers": model_cfg.n_layers,
            "bs_target_tokens": train_cfg.bs_target_tokens
        }
    )

    drive = authorize_gdrive()

    device = model_cfg.device
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay
    )
    total_steps = train_cfg.total_target_tokens // train_cfg.bs_target_tokens
    if task_type == 'finetune':
        warmup_steps = total_steps // 50
        scheduler = LambdaLR(
            optimizer,
            partial(
                get_warmup_lr,
                warmup_steps=warmup_steps
            )
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=train_cfg.eta_min
        )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    total_token_processed = 0
    accumulated_tokens = 0
    step_count = 0
    tokens_since_last_checkpoint = 0
    loss_accumulator = 0.0
    sentences_processed = 0

    try:
        total_token_processed, step_count = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, task_type
        )
        print(f"Resumed from checkpoint: {total_token_processed:,} tokens, step {step_count}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")

    model.train()

    pbar = tqdm(
        total=train_cfg.total_target_tokens, 
        desc="Training", 
        unit="token",
        initial=total_token_processed
    )

    while total_token_processed < train_cfg.total_target_tokens:
        for batch in loader:
            if batch is None:
                continue
            x = batch.to(device)
            num_batch_tokens = x.numel() - x.shape[0]

            with torch.autocast(device_type=device, dtype=torch.float16):
                output = model(x[:, :-1])
                output_true = x[:, 1:]
                loss = criterion(output.permute(0, 2, 1), output_true)

            scaled_loss = loss * (num_batch_tokens / train_cfg.bs_target_tokens)
            scaler.scale(scaled_loss).backward()
            loss_accumulator += loss.item() * num_batch_tokens
            tokens_since_last_checkpoint += num_batch_tokens
            accumulated_tokens += num_batch_tokens
            total_token_processed += num_batch_tokens
            sentences_processed += train_cfg.bs_sentence

            pbar.update(num_batch_tokens)
            pbar.set_description(
                f"Tokens: {total_token_processed:,}/{train_cfg.total_target_tokens:,} | Sentences: {sentences_processed:,}/1,000,000"
            )

            if accumulated_tokens >= train_cfg.bs_target_tokens:
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                avg_loss = loss_accumulator / accumulated_tokens
                wandb.log({
                    "loss": avg_loss,
                    "grad_norm": grad_norm,
                    "batch_tokens": accumulated_tokens,
                    "processed_tokens": total_token_processed,
                    "lr": optimizer.param_groups[0]['lr'],
                    "step": step_count
                })

                step_count += 1
                accumulated_tokens = 0
                loss_accumulator = 0
                del x, output, output_true, loss, scaled_loss
                torch.cuda.empty_cache()
            
            if tokens_since_last_checkpoint >= train_cfg.checkpoint_interval:
                save_checkpoint(
                    total_token_processed,
                    step_count,
                    model,
                    optimizer,
                    scheduler,
                    checkpoint_path,
                    task_type,
                    drive
                )
                tokens_since_last_checkpoint = 0
    pbar.close()
    wandb.finish()
