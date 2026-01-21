# train_ddp.py - DDP多卡训练版本
import torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_train_one import load_dataset
from gpt import GPT  
from config import *
from tqdm import tqdm
import os 
from tokenizers import Tokenizer
import math


def batch_proc(batch):
    """处理批次数据,添加padding和attention mask"""
    batch_x = []
    batch_chatml = []

    for ids, chatml in batch:
        batch_x.append(ids)
        batch_chatml.append(chatml)

    max_len = min(max(len(ids) for ids in batch_x), MAX_SEQ_LEN)
    
    padded_ids = []
    attention_masks = []

    for ids in batch_x:
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        padded_row = ids + [pad_id] * pad_len
        mask_row = [1] * len(ids) + [0] * pad_len
        padded_ids.append(padded_row)
        attention_masks.append(mask_row)

    batch_x_tensor = torch.tensor(padded_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.bool)

    return batch_x_tensor, attention_mask_tensor, batch_chatml


def generate_sample(model, tokenizer, prompt, device, max_tokens=50, temperature=0.8):
    """简单的生成函数"""
    model.eval()
    stop_ids = {pad_id, im_end_id}
    
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt).ids
        generated = input_ids.copy()
        
        for _ in range(max_tokens):
            current_input = torch.tensor([generated[-MAX_SEQ_LEN:]], dtype=torch.long).to(device)
            logits = model(current_input)
            next_token_logits = logits[0, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token in stop_ids:
                break
            
            generated.append(next_token)
        
        result = tokenizer.decode(generated)
    
    model.train()
    return result


def main():
    #初始化DDP
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank() #设置主gpu
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    
    # 只在rank 0打印信息（主GPU）
    if rank == 0:
        print(f"DDP Training with {world_size} GPUs")
        
    
    # 加载数据集和分词器
    if rank == 0:
        print("\nLoading dataset and tokenizer...")
        dataset = load_dataset()
        tokenizer = Tokenizer.from_file("tokenizer_hug.json")
    
    # 设置全局变量
    global pad_id, im_end_id
    pad_id = tokenizer.encode(PAD).ids[0]
    im_end_id = tokenizer.encode(IM_END).ids[0]
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocab size: {tokenizer.get_vocab_size()}")
        print(f"Pad ID: {pad_id}, IM_END ID: {im_end_id}")
    
    #初始化模型
    if rank == 0:
        print("\nInitializing model...")
    
    model = GPT(
        d_model=GPT_DIM,
        nhead=GPT_HEAD,
        feedforward=GPT_FF,
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    #加载checkpoint（如果存在）
    start_iter = 0
    best_loss = float('inf')
    loss_history = []
    checkpoint = None
    
    if os.path.exists('checkpoint.bin'):
        try:
            # rank 0 加载模型参数
            if rank == 0:
                model.load_state_dict(checkpoint['model'])
                start_iter = checkpoint['iter']
                best_loss = checkpoint.get('best_loss', float('inf'))
                loss_history = checkpoint.get('loss_history', [])
            
            if rank == 0:
                print(f"Resumed from iteration {start_iter}, best loss: {best_loss:.4f}")
        
        except Exception as e:
            if rank == 0:
                print(f"Failed to load checkpoint: {e}")
                print("Starting from scratch...")
            start_iter = 0
    
    #包装为DDP，这里主GPU会将模型广播到其他GPU上面
    model = DDP(model, device_ids=[rank])
    
    #设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),  
        weight_decay=0.01,
        eps=1e-8
    )
    
    # 加载optimizer状态
    if checkpoint and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                print("Loaded optimizer state from checkpoint")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load optimizer state: {e}")
    
    #学习率调度器
    def get_lr_multiplier(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, TRAIN_ITER - WARMUP_STEPS)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine_decay
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    if checkpoint and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            # 手动同步scheduler
            for _ in range(start_iter // ACCUM_STEPS):
                scheduler.step()
    
    #数据加载器（使用DistributedSampler）
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        persistent_workers=True,
        collate_fn=batch_proc,
        pin_memory=True
    )
    
    #训练循环
    accum_steps = ACCUM_STEPS
    log_interval = 100
    sample_interval = 1000
    save_interval = 1000
    
    if rank == 0:
        print(f"\nStarting training from iteration {start_iter}...")
        pbar = tqdm(total=TRAIN_ITER, initial=start_iter, desc="Training")
    
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    running_steps = 0
    
    # 计算需要的epoch数
    steps_per_epoch = len(dataloader)
    current_epoch = start_iter // steps_per_epoch
    
    for epoch in range(current_epoch, (TRAIN_ITER // steps_per_epoch) + 1):
        sampler.set_epoch(epoch)  # 重要：设置epoch以shuffle数据
        
        for batch_idx, (batch_ids, attention_mask, batch_chatml) in enumerate(dataloader):
            i = epoch * steps_per_epoch + batch_idx
            
            if i < start_iter:
                continue
            
            if i >= TRAIN_ITER:
                break
            
            batch_ids = batch_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            
            # 转换mask格式
            padding_mask = ~attention_mask
            
            # 前向传播
            logits = model(batch_ids, padding_mask)
            
            # 计算loss
            input_logits = logits[:, :-1, :].contiguous()
            targets = batch_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                input_logits.view(-1, input_logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id,
                label_smoothing=0
            )
            
            # 梯度累积
            loss = loss / accum_steps
            loss.backward()
            
            # 累积loss统计
            running_loss += loss.item() * accum_steps
            running_steps += 1
            
            # 梯度更新
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # 更新进度条（仅rank 0）
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * accum_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                pbar.update(1)
            
            # 打印详细信息（仅rank 0）
            if (i + 1) % log_interval == 0 and rank == 0:
                # 收集所有进程的loss
                loss_tensor = torch.tensor(running_loss / running_steps).to(device)
                dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / world_size
                
                loss_history.append(avg_loss)
                
                tqdm.write(f"\n[Step {i+1}/{TRAIN_ITER}] Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    tqdm.write(f"New best loss: {best_loss:.4f}")
                
                running_loss = 0.0
                running_steps = 0
            
            # 生成样本（仅rank 0）
            if (i + 1) % sample_interval == 0 and rank == 0:
                raw_model = model.module
                test_prompt = "user\n春天\n\nassistant\n"
                sample_text = generate_sample(raw_model, tokenizer, test_prompt, device, max_tokens=50)
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Sample at step {i+1}:")
                tqdm.write(f"{sample_text}")
                tqdm.write(f"{'='*60}\n")
            
            # 保存checkpoint（仅rank 0）
            if (i + 1) % save_interval == 0 and rank == 0:
                checkpoint_data = {
                    'iter': i + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'loss_history': loss_history[-100:],
                    'config': {
                        'd_model': GPT_DIM,
                        'nhead': GPT_HEAD,
                        'feedforward': GPT_FF,
                        'vocab_size': tokenizer.get_vocab_size(),
                        'max_seq_len': MAX_SEQ_LEN,
                        'learning_rate': LEARNING_RATE,
                        'train_iter': TRAIN_ITER
                    }
                }
                
                torch.save(checkpoint_data, 'checkpoint.bin.tmp')
                os.replace('checkpoint.bin.tmp', 'checkpoint.bin')
                
                # 保存最佳模型
                if len(loss_history) >= 10 and abs(best_loss - min(loss_history[-10:])) < 1e-6:
                    torch.save(checkpoint_data, 'best_model.bin')
                    tqdm.write(f"Saved best model at step {i+1}")
            
            # 等待rank 0 保存完成
            dist.barrier()
        
        if i >= TRAIN_ITER:
            break
    
    #训练结束
    if rank == 0:
        pbar.close()
        print("Training finished!")
        print(f"Best loss achieved: {best_loss:.4f}")

        # 最终保存
        final_checkpoint = {
            'iter': TRAIN_ITER,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'loss_history': loss_history,
            'config': {
                'd_model': GPT_DIM,
                'nhead': GPT_HEAD,
                'feedforward': GPT_FF,
                'vocab_size': tokenizer.get_vocab_size(),
                'max_seq_len': MAX_SEQ_LEN,
                'learning_rate': LEARNING_RATE,
                'train_iter': TRAIN_ITER
            }
        }
        torch.save(final_checkpoint, 'final_checkpoint.bin')
        print("Final model saved!")
    
    dist.destroy_process_group()


# 使用方法:
# torchrun --nproc-per-node 4 train_ddp.py  # 使用4张GPU
# 或
# torchrun --nproc-per-node 2 train_ddp.py  # 使用2张GPU
if __name__ == '__main__':
    main()