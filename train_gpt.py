import torch 
from dataset_train_one import load_dataset
from gpt import GPT  
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import *
from tqdm import tqdm
import os 
from tokenizers import Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 加载数据集和分词器
dataset = load_dataset() 
tokenizer = Tokenizer.from_file("tokenizer_hug.json")
print(f"Dataset size: {len(dataset)}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# 获取特殊 token ID
pad_id = tokenizer.encode(PAD).ids[0]  
im_end_id = tokenizer.encode(IM_END).ids[0]


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


def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """简单的生成函数"""
    model.eval()
    stop_ids = {pad_id, im_end_id}
    
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt).ids
        generated = input_ids.copy()
        
        for _ in range(max_tokens):
            current_input = torch.tensor([generated[-MAX_SEQ_LEN:]], dtype=torch.long).to(DEVICE)
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


if __name__ == '__main__':
    print("\nInitializing model...")
    model = GPT(
        d_model=GPT_DIM,
        nhead=GPT_HEAD,
        feedforward=GPT_FF,
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),  
        weight_decay=0.01,
        eps=1e-8
    )
    
    # 简化的学习率调度器：使用LambdaLR实现warmup + cosine
    def get_lr_multiplier(step):
        """获取学习率倍数"""
        # Warmup阶段
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        
        # Cosine annealing阶段
        progress = (step - WARMUP_STEPS) / (TRAIN_ITER - WARMUP_STEPS)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine_decay  # 从1.0衰减到0.1
    
    import math
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
      
    # 开始训练
    start_iter = 0
    best_loss = float('inf')
    loss_history = []
    # 梯度累积设置 
    accum_steps = 8  
    # 加载checkpoint（如果存在）
    if os.path.exists('checkpoint.bin'):
        try:
            print("\nLoading checkpoint...")
            checkpoint = torch.load('checkpoint.bin', map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            
            # 获取checkpoint中保存的学习率
            checkpoint_lr = checkpoint.get('config', {}).get('learning_rate', None)
            
            # 只在checkpoint中有optimizer状态时才加载
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("Loaded optimizer state from checkpoint")
                    
                    # === 关键：学习率调整逻辑 ===
                    current_lr_in_optimizer = optimizer.param_groups[0]['lr']
                    
                    # 如果config中的LEARNING_RATE与checkpoint不一致
                    if checkpoint_lr is not None and abs(LEARNING_RATE - checkpoint_lr) > 1e-9:
                        # 策略选择（可以根据需要修改）
                        FORCE_CONFIG_LR = True  # 设为False则保持checkpoint的学习率
                        
                        if FORCE_CONFIG_LR:
                            # 强制使用config中的学习率作为基础学习率
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = LEARNING_RATE
                            print(f"已强制应用config学习率: {LEARNING_RATE:.2e}")
                        else:
                            print("保持checkpoint中的学习率")
                    else:
                        print(f"学习率一致: {LEARNING_RATE:.2e}")
                    
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
                    print("Starting with fresh optimizer state")
            else:
                print("No optimizer state in checkpoint, using fresh optimizer")
            
            start_iter = checkpoint['iter']
            best_loss = checkpoint.get('best_loss', float('inf'))
            
            # 只在checkpoint中有scheduler状态且与当前scheduler兼容时才加载
            if 'scheduler' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print("Loaded scheduler state from checkpoint")
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
                    print("Manually stepping scheduler to sync with current iteration")
                    # 手动step scheduler到正确的步数
                    for _ in range(start_iter // accum_steps):
                        scheduler.step()
            else:
                print("ℹNo scheduler state in checkpoint")
                print("Manually stepping scheduler to sync with current iteration")
                # 手动step scheduler到正确的步数
                for _ in range(start_iter // accum_steps):
                    scheduler.step()
            
            loss_history = checkpoint.get('loss_history', [])
            
            # 显示最终的学习率状态
            final_lr = optimizer.param_groups[0]['lr']     
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")
            start_iter = 0
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=batch_proc,
        pin_memory=True
    )
    
    data_iter = iter(dataloader)
    pbar = tqdm(total=TRAIN_ITER, initial=start_iter, desc="Training")
    
    
    optimizer.zero_grad()
    
    # 训练统计
    running_loss = 0.0
    running_steps = 0
    log_interval = 100
    sample_interval = 1000
    save_interval = 1000
    
    model.train()
    
    for i in range(start_iter, TRAIN_ITER):
        try:
            batch_ids, attention_mask, batch_chatml = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_ids, attention_mask, batch_chatml = next(data_iter)
        
        batch_ids = batch_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        
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
            label_smoothing=0  # 添加label smoothing防止过拟合，loss降不下去可以去掉
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
            scheduler.step()  # 每步更新学习率
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item() * accum_steps:.4f}",
            'lr': f"{current_lr:.2e}"
        })
        pbar.update(1)
        
        # 打印详细信息
        if (i + 1) % log_interval == 0:
            if running_steps > 0:
                avg_loss = running_loss / running_steps
                loss_history.append(avg_loss)
                
                tqdm.write(f"\n[Step {i+1}/{TRAIN_ITER}] Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    tqdm.write(f"New best loss: {best_loss:.4f}")
            
            running_loss = 0.0
            running_steps = 0
        
        # 生成样本
        if (i + 1) % sample_interval == 0:
            test_prompt = "user\n一只鸟在想方向\n\nassistant\n亟欲飞翔却张不开翅膀"
            sample_text = generate_sample(model, tokenizer, test_prompt, max_tokens=50)
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Sample at step {i+1}:")
            tqdm.write(f"{sample_text}")
            tqdm.write(f"{'='*60}\n")
        
        # 保存checkpoint
        if (i + 1) % save_interval == 0:
            checkpoint = {
                'iter': i + 1,
                'model': model.state_dict(),
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
            
            torch.save(checkpoint, 'checkpoint.bin.tmp')
            os.replace('checkpoint.bin.tmp', 'checkpoint.bin')
            
            if abs(best_loss - min(loss_history[-10:] if len(loss_history) >= 10 else [float('inf')])) < 1e-6:
                torch.save(checkpoint, 'best_model.bin')
                tqdm.write(f"Saved best model at step {i+1}")
    
    pbar.close()
    
    print("\n" + "="*60)
    print("Training finished!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print("="*60)
    
    # 最终保存
    final_checkpoint = {
        'iter': TRAIN_ITER,
        'model': model.state_dict(),
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