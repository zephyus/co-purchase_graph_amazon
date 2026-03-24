# Chapter 27: Multi-GPU & Hyperparameters

## 1. The Intuition (引言與核心靈魂)
Imagine trying to paint a massive mural by yourself. It would take weeks. Now imagine hiring 8 painters. If they don't communicate, they will paint over each other's work or use different color palettes. But if they share a master blueprint and synchronize every hour, the mural is completed perfectly in a day. 

In Deep Learning, a single GPU can take days to process industrial datasets. Multi-GPU training (Distributed Data Parallel) divides the dataset across multiple GPUs. However, synchronizing the gradients (the strokes of the painters) requires extreme engineering precision. Furthermore, selecting the right Hyperparameters—learning rate, batch size, dropout—is like choosing the correct thickness of the brushes. 

**Learning Objectives:**
1. Understand the theoretical foundation of PyTorch's Distributed Data Parallel (DDP).
2. Master Gradient Accumulation to simulate massive batch sizes.
3. Decipher learning rate scheduling (Cosine Annealing) mathematically.

## 2. Deep Dive (核心概念與深度解析)
**Distributed Data Parallel (DDP)**
In DDP, each GPU hosts an identical copy of the model $\mathbf{W}$. The dataset $\mathcal{D}$ is partitioned into $K$ non-overlapping subsets $\mathcal{D}_1, \dots, \mathcal{D}_K$ across $K$ GPUs.
In every forward pass, GPU $k$ computes the loss $L_k$ on its local batch. In the backward pass, each GPU computes local gradients:
$$ \mathbf{g}_k = \nabla_{\mathbf{W}} L_k $$
Before the optimizer updates the weights, PyTorch executes an **All-Reduce** operation over the fast NVLink interconnect, averaging the gradients across all GPUs:
$$ \mathbf{g}_{global} = \frac{1}{K} \sum_{k=1}^K \mathbf{g}_k $$
Every GPU then applies $\mathbf{g}_{global}$, ensuring strict synchronization of weights $\mathbf{W}$.

**Learning Rate Scheduling**
A constant learning rate $\eta$ is sub-optimal. Cosine Annealing drops $\eta$ smoothly following a cosine curve, helping the model escape local minima early on, and settle cleanly into the global minimum at the end:
$$ \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right) $$

**Common Misconceptions:**
- *DataParallel (DP) is just as good as DistributedDataParallel (DDP).* False. DP relies on a single Python thread (due to the GIL) and repeatedly copies the model across GPUs every forward pass. It is horrifically inefficient. DDP uses multiprocessing.
- *Doubling GPUs means I don't need to change my learning rate.* False. If you use 4 GPUs, your effective batch size is $4 \times B$. Under the Linear Scaling Rule, you must multiply your learning rate by 4 to compensate.

## 3. Code & Engineering (程式碼實作與工程解密)
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank: int, world_size: int):
    """Initializes distributed process group."""
    dist.init_process_group(
        backend='nccl',          # NCCL is optimized for NVIDIA GPUs
        init_method='env://', 
        world_size=world_size, 
        rank=rank
    )
    torch.cuda.set_device(rank)

def train_worker(rank: int, world_size: int, model: 'nn.Module'):
    setup_distributed(rank, world_size)
    
    # Move model to local GPU
    model = model.to(rank)
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * world_size) # Linear scaling rule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Gradient Accumulation simulates batch size of 1024 even if we only fit 256
    accumulation_steps = 4 
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            batch = batch.to(rank)
            loss = model(batch) / accumulation_steps # Normalize loss
            
            # Gradients automatically all-reduce here due to DDP hooks
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        scheduler.step()
        
    dist.destroy_process_group()
```
*Engineering Note:* When combining DDP and Gradient Accumulation, DDP defaults to syncing gradients on *every* `.backward()` call. This wastes massive network bandwidth. Advanced implementations use `model.no_sync()` context managers during the first `accumulation_steps - 1` backward passes.

## 4. MIT-Level Exercises (課後思考與魔王挑戰)
1. **Conceptual Validation:** If you train a GNN with DDP, explain why Graph Partitioning (e.g., METIS, ClusterGCN) is strictly necessary to avoid massive communication overhead compared to standard Vision models (CNNs).
2. **Extreme Edge-Case:** You scale training to 128 A100 GPUs, driving your effective batch size to 2,000,000 nodes. Your loss immediately NaNs on epoch 1. Using the concepts of gradient variance and the Linear Scaling rule, diagnose why this happened and propose a warmup strategy to fix it.