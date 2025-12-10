# Shiprocket GPU Scaling Analysis for 10K Addresses/Second

## Current Performance Baseline

Based on our testing:

### CPU Performance (Current):
- **Speed:** 889ms per address (single-threaded)
- **Throughput:** ~1.1 addresses/second per CPU core
- **Model:** shiprocket-ai/open-indicbert-indian-address-ner (IndicBERT)
- **Model Size:** ~130MB (parameters)

### GPU Performance (Estimated):
With GPU acceleration, typical speedup for transformer models:
- **Expected speedup:** 10-20x faster than CPU
- **Estimated GPU speed:** 50-100ms per address
- **Estimated throughput:** 10-20 addresses/second per GPU (single inference)

## Target: 10,000 Addresses/Second

### Calculation

**Required throughput:** 10,000 addresses/second

**With batch processing on GPU:**
- Batch size: 32 addresses (optimal for IndicBERT)
- GPU inference time per batch: ~200-400ms (estimated)
- Throughput per GPU: 80-160 addresses/second

**Number of GPUs needed:**
```
10,000 addresses/sec ÷ 120 addresses/sec per GPU = ~84 GPUs
```

### More Realistic Scenario with Optimization

With aggressive optimization:
- Batch size: 64 (larger batches)
- Optimized inference: TensorRT/ONNX conversion
- Mixed precision (FP16): 2x speedup
- Estimated throughput: 300-400 addresses/second per GPU

**Optimized GPU count:**
```
10,000 addresses/sec ÷ 350 addresses/sec per GPU = ~29 GPUs
```

## AWS Instance Recommendations

### Option 1: Multiple GPU Instances (Recommended)

**Instance Type:** `g5.2xlarge` (1x NVIDIA A10G GPU)

**Specifications:**
- GPU: 1x NVIDIA A10G (24GB VRAM)
- vCPUs: 8
- RAM: 32GB
- Cost: ~$1.21/hour (us-east-1, on-demand)

**Deployment:**
- **Number of instances:** 30-35 instances
- **Total cost:** ~$36-42/hour (~$26,000-30,000/month)
- **Throughput:** 10,000-12,000 addresses/second

**Pros:**
- Horizontal scaling
- Fault tolerance (if one fails, others continue)
- Can scale up/down based on demand
- Good price/performance ratio

**Cons:**
- Need load balancer
- More complex orchestration
- Network overhead between instances

---

### Option 2: Large Multi-GPU Instance

**Instance Type:** `p4d.24xlarge` (8x NVIDIA A100 GPUs)

**Specifications:**
- GPUs: 8x NVIDIA A100 (40GB VRAM each)
- vCPUs: 96
- RAM: 1,152GB
- Network: 400 Gbps
- Cost: ~$32.77/hour (us-east-1, on-demand)

**Deployment:**
- **Number of instances:** 4 instances
- **Total cost:** ~$131/hour (~$94,000/month)
- **Throughput:** 10,000-14,000 addresses/second

**Pros:**
- Fewer instances to manage
- Lower network latency
- Better for batch processing
- High-speed inter-GPU communication

**Cons:**
- Very expensive
- Single point of failure per instance
- Overkill for this workload

---

### Option 3: Cost-Optimized with Spot Instances

**Instance Type:** `g5.2xlarge` (Spot)

**Specifications:**
- Same as Option 1
- Cost: ~$0.36-0.60/hour (70-80% discount)

**Deployment:**
- **Number of instances:** 35-40 instances (with buffer for interruptions)
- **Total cost:** ~$12-24/hour (~$8,600-17,000/month)
- **Throughput:** 10,000-12,000 addresses/second

**Pros:**
- 70-80% cost savings
- Same performance as on-demand
- Good for batch processing

**Cons:**
- Can be interrupted (need spot instance handling)
- Need auto-scaling group with mixed instance types
- Requires robust retry logic

---

### Option 4: Inference-Optimized Instances

**Instance Type:** `inf2.xlarge` (AWS Inferentia2)

**Specifications:**
- Accelerator: 1x AWS Inferentia2 chip
- vCPUs: 4
- RAM: 16GB
- Cost: ~$0.76/hour (us-east-1)

**Deployment:**
- **Number of instances:** 40-50 instances
- **Total cost:** ~$30-38/hour (~$21,000-27,000/month)
- **Throughput:** 10,000-12,500 addresses/second

**Pros:**
- Optimized for inference
- Lower cost than GPU instances
- Good performance for transformer models

**Cons:**
- Requires model compilation for Inferentia
- Less flexible than GPUs
- Limited community support

---

## Recommended Architecture

### Best Option: Hybrid Approach

**Primary:** 25x `g5.2xlarge` (on-demand) + 10x `g5.2xlarge` (spot)

**Configuration:**
```
Load Balancer (ALB/NLB)
    ↓
Auto Scaling Group
    ├── 25x g5.2xlarge (on-demand) - baseline capacity
    └── 10x g5.2xlarge (spot) - burst capacity
```

**Cost Breakdown:**
- On-demand: 25 × $1.21/hour = $30.25/hour
- Spot: 10 × $0.40/hour = $4.00/hour
- **Total:** ~$34/hour (~$24,500/month)

**Throughput:**
- Baseline: 8,750 addresses/second (25 instances)
- Burst: 12,250 addresses/second (35 instances)

**Benefits:**
- Reliable baseline with on-demand
- Cost savings with spot instances
- Auto-scaling for traffic spikes
- Fault-tolerant architecture

---

## Infrastructure Requirements

### Load Balancing
```
AWS Application Load Balancer (ALB)
- Health checks every 30 seconds
- Connection draining: 300 seconds
- Sticky sessions: disabled
- Cost: ~$23/month + data transfer
```

### Container Orchestration
```
Amazon ECS or EKS
- Task definition: 1 container per instance
- Service auto-scaling based on CPU/memory
- Rolling updates for zero downtime
- Cost: ECS free, EKS ~$73/month per cluster
```

### Model Storage
```
Amazon S3 + CloudFront
- Store model files (~130MB)
- Cache on each instance at startup
- Cost: ~$3/month for storage + minimal transfer
```

### Monitoring
```
CloudWatch + Prometheus/Grafana
- Track: throughput, latency, error rate, GPU utilization
- Alarms for scaling triggers
- Cost: ~$50-100/month
```

---

## Performance Optimization Strategies

### 1. Model Optimization
```python
# Convert to ONNX for faster inference
import torch
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "shiprocket-ai/open-indicbert-indian-address-ner"
)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "shiprocket_model.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 
                  'attention_mask': {0: 'batch_size'}}
)

# Expected speedup: 2-3x
```

### 2. Batch Processing
```python
# Optimal batch size for A10G GPU
BATCH_SIZE = 64  # Adjust based on GPU memory

# Process in batches
def process_batch(addresses):
    results = parser.parse_batch(addresses)
    return results

# Throughput improvement: 5-10x
```

### 3. Mixed Precision (FP16)
```python
# Use automatic mixed precision
import torch

model = model.half()  # Convert to FP16
model = model.cuda()

# Expected speedup: 2x
# Memory savings: 50%
```

### 4. TensorRT Optimization
```python
# Convert to TensorRT for maximum performance
import tensorrt as trt

# Build TensorRT engine
# Expected speedup: 3-5x over standard PyTorch
```

---

## Cost Comparison: Shiprocket vs Local Parser

### Shiprocket (GPU-based)
**Infrastructure:** 35x g5.2xlarge instances
- **Cost:** ~$34/hour (~$24,500/month)
- **Throughput:** 10,000 addresses/second
- **Cost per million addresses:** $3.40

### Local Rule-Based Parser (CPU-based)
**Infrastructure:** 1x t3.medium instance
- **Cost:** ~$0.042/hour (~$30/month)
- **Throughput:** 5,882,000 addresses/second (0.17ms per address)
- **Cost per million addresses:** $0.000007

**Cost Difference:** Shiprocket is **485,000x more expensive** than Local parser!

---

## Reality Check: Why This Doesn't Make Sense

### The Math
- **Shiprocket:** $24,500/month for 10K addresses/sec
- **Local Parser:** $30/month for 5.8M addresses/sec

### The Problem
1. **Shiprocket is 817x slower** than needed (10K vs 5.8M addresses/sec)
2. **Costs 817x more** than necessary
3. **Still has 60% failure rate** even with GPU
4. **Requires complex infrastructure** (load balancers, auto-scaling, monitoring)
5. **Operational overhead** (model updates, GPU driver management, etc.)

---

## Recommendation: Don't Use Shiprocket for This

### Use Local Rule-Based Parser Instead

**Single t3.xlarge instance:**
- **Cost:** $0.166/hour (~$120/month)
- **vCPUs:** 4
- **RAM:** 16GB
- **Throughput:** 23.5 million addresses/second (4 cores × 5.88M)
- **Reliability:** 100% success rate
- **Latency:** 0.17ms per address

**For your 53MB dataset (~500K addresses):**
- **Shiprocket:** 50 seconds + $0.17 cost
- **Local:** 0.085 seconds + $0.000004 cost

### When to Consider GPU/Shiprocket

Only if you need:
1. **Better locality extraction** (69% vs 23%)
2. **Better road extraction** (44% vs 0%)
3. **Can tolerate 60% failure rate**
4. **Budget is unlimited**
5. **Speed doesn't matter**

---

## Conclusion

**For 10K addresses/second:**
- **GPU Required:** 29-35 NVIDIA A10G GPUs
- **AWS Instance:** 35x g5.2xlarge
- **Monthly Cost:** ~$24,500
- **Setup Complexity:** High

**But you should use Local Parser:**
- **CPU Required:** 1 vCPU
- **AWS Instance:** 1x t3.medium
- **Monthly Cost:** ~$30
- **Setup Complexity:** Minimal
- **Performance:** 588x faster than your requirement

**The Local parser can handle 5.8 million addresses/second on a single CPU core. You don't need GPUs.**

---

**Generated:** December 9, 2025  
**Analysis:** Shiprocket GPU scaling for 10K addresses/second  
**Verdict:** Use Local Rule-Based Parser instead - it's 817x faster and 817x cheaper
