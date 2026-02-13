"""
GPU MEMORY OPTIMIZATION GUIDE (MACHINE LEARNING)
-----------------------------------------------

1. MODEL WEIGHTS (The Static Load)
   * Standard Cost: ~2 bytes/parameter (FP16/BF16) or 4 bytes/parameter (FP32).
   * Optimization:
     - Quantization: Convert FP16 -> INT8 or INT4.
       ** Benefit: Reduces VRAM footprint by 50-75%.
     - Pruning: Remove "dead" or low-impact neurons.
       ** Benefit: Physically shrinks the model architecture.
     - Distillation: Train a smaller "student" model to mimic a "teacher."

2. IMAGE DATA & ACTIVATIONS (The Input Load)
   * Formula: (Height x Width x Channels) x Batch Size.
   * Activation Explosion: GPU stores "Feature Maps" for every layer during training.
   * Optimization:
     - Downscaling: Reducing 512x512 to 256x256 = 4x less memory (quadratic scaling).
     - Patching: Break high-res images into smaller "tiles" to process separately.
     - Lower Batch Size: The most direct way to clear activation memory.

3. TRAINING OVERHEAD (The Hidden Load)
   * Optimizer States: Adam stores momentum/variance for every weight.
     - Total Cost: Can reach ~10+ bytes per parameter (Weight + Grad + States).
   * Optimization:
     - 8-bit Optimizers: Use `bitsandbytes` AdamW (reduces state memory by 75%).
     - Gradient Checkpointing: Trades compute for memory by deleting/re-calculating 
       activations on the fly.
     - Mixed Precision (AMP): Performs math in FP16 while keeping a master copy in FP32.

4. QUICK OOM (OUT OF MEMORY) CHECKLIST
   [ ] Try Batch Size = 1 (to see if it's an activation/data issue).
   [ ] Enable 4-bit/8-bit Loading (to see if it's a weight issue).
   [ ] Use Gradient Accumulation (to simulate large batches with small VRAM).
   [ ] Clear Cache: `torch.cuda.empty_cache()` (removes "zombie" memory).
"""

# Example: Checking Image Tensor Memory Size
import torch

def check_tensor_mem(height, width, batch=1, dtype=torch.float32):
    # Standard FP32 is 4 bytes per element
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = height * width * 3 * batch * bytes_per_element
    print(f"Image Tensor ({height}x{width}) Memory: {total_bytes / 1e6:.2f} MB")


#Function that reports gpu memory
import gc

def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


# check_tensor_mem(1024, 1024, batch=8)