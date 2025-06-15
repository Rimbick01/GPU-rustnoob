# GPU-rustnoob
# GPU Computing with OpenCL in Rust: For Masochists Who Enjoy Pain 🚀🔥

*Congratulations!* You've decided to burn your CPU's feelings by cheating on it with a GPU. This guide will help you write code that runs 1000x faster (or crashes 1000x more spectacularly). Let's add two whole arrays together like it's rocket science.

## Prerequisites (a.k.a. "The Setup for Failure")

- **A GPU that isn't a literal toaster**: Install vendor drivers. NVIDIA users get to download CUDA (5GB of "essential" bloatware). AMD folks get to Google "why is ROCm not working" for 6 hours. Intel users... bless your heart.
  

## Setup: The Part Where You Question Life Choices

1. **Summon a new Rust project** (your computer is already judging you):
   ```bash
   cargo new midlife_crisis && cd midlife_crisis

[dependencies]
ocl = { version = "0.19", features = ["ocl-libc"] }  # "libc" stands for "library of cosmic despair"

// Behold! The pinnacle of human achievement:
__kernel void vec_add(
    __global const float* a,  // The 'a' stands for "anguish"
    __global const float* b,  // The 'b' stands for "why am I doing this"
    __global float* result    // Where hope goes to die
) {
    int idx = get_global_id(0);  // Magic spell to summon parallel demons
    result[idx] = a[idx] + b[idx];  // WITCHCRAFT
}
