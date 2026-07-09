# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""LingBot-VA package.

Subpackages are intentionally not imported eagerly. Dataset metadata tools do not
need CUDA model kernels, and importing them should not require FlashAttention.
"""
