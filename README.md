# HAFM
HAFM: Hierarchical Autoregressive Foundation Model for Music Accompaniment Generation
> **Authors:**
Jian Zhu, Jianwei Cui, Shihao Chen, Yubang Zhang, Cheng Luo.

This repo contains the code and data of [Hierarchical Autoregressive Foundation Model](https://arxiv.org/abs/2604.09054).

## 1. Abstract
We present HAFM, a system that generates instrumental music audio to accompany input vocals. Given isolated singing voice, HAFM produces a coherent instrumental accompaniment that can be directly mixed with the input to create complete music. We propose three key innovations over prior work: (1) a dual-rate codec tokenization scheme using HuBERT semantic tokens at 50\,Hz for vocals and EnCodec acoustic tokens at 75\,Hz for instrumentals, enabling time-aligned yet rate-independent modeling; (2) a three-stage hierarchical autoregressive architecture (semantic $\rightarrow$ coarse acoustic $\rightarrow$ fine acoustic) with interleaved multi-codebook prediction and classifier-free guidance; and (3) modern Transformer design choices including QK-norm, GEGLU activations, RMSNorm, and T5-style relative position bias for improved training stability and sequence generalization. Experiments on MUSDB18 demonstrate that HAFM achieves a Fr\'{e}chet Audio Distance (FAD) of 2.08 on isolated vocal inputs, outperforming retrieval baselines and matching prior state-of-the-art systems with fewer parameters. The source code is available at https://github.com/HackerHyper/HAFM.

## 2. ARCH
<div align="center">
  <img src="https://github.com/HackerHyper/HAFM/blob/main/ARCH.jpg">
</div>

## 3. Demo
[![播放音频](https://img.shields.io/badge/点击播放-音频-blue?style=for-the-badge&logo=googlemusic)](https://raw.githubusercontent.com/HackerHyper/HAFM/main/accomp_3s_instrumental.wav)

If you have any problems, contact me via qijian.zhu@outlook.com.
