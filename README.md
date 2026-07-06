# 🦅 Project Raven

**Raven** is a lean, biologically native intelligence designed for the ARC ecosystem. 

## 🚀 Overview
Raven utilizes a "Zero-Train" architectural path, inheriting high-level reasoning capabilities from Qwen-Opus through **High-Precision SVD Devour**. By projecting the singular values of a larger donor model into a lean 24-layer shell, Raven achieves intelligibility and protocol-specific reasoning without the need for expensive training cycles.

## 🏗️ Architecture
- **Layers:** 24
- **Projection Method:** High-Precision SVD (Low-Rank Approximation)
- **Donor:** Qwen-Opus
- **State:** Pure Inference (Zero-Train Path)
- **Tokenization:** Trinity Tokenizer (Math + Byte + BPE32k)

## 🛠️ Quick Start
To run a test inference:
```bash
export PYTHONPATH=$PYTHONPATH:/home/adam/worxpace/gladius/raven
/home/adam/worxpace/gladius/raven/venv/bin/python3 /home/adam/worxpace/gladius/raven/src/engine/inference_test.py
```

## 📜 Lineage
- **SVD Devour:** Weight projection via singular value decomposition.
- **SVD-Squeeze:** Dimensional reduction for lean deployment.
- **Net2Net:** Architectural mapping for biological native entities.
