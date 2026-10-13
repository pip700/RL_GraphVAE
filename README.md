# RL_GraphVAE — Target-Biased Novel Molecule Discovery

A reinforcement learning-enhanced Graph Variational Autoencoder (GraphVAE) framework for **de novo molecular generation** with **target-biased optimization** using a downstream activity prediction reward model.

The workflow combines:

* **GraphVAE pretraining** for molecular graph reconstruction
* **Policy-gradient reinforcement learning** for target-specific optimization
* **XGBoost reward prediction** for molecular activity scoring
* **Dockerized GPU deployment** for reproducible training and inference

---

# Architecture Overview

```text
RL-GraphVAE-Driven Molecular Generation
│
├── PART I: GraphVAE Pretraining
│   GraphVAE_training
│   ├── GraphVAE
│   │   ├── Encoder: Molecular graph x → latent representation z
│   │   └── Decoder: Latent representation z → reconstructed graph
│   ├── Jointly train encoder and decoder
│   ├── Total pretraining loss:
│   │   L_pre = λ_t·L_topo + λ_b·L_bond + λ_l·L_label
│   │           + λ_r·L_root + λ_m·L_mol + β·L_KL
│   └── Save pretrained model checkpoints
│       └── model.iter-{epoch}
▼
[Pretrained checkpoint: model.iter-N]
▼
├── PART II: Reinforcement Learning Fine-Tuning
│   RL_GraphVAE_decoder_training
│   ├── Load pretrained GraphVAE weights
│   ├── Encoder: Graph_VAE.encoder()
│   │   ├── Input: Molecular graph x
│   │   ├── Output: z₀, mean (μ), log_var (log σ²)
│   │   └── Sample latent: z₀ ~ qφ(z|x) Encoder parameters φ: remain frozen
│   └── Decoder: Graph_VAE.decoder()
│       │  Decoder acts as a stochastic molecular-generation policy
│       ├── 1. Root selection
│       │   └── _stochastic_label_prediction_1(z₀) → root atom
│       ├── 2. Latent-state update
│       │   └── z₁ = tanh(update_fc(z₀ + update_z(...)))
│       ├── 3. Sequential graph construction
│       │   ├── _stochastic_topo_prediction(z₁)
│       │   │   └── Predict molecular graph topology
│       │   ├── _stochastic_bond_prediction(z₁)
│       │   │   └── Predict bond types
│       │   ├── _stochastic_label_prediction(z₁)
│       │   │   └── Predict atom/substructure labels
│       │   └── Update latent state using graph context:
│       │       z₍ₜ₊₁₎ = tanh(W_z zₜ + W_h hₜ + b)
│       ├── 4. Continue sequential generation
│       │   ├── Repeat graph-generation actions
│       │   ├── Update latent state at each step
│       │   ├── Apply backtracking when required
│       │   ├── connect_smiles() → dec_smi
│       │   └── Accumulate action log-probabilities:
│       │       log_prob_sum = Σₜ log πθ(aₜ|zₜ)
│       ▼
│   [Generated molecular SMILES: dec_smi]
│       ▼
│   ├── Biological Activity-Based Reward
│   │   XGBoostRewardModel.predict(dec_smi)
│   │   ├── Predict p_active and p_inactive
│   │   └── Assign molecular reward:
│   │       r(x) = Pψ(active|x)
│   ▼
├── Advantage Estimation
│   ├── Calculate advantage using the current baseline:
│   │   A(x) = r(x) − b
│   └── Update baseline for subsequent samples:
│       b_new = λ·b + (1 − λ)·r(x)
▼
├── Reinforcement-Learning Optimization
│   ├── Policy-gradient loss (REINFORCE):
│   │   L_PG = −A(x) · log_prob_sum
│   ├── KL-divergence loss:
│   │   L_KL = −0.5·Σ[1 + log_var − mean² − exp(log_var)]
│   ├── Combined objective:
│   │   L_total = rl_weight·L_PG + kl_beta·L_KL
│   └── Update trainable parameters using gradients:
│       θ ← θ − η·∇θ L_total
│       (Decoder parameters are intended to be optimized)
▼
[Repeat Generation → Reward Evaluation → Optimization]

```

---

# Docker Deployment

## Run Container

```bash
sudo docker run --gpus all -d --rm \
  -v $(pwd):/workspace \
  -p 8500:8501 \
  --name RL_GraphVAE \
  ghcr.io/pip700/rl_graphvae:v2
```

---

# Web Interface

Access the application via browser:

```text
http://0.0.0.0:8500/
```

---

# Stop Container

```bash
sudo docker rm RL_GraphVAE --force
```



# Citation

If used in research, cite:

```bibtex
  software={RL-GraphVAE},
  title={Generative AI-Assisted Molecular Design of AChEIs},
  author={Dhairiya Agarwal, Prabha Garga},
  year={2026}
  DOI:{https://doi.org/10.1021/acs.jcim.6c02233}
```
