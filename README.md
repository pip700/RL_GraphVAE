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
RL_GraphVAE Driven Molecular Generation

GraphVAE_training.py
│  status = 'train'
│  Graph_VAE.Chem_VAE (encoder + decoder trained jointly)
│  Saves: model.iter-{epoch}
│
│  Loss = λ_t·L_topo + λ_b·L_bond + λ_l·L_label
│       + λ_r·L_root + λ_m·L_mol + β·KL
│
▼
[Pretrained checkpoint: model.iter-N]
│
▼
RL_GraphVAE_decoder_training.py
   status = 'rl_train'
   Loads pretrained weights
   (Attempts to) freeze encoder
   │
   ├── Graph_VAE.encoder() → z₀, mean, log_var
   │   [frozen - intended]
   │
   └── Graph_VAE.decoder(status='rl_train')
       ├── _stochastic_label_prediction_1(z₀) → root
       ├── z₁ = tanh(update_fc(z₀ + update_z(...)))
       ├── _stochastic_topo_prediction(z₁)
       ├── _stochastic_bond_prediction(z₁)
       ├── _stochastic_label_prediction(z₁)
       ├── connect_smiles() → dec_smi
       └── z₂ = tanh(update_fc(z₁ + update_z(...)))
       ...
       └── log_prob_sum = Σ log π_θ(a_t|z_t)
   │
   ├── XGBoostRewardModel.predict(dec_smi)
   │   └── reward, p_active, p_inactive
   │
   ├── kl_loss = -0.5·Σ(1 + log_var - mean² - exp(log_var))
   ├── advantage = reward - baseline
   ├── pg_loss = -(advantage · log_prob_sum)
   └── total_loss = rl_weight·pg_loss + kl_beta·kl_loss
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
