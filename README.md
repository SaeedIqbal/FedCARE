### **FedCARE**  
The code integrates **structured pruning**, **quantization**, **sparsification**, and **DP+SMPC** for federated domain adaptation (FDA) on **Digit-Five**, **Office**, and **Office-Caltech10** datasets. 

---

### **Repository Layout**  
```
FedCARE/
├── README.md
├── main.py
├── config/
│   └── hyperparams.py
├── models/
│   ├── resnet.py
│   └── cnn.py
├── data/
│   ├── digit_five.py
│   ├── office.py
│   └── office_caltech10.py
├── utils/
│   ├── pruning.py
│   ├── quantization.py
│   ├── sparsification.py
│   └── dp_smppc.py
└── experiments/
    ├── train_digit_five.py
    ├── train_office.py
    └── train_office_caltech10.py
```

---

### **1. `README.md`**  
```markdown
# FedCARE: Federated Compression, Adaptation, and Robustness for Edge-Centric Domain Alignment under DP and SMPC

This repository contains the official implementation of **FedCARE**, a unified framework for **federated domain adaptation (FDA)** under non-i.i.d. data, communication constraints, and differential privacy (DP) + secure multiparty computation (SMPC). The code supports experiments on **Digit-Five**, **Office**, and **Office-Caltech10** datasets, validated via Lemma 1–4 and Theorem 1–4 in the manuscript.

## 📚 Datasets
### **1. Digit-Five**
- **Description**: Combines 5 digit datasets (MNIST, MNIST-M, USPS, SVHN, SYN) with domain shifts.
- **Path**: `/home/phd/Datasets/Digit-Five`
- **Classes**: 10
- **Image Size**: 32×32 pixels

### **2. Office**
- **Description**: 31 classes of office appliances from Amazon, DSLR, and Webcam.
- **Path**: `/home/phd/Datasets/Office`
- **Image Size**: 224×224 (ResNet input)

### **3. Office-Caltech10**
- **Description**: Merges Office and Caltech-256 datasets with 10 shared classes across 4 domains.
- **Path**: `/home/phd/Datasets/Office-Caltech10`
- **Image Size**: 224×224 (ResNet input)

---

## 🛠️ Installation

### **Dependencies**
```bash
pip install torch torchvision opacus scikit-learn matplotlib seaborn numpy pandas
```

### **Dataset Setup**
1. Download datasets from official sources:
   - **Digit-Five**: [Link](https://github.com/eriklindernoren/PyTorch-GAN)
   - **Office**: [Link](https://www.cc.gatech.edu/~hays/DA/)
   - **Office-Caltech10**: [Link](http://www.vision.soic.indiana.edu/egoipt/)

2. Organize directories:
```bash
/home/phd/Datasets/
├── Digit-Five/
│   ├── train/
│   └── test/
├── Office/
│   ├── Amazon/
│   ├── DSLR/
│   └── Webcam/
└── Office-Caltech10/
    ├── Amazon/
    ├── Caltech/
    ├── DSLR/
    └── Webcam/
```

---

## 🚀 Usage

### **Train on Digit-Five**
```bash
python experiments/train_digit_five.py --prune --quantize --sparsify --dp --smpc
```

### **Train on Office**
```bash
python experiments/train_office.py --prune --quantize --sparsify --dp --smpc
```

### **Train on Office-Caltech10**
```bash
python experiments/train_office_caltech10.py --prune --quantize --sparsify --dp --smpc
```

### **Hyperparameters**
| Flag | Description |
|------|-------------|
| `--prune` | Enable structured pruning ($\lambda=0.01$) |
| `--quantize` | Enable 8-bit precision ($b=8$) |
| `--sparsify` | Top-$k=50\%$ gradient sparsification |
| `--dp` | Apply $(\epsilon=2, \delta=10^{-5})$-DP |
| `--smpc` | Enable SMPC gradient reconstruction |

---

## 📌 Methodology Overview
### **Core Components**
1. **Structured Pruning**  
   - Sparsity-inducing $L_1$-regularization ($\lambda=0.01$) bounds model deviation:  
     $$
     \|\theta^* - \theta_{\text{dense}}\|_2 \leq \sqrt{\frac{2\lambda s}{\mu}} \quad \text{(Lemma 1)}
     $$

2. **Quantization-Stabilized Training**  
   - 8-bit precision ($b=8$) limits error:  
     $$
     \|\theta_q - \theta\|_2 \leq \frac{\Delta \sqrt{d}}{2} \quad \text{(Lemma 2)}
     $$

3. **IDD Minimization**  
   - Gradient alignment loss:  
     $$
     \mathcal{L}_{\text{IDD}} = \|\nabla F_1(G(x_t)) - \nabla F_2(G(x_t))\|_1 \quad \text{(Theorem 4)}
     $$

4. **Secure Aggregation**  
   - DP+SMPC ensures $\epsilon=2$-DP with 100% gradient reconstruction:  
     $$
     \mathcal{R}(\text{Enc}_{\text{SMPC}}(\nabla \theta_t)) = \nabla \theta_t \quad \text{(Lemma 4)}
     $$

---

## 🧪 Results

### **Accuracy vs. Label Skew ($\alpha=1$–50)**
| Dataset | FedCARE | FedAvg | FADA |
|--------|---------|--------|------|
| Digit-Five | 95.2% | 64.28% | 73.6% |
| Office | 95.7% | 84.9% | 75.6% |
| Office-Caltech10 | 96.8% | 86.7% | 76.8% |

### **Communication Efficiency**
| Dataset | CR (FedCARE) | CR (FedAvg) | SMPC Overhead |
|--------|---------------|--------------|----------------|
| Digit-Five | 4.3× (44.7→10.4 MB) | 1.0× | 30% increase |
| Office-Caltech10 | 4.5× (44.7→9.9 MB) | 1.0× | 30% increase |

---

## 📝 Citation  
If you use this code, please cite our work:  
```bibtex
@article{iqbal2025fedcare,
  title={FedCARE: Federated Compression, Adaptation, and Robustness for Edge-Centric Domain Alignment under DP and SMPC},
  author={Saeed Iqbal et al.},
  journal={Knowledge-Based Systems},
  year={2025}
}
```

## 📧 Contact  
For questions, contact [saeed.iqbal@szu.edu.cn](mailto:saeed.iqbal@szu.edu.cn).
```
```
## Scientific Enrichment  
```
1. Structured Pruning
   - Applied to convolutional layers with sparsity=40% (Lemma 1: $\|\theta^* - \theta_{\text{dense}}\|_2 \leq \sqrt{\frac{2\lambda s}{\mu}}$).  

2. 8-bit Quantization  
   - Limits error via $\|\theta_q - \theta\|_2 \leq \frac{\Delta \sqrt{d}}{2}$ (Lemma 2), enabling 4.3× compression (Digit-Five).  

3. Top-k Sparsification  
   - Reduces communication cost by 50% (22.4→11.2 MB/round) while maintaining convergence (Theorem 1: $B_{\text{joint}} = \eta^2 L^2 (\tau^2 + \sigma^2)$).  

4. DP+SMPC  
   - Ensures $\epsilon=2$-DP with SMPC reconstruction ($\mathcal{R}(\text{Enc}_{\text{SMPC}}(\nabla \theta_t)) = \nabla \theta_t$; Lemma 4).  

5. IDD Minimization  
   - $\mathcal{L}_{\text{IDD}} = \|\nabla F_1(G(x_t)) - \nabla F_2(G(x_t))\|_1$ reduces NT by 10× (e.g., 12.1%→1.2% on Digit-Five).  


```
## Key Adjustments
```
1. Dataset-Specific Architectures:
   - CNN for Digit-Five, ResNet-101 for Office/Office-Caltech10.  
2. DP Integration:  
   - Uses Opacus for $(\epsilon, \delta)$-DP. Replace with custom SMPC for secure aggregation.  
3. Sparsification:  
   - Top-k gradient masking (mock implementation; replace with gradient sparsification logic).  
