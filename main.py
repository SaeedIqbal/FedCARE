### **2. Python Implementation**  
#### **`main.py`**  
```python
import torch
import argparse
from models.resnet import ResNet101
from models.cnn import MSKCNN
from utils.pruning import structured_prune
from utils.quantization import QuantizedModel
from utils.sparsification import TopKSparsifier
from utils.dp_smppc import DPSMPC

def parse_args():
    parser = argparse.ArgumentParser(description="FedCARE Training")
    parser.add_argument("--dataset", choices=["MSK", "DR", "MM-WHS"], required=True)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--sparsify", action="store_true")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--smpc", action="store_true")
    return parser.parse_args()

def train_fedcare(args):
    # Load dataset
    if args.dataset == "MSK":
        model = MSKCNN()
        data_loader = "MSK_loader"
    elif args.dataset == "office":
        model = ResNet101()
        data_loader = "office_loader"
    else:
        model = ResNet101()
        data_loader = "office_caltech_loader"

    # Apply structured pruning
    if args.prune:
        model = structured_prune(model, sparsity=0.4)  # s=40%

    # Apply 8-bit quantization
    if args.quantize:
        model = QuantizedModel(model, bits=8)

    # Apply top-k sparsification
    sparsifier = TopKSparsifier(k=0.5)  # k=50%
    if args.sparsify:
        model = sparsifier(model)

    # DP+SMPC
    if args.dp and args.smpc:
        model = DPSMPC(model, epsilon=2)

    # Train loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):  # E=5 local epochs
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            if args.dataset == "MSK" and args.sparsify:
                loss += 0.02 * model.idd_loss()  # Theorem 4
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model.state_dict(), f"checkpoints/{args.dataset}_fedcare.pth")
    print(f"Training complete for {args.dataset} with FedCARE.")

if __name__ == "__main__":
    args = parse_args()
    train_fedcare(args)


'''
import torch
import argparse
from models.resnet import ResNet101
from models.cnn import MSKCNN
from utils.pruning import structured_prune
from utils.quantization import QuantizedModel
from utils.sparsification import TopKSparsifier
from utils.dp_smppc import DPSMPC

def parse_args():
    parser = argparse.ArgumentParser(description="FedCARE Training")
    parser.add_argument("--dataset", choices=["MSK", "DR", "MM-WHS"], required=True)
    parser.add_argument("--prune", action="store_true")  # Lemma 1
    parser.add_argument("--quantize", action="store_true")  # Lemma 2
    parser.add_argument("--sparsify", action="store_true")  # Theorem 1–2
    parser.add_argument("--dp", action="store_true")  # Theorem 3
    parser.add_argument("--smpc", action="store_true")  # Lemma 4
    return parser.parse_args()

def train_fedcare(args):
    # Load dataset-specific model
    if args.dataset == "MSK":
        model = MSKCNN()
    else:
        model = ResNet101()

    # Apply structured pruning (Lemma 1)
    if args.prune:
        model = structured_prune(model, sparsity=0.4)  # s=40%

    # Apply 8-bit quantization (Lemma 2)
    if args.quantize:
        model = QuantizedModel(model, bits=8)

    # Apply top-k sparsification (Theorem 1–2)
    sparsifier = TopKSparsifier(k=0.5)  # k=50%
    if args.sparsify:
        model = sparsifier(model)

    # Apply DP+SMPC (Theorem 3 & Lemma 4)
    if args.dp and args.smpc:
        model = DPSMPC(model, epsilon=2)  # σ=1.9 at ε=2

    # Training logic
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):  # E=5 local epochs
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            if args.sparsify:
                loss += 0.02 * model.idd_loss()  # Theorem 4
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f"checkpoints/{args.dataset}_fedcare.pth")
    print(f"Training complete for {args.dataset} with FedCARE.")
'''