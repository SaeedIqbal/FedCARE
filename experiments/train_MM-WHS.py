import torch
from data.MM_WHS import MM_WHSLoader
from models.resnet import ResNet101
from utils.pruning import structured_prune
from utils.quantization import QuantizedModel
from utils.sparsification import TopKSparsifier
from utils.dp_smppc import DPSMPC

def train_mm_whs(args):
    model = ResNet101(num_classes=10)
    if args.prune:
        model = structured_prune(model, sparsity=0.4)
    if args.quantize:
        model = QuantizedModel(model, bits=8)
    if args.sparsify:
        sparsifier = TopKSparsifier(k=0.5)
        model = sparsifier(model)
    if args.dp and args.smpc:
        model = DPSMPC(model, epsilon=2)

    # Training logic
    loader = MM_WHSLoader(root="/home/phd/Datasets/MM_WHS")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):  # E=5 local epochs
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            if args.sparsify:
                loss += 0.02 * model.idd_loss()  # Theorem 4
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "checkpoints/MM_WHS_fedcare.pth")

if __name__ == "__main__":
    args = parse_args()
    train_mm_whs(args)