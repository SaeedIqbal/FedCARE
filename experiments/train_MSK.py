import torch
from data.msk import MSKLoader
from models.cnn import MSKCNN
from utils.pruning import structured_prune
from utils.quantization import QuantizedModel
from utils.sparsification import TopKSparsifier
from utils.dp_smppc import DPSMPC

def msk(args):
    model = MSKCNN()
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
    loader = MSKLoader(root="/home/phd/Datasets/MSK")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):  # E=5 local epochs
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "checkpoints/msk_fedcare.pth")

if __name__ == "__main__":
    args = parse_args()
    msk(args)