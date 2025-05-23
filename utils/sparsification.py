class TopKSparsifier:
    def __init__(self, k=0.5):  # k=50%
        self.k = k

    def __call__(self, model):
        # Top-k gradient sparsification
        for param in model.parameters():
            threshold = torch.kthvalue(param.abs(), int((1 - self.k) * param.numel()))
            param.data[param.abs() < threshold] = 0
        return model