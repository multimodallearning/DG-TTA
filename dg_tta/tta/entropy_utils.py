import torch


def calc_model_entropy(model):
    with torch.no_grad():
        params = [p for p in model.parameters() if p.grad is not None]
        values = torch.cat([p.view(-1) for p in params])

        return get_entropy(values)


def get_entropy_change(entropies):
    filter_fn = lambda input: torch.nn.functional.conv1d(torch.cat([input[0:1], input]).view(1,1,-1), weight=torch.tensor([[[-1.,1.]]]))
    return filter_fn(entropies).view(-1)


def get_entropy(values):
    hist = torch.histogram(values.cpu(), bins=int(torch.tensor(values.numel()).sqrt().ceil().item()), density=False)[0]
    hist = hist / hist.sum()
    entropy = -(hist * hist.log().amax(0)).sum()
    return entropy


def entropy_is_increasing(entropies):
    return any(get_entropy_change(entropies) > 0.)