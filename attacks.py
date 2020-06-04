import torch
import torch.nn.functional as F

import plot


def ifgsm(model, test_loader, n_samples=100, n_iter=1000, epsilon=0.1, seed=0):
    # set seed
    torch.manual_seed(seed)

    epsilon_iter = epsilon / n_iter
    min, max = 0.0, 1.0
    count = 0

    adv_samples = []

    for x, y in test_loader:
        # x, y = x.to(device), y.to(device)

        y_hat = model(x)
        if y.item() != torch.argmax(y_hat).item():
            continue

        queries = 0
        noise = torch.zeros(x.size())
        for i in range(n_iter):
            adv = x + noise
            adv = torch.clamp(adv, min, max)
            adv.requires_grad = True
            y_hat = model(adv)

            if y.item() != torch.argmax(y_hat).item():
                queries = i + 1
                adv_samples.append((queries, adv))
                break

            model.zero_grad()
            loss = F.cross_entropy(y_hat, y)
            loss.backward()

            noise += epsilon_iter * torch.sign(adv.grad.data)
            noise = torch.clamp(noise, -epsilon, epsilon)

        print("queries", queries)
        plot.images(x, y, adv, y_hat)

        count += 1
        if count == n_samples:
            break

    return adv_samples
