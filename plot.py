import torch
import matplotlib.pyplot as plt


def images(x, y, adv, y_hat):
    x = torch.detach(x)
    x = torch.squeeze(x)
    adv = torch.detach(adv)
    adv = torch.squeeze(adv)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    axes[0].imshow(x, cmap="gray")
    axes[0].set_title(y.item())
    axes[1].imshow(adv, cmap="gray")
    axes[1].set_title(torch.argmax(y_hat).item())
    fig.tight_layout()
    fig.show()
