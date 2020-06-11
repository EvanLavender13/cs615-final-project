import torch
import matplotlib.pyplot as plt


def images(x, y, adv, y_hat):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    axes[0].imshow(x, cmap="gray")
    axes[0].set_title(y.item())
    axes[1].imshow(adv, cmap="gray")
    axes[1].set_title(torch.argmax(y_hat).item())
    fig.tight_layout()
    plt.show()


def save(adv, y_hat, file):
    file += ".png"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    ax.imshow(adv, cmap="gray")
    ax.set_title(torch.argmax(y_hat).item())
    fig.tight_layout()
    plt.savefig("images/" + file)


def history(hist):
    plt.plot(range(len(hist)), hist)
    plt.tight_layout()
    plt.show()
