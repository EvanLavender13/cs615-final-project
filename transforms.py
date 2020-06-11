import torch


class Adversarial:

    def __init__(self, model, attack, prob=0.1):
        self.model = model
        self.attack = attack
        self.prob = prob

    def __call__(self, tensor):
        if torch.rand(1) < self.prob:
            x = torch.unsqueeze(tensor, 1)
            y_target = torch.randint(low=0, high=9, size=(1,))
            _, _, adv, _ = self.attack.attack(
                self.model, x, y_target, target=True)

            if adv.size() == x.size():
                tensor = torch.squeeze(adv, 1)
            else:
                tensor = tensor

        return tensor
