from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler


class Attack(ABC):

    @abstractmethod
    def attack(self):
        pass


class ES(Attack):

    def __init__(self, n_iter=1000, n_population=5, epsilon=0.3, step_size=0.05, sigma=0.25, seed=0):
        self.n_iter = n_iter
        self.n_population = n_population
        self.epsilon = epsilon
        self.sigma = sigma
        self.step_size = step_size
        self.seed = seed

    def attack(self, model, x, y, target=False):
        self.model_ = model
        self.min_ = 0.0
        self.max_ = 1.0

        self.size_ = x.view(-1).size()[0]
        self.mu_ = torch.zeros(self.size_)
        self.height_ = x.size()[2]
        self.width_ = x.size()[3]
        self.scaler_ = MinMaxScaler(
            feature_range=(0.0, self.epsilon / 2))

        success = False
        n_queries = 0
        adv = None
        y_hat = None

        history = []
        for i in range(self.n_iter):
            noise = torch.randn(self.n_population, self.size_)
            noise = torch.tensor(self.scaler_.fit_transform(
                noise), dtype=torch.float32)
            population = self.mu_ + self.sigma * noise

            fitness = torch.zeros(self.n_population)
            for j, sample in enumerate(population):
                perturbation = torch.reshape(
                    sample, (1, 1, self.height_, self.width_))
                perturbation = torch.clamp(
                    perturbation, -self.epsilon, self.epsilon)
                adv = x + perturbation
                adv = torch.clamp(adv, self.min_, self.max_)
                y_hat = torch.detach(self.model_(adv))
                n_queries += 1

                if target:
                    success = torch.argmax(y_hat).item() == y.item()
                else:
                    success = torch.argmax(y_hat).item() != y.item()

                if success:
                    adv = torch.detach(adv)
                    adv = torch.squeeze(adv)
                    break

                loss = F.cross_entropy(y_hat, y) ** 2
                if target:
                    loss = -loss

                fitness[j] = loss

            if success:
                break

            history.append(torch.mean(fitness))
            if not torch.all(fitness == fitness[0]):
                fitness_std = torch.unsqueeze(
                    (fitness - torch.mean(fitness)) / torch.std(fitness), 1)

                update = self.step_size / \
                    (self.n_population * self.sigma) * \
                    torch.mm(noise.T, fitness_std)

                self.mu_ += torch.squeeze(update)
            else:
                self.mu_ += self.step_size * torch.randn(self.mu_.size())

        return success, n_queries, adv, y_hat  # , history


class IFGSM(Attack):

    def __init__(self, n_iter=1000, epsilon=0.3, step_size=0.001, seed=0):
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.step_size = step_size
        self.seed = seed

    def attack(self, model, x, y, target=False):
        self.model_ = model
        self.min_ = 0.0
        self.max_ = 1.0
        self.noise_ = torch.zeros(x.size())

        success = False
        n_queries = 0
        adv = None
        y_hat = None

        for i in range(self.n_iter):
            adv = x + self.noise_
            adv = torch.clamp(adv, self.min_, self.max_)
            adv.requires_grad = True
            y_hat = self.model_(adv)
            n_queries += 1

            if target:
                success = torch.argmax(y_hat).item() == y.item()
            else:
                success = torch.argmax(y_hat).item() != y.item()

            if success:
                adv = torch.detach(adv)
                adv = torch.squeeze(adv)
                break

            model.zero_grad()
            loss = F.cross_entropy(y_hat, y)
            loss.backward()

            if target:
                self.noise_ -= self.step_size * torch.sign(adv.grad.data)
            else:
                self.noise_ += self.step_size * torch.sign(adv.grad.data)

            self.noise_ = torch.clamp(self.noise_, -self.epsilon, self.epsilon)

        return success, n_queries, adv, y_hat
