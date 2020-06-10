import torch
from mnist import Net

from tqdm import tqdm
from attacks import ES, IFGSM
from argparse import ArgumentParser
import data
import plot


def get_args():
    parser = ArgumentParser(description="Black-box")
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--blackbox", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # load pre-trained model
    model = Net()
    model.load_state_dict(torch.load("data/mnist_cnn.pt", map_location="cpu"))

    # set model to evaluation mode
    model.eval()

    loader = data.mnist(batch_size=1)

    n_success = 0
    success_rate = 0.0
    t_count = 0
    t_queries = 0

    if args.blackbox:
        attack = ES(n_iter=args.n_iter)
    else:
        attack = IFGSM(n_iter=args.n_iter)

    print("Creating %d adversarial example(s)..." % args.n_samples)
    print("Running %s for %d iterations...\n" %
          (attack.__class__.__name__, args.n_iter))

    pbar = tqdm(total=args.n_samples)
    for i, (x, y) in enumerate(loader):
        if args.target:
            torch.manual_seed(i)
            y_target = torch.randint(low=0, high=9, size=(1,))
            while y_target == y:
                y_target = torch.randint(low=0, high=9, size=(1,))

            success, n_queries, adv, y_hat = attack.attack(
                model, x, y_target, target=target)
        else:
            success, n_queries, adv, y_hat = attack.attack(
                model, x, y)

        # if success:
        #     plot.images(x.squeeze(), y, adv.squeeze().detach(), y_hat)
        #     plot.history(history)

        if success:
            t_queries += n_queries
            n_success += 1

        success_rate = n_success / (i + 1)
        pbar.set_description("n_success %d success_rate %.5f" %
                             (n_success, success_rate))
        pbar.update(1)
        t_count += 1
        if t_count == args.n_samples:
            break
    pbar.close()

    success_rate = n_success / args.n_samples
    print("\nsuccess rate", success_rate)
    if n_success > 0:
        print("avg queries", t_queries / n_success)
