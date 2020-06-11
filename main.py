import torch
from mnist import Net

from tqdm import tqdm
from attacks import ES, IFGSM
from argparse import ArgumentParser
import data
import plot


def get_args():
    parser = ArgumentParser(description="MNIST Adversarial Examples")
    parser.add_argument("--n_iter", type=int, default=500,
                        help="Number of iterations")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to create")
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Amount of change allowed")
    parser.add_argument("--target", action="store_true",
                        help="Will create a random target to classify as")
    parser.add_argument("--blackbox", action="store_true",
                        help="Will use the blackbox attack")
    parser.add_argument("--save", action="store_true",
                        help="Save images")
    parser.add_argument("--n_images", type=int, default=0,
                        help="Number of images to save")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--model", type=str,
                        default="data/mnist_cnn.pt", help="Model to attack")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("EPSILON", args.epsilon)

    # load pre-trained model
    model = Net()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    # set model to evaluation mode
    model.eval()

    loader = data.mnist(batch_size=1, seed=args.seed)

    n_success = 0
    success_rate = 0.0
    t_count = 0
    t_queries = []
    images_saved = 0

    if args.blackbox:
        attack = ES(n_iter=args.n_iter, epsilon=args.epsilon)
    else:
        attack = IFGSM(n_iter=args.n_iter, epsilon=args.epsilon)

    print("Creating %d adversarial example(s)..." % args.n_samples)
    print("Running %s for %d iterations w/ epsilon %.2f...\n" %
          (attack.__class__.__name__, args.n_iter, args.epsilon))

    pbar = tqdm(total=args.n_samples)
    for i, (x, y) in enumerate(loader):
        torch.manual_seed(i)
        if args.target:
            y_target = torch.randint(low=0, high=9, size=(1,))
            while y_target == y:
                y_target = torch.randint(low=0, high=9, size=(1,))

            success, n_queries, adv, y_hat = attack.attack(
                model, x, y_target, target=args.target)
        else:
            success, n_queries, adv, y_hat = attack.attack(
                model, x, y)

        if success and args.save:
            # plot.images(x.squeeze(), y, adv.squeeze().detach(), y_hat)
            # plot.history(history)
            if images_saved < args.n_images:
                plot.save(adv, y_hat, "img_%d_%.2f_%s" %
                          (i, args.epsilon, attack.__class__.__name__))
                images_saved += 1

        if success:
            t_queries.append(n_queries)
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
        print("median queries", torch.median(torch.tensor(t_queries)))
