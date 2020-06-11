# CS615 Final Project
## Evan Lavender (edl43@drexel.edu)
## Adversarial Examples

### Use `run.sh`

### `main.py` usage - for running attacks
```
usage: main.py [-h] [--n_iter N_ITER] [--n_samples N_SAMPLES] [--epsilon EPSILON] [--target] [--blackbox] [--save] [--n_images N_IMAGES] [--seed SEED] [--model MODEL]

MNIST Adversarial Examples

optional arguments:
  -h, --help            show this help message and exit
  --n_iter N_ITER       Number of iterations
  --n_samples N_SAMPLES
                        Number of samples to create
  --epsilon EPSILON     Amount of change allowed
  --target              Will create a random target to classify as
  --blackbox            Will use the blackbox attack
  --save                Save images
  --n_images N_IMAGES   Number of images to save
  --seed SEED           Random seed
  --model MODEL         Model to attack
```

### `mnist.py` usage - for training
```
usage: mnist.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR] [--gamma M] [--no-cuda] [--seed S] [--log-interval N] [--save-model] [--perturb] [--blackbox]

PyTorch MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 14)
  --lr LR              learning rate (default: 1.0)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         For Saving the current Model
  --perturb            For perturbing training images
  --blackbox
```
