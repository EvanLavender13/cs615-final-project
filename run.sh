#!/bin/bash

# # Misclassification
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_.pt --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_.pt --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_.pt --blackbox

# # Target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_.pt --target --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_.pt --target --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_.pt --target --blackbox

# # Misclassification
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_p.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_p.pt --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_p.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_p.pt --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_p.pt
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_p.pt --blackbox

# # Target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_p.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_p.pt --target --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_p.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_p.pt --target --blackbox
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_p.pt --target
# python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_p.pt --target --blackbox

# Misclassification
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_pbb.pt
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 0 --model data/mnist_cnn_pbb.pt --blackbox
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_pbb.pt
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 1 --model data/mnist_cnn_pbb.pt --blackbox
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_pbb.pt
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 2 --model data/mnist_cnn_pbb.pt --blackbox

# Target
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_pbb.pt --target
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.1 --save --n_images 10 --seed 3 --model data/mnist_cnn_pbb.pt --target --blackbox
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_pbb.pt --target
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.2 --save --n_images 10 --seed 4 --model data/mnist_cnn_pbb.pt --target --blackbox
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_pbb.pt --target
python3 main.py --n_iter 500 --n_samples 100 --epsilon 0.3 --save --n_images 10 --seed 5 --model data/mnist_cnn_pbb.pt --target --blackbox