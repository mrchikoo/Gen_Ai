<div align="center">

# FashionGAN

**A DCGAN-style generative adversarial network that dreams up Fashion-MNIST clothing images from random noise.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

</div>

## Overview

This project (my B.E. final-year project, see `BE_ProjReport.pdf`) trains a generative adversarial network on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset (60,000 28x28 grayscale clothing images) and uses the trained generator to synthesize new, previously unseen fashion items. Everything lives in one notebook, `FashionGAN.ipynb`, and the trained generator is included as `generatormodel.h5`.

## How it works

1. **Data**: Fashion-MNIST is loaded with `tensorflow_datasets`, scaled to `[0, 1]`, cached, shuffled, batched (128) and prefetched.
2. **Generator**: a 128-dim noise vector goes through `Dense(7*7*128)`, is reshaped to 7x7x128, upsampled twice (to 28x28) with 5x5 convolutions, followed by two 4x4 convolutions and a final `Conv2D(1, 4, sigmoid)`. All hidden layers use LeakyReLU(0.2).
3. **Discriminator**: four `Conv2D` layers (32, 64, 128, 256 filters, 5x5 kernels) with LeakyReLU and Dropout(0.4), then Flatten, Dropout and a sigmoid `Dense(1)`.
4. **Training**: a custom `FashionGAN(Model)` subclass overrides `train_step` and updates the discriminator and generator with separate `GradientTape`s. Binary cross-entropy for both, Adam optimizers (generator lr 1e-4, discriminator lr 1e-5), and small random noise added to the discriminator labels to keep it from overpowering the generator. A `ModelMonitor` callback saves sample images each epoch. The notebook trains for 20 epochs.
5. **Inference**: load the saved generator weights, feed `tf.random.normal((16, 128, 1))` and plot the generated images.

## Tech stack

Python, TensorFlow / Keras, TensorFlow Datasets, Matplotlib, NumPy, Jupyter.

## Getting started

```bash
git clone https://github.com/mrchikoo/Gen_Ai.git
cd Gen_Ai
pip install tensorflow matplotlib tensorflow-datasets ipywidgets jupyter
jupyter notebook FashionGAN.ipynb
```

- To **retrain**, run the notebook top to bottom (a GPU is strongly recommended; the notebook enables GPU memory growth). Create an `images/` folder first, since the callback saves samples there.
- To **only generate images**, run the data/model-definition cells, then the cell that loads weights. The notebook loads them from `archive/generatormodel.h5`, so either place `generatormodel.h5` in an `archive/` folder or change the path to point at the file in the repo root.

## Repository contents

| File | Description |
|------|-------------|
| `FashionGAN.ipynb` | Data pipeline, generator, discriminator, custom GAN training loop, loss plot and sample generation |
| `generatormodel.h5` | Trained generator weights |
| `BE_ProjReport.pdf` | Full project report |

## Notes

- Output images are 28x28 grayscale; expect Fashion-MNIST-like quality, not photorealism.
- No quantitative metrics (e.g. FID) are computed in the notebook; judge results visually from the generated samples and the loss curves.
- `.h5` is a legacy Keras format; with very recent Keras versions you may need `tf_keras` or an older TensorFlow to load it.
