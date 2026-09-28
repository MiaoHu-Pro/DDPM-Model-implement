# DDPM diffusion-model implementation

This project contains a compact educational implementation of a **Denoising
Diffusion Probabilistic Model (DDPM)**. The main program is
`code-3-diff-unet-fixed.py`, which trains a time-conditioned U-Net to reverse a
gradual Gaussian-noising process.

## Core idea

During training, a clean image $x_0$ is moved directly to a randomly selected
diffusion step $t$:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0
    + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

Here, $\alpha_t=1-\beta_t$ and
$\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$. The U-Net receives the noisy image
$x_t$ and a sinusoidal embedding of timestep $t$, and learns to predict the
noise that was added:

$$
L_{\mathrm{DDPM}}
=\mathbb{E}_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].
$$

To generate an image, sampling starts from Gaussian noise $x_T$ and repeatedly
applies the learned reverse-denoising step until an estimate of $x_0$ is
obtained. The implementation uses residual convolution blocks, U-Net skip
connections, timestep conditioning, exponential-moving-average model weights,
mixed-precision training, validation, and checkpointing.

## Files and datasets

- `code-1.py`: visualizes the iterative forward noising process.
- `code-2.py`: demonstrates the closed-form formula for obtaining $x_t$.
- `code-3.py`: the original small demonstration.
- `code-3-diff-unet-fixed.py`: the corrected, server-friendly training and
  sampling implementation.
- `submit-code-3-diff-unet-fixed.sh`: Slurm launcher for the fixed program.

The fixed program supports two dataset modes:

- `--dataset demo` repeatedly trains on the two local demonstration images.
- `--dataset flickr8k` trains on the images stored in the local Flickr8k
  Parquet files and evaluates denoising loss on its validation/test data.

This is an **unconditional image model**: Flickr8k captions are not passed to
the U-Net. Therefore, it learns $p(x)$ rather than text-conditioned
$p(x\mid\text{caption})$ and cannot generate an image from a text prompt.

## Example Slurm run

```bash
cd ./diffusion-model

sbatch submit-code-3-diff-unet-fixed.sh \
    --dataset flickr8k \
    --epochs 100 \
    --sample-every 10 \
    --num-samples 16
```

`--sample-every 10` generates a preview after every ten training epochs; it
does not sample every training image ten times. Outputs include the best model
checkpoint, intermediate preview grids, `final-samples.png`, and
`metrics.json`.
