# SimbaV2

## Introduction

We introduce SimbaV2, a reinforcement learning architecture that stabilizes training on non-stationary data through hyperspherical normalization and distributional value estimation with reward scaling. By scaling with larger models and compute, SimbaV2 achieves state-of-the-art performance on 57 continuous control tasks across MuJoCo, DMC, MyoSuite, and Humanoid-bench. The Gymnasium 1.0 API implementation ensures seamless integration with diverse RL environments.

<p align="left">
  <img src="docs/images/overview.png" style="max-height: 200px; object-fit: contain;" class="figure">
</p>

<a href="https://joonleesky.github.io" class="nobreak">Hojoon Lee</a><sup>\*</sup>&ensp;
<a href="https://leeyngdo.github.io/" class="nobreak">Youngdo Lee</a><sup>\*</sup>&ensp; 
<a href="https://takuseno.github.io/" class="nobreak">Takuma Seno</a><sup></sup>&ensp;
<a href="https://i-am-proto.github.io" class="nobreak">Donghu Kim</a><sup></sup>&ensp;
<a href="https://www.cs.utexas.edu/~pstone/" class="nobreak">Peter Stone</a><sup></sup>&ensp;
<a href="https://sites.google.com/site/jaegulchoo" class="nobreak">Jaegul Choo</a><sup></sup>&ensp;

[[Website]](https://dojeon-ai.github.io/SimbaV2/) [[Paper]](https://arxiv.org/abs/2310.16828) [[Dataset]](https://www.tdmpc2.com/dataset)

## Result

<span class="simbav2">SimbaV2</span> outperforms other RL algorithms, where performance scales as compute increases. The numbers below each dot indicate the update-to-data (UTD) ratio. <span class="simbav2">SimbaV2</span>, with UTD=1, achieves a performance of $0.848$, surpassing <span class="tdmpc2">TD-MPC2</span> ($0.749$), the most computationally intensive version of <span class="simba">Simba</span> ($0.818$), and <span class="bro">BRO</span> ($0.807$). The results show normalized returns, averaged over $57$ continuous control tasks from MuJoCo, DMControl, MyoSuite, and HumanoidBench, each trained on $1$ million samples.


<p align="center">
  <img src="docs/images/online.png" style="max-height: 400px; object-fit: contain;" class="figure">
</p>

We scale the number of model parameters by increasing the width of the critic network, and scale compute by increasing the update-to-data (UTD) ratio with and without periodic reset. For an empirical analysis, we define two challenging benchmark subsets: DMC-Hard ($7$ tasks involving $\texttt{dog}$ and $\texttt{humanoid}$ embodiments) and HBench-Hard ($5$ tasks: $\texttt{run}$, $\texttt{balance-simple}$, $\texttt{sit-hard}$, $\texttt{stair}$, $\texttt{walk}$). On both benchmarks, <span class="bold simbav2">SimbaV2</span> benefit from both increased model size and UTD ratio, while <span class="bold simba">Simba</span> plateaus at some moment. Notably, <span class="bold simbav2">SimbaV2</span> <strong>scales smoothly alongside UTD ratio even without reset, where using reset slightly degrades its performance</strong>.

<p align="center">
  <img src="docs/images/param_scaling.png" style="max-height: 160px; object-fit: contain;" class="figure">
  <img src="docs/images/utd_scaling.png" style="max-height: 160px; object-fit: contain;" class="figure">
</p>

## Getting strated

### Docker

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
docker build . -t scale_rl .
docker run --gpus all -v .:/home/user/scale_rl -it scale_rl /bin/bash
```

### Pip/Conda

If you prefer to install dependencies manually, start by installing dependencies via conda by following the guidelines.
```
# Use pip
pip install -e .

# Or use conda
conda env create -f deps/environment.yaml
```

#### Jax for GPU
```
pip install -U "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# If you want to execute multiple runs with a single GPU, we recommend to set this variable.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

#### Mujoco
Please see installation instruction at [MuJoCo](https://github.com/google-deepmind/mujoco).
```
# Additional environmental evariables for headless rendering
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"
export MKL_SERVICE_FORCE_INTEL="0"
```

#### Humanoid Bench

```
git clone https://github.com/joonleesky/humanoid-bench
cd humanoid-bench
pip install -e .
```

#### Myosuite
```
git clone --recursive https://github.com/joonleesky/myosuite
cd myosuite
pip install -e .
```


##  Example usage

We provide examples on how to train SAC agents with SimBa architecture.  

To run a single online RL experiment
```
python run_online.py
```

To run a single offline RL experiment
```
python run_offline.py
```

To benchmark the algorithm with all environments
```
python run_parallel.py \
    --task all \
    --device_ids <list of gpu devices to use> \
    --num_seeds <num_seeds> \
    --num_exp_per_device <number>  
```


## Analysis

Please refer to `/analysis` to visualize the experimental results provided in the paper.


## License
This project is released under the [Apache 2.0 license](/LICENSE).

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@article{lee2025simbav2,
  title={Hyperspherical Normalization for Scalable Deep Reinforcement Learning}, 
  author={Hojoon Lee and Youngdo Lee and Takuma Seno and Donghu Kim and Peter Stone and Jaegul Choo},
  year={2025}
}
```
