# Large Scale Distributed Parallel MCTS algorithm

This project developed highly scalable distributed parallel MCTS algorithms and their applications for molecular design probelm.
Check the paper for details: [Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design.](https://arxiv.org/abs/2006.10504)

## Requirements

The code was tested on Linux and MacOS, we recommend using anaconda to install the following softwares.

1. [Python](https://www.anaconda.com/products/individual)(version 3.7)
2. [mpi4py](https://anaconda.org/anaconda/mpi4py)(version 3.0.3)
3. [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2)

### How to setup the environment

NOTE: You need to run MPChemTS on a server where OpenMPI or MPICH is installed. If you can't find `mpiexec` command, please consult your server administrator to install such an MPI library.

```bash
cd YOUR_WORKSPACE
python3.7 -m venv .venv
source .venv/bin/activate
pip install chemtsv2
pip install mpi4py==3.0.3
```

## Run parallel MCTS algorithms for molecular design

### optimization of logP property

```bash
mpiexec -n 4 python run_mpchemts.py --config config/setting.yaml
```

> where 4 is the number of cores or processes to use. You can use more cores by changing 4 to 1024 for example. The example code used D-MCTS algorithm as default, you can simply change to H-MCTS for your own purpose by checking the source code of example_logp.py.

## Implement your own property simulator

> Go to pmcts folder and add your simulator to property_simulator.py
