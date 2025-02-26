# Cache-DFL

A **cache-based Distributed Federated Learning** framework (experimental) for **"Decentralized Federated Learning with Model Caching on Mobile Agents."**. This repository explores new strategies for caching and data loading in distributed or federated learning scenarios.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Federated Learning Trainer**: A customizable trainer for simulating federated learning across multiple nodes.
- **Caching Mechanism**: Implements a caching module to efficiently manage repeated data access.
- **Data Handling Utilities**: Convenient data loaders for local or remote datasets.
- **Extensible Design**: Easily modify or extend the core classes to fit your own research.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shawnxiaoyuwang/Cache-DFL.git
   cd Cache-DFL
   
2. **Install dependencies (if using requirements.txt)**:
   ```bash
   pip install -r requirements.txt
   ```
   Or if you prefer a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

# Usage
1. **Configure your settings:**
   * Update `config.yaml` or `config.json` with your desired parameters (learning rate, federation rounds, caching strategy, etc.).
2. **Run the main script:**
   ```bash
   python main.py --config config.yaml
   ```

# Project Structure
   ```plaintext
   Cache-DFL/
   ├── cache_dfl/                # Main Python package
   │   ├── aggregation.py
   │   ├── cache_algorithm.py    #
   │   ├── trainer_single.py     # Contains main training logic on single thread
   │   ├── trainer_mpi.py        # Contains main training logic on multiple threads/machines (using mpi)
   │   ├── data_loader.py        # Data loading and caching logic
   │   ├── model.py              # Models or model architectures
   │   ├── road_sim.py           # Road simulation
   │   ├── seed_setter.py        # Random seeds
   │   ├── utils.py              # Utility functions
   │   └── utils_cnn.py          # Utility functions for CNN
   |
   ├── examples/
   │   └── demo.py (To be develop)
   │   └── ...
   |
   ├── tests/
   │   ├── test_trainer.py (To be develop)
   │   └── ...
   ├── config.json               # Example config file
   ├── requirements.txt          # Python dependencies
   ├── LICENSE
   ├── main.py                   # Entry point for running the federated learning pipeline using mpi process (to be developed, currently you can excute trainer_single.py ot trainer_mpi.py with argparse directly)
   └── README.md
```
* `cachedfl/`: Core Python package containing the main source code.
* `examples/`: Example scripts that show how to use the code.
* `tests/`: Automated tests using pytest or unittest.
* `requirements.txt`: Python dependencies list.
* `README.md`: This file.


# Configuration
* **Hyperparameters**: Learning rate, batch size, number of epochs, and federated rounds can be modified in the config file or via command-line flags.
* **Caching Options**: Customize parameters such as cache size or eviction policies.
* **Distributed Settings**: Simulation on single thread or multiple threads/machines (using mpi)
# Examples
## to be developed, currently you can excute trainer_single.py ot trainer_mpi.py with argparse directly
   ``` bash
   python examples/demo.py
```
## License

* This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments
* **Libraries/Frameworks**: PyTorch, NumPy, or others as relevant.
* **Contributors**:  Xiaoyu Wang, New York University
* **References:**
  Wang, Xiaoyu, Guojun Xiong, Houwei Cao, Jian Li, and Yong Liu. "Decentralized Federated Learning with Model Caching on Mobile Agents." arXiv preprint arXiv:2408.14001 (2024).
  https://arxiv.org/abs/2408.14001 (**Oral Presentation at AAAI 2025**)
