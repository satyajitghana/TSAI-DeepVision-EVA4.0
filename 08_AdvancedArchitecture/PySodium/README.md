# PySodium V0.0.1

## Usage

1. Install the PySodium Library

`pip install git+https://github.com/satyajitghana/PySodium.git#egg=sodium`

2. Create a config.yml

```yaml
name: CIFAR10_MyNet
save_dir: saved/
seed: 1
target_device: 0

arch:
    type: CIFAR10Model
    args: {}

augmentation:
    type: CIFAR10Transforms
    args: {}

data_loader:
    type: CIFAR10DataLoader
    args:
        batch_size: 64
        data_dir: data/
        nworkers: 4
        shuffle: True

criterion: cross_entropy_loss

lr_scheduler:
    type: OneCycleLR
    args:
        max_lr: 0.1

optimizer:
    type: SGD
    args:
        lr: 0.008
        momentum: 0.95

training:
    epochs: 50
```

3. Run the Model !

```python
# import my baby-library
from sodium.utils import load_config
import sodium.runner as runner

# create a runner
config = load_config('config.yml', tsai_mode=False)

# train the network
runner.train(config)

# plot metrics
runner.plot()
```

## NOTE

if you are using the library on a terminal, you can use the main.py and call

`python main.py --config=config.yml`


---

Made with ❤ by shadowleaf.satyajit