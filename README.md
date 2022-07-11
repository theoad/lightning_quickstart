<div align="center">    
 
# Lightning Quick

</div>

## Installation 
First, install dependencies
```bash
# You'll need pytorch 1.8 or newer.
# Visit https://pytorch.org/ to get the relevant installation command. I used:
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install the module
pip install git+https://github.com/theoad/lightning_quick
 ```   

## Usage
### Inherit from BaseModule and quickly create a powerful module.
```python
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.classification.accuracy import Accuracy

from lightning_quick.modules import BaseModule
from lightning_quick.nets.mlp import MLP
from lightning_quick.data.mnist import MNISTDatamodule


class Classification(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = self.cross_entropy  # TODO: Assign here the loss function

    # TODO: Change project name and hparams to display in the name of the run
    project_name = 'mnist_classification'
    run_hparam_disp = ['learning_rate', 'hidden_size']
    datamodule_cls = MNISTDatamodule  # TODO: Define here on which data to train/test

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Every argument added here is automatically accessible through self.hparam !
        parent_parser = BaseModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--img_size", type=int, nargs="*", default=[1, 32, 32])
        parser.add_argument("--hidden_size", type=int, default=1024)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--num_layers", type=int, default=3)
        return parent_parser

    def batch_preprocess(self, batch):
        # This method should return a dictionary with at least 'samples' and 'targets'
        samples, target = batch
        return {'samples': samples, 'targets': target}

    def cross_entropy(self, batch, batch_idx):
        samples, target = batch['samples'], batch['targets']
        preds = self(samples)
        loss = F.cross_entropy(preds, target, reduction='mean')
        logs = self.train_metrics(preds, target)
        logs['cross_entropy'] = loss.item()
        return loss, logs

    def forward(self, samples):
        return self.model(samples)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def _init_modules(self):
        # TODO: Add other models, preprocessing and so on
        self.model = MLP(
            self.hparams.img_size,
            self.hparams.num_classes,
            self.hparams.hidden_size,
            self.hparams.num_layers,
        )

    def _init_metrics(self):
        # TODO: Add other metrics (see lightning_quick.modules.classification_example.py)
        return Accuracy(num_classes=self.hparams.num_classes)

    @classmethod
    def _init_logger(cls, args):
        # TODO: Replace with the logger of your heart (any logger supported by lightning)
        return pl_loggers.WandbLogger(
            name=args.run_name,
            project=args.proj_name,
            entity='your-team-name'
        )

    @classmethod
    def _init_callbacks(cls, args):
        # TODO: Add your callbacks here
        return [
            pl.callbacks.ModelCheckpoint(
                dirpath=f'checkpoints/{args.run_name}',
                filename='{epoch}-{val_accuracy:.3f}',
                monitor='val_accuracy',
                save_top_k=5,
                save_last=True,
                mode='max'
            ),
        ]


if __name__ == "__main__":
    Classification.train_routine()
```

### Train
```bash
# Single GPU
python classification.py --gpus 1 --hidden_size 128 --num_layers 5

# Multi GPU
python classification.py --gpus 8 --hidden_size 128 --num_layers 5

# Debug
python classification.py --gpus 1 --fast_dev_run

# Help
python classification.py --help
```

### Test
```bash
python classification.py --gpus 1 --test --checkpoint mnist_classification/2mwq89zf/checkpoints/last.ckpt
```

### Inference
```python
import torch
from lightning_quick.nets.mlp import MLP

checkpoint = torch.load("mnist_classification/2mwq89zf/checkpoints/last.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]
model_weights = checkpoint["state_dict"]

# MLP is a simple nn.Module !
model = MLP(**hyper_parameters)

# update keys by dropping `model` (in Classification kept the MLP in the field self.model).
for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)

model.load_state_dict(model_weights)
model.eval()
x = torch.randn(1, 1, 32, 32)

with torch.no_grad():
    y_hat = model(x)
```

### Reuse weights
```python
from lightning_quick.modules import BaseModule
from lightning_quick.nets.mlp import MLP

class ModelReuse(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ...

    def _init_modules(self):
        self.model = MLP(
            self.hparams.img_size,
            self.hparams.num_classes,
            self.hparams.hidden_size,
            self.hparams.num_layers,
        )
        self.cnn_backbone = ... # Another architecture
    # ...
```
```bash
# Will automatically initialize the weights of the MLP in self.model with the checkpoint
python model_reuse.py --gpus 1 --checkpoint mnist_classification/2mwq89zf/checkpoints/last.ckpt
```

