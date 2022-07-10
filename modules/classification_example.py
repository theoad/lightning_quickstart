import modules
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from nets.mlp import MLP
from data.mnist import MNISTDatamodule
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall


class Classification(modules.BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = self.cross_entropy

    project_name = 'mnist_classification'
    run_hparam_disp = ['learning_rate', 'hidden_size']
    datamodule_cls = MNISTDatamodule

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = modules.BaseModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--img_size", type=int, nargs="*", default=[1, 32, 32])
        parser.add_argument("--hidden_size", type=int, default=1024)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--betas", type=int, nargs="*", default=[0.9, 0.999])
        return parent_parser

    def batch_preprocess(self, batch):
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

    def _update_metrics(self, batch, batch_idx, mode):
        pbatch = super()._update_metrics(batch, batch_idx, mode)
        pbatch['preds'] = pbatch['preds'].argmax(dim=-1)
        return pbatch

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=1e-8
        )

    def _init_modules(self):
        self.model = MLP(
            self.hparams.img_size,
            self.hparams.num_classes,
            self.hparams.hidden_size,
            self.hparams.num_layers,
        )

    def _init_metrics(self):
        return MetricCollection(dict(
            accuracy=Accuracy(num_classes=self.hparams.num_classes),
            top5_accuracy=Accuracy(top_k=5, num_classes=self.hparams.num_classes),
            precision=Precision(num_classes=self.hparams.num_classes, average='macro'),
            recall=Recall(num_classes=self.hparams.num_classes, average='macro'),
        ))

    @classmethod
    def _init_logger(cls, args):
        return pl_loggers.WandbLogger(
            name=args.run_name,
            project=args.proj_name,
            entity='gip'
        )

    @classmethod
    def _init_callbacks(cls, args):
        return [
            pl.callbacks.ModelCheckpoint(
                dirpath=None,
                filename='{epoch}-{val_accuracy:.3f}',
                monitor='val_accuracy',
                save_top_k=5,
                save_last=True,
                mode='max'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode="max",
                patience=5,
                verbose=True,
            )
        ]

    @staticmethod
    def _trainer_kwargs():
        kwargs = modules.BaseModule._trainer_kwargs()
        kwargs['benchmark'] = True
        return kwargs


if __name__ == "__main__":
    Classification.train_routine()
