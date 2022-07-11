import os
from typing import List
from collections import OrderedDict
from argparse import ArgumentParser
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import model_summary


class BaseModule(pl.LightningModule, ABC):
    """`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    Example:

        ::

            - Print help & arguments of the module
              python my_module.py --help

            - Fast run (train on single batch without logger)
              python my_module.py --gpus 1 --fast_dev_run

            - Train using cmd line
              python my_module.py --gpus 1 --arg1 5 --arg2 true

            - Test (inference)
              python my_module.py --gpus 1 --test -ckpt <path/to/checkpoint>

    TODO:
        - Support config file (like Lighting CLI): python --config config.yaml
        - Support push to torch.hub
        - Add documentation for external dataset (as pl.Datamodule or pl.Callback)
        - Improve checkpoint loading.

    .. _TheoA: https://github.com/theoad
    """

    project_name = ...  # assign project name ex: 'lightning_quick'
    run_hparam_disp = ...  # add list of hyper-params to display in run names ex: ['learning_rate', 'hidden_size']
    datamodule_cls = ...  # assign a datamodule class (yes, the class itself, not some instantiated object)

    def __init__(self, *args, **kwargs):
        """
        DO Initialize here::

            - Fields that don't contain nets.Parameters
            - Assign loss function to self.loss

        DON'T Initialize here::

            - nets.Modules - instead override self._init_modules()
            - Datasets - instead override self._init_datasets()
            - Metrics - instead override self._init_metrics()
        """
        super().__init__()
        self.save_hyperparameters(kwargs)

        self.loss = ...  # assign the loss function. (can be a list for alternate updates like GANs)

        # evaluation
        self.train_metrics, self.val_metrics, self.test_metrics = None, None, None

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Module arguments to be parsed and added to `self.hparams`.

        Example:

            Say you want to add the argument `hidden_size` to the model and access it in `self._init_module()`.
            just override this method::
                @staticmethod
                def add_model_specific_args(parent_parser):
                    parent_parser = BaseModule.add_model_specific_args(parent_parser)
                    parser.add_argument("--hidden_size", type=int, default=1024)
                    return parent_parser


            Then access `hidden_size` everywhere in your module using `self.hparams.hidden_size`.

            That's it ! No need to update `self.__init__()`, no need to set `self.hidden_size = hidden_size` and so on...
        """
        parser = parent_parser.add_argument_group("Run")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility")
        parser.add_argument("--test", action="store_true", default=False, help="Set this flag in order to skip training")
        parser.add_argument("-ckpt", "--checkpoint", type=str, default=None, help="path to pretrained model."
                            " Use for test only. To resume training, use --resume_from_checkpoint")
        return parent_parser

    @abstractmethod
    def _init_modules(self):
        """
        Init method for nets.Modules.

        Initialize here any modules as you'd have done in self.__init__().

        Example:

            ::

                def _init_modules(self):
                in_flatten_size = torch.prod(self.hparams.input_size)
                hidden_size = self.hparams.hidden_size
                num_classes = self.hparams.num_classes

                self.model = nets.Sequential(
                    nets.Flatten(),
                    nets.Linear(in_flatten_size, hidden_size),
                    nets.Linear(hidden_size, hidden_size),
                    nets.Linear(hidden_size, num_classes),
                )
        """
        pass

    @abstractmethod
    def _init_metrics(self):
        """
        Init method for train, validation and test metrics. Should return::

            - torchmetrics.Metric
            - torchmetrics.MetricCollection
            - Any derived class from torchmetrics.Metric

        Example:

            ::

                def _init_metrics(self):
                    from torchmetrics.classification.accuracy import Accuracy
                    from torchmetrics.classification.precision_recall import Precision, Recall
                    return MetricCollection({
                        "accuracy": Accuracy(),
                        "top-5 accuracy": Accuracy(top_k=5),
                        "precision": Precision(),
                        "recall": Recall(),
                    })
        """
        pass

    @classmethod
    def _init_logger(cls, args) -> pl_loggers.LightningLoggerBase:
        """
        Init method for the logger.

        Example:

            ::

                def _init_logger(cls, args) -> pl_loggers.LightningLoggerBase:
                    return pl_loggers.TensorBoardLogger(
                        save_dir=args.log_dir,
                        name=args.proj_name,
                        sub_dir=args.run_name,
                    )
        """
        pass

    @classmethod
    def _init_callbacks(cls, args) -> List[pl.callbacks.base.Callback]:
        """
        Init method for callbacks.

        Example:

            ::

                def _init_callbacks(self):
                checkpoint_callback = ModelCheckpoint(
                    dirpath=args.logdir,
                    filename='{epoch}-{val_accuracy:.3f}',
                    monitor='val_accuracy',
                    save_top_k=5,
                    every_n_val_epochs=2,
                    save_last=True,
                    mode='max'
                )
                return [checkpoint_callback]
        """
        return []

    @staticmethod
    def _trainer_kwargs():
        """
        Set default pl.Trainer arguments.

        Each default argument can be overloaded by command-line/config argument:

        Example:

            ::

                python base_module.py --strategy ddp
        """
        return dict()

    @abstractmethod
    def batch_preprocess(self, batch):
        """
        Pre-process batch before feeding to self.loss and computing metrics.

        :param batch: Output of self.train_ds.__getitem__()
        :return: Dictionary (or any key-valued container) with at least the keys `samples` and `targets`
        """
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.loss[optimizer_idx] if hasattr(self.loss, '__getitem__') else self.loss
        loss, logs = loss(self.batch_preprocess(batch), batch_idx)
        self.log_dict(logs, sync_dist=True, prog_bar=True)
        return loss

    def _update_metrics(self, batch, batch_idx, mode):
        pbatch = self.batch_preprocess(batch)
        preds, targets = self(pbatch['samples']), pbatch['targets']
        getattr(self, f'{mode}_metrics').update(preds, targets)
        pbatch['preds'] = preds
        return pbatch

    def _log_metric(self, mode):
        res = getattr(self, f'{mode}_metrics').compute()
        getattr(self, f'{mode}_metrics').reset()
        return self.log_dict(res, sync_dist=True, logger=True)

    def _load_attr_state_dict(self, attr):
        if getattr(self, attr) is None or not isinstance(getattr(self, attr), torch.nn.Module):
            return
        assert os.path.exists(self.hparams.checkpoint), f'Error: Path {self.hparams.checkpoint} not found.'
        checkpoint = torch.load(self.hparams.checkpoint)
        state_dict = OrderedDict()
        found = False
        for key, val in checkpoint['state_dict'].items():
            if attr == key.split('.')[0]:
                found = True
                state_dict['.'.join(key.split('.')[1:])] = val
        if found:
            getattr(self, attr).load_state_dict(state_dict, strict=False)

    def setup(self, stage=None):
        self._init_modules()
        metrics = self._init_metrics()
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Checkpoint loading. Let you load partial attributes.
        if self.hparams.checkpoint is not None:
            for attr, _ in self.named_children():
                self._load_attr_state_dict(attr)

    def _prepare_metrics(self, mode):
        for metric in getattr(self, f'{mode}_metrics').values():
            if hasattr(metric, 'prepare_metric'):
                metric.prepare_metric(self)

    def on_fit_start(self) -> None:
        return self._prepare_metrics('val')

    def on_test_start(self) -> None:
        return self._prepare_metrics('test')

    def validation_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._log_metric('val')

    def test_epoch_end(self, outputs):
        return self._log_metric('test')

    @classmethod
    def _parse_args(cls):
        parser = ArgumentParser()
        parser = cls.add_model_specific_args(parser)
        parser = cls.datamodule_cls.add_argparse_args(parser)
        parser = pl.Trainer.add_argparse_args(parser)
        args = parser.parse_args()
        args.proj_name = cls.project_name
        args.run_name = f'{cls.__name__}-' + cls.hparams2desc(parser, args, cls.run_hparam_disp, verbose='vvv')
        return args

    @staticmethod
    def hparams2desc(parser, args, hparam_names, delimiter='-', verbose='vvv'):
        desc = ''
        for action in parser._get_optional_actions():
            if len(set(hparam_names) & set(map(lambda s: s.replace('-', ''), action.option_strings))) > 0:
                arg_short_name = action.option_strings[0].replace('-', '')
                arg_name = action.option_strings[-1].replace('-', '')
                desc += delimiter
                if verbose == 'v': pass
                elif verbose == 'vv': desc += arg_short_name
                elif verbose == 'vvv': desc += arg_name + '='
                else: raise ValueError(f"verbose choice is ['v', 'vv', 'vvv']. Provided verbose={verbose}")
                desc += str(getattr(args, arg_name))
        return desc

    @classmethod
    def train_routine(cls):
        args = cls._parse_args()

        pl.seed_everything(args.seed)
        gpu_num = (args.gpus if torch.cuda.is_available() else 0)
        logger = None

        if not args.fast_dev_run:
            logger = cls._init_logger(args)
            logger.log_hyperparams(args)

        callbacks = cls._init_callbacks(args)
        model = cls(**vars(args))

        datamodule: pl.LightningDataModule = cls.datamodule_cls.from_argparse_args(args)

        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args,
            gpus=gpu_num,
            logger=logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False) if gpu_num > 1 else None,
            max_epochs=-1,
            **cls._trainer_kwargs(),
        )

        if not args.test:
            trainer.tune(model, datamodule=datamodule)
            trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

    def __repr__(self):
        return model_summary.ModelSummary(self)
