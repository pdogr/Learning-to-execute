import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from base import Optimizer
from torchvision.utils import make_grid
from math import sqrt


class Trainer(BaseTrainer):
    def __init__(
            self,
            model, criterion, metric_ftns, optimizer: Optimizer,
            config, data_loader, valid_data_loader, len_epoch=None,
            log_step=2):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = log_step
        if len_epoch is None:
            self.data_loader_iter = data_loader
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader_iter = inf_loop(self.data_loader)
            self.valid_loader_iter = inf_loop(self.valid_data_loader)
            self.len_epoch = len_epoch
            self.valid_len_epoch = 53
        self.train_metrics = MetricTracker(
            'train_loss',
            *['train_'+m.__name__ for m in self.metric_ftns],
            writer=self.writer)
        self.valid_metrics = MetricTracker(
            'val_loss',
            *['val_'+m.__name__ for m in self.metric_ftns],
            writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.logger.info(epoch)
        self.logger.info("Current difficulty: {}".format(
            self.data_loader.gen.difficulty))
        for batch_idx, batch in enumerate(self.data_loader_iter):

            (input_variables, input_lengths, target) = batch
            self.optimizer.zero_grad()
            output, _, sequence_info = self.model.forward(
                input=input_variables, input_lens=input_lengths,
                target=target,
                teacher_forcing_ratio=0.5)
            loss = self.criterion.__call__(
                output,
                target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # set train metrics
            self.train_metrics.update('train_loss', loss.item())
            for metric in self.metric_ftns:
                self.train_metrics.update(
                    'train_' + metric.__name__,
                    metric(output, target, sequence_info))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.8f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))
            if self.train_metrics.result().get('train_Accuracy')>=0.93:
                self.optimizer.step_lr(self.train_metrics.result().get('train_loss'), epoch)
            if batch_idx == self.len_epoch:
                break

        history = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if val_log.get('val_Accuracy') >= 0.95 and self.train_metrics.result().get('train_Accuracy') >= 0.80:
                self.logger.info("Increasing difficulty")
                self.data_loader.gen.increase_difficulty()
                self.valid_data_loader.gen.increase_difficulty()
                
                self.logger.info("Current difficulty: {}".format(
                    self.data_loader.gen.difficulty))
            history.update(**{k: v for k, v in val_log.items()})

        self.optimizer.step_lr(history['train_loss'], epoch)
        return history

    def _valid_epoch(self, epoch):
        """
                Validate after training an epoch

                :param epoch: Integer, current training epoch.
                :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader_iter):

                (input_variables, input_lengths, target) = batch
                output, _, sequence_info = self.model.forward(
                    input=input_variables, input_lens=input_lengths,
                    target=target)
                loss = self.criterion.__call__(
                    output,
                    target)

                # set writer step
                self.writer.set_step(
                    (epoch - 1) * self.len_epoch + batch_idx,
                    'valid')

                # set val metrics
                self.valid_metrics.update('val_loss', loss.item())
                for metric in self.metric_ftns:
                    self.valid_metrics.update(
                        'val_'+metric.__name__,
                        metric(output, target, sequence_info))
                if batch_idx == self.valid_len_epoch:
                    break

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
