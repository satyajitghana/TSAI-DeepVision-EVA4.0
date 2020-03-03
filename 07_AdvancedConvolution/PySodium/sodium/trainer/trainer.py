from sodium.utils import setup_logger
from sodium.base import BaseTrainer

from tqdm.auto import tqdm, trange
import torch

logger = setup_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, config, device, train_loader, test_loader, lr_scheduler=None):
        super().__init__(model, loss, optimizer, config, device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch: int) -> dict:

        self.model.train()  # set the model in training mode

        correct = 0
        processed = 0

        pbar = tqdm(self.train_loader)

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.loss(output, target)

            loss.backward()

            self.optimizer.step()

            # pred = output.argmax(dim=1, keepdim=True)
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # processed += len(data)

            # pbar.set_description(
            #     desc=f'epoch={epoch} loss={loss.item():.10f} batch_id={batch_idx} accuracy={100*correct/processed:0.3f}')

            pbar.set_description(
                desc=f'epoch={epoch} loss={loss.item():.10f} batch_id={batch_idx}')

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()

        # check if there's a lr scheduler
        if (self.lr_scheduler is not None) and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_scheduler.step()

        self._test_epoch(epoch)  # test this epoch

    def _test_epoch(self, epoch: int) -> dict:

        self.model.eval()  # set the model in evaluation mode

        # test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == labels).sum().item()

        #         test_loss += self.loss(output, target, reduction='sum').item()
        #         pred = output.argmax(dim=1, keepdim=True)
        #         correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(self.test_loader.dataset)

        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #     test_loss, correct, len(self.test_loader.dataset),
        #     100. * correct / len(self.test_loader.dataset)))
        print(f'Test set: Accuracy: {100 * correct / total}')
