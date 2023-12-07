import torchvision
from utils_n2n.utils import *
from model.losses import training_loss
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Trainer class
    """
    def __init__(self, network, pass_network, criterion, optimizer, scheduler, config,
                 trainloader, validloader, epoch_frac_save=100):
        self.config = config
        self.trainloader = trainloader
        self.validloader = validloader
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.net = lambda y: pass_network(y, network)
        self.cuda_avail = torch.cuda.is_available()

        self.epoch_frac_save = epoch_frac_save
        self.nshow = trainloader.__len__() // epoch_frac_save if self.cuda_avail else 1

        self.lambda_param = self.config['mask']['us_fac_lambda']
        self.alpha_param = torch.tensor([float(self.config['noise']['alpha'])])

        self.logdir = config['network']['save_loc']
        self.writer = SummaryWriter(self.logdir)

    def train(self):
        for epoch in range(self.config['optimizer']['epochs']):
            self._train_epoch(epoch)
            valid_loss, matched_loss, y0, y0_est = self._valid_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            print('Validation loss is {:e}'.format(valid_loss))
            print('Matched loss is {:e}'.format(matched_loss))
            self.writer.add_scalar('Validation_losses/MSE_loss', valid_loss, epoch)
            self.writer.add_scalar('Validation_losses/MSE_loss_matched', matched_loss, epoch)
            torch.save(self.network.state_dict(), self.logdir + '/state_dict')

            self._save_examples(epoch, y0, y0_est)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        """
        print('training...')
        print('epoch|minibatch|Label loss')
        j = 0
        running_loss = 0
        self.network.train()
        for i, data in enumerate(self.trainloader, 0):
            self.optimizer.zero_grad()

            loss, _ = training_loss(data, self.net, self.config, self.criterion,
                                          self.lambda_param, self.alpha_param)

            if not torch.isnan(loss):
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            else:
                warnings.warning('NaN in loss')

            if i % self.nshow == (self.nshow - 1):  # print every nshow mini-batches
                print('%d    |   %d    |%e ' %
                      (epoch + 1, i + 1, running_loss / self.nshow))
                self.writer.add_scalar('Training_losses/MSE_loss', running_loss / self.nshow, epoch * self.epoch_frac_save + j)
                self.writer.add_scalar('Training_losses/alpha_param', self.alpha_param, epoch * self.epoch_frac_save + j)
                self.writer.add_scalar('Training_losses/lambda_param', self.lambda_param, epoch * self.epoch_frac_save + j)

                if not self.cuda_avail:  # truncate training when on my machine
                    break

                running_loss = 0
                j += 1

        return 1

    def _valid_epoch(self):
        """
        Validate after training an epoch
        """
        running_loss_val = 0
        running_matched_loss = 0
        with torch.no_grad():
            print('validation...')
            self.network.eval()
            for i, data in enumerate(self.validloader, 0):
                loss, y0_est = training_loss(data, self.net, self.config, self.criterion, self.lambda_param, self.alpha_param)
                running_matched_loss += loss.item()
                y0 = data[0].to(y0_est)
                running_loss_val += self.criterion(y0_est, y0)
                if not self.cuda_avail:
                    if i % self.nshow == (self.nshow - 1):
                        break

        return running_loss_val / (i + 1), running_matched_loss / (i + 1), y0, y0_est

    def _save_examples(self, epoch, y0, y0_est):
        if epoch == 0:
            x0 = kspace_to_rss(y0[0:4])
            x0 = torchvision.utils.make_grid(x0).detach().cpu()
            self.xnorm = torch.max(x0)
            self.writer.add_image('ground_truth', x0 / self.xnorm)

        x0_est = kspace_to_rss(y0_est[0:4])
        x0_est = torchvision.utils.make_grid(x0_est).detach().cpu()
        self.writer.add_image('estimate', x0_est / self.xnorm)
