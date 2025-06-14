""" BaseModel
"""
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

class BaseModel_Aug():
    """ Base Model for ocr-gan
    """
    def __init__(self, opt, data, classes):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def seed(self, seed_value):
        """ Seed 

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input_lap.resize_(input[0].size()).copy_(input[0])
            self.input_res.resize_(input[1].size()).copy_(input[1])
            self.fake_aug.resize_(input[2].size()).copy_(input[2])
            self.gt.resize_(input[3].size()).copy_(input[3])
            self.label.resize_(input[3].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input_lap.resize_(input[0].size()).copy_(input[0])
                self.fixed_input_res.resize_(input[1].size()).copy_(input[1])

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item())])

        return errors

    ##
    def reinit_d(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input_lap.data + self.input_res.data
        fakes = self.fake.data
        fake_lap = self.fake_lap.data
        fake_res = self.fake_res.data
        return reals, fakes, fake_lap, fake_res

    ##
    def save_weights(self, epoch:int, is_best:bool=False):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(
            self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")

    def load_weights(self, epoch=None, is_best:bool=False, path=None):
        """ Load pre-trained weights of NetG and NetD

        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})

        Raises:
            Exception -- [description]
            IOError -- [description]
        """

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        if is_best:
            fname_g = f"netG_best.pth"
            fname_d = f"netD_best.pth"
        else:
            fname_g = f"netG_{epoch}.pth"
            fname_d = f"netD_{epoch}.pth"

        if path is None:
            path_g = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_g}"
            path_d = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_d}"

        # Load the weights of netg and netd.
        print('>> Loading weights...')
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.data.train.dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fake_lap, fake_res = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fake_lap, fake_res)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fake_lap, fake_res)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training {self.name} on {self.classes} to detect {self.opt.note}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)
        return best_auc

    ##
    def test(self):
        """ Test model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        import time
        with torch.no_grad():
            # Load the weights of netg and netd if requested.
            if getattr(self.opt, "load_weights", False):
                path = f"./output/{self.name.lower()}/{self.opt.dataset}/train/weights/netG.pth"
                pretrained_dict = torch.load(path, map_location=self.device)['state_dict']
                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big tensors for the test set.
            num_samples = len(self.data.valid.dataset)
            self.an_scores = torch.zeros(size=(num_samples,), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(num_samples,), dtype=torch.long, device=self.device)
            self.latent_i = torch.zeros(size=(num_samples, self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o = torch.zeros(size=(num_samples, self.opt.nz), dtype=torch.float32, device=self.device)

            test_losses_g = []
            test_losses_d = []
            test_losses_adv = []
            test_losses_con = []
            test_losses_lat = []
            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                # Calculate error (anomaly score)
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                # Fill big tensors
                start_idx = i * self.opt.batchsize
                end_idx = start_idx + error.size(0)
                self.an_scores[start_idx:end_idx] = error.reshape(error.size(0))
                self.gt_labels[start_idx:end_idx] = self.gt.reshape(error.size(0))
                self.latent_i[start_idx:end_idx, :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o[start_idx:end_idx, :] = latent_o.reshape(error.size(0), self.opt.nz)

                # If possible, calculate and store losses (if your model tracks these in test)
                if hasattr(self, "err_g"):
                    test_losses_g.append(float(self.err_g))
                if hasattr(self, "err_d"):
                    test_losses_d.append(float(self.err_d))
                if hasattr(self, "err_g_adv"):
                    test_losses_adv.append(float(self.err_g_adv))
                if hasattr(self, "err_g_con"):
                    test_losses_con.append(float(self.err_g_con))
                if hasattr(self, "err_g_lat"):
                    test_losses_lat.append(float(self.err_g_lat))

                self.times.append(time_o - time_i)

                # Save test images
                if getattr(self.opt, "save_test_images", False):
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, *_ = self.get_current_images()
                    vutils.save_image(real, f'{dst}/real_{i+1:03d}.eps', normalize=True)
                    vutils.save_image(fake, f'{dst}/fake_{i+1:03d}.eps', normalize=True)

            # Measure inference time (average over first 100 if available)
            self.times = np.array(self.times)
            avg_time_ms = np.mean(self.times[:100]) * 1000 if len(self.times) >= 100 else np.mean(self.times) * 1000

            # Normalize anomaly scores
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels.cpu().numpy(), self.an_scores.cpu().numpy())
            # Add more metrics if needed, e.g. threshold-based accuracy, F1, etc.

            # Aggregate mean losses
            mean_err_g = np.mean(test_losses_g) if test_losses_g else None
            mean_err_d = np.mean(test_losses_d) if test_losses_d else None
            mean_err_g_adv = np.mean(test_losses_adv) if test_losses_adv else None
            mean_err_g_con = np.mean(test_losses_con) if test_losses_con else None
            mean_err_g_lat = np.mean(test_losses_lat) if test_losses_lat else None

            # Compose performance dict
            performance = OrderedDict([
                ('Epoch', getattr(self, 'epoch', None)),
                ('Avg Run Time (ms/batch)', avg_time_ms),
                ('AUC', auc),
                ('Mean Generator Loss', mean_err_g),
                ('Mean Discriminator Loss', mean_err_d),
                ('Mean Adv Loss', mean_err_g_adv),
                ('Mean Context Loss', mean_err_g_con),
                ('Mean Latent Loss', mean_err_g_lat),
                ('Dataset', self.opt.dataset),
                ('Model Name', self.name if hasattr(self, 'name') else None),
                ('Num Test Samples', num_samples),
                ('Batch Size', self.opt.batchsize),
                ('Latent Dim', getattr(self.opt, 'nz', None)),
                ('Metric', getattr(self.opt, 'metric', None)),
                ('Test Time', time.strftime('%Y-%m-%d %H:%M:%S'))
            ])

            # Optionally, visualize or log performance
            if getattr(self.opt, "display_id", 0) > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / num_samples
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            return performance

        ##
        def update_learning_rate(self):
            """ Update learning rate based on the rule provided in options.
            """
    
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('   LR = %.7f' % lr)        