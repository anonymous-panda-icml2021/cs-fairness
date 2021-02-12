"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

import copy
import heapq
import torch
import numpy as np
import utils
import scipy.fftpack as fftpack
import sys
import os
import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import models as nvp_model
from celebA_stylegan2.model import Generator
from glow_256 import model as glow_model
from ncsnv2.models import get_sigmas, ema
from ncsnv2.models.ncsnv2 import NCSNv2, NCSNv2Deepest

import tensorflow as tf
import torch.nn.functional as F

from celebA_utils import *
from lpips import PerceptualLoss

from PULSE import PULSE
import yaml
import argparse
import time
from include import fit, decoder

def get_measurements_torch(x_hat_batch, A, measurement_type, hparams):
    batch_size = hparams.batch_size
    if measurement_type == 'project':
        y_hat_batch = x_hat_batch
    elif measurement_type == 'gaussian':
        y_hat_batch = torch.mm(xhat_batch.view(batch_size,-1), A)
    elif measurement_type == 'circulant':
        sign_pattern = torch.Tensor(hparams.sign_pattern).to(hparams.device)
        y_hat_batch = utils.partial_circulant_torch(x_hat_batch, A, hparams.train_indices,sign_pattern)
    elif measurement_type == 'superres':
        x_hat_reshape_batch = x_hat_batch.view((batch_size,) + hparams.image_shape)
        y_hat_batch = F.avg_pool2d(x_hat_reshape_batch, hparams.downsample)
    return y_hat_batch.view(batch_size, -1)

def realnvp_map_estimator(hparams):

    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()

    mse = torch.nn.MSELoss(reduction='none')
    annealed = hparams.annealed

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            # z = np.sqrt(hparams.ploss_weight) * model.sample_z(n=hparams.batch_size)/0.7
            if hparams.fixed_init:
                z0 = torch.load('initializations/torch_z_init.pt')[:hparams.batch_size]
                z = z0.detach() * hparams.zprior_sdev
                z = z.to(hparams.device)
            else:
                z =  model.sample_z(n=hparams.batch_size, temp=hparams.zprior_init_sdev)
            z.requires_grad_()
            opt = utils.get_optimizer(z, hparams.learning_rate, hparams)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                x_hat_batch = model.postprocess(model.inverse(z))
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                # p_loss_batch, _, _ = model.log_prob(xhat_batch)
                p_loss_batch = torch.norm(z,dim=-1).pow(2)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = hparams.num_measurements / (2 * sigma**2)

                if hparams.zprior_weight is None:
                    if hparams.zprior_sdev != 0:
                        zprior_weight = 1/(2 * (hparams.zprior_sdev **2))
                    else:
                        zprior_weight = 0
                else:
                    zprior_weight = hparams.zprior_weight
                total_loss_batch = mloss_weight * m_loss_batch + zprior_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                total_loss = total_loss_batch.sum()

                total_loss.backward()

                opt.step()
                scheduler.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss {} p_loss {}'
                print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), m_loss.item(), p_loss.item()))

            x_hat_batch = model.postprocess(model.inverse(z))
            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch = torch.norm(z,dim=-1).pow(2)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    + zprior_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
            best_keeper_z.report(z.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator

def realnvp_xmap_estimator(hparams):

    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()

    mse = torch.nn.MSELoss(reduction='none')
    annealed = hparams.annealed

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            # z = np.sqrt(hparams.ploss_weight) * model.sample_z(n=hparams.batch_size)/0.7
            if hparams.fixed_init:
                z0 = torch.load('initializations/torch_z_init.pt')[:hparams.batch_size]
                z = z0.detach() * hparams.zprior_sdev
                z = z.to(hparams.device)
            else:
                z =  model.sample_z(n=hparams.batch_size, temp=hparams.zprior_init_sdev)
                x0 = model.postprocess(model.inverse(z))

            x_hat_batch = x0.detach().to(hparams.device)
            x_hat_batch.requires_grad_()
            opt = utils.get_optimizer(x_hat_batch, hparams.learning_rate, hparams)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                p_loss_batch, _, _ = model.log_prob(x_hat_batch)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = hparams.num_measurements / (2 * sigma**2)

                if hparams.ploss_weight is None:
                    ploss_weight = 1.
                else:
                    ploss_weight = hparams.ploss_weight
                total_loss_batch = mloss_weight * m_loss_batch - ploss_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                total_loss = total_loss_batch.sum()

                total_loss.backward()

                opt.step()
                scheduler.step()
                x_hat_batch.data.clamp_(0,1)

                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss {} p_loss {}'
                print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), m_loss.item(), p_loss.item()))

            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch, _, _ = model.log_prob(x_hat_batch)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    - ploss_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
            z = model(model.preprocess(x_hat_batch))
            best_keeper_z.report(z.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator

def noisy_estimator(hparams):

    assert hparams.ploss_weight > 0

    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()

    mse = torch.nn.MSELoss(reduction='none')

    def estimator(A_train, y_batch_train, A_eval, y_batch_eval, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        y = torch.Tensor(y_batch_train).cuda()

        best_keeper = utils.BestKeeper(hparams)

        for i in range(hparams.num_random_restarts):
            z =  model.sample_z(n=hparams.batch_size, temp=hparams.zprior_init_sdev)
            z.requires_grad_()
            opt = utils.get_optimizer(z, hparams.learning_rate, hparams)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                xhat_batch = model.postprocess(model.inverse(z))
                m_loss_batch = mse(torch.mm(xhat_batch.view(hparams.batch_size,-1), A), y).sum(dim=1)
                p_loss_batch, _, _ = model.log_prob(xhat_batch)
                p_loss_batch = p_loss_batch.view(-1)
                total_loss_batch = hparams.mloss_weight * m_loss_batch \
                        - hparams.ploss_weight * p_loss_batch

                m_loss = m_loss_batch.mean()
                p_loss = p_loss_batch.mean()
                total_loss = total_loss_batch.mean()

                total_loss.backward()
                z.grad += np.sqrt(hparams.gradient_noise_weight/hparams.n_input) * torch.randn(z.shape).cuda()
                opt.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss {} p_loss {}'
                print(logging_format.format(i, j, hparams.learning_rate, total_loss.item(), m_loss.item(), p_loss.item()))

            xhat_batch = model.postprocess(model.inverse(z))
            m_loss_batch = mse(torch.mm(xhat_batch.view(hparams.batch_size,-1), A), y).sum(dim=1)
            p_loss_batch, _, _ = model.log_prob(xhat_batch)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = hparams.mloss_weight * m_loss_batch \
                    - hparams.ploss_weight * p_loss_batch
            best_keeper.report(xhat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best()

    return estimator


def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
    x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, hparams)

    # Create the discriminator
    prob, restore_dict_discrim, restore_path_discrim = celebA_model_def.dcgan_discrim(x_hat_batch, hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')
    else:
        measurement_is_sparse = (hparams.measurement_type in ['inpaint', 'superres'])
        y_hat_batch = tf.matmul(x_hat_batch, A, b_is_sparse=measurement_is_sparse, name='y2_batch')

    # define all losses
    m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch =  tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
    d_loss1_batch = -tf.log(prob)
    d_loss2_batch =  tf.log(1-prob)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch \
                     + hparams.dloss1_weight * d_loss1_batch \
                     + hparams.dloss2_weight * d_loss2_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    d_loss1 = tf.reduce_mean(d_loss1_batch)
    d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(A_train, y_batch_train, A_eval, y_batch_eval, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)

        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_train}
        else:
            feed_dict = {A: A_train, y_batch: y_batch_train}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} d_loss1 {} d_loss2 {}'
                print(logging_format.format(i, j, lr_val, total_loss_val, m_loss1_val, m_loss2_val, zp_loss_val, d_loss1_val, d_loss2_val))

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator





def realnvp_langevin_estimator(hparams):


    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()

    mse = torch.nn.MSELoss(reduction='none')
    annealed = hparams.annealed

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            # z = np.sqrt(hparams.ploss_weight) * model.sample_z(n=hparams.batch_size)/0.7
            if hparams.fixed_init:
                z0 = torch.load('initializations/torch_z_init.pt')[:hparams.batch_size]
                z = z0.clone() * hparams.zprior_sdev
                z = z.to(hparams.device)
            else:
                z =  model.sample_z(n=hparams.batch_size, temp=hparams.zprior_init_sdev)
            z.requires_grad_()
            opt = utils.get_optimizer(z, hparams.learning_rate, hparams)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                x_hat_batch = model.postprocess(model.inverse(z))
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                # p_loss_batch, _, _ = model.log_prob(xhat_batch)
                p_loss_batch = torch.norm(z,dim=-1).pow(2)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = hparams.num_measurements / (2 * sigma**2)

                if hparams.zprior_weight is None:
                    if hparams.zprior_sdev != 0:
                        sdev = hparams.zprior_sdev
                        zprior_weight = 1/(2 * (sdev**2))
                    else:
                        zprior_weight = 0
                else:
                    zprior_weight = hparams.zprior_weight
                total_loss_batch = mloss_weight * m_loss_batch + zprior_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                total_loss = total_loss_batch.sum()

                total_loss.backward()
                opt.step()

                if hparams.gradient_noise_weight is None:
                    gradient_noise_weight = np.sqrt(2*opt.param_groups[0]['lr']/(1-hparams.momentum))
                else:
                    gradient_noise_weight = hparams.gradient_noise_weight
                z.data += gradient_noise_weight*torch.randn(z.shape).cuda()

                scheduler.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss {} p_loss {}'
                print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), m_loss.item(), p_loss.item()))

            x_hat_batch = model.postprocess(model.inverse(z))
            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch = torch.norm(z,dim=-1).pow(2)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    + zprior_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
            best_keeper_z.report(z.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator

def realnvp_xlangevin_estimator(hparams):

    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()

    mse = torch.nn.MSELoss(reduction='none')
    annealed = hparams.annealed

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams)
        best_keeper_z = utils.BestKeeper(hparams)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            if hparams.fixed_init:
                z0 = torch.load('initializations/torch_z_init.pt')[:hparams.batch_size]
                z = z0 * hparams.zprior_sdev
            else:
                z =  model.sample_z(n=hparams.batch_size, temp=hparams.zprior_init_sdev)

            x0 = model.postprocess(model.inverse(z))
            x_hat_batch = x0.detach().to(hparams.device)
            x_hat_batch.requires_grad_()
            opt = utils.get_optimizer(x_hat_batch, hparams.learning_rate, hparams)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                p_loss_batch, _, _ = model.log_prob(x_hat_batch)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = hparams.num_measurements / (2 * sigma**2)

                if hparams.ploss_weight is None:
                    ploss_weight = 1.
                else:
                    ploss_weight = hparams.ploss_weight
                total_loss_batch = mloss_weight * m_loss_batch - ploss_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                total_loss = total_loss_batch.sum()

                total_loss.backward()

                opt.step()

                if hparams.gradient_noise_weight is None:
                    gradient_noise_weight = np.sqrt(2*opt.param_groups[0]['lr'])
                else:
                    gradient_noise_weight = hparams.gradient_noise_weight
                x_hat_batch.data += gradient_noise_weight*torch.randn_like(x_hat_batch).to(hparams.device)
                x_hat_batch.data.clamp_(0,1)

                scheduler.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss {} p_loss {}'
                print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), m_loss.item(), p_loss.item()))

            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch, _, _ = model.log_prob(x_hat_batch)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    - ploss_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
            print(x_hat_batch.max(), x_hat_batch.min())
            z , _ = model(model.preprocess(x_hat_batch)[0])
            best_keeper_z.report(z.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator

def stylegan_map_estimator(hparams):

    model = Generator(hparams.image_size, 512, 8)
    model.load_state_dict(torch.load(hparams.checkpoint_path)["g_ema"], strict=False)
    model.eval()
    model = model.to(hparams.device)

    for p in model.parameters():
        p.requires_grad = False

    mse = torch.nn.SmoothL1Loss(reduction='none')
    # mse = torch.nn.MSELoss(reduction='none')
    l1 = torch.nn.L1Loss(reduction='none')
    annealed = hparams.annealed


    lpips = PerceptualLoss(net='vgg')
    batch_size = hparams.batch_size

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        else:
            A = None
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        noises_single = model.make_noise()
        count = 512
        for noise in noises_single:
            count += np.prod(noise.shape)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, count)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            z = hparams.zprior_init_sdev * torch.randn(hparams.batch_size, 512, device=hparams.device)
            z.requires_grad_()
            noises_single = model.make_noise()
            noises = []
            noise_vars = []

            count = 512
            # optimize over a certain number of noise variables
            for idx, noise in enumerate(noises_single):
                noises.append(hparams.zprior_init_sdev * noise.repeat(hparams.batch_size, 1, 1, 1).normal_())
                if idx < hparams.num_noise_variables:
                    noise_vars.append(noises[-1])
                    noises[-1].requires_grad = True
                    count += np.prod(noises[-1].shape)
            print(count)

            opt = utils.get_optimizer([z] + noise_vars, hparams.learning_rate, hparams)
            # check whether cosine annealing of lr helps
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                # stylegan2 adds some noise to the latent. dunno if this
                # is important

                # noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                # latent_n = latent_noise(latent_in, noise_strength.item())

                # the flag input_is_latent determines whether z is passed through the
                # styling network
                # can be done explicitly as well
                x_hat_batch = 0.5 * model([z], input_is_latent=False, noise=noises)[0] + 0.5
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                y_hat_batch_nchw = y_hat_batch.view( hparams.y_shape)
                y_batch_nchw = y.view( hparams.y_shape)
                if hparams.lpips:
                    m_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
                else:
                    m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                # m_loss_batch = l1(y_hat_batch, y).sum(dim=1)
                # p_loss_batch, _, _ = model.log_prob(xhat_batch)
                p_loss_batch = torch.norm(z,dim=-1).pow(2)
                try:
                    l_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
                except:
                    l_loss_batch = None
                for noise in noise_vars:
                    p_loss_batch += torch.norm(noise.view(hparams.batch_size, -1), dim=-1).pow(2)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = 1/(2 * sigma**2 ) #hparams.num_measurements / (2 * sigma**2)

                if hparams.zprior_weight is None:
                    zprior_weight = 1/(2 * (hparams.zprior_sdev **2))
                else:
                    zprior_weight = hparams.zprior_weight
                total_loss_batch = mloss_weight * m_loss_batch + zprior_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                try:
                    l_loss = l_loss_batch.sum()
                except:
                    l_loss = None
                total_loss = total_loss_batch.sum()

                total_loss.backward()
                opt.step()

                if hparams.project:
                    for temp in noise_vars:
                        temp_dim = np.prod(temp.shape[1:])
                        temp.data = hparams.zprior_sdev * np.sqrt(temp_dim) * temp.data / torch.norm(temp.view(batch_size, -1), dim=1)
                    z.data = hparams.zprior_sdev * np.sqrt(512) * z.data / torch.norm(z.view(batch_size,-1), dim=1)
                else:
                    pass



                scheduler.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} l_loss {} m_loss {} p_loss {}'
                try:
                    print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), l_loss.item(), m_loss.item(), p_loss.item()))
                except:
                    print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), None, m_loss.item(), p_loss.item()))

            x_hat_batch = 0.5 * model([z], input_is_latent=False, noise=noises)[0] + 0.5
            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            y_hat_batch_nchw = y_hat_batch.view( hparams.y_shape)
            y_batch_nchw = y.view( hparams.y_shape)
            if hparams.lpips:
                m_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
            else:
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch = torch.norm(z,dim=-1).pow(2)
            for noise in noise_vars:
                p_loss_batch += torch.norm(noise.view(hparams.batch_size, -1), dim=-1).pow(2)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    + zprior_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), total_loss_batch.detach().cpu().numpy())
            z_hat_batch = z.view(hparams.batch_size,-1).detach().cpu().numpy()
            for noise in noises:
                z_hat_batch = np.c_[z_hat_batch, noise.view(hparams.batch_size,-1).detach().cpu().numpy()]

            print(z_hat_batch)
            best_keeper_z.report(z_hat_batch, total_loss_batch.detach().cpu().numpy())
        # return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best
        return best_keeper.get_best(), np.zeros(hparams.batch_size), best_keeper.losses_val_best

    return estimator

def stylegan_langevin_estimator(hparams):

    model = Generator(hparams.image_size, 512, 8)
    model.load_state_dict(torch.load(hparams.checkpoint_path)["g_ema"], strict=False)
    model.eval()
    model = model.to(hparams.device)

    for p in model.parameters():
        p.requires_grad = False

    #mse = torch.nn.SmoothL1Loss(reduction='none')
    mse = torch.nn.MSELoss(reduction='none')
    l1 = torch.nn.L1Loss(reduction='none')
    annealed = hparams.annealed


    lpips = PerceptualLoss(net='vgg')

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        if A_val is not None:
            A = torch.Tensor(A_val).cuda()
        else:
            A = None
        y = torch.Tensor(y_val).cuda()

        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        noises_single = model.make_noise()
        count = 512
        for noise in noises_single:
            count += np.prod(noise.shape)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, count)

        # run T steps of langevin for L different noise levels
        T = hparams.T
        L = hparams.L
        sigma1 = hparams.sigma_init
        sigmaT = hparams.sigma_final
        # geometric factor for tuning sigma and learning rate
        factor = np.power(sigmaT / sigma1, 1/(L-1))

        # if you're running regular langevin, step size is fixed
        # and noise std doesn't change
        if annealed:
            lr_lambda = lambda i: (sigma1 * np.power(factor, (i-1)//T))**2 / (sigmaT **2)
            sigma_lambda = lambda i: sigma1 * np.power(factor, i//T)
        else:
            lr_lambda = lambda i: 1
            sigma_lambda = lambda i: hparams.noise_std

        for i in range(hparams.num_random_restarts):

            z = hparams.zprior_init_sdev * torch.randn(hparams.batch_size, 512, device=hparams.device)
            z.requires_grad_()
            noises_single = model.make_noise()
            noises = []
            noise_vars = []

            count = 512
            # optimize over a certain number of noise variables
            for idx, noise in enumerate(noises_single):
                noises.append(hparams.zprior_init_sdev * noise.repeat(hparams.batch_size, 1, 1, 1).normal_())
                if idx < hparams.num_noise_variables:
                    noise_vars.append(noises[-1])
                    noises[-1].requires_grad = True
                    count += np.prod(noises[-1].shape)
            print(count)

            opt = utils.get_optimizer([z] + noise_vars, hparams.learning_rate, hparams)
            # check whether cosine annealing of lr helps
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

            for j in range(hparams.max_update_iter):
                opt.zero_grad()
                # stylegan2 adds some noise to the latent. dunno if this
                # is important

                # noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                # latent_n = latent_noise(latent_in, noise_strength.item())

                # the flag input_is_latent determines whether z is passed through the
                # styling network
                # can be done explicitly as well
                x_hat_batch = 0.5 * model([z], input_is_latent=False, noise=noises)[0] + 0.5
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = x_hat_batch.detach().cpu().numpy()
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)
                y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
                y_hat_batch_nchw = y_hat_batch.view( hparams.y_shape)
                y_batch_nchw = y.view( hparams.y_shape)
                if hparams.lpips:
                    m_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
                else:
                    m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
                # m_loss_batch = l1(y_hat_batch, y).sum(dim=1)
                # p_loss_batch, _, _ = model.log_prob(xhat_batch)
                p_loss_batch = torch.norm(z,dim=-1).pow(2)
                #try:
                #    l_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
                #except:
                #    l_loss_batch = None
                for noise in noise_vars:
                    p_loss_batch += torch.norm(noise.view(hparams.batch_size, -1), dim=-1).pow(2)
                p_loss_batch = p_loss_batch.view(-1)
                if (hparams.mloss_weight is not None) and (not annealed):
                    mloss_weight = hparams.mloss_weight
                else:
                    sigma = sigma_lambda(j)
                    mloss_weight = hparams.num_measurements / (2 * sigma**2)

                if (hparams.zprior_weight is not None) :
                    zprior_weight = hparams.zprior_weight
                else:
                    zprior_weight = 0.5 / (hparams.zprior_sdev **2)
                total_loss_batch = mloss_weight * m_loss_batch + zprior_weight * p_loss_batch

                m_loss = m_loss_batch.sum()
                p_loss = p_loss_batch.sum()
                #try:
                #    l_loss = l_loss_batch.sum()
                #except:
                #    l_loss = None
                total_loss = total_loss_batch.sum()

                total_loss.backward()
                opt.step()

                if hparams.gradient_noise_weight is None:
                    gradient_noise_weight = np.sqrt(2*opt.param_groups[0]['lr'] /(1- hparams.momentum))
                else:
                    gradient_noise_weight = hparams.gradient_noise_weight
                z.data += gradient_noise_weight*torch.randn_like(z).to(hparams.device)
                for noise in noise_vars:
                    noise.data += gradient_noise_weight * torch.randn_like(noise).to(hparams.device)

                scheduler.step()

                logging_format = 'rr {} iter {} lr {} total_loss {} l_loss {} m_loss {} p_loss {}'
                #try:
                #    print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), l_loss.item(), m_loss.item(), p_loss.item()))
                #except:
                print(logging_format.format(i, j, opt.param_groups[0]['lr'], total_loss.item(), None, m_loss.item(), p_loss.item()))

            x_hat_batch = 0.5 * model([z], input_is_latent=False, noise=noises)[0] + 0.5
            y_hat_batch = get_measurements_torch(x_hat_batch, A, hparams.measurement_type, hparams)
            y_hat_batch_nchw = y_hat_batch.view( hparams.y_shape)
            y_batch_nchw = y.view( hparams.y_shape)
            if hparams.lpips:
                m_loss_batch = lpips(y_hat_batch_nchw, y_batch_nchw)
            else:
                m_loss_batch = mse(y_hat_batch, y).sum(dim=1)
            p_loss_batch = torch.norm(z,dim=-1).pow(2)
            for noise in noise_vars:
                p_loss_batch += torch.norm(noise.view(hparams.batch_size, -1), dim=-1).pow(2)
            p_loss_batch = p_loss_batch.view(-1)
            total_loss_batch = mloss_weight * m_loss_batch \
                    + zprior_weight * p_loss_batch
            best_keeper.report(x_hat_batch.view(hparams.batch_size,-1).detach().cpu().numpy(), m_loss_batch.detach().cpu().numpy())
            z_hat_batch = z.view(hparams.batch_size,-1).detach().cpu().numpy()
            for noise in noises:
                z_hat_batch = np.c_[z_hat_batch, noise.view(hparams.batch_size,-1).detach().cpu().numpy()]

            print(z_hat_batch)
            best_keeper_z.report(z_hat_batch, m_loss_batch.detach().cpu().numpy())
            if m_loss_batch.mean()<= hparams.error_threshold:
                break
        # return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best
        return best_keeper.get_best(), np.zeros(hparams.batch_size), best_keeper.losses_val_best

    return estimator

def stylegan_pulse_estimator(hparams):

    model = PULSE(image_size=hparams.image_size, checkpoint_path=hparams.checkpoint_path, dataset=hparams.dataset)
    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""
        kwargs = vars(hparams)

        y_val_tensor = torch.Tensor(y_val.reshape(hparams.y_shape)).cuda()

        for (HR, LR) in model(y_val_tensor, **kwargs):

            return HR.detach().cpu().numpy().reshape(hparams.batch_size,-1), np.zeros(hparams.batch_size), np.zeros(hparams.batch_size)
        #return best_keeper.get_best(), np.zeros(hparams.batch_size), best_keeper.losses_val_best

    return estimator


# def glow_map_estimator(hparams):

#     # set up model and session
#     dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev, hparams.fixed_init)

#     x_hat_batch_nhwc = dec_x + 0.5


#     # Set up palceholders
#     A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
#     y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

#     # convert from NHWC to NCHW
#     # since I used pytorch for reading data, the measurements
#     # are from a ground truth of data format NCHW
#     # Meanwhile GLOW's output has format NHWC
#     x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])

#     # measure the estimate
#     if hparams.measurement_type == 'project':
#         y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
#     elif hparams.measurement_type == 'circulant':
#         sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
#         y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
#     elif hparams.measurement_type == 'superres':
#         y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))

#     # define all losses
#     z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
#     z_stack = tf.concat(z_list, axis=1)
#     z_loss_batch = tf.reduce_sum(z_stack**2, 1)
#     z_loss = tf.reduce_sum(z_loss_batch)
#     y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
#     y_loss = tf.reduce_sum(y_loss_batch)
#     if hparams.mloss_weight is None:
#         mloss_weight = 1.
#     else:
#         mloss_weight = hparams.mloss_weight
#     if hparams.zprior_weight is None:
#         zprior_weight = (hparams.noise_std ** 2)/(hparams.num_measurements * 0.49)
#     else:
#         zprior_weight = hparams.zprior_weight
#     total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
#     total_loss = tf.reduce_sum(total_loss_batch)

#     # Set up gradient descent
#     global_step = tf.Variable(0, trainable=False, name='global_step')
#     learning_rate = utils.get_learning_rate(global_step, hparams)
#     with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#         opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
#         update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')
#     # opt_reinit_op = utils.get_opt_reinit_op(opt, dec_eps, global_step)

#         sess = utils.tensorflow_session()

#         # initialize variables
#         uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
#         init_op = tf.variables_initializer(
#             [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])
#     # sess.run(init_op)

#     def estimator(A_val, y_val, hparams):
#         """Function that returns the estimated image"""
#         best_keeper = utils.BestKeeper(hparams)
#         best_keeper_z = utils.BestKeeper(hparams)
#         feed_dict = hparams.feed_dict.copy()

#         if hparams.measurement_type == 'circulant':
#             feed_dict.update({A: A_val, y_batch: y_val})
#         else:
#             feed_dict.update({y_batch: y_val})

#         for i in range(hparams.num_random_restarts):
#             sess.run(init_op)
#             # sess.run(opt_reinit_op)
#             for j in range(hparams.max_update_iter):
#                 if hparams.gif and ((j % hparams.gif_iter) == 0):
#                     images = sess.run(x_hat_batch_nchw, feed_dict=feed_dict)
#                     for im_num, image in enumerate(images):
#                         save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
#                         utils.set_up_dir(save_dir)
#                         save_path = save_dir + '{0}.png'.format(j)
#                         image = image.reshape(hparams.image_shape)
#                         save_image(image, save_path)

#                 _, lr_value, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, learning_rate, total_loss, z_loss, y_loss], feed_dict)
#                 logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
#                 print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))

#             x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)

#             x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
#             best_keeper.report(x_hat_batch_value, total_loss_batch_value)
#             best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
#         return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best


#     return estimator

# def glow_langevin_estimator(hparams):

#     # set up model and session
#     dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev, hparams.fixed_init)

#     x_hat_batch_nhwc = dec_x + 0.5


#     # Set up palceholders
#     A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
#     y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')


#     # convert from NHWC to NCHW
#     # since I used pytorch for reading data, the measurements
#     # are from a ground truth of data format NCHW
#     # Meanwhile GLOW's output has format NHWC
#     x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])

#     # measure the estimate
#     if hparams.measurement_type == 'project':
#         y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
#     elif hparams.measurement_type == 'circulant':
#         sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
#         y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
#     elif hparams.measurement_type == 'superres':
#         y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))

#     # create noise placeholders for langevin
#     noise_vars = [tf.placeholder(tf.float32, shape=dec_eps[i].get_shape()) for i in range(len(dec_eps))]

#     # define all losses
#     z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
#     z_stack = tf.concat(z_list, axis=1)
#     z_loss_batch = tf.reduce_sum(z_stack**2, 1)
#     z_loss = tf.reduce_sum(z_loss_batch)
#     y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
#     y_loss = tf.reduce_sum(y_loss_batch)

#     # mloss_weight should be m/2sigma^2 for proper langevin
#     # zprior_weight should be 1/(2*0.49) for proper langevin
#     if hparams.mloss_weight is None:
#         mloss_weight = 0.5 * hparams.num_measurements / (hparams.noise_std ** 2)
#     else:
#         mloss_weight = hparams.mloss_weight
#     if hparams.zprior_weight is None:
#         zprior_weight = 1/(2 * 0.49)
#     else:
#         zprior_weight = hparams.zprior_weight
#     total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
#     total_loss = tf.reduce_sum(total_loss_batch)

#     # Set up gradient descent
#     global_step = tf.Variable(0, trainable=False, name='global_step')
#     learning_rate = utils.get_learning_rate(global_step, hparams)
#     with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#         opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
#         update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')
#     # opt_reinit_op = utils.get_opt_reinit_op(opt, dec_eps, global_step)
#         noise_ops = [dec_eps[i].assign_add(noise_vars[i]) for i in range(len(dec_eps))]

#         sess = utils.tensorflow_session()

#         # initialize variables
#         uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
#         init_op = tf.variables_initializer(
#             [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])
#     # sess.run(init_op)

#     def estimator(A_val, y_val, hparams):
#         """Function that returns the estimated image"""
#         best_keeper = utils.BestKeeper(hparams)
#         best_keeper_z = utils.BestKeeper(hparams)
#         feed_dict = hparams.feed_dict.copy()

#         if hparams.measurement_type == 'circulant':
#             feed_dict.update({A: A_val, y_batch: y_val})
#         else:
#             feed_dict.update({y_batch: y_val})

#         for i in range(hparams.num_random_restarts):
#             sess.run(init_op)
#             # sess.run(opt_reinit_op)
#             for j in range(hparams.max_update_iter):
#                 if hparams.gif and ((j % hparams.gif_iter) == 0):
#                     images = sess.run(x_hat_batch_nchw, feed_dict=feed_dict)
#                     for im_num, image in enumerate(images):
#                         save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
#                         utils.set_up_dir(save_dir)
#                         save_path = save_dir + '{0}.png'.format(j)
#                         image = image.reshape(hparams.image_shape)
#                         save_image(image, save_path)

#                 # feed_dict.update({beta: 1})# 1 - np.exp(-(j+1)/500.)})
#                 _, lr_value, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, learning_rate, total_loss, z_loss, y_loss], feed_dict)
#                 logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
#                 print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))

#                 if hparams.gradient_noise_weight is None:
#                     gradient_noise_weight = np.sqrt(2*lr_value)
#                 else:
#                     gradient_noise_weight = hparams.gradient_noise_weight
#                 for noise_var in noise_vars:
#                     noise_shape = noise_var.get_shape().as_list()
#                     # gradient_noise_weight should be sqrt(2*lr) for proper langevin
#                     feed_dict.update({noise_var: gradient_noise_weight *np.random.randn(hparams.batch_size, noise_shape[1], noise_shape[2], noise_shape[3])})
#                 results = run(sess,noise_ops,feed_dict)


#             x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)

#             x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
#             best_keeper.report(x_hat_batch_value, total_loss_batch_value)
#             best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
#         return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

#     return estimator

def glow_annealed_map_estimator(hparams):

    annealed = hparams.annealed
    # set up model and session
    dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev, hparams.fixed_init)

    x_hat_batch_nhwc = dec_x + 0.5


    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # convert from NHWC to NCHW
    # since I used pytorch for reading data, the measurements
    # are from a ground truth of data format NCHW
    # Meanwhile GLOW's output has format NHWC
    x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
    elif hparams.measurement_type == 'circulant':
        sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
        y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
    elif hparams.measurement_type == 'superres':
        y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))

    # define all losses
    z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
    z_stack = tf.concat(z_list, axis=1)
    z_loss_batch = tf.reduce_sum(z_stack**2, 1)
    z_loss = tf.reduce_sum(z_loss_batch)
    y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_sum(y_loss_batch)

    # mloss_weight should be m/2sigma^2 for proper langevin
    # zprior_weight should be 1/(2*0.49) for proper langevin
    sigma = tf.placeholder(tf.float32, shape=[])
    if (hparams.mloss_weight is not None) and (not annealed):
        mloss_weight = hparams.mloss_weight
    else:
        mloss_weight = 0.5 * hparams.num_measurements / (sigma ** 2)
    if hparams.zprior_weight is None:
        if hparams.zprior_sdev != 0:
            zprior_weight = 1/(2 * hparams.zprior_sdev**2)
        else:
            zprior_weight = 0
    else:
        zprior_weight = hparams.zprior_weight
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')

        sess = utils.tensorflow_session()

        # initialize variables
        uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])
    # sess.run(init_op)

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        best_keeper_z = utils.BestKeeper(hparams)
        feed_dict = hparams.feed_dict.copy()

        if hparams.measurement_type == 'circulant':
            feed_dict.update({A: A_val, y_batch: y_val})
        else:
            feed_dict.update({y_batch: y_val})

        for i in range(hparams.num_random_restarts):
            sess.run(init_op)
            # sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch_nchw, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

                factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1))

                if annealed:
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2)
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T)
                    lr_value = hparams.learning_rate * lr_lambda(j)
                else:
                    sigma_value = hparams.sigma_final
                    lr_value = hparams.learning_rate

                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})
                _, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, total_loss, z_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))


            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)

            x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
            best_keeper.report(x_hat_batch_value, total_loss_batch_value)
            best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best


    return estimator


def glow_annealed_langevin_estimator(hparams):

    annealed = hparams.annealed
    # set up model and session
    dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev, hparams.fixed_init)

    x_hat_batch_nhwc = dec_x + 0.5


    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')


    # convert from NHWC to NCHW
    # since I used pytorch for reading data, the measurements
    # are from a ground truth of data format NCHW
    # Meanwhile GLOW's output has format NHWC
    x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
    elif hparams.measurement_type == 'circulant':
        sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
        y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
    elif hparams.measurement_type == 'superres':
        y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))

    # create noise placeholders for langevin
    noise_vars = [tf.placeholder(tf.float32, shape=dec_eps[i].get_shape()) for i in range(len(dec_eps))]

    # define all losses
    z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
    z_stack = tf.concat(z_list, axis=1)
    z_loss_batch = tf.reduce_sum(z_stack**2, 1)
    z_loss = tf.reduce_sum(z_loss_batch)
    # y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
    # y_loss = tf.reduce_sum(y_loss_batch)
    y_hat_batch_nchw = tf.reshape(y_hat_batch, hparams.y_shape)
    y_hat_batch_nhwc = tf.transpose(y_hat_batch_nchw, perm=[0,2,3,1])
    y_batch_nchw = tf.reshape(y_batch, hparams.y_shape)
    y_batch_nhwc = tf.transpose(y_batch_nchw, perm=[0,2,3,1])
    y_loss_batch = lpips_tf.lpips(y_hat_batch_nhwc, y_batch_nhwc, model='net-lin', net='vgg')
    y_loss = tf.reduce_sum(y_loss_batch)

    # mloss_weight should be m/2sigma^2 for proper langevin
    # zprior_weight should be 1/(2*0.49) for proper langevin
    sigma = tf.placeholder(tf.float32, shape=[])
    # mloss_weight = 0.5 * hparams.num_measurements / (sigma ** 2)
    mloss_weight = 1 / (sigma ** 2)
    if hparams.zprior_weight is None:
        zprior_weight = 1/(2 * 0.49)
    else:
        zprior_weight = hparams.zprior_weight
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')
    # opt_reinit_op = utils.get_opt_reinit_op(opt, dec_eps, global_step)
        noise_ops = [dec_eps[i].assign_add(noise_vars[i]) for i in range(len(dec_eps))]

        sess = utils.tensorflow_session()

        # initialize variables
        uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])
    # sess.run(init_op)

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        best_keeper_z = utils.BestKeeper(hparams)
        feed_dict = hparams.feed_dict.copy()

        if hparams.measurement_type == 'circulant':
            feed_dict.update({A: A_val, y_batch: y_val})
        else:
            feed_dict.update({y_batch: y_val})

        for i in range(hparams.num_random_restarts):
            sess.run(init_op)
            # sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch_nchw, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)


                factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1))

                if annealed:
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2)
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T)
                else:
                    lr_lambda = lambda t: 1
                    sigma_value = hparams.sigma_final
                lr_value = hparams.learning_rate * lr_lambda(j)

                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})
                _, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, total_loss, z_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))

                if hparams.gradient_noise_weight is None:
                    gradient_noise_weight = np.sqrt(2*lr_value/(1-hparams.momentum))
                else:
                    gradient_noise_weight = hparams.gradient_noise_weight
                for noise_var in noise_vars:
                    noise_shape = noise_var.get_shape().as_list()
                    # gradient_noise_weight should be sqrt(2*lr) for proper langevin
                    feed_dict.update({noise_var: gradient_noise_weight *np.random.randn(hparams.batch_size, noise_shape[1], noise_shape[2], noise_shape[3])})
                results = run(sess,noise_ops,feed_dict)


            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)

            x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
            best_keeper.report(x_hat_batch_value, total_loss_batch_value)
            best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator


def ncsnv2_langevin_estimator(hparams, MAP=False):
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def gradient_log_conditional_likelihood(A, y_batch, y_hat_batch):
        err = y_batch - y_hat_batch
        if hparams.measurement_type == 'superres':
            err = err.view(hparams.y_shape)
            ans = F.interpolate(err, scale_factor = hparams.downsample)
        elif hparams.measurement_type == 'circulant':
            err_padded = torch.zeros(hparams.batch_size, hparams.n_input).to(hparams.device)
            err_padded[:,hparams.train_indices] = err
            A_shift = torch.zeros_like(A)
            A_shift[0,0] = A[0,0]
            A_shift[0,1:] = A[0,1:].flip(dims=[0])

            err_A = utils.partial_circulant_torch(err_padded, A_shift, range(hparams.n_input), sign_pattern=ones_torch)

            ans = err_A * sign_pattern_torch
            ans = ans.view((-1,) + hparams.image_shape)
        else:
            return NotImplementedError

        return ans


    batch_size = hparams.batch_size
    if hparams.measurement_type == 'circulant':
        if hparams.sign_pattern is not None:
            sign_pattern_torch = torch.Tensor(hparams.sign_pattern).to(hparams.device)
            ones_torch = torch.ones(1,hparams.n_input).to(hparams.device)
    else:
        pass
    new_config = dict2namespace(hparams.ncsnv2_configs)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.sampling.batch_size = batch_size

    if 'ffhq' in hparams.dataset :
        score = NCSNv2Deepest(new_config).to(new_config.device)
    elif hparams.dataset == 'celebA':
        score = NCSNv2(new_config).to(new_config.device)


    sigmas_torch = get_sigmas(new_config)
    sigmas = sigmas_torch.cpu().numpy()

    states = torch.load(hparams.checkpoint_path,
                        map_location=new_config.device)

    score = torch.nn.DataParallel(score)

    score.load_state_dict(states[0], strict=True)

    for p in score.parameters():
        p.requires_grad = False

    if new_config.model.ema:
        ema_helper = ema.EMAHelper(mu=new_config.model.ema_rate)
        ema_helper.register(score)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(score)


    score.eval()

    mse = torch.nn.MSELoss(reduction='none')

    def estimator(A_val, y_val, hparams):
        x_hat_nchw_batch = torch.rand((hparams.batch_size,) + hparams.image_shape,
                device=new_config.device)
        n_steps_each = new_config.sampling.n_steps_each
        step_lr = new_config.sampling.step_lr

        y_batch = torch.Tensor(y_val).to(new_config.device)
        if A_val is not None:
            A = torch.Tensor(A_val).to(new_config.device)
        else:
            A = None
        with torch.no_grad():
            start = time.time()
            for c, sigma in enumerate(sigmas[:int(hparams.L)]):
                labels = torch.ones(x_hat_nchw_batch.shape[0], device=x_hat_nchw_batch.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    j = c*n_steps_each + s
                    if hparams.gif and (( j % hparams.gif_iter) == 0):
                        images = x_hat_nchw_batch.detach().cpu().numpy()
                        for im_num, image in enumerate(images):
                            save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                            utils.set_up_dir(save_dir)
                            save_path = save_dir + '{0}.png'.format(j)
                            image = image.reshape(hparams.image_shape)
                            save_image(image, save_path)
                    noise = torch.randn_like(x_hat_nchw_batch) * np.sqrt(step_size * 2)
                    grad = score(x_hat_nchw_batch, labels)
                    #error = (up(torch_downsample(x_hat_nchw_batch,factor) - y) / sigma**2)

                    y_hat_batch = get_measurements_torch(x_hat_nchw_batch.view(hparams.batch_size, -1), A, hparams.measurement_type, hparams)
                    m_loss_grad_nchw = gradient_log_conditional_likelihood(A, y_batch, y_hat_batch)/(sigma**2 + hparams.noise_std**2/hparams.num_measurements)
                    if hparams.mloss_weight is None:
                        mloss_weight = 1.0
                    else:
                        mloss_weight = hparams.mloss_weight
                    if MAP:
                        x_hat_nchw_batch = x_hat_nchw_batch + step_size * (grad + mloss_weight * m_loss_grad_nchw)
                    else:
                        x_hat_nchw_batch = x_hat_nchw_batch + step_size * (grad + mloss_weight * m_loss_grad_nchw) + noise

                    m_loss_batch = mse(y_hat_batch, y_batch).sum(dim=1)

                    print("class: {}, step_size: {}, mean {}, max {}, y_mse {}".format(c, step_size, grad.abs().mean(),
                                                                             grad.abs().max(), m_loss_batch.mean()))
            end = time.time()
            print(f'Time on batch:{(end - start)/60:.3f} minutes')
        return x_hat_nchw_batch.view(hparams.batch_size,-1).cpu().numpy(), np.zeros(hparams.batch_size), m_loss_batch.cpu().numpy()

    return estimator

def deep_decoder_estimator(hparams):
    num_channels = [700]*7
    output_depth = 3 # hparams.image_size

    def estimator(A_val, y_val, hparams):

        y_batch = torch.Tensor(y_val).to(hparams.device)
        if A_val is not None:
            A = torch.Tensor(A_val).to(hparams.device)
        else:
            A = None

        def apply_f(x):
            return get_measurements_torch(x.view(hparams.batch_size,-1),A,hparams.measurement_type,hparams)

        net = decoder.decodernw(output_depth, num_channels_up=num_channels, upsample_first=True).cuda()

        rn = 0.005
        rnd = 500
        numit = 4000

        print(hparams.max_update_iter)
        mse_n, mse_t, ni, net = fit(
                       num_channels=num_channels,
                        reg_noise_std=rn,
                        reg_noise_decayevery = rnd,
                        num_iter=hparams.max_update_iter,
                        LR=hparams.learning_rate,
                        OPTIMIZER=hparams.optimizer_type,
                        img_noisy_var=y_batch,
                        net=net,
                        img_clean_var=torch.zeros_like(y_batch),
                        find_best=True,
                        apply_f=apply_f,
                        )
        return net(ni.cuda()).view(hparams.batch_size,-1).detach().cpu().numpy(), np.zeros(hparams.batch_size), np.zeros(hparams.batch_size)

    return estimator

