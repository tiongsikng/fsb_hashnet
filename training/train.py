import numpy as np
import time
import sys
import itertools 
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import random
from torch.distributions import Beta
import math
from PIL import Image
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# **********    

class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x
        self.fn_no_mean = lambda x, i: x

    def __call__(self, loss, peri_loss, face_loss, d_loss, g_loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:5.4f} | '.format(self.fn(loss, i))
        peri_loss = 'peri_loss: {:5.4f} | '.format(self.fn_no_mean(peri_loss, i))
        face_loss = 'face_loss: {:5.4f} | '.format(self.fn_no_mean(face_loss, i))
        d_loss = 'd_loss: {:5.4f} | '.format(self.fn_no_mean(d_loss, i))
        g_loss = 'g_loss: {:5.4f} | '.format(self.fn_no_mean(g_loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + peri_loss + face_loss + d_loss + g_loss + metric_str + ' ', end='')
        if i + 1 == self.length:
            print('')

# **********

class BatchTimer(object):
    
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)

# **********

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()

# **********

def run_train(feature_extractor, generator, discriminator, feat_fc, hash_fc, 
                face_loader, peri_loader, face_loader_tl, peri_loader_tl, 
                epoch = 1, net_params = None, loss_fn = None, optimizer_G = None, optimizer_D = None,
                scheduler_G = None, scheduler_D = None, batch_metrics = {'time': BatchTimer()}, 
                show_running = True, device = 'cuda:0', writer = None):
    
    mode = 'Train'
    iter_max = len(face_loader)
    logger = Logger(mode, length = iter_max, calculate_mean = show_running)
    
    loss = 0
    metrics = {}
    
    # **********
    
    face_iterator_tl = iter(face_loader_tl)
    peri_iterator_tl = iter(peri_loader_tl)
    wting = math.exp(-5.0 * (1 - (epoch / net_params['epochs'])**2) )

    for batch_idx, ( face_in, peri_in ) in enumerate( zip( face_loader, peri_loader ) ):
        #### *** source : face ***
        #### random sampling
        face_in = face_in

        face_x, face_y = face_in

        face_x = face_x.to(device)
        face_y = face_y.to(device)      

        #### balanced sampling
        try:
            face_in_tl = next(face_iterator_tl)
        except StopIteration:
            face_iterator_tl = iter(face_loader_tl)
            face_in_tl = next(face_iterator_tl)
        
        face_x_tl, face_y_tl = face_in_tl

        face_x_tl = face_x_tl.to(device)
        face_y_tl = face_y_tl.to(device)

        del face_in
        del face_in_tl
        
        # *** ***

        face_x_r = face_x
        face_x_tl = face_x_tl
        face_lbl_r = face_y
        face_lbl_tl = face_y_tl
        
        face_emb = feature_extractor(torch.cat((face_x_r, face_x_tl)))
        face_emb_r = face_emb[:int(face_emb.shape[0]/2)]
        face_emb_tl = face_emb[int(face_emb.shape[0]/2):]        
        face_lbl = torch.cat((face_lbl_r, face_lbl_tl))

        face_emb_gen = generator(face_emb, face_lbl, training=True)
        face_emb_gen_r = face_emb_gen[:int(face_emb_gen.shape[0]/2)]
        face_emb_gen_tl = face_emb_gen[int(face_emb_gen.shape[0]/2):]

        # ***

        face_pred = feat_fc(face_emb, face_lbl)
        face_loss_ce = wting*loss_fn['loss_ce'](face_pred, face_lbl)

        face_pred_gen = hash_fc(face_emb_gen, face_lbl)
        face_loss_ce_gen = wting*loss_fn['loss_ce'](face_pred_gen, face_lbl)

        #### *** target : periocular ***
        #### random sampling
        peri_in = peri_in

        peri_x, peri_y = peri_in

        peri_x = peri_x.to(device)
        peri_y = peri_y.to(device)     

        #### balanced sampling
        try:
            peri_in_tl = next(peri_iterator_tl)
        except StopIteration:
            peri_iterator_tl = iter(peri_loader_tl)
            peri_in_tl = next(peri_iterator_tl)
        
        peri_x_tl, peri_y_tl = peri_in_tl

        peri_x_tl = peri_x_tl.to(device)
        peri_y_tl = peri_y_tl.to(device)

        del peri_in
        del peri_in_tl
        
        # *** ***

        peri_x_r = peri_x
        peri_x_tl = peri_x_tl
        peri_lbl_r = peri_y
        peri_lbl_tl = peri_y_tl
        
        peri_emb = feature_extractor(torch.cat((peri_x_r, peri_x_tl)))
        peri_emb_r = peri_emb[:int(peri_emb.shape[0]/2)]
        peri_emb_tl = peri_emb[int(peri_emb.shape[0]/2):]
        peri_lbl = torch.cat((peri_lbl_r, peri_lbl_tl))

        peri_emb_gen = generator(peri_emb, peri_lbl, training=True)
        peri_emb_gen_r = peri_emb_gen[:int(peri_emb_gen.shape[0]/2)]
        peri_emb_gen_tl = peri_emb_gen[int(peri_emb_gen.shape[0]/2):]

        # ***

        peri_pred = feat_fc(peri_emb, peri_lbl)
        peri_loss_ce = wting*loss_fn['loss_ce'](peri_pred, peri_lbl)

        peri_pred_gen = hash_fc(peri_emb_gen, peri_lbl)
        peri_loss_ce_gen = wting*loss_fn['loss_ce'](peri_pred_gen, peri_lbl)
        
        #
        # *** ***
        #          

        # *** *** 
        real = torch.FloatTensor((face_emb).size(0), 1).fill_(1.0).to(device) # 진짜(real): 1 # source
        fake = torch.FloatTensor((peri_emb).size(0), 1).fill_(0.0).to(device) # 가짜(fake): 0 # target

        g_loss = (peri_loss_ce + face_loss_ce) + peri_loss_ce_gen + face_loss_ce_gen \
                + loss_fn['loss_bce'](discriminator(peri_emb_gen), real)

        # *** ***
        real_loss = loss_fn['loss_bce'](discriminator(face_emb_gen), real)
        fake_loss = loss_fn['loss_bce'](discriminator(peri_emb_gen), fake)
        d_loss = ((fake_loss + real_loss) / 2)
        optimizer_G.zero_grad()    
        optimizer_D.zero_grad()
        loss_batch = g_loss + d_loss
        loss_batch.backward()
        optimizer_G.step()
        optimizer_D.step()

        del face_emb, face_emb_tl, face_emb_r, face_emb_gen
        del peri_emb, peri_emb_tl, peri_emb_r, peri_emb_gen


        # *** ***
        
        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(peri_pred, peri_lbl).detach().cpu()            
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            
        if writer is not None:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, (peri_loss_ce + peri_loss_ce_gen), (face_loss_ce + face_loss_ce_gen), d_loss, g_loss, metrics, batch_idx)
        else:
            logger(loss_batch, metrics_batch, batch_idx)
    
    # *** ***

    if scheduler_G is not None:
        scheduler_G.step()

    if scheduler_D is not None:
        scheduler_D.step()

    loss = loss / (batch_idx + 1)
    metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
    
    return metrics, loss