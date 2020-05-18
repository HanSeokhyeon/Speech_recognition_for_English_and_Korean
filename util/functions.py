import torch
import torch.nn as nn
from torch.autograd import Variable  
import numpy as np
import editdistance as ed
import time


# letter_error_rate function
# Merge the repeated prediction and calculate editdistance of prediction and ground truth
def letter_error_rate(pred_y,true_y,data):
    ed_accumalate = []
    for p,t in zip(pred_y,true_y):
        compressed_t = [w for w in t if (w!=1 and w!=0)]
        
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        if data == 'timit':
            compressed_t = collapse_phn(compressed_t)
            compressed_p = collapse_phn(compressed_p)
        ed_accumalate.append(ed.eval(compressed_p,compressed_t)/len(compressed_t))
    return ed_accumalate


def label_smoothing_loss(pred_y,true_y,label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y,dim=-1),dim=-1,keepdim=True)
    
    # calculate smoothen label, last term ensures padding vector remains all zero
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0-label_smoothing)*true_y+(label_smoothing/class_dim))*torch.sum(true_y,dim=-1,keepdim=True)

    loss = - torch.mean(torch.sum((torch.sum(smooth_y * pred_y,dim=-1)/seq_len),dim=-1))

    return loss


def train(train_set, model, optimizer, tf_rate, conf, global_step, log_writer, data='timit'):
    bucketing = conf['model_parameter']['bucketing']
    use_gpu = conf['model_parameter']['use_gpu']
    label_smoothing = conf['model_parameter']['label_smoothing']

    verbose_step = conf['training_parameter']['verbose_step']

    model.train()

    # Training
    for batch_index, (batch_data, batch_label) in enumerate(train_set):
        if bucketing:
            batch_data = batch_data.squeeze(dim=0)
            batch_label = batch_data.squeeze(dim=0)
        max_label_len = min([batch_label.size()[1], conf['model_parameter']['max_label_len']])

        batch_data = Variable(batch_data).type(torch.FloatTensor)
        batch_label = Variable(batch_label, requires_grad=False)
        criterion = nn.NLLLoss(ignore_index=0)
        if use_gpu:
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            criterion = criterion.cuda()

        optimizer.zero_grad()

        raw_pred_seq = model(batch_data, batch_label, tf_rate, batch_label)

        pred_y = (torch.cat([torch.unsqueeze(each_y, 1) for each_y in raw_pred_seq], 1)[:, :max_label_len, :]).contiguous()

        if label_smoothing == 0.0:
            pred_y = pred_y.permute(0, 2, 1)  # pred_y.contiguous().view(-1,output_class_dim)
            true_y = torch.max(batch_label, dim=2)[1][:, :max_label_len].contiguous()  # .view(-1)

            loss = criterion(pred_y, true_y)
            # variable -> numpy before sending into LER calculator
            batch_ler = letter_error_rate(torch.max(pred_y.permute(0, 2, 1), dim=2)[1].cpu().numpy(),
                                        # .reshape(current_batch_size,max_label_len),
                                        true_y.cpu().data.numpy(),
                                        data)  # .reshape(current_batch_size,max_label_len), data)

        else:
            true_y = batch_label[:, :max_label_len, :].contiguous()
            true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
            loss = label_smoothing_loss(pred_y, true_y, label_smoothing=label_smoothing)
            batch_ler = letter_error_rate(torch.max(pred_y, dim=2)[1].cpu().numpy(),
                                        # .reshape(current_batch_size,max_label_len),
                                        torch.max(true_y, dim=2)[1].cpu().data.numpy(),
                                        data)  # .reshape(current_batch_size,max_label_len), data)

        loss.backward()
        optimizer.step()

        batch_loss = loss.cpu().data.numpy()

        global_step += 1

        if global_step % verbose_step == 0:
            log_writer.add_scalars('loss', {'train': batch_loss}, global_step)
            log_writer.add_scalars('cer', {'train': np.array([np.array(batch_ler).mean()])}, global_step)

    return global_step


def evaluate(evaluate_set, model, tf_rate, conf, global_step, log_writer, epoch_begin, train_begin, logger, epoch, is_valid, data='timit', test=False):
    bucketing = conf['model_parameter']['bucketing']
    use_gpu = conf['model_parameter']['use_gpu']

    model.eval()

    # Validation
    eval_loss = []
    eval_ler = []

    with torch.no_grad():
        for _, (batch_data, batch_label) in enumerate(evaluate_set):
            if bucketing:
                batch_data = batch_data.squeeze(dim=0)
                batch_label = batch_data.squeeze(dim=0)
            max_label_len = min([batch_label.size()[1], conf['model_parameter']['max_label_len']])

            batch_data = Variable(batch_data).type(torch.FloatTensor)
            batch_label = Variable(batch_label, requires_grad=False)
            criterion = nn.NLLLoss(ignore_index=0)
            if use_gpu:
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                criterion = criterion.cuda()

            raw_pred_seq = model(batch_data, batch_label, 0, None)

            pred_y = (torch.cat([torch.unsqueeze(each_y, 1) for each_y in raw_pred_seq], 1)[:, :max_label_len, :]).contiguous()

            pred_y = pred_y.permute(0, 2, 1)  # pred_y.contiguous().view(-1,output_class_dim)
            true_y = torch.max(batch_label, dim=2)[1][:, :max_label_len].contiguous()  # .view(-1)

            loss = criterion(pred_y, true_y)
            # variable -> numpy before sending into LER calculator
            batch_ler = letter_error_rate(torch.max(pred_y.permute(0, 2, 1), dim=2)[1].cpu().numpy(),
                                        # .reshape(current_batch_size,max_label_len),
                                        true_y.cpu().data.numpy(),
                                        data)  # .reshape(current_batch_size,max_label_len), data)

            batch_loss = loss.cpu().data.numpy()

            eval_loss.append(batch_loss)
            eval_ler.extend(batch_ler)

    now_loss, now_cer = np.array([sum(eval_loss) / len(eval_loss)]), np.mean(eval_ler)
    if is_valid:
        log_writer.add_scalars('loss', {'dev': now_loss}, global_step)
        log_writer.add_scalars('cer', {'dev': now_cer}, global_step)
    else:
        log_writer.add_scalars('loss', {'test': now_loss}, global_step)
        log_writer.add_scalars('cer', {'test': now_cer}, global_step)

    current = time.time()
    epoch_elapsed = (current - epoch_begin) / 60.0
    train_elapsed = (current - train_begin) / 3600.0

    if not test:
        logger.info("epoch: {}, global step: {:6d}, loss: {:.4f}, cer: {:.4f}, elapsed: {:.2f}m {:.2f}h"
                    .format(epoch, global_step, float(now_loss), float(now_cer), epoch_elapsed, train_elapsed))
    else:
        logger.info("test epoch: {}, cer: {:.6f}".format(epoch, float(now_cer)))

    return now_cer


def log_parser(log_file_path):
    tr_loss,tt_loss,tr_ler,tt_ler = [], [], [], []
    with open(log_file_path,'r') as log_f:
        for line in log_f:
            tmp = line.split('_')
            tr_loss.append(float(tmp[3]))
            tr_ler.append(float(tmp[5]))
            tt_loss.append(float(tmp[7]))
            tt_ler.append(float(tmp[9]))

    return tr_loss,tt_loss,tr_ler,tt_ler


def collapse_phn(seq, return_phn = False, drop_q = True):
    # Collapse 61 phns to 39 phns
    # http://cdn.intechopen.com/pdfs/15948/InTech-Phoneme_recognition_on_the_timit_database.pdf
    phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl",
                "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
                "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng",
                "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih",
                "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow",
                "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

    phonemes2index = {k:(v+2) for v,k in enumerate(phonemes)}
    index2phonemes = {(v+2):k for v,k in enumerate(phonemes)}

    phonemse_reduce_mapping = {"b": "b", "bcl": "h#", "d": "d", "dcl": "h#", "g": "g",
                               "gcl": "h#", "p": "p", "pcl": "h#", "t": "t", "tcl": "h#",
                               "k": "k", "kcl": "h#", "dx": "dx", "q": "q", "jh": "jh",
                               "ch": "ch", "s": "s", "sh": "sh", "z": "z", "zh": "sh",
                               "f": "f", "th": "th", "v": "v", "dh": "dh", "m": "m",
                               "n": "n", "ng": "ng", "em": "m", "en": "n", "eng": "ng",
                               "nx": "n", "l": "l", "r": "r", "w": "w", "y": "y",
                               "hh": "hh", "hv": "hh", "el": "l", "iy": "iy", "ih": "ih",
                               "eh": "eh", "ey": "ey", "ae": "ae", "aa": "aa", "aw": "aw",
                               "ay": "ay", "ah": "ah", "ao": "aa", "oy": "oy", "ow": "ow",
                               "uh": "uh", "uw": "uw", "ux": "uw", "er": "er", "ax": "ah",
                               "ix": "ih", "axr": "er", "ax-h": "ah", "pau": "h#", "epi": "h#",
                               "h#": "h#"}

    # inverse index into phn
    seq = [index2phonemes[idx] for idx in seq]
    # collapse phn
    seq = [phonemse_reduce_mapping[phn] for phn in seq]
    # Discard phn q
    if drop_q:
        seq = [phn for phn in seq if phn != "q"]
    else:
        seq = [phn if phn != "q" else ' ' for phn in seq]
    if return_phn:
        return seq

    # Transfer back into index seqence for Evaluation
    seq = [phonemes2index[phn] for phn in seq]

    return seq
