# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils
import torch
from . import FairseqCriterion, register_criterion


@register_criterion('cokd_loss')
class COKDCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.kd_alpha = args.kd_alpha
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--kd-alpha', default=0.5, type=float)
        parser.add_argument('--num-teachers', default=1, type=int)
        # fmt: on

    def forward(self, model, sample, reduce=True, teachers = None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if teachers is None:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        else:
            net_output_teachers = [teacher(**sample['net_input']) for teacher in teachers]
            loss, nll_loss = self.compute_kd_loss(model, net_output, net_output_teachers, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)#fairseq/models/fairseq_model.py:sample['target']
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_kd_loss(self, model, net_output, net_output_teachers, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        teacher_probs = [model.get_normalized_probs(net_output_teacher, log_probs=False) for net_output_teacher in net_output_teachers]
        teacher_prob = torch.mean(torch.stack(teacher_probs, dim = 0), dim = 0)
        teacher_prob = teacher_prob.view(-1, teacher_prob.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        kd_loss = (-lprobs * teacher_prob).sum(dim = -1, keepdim=True)[non_pad_mask]
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
            kd_loss = kd_loss.sum()
        loss = nll_loss * (1 - self.kd_alpha) + kd_loss * self.kd_alpha
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
