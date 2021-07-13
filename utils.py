# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import heapq
import io
import os
import time
from collections import defaultdict, deque
from typing import List

import torch
import torch.distributed as tdist
import torch.nn.functional as F


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def master_echo(is_master, msg: str, color='33', tail=''):
    if is_master:
        os.system(f'echo -e "\033[{color}m{msg}\033[0m{tail}"')


def get_bn(bn_mom):
    def BN_func(*args, **kwargs):
        kwargs.update({'momentum': bn_mom})
        return torch.nn.BatchNorm2d(*args, **kwargs)
    
    return BN_func


class TopKHeap(list):
    
    def __init__(self, maxsize):
        super(TopKHeap, self).__init__()
        self.maxsize = maxsize
        assert self.maxsize >= 1
    
    def push_q(self, x):
        if len(self) < self.maxsize:
            heapq.heappush(self, x)
        elif x > self[0]:
            heapq.heappushpop(self, x)
    
    def pop_q(self):
        return heapq.heappop(self)
    
    def __repr__(self):
        return str(sorted([x for x in self], reverse=True))


class AmpScaler:
    state_dict_key = "amp_scaler"
    
    def __init__(self):
        ...
        # todo: 关闭 amp 自动精度 半精度
        # self._scaler = None # torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        # self._scaler.scale(loss).backward(create_graph=create_graph)
        # if clip_grad is not None:
        #     assert parameters is not None
        #     self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        #     norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_grad))
        # else:
        #     norm = None
        # self._scaler.step(optimizer)
        # self._scaler.update()
        # return norm
        
        optimizer.step()
        if clip_grad is not None:
            assert parameters is not None
            print(f'============== type(parameters): {type(parameters)}')
            print(f'============== len (parameters): {len(parameters)}')
            print(f'============== len (gard not N): {len([p for p in parameters if p.grad is not None])}')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_grad))
        else:
            norm = None
        return norm
    
    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        else:
            return {AmpScaler.state_dict_key: []}
    
    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)
        else:
            pass
        
        
class AverageMeter(object):
    def __init__(self, length=0):
        self.length = round(length)
        if self.length > 0:
            self.queuing = True
            self.val_history = []
            self.num_history = []
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def reset(self):
        if self.length > 0:
            self.val_history.clear()
            self.num_history.clear()
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def update(self, val, num=1):
        self.val_sum += val * num
        self.num_sum += num
        self.last = val / num
        if self.queuing:
            self.val_history.append(val)
            self.num_history.append(num)
            if len(self.val_history) > self.length:
                self.val_sum -= self.val_history[0] * self.num_history[0]
                self.num_sum -= self.num_history[0]
                del self.val_history[0]
                del self.num_history[0]
        self.avg = self.val_sum / self.num_sum
    
    def time_preds(self, counts):
        remain_secs = counts * self.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        return remain_time, finish_time


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # if not is_dist_avail_and_initialized():
        #     return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def change_builtin_print(lg, is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


def get_world_size():
    # if not is_dist_avail_and_initialized():
    #     return 1
    return tdist.get_world_size()


def get_rank():
    # if not is_dist_avail_and_initialized():
    #     return 0
    return tdist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def print_args(args):
    print('***********************************************')
    print('*', ' '.ljust(9), 'Training Mode is ', args.training_mode.ljust(15), '*')
    print('***********************************************')
    print('Dataset Name: ', args.data_set)
    print('Dataset will be downloaded/loaded from: ', args.dataset_location)
    print('---------- Model ----------')
    print('Model name: ', args.model)
    print('Finetune From: ', args.finetune)
    print('Resume From: ', args.resume)
    print('Output To: ', args.output_dir)
    print('Number of GPUs: ', args.world_size)
    if args.SiT_LinearEvaluation:
        print('NOTE: The backbone is freezed, only last classifier is trainable!')
    print('---------- Optimizer ----------')
    print('Learning Rate: ', args.lr)
    print('Weight Decay: ', args.weight_decay)
    print('Batchsize: ', args.batch_size)


def _compute_padding(kernel_size: List[int]) -> List[int]:
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def kornia_filter_2D(input, kernel, padding_mode):
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = _compute_padding([height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=padding_mode)
    
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))
    
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    
    return output.view(b, c, h, w)
