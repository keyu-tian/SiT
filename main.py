import argparse
import datetime
import numpy as np
import time
import torch
import os
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets.prepare_datasets import build_dataset
from dist import TorchDistManager
from engine import train_SSL, evaluate_SSL, train_finetune, evaluate_finetune
from log import create_loggers
from losses import MTL_loss

from samplers import RASampler
# for register
from models import vision_transformer_SiT   # SiT_base_patch16_224
from models import swin_transformer         # SwinT_base_patch4_224

import utils
from utils import change_builtin_print, LossOpt


def get_args_parser():
    parser = argparse.ArgumentParser('SiT training and evaluation script', add_help=False)

    parser.add_argument('--main_py_rel_path', type=str, required=True)
    parser.add_argument('--exp_dirname', type=str, required=True)
    
    parser.add_argument('--batch-size', default=72, type=int)
    parser.add_argument('--epochs', default=501, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='SiT_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # todo: --model-ema
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=4, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data-set', default='stl10', choices=['cifar100', 'cifar10', 'stl10', 'tinyimagenet'],
                        type=str, help='dataset name')
    parser.add_argument('--dataset_location', default='downloaded_datasets', type=str,
                        help='dataset location - dataset will be downloaded to this folder')
    
    parser.add_argument('--num_imgs_per_cat', default=None, type=int, help='Number of images per training category')
    parser.add_argument('--SiT_LinearEvaluation', default=0, type=int, help='If true, the backbone of the system will be freezed')
    parser.add_argument('--representation-size', default=None, type=int, help='nonLinear head')
    
    ## if training-mode is SSL, self-supervised training, else, finetune
    parser.add_argument('--training-mode', default='SSL', choices=['SSL', 'finetune'],
                        type=str, help='training mode')
    parser.add_argument('--validate-every', default=1, type=int, help='validate and save the checkpoints every n epochs')
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    return parser


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main(args, dist, st_lg, tb_lg):
    # disable any harsh augmentation in case of Self-supervise training
    if args.training_mode == 'SSL':
        print("NOTE: Smoothing, Mixup, CutMix, and AutoAugment will be disabled in case of Self-supervise training")
        args.smoothing = args.reprob = args.reprob = args.recount = args.mixup = args.cutmix = 0.0
        args.aa = ''
        
        if args.SiT_LinearEvaluation == 1:
            print("Warning: Linear Evaluation should be set to 0 during SSL training - changing SiT_LinearEvaluation to 0")
            args.SiT_LinearEvaluation = 0
    
    utils.print_args(args)
    
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    print("Loading dataset ....")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem, drop_last=True, collate_fn=collate_fn)
    
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
                                                  batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
                                                  pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn)
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model, pretrained=False, num_classes=args.nb_classes,
        # todo：根据以下kw的字段，添加swin的构造器参数，尤其注意最后一层变成MLP
        drop_rate=args.drop, drop_path_rate=args.drop_path, representation_size=args.representation_size,
        drop_block_rate=None, training_mode=args.training_mode)
    
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['rot_head.weight', 'rot_head.bias', 'contrastive_head.weight', 'contrastive_head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        
        model.load_state_dict(checkpoint_model, strict=False)
    
    model.to(device)
    
    # Freeze the backbone in case of linear evaluation
    if args.SiT_LinearEvaluation == 1:
        requires_grad(model, False)
        
        # todo: 根据这些字段，修改swin
        model.rot_head.weight.requires_grad = True
        model.rot_head.bias.requires_grad = True
        
        model.contrastive_head.weight.requires_grad = True
        model.contrastive_head.bias.requires_grad = True
        
        if args.representation_size is not None:
            model.pre_logits_rot.fc.weight.requires_grad = True
            model.pre_logits_rot.fc.bias.requires_grad = True
            
            model.pre_logits_contrastive.fc.weight.requires_grad = True
            model.pre_logits_contrastive.fc.bias.requires_grad = True
    
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '', resume='')
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.dev_idx], output_device=dist.dev_idx)
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = LossOpt()
    
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    if args.training_mode == 'SSL':
        criterion = MTL_loss(args.device, args.batch_size)
    elif args.training_mode == 'finetune' and args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    tr_iters = len(data_loader_train)
    va_iters = len(data_loader_val)
    if args.eval:
        test_stats = evaluate_SSL(st_lg, tb_lg, model, data_loader_val, device, -1, tr_iters * -1, va_iters, args.output_dir)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
        return
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        val_freq = args.validate_every * (1 if args.start_epoch >= args.epochs // 2 else 3)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if args.training_mode == 'SSL':
            train_stats = train_SSL(
                st_lg, tb_lg, model, criterion, data_loader_train, optimizer, device, epoch, args.epochs, tr_iters, loss_scaler,
                args.clip_grad, model_ema, mixup_fn)
        else:
            train_stats = train_finetune(
                st_lg, tb_lg, model, criterion, data_loader_train, optimizer, device, epoch, args.epochs, tr_iters, loss_scaler,
                args.clip_grad, model_ema, mixup_fn)
        
        lr_scheduler.step(epoch)
        
        if epoch % val_freq == 0 or epoch == args.epochs - 1:
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

            if args.training_mode == 'SSL':
                test_stats = evaluate_SSL(st_lg, tb_lg, model, data_loader_val, device, epoch, tr_iters * epoch, va_iters, args.output_dir)
            else:
                test_stats = evaluate_finetune(st_lg, tb_lg, model, data_loader_val, device, epoch, tr_iters * epoch, va_iters)
                
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')
                tb_lg.add_scalars('finetune_va_ep/acc1', {'best': max_accuracy}, epoch)
                tb_lg.add_scalars('finetune_va_it/acc1', {'best': max_accuracy}, tr_iters * epoch)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def dist_main():
    parser = argparse.ArgumentParser('SiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    sh_root = os.getcwd()
    exp_root = os.path.join(sh_root, args.exp_dirname)
    os.chdir(args.main_py_rel_path)
    prj_root = os.getcwd()
    os.chdir(sh_root)
    
    dist = TorchDistManager(args.exp_dirname, 'auto', 'auto')
    lg, st_lg, tb_lg = create_loggers(prj_root, sh_root, exp_root, dist)
    change_builtin_print(lg)
    
    args.distributed = True
    args.world_size = dist.world_size
    args.output_dir = exp_root
    args.data_set = args.data_set.strip().lower()
    args.dataset_location = os.path.join(os.path.expanduser('~'), 'datasets', args.data_set)
    
    try:
        main(args, dist, st_lg, tb_lg)
    except Exception as e:
        if dist.rank == 0:
            raise e
        else:
            exit(-1)


if __name__ == '__main__':
    dist_main()
