import os
import time
import random
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from os import makedirs
from os.path import exists, join

from src.ops import get_wjs
from datasets.train_feeder_r2et_smal33 import Feeder
from src.model_skeleton_aware_smal33 import RetNet, MotionDis

# SMAL33 limb joints for recon / sem loss weighting (includes paws).
#  9 LeftForeLeg,  10 LeftFrontPaw
# 13 RightForeLeg, 14 RightFrontPaw
# 19 LeftShin,      20 LeftHock,      21 LeftHindPaw
# 23 RightShin,     24 RightHock,     25 RightHindPaw
ATTENTION_LIST_RECON = [9, 10, 13, 14, 19, 20, 21, 23, 24, 25]
ATTENTION_LIST_SEM = [9, 10, 13, 14, 19, 20, 21, 23, 24, 25]


def get_parser():
    parser = argparse.ArgumentParser(
        description="R2ET skeleton-aware training for SMAL33 pet data"
    )
    parser.add_argument(
        "--config",
        default="./config/train_skeleton_aware_smal33.yaml",
        help="path to the configuration file",
    )
    parser.add_argument("--phase", default="train", help="train or test")
    parser.add_argument(
        "--work-dir",
        default="./work_dir/r2et_skeleton_aware_smal33",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--model-save-name",
        default="r2et_skeleton_aware_smal33",
        help="model saved name",
    )
    parser.add_argument(
        "--train-feeder-args",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--alpha", type=float, default=100.0, help="threshold for euler angle"
    )
    parser.add_argument(
        "--nu", type=float, default=100.0, help="weight factor for semantic geometry loss"
    )
    parser.add_argument(
        "--mu", type=float, default=10.0, help="weight factor for twist loss"
    )
    parser.add_argument(
        "--omega", type=float, default=5.0, help="weight factor for temporal smooth loss"
    )
    parser.add_argument("--euler-ord", default="yzx", help="order of the euler angle")
    parser.add_argument(
        "--max-length", type=int, default=32, help="max sequence length: T"
    )
    parser.add_argument(
        "--num-joint", type=int, default=33, help="number of the joints"
    )
    parser.add_argument(
        "--kp", type=float, default=0.8, help="keep prob in dropout layers"
    )
    parser.add_argument("--margin", type=float, default=0.3, help="fake score margin")
    parser.add_argument(
        "--lam", type=int, default=2, help="balancing factor for GAN loss"
    )
    parser.add_argument(
        "--ret-model-args",
        type=dict,
        default=dict(),
        help="the arguments of retargetor",
    )
    parser.add_argument(
        "--dis-model-args",
        type=dict,
        default=dict(),
        help="the arguments of discriminator",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )
    parser.add_argument("--epoch", type=int, default=30, help="training epoch")
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of DataLoader workers",
    )
    return parser


class AverageMeter(object):
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_cuda_devices(device_ids):
    """
    Bind physical GPU ids via CUDA_VISIBLE_DEVICES, then use logical cuda:0..N-1.
    Example: device_ids=[1] -> only physical GPU1 visible as cuda:0.
    """
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
    torch_device = torch.device("cuda:0")
    parallel_device_ids = list(range(len(device_ids)))
    return torch_device, parallel_device_ids, device_ids


def print_log_txt(s, work_dir, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f"[ {localtime} ] {s}"
    print(s)
    log_path = join(work_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        print(s, file=f)


def train(
    retarget_net,
    discriminator,
    data_loader,
    optimizer_r,
    optimizer_d,
    scheduler_r,
    scheduler_d,
    global_mean,
    global_std,
    local_mean,
    local_std,
    quat_mean,
    quat_std,
    parents,
    epoch,
    logger,
    arg,
    device,
):
    pbar = tqdm(total=len(data_loader), ncols=160)
    meters = {
        "ret": AverageMeter(),
        "disc": AverageMeter(),
        "base": AverageMeter(),
        "local_ae": AverageMeter(),
        "quat_ae": AverageMeter(),
        "twist": AverageMeter(),
        "gen": AverageMeter(),
        "sem": AverageMeter(),
        "sem_weighted": AverageMeter(),
        "smooth": AverageMeter(),
        "smooth_weighted": AverageMeter(),
        "time": AverageMeter(),
    }

    global_mean = torch.from_numpy(global_mean).to(device)
    global_std = torch.from_numpy(global_std).to(device)

    for batch_idx, (
        indexesA,
        indexesB,
        seqA,
        skelA,
        seqB,
        skelB,
        aeReg,
        mask,
        inp_height,
        tgt_height,
        shapeA,
        shapeB,
        quatA_cp,
    ) in enumerate(data_loader):
        seqA = seqA.float().to(device)
        skelA = skelA.float().to(device)
        seqB = seqB.float().to(device)
        skelB = skelB.float().to(device)
        aeReg = aeReg.float().to(device)
        mask = mask.float().to(device)
        inp_height = inp_height.float().to(device)
        tgt_height = tgt_height.float().to(device)
        quatA_cp = quatA_cp.float().to(device)
        shapeA = shapeA.float().to(device)
        shapeB = shapeB.float().to(device)

        pbar.set_description("Train Epoch %i  Step %i" % (epoch + 1, batch_idx))
        start_time = time.time()

        retarget_net.train()
        discriminator.eval()
        optimizer_r.zero_grad()

        (
            localA_gt,
            localB_rt,
            localB_gt,
            globalA_gt,
            globalB_rt,
            quatB_rt,
        ) = retarget_net(
            seqA,
            seqB,
            skelA,
            skelB,
            shapeA,
            shapeB,
            quatA_cp,
            inp_height,
            tgt_height,
            local_mean,
            local_std,
            quat_mean,
            quat_std,
            parents,
        )

        num_joint = arg.num_joint
        batch_size = localA_gt.shape[0]
        max_len = arg.max_length

        wjsA = get_wjs(localA_gt, globalA_gt)
        wjsA = torch.reshape(wjsA, [batch_size, max_len, num_joint, 3])
        wjsB = get_wjs(localB_rt, globalB_rt)
        wjsB = torch.reshape(wjsB, [batch_size, max_len, num_joint, 3])

        inpxyz = torch.mean(wjsA, dim=2)
        motion_real = torch.divide(inpxyz, inp_height[:, :, None]).float()
        motion_real = motion_real.permute(0, 2, 1).contiguous()
        score_real = discriminator(motion_real)

        tgtxyz = torch.mean(wjsB, dim=2)
        motion_fake = torch.divide(tgtxyz, tgt_height[:, :, None]).float()
        motion_fake = motion_fake.permute(0, 2, 1).contiguous()
        score_fake = discriminator(motion_fake)

        quatA_denorm = (
            quatA_cp * torch.from_numpy(quat_std).cuda(quatA_cp.device)[None, :]
            + torch.from_numpy(quat_mean).cuda(quatA_cp.device)[None, :]
        )
        quatA_denorm = quatA_denorm.float()

        local_ae_loss, quat_ae_loss = RetNet.get_recon_loss(
            ATTENTION_LIST_RECON,
            num_joint,
            aeReg,
            mask,
            localB_rt,
            localB_gt,
            quatA_denorm,
            quatB_rt,
        )

        twist_loss = RetNet.get_rot_cons_loss(arg.alpha, arg.euler_ord, quatB_rt)
        gen_loss = RetNet.get_gen_loss(score_fake, aeReg)
        base_loss = local_ae_loss + quat_ae_loss + arg.mu * twist_loss

        local_std_ts = torch.from_numpy(local_std).cuda(skelB.device)
        local_mean_ts = torch.from_numpy(local_mean).cuda(skelB.device)

        bs = quatB_rt.shape[0]
        localB_rt_denorm = (
            localB_rt * local_std_ts[:, None, :, :] + local_mean_ts[:, None, :, :]
        ).float()
        localA_gt_denorm = (
            localA_gt * local_std_ts[:, None, :, :] + local_mean_ts[:, None, :, :]
        ).float()
        smooth_loss = RetNet.get_smooth_loss(localB_rt_denorm, mask)

        normed_matrixB, normed_matrixA = RetNet.get_rela_matrix(
            localB_rt_denorm, localA_gt_denorm, tgt_height, inp_height
        )
        sem_loss = RetNet.get_sem_loss(
            ATTENTION_LIST_SEM,
            num_joint,
            normed_matrixA,
            normed_matrixB,
            mask,
        )

        ret_loss = (
            arg.lam * gen_loss
            + base_loss
            + arg.nu * sem_loss
            + arg.omega * smooth_loss
        )
        ret_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(retarget_net.parameters(), max_norm=25)
        optimizer_r.step()

        retarget_net.eval()
        discriminator.train()
        optimizer_d.zero_grad()

        score_real = discriminator(motion_real.detach())
        score_fake = discriminator(motion_fake.detach())
        disc_loss = RetNet.get_dis_loss(score_real, score_fake, aeReg)

        disc_updated = False
        for i in range(score_fake.shape[0]):
            if score_fake[i] > arg.margin:
                disc_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=25)
                optimizer_d.step()
                disc_updated = True
                break

        end_time = time.time()
        step_time = end_time - start_time
        meters["time"].update(step_time)
        meters["ret"].update(float(ret_loss.item()))
        meters["disc"].update(float(disc_loss.item()))
        meters["base"].update(float(base_loss.item()))
        meters["local_ae"].update(float(local_ae_loss.item()))
        meters["quat_ae"].update(float(quat_ae_loss.item()))
        meters["twist"].update(float(twist_loss.item()))
        meters["gen"].update(float(gen_loss.item()))
        meters["sem"].update(float(sem_loss.item()))
        meters["sem_weighted"].update(float((arg.nu * sem_loss).item()))
        meters["smooth"].update(float(smooth_loss.item()))
        meters["smooth_weighted"].update(float((arg.omega * smooth_loss).item()))

        pbar.set_postfix(
            loss_r=float(ret_loss.item()),
            loss_d=float(disc_loss.item()),
            local_ae=float(local_ae_loss.item()),
            sem=float(sem_loss.item()),
            smooth=float(smooth_loss.item()),
            time=step_time,
        )
        pbar.update(1)

    scheduler_r.step()
    scheduler_d.step()
    pbar.close()

    logger.add_scalar("train/loss_ret", meters["ret"].avg, epoch)
    logger.add_scalar("train/loss_disc", meters["disc"].avg, epoch)
    logger.add_scalar("train/loss_base", meters["base"].avg, epoch)
    logger.add_scalar("train/loss_local_ae", meters["local_ae"].avg, epoch)
    logger.add_scalar("train/loss_quat_ae", meters["quat_ae"].avg, epoch)
    logger.add_scalar("train/loss_twist", meters["twist"].avg, epoch)
    logger.add_scalar("train/loss_gen", meters["gen"].avg, epoch)
    logger.add_scalar("train/loss_sem", meters["sem"].avg, epoch)
    logger.add_scalar("train/loss_sem_weighted", meters["sem_weighted"].avg, epoch)
    logger.add_scalar("train/loss_smooth", meters["smooth"].avg, epoch)
    logger.add_scalar(
        "train/loss_smooth_weighted", meters["smooth_weighted"].avg, epoch
    )
    logger.add_scalar("train/lr_ret", optimizer_r.param_groups[0]["lr"], epoch)
    logger.add_scalar("train/lr_dis", optimizer_d.param_groups[0]["lr"], epoch)
    logger.add_scalar("train/step_time", meters["time"].avg, epoch)

    return meters


def format_epoch_log(epoch_idx, total_epochs, meters, lr_ret, lr_dis):
    return (
        f"Epoch {epoch_idx + 1}/{total_epochs} | "
        f"lr_ret={lr_ret:.6g} lr_dis={lr_dis:.6g} | "
        f"time={meters['time'].avg:.4f}s | "
        f"ret={meters['ret'].avg:.6f} disc={meters['disc'].avg:.6f} | "
        f"base={meters['base'].avg:.6f} local_ae={meters['local_ae'].avg:.6f} "
        f"quat_ae={meters['quat_ae'].avg:.6f} twist={meters['twist'].avg:.6f} "
        f"gen={meters['gen'].avg:.6f} sem={meters['sem'].avg:.6f} "
        f"sem_w={meters['sem_weighted'].avg:.6f} smooth={meters['smooth'].avg:.6f} "
        f"smooth_w={meters['smooth_weighted'].avg:.6f}"
    )


def main(arg):
    if not exists(arg.work_dir):
        makedirs(arg.work_dir)

    device = arg.torch_device
    print_log_txt("Initializing SMAL33 skeleton-aware training", arg.work_dir)
    print_log_txt(
        f"Using physical GPUs {arg.physical_device_ids} "
        f"-> logical {device} (DataParallel ids={arg.parallel_device_ids})",
        arg.work_dir,
    )
    data_feeder = Feeder(**arg.train_feeder_args)
    retarget_net = RetNet(**arg.ret_model_args).to(device)
    discriminator_net = MotionDis(**arg.dis_model_args).to(device)
    if len(arg.parallel_device_ids) > 1:
        retarget_net = nn.DataParallel(
            retarget_net, device_ids=arg.parallel_device_ids
        )
        discriminator_net = nn.DataParallel(
            discriminator_net, device_ids=arg.parallel_device_ids
        )

    data_loader = torch.utils.data.DataLoader(
        dataset=data_feeder,
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        shuffle=True,
    )

    optimizer_ret = optim.Adam(
        retarget_net.parameters(),
        lr=arg.base_lr,
        weight_decay=arg.weight_decay,
        betas=(0.5, 0.999),
    )
    optimizer_dis = optim.Adam(
        discriminator_net.parameters(),
        lr=arg.base_lr,
        weight_decay=arg.weight_decay,
        betas=(0.5, 0.999),
    )

    scheduler_ret = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ret, milestones=arg.step, gamma=0.1, last_epoch=-1
    )
    scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_dis, milestones=arg.step, gamma=0.1, last_epoch=-1
    )

    train_writer = SummaryWriter(
        join(arg.work_dir, arg.model_save_name, "train_log"), "train"
    )

    arg_dict = vars(arg)
    with open(join(arg.work_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(arg_dict, f)

    print_log_txt(
        f"Dataset sequences={len(data_feeder)} | "
        f"num_joint={arg.num_joint} | max_length={arg.max_length} | "
        f"attention_recon={ATTENTION_LIST_RECON} | omega={arg.omega}",
        arg.work_dir,
    )

    for i in range(arg.epoch):
        meters = train(
            retarget_net,
            discriminator_net,
            data_loader,
            optimizer_ret,
            optimizer_dis,
            scheduler_ret,
            scheduler_dis,
            data_feeder.global_mean,
            data_feeder.global_std,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
            i,
            train_writer,
            arg,
            device,
        )

        lr_ret = optimizer_ret.param_groups[0]["lr"]
        lr_dis = optimizer_dis.param_groups[0]["lr"]
        print_log_txt(
            format_epoch_log(i, arg.epoch, meters, lr_ret, lr_dis),
            arg.work_dir,
            print_time=True,
        )

        if (i + 1) % 2 == 0:
            state_dict_ret = retarget_net.state_dict()
            state_dict_dis = discriminator_net.state_dict()

            weights_gen = OrderedDict([[k, v.cpu()] for k, v in state_dict_ret.items()])
            ret_path = join(
                arg.work_dir, arg.model_save_name + "_ret-" + str(i + 1) + ".pt"
            )
            torch.save(weights_gen, ret_path)
            print_log_txt(f"Saved retargetor checkpoint: {ret_path}", arg.work_dir)

            weights_dis = OrderedDict([[k, v.cpu()] for k, v in state_dict_dis.items()])
            dis_path = join(
                arg.work_dir, arg.model_save_name + "_dis-" + str(i + 1) + ".pt"
            )
            torch.save(weights_dis, dis_path)
            print_log_txt(f"Saved discriminator checkpoint: {dis_path}", arg.work_dir)

    print_log_txt("Training finished.", arg.work_dir)
    train_writer.close()


if __name__ == "__main__":
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r", encoding="utf-8") as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG:", k)
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    torch_device, parallel_device_ids, physical_device_ids = setup_cuda_devices(
        arg.device
    )
    arg.torch_device = torch_device
    arg.parallel_device_ids = parallel_device_ids
    arg.physical_device_ids = physical_device_ids
    init_seed(arg.seed)
    main(arg)
