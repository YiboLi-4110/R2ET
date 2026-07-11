import os
import sys
import time
import random
import yaml
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import torch.optim as optim
from os import listdir, makedirs
from os.path import exists, join

from src.ops import get_wjs
from src.mesh_geometry_cache import build_mesh_geometry_cache
from datasets.train_feeder_r2et_smal33 import Feeder
from src.model_shape_aware_smal33 import RetNet, MotionDis, PAW_JOINT_INDICES

# Limb joints for cons-loss weighting (paws + hind chain).
ATTENTION_LIST_CONS = [9, 10, 13, 14, 19, 20, 21, 23, 24, 25]


def get_parser():
    parser = argparse.ArgumentParser(
        description="R2ET shape-aware training for SMAL33 pet data"
    )
    parser.add_argument(
        "--config",
        default="./config/train_shape_aware_smal33.yaml",
        help="path to the configuration file",
    )
    parser.add_argument("--phase", default="train", help="train or test")
    parser.add_argument(
        "--work_dir",
        default="./work_dir/train_shapeaware_smal33",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--mesh_path",
        default="./datasets/Planet_Zoo_FBX-smal2/train_shape/",
        help="directory of shape .npz files (fallback when mesh_paths is unset)",
    )
    parser.add_argument(
        "--mesh_paths",
        nargs="+",
        default=None,
        help=(
            "one or more shape .npz directories for geometry loss; "
            "use both shepherd train_shape and batch2_dogs_shape for mixed training"
        ),
    )
    parser.add_argument(
        "--model_save_name",
        default="r2et_shape_aware_smal33",
        help="model saved name",
    )
    parser.add_argument(
        "--train_feeder_args",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="physical GPU id(s), e.g. --device 0 1 2 3 (auto DDP via torchrun)",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size per GPU (global batch = batch_size * num_gpus)",
    )
    parser.add_argument(
        "--alpha", type=float, default=100.0, help="threshold for euler angle"
    )
    parser.add_argument(
        "--mu", type=float, default=10.0, help="weight factor for twist loss"
    )
    parser.add_argument(
        "--omega_smooth",
        type=float,
        default=0.0,
        help="weight factor for optional second-order temporal smooth loss",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.5,
        help="global weight factor for Repulsive and Attractive loss",
    )
    parser.add_argument(
        "--w_front",
        type=float,
        default=1.5,
        help="RDF weight for left/right front legs vs torso+head (when enabled)",
    )
    parser.add_argument(
        "--w_hind",
        type=float,
        default=3.0,
        help="RDF weight for left/right hind legs vs torso+head",
    )
    parser.add_argument(
        "--w_tail_hind",
        type=float,
        default=3.5,
        help="RDF weight for tail vs hind legs (highest priority)",
    )
    parser.add_argument(
        "--enable_front_rdf",
        action="store_true",
        default=False,
        help="train front-leg RDF (left/right front vs torso+head)",
    )
    parser.add_argument(
        "--disable_front_rdf",
        action="store_true",
        help="skip front-leg RDF entirely (default; saves compute)",
    )
    parser.add_argument(
        "--w_att",
        type=float,
        default=1.0,
        help="ADF weight for paw contact on torso",
    )
    parser.add_argument(
        "--enable_att_loss",
        action="store_true",
        default=True,
        help="compute ADF paw-contact loss (disable for extra speed)",
    )
    parser.add_argument(
        "--disable_att_loss",
        action="store_true",
        help="skip ADF loss entirely",
    )
    parser.add_argument(
        "--sdf_grid_size",
        type=int,
        default=24,
        help="SDF voxel grid resolution (lower=faster, default 24; original 32)",
    )
    parser.add_argument(
        "--geo_frame_stride",
        type=int,
        default=2,
        help="evaluate geometry losses every N frames (default 2)",
    )
    parser.add_argument(
        "--tao", type=float, default=0.005, help="weight factor for gate regularization"
    )
    parser.add_argument("--euler_ord", default="yzx", help="order of the euler angle")
    parser.add_argument(
        "--max_length", type=int, default=32, help="max sequence length: T"
    )
    parser.add_argument(
        "--num_joint", type=int, default=33, help="number of the joints"
    )
    parser.add_argument(
        "--kp", type=float, default=0.8, help="keep prob in dropout layers"
    )
    parser.add_argument("--margin", type=float, default=0.3, help="fake score margin")
    parser.add_argument(
        "--lam", type=int, default=2, help="balance the GAN loss"
    )
    parser.add_argument(
        "--ret_model_args",
        type=dict,
        default=dict(),
        help="the arguments of retargetor",
    )
    parser.add_argument(
        "--dis_model_args",
        type=dict,
        default=dict(),
        help="the arguments of discriminator",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )
    parser.add_argument(
        "--epoch", type=int, default=50, help="stop training in which epoch"
    )
    parser.add_argument(
        "--ret_weights",
        default="",
        help="stage-1 retargetor checkpoint (.pt)",
    )
    parser.add_argument(
        "--dis_weights",
        default="",
        help="stage-1 discriminator checkpoint (.pt)",
    )
    parser.add_argument(
        "--ignore_weights", default=[], help="ret weights keys to skip when loading"
    )
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="DataLoader worker count per GPU"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="TCP port for DDP init when auto-launching torchrun",
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


def unwrap_module(model):
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


def relaunch_with_torchrun(script_args, device_ids, master_port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(device_ids)}",
        f"--master_port={master_port}",
        os.path.abspath(__file__),
    ] + script_args
    print(
        f"Launching DDP on physical GPUs {device_ids} "
        f"({len(device_ids)} processes)...",
        flush=True,
    )
    subprocess.check_call(cmd, env=env)


def init_distributed_env(device_ids):
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    physical_device_ids = list(device_ids)

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return device, rank, world_size, local_rank, True, physical_device_ids

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in physical_device_ids)
    device = torch.device("cuda:0")
    return device, 0, 1, 0, False, physical_device_ids


def is_main_process(arg):
    return arg.rank == 0


def print_log_txt(s, work_dir, arg, print_time=True):
    if not is_main_process(arg):
        return
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f"[ {localtime} ] {s}"
    print(s)
    with open(join(work_dir, "log.txt"), "a", encoding="utf-8") as f:
        print(s, file=f)


def all_reduce_meter(meter, device, world_size):
    if world_size == 1:
        return meter.avg
    stats = torch.tensor(
        [meter.sum, float(meter.count)], dtype=torch.float64, device=device
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_sum, total_count = stats.tolist()
    if total_count <= 0:
        return 0.0
    return total_sum / total_count


def reduce_epoch_meters(arg, device, meters):
    """Reduce AverageMeters across all DDP ranks (every rank must call this)."""
    return {
        name: all_reduce_meter(meter, device, arg.world_size)
        for name, meter in meters.items()
    }


def load_model(ret_model, dis_model, arg, device):
    if arg.ret_weights:
        print_log_txt(f"Loading ret weights from {arg.ret_weights}", arg.work_dir, arg)
        ret_weights = torch.load(arg.ret_weights, map_location=device)
        ret_weights = OrderedDict(
            [
                [k.split("module.")[-1], v.to(device)]
                for k, v in ret_weights.items()
            ]
        )
        pop_lst = []
        for k in ret_weights.keys():
            for w in arg.ignore_weights:
                if w in k:
                    pop_lst.append(k)
        for k in pop_lst:
            ret_weights.pop(k)
            print_log_txt(f"Removed weight key: {k}", arg.work_dir, arg)

        try:
            ret_model.load_state_dict(ret_weights, strict=False)
            print_log_txt("Loaded stage-1 ret weights (strict=False).", arg.work_dir, arg)
        except Exception as exc:
            print_log_txt(f"Partial load failed: {exc}", arg.work_dir, arg)
            state = ret_model.state_dict()
            state.update(ret_weights)
            ret_model.load_state_dict(state, strict=False)

    if arg.dis_weights:
        print_log_txt(f"Loading dis weights from {arg.dis_weights}", arg.work_dir, arg)
        dis_weights = torch.load(arg.dis_weights, map_location=device)
        dis_weights = OrderedDict(
            [
                [k.split("module.")[-1], v.to(device)]
                for k, v in dis_weights.items()
            ]
        )
        dis_model.load_state_dict(dis_weights, strict=False)


def load_mesh_file_dic(mesh_paths, work_dir=None, arg=None):
    """Load shape .npz files from one or more directories keyed by stem name."""
    if isinstance(mesh_paths, (str, os.PathLike)):
        mesh_paths = [mesh_paths]
    mesh_file_dic = {}
    for mesh_root in mesh_paths:
        mesh_root = os.fspath(mesh_root)
        if not exists(mesh_root):
            raise FileNotFoundError(f"mesh path not found: {mesh_root}")
        file_names = sorted(
            f
            for f in listdir(mesh_root)
            if not f.startswith(".") and f.endswith(".npz")
        )
        if not file_names:
            raise FileNotFoundError(f"No .npz mesh files under {mesh_root}")
        for mesh_name in file_names:
            key = mesh_name.split(".")[0]
            if key in mesh_file_dic:
                msg = (
                    f"WARN: duplicate mesh key '{key}' in {mesh_root}, "
                    "keeping the first copy"
                )
                if arg is not None:
                    print_log_txt(msg, work_dir, arg)
                else:
                    print(msg)
                continue
            mesh_file_dic[key] = np.load(join(mesh_root, mesh_name))
    return mesh_file_dic


def validate_mesh_keys(mesh_file_dic, shape_keys, work_dir, arg):
    missing = sorted(set(shape_keys) - set(mesh_file_dic.keys()))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" ... (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise KeyError(
            "Missing mesh .npz for shape keys used by the feeder: "
            f"{preview}{suffix}. "
            "Set mesh_paths to include both shepherd train_shape and "
            "batch2_dogs_shape (or merge them into one directory)."
        )
    print_log_txt(
        f"Mesh keys validated: {len(mesh_file_dic)} npz loaded, "
        f"{len(shape_keys)} shape keys in feeder",
        work_dir,
        arg,
    )


def build_mesh_vertex_groups(mesh_file_dic, data_feeder):
    torso_bone_lst = data_feeder.torso_bone_lst.tolist()
    head_bone_lst = data_feeder.head_bone_lst.tolist()
    left_front_bone_lst = data_feeder.left_front_leg_lst.tolist()
    right_front_bone_lst = data_feeder.right_front_leg_lst.tolist()
    left_hind_bone_lst = data_feeder.left_hind_leg_lst.tolist()
    right_hind_bone_lst = data_feeder.right_hind_leg_lst.tolist()
    tail_bone_lst = data_feeder.tail_bone_lst.tolist()
    paw_bone_lst = data_feeder.paw_bone_lst.tolist()

    groups = {
        "torso": {},
        "head": {},
        "left_front": {},
        "right_front": {},
        "left_hind": {},
        "right_hind": {},
        "tail": {},
        "paws": {},
    }

    for mesh_name, fbx_data in mesh_file_dic.items():
        vertex_part_np = fbx_data["vertex_part"]
        vertex_num = vertex_part_np.shape[0]
        lst = {k: [] for k in groups.keys()}

        for i in range(vertex_num):
            part = int(vertex_part_np[i])
            if part in torso_bone_lst:
                lst["torso"].append(i)
            if part in head_bone_lst:
                lst["head"].append(i)
            if part in left_front_bone_lst:
                lst["left_front"].append(i)
            if part in right_front_bone_lst:
                lst["right_front"].append(i)
            if part in left_hind_bone_lst:
                lst["left_hind"].append(i)
            if part in right_hind_bone_lst:
                lst["right_hind"].append(i)
            if part in tail_bone_lst:
                lst["tail"].append(i)
            if part in paw_bone_lst:
                lst["paws"].append(i)

        for key in groups:
            groups[key][mesh_name] = lst[key]

    return groups


def save_checkpoint(retarget_net, arg, epoch_idx):
    """Save weights in the same key layout as single-GPU training (no module. prefix)."""
    raw_model = unwrap_module(retarget_net)
    state_dict = raw_model.state_dict()
    weights_gen = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
    ckpt_path = join(
        arg.work_dir, arg.model_save_name + "_ret-" + str(epoch_idx) + ".pt"
    )
    torch.save(weights_gen, ckpt_path)
    print_log_txt(f"Saved checkpoint: {ckpt_path}", arg.work_dir, arg)


def train(
    retarget_net,
    discriminator,
    data_loader,
    optimizer_ret,
    scheduler,
    local_mean,
    local_std,
    quat_mean,
    quat_std,
    parents,
    mesh_file_dic,
    mesh_groups,
    mesh_geom_cache,
    epoch,
    logger,
    arg,
    device,
):
    use_ddp = arg.world_size > 1
    module = unwrap_module(retarget_net)

    pbar = tqdm(
        total=len(data_loader),
        ncols=160,
        disable=not is_main_process(arg),
    )
    epoch_loss_ret = AverageMeter()
    epoch_loss_rep = AverageMeter()
    epoch_time = AverageMeter()
    epoch_rep_lf = AverageMeter()
    epoch_rep_rf = AverageMeter()
    epoch_rep_lh = AverageMeter()
    epoch_rep_rh = AverageMeter()
    epoch_rep_tail_hind = AverageMeter()
    epoch_att = AverageMeter()
    epoch_smooth = AverageMeter()
    epoch_smooth_weighted = AverageMeter()

    local_mean_np = local_mean
    local_std_np = local_std

    for batch_idx, (
        indexesA,
        indexesB,
        seqA,
        skelA,
        seqB,
        skelB,
        aeReg,
        mask,
        heightA,
        heightB,
        shapeA,
        shapeB,
        quatA_cp,
        shape_keyA,
        shape_keyB,
    ) in enumerate(data_loader):
        seqA = seqA.float().to(device, non_blocking=True)
        skelA = skelA.float().to(device, non_blocking=True)
        seqB = seqB.float().to(device, non_blocking=True)
        skelB = skelB.float().to(device, non_blocking=True)
        aeReg = aeReg.float().to(device, non_blocking=True)
        mask = mask.float().to(device, non_blocking=True)
        heightA = heightA.float().to(device, non_blocking=True)
        heightB = heightB.float().to(device, non_blocking=True)
        quatA_cp = quatA_cp.float().to(device, non_blocking=True)
        shapeA = shapeA.float().to(device, non_blocking=True)
        shapeB = shapeB.float().to(device, non_blocking=True)

        pbar.set_description("Train Epoch %i  Step %i" % (epoch + 1, batch_idx))
        start_time = time.time()

        retarget_net.train()
        discriminator.eval()
        optimizer_ret.zero_grad(set_to_none=True)

        (
            localA_gt,
            localB_rt,
            localB_gt,
            globalA_gt,
            globalB_rt,
            quatB_rt,
            quatB_base,
            localB_base,
            weights_sp,
        ) = retarget_net(
            seqA,
            seqB,
            skelA,
            skelB,
            shapeA,
            shapeB,
            quatA_cp,
            heightA,
            heightB,
            local_mean_np,
            local_std_np,
            quat_mean,
            quat_std,
            parents,
        )

        num_joint = arg.num_joint
        batch_size = localA_gt.shape[0]
        max_len = arg.max_length

        wjsB = get_wjs(localB_rt, globalB_rt)
        wjsB = torch.reshape(wjsB, [batch_size, max_len, num_joint, 3])
        tgtxyz = torch.mean(wjsB, dim=2)
        motion_fake = torch.divide(tgtxyz, heightB[:, :, None]).float()
        motion_fake = motion_fake.permute(0, 2, 1).contiguous()
        score_fake = discriminator(motion_fake)

        local_ae_loss, quat_ae_loss = RetNet.get_cons_loss(
            ATTENTION_LIST_CONS,
            num_joint,
            mask,
            localB_rt,
            localB_base,
            quatB_rt,
            quatB_base,
        )
        twist_loss = RetNet.get_rot_cons_loss(arg.alpha, arg.euler_ord, quatB_rt)
        gen_loss = RetNet.get_gen_loss(score_fake, aeReg)
        regular_loss = RetNet.get_regularization_loss(weights_sp, mask)
        if arg.omega_smooth > 0:
            local_std_ts = torch.from_numpy(local_std_np).to(localB_rt.device)
            local_mean_ts = torch.from_numpy(local_mean_np).to(localB_rt.device)
            local_std_ts = local_std_ts.reshape(1, 1, num_joint, 3)
            local_mean_ts = local_mean_ts.reshape(1, 1, num_joint, 3)
            localB_rt_denorm = (
                localB_rt * local_std_ts
                + local_mean_ts
            ).float()
            smooth_loss = RetNet.get_smooth_loss(localB_rt_denorm, mask)
        else:
            smooth_loss = torch.tensor(0.0, device=localB_rt.device)
        base_loss = (
            local_ae_loss + quat_ae_loss + arg.mu * twist_loss + arg.tao * regular_loss
        )

        bs = quatB_rt.shape[0]
        t_poseB = torch.reshape(skelB[:, 0, :], [bs, num_joint, 3])
        t_poseB = (
            t_poseB * torch.from_numpy(local_std_np).to(t_poseB.device)
            + torch.from_numpy(local_mean_np).to(t_poseB.device)
        )
        t_poseB = t_poseB.float()

        rep_loss_lf, rep_loss_rf, rep_loss_lh, rep_loss_rh = 0, 0, 0, 0
        rep_loss_tail_hind = 0
        att_loss = 0
        compute_att = arg.enable_att_loss and not arg.disable_att_loss
        compute_front = arg.enable_front_rdf and not arg.disable_front_rdf

        for i in range(bs):
            mesh_name = shape_keyB[i]
            if mesh_name not in mesh_geom_cache:
                raise KeyError(
                    f"Missing mesh geometry cache for shape key '{mesh_name}'. "
                    "Ensure mesh_paths covers shepherd train_shape and "
                    "batch2_dogs_shape."
                )
            cache_entry = mesh_geom_cache[mesh_name]
            vertices = cache_entry["vertices"].to(device, non_blocking=True)
            sk_weights = cache_entry["skin_weights"].to(device, non_blocking=True)

            lf, rf, lh, rh, tail_hind = RetNet.get_rep_loss_part(
                parents,
                quatB_rt[i],
                t_poseB[i],
                vertices,
                sk_weights,
                mesh_groups["torso"][mesh_name],
                mesh_groups["head"][mesh_name],
                mesh_groups["left_front"][mesh_name],
                mesh_groups["right_front"][mesh_name],
                mesh_groups["left_hind"][mesh_name],
                mesh_groups["right_hind"][mesh_name],
                mesh_groups["tail"][mesh_name],
                True,
                sdf_grid_size=arg.sdf_grid_size,
                frame_stride=arg.geo_frame_stride,
                hull_cache=cache_entry,
                compute_front_rdf=compute_front,
            )
            if compute_att:
                att_loss += RetNet.get_att_loss(
                    parents,
                    quatB_rt[i],
                    t_poseB[i],
                    vertices,
                    sk_weights,
                    mesh_groups["torso"][mesh_name],
                    mesh_groups["paws"][mesh_name],
                    sdf_grid_size=arg.sdf_grid_size,
                    frame_stride=arg.geo_frame_stride,
                    hull_cache=cache_entry,
                )
            if compute_front:
                rep_loss_lf += lf
                rep_loss_rf += rf
            rep_loss_lh += lh
            rep_loss_rh += rh
            rep_loss_tail_hind += tail_hind

        if compute_front:
            rep_loss_lf /= bs
            rep_loss_rf /= bs
        else:
            rep_loss_lf = torch.tensor(0.0, device=quatB_rt.device)
            rep_loss_rf = torch.tensor(0.0, device=quatB_rt.device)
        rep_loss_lh /= bs
        rep_loss_rh /= bs
        rep_loss_tail_hind /= bs
        if compute_att:
            att_loss /= bs
        else:
            att_loss = torch.tensor(0.0, device=quatB_rt.device)

        w_front = arg.w_front if compute_front else 0.0
        w_hind = arg.w_hind
        w_tail_hind = arg.w_tail_hind
        w_att = arg.w_att if compute_att else 0.0

        def freeze_all():
            for para in retarget_net.parameters():
                para.requires_grad = False

        def partial_backward(loss):
            if use_ddp:
                with retarget_net.no_sync():
                    loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

        if compute_front:
            freeze_all()
            for para in module.delta_leftFront_dec.parameters():
                para.requires_grad = True
            partial_backward(w_front * rep_loss_lf)

            freeze_all()
            for para in module.delta_rightFront_dec.parameters():
                para.requires_grad = True
            partial_backward(w_front * rep_loss_rf)

        freeze_all()
        for para in module.delta_leftHind_dec.parameters():
            para.requires_grad = True
        partial_backward(w_hind * rep_loss_lh)

        freeze_all()
        for para in module.delta_rightHind_dec.parameters():
            para.requires_grad = True
        partial_backward(w_hind * rep_loss_rh)

        tail_geo_loss = w_tail_hind * rep_loss_tail_hind
        freeze_all()
        for para in module.delta_tail_dec.parameters():
            para.requires_grad = True
        partial_backward(tail_geo_loss)

        rep_loss_terms = [
            w_hind * (rep_loss_lh + rep_loss_rh),
            w_tail_hind * rep_loss_tail_hind,
        ]
        if compute_front:
            rep_loss_terms.insert(0, w_front * (rep_loss_lf + rep_loss_rf))
        if compute_att:
            rep_loss_terms.append(w_att * att_loss)
        rep_loss = arg.kappa * sum(rep_loss_terms)
        smooth_weighted = arg.omega_smooth * smooth_loss
        ret_loss = arg.lam * gen_loss + base_loss + smooth_weighted

        freeze_all()
        for para in module.weights_dec.parameters():
            para.requires_grad = True
        partial_backward(rep_loss)

        for para in retarget_net.parameters():
            para.requires_grad = True
        ret_loss.backward()
        nn.utils.clip_grad_norm_(retarget_net.parameters(), max_norm=25)
        optimizer_ret.step()

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        epoch_loss_ret.update(float(ret_loss.item()))
        epoch_loss_rep.update(float(rep_loss.item()))
        if compute_front:
            epoch_rep_lf.update(float(rep_loss_lf.item()))
            epoch_rep_rf.update(float(rep_loss_rf.item()))
        epoch_rep_lh.update(float(rep_loss_lh.item()))
        epoch_rep_rh.update(float(rep_loss_rh.item()))
        epoch_rep_tail_hind.update(float(rep_loss_tail_hind.item()))
        epoch_smooth.update(float(smooth_loss.item()))
        epoch_smooth_weighted.update(float(smooth_weighted.item()))
        if compute_att:
            epoch_att.update(float(att_loss.item()))

        pbar.set_postfix(
            loss_r=float(ret_loss.item()),
            loss_sp=float(rep_loss.item()),
            tail_hind=float(rep_loss_tail_hind.item()),
            lh=float(rep_loss_lh.item()),
            smooth=float(smooth_loss.item()),
            time=end_time - start_time,
        )
        pbar.update(1)

    scheduler.step()
    pbar.close()

    meters = {
        "loss_ret": epoch_loss_ret,
        "loss_rep": epoch_loss_rep,
        "rep_lh": epoch_rep_lh,
        "rep_rh": epoch_rep_rh,
        "rep_tail_hind": epoch_rep_tail_hind,
        "smooth": epoch_smooth,
        "smooth_weighted": epoch_smooth_weighted,
        "epoch_time": epoch_time,
    }
    compute_front = arg.enable_front_rdf and not arg.disable_front_rdf
    compute_att = arg.enable_att_loss and not arg.disable_att_loss
    if compute_front:
        meters["rep_lf"] = epoch_rep_lf
        meters["rep_rf"] = epoch_rep_rf
    if compute_att:
        meters["att"] = epoch_att

    # All ranks must participate in collective reduction (DDP deadlock otherwise).
    reduced = reduce_epoch_meters(arg, device, meters)

    if is_main_process(arg) and logger is not None:
        logger.add_scalar("train_loss_ret", reduced["loss_ret"], epoch)
        logger.add_scalar("train_loss_rep", reduced["loss_rep"], epoch)
        logger.add_scalar("train_rep_lh", reduced["rep_lh"], epoch)
        logger.add_scalar("train_rep_rh", reduced["rep_rh"], epoch)
        logger.add_scalar("train_rep_tail_hind", reduced["rep_tail_hind"], epoch)
        logger.add_scalar("train_smooth", reduced["smooth"], epoch)
        logger.add_scalar("train_smooth_weighted", reduced["smooth_weighted"], epoch)
        if compute_front:
            logger.add_scalar("train_rep_lf", reduced["rep_lf"], epoch)
            logger.add_scalar("train_rep_rf", reduced["rep_rf"], epoch)
        if compute_att:
            logger.add_scalar("train_att", reduced["att"], epoch)

    return reduced


def main(arg):
    if is_main_process(arg) and not exists(arg.work_dir):
        makedirs(arg.work_dir)

    if arg.distributed:
        dist.barrier()

    device = arg.torch_device
    world_size = arg.world_size
    global_batch = arg.batch_size * world_size

    print_log_txt("Initializing SMAL33 shape-aware training", arg.work_dir, arg)
    print_log_txt(
        f"GPUs: physical={arg.physical_device_ids} | DDP={'on' if world_size > 1 else 'off'} "
        f"| world_size={world_size} | batch_per_gpu={arg.batch_size} "
        f"| global_batch={global_batch}",
        arg.work_dir,
        arg,
    )
    compute_front = arg.enable_front_rdf and not arg.disable_front_rdf
    print_log_txt(
        f"Geo speed: sdf_grid={arg.sdf_grid_size} frame_stride={arg.geo_frame_stride} "
        f"att_loss={'on' if arg.enable_att_loss and not arg.disable_att_loss else 'off'} "
        f"front_rdf={'on' if compute_front else 'off'} | hull_precompute=on",
        arg.work_dir,
        arg,
    )
    print_log_txt(
        f"RDF weights front={arg.w_front if compute_front else 0}(off) "
        f"hind={arg.w_hind} tail_hind={arg.w_tail_hind} "
        f"omega_smooth={arg.omega_smooth}",
        arg.work_dir,
        arg,
    )

    data_feeder = Feeder(**arg.train_feeder_args)
    retarget_net = RetNet(**arg.ret_model_args).to(device)
    discriminator_net = MotionDis(**arg.dis_model_args).to(device)

    load_model(retarget_net, discriminator_net, arg, device)

    if world_size > 1:
        retarget_net = DDP(
            retarget_net,
            device_ids=[arg.local_rank],
            output_device=arg.local_rank,
            find_unused_parameters=True,
        )

    train_sampler = None
    loader_kwargs = {
        "batch_size": arg.batch_size,
        "num_workers": arg.num_workers,
        "pin_memory": True,
    }
    if world_size > 1:
        train_sampler = DistributedSampler(
            data_feeder,
            num_replicas=world_size,
            rank=arg.rank,
            shuffle=True,
            drop_last=False,
        )
        loader_kwargs["shuffle"] = False
        loader_kwargs["sampler"] = train_sampler
    else:
        loader_kwargs["shuffle"] = True

    if arg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    data_loader = torch.utils.data.DataLoader(dataset=data_feeder, **loader_kwargs)

    module = unwrap_module(retarget_net)
    opt_para_lst = [
        module.delta_leftFront_dec.parameters(),
        module.delta_rightFront_dec.parameters(),
        module.delta_leftHind_dec.parameters(),
        module.delta_rightHind_dec.parameters(),
        module.delta_tail_dec.parameters(),
        module.weights_dec.parameters(),
    ]
    params = []
    for pl in opt_para_lst:
        for p in pl:
            params.append(p)
    print_log_txt(f"Optimize params: {len(params)}", arg.work_dir, arg)

    optimizer_ret = optim.Adam(
        params, lr=arg.base_lr, weight_decay=arg.weight_decay, betas=(0.5, 0.999)
    )
    scheduler_ret = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ret, milestones=arg.step, gamma=0.1, last_epoch=-1
    )

    train_writer = None
    if is_main_process(arg):
        train_writer = SummaryWriter(
            join(arg.work_dir, arg.model_save_name, "train_log"), "train"
        )

    mesh_paths = getattr(arg, "mesh_paths", None) or [arg.mesh_path]
    mesh_file_dic = load_mesh_file_dic(mesh_paths, arg.work_dir, arg)
    validate_mesh_keys(mesh_file_dic, data_feeder.shape_dic.keys(), arg.work_dir, arg)

    mesh_groups = build_mesh_vertex_groups(mesh_file_dic, data_feeder)
    mesh_geom_cache = build_mesh_geometry_cache(mesh_file_dic, mesh_groups)
    cross_prob = getattr(data_feeder, "cross_external_target_prob", None)
    target_pool_size = len(getattr(data_feeder, "target_pool", []) or [])
    print_log_txt(
        f"Loaded {len(mesh_file_dic)} shape meshes from {len(mesh_paths)} path(s) "
        f"(hull precomputed) | paw bones={PAW_JOINT_INDICES} | "
        f"sequences={len(data_feeder)} | target_pool={target_pool_size} | "
        f"cross_external_target_prob={cross_prob}",
        arg.work_dir,
        arg,
    )

    if is_main_process(arg):
        arg_dict = vars(arg).copy()
        for k in (
            "torch_device",
            "rank",
            "world_size",
            "local_rank",
            "distributed",
            "is_distributed_worker",
        ):
            arg_dict.pop(k, None)
        with open(join(arg.work_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(arg_dict, f)

    if arg.distributed:
        dist.barrier()

    for i in range(arg.epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(i)

        epoch_stats = train(
            retarget_net,
            discriminator_net,
            data_loader,
            optimizer_ret,
            scheduler_ret,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
            mesh_file_dic,
            mesh_groups,
            mesh_geom_cache,
            i,
            train_writer,
            arg,
            device,
        )

        lr = optimizer_ret.param_groups[0]["lr"]
        log_txt = (
            f"epoch:{i + 1}  ret loss:{epoch_stats['loss_ret']:.6f}  "
            f"sfpen loss:{epoch_stats['loss_rep']:.6f}  "
            f"rep_lh:{epoch_stats['rep_lh']:.4f}  "
            f"rep_rh:{epoch_stats['rep_rh']:.4f}  "
            f"rep_tail_hind:{epoch_stats['rep_tail_hind']:.4f}  "
            f"smooth:{epoch_stats['smooth']:.6f}  "
            f"smooth_w:{epoch_stats['smooth_weighted']:.6f}  "
            f"epoch time:{epoch_stats['epoch_time']:.4f}  lr:{lr:.6g}"
        )
        print_log_txt(log_txt, arg.work_dir, arg)

        if (i + 1) % 5 == 0:
            if is_main_process(arg):
                save_checkpoint(retarget_net, arg, i + 1)
            if world_size > 1:
                dist.barrier()

    print_log_txt("Training finished.", arg.work_dir, arg)
    if train_writer is not None:
        train_writer.close()

    if world_size > 1:
        dist.destroy_process_group()


def parse_args():
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
    return parser.parse_args()


if __name__ == "__main__":
    arg = parse_args()
    physical_devices = arg.device if isinstance(arg.device, list) else [arg.device]

    if len(physical_devices) > 1 and os.environ.get("LOCAL_RANK") is None:
        relaunch_with_torchrun(sys.argv[1:], physical_devices, arg.master_port)
        sys.exit(0)

    device, rank, world_size, local_rank, distributed, physical_device_ids = (
        init_distributed_env(physical_devices)
    )
    arg.torch_device = device
    arg.rank = rank
    arg.world_size = world_size
    arg.local_rank = local_rank
    arg.distributed = distributed
    arg.physical_device_ids = physical_device_ids
    init_seed(arg.seed + rank)
    main(arg)
