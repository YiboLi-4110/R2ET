"""
SMAL33 shape-aware retargeting model.

Changes vs src/model_shape_aware.py:
  - 33-joint quadruped topology (see datasets/bvh_smal33.py)
  - delta_dec uses identity-residual quaternion (aligned with stage-1 SMAL33 fix)
  - Five shape decoders: left/right front, left/right hind, tail
  - Balancing gate (WeightsDecoder) unchanged in role, sized for 33 joints
  - RDF / ADF geometry losses extended for tail vertices and paw end-effectors
  - Symmetric RDF thresholds; weighted rep losses; tail-vs-hind cross RDF
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import trimesh
from sdf import SDF, SDF2
from torch import asin, atan2

from src.forward_kinematics import FK
from src.linear_blend_skin import linear_blend_skinning
from src.ops import qlinear, q_mul_q

NUM_JOINTS = 33

# Shape-decoder joint groups (upper chains; paws/hocks corrected via FK + ADF).
LEFT_FRONT_JOINTS = [7, 8, 9]       # LeftScapula, LeftUpperArm, LeftForeLeg
RIGHT_FRONT_JOINTS = [11, 12, 13]   # RightScapula, RightUpperArm, RightForeLeg
LEFT_HIND_JOINTS = [18, 19]         # LeftThigh, LeftShin
RIGHT_HIND_JOINTS = [22, 23]        # RightThigh, RightShin
TAIL_JOINTS = [26, 27, 28]          # Tail1, Tail2, Tail3

# Paw / end-effector joint indices for ADF (Attractive Distance Field).
PAW_JOINT_INDICES = [10, 14, 21, 25]  # LeftFrontPaw, RightFrontPaw, LeftHindPaw, RightHindPaw

# Symmetric RDF activation thresholds (ifth=True: penalize when phi_val > threshold).
RDF_THRESHOLD_FRONT = 4.0
RDF_THRESHOLD_HIND = 4.0
RDF_THRESHOLD_TAIL_HIND = 4.0    # tail-into-hind legs: primary tail penetration mode
DEFAULT_SDF_GRID_SIZE = 24       # 32 is slower; 24 is a good speed/quality tradeoff
DEFAULT_GEO_FRAME_STRIDE = 2     # evaluate geometry loss every N frames


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0, :, :, :, :], qkv[1, :, :, :, :], qkv[2, :, :, :, :]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Residual(
                                Attention(dim, mlp_dim, heads=heads, dropout=dropout)
                            ),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=33):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                f"odd dim (got dim={dim})"
            )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, 0 : emb.size(1), :]
        emb = self.dropout(emb)
        return emb


class QuatEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, hidden_channels, kp):
        super().__init__()
        self.num_joint = num_joint
        self.token_linear = nn.Linear(4, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp)

    def forward(self, pose_t):
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)
        return embed_q


class SeklEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super().__init__()
        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, skel):
        token_s = self.token_linear(skel)
        embed_s = self.trans1(token_s)
        return embed_s


class DeltaDecoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, hidden_channels, kp):
        super().__init__()
        self.num_joint = num_joint
        self.q_encoder = QuatEncoder(num_joint, token_channels, hidden_channels, kp)
        self.skel_encoder = SeklEncoder(num_joint, token_channels, embed_channels, kp)
        self.pos_encoder = PositionalEncoding(
            1 - kp, hidden_channels + (2 * embed_channels), max_len=num_joint
        )
        self.embed_linear = nn.Linear(
            hidden_channels + (2 * embed_channels), embed_channels
        )
        self.embed_acti = nn.ReLU()
        self.embed_drop = nn.Dropout(1 - kp)
        self.delta_linear = nn.Linear(embed_channels, 4)

    def forward(self, q_t, skelA, skelB):
        q_embed = self.q_encoder(q_t)
        skelA_embed = self.skel_encoder(skelA)
        skelB_embed = self.skel_encoder(skelB)
        cat_embed = torch.cat([q_embed, skelA_embed, skelB_embed], dim=-1)
        pos_embed = self.pos_encoder(cat_embed)
        embed = self.embed_drop(self.embed_acti(self.embed_linear(pos_embed)))
        deltaq_t = self.delta_linear(embed)
        return deltaq_t


class DeltaShapeDecoder(nn.Module):
    def __init__(self, num_limb_joint, skeleton_num_joint, hidden_channels, kp):
        super().__init__()
        self.num_limb_joint = num_limb_joint
        self.skeleton_num_joint = skeleton_num_joint

        self.joint_linear1 = nn.Linear(10 * skeleton_num_joint, hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)
        self.delta_linear = qlinear(hidden_channels, 4 * num_limb_joint)

    def forward(self, shapeA, shapeB, x):
        bs = shapeB.shape[0]
        x_cat = torch.cat([shapeA, shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))
        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        deltaq_t = self.delta_linear(x_embed)
        return deltaq_t


class WeightsDecoder(nn.Module):
    """Balancing gate Fw: per-joint blend between skeleton and shape outputs."""

    def __init__(self, num_joint, hidden_channels, kp):
        super().__init__()
        self.num_joint = num_joint
        self.joint_linear1 = nn.Linear(10 * num_joint, 2 * hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(2 * hidden_channels, 2 * hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)
        self.joint_linear3 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.joint_acti3 = nn.ReLU()
        self.joint_drop3 = nn.Dropout(p=1 - kp)
        self.weights_linear = nn.Linear(hidden_channels, num_joint)
        self.weights_acti = nn.Sigmoid()

    def forward(self, refB, shapeB, x):
        bs = refB.shape[0]
        x_cat = torch.cat([refB, shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))
        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        x_embed = self.joint_drop3(self.joint_acti3(self.joint_linear3(x_embed)))
        weights = self.weights_acti(self.weights_linear(x_embed))
        return weights


class MotionDis(nn.Module):
    def __init__(self, kp):
        super().__init__()
        pad = int((4 - 1) / 2)
        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(p=1 - kp)),
                    ("h0", nn.Conv1d(3, 16, kernel_size=4, padding=pad, stride=2)),
                    ("acti0", nn.LeakyReLU(0.2)),
                    ("h1", nn.Conv1d(16, 32, kernel_size=4, padding=pad, stride=2)),
                    ("bn1", nn.BatchNorm1d(32)),
                    ("acti1", nn.LeakyReLU(0.2)),
                    ("h2", nn.Conv1d(32, 64, kernel_size=4, padding=pad, stride=2)),
                    ("bn2", nn.BatchNorm1d(64)),
                    ("acti2", nn.LeakyReLU(0.2)),
                    ("h3", nn.Conv1d(64, 64, kernel_size=4, padding=pad, stride=2)),
                    ("bn3", nn.BatchNorm1d(64)),
                    ("acti3", nn.LeakyReLU(0.2)),
                    ("h4", nn.Conv1d(64, 1, kernel_size=2, stride=2)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        bs = x.size(0)
        y = self.seq(x)
        return y.view(bs, 1)


def normalized(angles):
    lengths = torch.sqrt(torch.sum(torch.square(angles), dim=-1))
    lengths = torch.clamp(lengths, min=1e-8)
    return angles / lengths[..., None]


class RetNet(nn.Module):
    def __init__(
        self,
        num_joint=33,
        token_channels=64,
        hidden_channels_p=256,
        embed_channels_p=128,
        kp=0.8,
    ):
        super().__init__()
        self.num_joint = num_joint
        self.delta_dec = DeltaDecoder(
            num_joint, token_channels, embed_channels_p, hidden_channels_p, kp
        )
        self.delta_leftFront_dec = DeltaShapeDecoder(
            3, num_joint, hidden_channels_p, kp
        )
        self.delta_rightFront_dec = DeltaShapeDecoder(
            3, num_joint, hidden_channels_p, kp
        )
        self.delta_leftHind_dec = DeltaShapeDecoder(2, num_joint, hidden_channels_p, kp)
        self.delta_rightHind_dec = DeltaShapeDecoder(
            2, num_joint, hidden_channels_p, kp
        )
        self.delta_tail_dec = DeltaShapeDecoder(3, num_joint, hidden_channels_p, kp)
        self.weights_dec = WeightsDecoder(num_joint, hidden_channels_p, kp)

    def forward(
        self,
        seqA,
        seqB,
        skelA,
        skelB,
        shapeA,
        shapeB,
        quatA,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        parents,
        k=-1,
        phase="train",
    ):
        self.parents = parents
        bs, T = seqA.size(0), seqA.size(1)

        local_mean = torch.from_numpy(local_mean).float().cuda(seqA.device)
        local_std = torch.from_numpy(local_std).float().cuda(seqA.device)
        quat_mean = torch.from_numpy(quat_mean).float().cuda(seqA.device)
        quat_std = torch.from_numpy(quat_std).float().cuda(seqA.device)
        parents = torch.from_numpy(parents).cuda(seqA.device)

        t_poseB = torch.reshape(skelB[:, 0, :], [bs, self.num_joint, 3])
        t_poseB = t_poseB * local_std + local_mean
        refB = t_poseB
        refB_feed = skelB[:, 0, :]
        refA_feed = skelA[:, 0, :]
        shapeB = shapeB.view((bs, self.num_joint, 3))
        shapeA = shapeA.view((bs, self.num_joint, 3))

        quatA_denorm = quatA * quat_std[None, :] + quat_mean[None, :]

        delta_qs = []
        B_locals_rt = []
        B_quats_rt = []
        B_quats_base = []
        B_locals_base = []
        B_gates = []
        delta_qg = []

        if k == -1:
            k = 1.0

        left_front_joints = LEFT_FRONT_JOINTS
        right_front_joints = RIGHT_FRONT_JOINTS
        left_hind_joints = LEFT_HIND_JOINTS
        right_hind_joints = RIGHT_HIND_JOINTS
        tail_joints = TAIL_JOINTS

        for t in range(T):
            qoutA_t = quatA[:, t, :, :]
            qoutA_t_denorm = quatA_denorm[:, t, :, :]

            refA_feed = refA_feed.view((bs, self.num_joint, 3))
            refB_feed = refB_feed.view((bs, self.num_joint, 3))

            delta1 = self.delta_dec(qoutA_t, refA_feed, refB_feed)
            delta1[..., 0] = delta1[..., 0] + 1.0
            delta1 = normalized(delta1)
            delta_qs.append(delta1)

            qB_base = q_mul_q(qoutA_t_denorm, delta1)
            qB_base = qB_base.detach()
            qB_base_norm = (qB_base - quat_mean) / quat_std

            delta2_lf = self.delta_leftFront_dec(shapeA, shapeB, qB_base_norm)
            delta2_lf = torch.reshape(delta2_lf, [bs, 3, 4])
            delta2_lf = (
                delta2_lf * quat_std[:, left_front_joints, :]
                + quat_mean[:, left_front_joints, :]
            )
            delta2_lf = normalized(delta2_lf)

            delta2_rf = self.delta_rightFront_dec(shapeA, shapeB, qB_base_norm)
            delta2_rf = torch.reshape(delta2_rf, [bs, 3, 4])
            delta2_rf = (
                delta2_rf * quat_std[:, right_front_joints, :]
                + quat_mean[:, right_front_joints, :]
            )
            delta2_rf = normalized(delta2_rf)

            delta2_lh = self.delta_leftHind_dec(shapeA, shapeB, qB_base_norm)
            delta2_lh = torch.reshape(delta2_lh, [bs, 2, 4])
            delta2_lh = (
                delta2_lh * quat_std[:, left_hind_joints, :]
                + quat_mean[:, left_hind_joints, :]
            )
            delta2_lh = normalized(delta2_lh)

            delta2_rh = self.delta_rightHind_dec(shapeA, shapeB, qB_base_norm)
            delta2_rh = torch.reshape(delta2_rh, [bs, 2, 4])
            delta2_rh = (
                delta2_rh * quat_std[:, right_hind_joints, :]
                + quat_mean[:, right_hind_joints, :]
            )
            delta2_rh = normalized(delta2_rh)

            delta2_tail = self.delta_tail_dec(shapeA, shapeB, qB_base_norm)
            delta2_tail = torch.reshape(delta2_tail, [bs, 3, 4])
            delta2_tail = (
                delta2_tail * quat_std[:, tail_joints, :]
                + quat_mean[:, tail_joints, :]
            )
            delta2_tail = normalized(delta2_tail)

            delta2 = (
                torch.tensor([1, 0, 0, 0], dtype=torch.float32)
                .cuda(seqA.device)
                .repeat(bs, self.num_joint, 1)
            )
            delta2[:, left_front_joints, :] = delta2_lf
            delta2[:, right_front_joints, :] = delta2_rf
            delta2[:, left_hind_joints, :] = delta2_lh
            delta2[:, right_hind_joints, :] = delta2_rh
            delta2[:, tail_joints, :] = delta2_tail
            delta_qg.append(delta2)

            bala_gate = self.weights_dec(refB_feed, shapeB, qB_base_norm)
            qB_hat = q_mul_q(qB_base, delta2)

            if phase == "train":
                one_w = np.random.binomial(1, p=0.4)
                if one_w:
                    bala_gate = torch.ones(bala_gate.shape, dtype=torch.float32).cuda(
                        seqA.device
                    )

            qB_t = torch.lerp(qB_base, qB_hat, bala_gate[:, :, None] * k)

            B_quats_base.append(qB_base)
            B_quats_rt.append(qB_t)
            B_gates.append(bala_gate)

            localB_t = FK.run(parents, refB, qB_t)
            localB_t = (localB_t - local_mean) / local_std
            B_locals_rt.append(localB_t)

            localB_base_t = FK.run(parents, refB, qB_base)
            localB_base_t = (localB_base_t - local_mean) / local_std
            B_locals_base.append(localB_base_t)

        quatB_rt = torch.stack(B_quats_rt, dim=1)
        delta_qs = torch.stack(delta_qs, dim=1)
        localB_rt = torch.stack(B_locals_rt, dim=1)
        quatB_base = torch.stack(B_quats_base, dim=1)
        bala_gates = torch.stack(B_gates, dim=1)
        delta_qg = torch.stack(delta_qg, dim=1)
        localB_base = torch.stack(B_locals_base, dim=1)

        globalA_vel = seqA[:, :, -4:-1]
        globalA_rot = seqA[:, :, -1]
        normalized_vin = torch.cat(
            (
                torch.divide(globalA_vel, inp_height[:, :, None]),
                globalA_rot[:, :, None],
            ),
            dim=-1,
        )
        normalized_vout = normalized_vin.clone()
        globalB_vel = normalized_vout[:, :, :-1]
        globalB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat(
            (
                torch.multiply(globalB_vel, tgt_height[:, :, None]),
                globalB_rot[:, :, None],
            ),
            dim=-1,
        )

        if self.training:
            localA_gt = torch.reshape(seqA[:, :, :-4], [bs, T, self.num_joint, 3])
            localB_gt = torch.reshape(seqB[:, :, :-4], [bs, T, self.num_joint, 3])
            globalA_gt = seqA[:, :, -4:]
            return (
                localA_gt,
                localB_rt,
                localB_gt,
                globalA_gt,
                globalB_rt,
                quatB_rt,
                quatB_base,
                localB_base,
                bala_gates,
            )

        return localB_rt, globalB_rt, quatB_rt, delta_qs, delta_qg

    @staticmethod
    def get_cons_loss(
        atte_lst, num_joint, mask, localB_rt, localB_base, quatB_rt, quatB_base
    ):
        attW = torch.ones(num_joint).cuda(localB_rt.device)
        attW[atte_lst] = 2

        ae_joints_err = torch.sum(
            (
                torch.multiply(
                    mask[:, :, None, None], torch.subtract(localB_rt, localB_base)
                )
            )
            ** 2,
            dim=[0, 1, 3],
        )
        local_ae_loss = torch.sum(attW * ae_joints_err)
        local_ae_loss = torch.divide(
            local_ae_loss,
            torch.maximum(torch.sum(mask), torch.tensor(1).cuda(localB_base.device)),
        )

        quat_ae_loss = torch.sum(
            (
                torch.multiply(
                    mask[:, :, None, None], torch.subtract(quatB_rt, quatB_base)
                )
            )
            ** 2,
            dim=[0, 1, 3],
        )
        quat_ae_loss = torch.sum(attW * quat_ae_loss)
        quat_ae_loss = torch.divide(
            quat_ae_loss,
            torch.maximum(torch.sum(mask), torch.tensor(1).cuda(localB_base.device)),
        )
        return local_ae_loss, quat_ae_loss

    @staticmethod
    def get_rot_cons_loss(alpha, euler_ord, quatB_rt):
        rads = alpha / 180.0
        twistB_loss = torch.mean(
            torch.square(
                torch.maximum(
                    torch.tensor(0).cuda(quatB_rt.device),
                    torch.abs(euler_y(quatB_rt, euler_ord)) - rads * np.pi,
                )
            )
        )
        return twistB_loss

    @staticmethod
    def get_gen_loss(fake_score, ae_reg):
        bceloss = nn.BCELoss(reduction="none")
        gen_motion_loss = bceloss(
            fake_score, torch.ones(fake_score.shape).cuda(fake_score.device)
        )
        gen_motion_loss = torch.sum(torch.multiply((1 - ae_reg), gen_motion_loss))
        gen_motion_loss = torch.divide(
            gen_motion_loss,
            torch.maximum(torch.sum(1 - ae_reg), torch.tensor(1).cuda(ae_reg.device)),
        )
        return gen_motion_loss

    @staticmethod
    def get_regularization_loss(weights_sp, mask):
        bs = weights_sp.shape[0]
        regular_weights_loss = torch.sum(
            torch.multiply(mask[:, :, None], weights_sp**2)
        ) / (2 * bs)
        return regular_weights_loss

    @staticmethod
    def get_smooth_loss(localB_rt_denorm, mask):
        """
        Temporal smoothness penalty on denormalized local joint trajectories.

        This mirrors the skeleton-aware stage and uses second-order finite
        differences, so it damps frame-wise jitter without penalizing constant
        velocity motion. Only valid, unpadded frame triplets contribute.
        """
        if localB_rt_denorm.shape[1] < 3:
            return torch.tensor(0.0, device=localB_rt_denorm.device)
        accel = (
            localB_rt_denorm[:, 2:]
            - 2 * localB_rt_denorm[:, 1:-1]
            + localB_rt_denorm[:, :-2]
        )
        valid = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        accel_sq = torch.mean(torch.square(accel), dim=[2, 3])
        smooth_loss = torch.sum(accel_sq * valid)
        denom = torch.maximum(
            torch.sum(valid),
            torch.tensor(1.0, device=valid.device),
        )
        return torch.divide(smooth_loss, denom)

    @staticmethod
    def _build_sdf_from_hull(
        vertices_centered_scaled,
        hull_spec,
        device,
        grid_size=DEFAULT_SDF_GRID_SIZE,
    ):
        """SDF from cached rest-pose hull topology + current deformed vertex positions."""
        if hull_spec is None:
            return None
        get_sdf = SDF()
        idx = hull_spec["vertex_indices"]
        faces_np = hull_spec["faces"]
        hull_verts = vertices_centered_scaled[0, idx, :]
        faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device)
        return get_sdf(faces_t, hull_verts[None,], grid_size=grid_size)

    @staticmethod
    def _build_body_head_sdf(
        vertices_centered_scaled,
        body_vertices_lst,
        head_vertices_lst,
        device,
        grid_size=DEFAULT_SDF_GRID_SIZE,
        torso_hull=None,
        head_hull=None,
    ):
        get_sdf = SDF()
        if torso_hull is not None:
            body_sdf = RetNet._build_sdf_from_hull(
                vertices_centered_scaled, torso_hull, device, grid_size=grid_size
            )
        else:
            body_vertices = (
                vertices_centered_scaled[0, body_vertices_lst, :]
                .detach()
                .cpu()
                .numpy()
            )
            body_point_cloud = trimesh.points.PointCloud(vertices=body_vertices)
            body_mesh = body_point_cloud.convex_hull
            body_vertices_t = torch.from_numpy(
                np.array(body_mesh.vertices).astype(np.single)
            ).to(device)
            body_faces_t = torch.from_numpy(
                np.array(body_mesh.faces).astype(np.int32)
            ).to(device)
            body_sdf = get_sdf(
                body_faces_t, body_vertices_t[None,], grid_size=grid_size
            )

        if head_hull is not None:
            head_sdf = RetNet._build_sdf_from_hull(
                vertices_centered_scaled, head_hull, device, grid_size=grid_size
            )
        else:
            head_vertices = (
                vertices_centered_scaled[0, head_vertices_lst, :]
                .detach()
                .cpu()
                .numpy()
            )
            head_point_cloud = trimesh.points.PointCloud(vertices=head_vertices)
            head_mesh = head_point_cloud.convex_hull
            head_vertices_t = torch.from_numpy(
                np.array(head_mesh.vertices).astype(np.single)
            ).to(device)
            head_faces_t = torch.from_numpy(
                np.array(head_mesh.faces).astype(np.int32)
            ).to(device)
            head_sdf = get_sdf(
                head_faces_t, head_vertices_t[None,], grid_size=grid_size
            )
        return body_sdf + head_sdf

    @staticmethod
    def _build_vertices_sdf(
        vertices_centered_scaled,
        vertices_lst,
        device,
        grid_size=DEFAULT_SDF_GRID_SIZE,
        hull_spec=None,
    ):
        """Convex-hull SDF from an arbitrary vertex subset."""
        if hull_spec is not None:
            return RetNet._build_sdf_from_hull(
                vertices_centered_scaled, hull_spec, device, grid_size=grid_size
            )
        if not vertices_lst:
            return None
        verts_np = (
            vertices_centered_scaled[0, vertices_lst, :].detach().cpu().numpy()
        )
        if verts_np.shape[0] < 4:
            return None
        get_sdf = SDF()
        point_cloud = trimesh.points.PointCloud(vertices=verts_np)
        mesh = point_cloud.convex_hull
        mesh_vertices = torch.from_numpy(
            np.array(mesh.vertices).astype(np.single)
        ).to(device)
        mesh_faces = torch.from_numpy(
            np.array(mesh.faces).astype(np.int32)
        ).to(device)
        return get_sdf(mesh_faces, mesh_vertices[None,], grid_size=grid_size)

    @staticmethod
    def _rep_loss_on_vertices(
        total_sdf, vertices_local, threshold, device, ifth=True
    ):
        vert_num = vertices_local.shape[0]
        if vert_num == 0:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True).cuda(
                device
            )
        vertices_grid = vertices_local.view(1, -1, 1, 1, 3)
        phi_val = (
            nn.functional.grid_sample(
                total_sdf[0][None, None], vertices_grid, align_corners=False
            )
            .view(-1)
            .sum()
            / vert_num
        ) * 1000
        if ifth and phi_val <= threshold:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True).cuda(
                device
            )
        return phi_val

    @staticmethod
    def get_rep_loss_part(
        parents,
        quatB_rt,
        rest_skelB,
        meshB,
        skinB_weights,
        body_vertices_lst,
        head_vertices_lst,
        left_front_vertices_lst,
        right_front_vertices_lst,
        left_hind_vertices_lst,
        right_hind_vertices_lst,
        tail_vertices_lst,
        ifth=False,
        sdf_grid_size=DEFAULT_SDF_GRID_SIZE,
        frame_stride=DEFAULT_GEO_FRAME_STRIDE,
        hull_cache=None,
        compute_front_rdf=True,
    ):
        vertices_lbs = linear_blend_skinning(
            parents, quatB_rt, rest_skelB, meshB, skinB_weights
        )
        torso_hull = head_hull = hind_hull = None
        if hull_cache is not None:
            hulls = hull_cache.get("hulls", {})
            torso_hull = hulls.get("torso")
            head_hull = hulls.get("head")
            hind_hull = hulls.get("hind")

        scale_factor = 0.2
        boxes = get_bounding_boxes(vertices_lbs)
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (
            (1 + scale_factor)
            * 0.5
            * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None]
        )
        vertices_centered = vertices_lbs - boxes_center
        vertices_centered_scaled = vertices_centered / boxes_scale

        T = vertices_lbs.shape[0]
        frame_stride = max(1, int(frame_stride))
        frame_indices = list(range(0, T, frame_stride))
        n_frames = max(len(frame_indices), 1)

        rep_loss_lf, rep_loss_rf, rep_loss_lh, rep_loss_rh = 0, 0, 0, 0
        rep_loss_tail_hind = 0

        hind_vertices_lst = list(left_hind_vertices_lst) + list(
            right_hind_vertices_lst
        )

        for i in frame_indices:
            frame_verts = vertices_centered_scaled[i : i + 1]
            device = quatB_rt.device

            # Torso + head obstacle: front/hind legs vs body (incl. front vs head/neck).
            total_sdf = RetNet._build_body_head_sdf(
                frame_verts,
                body_vertices_lst,
                head_vertices_lst,
                device,
                grid_size=sdf_grid_size,
                torso_hull=torso_hull,
                head_hull=head_hull,
            )

            if compute_front_rdf:
                rep_loss_lf += RetNet._rep_loss_on_vertices(
                    total_sdf,
                    vertices_centered_scaled[i, left_front_vertices_lst, :],
                    RDF_THRESHOLD_FRONT,
                    device,
                    ifth,
                )
                rep_loss_rf += RetNet._rep_loss_on_vertices(
                    total_sdf,
                    vertices_centered_scaled[i, right_front_vertices_lst, :],
                    RDF_THRESHOLD_FRONT,
                    device,
                    ifth,
                )
            rep_loss_lh += RetNet._rep_loss_on_vertices(
                total_sdf,
                vertices_centered_scaled[i, left_hind_vertices_lst, :],
                RDF_THRESHOLD_HIND,
                device,
                ifth,
            )
            rep_loss_rh += RetNet._rep_loss_on_vertices(
                total_sdf,
                vertices_centered_scaled[i, right_hind_vertices_lst, :],
                RDF_THRESHOLD_HIND,
                device,
                ifth,
            )

            # Hind-legs obstacle: tail vs hind (primary tail penetration mode).
            hind_sdf = RetNet._build_vertices_sdf(
                frame_verts,
                hind_vertices_lst,
                device,
                grid_size=sdf_grid_size,
                hull_spec=hind_hull,
            )
            if hind_sdf is not None:
                rep_loss_tail_hind += RetNet._rep_loss_on_vertices(
                    hind_sdf,
                    vertices_centered_scaled[i, tail_vertices_lst, :],
                    RDF_THRESHOLD_TAIL_HIND,
                    device,
                    ifth,
                )

        return (
            rep_loss_lf / n_frames,
            rep_loss_rf / n_frames,
            rep_loss_lh / n_frames,
            rep_loss_rh / n_frames,
            rep_loss_tail_hind / n_frames,
        )

    @staticmethod
    def get_rep_eval_stats(
        parents,
        quatB_rt,
        rest_skelB,
        meshB,
        skinB_weights,
        body_vertices_lst,
        head_vertices_lst,
        left_front_vertices_lst,
        right_front_vertices_lst,
        left_hind_vertices_lst,
        right_hind_vertices_lst,
        tail_vertices_lst,
        sdf_grid_size=DEFAULT_SDF_GRID_SIZE,
        frame_stride=DEFAULT_GEO_FRAME_STRIDE,
        hull_cache=None,
        compute_front_rdf=True,
    ):
        """
        Evaluation helper: RDF magnitudes (ifth=False) and per-part penetration
        frame rates (fraction of sampled frames with phi > threshold).
        """
        vertices_lbs = linear_blend_skinning(
            parents, quatB_rt, rest_skelB, meshB, skinB_weights
        )
        torso_hull = head_hull = hind_hull = None
        if hull_cache is not None:
            hulls = hull_cache.get("hulls", {})
            torso_hull = hulls.get("torso")
            head_hull = hulls.get("head")
            hind_hull = hulls.get("hind")

        scale_factor = 0.2
        boxes = get_bounding_boxes(vertices_lbs)
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (
            (1 + scale_factor)
            * 0.5
            * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None]
        )
        vertices_centered = vertices_lbs - boxes_center
        vertices_centered_scaled = vertices_centered / boxes_scale

        T = vertices_lbs.shape[0]
        frame_stride = max(1, int(frame_stride))
        frame_indices = list(range(0, T, frame_stride))
        n_frames = max(len(frame_indices), 1)

        rep_sum = {
            "rep_lf": 0.0,
            "rep_rf": 0.0,
            "rep_lh": 0.0,
            "rep_rh": 0.0,
            "rep_tail_hind": 0.0,
        }
        pen_counts = {k: 0 for k in rep_sum}
        hind_vertices_lst = list(left_hind_vertices_lst) + list(
            right_hind_vertices_lst
        )

        def _phi_scalar(total_sdf, verts, threshold, device):
            vert_num = verts.shape[0]
            if vert_num == 0:
                return 0.0
            vertices_grid = verts.view(1, -1, 1, 1, 3)
            phi_val = (
                nn.functional.grid_sample(
                    total_sdf[0][None, None], vertices_grid, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            return float(phi_val.item())

        for i in frame_indices:
            frame_verts = vertices_centered_scaled[i : i + 1]
            device = quatB_rt.device

            total_sdf = RetNet._build_body_head_sdf(
                frame_verts,
                body_vertices_lst,
                head_vertices_lst,
                device,
                grid_size=sdf_grid_size,
                torso_hull=torso_hull,
                head_hull=head_hull,
            )

            if compute_front_rdf:
                phi_lf = _phi_scalar(
                    total_sdf,
                    vertices_centered_scaled[i, left_front_vertices_lst, :],
                    RDF_THRESHOLD_FRONT,
                    device,
                )
                phi_rf = _phi_scalar(
                    total_sdf,
                    vertices_centered_scaled[i, right_front_vertices_lst, :],
                    RDF_THRESHOLD_FRONT,
                    device,
                )
                rep_sum["rep_lf"] += phi_lf
                rep_sum["rep_rf"] += phi_rf
                if phi_lf > RDF_THRESHOLD_FRONT:
                    pen_counts["rep_lf"] += 1
                if phi_rf > RDF_THRESHOLD_FRONT:
                    pen_counts["rep_rf"] += 1

            phi_lh = _phi_scalar(
                total_sdf,
                vertices_centered_scaled[i, left_hind_vertices_lst, :],
                RDF_THRESHOLD_HIND,
                device,
            )
            phi_rh = _phi_scalar(
                total_sdf,
                vertices_centered_scaled[i, right_hind_vertices_lst, :],
                RDF_THRESHOLD_HIND,
                device,
            )
            rep_sum["rep_lh"] += phi_lh
            rep_sum["rep_rh"] += phi_rh
            if phi_lh > RDF_THRESHOLD_HIND:
                pen_counts["rep_lh"] += 1
            if phi_rh > RDF_THRESHOLD_HIND:
                pen_counts["rep_rh"] += 1

            hind_sdf = RetNet._build_vertices_sdf(
                frame_verts,
                hind_vertices_lst,
                device,
                grid_size=sdf_grid_size,
                hull_spec=hind_hull,
            )
            if hind_sdf is not None:
                phi_tail = _phi_scalar(
                    hind_sdf,
                    vertices_centered_scaled[i, tail_vertices_lst, :],
                    RDF_THRESHOLD_TAIL_HIND,
                    device,
                )
                rep_sum["rep_tail_hind"] += phi_tail
                if phi_tail > RDF_THRESHOLD_TAIL_HIND:
                    pen_counts["rep_tail_hind"] += 1

        stats = {k: v / n_frames for k, v in rep_sum.items()}
        stats["pen_rate_lf"] = pen_counts["rep_lf"] / n_frames
        stats["pen_rate_rf"] = pen_counts["rep_rf"] / n_frames
        stats["pen_rate_lh"] = pen_counts["rep_lh"] / n_frames
        stats["pen_rate_rh"] = pen_counts["rep_rh"] / n_frames
        stats["pen_rate_tail_hind"] = pen_counts["rep_tail_hind"] / n_frames
        stats["geo_frames_evaluated"] = n_frames
        return stats

    @staticmethod
    def get_att_loss(
        parents,
        quatB_rt,
        rest_skelB,
        meshB,
        skinB_weights,
        body_vertices_lst,
        paw_vertices_lst,
        sdf_grid_size=DEFAULT_SDF_GRID_SIZE,
        frame_stride=DEFAULT_GEO_FRAME_STRIDE,
        hull_cache=None,
    ):
        get_sdf_out = SDF2()
        vertices_lbs = linear_blend_skinning(
            parents, quatB_rt, rest_skelB, meshB, skinB_weights
        )
        torso_hull = None
        if hull_cache is not None:
            torso_hull = hull_cache.get("hulls", {}).get("torso_adf")

        scale_factor = 0.2
        boxes = get_bounding_boxes(vertices_lbs)
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (
            (1 + scale_factor)
            * 0.5
            * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None]
        )
        vertices_centered = vertices_lbs - boxes_center
        vertices_centered_scaled = vertices_centered / boxes_scale

        T = vertices_lbs.shape[0]
        frame_stride = max(1, int(frame_stride))
        frame_indices = list(range(0, T, frame_stride))
        n_frames = max(len(frame_indices), 1)
        att_loss = 0

        for i in frame_indices:
            frame_verts = vertices_centered_scaled[i : i + 1]
            if torso_hull is not None:
                body_sdf_out = RetNet._build_sdf_from_hull(
                    frame_verts,
                    torso_hull,
                    quatB_rt.device,
                    grid_size=sdf_grid_size,
                )
            else:
                body_vertices = (
                    vertices_centered_scaled[i, body_vertices_lst, :]
                    .detach()
                    .cpu()
                    .numpy()
                )
                body_point_cloud = trimesh.points.PointCloud(vertices=body_vertices)
                body_mesh = body_point_cloud.convex_hull
                body_vertices_t = torch.from_numpy(
                    np.array(body_mesh.vertices).astype(np.single)
                ).to(quatB_rt.device)
                body_faces_t = torch.from_numpy(
                    np.array(body_mesh.faces).astype(np.int32)
                ).to(quatB_rt.device)
                body_sdf_out = get_sdf_out(
                    body_faces_t, body_vertices_t[None,], grid_size=sdf_grid_size
                )

            vertices_local = vertices_centered_scaled[i, paw_vertices_lst, :]
            vert_num = vertices_local.shape[0]
            if vert_num == 0:
                continue
            vertices_grid = vertices_local.view(1, -1, 1, 1, 3)
            phi_val = (
                nn.functional.grid_sample(
                    body_sdf_out[0][None, None], vertices_grid, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            att_loss += phi_val

        return att_loss / n_frames


def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes


def euler_y(angles, order="yzx"):
    q = normalized(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    if order == "xyz":
        ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        ey = asin(torch.clamp(2 * (q0 * q2 - q3 * q1), -1, 1))
        ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        return torch.stack(values=[ex, ez], dim=-1)[:, :, 1:]
    if order == "yzx":
        ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        ez = asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))
        return ey[:, :, 1:]
    raise Exception("Unknown Euler order!")
