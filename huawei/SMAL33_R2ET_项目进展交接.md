# SMAL33 宠物 R2ET 迁移 — 项目进展交接文档

> **用途**：供新对话快速恢复上下文。  
> **基线仓库**：CVPR 2023 R2ET（Mixamo 人体，22 关节，`max_length=60`）。  
> **当前目标**：Planet Zoo SMAL33 四足宠物 motion retarget（33 关节，`max_length=32`）。  
> **最后更新**：2026-06（第一轮 shape 训练与评估已完成；第二轮训练已启动；外部数据对接方案已梳理）。

---

## 1. 一句话状态

| 阶段 | 状态 |
|------|------|
| 数据预处理（FBX→BVH→NPY、Shape NPZ、Stats） | ✅ 完成并验证 |
| Stage-1 Skeleton-aware 训练 | ✅ 完成（checkpoint 如 `ret-96` / `ret-180`） |
| Stage-1 检验（数值 + 可视化） | ✅ 有 `eval_skeleton_aware_smal33.py`、`inspect_stage1_outputs_smal33.py` |
| Stage-2 Shape-aware 训练（第一轮） | ✅ 30 epoch，4 卡 DDP，约 ~20 min/epoch |
| Stage-2 检验 Phase A（数值） | ✅ `eval_shape_aware_smal33.py` → `eval_summary.json` |
| Stage-2 检验 Phase B（可视化） | ✅ `inspect_shape_aware_smal33.py`（骨架 + 蒙皮 mesh） |
| Stage-2 检验 Phase C（批量 BVH 推理） | ⏳ 未单独封装（可用 inspect + `retarget_to_bvh` 组合） |
| 外部数据（shepherd → batch2_dogs） | 📋 流程已明确，待对方 SMAL33 资产验证 |
| 第二轮 shape 训练 | 🔄 进行中（恢复 att/更高 sdf 等「高质量」设置） |

---

## 2. 与原版 R2ET 的核心差异（必读）

| 维度 | 原版 R2ET (Mixamo) | 本项目 SMAL33 |
|------|-------------------|---------------|
| 关节数 | 22 | **33** |
| 骨架语义 | 人体：body/head/leftarm/rightarm/leftleg/rightleg | **四足**：torso/head/left&right front/hind/**tail** |
| 训练窗口 `max_length` | 60 | **32** |
| 最短样本 | FBX→BVH 常补到 60 帧 | **保留真实帧长**；训练 **`min_frames=32`**，丢弃更短样本 |
| Stats 文件前缀 | `mixamo_*` | **`smal33_*`** |
| Shape 几何 RDF 部位 | 4 肢 + torso/head | **前肢(可选) + 后肢 + tail→hind**；**已删除 tail→torso RDF** |
| Shape delta 解码器 | leftArm/rightArm/leftLeg/rightLeg | **leftFront/rightFront/leftHind/rightHind/tail** |
| ATT（爪-躯干）损失 | 人体脚 | **四足 paw 顶点**（`mesh_groups["paws"]`） |
| 高度计算 | 人体骨长启发式 | **`get_height_from_skel`**：spine1–6 + tail 段 |
| Skeleton 四元数 delta | Mixamo 惯例 | SMAL33：**残差绕 identity**，避免 quat_mean/std 对前链的固定偏置 |
| 推理输入 | Stage-1：`seqB=None` | Stage-2：**需要 `seqB`**（目标 motion 特征） |
| 数据路径 | `datasets/mixamo/` | `datasets/Planet_Zoo_FBX-smal2/` |

### 2.1 为何用 32 帧而非 60

Planet Zoo 宠物动作大量短 clip（`<60` 帧约占 **66%**）。若强行补到 60 帧会污染 root velocity 统计并引入假静止尾。  
采用 **`min_frames=32`** 仅丢弃约 **27.6%** 样本，是数据量与分布干净度的折中。详见 `datasets/SMAL33_TRAINING_HANDOFF.md`。

### 2.2 Shape 阶段几何设计要点

- **Tail RDF**：`tail` 顶点相对 **hind（后肢 hull）** 的 SDF，权重 **`w_tail_hind=3.5`**（最高档之一）。
- **Hind RDF**：后肢相对 **torso hull**，`w_hind=3.0`。
- **Front RDF**：默认 **关闭**（`--disable_front_rdf` / `enable_front_rdf: false`），因前肢穿模率低。
- **ATT loss**：爪部顶点相对 torso ADF，可通过 `--disable_att_loss` 关闭以提速。
- **训练时 RDF hinge**：`RDF_THRESHOLD_* = 4.0`，`ifth=True` 时低于阈值无梯度（后腿易平台，第二轮可考虑降到 3.5 或 `ifth=False`）。
- **Hull 预计算**：`src/mesh_geometry_cache.py`，避免每帧 `trimesh.convex_hull`。
- **SDF backward**：需在 `outside-code/sdf` 安装处修复 `backward(ctx, grad_output)`（用户 site-packages 已改）。

---

## 3. 数据目录与预处理流程

### 3.1 目录结构

```
datasets/Planet_Zoo_FBX-smal2/
├── train_char/          # 每角色子目录，*.fbx → *.bvh
├── train_q/             # 预处理 motion：*_seq.npy, *_skel.npy, *_quat.npy
├── train_shape/         # 每角色 *.npz（mesh + shape 向量）
└── stats/               # smal33_*_mean/std.npy 等
```

**外部对接（进行中）**：

```
datasets/Planet_Zoo_FBX-smal2/
├── smal@shepherd/       # 源动作 FBX + blend
└── batch2_dogs_gt/      # 目标犬 blend（rest）
```

### 3.2 预处理脚本链

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 帧数统计 | `datasets/check_fbx_frame_stats.py` | 论证不用 60 帧补齐 |
| FBX→BVH | `datasets/fbx2bvh_smal33.py` | **保留真实帧长**；Blender 并行 |
| BVH→NPY | `datasets/preprocess_q_smal33.py` | quat/skel/seq |
| 检查 | `datasets/check_smal33_preprocess.py` | FK 重建误差等 |
| Shape NPZ | `datasets/extract_shape_smal33.py` | rest mesh、skin、joint_shape |
| Shape 检查 | `datasets/check_extract_shape_smal33.py` | 33 骨命名必须齐全 |
| Stats | `datasets/compute_stats_smal33.py` | 或 feeder 首次加载时缓存 |

### 3.3 Shape NPZ 关键字段

`joint_shape`, `full_width`, `body_width`, `rest_vertices`, `rest_faces`, `skinning_weights`, `vertex_part`, `skeleton`, `joint_names`（SMAL33 标准 33 名）。

### 3.4 共享 I/O 模块

**`datasets/smal33_motion_io.py`**：`get_inp_from_bvh`, `load_stats`, `load_shape_vector`, `load_retnet`, `load_shape_retnet`, `retarget_to_bvh`, `world_joints_from_motion`, `split_characters`（val 10%, seed **3047**）。

---

## 4. Stage-1：Skeleton-aware

### 4.1 代码与配置

| 文件 | 作用 |
|------|------|
| `src/model_skeleton_aware_smal33.py` | 33 关节 RetNet（仅 delta skeleton） |
| `train_skeleton_aware_smal33.py` | 训练 |
| `config/train_skeleton_aware_smal33.yaml` | `max_length=32`, `min_frames=32`, epoch 180 |
| `datasets/train_feeder_r2et_smal33.py` | 数据加载、stats、shape 向量 |
| `inference_bvh_smal33.py` | BVH 推理导出（Stage-1 权重） |
| `eval_skeleton_aware_smal33.py` | val split 数值评估 |
| `inspect_stage1_outputs_smal33.py` | 世界空间骨架可视化诊断 |

### 4.2 训练要点

- Work dir 示例：`work_dir/train_skeleton_aware_smal33/`
- Checkpoint 命名：`r2et_skeleton_aware_smal33_ret-{epoch}.pt` + `dis-*.pt`
- 第一轮 shape 训练 init 用过 **`ret-96` / `dis-96`**；yaml 默认 **`ret-180`**

### 4.3 检验

- **数值**：`config/eval_skeleton_aware_smal33.yaml`
- **可视化**：`inspect_stage1_outputs_smal33.py` — 绿=输入，红=输出，可选蓝=target rest；`compare_spacing` / `view_mode` 可调

---

## 5. Stage-2：Shape-aware

### 5.1 代码与配置

| 文件 | 作用 |
|------|------|
| `src/model_shape_aware_smal33.py` | Shape RetNet + RDF/ATT + `get_rep_eval_stats` |
| `train_shape_aware_smal33.py` | **DDP 多卡**训练 |
| `config/train_shape_aware_smal33.yaml` | 默认 sdf=24, stride=2, att=on |
| `train_shape.sh` | 启动示例（多卡 batch 32/GPU） |
| `src/mesh_geometry_cache.py` | Rest-pose hull 预计算 |

### 5.2 第一轮实际训练设置（`log.txt` 记录）

```
4×GPU DDP, batch 32/GPU, global 128
init: ret-96 / dis-96
--disable_att_loss --disable_front_rdf
--sdf_grid_size 20 --geo_frame_stride 4
30 epochs, ~20 min/epoch
work_dir: work_dir/train_shapeaware_smal33/
```

训练曲线摘要：`ret loss` 43→8.4；`rep_rh`~2.9、`rep_tail_hind`~0.6–0.7 平台；几何项 ep8 后增幅变小。

### 5.3 第二轮训练（已启动，建议配置）

在恢复 **`att_loss=on`、`sdf=24`、`stride=2`** 基础上，建议：

- init **`ret-180`** 或从 **`ret-12.pt` 微调**
- `step: [12, 22]`，`kappa: 0.65`，`w_hind: 3.5`，`w_tail_hind: 4.0`
- 代码侧：`RDF_THRESHOLD_HIND: 3.5`（可选）

---

## 6. Stage-2 评估与可视化

### 6.1 Phase A：数值评估

**脚本**：`eval_shape_aware_smal33.py`  
**配置**：`config/eval_shape_aware_smal33.yaml`  
**输出**：`work_dir/train_shapeaware_smal33/eval/eval_summary.json`

**设计**：

- Val split：10% 角色，seed 3047（与训练 feeder 一致）
- **Baseline**：必须用 **skeleton** `load_retnet` + `forward_skeleton`，不能把 stage-1 权重载入 shape 模型当 baseline
- **Stage-2**：`load_shape_retnet` + `forward_shape`
- 指标：skeleton（sem, twist, jerk…）+ **cross geometry**（`rep_lh/rh/tail_hind`, `pen_rate_*`, `geo_priority_score`）
- `delta_vs_baseline` 负值表示优于 stage-1

**第一轮结果摘要**（50 cross pairs）：

| 指标 | Baseline (skel ret-96) | Stage-2 最佳 (~ret-12) |
|------|------------------------|-------------------------|
| `cross.sem` | 0.0198 | ~0.019（基本保持） |
| `rep_tail_hind` | 2.21 | **~1.68**（↓约 24%） |
| `pen_rate_tail_hind` | 10.75% | **~7.5–8.25%** |
| `pen_rate_rh` | 42.25% | **~41.25%**（几乎平台） |
| `geo_priority_score` | 9.38 | **8.78**（推荐 ckpt） |

**推荐 checkpoint**：`r2et_shape_aware_smal33_ret-12.pt`（`geo_priority_score` 最低）。  
**注意**：`delta rep_rh > 0` 在部分 epoch 出现，幅度小（~+0.02），属统计噪声；**右后腿是主要瓶颈**。

**已修复 bug**：`compute_geometry_metrics` 中 `t_pose_b` 须为 `(33,3)`，用 `batch_skel_to_tpose(...)[0]`；`parents` 传 numpy。

### 6.2 Phase B：可视化检验

**脚本**：`inspect_shape_aware_smal33.py`  
**配置**：`config/inspect_shape_aware_smal33_cfg.yaml`

**输出**（每对 case 一个子目录）：

- `skeleton_compare.mp4`：绿=源，红=stage-2；可选 `--show_target_rest` 蓝=目标 rest
- `mesh_compare.mp4`：绿=源蒙皮，红=stage-2@目标 mesh；**默认不含 stage-1**（`--show_stage1_mesh` 可选）
- `predictions.npz`, `summary.json`, 静帧 png

**相机参数**：`camera_zoom`, `view_elev`, `view_azim`, `frame_margin`, `mesh_spacing`

**已修复问题**：

1. Cross 时 mesh faces 须分 `inp_faces` / `tgt_faces`（源/目标顶点数不同）
2. `Poly3DCollection` 无 `shade` 参数 → 自实现 face 光照
3. 视频仅 ~1s：因 `num_frames=min(..., len(tgt_bvh))` 被**短目标 BVH**截断 → **改为以源 BVH 长度为准**（`len(inp_motion["quat"])`）

### 6.3 Phase C（待做）

**`inference_bvh_shape_aware_smal33.py`**：批量 BVH 导出；shape 模型需 **seqA + seqB**，不同于 skeleton 版 `seqB=None`。

---

## 7. 外部数据使用流程（shepherd → batch2_dogs）

**结论**：可用现有权重，**前提是 SMAL33 骨骼命名一致**。

| 需要 | 来源 |
|------|------|
| 权重 + stats | 本地已有 |
| 源/目标 shape `.npz` | `extract_shape_smal33.py` ← rest FBX |
| 源动作 BVH | `fbx2bvh_smal33.py` ← 源 FBX |
| 目标 rest BVH | 目标 rest FBX→BVH（仅需 t-pose） |
| `.blend` | Blender 导出 FBX 后再走上述流程 |

**必须向对方确认**：batch2_dogs 是否为 **SMAL33 拓扑**（`JOINT_NAME_SMAL_33` 全套骨名）。若不是，需重绑资产，非脚本可解决。

---

## 8. 工程化改动备忘

| 项 | 说明 |
|----|------|
| DDP | `train_shape_aware_smal33.py`；`all_reduce` 全 rank 参与，仅 rank0 写 log |
| DataParallel | 已弃用（无效） |
| SDF CUDA | `backward(ctx, grad_output)` 修复 |
| 路径 typo | 训练 log 在 `train_shape_aware_smal33/`，ckpt 在 `train_shapeaware_smal33/`（无下划线），评估时注意 |

---

## 9. 关键文件索引

```
config/
  train_skeleton_aware_smal33.yaml
  train_shape_aware_smal33.yaml
  eval_skeleton_aware_smal33.yaml
  eval_shape_aware_smal33.yaml
  inference_bvh_smal33_cfg.yaml
  inspect_shape_aware_smal33_cfg.yaml

datasets/
  smal33_motion_io.py          # 推理/评估共享 I/O
  train_feeder_r2et_smal33.py
  fbx2bvh_smal33.py
  preprocess_q_smal33.py
  extract_shape_smal33.py
  SMAL33_TRAINING_HANDOFF.md   # 数据阶段详细说明

src/
  model_skeleton_aware_smal33.py
  model_shape_aware_smal33.py
  mesh_geometry_cache.py

train_skeleton_aware_smal33.py
train_shape_aware_smal33.py
train_shape.sh

eval_skeleton_aware_smal33.py
eval_shape_aware_smal33.py

inference_bvh_smal33.py
inspect_stage1_outputs_smal33.py
inspect_shape_aware_smal33.py

work_dir/
  train_skeleton_aware_smal33/     # stage-1 log 等
  train_shapeaware_smal33/         # stage-2 ckpt + eval/
```

---

## 10. 常用命令速查

```bash
# Stage-1 推理
python inference_bvh_smal33.py --config config/inference_bvh_smal33_cfg.yaml

# Stage-2 数值评估
python eval_shape_aware_smal33.py --config config/eval_shape_aware_smal33.yaml

# Stage-2 可视化
python inspect_shape_aware_smal33.py --config config/inspect_shape_aware_smal33_cfg.yaml

# Stage-2 多卡训练
bash train_shape.sh

# FBX→BVH
blender -b -P datasets/fbx2bvh_smal33.py -- --data_path ./datasets/Planet_Zoo_FBX-smal2/train_char

# Shape 提取
blender -b -P datasets/extract_shape_smal33.py -- \
  --fbx_root ./datasets/Planet_Zoo_FBX-smal2/train_char \
  --save_path ./datasets/Planet_Zoo_FBX-smal2/train_shape
```

---

## 11. 给下一段对话的最短上下文（可直接粘贴）

> 我们在 R2ET 上完成了 SMAL33 四足迁移：33 关节、`max_length=32`、`min_frames=32`、真实帧长 FBX 导出。Stage-1 skeleton 已训完；Stage-2 shape 第一轮 30 epoch（DDP，disable att/front RDF，sdf20/stride4，init ret-96）已完成 eval + inspect。相对 baseline，**tail/hind 几何明显改善**，**右后腿 pen_rate 仍 ~41% 平台**；推荐 ckpt **ret-12**。评估 baseline 必须用 skeleton forward；inspect 已支持源+stage2 蒙皮可视化，帧长以源 BVH 为准。第二轮 shape 训练已开（恢复 att、更高 sdf）。外部 shepherd→batch2_dogs 需 SMAL33 绑定 FBX/blend→现有脚本。待做：Phase C 批量 shape 推理脚本。

---

## 12. 参考文档

- 数据阶段细节：`datasets/SMAL33_TRAINING_HANDOFF.md`
- 原版论文：Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry (CVPR 2023)
