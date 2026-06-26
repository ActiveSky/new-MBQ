# MBQ 每日规划记录

本文档用于记录每天讨论出的实验规划、关键判断、风险约束和下一步行动。后续更新时建议按日期追加，不要随意覆盖已有记录，这样之后复盘实验结果时，可以追溯每一步的假设、动机和决策依据。

## 2026-06-09

### 主题

InternVL2-8B 的 2-bit MBQ 量化中，尝试通过自定义敏感层保护来寻找更优的成本和性能折中点。

当前方向是从单一的全局 `keep_ratio` 控制，转向显式选择敏感 linear 层并将这些层提升到更高 bit。目标不是单纯降低 PPL，而是在 2-bit 主目标下，把存储成本作为核心约束，找到质量收益和额外存储之间更合理的甜点区。

### 当前证据

已有的 `keep_ratio` sweep 表明，增加 3-bit 保护层的数量会改善 PPL，但不同区间的边际收益并不均匀。

目前已知的 Wikitext2 PPL 结果如下，对应 `w2g32_scale_reweight_true_svd_1.0_mixed_*` 系列 cache：

| keep_ratio | 3-bit linears | 2-bit linears | low-rank count | PPL |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 160 | 160 | 29.7054 |
| 0.1 | 16 | 144 | 144 | 27.5484 |
| 0.2 | 32 | 128 | 128 | 26.1439 |
| 0.3 | 48 | 112 | 112 | 23.9118 |
| 0.4 | 64 | 96 | 96 | 22.6093 |
| 0.5 | 80 | 80 | 80 | 20.6337 |
| 0.6 | 96 | 64 | 64 | 19.6061 |
| 0.7 | 112 | 48 | 48 | 18.9858 |

目前观察到的主要边际收益如下：

| 区间 | 新增保护层构成 | PPL 下降 |
| --- | --- | ---: |
| 0.2 -> 0.3 | 10 个 `feed_forward.w2`，4 个 `feed_forward.w3`，2 个 `attention.wo` | 2.2321 |
| 0.4 -> 0.5 | 10 个 `feed_forward.w2`，3 个 `attention.wo`，2 个 `feed_forward.w3`，1 个 `feed_forward.w1` | 1.9756 |
| 0.1 -> 0.2 | 8 个 `attention.wo`，8 个 `feed_forward.w3` | 1.4045 |

这些结果说明，全局 score 排序是有参考价值的，但 module 类型对收益和成本都有重要影响。当前证据支持以下粗略敏感性判断：

1. `attention.wo` 很敏感，并且相对便宜，值得优先保护。
2. `feed_forward.w2` 对 PPL 的改善非常关键，但它是较大的 MLP projection，提升 bit 的成本较高。
3. `feed_forward.w3` 也有持续敏感性，但同样成本较高，需要谨慎选择。
4. `feed_forward.w1` 在排序中出现较晚，不应优先于更敏感的 `wo`、`w2`、`w3`。
5. `attention.wqkv` 进入排序较晚，目前看对 PPL 的边际收益较弱，不应在存储敏感方案中优先保护。

### 当前代码行为确认

现在的评估路径支持自定义修改 `linear_bit_map`，不需要大幅修改主流程。

PPL 脚本会加载 scale cache：

```python
quant_state = torch.load(scale_path, map_location="cpu")
```

当启用 `--mixed_probe` 时，会读取：

```python
linear_bit_map = quant_state.get("linear_bit_map", {})
```

在 pseudo quantization 阶段，每个 linear 层会优先使用 `linear_bit_map` 中指定的 bit：

```python
linear_w_bit = int(override_bit_map.get(full_name, w_bit))
```

因此，只要编辑保存好的 cache 中的 `linear_bit_map`，就可以改变哪些层用 3-bit、哪些层继续用 2-bit 进行 PPL 评估。

### 关键约束

`linear_bit_map` 和 `low_rank` 必须保持一致。

当前 scale cache 是在如下设置下生成的：

```yaml
exclusive_with_low_rank: True
```

在这个语义下，被提升到更高 bit 的层应该从 `low_rank` 残差列表中排除。如果某一层在 `linear_bit_map` 中被设置为 3-bit，同时它仍然保留在 `low_rank` 里，那么当前代码会对同一层施加两种补偿机制：

```text
3-bit base weight + low-rank residual
```

这种配置会比当前 `keep_ratio` 实验更强，也更贵。它得到的 PPL 可能更好看，但不符合我们想比较的机制，也无法公平反映真实存储成本。

因此，自定义 cache 的目标语义应该是：

```text
敏感层: 3-bit base weight，不保留 low-rank residual
非敏感层: 2-bit base weight，保留 low-rank residual
```

### 母版 cache 选择

建议使用以下 cache 作为自定义实验的母版：

```text
scale_cache/mbq/internvl2_8b_w2g32_scale_reweight_true_svd_1.0_mixed_0.0.pt
```

理由：

1. 它的 `linear_bit_map` 中包含 160 个候选 linear，初始都设为 2-bit。
2. 它的 `low_rank` 中包含 160 个候选 linear 的 SVD 残差。
3. 它保存了完整的 `linear_score_map`，后续可以根据已有 score 元数据生成自定义 mask。
4. 它避免了从 `mixed_0.2` 或 `mixed_0.3` 往回改的问题。非零 `keep_ratio` 的 cache 中，有些已经被提升到 3-bit 的层在生成时已经从 `low_rank` 删除，后续如果再把这些层降回 2-bit，代码不会自动补回对应的 SVD 残差。

不要使用已有的非零 `keep_ratio` cache 作为“降层回 2-bit”的母版，因为对应的 low-rank residual 可能已经缺失。

### 自定义 cache 工作流程

对每一种自定义策略，建议按以下步骤进行：

1. 使用 `mbq` 环境加载 `mixed_0.0.pt`。
2. 读取 `linear_score_map`，根据策略构造敏感层集合。
3. 将敏感层在 `linear_bit_map` 中设置为 `3`。
4. 将未选中的层在 `linear_bit_map` 中保持为 `2`。
5. 过滤 `low_rank`，确保所有 3-bit 敏感层都不再出现在 low-rank 列表里。
6. 更新 `linear_mixed_config` 元数据，记录自定义策略名称、选中层数量和选择规则。
7. 保存成新的 cache 文件，文件名必须清晰描述策略，避免覆盖已有 sweep cache。
8. 使用现有 `tests/test_PPL/2_compute_quant_ppl.sh` 流程评估 PPL。
9. 在本文档中记录 PPL、3-bit 层数量、删除的 low-rank 数量、估算存储变化和完整选层列表。

自定义 cache 文件名应使用明确命名，例如：

```text
internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_wo_core.pt
internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_wo_w2_top32.pt
internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_costaware_40.pt
```

### 候选敏感层策略

下一批实验不应只依赖全局 score top-k，而应比较多个结构化 mask。

#### 策略 A：`wo_core`

只保护最敏感的 `attention.wo` 层。

目的：

评估只保护较便宜的 attention output projection 能恢复多少质量。

初始候选：

```text
attention.wo: L7-L17, L20-L22, L24-L26
```

这个策略的吸引力在于：`attention.wo` 参数量低于 MLP projection，并且在 score 排名中很早出现，可能具有较好的质量收益和存储成本比。

#### 策略 B：`wo_core_plus_w2_early_mid`

保护核心 `attention.wo`，再加入最先出现的高价值 `feed_forward.w2` 层。

目的：

测试是否可以用少于完整 top-48 的保护层数量，吃到 `0.2 -> 0.3` 中最明显的 PPL 收益。

初始候选：

```text
attention.wo: 策略 A 的 core set
feed_forward.w2: L3, L5-L13
```

这个策略直接针对 `0.2 -> 0.3` 区间中表现最突出的 `mlp_out` 敏感组。

#### 策略 C：`wo_w2_plus_w3_selective`

保护 `attention.wo`、关键 `feed_forward.w2`，再选择少量 `feed_forward.w3`。

目的：

评估少量 `w3` 保护是否能带来足够收益，从而证明其额外存储成本是值得的。

初始候选：

```text
attention.wo: 策略 A 的 core set
feed_forward.w2: L3, L5-L13
feed_forward.w3: L2-L9，预算允许时可考虑 L21-L27
```

这个策略应该在多个预算下测试，因为 `w3` 虽然敏感，但属于较大的 MLP projection，成本不能忽略。

#### 策略 D：`top_score_fixed_budget`

使用原始全局 score 排序，并固定选层预算，例如 24、32、40、48 个保护层。

目的：

为结构化、module-aware 策略提供对照基线。

如果自定义启发式策略不能优于或接近简单 global top-k，那么它就不值得保留。

#### 策略 E：`cost_aware_score`

使用带成本归一化的选择目标，例如：

```text
sensitivity_score / estimated_extra_storage_cost
```

目的：

优先选择单位存储成本下敏感度更高的层。

这种策略可能会选择更多 `attention.wo`，并减少大 MLP 层数量。不过需要用实际 PPL 验证，因为它可能低估 `feed_forward.w2` 对质量的关键作用。

### 存储成本估算说明

对每一个被选为 3-bit 的层，base weight 相比 2-bit 会额外增加约 1 bit per parameter。

但由于被选中的 3-bit 层应该从 `low_rank` 中删除，因此同时也会节省该层的 SVD residual 存储。比较自定义策略时，应该关注：

```text
净存储变化 ~= base weight 额外 bit 成本 - 删除的 low-rank 存储
```

这个估算只有在最终部署打包格式一致时才是精确的，但在搜索阶段仍然可以用于比较不同 mask 的相对成本。

已有观察例子：

1. `0.2 -> 0.3` 新增 16 个保护层，其中包含 10 个 `w2`。
   - bit-packed 情况下，base weight 额外成本估计约 102 MiB。
   - 删除的 low-rank cache tensor 约 67 MiB。
   - 净增加估计约 35 MiB。

2. `0.1 -> 0.2` 新增 8 个 `wo` 和 8 个 `w3`。
   - bit-packed 情况下，base weight 额外成本估计约 72 MiB。
   - 删除的 low-rank cache tensor 约 52 MiB。
   - 净增加估计约 20 MiB。

这些数字说明，不能只看 3-bit 层数量。不同 module 类型的参数量不同，选择同样数量的保护层可能带来完全不同的存储成本。

### 风险与检查项

在信任任何自定义 cache 的 PPL 结果之前，必须检查以下内容：

1. 所有选中的 3-bit 层都已经从 `low_rank` 中删除。
2. 每个候选 linear 都在 `linear_bit_map` 中有条目。
3. 输出 JSON 中的 `linear_bit_counts` 和预期一致。
4. 输出 JSON 中的 `low_rank_count` 和预期一致。
5. 输出 JSON 中记录的 `scale_path` 指向正确的自定义 cache。
6. 每个自定义策略的完整选层列表需要保存或记录，保证可复现。
7. 每个结果都需要和 `mixed_0.0` 以及最接近的 global `keep_ratio` baseline 对比。

重要 caveat：

当前自定义 mask 实验不会为每个 mask 重新进行 scale search。这一点在快速搜索阶段可以接受，因为已有 `keep_ratio` sweep 也是复用保存好的 scale 信息。等找到有希望的自定义策略后，如果要作为主要结果使用，应考虑将最终 mask 固化到生成流程中，再进行更严格的验证。

### 实验执行规范

为了尽快搜索到好的甜点区，后续应尽量多跑自定义 mask 实验。但每次运行前必须先确认 GPU 资源、环境变量和运行环境，避免因为显存不足、网络波动或导入路径问题造成无效实验。

#### GPU 选择

每次启动 PPL 或 scale-cache 生成实验前，先查看 GPU 剩余显存：

```bash
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
```

选择原则：

1. 优先选择 `memory.free` 最高的 GPU。
2. 如果多个 GPU 剩余显存接近，优先选择 `utilization.gpu` 更低的 GPU。
3. 不要默认沿用脚本中的 `GPU_ID=5`，除非确认该卡当前最空。
4. 开始实验前不要依赖脚本默认值，优先在启动命令中显式传入 `GPU_ID=<选择的GPU编号>`。
5. 同一时间尽量不要在同一张卡上叠加多个 8B PPL 实验，避免中途 OOM 导致结果不可用。

#### 环境变量和运行环境

当前机器和仓库环境已经确认安全，后续实验可以直接使用固定的 Hugging Face token 和 mirror 配置。每次运行前统一设置以下环境变量：

```bash
export PYTHONPATH=.
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=YOUR_HF_TOKEN_HERE
```

注意：

1. 当前环境已由使用者明确确认安全，因此本文档中直接记录可用的 HF token，便于后续快速启动实验；在本机、本仓库的实验中可以直接使用，不需要每次再确认。
2. 如果之后需要迁移到共享仓库、公开仓库或多人可见环境，应先移除或替换该 token。
3. 如果需要长期使用 token，也可以在当前 shell、`~/.bashrc`、`~/.zshrc` 或安全的私有环境配置中设置。
4. `PYTHONPATH=.` 用于避免从仓库根目录运行脚本时出现本地包导入错误。
5. `HF_ENDPOINT=https://hf-mirror.com` 用于降低 Hugging Face 网络波动对实验启动的影响。
6. `tests/test_PPL/2_compute_quant_ppl.sh` 中已经设置了这些默认值，但正式实验仍建议在命令行显式 export 一遍，方便从日志中确认运行环境。

后续默认把下面三条命令视为每轮实验的固定启动前缀：

```bash
export PYTHONPATH=.
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export HF_ENDPOINT=https://hf-mirror.com
```

也可以直接使用一行命令完成设置：

```bash
export PYTHONPATH=. && export HF_ENDPOINT=https://hf-mirror.com && export HF_TOKEN=YOUR_HF_TOKEN_HERE
```

所有实验应使用 `mbq` 环境。交互式运行时可以使用：

```bash
source /home/users/Mayinyi/anaconda3/etc/profile.d/conda.sh
conda activate mbq
```

如果从脚本或非交互命令中调用 Python，优先使用明确的解释器路径，避免误用 base 环境：

```bash
/home/users/Mayinyi/anaconda3/envs/mbq/bin/python
```

#### 标准实验启动命令

后续 PPL 实验尽量使用同一套启动格式，减少环境差异带来的噪声。推荐从仓库根目录执行：

```bash
cd /home/users/Mayinyi/jikuixie/new-MBQ
export PYTHONPATH=.
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=YOUR_HF_TOKEN_HERE
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
GPU_ID=<选择的GPU编号> SCALE_FILE_NAME=<目标scale_cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

其中 `GPU_ID` 必须根据 `nvidia-smi` 的实时结果填写，不要机械使用固定编号。`SCALE_FILE_NAME` 只写文件名即可，例如：

```bash
GPU_ID=3 SCALE_FILE_NAME=internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_wo_w2_w3_40.pt bash tests/test_PPL/2_compute_quant_ppl.sh
```

#### 安全环境下的直接启动模板

当前环境已由使用者确认安全，后续在这台机器上可以直接使用下面的模板启动实验。注意用户口头提到的 `export export HF_TOKEN=...` 在实际 shell 中应写成单个 `export`，也就是 `export HF_TOKEN=...`。

启动前先查显存，并把 `GPU_ID` 替换成剩余显存最高、利用率较低的 GPU：

```bash
cd /home/users/Mayinyi/jikuixie/new-MBQ
export PYTHONPATH=.
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export HF_ENDPOINT=https://hf-mirror.com
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
GPU_ID=<显存最空的GPU编号> N_SAMPLES=256 SCALE_FILE_NAME=<目标scale_cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

如果要做更稳的验证，把 `N_SAMPLES` 改成 512：

```bash
GPU_ID=<显存最空的GPU编号> N_SAMPLES=512 SCALE_FILE_NAME=<目标scale_cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

如果 shell 中需要显式进入 conda 环境，使用：

```bash
source /home/users/Mayinyi/anaconda3/etc/profile.d/conda.sh
conda activate mbq
```

如果不想依赖 shell 是否已激活 conda，则直接指定 `mbq` 环境的 Python 路径。当前 `tests/test_PPL/2_compute_quant_ppl.sh` 已默认使用：

```bash
PYTHON_BIN=/home/users/Mayinyi/anaconda3/envs/mbq/bin/python
```

脚本会在日志开头再次打印 GPU 状态、`PYTHONPATH`、`HF_ENDPOINT`、scale cache 路径和输出 JSON 路径。实验结束后先检查这些字段是否正确，再把 PPL 记录进结果表。

如果需要直接调用 Python 工具生成 cache，也使用同样的环境变量和 `mbq` 解释器：

```bash
cd /home/users/Mayinyi/jikuixie/new-MBQ
export PYTHONPATH=.
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=YOUR_HF_TOKEN_HERE
/home/users/Mayinyi/anaconda3/envs/mbq/bin/python tools/build_custom_mbq_cache.py --policy <策略名> --budget <层数> --name <输出名称>
```

#### 批量实验排队原则

为了更快找到甜点区，可以连续跑多组 PPL，但每次启动前仍然要重新查看 GPU。不要假设上一轮空闲的 GPU 在下一轮仍然空闲。

建议把实验按“小批次队列”推进：每次先准备 3 到 5 个候选 cache，逐个检查 GPU、逐个启动 PPL、逐个记录结果。不要一次性生成太多没有验证价值的远端预算点；优先围绕已有 Pareto 候选做密集搜索。

建议按以下顺序推进：

1. 先跑低成本和中等预算的关键对照，例如 `custom_wo_core`、`custom_global_top24`、`custom_wo_w2_24`。
2. 再跑当前最有希望的结构化策略附近的细分预算，例如 `custom_wo_w2_w3_36`、`custom_wo_w2_w3_40`、`custom_wo_w2_w3_44`、必要时扩到 48。
3. 每跑完一个实验，先确认输出 JSON 中的 `scale_path`、`linear_bit_counts`、`low_rank_count` 和预期一致，再把 PPL 纳入比较表。
4. 如果某个策略 PPL 明显弱于同成本或低成本策略，先不要继续在它附近扩展预算，优先把算力给 Pareto 更有希望的策略。
5. 如果某个策略接近或超过 `mixed_0.3`，下一步应该围绕它做删层实验，确认哪些 3-bit 层是真正必要的。
6. 每完成一轮小批次实验后，重新判断下一批预算，不要机械延续原先队列；如果结果显示某个层族收益很弱，应及时把实验资源转向更有希望的层族或 swap/ablation。

实验运行时优先前台执行，方便及时看到 OOM、导入失败或网络异常。如果确实要后台运行，需要记录日志文件路径，并在实验结束后检查日志末尾是否正常输出 PPL。

#### 推荐运行前检查清单

每次实验启动前，按以下顺序检查：

1. 当前目录是否为仓库根目录：`/home/users/Mayinyi/jikuixie/new-MBQ`。
2. 是否已经设置 `PYTHONPATH=.`。
3. 是否已经设置 `HF_ENDPOINT=https://hf-mirror.com`。
4. 是否已经在安全位置设置 HF token。
5. 当前使用的 Python 是否来自 `mbq` 环境。
6. 目标 scale cache 是否存在。
7. 输出文件名是否不会覆盖已有结果。
8. GPU 是否是当前剩余显存较高的卡。
9. 当前实验是否会覆盖已有输出或已有 cache。
10. 自定义 cache 是否从 `mixed_0.0.pt` 生成，而不是从非零 `keep_ratio` cache 反向修改。
11. 自定义 cache 中被提升到 3-bit 的层是否已经从 `low_rank` 中过滤掉。

#### 失败实验记录要求

失败实验也要记录，避免后续重复踩同一个问题。记录时至少包含：

```text
时间:
策略/cache:
GPU_ID:
失败类型: OOM / 网络 / 导入错误 / cache错误 / 输出异常 / 其他
日志路径:
是否产生 PPL JSON:
下一步处理:
```

如果失败原因是 GPU 显存不足，下一次不要只换随机 GPU，而是重新执行显存查询命令后再选择。如果失败原因是网络或导入错误，优先确认 `HF_ENDPOINT`、`HF_TOKEN`、`PYTHONPATH` 和 `mbq` Python 是否生效。

### 甜点区搜索策略

接下来应扩大实验覆盖范围，而不是只跑少数手工策略。目标是用尽量少的额外存储，逼近或超过 global `keep_ratio=0.3` 的质量，同时确认哪些策略在 PPL 和存储估算上位于 Pareto frontier。

建议第一阶段固定几个预算档位：

| 档位 | 目的 |
| --- | --- |
| 16 个 3-bit 层 | 对比 `mixed_0.1`，确认只保护少量关键层时的收益 |
| 24 个 3-bit 层 | 位于 `0.1` 和 `0.2` 中间，测试更细的低成本甜点 |
| 32 个 3-bit 层 | 对比 `mixed_0.2`，判断自定义策略是否优于 global top32 |
| 40 个 3-bit 层 | 位于 `0.2` 和 `0.3` 中间，重点测试是否提前吃到 `w2` 收益 |
| 48 个 3-bit 层 | 对比 `mixed_0.3`，判断同预算下 module-aware 是否更优 |

每个预算档位至少考虑以下策略：

1. `global_topK`：完全按 `linear_score` 排名前 K，作为对照基线。
2. `wo_first_K`：优先选择高分 `attention.wo`，用于评估便宜层的质量收益。
3. `wo_w2_K`：优先选择核心 `attention.wo` 和高敏 `feed_forward.w2`。
4. `wo_w2_w3_K`：在 `wo + w2` 基础上加入少量高分 `feed_forward.w3`。
5. `costaware_K`：按照 `linear_score / estimated_net_cost` 或类似指标选择。

第一批优先生成并评估以下 cache：

```text
custom_global_top24
custom_global_top40
custom_wo_core
custom_wo_w2_24
custom_wo_w2_32
custom_wo_w2_w3_40
custom_costaware_32
custom_costaware_40
```

其中 `custom_global_top24` 和 `custom_global_top40` 很重要，因为已有 sweep 只有 0.1、0.2、0.3 等 0.1 间隔，缺少中间点。它们可以帮助判断 PPL 改善是否平滑，还是在某些层进入后突然改善。

### 实验结果记录格式

每个自定义实验完成后，在本文档追加一行或一小节，至少记录：

```text
策略名称:
cache 文件:
PPL:
NLL:
3-bit 层数量:
2-bit 层数量:
low-rank count:
选中 module 分布:
估算额外 base-weight 存储:
删除 low-rank 存储:
估算净存储变化:
对比 baseline:
备注:
```

如果某个实验失败，也要记录失败原因，例如 OOM、网络下载失败、cache 不存在、环境错误或输出 JSON 异常。失败实验不要和有效 PPL 结果混在一起。

### 已完成自定义实验结果

第一批和第二批自定义 cache 均从 `mixed_0.0.pt` 生成。每个 cache 都完成了一致性检查：`linear_bit_map` 的 bit 数量与 summary 一致，所有 3-bit 层都已经从 `low_rank` 中删除，实际 `low_rank_count` 与 summary 的 `new_low_rank_count` 一致。

下面的净存储估算统一使用同一口径：从 `mixed_0.0.pt` 的完整 `low_rank` 张量估算每个层提升到 3-bit 增加的 base weight 成本，并扣除被删除的 low-rank residual 成本。该数字用于搜索阶段比较，不等价于最终打包格式的精确文件大小。

#### Baseline sweep

| 策略 | 3-bit 层数 | module 分布 | low-rank count | 估算净存储变化 | PPL | 备注 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `mixed_0.0` | 0 | 无 | 160 | 0 MiB | 29.7054 | 全 2-bit + 全 low-rank baseline。 |
| `mixed_0.1` | 16 | `wo` 12, `w3` 4 | 144 | 10.0 MiB | 27.5484 | 全局 score top16。 |
| `mixed_0.2` | 32 | `wo` 20, `w3` 12 | 128 | 30.0 MiB | 26.1439 | 全局 score top32。 |
| `mixed_0.3` | 48 | `wo` 22, `w2` 10, `w3` 16 | 112 | 65.0 MiB | 23.9118 | 当前主要对照点。 |
| `mixed_0.4` | 64 | `wo` 23, `w1` 1, `w2` 14, `w3` 26 | 96 | 102.5 MiB | 22.6093 | 更高成本的强 baseline。 |

#### 自定义策略结果

| 策略 | 3-bit 层数 | module 分布 | low-rank count | 估算净存储变化 | PPL | 观察 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `custom_wo_core` | 17 | `wo` 17 | 143 | 0.0 MiB | 28.2964 | 几乎零净成本即可比 `mixed_0.0` 降 1.4090 PPL。 |
| `custom_global_top24` | 24 | `wo` 17, `w3` 7 | 136 | 17.5 MiB | 27.1086 | 低预算 global 中间点。 |
| `custom_wo_w2_24` | 24 | `wo` 17, `w2` 7 | 136 | 17.5 MiB | 26.7413 | 同成本优于 global top24，说明低预算也应提前纳入 `w2`。 |
| `custom_wo_w2_32` | 32 | `wo` 17, `w2` 15 | 128 | 37.5 MiB | 25.5038 | 同 32 层优于 `mixed_0.2`，再次证明 `w2` 敏感。 |
| `custom_global_top40` | 40 | `wo` 21, `w2` 6, `w3` 13 | 120 | 47.5 MiB | 25.1731 | 比 32 层有所改善，但仍未充分利用 `w2`。 |
| `custom_wo_w2_w3_36` | 36 | `wo` 17, `w2` 11, `w3` 8 | 124 | 47.5 MiB | 25.2790 | 同净存储下略弱于 global top40，但为后续结构化策略提供中间点。 |
| `custom_wo_w2_w3_40` | 40 | `wo` 17, `w2` 15, `w3` 8 | 120 | 57.5 MiB | 24.6909 | 明显优于 global top40。 |
| `custom_wo_w2_w3_41` | 41 | `wo` 17, `w2` 16, `w3` 8 | 119 | 60.0 MiB | 24.5302 | 从 40 层开始逐个补后段 `w2`。 |
| `custom_wo_w2_w3_42` | 42 | `wo` 17, `w2` 17, `w3` 8 | 118 | 62.5 MiB | 24.2660 | 继续补 `w2` 仍有明显收益。 |
| `custom_wo_w2_w3_43` | 43 | `wo` 17, `w2` 18, `w3` 8 | 117 | 65.0 MiB | 24.1117 | 与 `mixed_0.3` 净存储相同，但 PPL 仍高 0.1999。 |
| `custom_wo_w2_w3_44` | 44 | `wo` 17, `w2` 19, `w3` 8 | 116 | 67.5 MiB | 23.9151 | PPL 几乎贴近 `mixed_0.3`，但净存储略高。 |
| `custom_costaware_32` | 32 | `wo` 32 | 128 | 0.0 MiB | 27.2895 | 成本很低，但只保 `wo` 质量不足。 |
| `custom_costaware_40` | 40 | `wo` 32, `wqkv` 8 | 120 | 4.0 MiB | 26.7521 | 说明 `wqkv` 不是当前优先保护对象。 |
| `custom_wo_all_w2_w3_48` | 48 | `wo` 32, `w2` 8, `w3` 8 | 112 | 40.0 MiB | 24.8853 | 全 `wo` 底座在低成本段有效。 |
| `custom_wo_all_w2_w3_51` | 51 | `wo` 32, `w2` 11, `w3` 8 | 109 | 47.5 MiB | 24.5763 | 同净存储下明显优于 global top40 和 `wo_w2_w3_36`。 |
| `custom_wo_all_w2_w3_55` | 55 | `wo` 32, `w2` 15, `w3` 8 | 105 | 57.5 MiB | 24.0250 | 比 `mixed_0.3` 只高 0.1132，但少约 7.5 MiB 净存储。 |
| `custom_wo_all_w2_w3_56` | 56 | `wo` 32, `w2` 16, `w3` 8 | 104 | 60.0 MiB | 23.8751 | 当前最低越过 `mixed_0.3` 的点，比 0.3 低 0.0368 PPL，且少约 5.0 MiB 净存储。 |
| `custom_wo_all_w2_w3_57` | 57 | `wo` 32, `w2` 17, `w3` 8 | 103 | 62.5 MiB | 23.5857 | 比 0.3 低 0.3261 PPL，仍少约 2.5 MiB 净存储。 |
| `custom_wo_all_w2_w3_58` | 58 | `wo` 32, `w2` 18, `w3` 8 | 102 | 65.0 MiB | 23.4276 | 同净存储明显优于 `mixed_0.3`，是当前最重要候选。 |
| `custom_wo_all_w2_w3_59` | 59 | `wo` 32, `w2` 19, `w3` 8 | 101 | 67.5 MiB | 23.2199 | 当前最佳 PPL，自 58 再补一个 `w2` 仍有约 0.2078 PPL 收益。 |

### 当前分析

1. `attention.wo` 应该成为默认保护底座。`wo` 的 3-bit base 增量基本会被删除 low-rank residual 抵消，因此净存储接近 0。全量保护 `wo` 后，同净存储策略明显改善：`custom_wo_all_w2_w3_58` 比 `custom_wo_w2_w3_43` 同为约 65.0 MiB，却低 0.6839 PPL。
2. `feed_forward.w2` 是当前最明确的高敏层。24 层预算下，`wo_w2_24` 比 `global_top24` 低 0.3673 PPL；40 到 44 的逐层实验中，每新增一个后段 `w2` 都降低 PPL。尤其 43 -> 44 加入的 `L22:w2` 带来约 0.1965 PPL 改善，58 -> 59 也再次验证了这个层的价值。
3. `feed_forward.w3` 仍然有价值，但不能只按 global score 让它过早挤掉 `w2`。global top24/top40 都包含较多 `w3`，但同等或接近成本下，提前纳入 `w2` 的结构化策略表现更好。
4. `attention.wqkv` 目前不是优先保护对象。`custom_costaware_40` 增加 8 个 `wqkv`，净存储只增加约 4.0 MiB，但 PPL 仍为 26.7521，明显弱于含 `w2` 的策略。
5. 当前甜点区开始清晰：如果净存储预算约 47.5 MiB，`custom_wo_all_w2_w3_51` 是当前最强；如果预算约 57.5 MiB，`custom_wo_all_w2_w3_55` 很接近 `mixed_0.3` 且更省；如果预算约 60.0 MiB，`custom_wo_all_w2_w3_56` 已经以更低成本略优于 `mixed_0.3`；如果预算约 62.5-65.0 MiB，`custom_wo_all_w2_w3_57/58` 明显支配 `mixed_0.3`。

### 58 层同成本 swap 观察

这一轮实验固定 58 个 3-bit 层，净存储变化均约为 65.0 MiB。母版为 `custom_wo_all_w2_w3_58`，也就是全 `wo`、18 个 `w2`、8 个 `w3`。通过只替换一个层，可以观察候选层在同成本约束下的真实边际价值。

| 策略 | 变更 | module 分布 | PPL | 相对原始 58 | 观察 |
| --- | --- | --- | ---: | ---: | --- |
| `custom_wo_all_w2_w3_58` | 原始 58 层 | `wo` 32, `w2` 18, `w3` 8 | 23.4276 | 0.0000 | 当前 58 层基线。 |
| `custom_58_swap_l16w2_l22w2` | 去掉 `L16:w2`，加入 `L22:w2` | `wo` 32, `w2` 18, `w3` 8 | 23.3534 | -0.0742 | 当前最好的同成本 58 层；`L22:w2` 明显比 `L16:w2` 更值得保护。 |
| `custom_58_swap_l20w2_l22w2` | 去掉 `L20:w2`，加入 `L22:w2` | `wo` 32, `w2` 18, `w3` 8 | 23.3953 | -0.0324 | `L22:w2` 也略强于 `L20:w2`，但差距小于替换 `L16:w2`。 |
| `custom_58_swap_l19w2_l22w2` | 去掉 `L19:w2`，加入 `L22:w2` | `wo` 32, `w2` 18, `w3` 8 | 23.4971 | +0.0695 | `L19:w2` 不能轻易删，实际 PPL 价值高于 `L22:w2`。 |
| `custom_58_swap_l20w2_l24w3` | 去掉 `L20:w2`，加入 `L24:w3` | `wo` 32, `w2` 17, `w3` 9 | 23.4586 | +0.0309 | 后段高分 `w3` 暂时不能替代 `L20:w2`。 |
| `custom_58_swap_l20w2_l26w3` | 去掉 `L20:w2`，加入 `L26:w3` | `wo` 32, `w2` 17, `w3` 9 | 23.4620 | +0.0343 | 与 `L24:w3` 结论一致，额外 `w3` 的优先级应低于关键 `w2`。 |

这一轮最重要的信号是：`linear_score` 和实际 PPL 边际并不完全一致。按 `linear_score`，后段 `w2` 的排序是 `L16 > L19 > L20 > L22`；但 PPL swap 显示，至少在 58 层上下文中，`L19:w2` 很关键，`L22:w2` 应提前进入，`L16:w2` 反而更可替换。因此后续选层不能只依赖 score top-k，需要用局部 PPL swap/ablation 对 score 排名做校准。

当前临时排序可以写成：

```text
L19:w2 需要优先保留；L22:w2 应提前进入；L20:w2 仍有价值；L16:w2 需要继续验证，可能不是 56/57/58 甜点区中的最优选择。
```

对 `w3` 的判断也更清楚：已有的 8 个高分 `w3` 可以保留作为候选底座，但继续增加 `L24:w3` 或 `L26:w3` 来替换 `L20:w2` 并不划算。后续如果要增加 `w3`，应先做严格同成本对照，而不是仅凭 score 将其插入。

### 56 层同成本 swap 观察

这一轮固定 56 个 3-bit 层，净存储变化均约为 60.0 MiB。母版 `custom_wo_all_w2_w3_56` 是当前“最低越过 `mixed_0.3`”的甜点候选，原始选择里包含 `L16:w2`，但还没有 `L19/L20/L22:w2`。因此这组实验直接回答：在 60.0 MiB 预算下，后段 `w2` 里谁应该最先替代 `L16:w2`。

| 策略 | 变更 | PPL | 相对原始 56 | 观察 |
| --- | --- | ---: | ---: | --- |
| `custom_wo_all_w2_w3_56` | 原始 56 层，包含 `L16:w2` | 23.8751 | 0.0000 | 已经比 `mixed_0.3` 略好，且少约 5.0 MiB。 |
| `custom_56_swap_l16w2_l19w2` | 去掉 `L16:w2`，加入 `L19:w2` | 23.7345 | -0.1406 | 当前 60.0 MiB 最佳点，说明 `L19:w2` 应比 `L16:w2` 更早进入。 |
| `custom_56_swap_l16w2_l22w2` | 去掉 `L16:w2`，加入 `L22:w2` | 23.8176 | -0.0574 | `L22:w2` 也优于 `L16:w2`，但弱于 `L19:w2`。 |
| `custom_56_swap_l16w2_l20w2` | 去掉 `L16:w2`，加入 `L20:w2` | 23.8434 | -0.0316 | `L20:w2` 只小幅优于 `L16:w2`，收益不如 `L19/L22`。 |

这组结果把 60.0 MiB 预算下的局部排序写得更清楚：

```text
L19:w2 > L22:w2 > L20:w2 > L16:w2
```

这个排序和 `linear_score` 明显不同。`linear_score` 中 `L16:w2` 高于 `L19/L20/L22`，但 PPL 结果显示 `L16:w2` 在当前结构化 mask 中是可替换的。更谨慎的理解是：`linear_score` 可以给候选池，但最终 mask 应由 PPL 的局部 ablation/swap 来校准。

### 57 层同成本 swap 观察

这一轮固定 57 个 3-bit 层，净存储变化均约为 62.5 MiB。母版 `custom_wo_all_w2_w3_57` 在原始 56 的基础上加入了 `L19:w2`，因此原始 57 已包含 `L16:w2` 和 `L19:w2`，但还没有 `L20/L22:w2`。本轮实验用于判断：当 `L19:w2` 已经被保护时，第二个后段候选应该是 `L16`、`L20` 还是 `L22`。

| 策略 | 变更 | PPL | 相对原始 57 | 观察 |
| --- | --- | ---: | ---: | --- |
| `custom_wo_all_w2_w3_57` | 原始 57 层，包含 `L16:w2`、`L19:w2` | 23.5857 | 0.0000 | 比 `mixed_0.3` 低 0.3261 PPL，且少约 2.5 MiB。 |
| `custom_57_swap_l16w2_l22w2` | 去掉 `L16:w2`，保留 `L19:w2`，加入 `L22:w2` | 23.5287 | -0.0570 | 当前 62.5 MiB 最佳点，说明 `L22:w2` 应排在 `L16:w2` 前面。 |
| `custom_57_swap_l16w2_l20w2` | 去掉 `L16:w2`，保留 `L19:w2`，加入 `L20:w2` | 23.5711 | -0.0146 | `L20:w2` 也略优于 `L16:w2`，但收益远小于 `L22:w2`。 |

结合 56 层和 58 层实验，可以得到更稳定的局部判断：

```text
在 60.0 MiB 附近：L19:w2 最应该优先替代 L16:w2。
在 62.5 MiB 附近：保留 L19:w2 后，L22:w2 比 L20:w2 更应该替代 L16:w2。
在 65.0 MiB 附近：L19、L20、L22 同时保留，并去掉 L16，目前是已测 58 层中的最好组合。
```

这说明后段 `w2` 不是简单按 `linear_score` 顺序进入。`L16:w2` 的 score 虽然高于 `L19/L20/L22`，但在 56/57/58 三个预算点中都表现为相对可替换；`L19:w2` 和 `L22:w2` 的实际 PPL 价值更高。

#### 57 层进一步校准

在确认 `L22:w2` 应进入 57 层候选后，继续做了一组更细的同成本替换：固定 57 个 3-bit 层，分别用 `L22:w2` 替换原始 57 中的 `L17/L18/L19:w2`。这组实验用于判断 `L16/L17/L18/L19` 中哪些是强保留层，哪些只是因为 `linear_score` 较高而被提前选入。

| 策略 | 变更 | PPL | 相对原始 57 | 观察 |
| --- | --- | ---: | ---: | --- |
| `custom_57_swap_l17w2_l22w2` | 去掉 `L17:w2`，加入 `L22:w2` | 23.5193 | -0.0664 | 当前 57 层最佳点；`L22:w2` 比 `L17:w2` 更值得保留。 |
| `custom_57_swap_l16w2_l22w2` | 去掉 `L16:w2`，加入 `L22:w2` | 23.5287 | -0.0570 | 也明显优于原始 57，但略弱于替换 `L17:w2`。 |
| `custom_57_swap_l18w2_l22w2` | 去掉 `L18:w2`，加入 `L22:w2` | 23.5858 | +0.0001 | 基本等于原始 57，说明 `L18:w2` 不宜优先删。 |
| `custom_57_swap_l19w2_l22w2` | 去掉 `L19:w2`，加入 `L22:w2` | 23.6791 | +0.0934 | 明显退化，强烈支持 `L19:w2` 是 57 层附近的强敏感层。 |

因此，57 层预算下目前最好的局部组合不是简单把 `L16` 替换成 `L22`，而是保留 `L16/L18/L19`，用 `L22` 替换 `L17`：

```text
custom_57_swap_l17w2_l22w2: PPL 23.5193, net 62.5 MiB
```

这一点说明 `L16` 和 `L17` 的相对价值非常接近，不能仅凭一次替换把 `L16` 判为绝对低价值层；但 `L19` 的强保留结论已经比较稳定，`L18` 也比 `L16/L17` 更不适合被删。

### 58 层进一步校准

当前 58 层最佳候选是 `custom_58_swap_l16w2_l22w2`，即在原始 58 的基础上去掉 `L16:w2`，加入 `L22:w2`，PPL 为 23.3534。为了判断这个最佳组合里 `L17/L18` 是否也可能被 `L16` 替代，又做了两个同成本对照。

| 策略 | 变更 | PPL | 相对当前最佳 58 | 观察 |
| --- | --- | ---: | ---: | --- |
| `custom_58_swap_l16w2_l22w2` | 当前最佳 58，保留 `L17/L18/L19/L20/L22`，不保留 `L16` | 23.3534 | 0.0000 | 当前已测 65.0 MiB 最佳点。 |
| `custom_58_best_swap_l17w2_l16w2` | 在最佳 58 中去掉 `L17:w2`，换回 `L16:w2` | 23.3555 | +0.0021 | 几乎等价，说明 `L16/L17` 在 58 点是边界层。 |
| `custom_58_best_swap_l18w2_l16w2` | 在最佳 58 中去掉 `L18:w2`，换回 `L16:w2` | 23.4084 | +0.0550 | 明显退化，说明 `L18:w2` 比 `L16:w2` 更应保留。 |

结合 57 和 58 的校准结果，后段 `w2` 可以暂时分成三类：

```text
强保留层: L18:w2, L19:w2, L22:w2
有价值但需上下文判断: L20:w2
边界层: L16:w2, L17:w2
```

注意这里的“边界层”不是“不敏感”，而是在当前 `wo_all + 8个w3 + 后段w2` 的上下文中，它们与 `L22` 或彼此之间的差异很小。最终 mask 仍需要围绕候选预算点做局部验证。

### 59 层精炼候选观察

在 58 层最佳候选 `custom_58_swap_l16w2_l22w2` 基础上，继续测试增加 1 个 3-bit 层时应该优先加入谁。所有候选均为 59 个 3-bit 层，净存储变化均约为 67.5 MiB。已有 `custom_wo_all_w2_w3_59` 等价于“58 层最佳组合 + 加回 `L16:w2`”，因此新实验重点测试 `L2:w2`、`L23:w2`、`L24:w2` 和 `L24:w3`。

| 策略 | 相对最佳 58 的新增层 | module 分布 | PPL | 相对最佳 58 | 观察 |
| --- | --- | --- | ---: | ---: | --- |
| `custom_58_swap_l16w2_l22w2` | 无 | `wo` 32, `w2` 18, `w3` 8 | 23.3534 | 0.0000 | 当前 65.0 MiB 最佳点。 |
| `custom_59_from_best58_add_l23w2` | `L23:w2` | `wo` 32, `w2` 19, `w3` 8 | 23.1974 | -0.1560 | 当前 67.5 MiB 最佳点，比原始 59 更好。 |
| `custom_59_from_best58_add_l2w2` | `L2:w2` | `wo` 32, `w2` 19, `w3` 8 | 23.2036 | -0.1498 | 略弱于 `L23:w2`，但仍优于原始 59。 |
| `custom_wo_all_w2_w3_59` | `L16:w2` | `wo` 32, `w2` 19, `w3` 8 | 23.2199 | -0.1335 | 原始 59，说明加回边界层 `L16:w2` 有收益，但不是当前最优新增层。 |
| `custom_59_from_best58_add_l24w3` | `L24:w3` | `wo` 32, `w2` 18, `w3` 9 | 23.2320 | -0.1214 | 高分后段 `w3` 有收益，但弱于 `L23/L2:w2`。 |
| `custom_59_from_best58_add_l24w2` | `L24:w2` | `wo` 32, `w2` 19, `w3` 8 | 23.2338 | -0.1196 | 收益接近 `L24:w3`，但弱于 `L23/L2:w2`。 |

这一组结果说明，继续从 65.0 MiB 增加到 67.5 MiB 时，最优新增层仍然来自 `w2`，但不再是简单加回 `L16` 这个边界层。`L23:w2` 和 `L2:w2` 的实际 PPL 价值都高于 `L16:w2`。同时，`L24:w3` 的 `linear_score` 高于这些后段 `w2`，但 PPL 仍不如 `L23/L2:w2`，进一步证明 w3 的 score 不能直接和 w2 的实际保护收益等价比较。

当前 67.5 MiB 档位的候选排序为：

```text
custom_59_from_best58_add_l23w2 < custom_59_from_best58_add_l2w2 < custom_wo_all_w2_w3_59 < custom_59_from_best58_add_l24w3 < custom_59_from_best58_add_l24w2
```

这里用 PPL 从低到高排列。最好的 `custom_59_from_best58_add_l23w2` 比 `mixed_0.3` 低约 0.7144 PPL，但净存储只比 `mixed_0.3` 高约 2.5 MiB；如果 67.5 MiB 可以接受，它是当前最强的局部优化候选。

### 60 层精炼候选观察

在 59 层最佳候选 `custom_59_from_best58_add_l23w2` 基础上，继续测试第二个新增层。所有候选均为 60 个 3-bit 层，净存储变化均约为 70.0 MiB。主要比较对象包括再加入 `L2:w2`、`L16:w2`、`L24:w3` 和 `L24:w2`。其中 `custom_wo_all_w2_w3_60` 是未校准的原始 policy 60 层，它包含 `L16:w2` 和 `L2:w2`，但不包含 `L23:w2`，因此后续仍需补跑作为对照。

| 策略 | 相对最佳 59 的新增层 | module 分布 | PPL | 相对最佳 59 | 观察 |
| --- | --- | --- | ---: | ---: | --- |
| `custom_59_from_best58_add_l23w2` | 无 | `wo` 32, `w2` 19, `w3` 8 | 23.1974 | 0.0000 | 当前 67.5 MiB 最佳点。 |
| `custom_60_from_best59_l23_l2w2` | `L2:w2` | `wo` 32, `w2` 20, `w3` 8 | 23.0317 | -0.1657 | 当前 70.0 MiB 最佳点，继续补 `w2` 仍有明显收益。 |
| `custom_60_from_best59_l23_l16w2` | `L16:w2` | `wo` 32, `w2` 20, `w3` 8 | 23.0660 | -0.1314 | 边界层 `L16:w2` 在 60 层仍有收益，但弱于 `L2:w2`。 |
| `custom_60_from_best59_l23_l24w3` | `L24:w3` | `wo` 32, `w2` 19, `w3` 9 | 23.0669 | -0.1305 | 与 `L16:w2` 几乎打平，说明高分后段 `w3` 开始接近边界 `w2`。 |
| `custom_wo_all_w2_w3_60` | 原始 policy 60：`L16:w2`、`L2:w2`，不含 `L23:w2` | `wo` 32, `w2` 20, `w3` 8 | 23.0691 | -0.1283 | 未经 59 层校准的 policy 60，比最佳 60 高 0.0374 PPL。 |
| `custom_60_from_best59_l23_l24w2` | `L24:w2` | `wo` 32, `w2` 20, `w3` 8 | 23.0746 | -0.1228 | 有收益，但弱于 `L2/L16:w2` 和 `L24:w3`。 |

60 层结果说明，70.0 MiB 附近继续增加 1 个保护层仍有比较可观的边际收益，尤其 `L2:w2` 带来约 0.1657 PPL 下降。此时 `w3` 的边际价值开始接近部分边界 `w2`：`L24:w3` 几乎和 `L16:w2` 打平，并优于 `L24:w2`。因此后续如果继续往 61/62 层扩展，不能再简单假设所有新增都应该是 `w2`；应在 `L16/L24:w3/L24:w2/L23:w3/L9:w3` 等候选之间做小批量对照。

当前 70.0 MiB 档位的候选排序为：

```text
custom_60_from_best59_l23_l2w2 < custom_60_from_best59_l23_l16w2 ~= custom_60_from_best59_l23_l24w3 ~= custom_wo_all_w2_w3_60 < custom_60_from_best59_l23_l24w2
```

这里同样用 PPL 从低到高排列。最好的 `custom_60_from_best59_l23_l2w2` 比 `mixed_0.3` 低约 0.8801 PPL，但净存储高约 5.0 MiB；它也比 `mixed_0.4` 仍高约 0.4224 PPL，但比 `mixed_0.4` 少约 32.5 MiB 净存储，因此是当前更偏性能但仍明显省成本的候选。

### 61 层精炼候选观察

在 60 层最佳候选 `custom_60_from_best59_l23_l2w2` 基础上，继续测试第三个新增层。所有候选均为 61 个 3-bit 层，净存储变化均约为 72.5 MiB。此时候选从单纯 `w2` 扩展到后段 `w3`，重点观察 `w3` 是否开始超过边界 `w2`。

| 策略 | 相对最佳 60 的新增层 | module 分布 | PPL | 相对最佳 60 | 观察 |
| --- | --- | --- | ---: | ---: | --- |
| `custom_60_from_best59_l23_l2w2` | 无 | `wo` 32, `w2` 20, `w3` 8 | 23.0317 | 0.0000 | 当前 70.0 MiB 最佳点。 |
| `custom_61_from_best60_l23w3` | `L23:w3` | `wo` 32, `w2` 20, `w3` 9 | 22.9020 | -0.1298 | 当前 72.5 MiB 最佳点，特定后段 `w3` 已经超过边界 `w2`。 |
| `custom_wo_all_w2_w3_61` | `L16:w2` | `wo` 32, `w2` 21, `w3` 8 | 22.9050 | -0.1267 | 原始 61 policy，补回 `L16:w2` 也很强，但略弱于 `L23:w3`。 |
| `custom_61_from_best60_l24w3` | `L24:w3` | `wo` 32, `w2` 20, `w3` 9 | 22.9156 | -0.1162 | 接近 `L16:w2`，说明部分后段 `w3` 已进入有效候选区。 |
| `custom_61_from_best60_l9w3` | `L9:w3` | `wo` 32, `w2` 20, `w3` 9 | 22.9928 | -0.0399 | 明显弱于 `L23/L24:w3`，不能把所有高分 `w3` 一概提前。 |

61 层结果说明，从 70.0 MiB 继续增加到 72.5 MiB 时仍有收益，但层族边界发生变化：此前 58-60 层主要由 `w2` 主导，而 61 层最佳新增层变成了 `L23:w3`。这不是说 `w3` 整体优先级高于 `w2`，而是说明在 `L23:w2`、`L2:w2`、`L16:w2` 等关键或边界 `w2` 已经处理到一定程度后，部分后段 `w3` 的边际收益开始超过继续加入更弱的 `w2`。

当前 72.5 MiB 档位的候选排序为：

```text
custom_61_from_best60_l23w3 < custom_wo_all_w2_w3_61 < custom_61_from_best60_l24w3 < custom_61_from_best60_l9w3
```

这里用 PPL 从低到高排列。当前最强的 `custom_61_from_best60_l23w3` 比 `mixed_0.4` 仍高约 0.2926 PPL，但比 `mixed_0.4` 少约 30.0 MiB 净存储，因此 72.5 MiB 档位已经是很有竞争力的高性能候选。不过从 60 到 61 的最佳边际收益约 0.1298 PPL，低于 59 到 60 的 0.1657 PPL，开始能看到边际收益下降的迹象。

### 62 层精炼候选观察

在 61 层最佳候选 `custom_61_from_best60_l23w3` 基础上，继续测试第四个新增层。所有候选均为 62 个 3-bit 层，净存储变化均约为 75.0 MiB。这一轮主要比较 `L16:w2`、`L24:w3`、`L24:w2`，并把原始 `custom_wo_all_w2_w3_62` 作为未校准 policy 对照。

| 策略 | 相对最佳 61 的新增层 | module 分布 | PPL | 相对最佳 61 | 观察 |
| --- | --- | --- | ---: | ---: | --- |
| `custom_61_from_best60_l23w3` | 无 | `wo` 32, `w2` 20, `w3` 9 | 22.9020 | 0.0000 | 当前 72.5 MiB 最佳点。 |
| `custom_62_from_best61_l16w2` | `L16:w2` | `wo` 32, `w2` 21, `w3` 9 | 22.7795 | -0.1225 | 当前 75.0 MiB 最佳点，`L16:w2` 在高预算上下文中重新变得重要。 |
| `custom_wo_all_w2_w3_62` | 原始 policy 62：`L16:w2`、`L24:w2`，不含 `L23:w3` | `wo` 32, `w2` 22, `w3` 8 | 22.7857 | -0.1162 | 与最佳 62 只差 0.0062 PPL，说明两条路线都很接近。 |
| `custom_62_from_best61_l24w2` | `L24:w2` | `wo` 32, `w2` 21, `w3` 9 | 22.7876 | -0.1143 | 略弱于原始 62 和 `L16:w2`。 |
| `custom_62_from_best61_l24w3` | `L24:w3` | `wo` 32, `w2` 20, `w3` 10 | 22.7880 | -0.1140 | 与 `L24:w2` 几乎相同，也接近原始 62。 |

62 层的 256 samples 结果说明，从 72.5 MiB 到 75.0 MiB 仍有约 0.12 PPL 的收益，但同一预算内候选之间的差距已经非常小。256 samples 下最佳 `L16:w2` 路线只比原始 62 低 0.0062 PPL，比 `L24:w2`/`L24:w3` 低约 0.008-0.009 PPL。这意味着 75.0 MiB 附近已经进入局部微调区：单次 Wikitext2 256 samples 的噪声可能开始影响细微排序，后续若要把 62 层作为主候选，应该做 512 samples 或换数据集验证。

当前 75.0 MiB 档位的候选排序为：

```text
custom_62_from_best61_l16w2 < custom_wo_all_w2_w3_62 < custom_62_from_best61_l24w2 ~= custom_62_from_best61_l24w3
```

这里用 256-sample PPL 从低到高排列。`custom_62_from_best61_l16w2` 比 `mixed_0.4` 仍高约 0.1701 PPL，但比 `mixed_0.4` 少约 27.5 MiB 净存储。与 58-61 层相比，62 层已更接近 `mixed_0.4`，但边际收益和同预算差异都在变小。

### 512 samples 验证

评估脚本对 Wikitext2 不是随机采样，而是顺序读取前 N 个有效样本。因此，简单重复同一个 256 samples 命令只会复现同一批样本，不能有效估计随机波动。更有意义的验证方式是提高 `N_SAMPLES`，例如从 256 增加到 512。注意 256 和 512 的绝对 PPL 不能直接混合比较；应该只比较同一 `N_SAMPLES` 下的排序和边际差距。

本轮对 60/61/62 候选和 `mixed_0.4` 做了 512 samples 验证。第一批先验证 60/61/62 的跨预算 Pareto 阶梯：

| 策略 | 3-bit 层数 | 估算净存储变化 | 512-sample PPL | 512-sample NLL | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| `custom_60_from_best59_l23_l2w2` | 60 | 70.0 MiB | 21.9463 | 3.0886 | 70.0 MiB 档位仍是稳定候选。 |
| `custom_61_from_best60_l23w3` | 61 | 72.5 MiB | 21.8344 | 3.0835 | 相比 60 层继续降低 0.1119 PPL。 |
| `custom_62_from_best61_l16w2` | 62 | 75.0 MiB | 21.7138 | 3.0779 | 相比 61 层继续降低 0.1206 PPL，验证了 61 -> 62 的收益。 |
| `mixed_0.4` | 64 | 102.5 MiB | 21.5957 | 3.0725 | 高成本 baseline，仍低于 62 层候选 0.1181 PPL。 |

512 samples 下，60/61/62 的排序仍然保持单调改善，并且 61 -> 62 的收益与 256 samples 下的约 0.1225 PPL 非常接近。这说明“62 层候选相对 61 层有稳定收益”这个判断比单次 256 更可靠。

第一批 512 验证中，`custom_62_from_best61_l16w2` 与 `mixed_0.4` 的差距为约 0.1181 PPL，而净存储估算仍少约 27.5 MiB，已经支持 62 层候选作为高性能省成本点。但由于 62 层同预算内多个候选在 256 samples 下只差 0.01 PPL 以内，又继续对 62 层同成本候选做了 512 samples 验证。补充验证后，当前 512 samples 最稳的 62 层候选更新为 `custom_wo_all_w2_w3_62`，它与 `mixed_0.4` 的差距为约 0.0925 PPL。

62 层同成本候选的 512 samples 结果如下：

| 策略 | 256-sample PPL | 512-sample PPL | 512-sample NLL | 观察 |
| --- | ---: | ---: | ---: | --- |
| `custom_wo_all_w2_w3_62` | 22.7857 | 21.6881 | 3.0768 | 512 samples 下反超，成为当前 75.0 MiB 最稳候选。 |
| `custom_62_from_best61_l24w2` | 22.7876 | 21.6932 | 3.0770 | 512 samples 下排第二，接近原始 62。 |
| `custom_62_from_best61_l24w3` | 22.7880 | 21.7074 | 3.0777 | 512 samples 下优于 `L16:w2` 路线。 |
| `custom_62_from_best61_l16w2` | 22.7795 | 21.7138 | 3.0779 | 256 samples 最好，但 512 samples 下变成四者中最弱。 |

这个反转非常重要：它说明 62 层内部 0.01 PPL 以内的 256-sample 排序不能直接作为最终 mask 决策。更长样本下，原始 62 policy `custom_wo_all_w2_w3_62` 更稳；它包含 `L16:w2` 和 `L24:w2`，但不包含 `L23:w3`。因此当前 75.0 MiB 档位应暂时把 `custom_wo_all_w2_w3_62` 视为首选候选，而不是 256 samples 下的 `custom_62_from_best61_l16w2`。

#### 补齐后的 512 samples Pareto 曲线

随后补跑了 `mixed_0.3`、58/59 层候选，并进一步补齐 54/55/56/57 的低成本边界点。所有结果均为 Wikitext2 512 samples，同一采样口径下可以直接比较排序和边际收益。

| 策略 | 3-bit 层数 | module 分布 | 估算净存储变化 | 512-sample PPL | 512-sample NLL | 观察 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `mixed_0.3` | 48 | `wo` 22, `w2` 10, `w3` 16 | 65.0 MiB | 23.0088 | 3.1359 | 同口径 baseline。 |
| `custom_wo_all_w2_w3_54` | 54 | `wo` 32, `w2` 14, `w3` 8 | 55.0 MiB | 23.0733 | 3.1387 | 低于 `mixed_0.3` 的成本，但 PPL 仍高 0.0645。 |
| `custom_wo_all_w2_w3_55` | 55 | `wo` 32, `w2` 15, `w3` 8 | 57.5 MiB | 22.9076 | 3.1315 | 刚越过 `mixed_0.3`，但优势只有约 0.1012 PPL。 |
| `custom_56_swap_l16w2_l19w2` | 56 | `wo` 32, `w2` 16, `w3` 8 | 60.0 MiB | 22.6492 | 3.1201 | 低成本更稳候选，比 `mixed_0.3` 低约 0.3595 PPL。 |
| `custom_57_swap_l17w2_l22w2` | 57 | `wo` 32, `w2` 17, `w3` 8 | 62.5 MiB | 22.4349 | 3.1106 | 继续明显改善，比 `mixed_0.3` 低约 0.5739 PPL。 |
| `custom_58_swap_l16w2_l22w2` | 58 | `wo` 32, `w2` 18, `w3` 8 | 65.0 MiB | 22.2984 | 3.1045 | 与 `mixed_0.3` 估算净存储相同，但低约 0.7104 PPL，明确支配。 |
| `custom_59_from_best58_add_l23w2` | 59 | `wo` 32, `w2` 19, `w3` 8 | 67.5 MiB | 22.1449 | 3.0976 | 58 -> 59 仍降低约 0.1535 PPL，`L23:w2` 的收益稳定。 |
| `custom_60_from_best59_l23_l2w2` | 60 | `wo` 32, `w2` 20, `w3` 8 | 70.0 MiB | 21.9463 | 3.0886 | 59 -> 60 降低约 0.1987 PPL，`L2:w2` 也很敏感。 |
| `custom_61_from_best60_l23w3` | 61 | `wo` 32, `w2` 20, `w3` 9 | 72.5 MiB | 21.8344 | 3.0835 | 特定后段 `w3` 在该预算附近开始变得有价值。 |
| `custom_wo_all_w2_w3_62` | 62 | `wo` 32, `w2` 22, `w3` 8 | 75.0 MiB | 21.6881 | 3.0768 | 当前 75.0 MiB 首选，比 `mixed_0.4` 只高约 0.0925 PPL。 |
| `mixed_0.4` | 64 | `wo` 23, `w1` 1, `w2` 14, `w3` 26 | 102.5 MiB | 21.5957 | 3.0725 | 高成本强 baseline。 |

这一轮补齐后，低成本边界更清楚：`custom_wo_all_w2_w3_54` 还没有越过 `mixed_0.3`，而 `custom_wo_all_w2_w3_55` 刚刚越过。因此在当前 policy 和 Wikitext2 512 samples 下，最低越线点大约落在 55 层、57.5 MiB。不过 55 层只比 `mixed_0.3` 低约 0.10 PPL，仍属于较窄优势；如果要作为稳健甜点，56/57 层更可信。

边界层也值得记录：54 -> 55 新增的是 `L18:w2`，让模型刚越过 `mixed_0.3`；55 -> 56 新增的是 `L19:w2`，带来约 0.2583 PPL 的明显收益；56 -> 57 不是简单追加一个层，而是加入 `L16:w2` 和 `L22:w2`，同时移除 `L17:w2`，整体再降低约 0.2144 PPL。这进一步说明，当前最敏感的边界层仍集中在 `w2`，并且 56/57 附近需要用局部 swap 校准，不能只按层数或线性 score 顺序外推。

本轮也记录一次失败实验：`custom_59_from_best58_add_l23w2` 首次尝试在 GPU 5 上运行 512 samples 时，启动后 GPU 5 显存被其他进程占用，模型 `.cuda()` 阶段 OOM，没有产生有效 PPL JSON。随后重新查询 GPU，改用 GPU 7 成功完成。这再次确认每次实验前必须实时查看显存，不能沿用上一轮 GPU 选择。

### 跨数据集验证尝试

为了确认 Wikitext2 上的结论是否泛化，尝试使用 PTB 对当前 75.0 MiB 首选 `custom_wo_all_w2_w3_62` 做 256 samples 验证。实验启动环境和 PPL 流程正常，但数据集下载阶段失败：

```text
失败类型: 网络 / 数据下载超时
失败位置: load_dataset("ptb_text_only", "penn_treebank", split="test")
失败原因: raw.githubusercontent.com ReadTimeout, read timeout=100.0
是否产生有效 PPL JSON: 否
```

这个失败不说明 cache、量化流程或 GPU 有问题，只说明 PTB 数据源依赖 `raw.githubusercontent.com`，当前网络环境下不稳定。后续如果要做 PTB/C4/Pile-val 验证，建议先把数据集缓存好，或改用可访问的镜像/本地数据路径，再启动长时间 PPL。

### 下一步行动

1. 512 samples 已确认最低越线点大约落在 `custom_wo_all_w2_w3_55`，但该点只比 `mixed_0.3` 低约 0.10 PPL，优势偏窄；更稳的低成本甜点应优先看 `custom_56_swap_l16w2_l19w2` 和 `custom_57_swap_l17w2_l22w2`。
2. 围绕 `custom_wo_all_w2_w3_56/57/58` 做删层实验，重点验证后段 `w2` 的真实贡献：`L16:w2`、`L19:w2`、`L20:w2`、`L22:w2`。
3. 做少量同成本替换实验：用 `mixed_0.3` 中的中后段高分 `w3` 替换部分后段 `w2`，确认 `w2` 与 `w3` 的边际收益边界。
4. 若 `custom_wo_all_w2_w3_56/57/58` 稳定成为候选，应将最终 mask 固化到生成流程，而不只是在已有 cache 上做后处理。
5. 在最终候选上考虑增加验证集或重复评估，避免只依赖 Wikitext2 256 samples 的单次结果。
6. 56 层同成本 swap 已确认 `L19:w2` 是当前 60.0 MiB 点最优替换，后续应将 `custom_56_swap_l16w2_l19w2` 作为新的低成本甜点候选。
7. 57 层进一步校准已确认 `custom_57_swap_l17w2_l22w2` 是当前 62.5 MiB 最佳点，而不是早先的 `custom_57_swap_l16w2_l22w2`。
8. `L19:w2` 在 56/57/58 的多个对照中都表现为强保留层；`L18:w2` 也不应优先删；`L16/L17:w2` 属于边界层，需要在最终预算点附近继续局部确认。
9. 59 层精炼已确认 `custom_59_from_best58_add_l23w2` 是当前 67.5 MiB 最佳点，优于原始 59 的加回 `L16:w2`。
10. 60 层精炼已确认 `custom_60_from_best59_l23_l2w2` 是当前 70.0 MiB 最佳点，继续补 `w2` 仍然有效，但 `L24:w3` 已经接近边界 `L16:w2`。
11. 61 层精炼已确认 `custom_61_from_best60_l23w3` 是当前 72.5 MiB 最佳点，说明特定后段 `w3` 已开始超过边界 `w2`。
12. 62 层精炼的 256 samples 最佳是 `custom_62_from_best61_l16w2`，但 512 samples 同成本验证发生排序反转，当前更稳的 75.0 MiB 候选应改为 `custom_wo_all_w2_w3_62`。
13. 512 samples 验证已确认 54/55/56/57/58/59/60/61/62 跨预算候选形成清晰 Pareto 阶梯：54 尚未越过 `mixed_0.3`，55 刚越过，56/57 更稳，58 在同净存储下明确支配 `mixed_0.3`，62 与 `mixed_0.4` 的差距缩小到约 0.0925 PPL，同时仍少约 27.5 MiB 净存储。
14. PTB 跨数据集验证已尝试，但失败于 `raw.githubusercontent.com` 数据下载超时，没有产生有效 PPL。后续跨数据集验证前应先解决数据缓存或镜像问题。
15. 下一批实验可以补跑 `custom_62_from_best61_l9w3`，但优先级不高；更重要的是对 55/56/57/58/59/60/61/62 的 Pareto 候选做更多数据集验证，特别是 PTB 或 C4，以确认 Wikitext2 上的结论是否泛化。
16. 当前应重点保留 55/56/57/58/59/60/61/62 作为 Pareto 阶梯：`custom_wo_all_w2_w3_55`、`custom_56_swap_l16w2_l19w2`、`custom_57_swap_l17w2_l22w2`、`custom_58_swap_l16w2_l22w2`、`custom_59_from_best58_add_l23w2`、`custom_60_from_best59_l23_l2w2`、`custom_61_from_best60_l23w3`、`custom_wo_all_w2_w3_62`。如果要写论文或主结果，应对这些候选做重复评估或换数据集验证。

### 当前优先实验队列

Wikitext2 512 samples 的主 Pareto 曲线已经补齐，下一轮不应继续盲目扩高层数，而应围绕两个问题做验证：低成本边界是否稳定、Wikitext2 结论是否能泛化到其他数据集。

优先级如下：

1. 先解决 PTB/C4 的数据缓存或镜像问题，再对 `custom_wo_all_w2_w3_55`、`custom_56_swap_l16w2_l19w2`、`custom_57_swap_l17w2_l22w2`、`custom_58_swap_l16w2_l22w2`、`custom_wo_all_w2_w3_62` 做跨数据集验证。
2. 围绕 55/56/57 做局部 swap，重点确认 `L18:w2`、`L19:w2`、`L16:w2`、`L17:w2`、`L22:w2` 的真实边界价值。
3. 如果跨数据集验证资源有限，优先验证三个代表点：55 层最低越线点、57 层低成本稳健点、62 层高性能省成本点。
4. 若 55 层在其他数据集上不稳定，应把 56 或 57 作为低成本甜点；若 55 层也稳定，则可以把 57.5 MiB 作为最小成本主候选。

后续命令仍使用统一模板。每条命令执行前都要重新查询 GPU，并把 `GPU_ID` 替换成当时剩余显存最高的卡：

```bash
cd /home/users/Mayinyi/jikuixie/new-MBQ
export PYTHONPATH=.
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export HF_ENDPOINT=https://hf-mirror.com
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
GPU_ID=<显存最空的GPU编号> N_SAMPLES=512 SCALE_FILE_NAME=<目标scale_cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

每个实验结束后，应从最新 JSON 中核对以下字段，再写入结果表：

```text
args.dataset
args.n_samples
args.scale_path
summary.avg_ppl
summary.avg_nll
quant_state.linear_bit_counts
quant_state.low_rank_count
```

如果跨数据集下 55/56/57 仍稳定优于或接近 `mixed_0.3`，说明甜点区可以向 57.5-62.5 MiB 收敛；如果低成本点优势消失，则应把 58/59/60 或 62 作为更稳候选，并继续分析具体退化来自哪些层。

跨数据集验证暂时不要直接重复 PTB 命令。上一次失败点是数据集下载超时，而不是量化流程问题。下一步应先解决数据集缓存或镜像路径，再对当前 Pareto 候选跑 PTB/C4；否则容易消耗 GPU 时间却得不到有效 PPL。

### 2026-06-10 暂停实验时的阶段总结

目前已经暂停后续实验，阶段性结论应以已有 Wikitext2 512 samples 为主证据，以刚完成的本地 PTB 256 samples 为弱交叉验证证据。

#### 已确认的主结论

1. 自定义 `linear_bit_map` 保护敏感层是可行的，而且比单纯 global `keep_ratio` 更有成本优势。关键前提仍然是：被提升到 3-bit 的层必须从 `low_rank` 中过滤掉，避免同一层同时获得 3-bit 和 SVD residual 两种补偿。
2. `w2` 是当前最重要的敏感层族。低成本边界和中等预算收益几乎都由新增或替换 `feed_forward.w2` 决定，而不是由继续增加 `wqkv` 或盲目增加 `w3` 决定。
3. 全部 `attention.wo` 作为底座很划算。`wo` 的净存储代价很低，能提供稳定基础收益；但只保护 `wo` 不够，后续必须加入关键 `w2` 才能接近或超过 `mixed_0.3`。
4. `wqkv` 在当前实验中仍然不是优先保护对象。`costaware_40` 加入了 `wqkv`，但 PPL 明显弱于加入 `w2` 的策略。
5. `w3` 不是完全无效，但它更像中高预算之后的补充项；在低成本甜点区，关键 `w2` 的优先级明显高于大多数边界 `w3`。
6. `w1` 目前也不应作为低成本优先保护对象。现有证据不能说明 `w1` 完全无效，但它进入 score 排名较晚，且当前不含 `w1` 的自定义 Pareto 候选已经能接近或超过 global baseline。

#### w1 的补充分析

目前没有专门针对 `feed_forward.w1` 的同成本 swap 或 ablation，因此对 `w1` 的判断应当比 `w2` 更谨慎。现有实验能提供的是间接证据：`w1` 不是当前低成本甜点区的优先保护层族，但还不能被判定为完全不敏感。

从 `linear_score_map` 看，`w1` 进入排序明显晚于 `wo/w3/w2`：

```text
第一个 w1: rank 64, L3:w1
第二个 w1: rank 65, L2:w1
第一个 w2: rank 33, L6:w2
第一个 w3: rank 9,  L4:w3
第一个 wo: rank 1,  L14:wo
```

按 global top-k 的 module 分布看：

```text
top48 / mixed_0.3: w1 = 0
top64 / mixed_0.4: w1 = 1
top80 / mixed_0.5: w1 = 2
top96 / mixed_0.6: w1 = 8
top112 / mixed_0.7: w1 = 14
```

这说明在 score 排名中，`w1` 主要是高预算阶段才开始大量进入；它不是 55-62 层低成本搜索区间自然会优先选中的层族。

从已有 PPL 结果看，不含 `w1` 的自定义策略已经很强：`custom_wo_all_w2_w3_62` 不包含任何 `w1`，实际等效 bit 为 2.4426，Wikitext2-512 PPL 为 21.6881；`mixed_0.4` 包含 1 个 `w1`，实际等效 bit 为 2.5041，Wikitext2-512 PPL 为 21.5957。二者只差约 0.0925 PPL，但 `custom_62` 少约 51.2 MiB 实际成本。这不能证明 `w1` 没有贡献，因为 `mixed_0.4` 同时还多了很多 `w3/wo/w2`；但它说明在当前预算下，没有必要为了追求 `w1` 而牺牲已经验证更强的 `w2` 选择。

`mixed_0.4 -> mixed_0.5` 的提升也不能直接归因于 `w1`。这一段新增了 10 个 `w2`、3 个 `wo`、2 个 `w3` 和 1 个 `w1`，PPL 下降很大，但变量太多。结合前面的结构化实验和 swap 结果，更合理的解释是：主要收益仍然来自继续补关键 `w2`，`w1` 的单独贡献尚未被隔离。

当前对 `w1` 的阶段判断：

```text
低成本区间: 不优先保护 w1。
中高预算区间: w1 可以进入候选池，但需要专门 swap 验证。
当前主线: 先保护全 wo + 关键 w2 + 少量已验证 w3，再考虑 w1。
```

如果后续要验证 `w1`，建议只做少量同成本对照，例如在 62-64 层附近尝试把一个边界 `w3` 或边界 `w2` 替换为 `L3:w1` / `L2:w1`。在没有这类直接实验前，`w1` 应保持为“晚进入候选层”，不应写成强敏感层。

#### 当前 Pareto 阶梯

Wikitext2 512 samples 下，当前最重要的候选如下：

| 候选 | 净存储变化 | PPL | 阶段判断 |
| --- | ---: | ---: | --- |
| `custom_wo_all_w2_w3_54` | 55.0 MiB | 23.0733 | 成本低，但还没有超过 `mixed_0.3`。 |
| `custom_wo_all_w2_w3_55` | 57.5 MiB | 22.9076 | Wikitext2 上刚越过 `mixed_0.3`，但优势只有约 0.10 PPL。 |
| `custom_56_swap_l16w2_l19w2` | 60.0 MiB | 22.6492 | 更可信的低成本甜点，比 `mixed_0.3` 低约 0.36 PPL。 |
| `custom_57_swap_l17w2_l22w2` | 62.5 MiB | 22.4349 | 当前低成本稳健点，比 `mixed_0.3` 低约 0.57 PPL。 |
| `custom_58_swap_l16w2_l22w2` | 65.0 MiB | 22.2984 | 与 `mixed_0.3` 同净存储，但低约 0.71 PPL，明确支配。 |
| `custom_59_from_best58_add_l23w2` | 67.5 MiB | 22.1449 | 只比 `mixed_0.3` 多约 2.5 MiB，但收益继续明显。 |
| `custom_60_from_best59_l23_l2w2` | 70.0 MiB | 21.9463 | 偏性能的低成本候选，仍明显低于 `mixed_0.4` 的成本。 |
| `custom_61_from_best60_l23w3` | 72.5 MiB | 21.8344 | 说明特定 `w3` 在中高预算开始值得加入。 |
| `custom_wo_all_w2_w3_62` | 75.0 MiB | 21.6881 | 当前 75 MiB 首选，距离 `mixed_0.4` 只差约 0.0925 PPL，但省约 27.5 MiB。 |

对照点：`mixed_0.3` 为 65.0 MiB、PPL 23.0088；`mixed_0.4` 为 102.5 MiB、PPL 21.5957。

#### 低成本边界的解释

54 层还没有超过 `mixed_0.3`，55 层刚越过。54 -> 55 新增 `L18:w2`，说明 `L18:w2` 是低成本边界上的关键层之一。55 -> 56 加入 `L19:w2` 后收益明显扩大，说明 `L19:w2` 比单纯“刚好越线”的边界层更稳。57 层最佳点通过加入 `L22:w2` 并移除 `L17:w2` 获得进一步收益，说明 `L22:w2` 应提前进入保护集合。

当前更可信的敏感层判断：

```text
强敏感: L18:w2, L19:w2, L22:w2, L23:w2, L2:w2
边界敏感: L16:w2, L17:w2, L20:w2, L24:w2
中高预算后开始有价值: L23:w3, 少量后段 w3
低优先级: wqkv
```

#### PTB 交叉验证的新信息

本地 PTB 路径已经可用，已完成两个 256-sample 点：

| 候选 | PTB 256 PPL | 观察 |
| --- | ---: | --- |
| `mixed_0.3` | 225.3134 | PTB baseline。 |
| `custom_wo_all_w2_w3_55` | 225.3422 | 比 `mixed_0.3` 略差约 0.0288。 |

PTB 的绝对 PPL 不能和 Wikitext2 直接比较；这里只看同一 PTB 设置下的相对排序。这个结果说明：55 层虽然在 Wikitext2 512 上刚越过 `mixed_0.3`，但优势很窄，换到 PTB 256 后没有保持住。因此 55 层不能作为稳健主结论，只能作为“最低成本边界点”。

PTB 目前只跑了两个点，不能据此否定 56/57/58/62 的 Wikitext2 结论。更合理的判断是：低成本边界需要上移，56/57 比 55 更适合作为低成本甜点候选；后续若继续交叉验证，应优先跑 57、58、62，而不是继续围绕 55 做太多微调。

#### 暂停时的决策

1. 不再盲目继续扩高层数；目前主线已经从“发现策略”进入“验证策略”。
2. 55 层是最低越线点，但不够稳；如果需要一个低成本主候选，优先考虑 56 或 57。
3. 58 层是最干净的同成本胜出点，因为它与 `mixed_0.3` 同净存储但 PPL 明显更低。
4. 62 层是高性能省成本点，适合和 `mixed_0.4` 对比。


### 实际 bit 宽度核算

这一节按更接近部署打包的实际 bit 口径重新计算所有已跑实验的等效 bit 宽度。这里与前文“估算净存储变化”的口径不同：前文主要用于搜索期相对比较，扣除的是 cache 中 half tensor 形式保存的 low-rank 张量大小；本节按实际 bit 计算，SVD residual 按 4-bit 计入。

当前成本口径如下：

1. 主 weight 使用 `linear_bit_map` 中的 bit：未保护层为 2-bit，被保护层为 3-bit。
2. `scale` 来自 scale search，并且后续可以融合到前后层，因此不计入成本。
3. double quant 中的 primary scale / zero-point 也不计入本节成本；它们只影响伪量化数值。
4. 仍保留在 `low_rank` 中的 SVD residual 按 `up` 和 `down` 两个因子各自 4-bit 计入成本。
5. 被提升到 3-bit 的层已从 `low_rank` 删除，因此这些层只计 3-bit base weight，不再计 SVD residual。
6. 分母只使用当前 MBQ cache 覆盖的 160 个目标 linear 的原始 weight 参数量，不包含 embedding、lm head、vision tower 或未纳入 `linear_bit_map` 的其他模块。

从 `mixed_0.0.pt` 读取到的目标 linear 总量为：

```text
目标 linear weight 参数量: 6,979,321,856
全量 low-rank up/down 参数量: 301,989,888
1 bit over target linears: 832.0 MiB
全量 4-bit SVD residual: 144.0 MiB，折合 0.1731 bit/weight
mixed_0.0 实际等效 bit: 2.1731 bit
```

计算公式：

```text
actual_bits = sum_i(numel(W_i) * bit_i)
              + sum_{i in low_rank}(numel(up_i) + numel(down_i)) * 4

实际等效 bit = actual_bits / sum_i(numel(W_i))
相对 mixed_0.0 实际增量 MiB = (实际等效 bit - 2.1731) * 832.0
```

注意：本节的“相对 `mixed_0.0` 实际增量”不是前文的“估算净存储变化”。在实际 4-bit SVD 口径下，`attention.wo` 仍然比 MLP 层便宜，但不再是零成本；因此后续正式汇报成本时，应优先使用本节的“实际等效 bit”。

#### Baseline sweep 实际 bit

| 实验/cache | 实际等效 bit | 相对 `mixed_0.0` 实际增量 | 3-bit/2-bit | low-rank | 3-bit module 分布 | PPL 记录 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mixed_0.0` | 2.1731 | 0.0 MiB | 0/160 | 160 | `-` | W2-256 29.7054 |
| `mixed_0.1` | 2.2230 | 41.5 MiB | 16/144 | 144 | `w3:4, wo:12` | W2-256 27.5484 |
| `mixed_0.2` | 2.2939 | 100.5 MiB | 32/128 | 128 | `w3:12, wo:20` | W2-256 26.1439 |
| `mixed_0.3` | 2.3963 | 185.8 MiB | 48/112 | 112 | `w2:10, w3:16, wo:22` | W2-512 23.0088<br>W2-256 23.9118<br>PTB-256 225.3134 |
| `mixed_0.4` | 2.5041 | 275.4 MiB | 64/96 | 96 | `w1:1, w2:14, w3:26, wo:23` | W2-512 21.5957<br>W2-256 22.6093 |
| `mixed_0.5` | 2.6013 | 356.3 MiB | 80/80 | 80 | `w1:2, w2:24, w3:28, wo:26` | W2-256 20.6337 |
| `mixed_0.6` | 2.6985 | 437.1 MiB | 96/64 | 64 | `w1:8, w2:30, w3:29, wo:29` | W2-256 19.6061 |
| `mixed_0.7` | 2.7767 | 502.2 MiB | 112/48 | 48 | `w1:14, w2:30, w3:31, wo:30, wqkv:7` | W2-256 18.9858 |

#### 初始结构化策略实际 bit

| 实验/cache | 实际等效 bit | 相对 `mixed_0.0` 实际增量 | 3-bit/2-bit | low-rank | 3-bit module 分布 | PPL 记录 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `custom_wo_core` | 2.2037 | 25.5 MiB | 17/143 | 143 | `wo:17` | W2-256 28.2964 |
| `custom_global_top24` | 2.2532 | 66.6 MiB | 24/136 | 136 | `w3:7, wo:17` | W2-256 27.1086 |
| `custom_wo_w2_24` | 2.2532 | 66.6 MiB | 24/136 | 136 | `w2:7, wo:17` | W2-256 26.7413 |
| `custom_wo_w2_32` | 2.3096 | 113.6 MiB | 32/128 | 128 | `w2:15, wo:17` | W2-256 25.5038 |
| `custom_global_top40` | 2.3451 | 143.1 MiB | 40/120 | 120 | `w2:6, w3:13, wo:21` | W2-256 25.1731 |
| `custom_wo_w2_w3_36` | 2.3379 | 137.1 MiB | 36/124 | 124 | `w2:11, w3:8, wo:17` | W2-256 25.2790 |
| `custom_wo_w2_w3_40` | 2.3661 | 160.6 MiB | 40/120 | 120 | `w2:15, w3:8, wo:17` | W2-256 24.6909 |
| `custom_wo_w2_w3_41` | 2.3732 | 166.5 MiB | 41/119 | 119 | `w2:16, w3:8, wo:17` | W2-256 24.5302 |
| `custom_wo_w2_w3_42` | 2.3803 | 172.4 MiB | 42/118 | 118 | `w2:17, w3:8, wo:17` | W2-256 24.2660 |
| `custom_wo_w2_w3_43` | 2.3873 | 178.3 MiB | 43/117 | 117 | `w2:18, w3:8, wo:17` | W2-256 24.1117 |
| `custom_wo_w2_w3_44` | 2.3944 | 184.1 MiB | 44/116 | 116 | `w2:19, w3:8, wo:17` | W2-256 23.9151 |
| `custom_costaware_32` | 2.2308 | 48.0 MiB | 32/128 | 128 | `wo:32` | W2-256 27.2895 |
| `custom_costaware_40` | 2.2536 | 67.0 MiB | 40/120 | 120 | `wo:32, wqkv:8` | W2-256 26.7521 |

#### `wo_all_w2_w3` Pareto 主线实际 bit

| 实验/cache | 实际等效 bit | 相对 `mixed_0.0` 实际增量 | 3-bit/2-bit | low-rank | 3-bit module 分布 | PPL 记录 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `custom_wo_all_w2_w3_48` | 2.3438 | 142.0 MiB | 48/112 | 112 | `w2:8, w3:8, wo:32` | W2-256 24.8853 |
| `custom_wo_all_w2_w3_51` | 2.3649 | 159.6 MiB | 51/109 | 109 | `w2:11, w3:8, wo:32` | W2-256 24.5763 |
| `custom_wo_all_w2_w3_54` | 2.3861 | 177.3 MiB | 54/106 | 106 | `w2:14, w3:8, wo:32` | W2-512 23.0733 |
| `custom_wo_all_w2_w3_55` | 2.3932 | 183.1 MiB | 55/105 | 105 | `w2:15, w3:8, wo:32` | W2-512 22.9076<br>W2-256 24.0250<br>PTB-256 225.3422 |
| `custom_wo_all_w2_w3_56` | 2.4002 | 189.0 MiB | 56/104 | 104 | `w2:16, w3:8, wo:32` | W2-256 23.8751 |
| `custom_wo_all_w2_w3_57` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-256 23.5857 |
| `custom_wo_all_w2_w3_58` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-256 23.4276 |
| `custom_wo_all_w2_w3_59` | 2.4214 | 206.6 MiB | 59/101 | 101 | `w2:19, w3:8, wo:32` | W2-256 23.2199 |
| `custom_wo_all_w2_w3_60` | 2.4285 | 212.5 MiB | 60/100 | 100 | `w2:20, w3:8, wo:32` | W2-256 23.0691 |
| `custom_wo_all_w2_w3_61` | 2.4355 | 218.4 MiB | 61/99 | 99 | `w2:21, w3:8, wo:32` | W2-256 22.9050 |
| `custom_wo_all_w2_w3_62` | 2.4426 | 224.2 MiB | 62/98 | 98 | `w2:22, w3:8, wo:32` | W2-512 21.6881<br>W2-256 22.7857 |

#### 56/57/58 swap 实际 bit

| 实验/cache | 实际等效 bit | 相对 `mixed_0.0` 实际增量 | 3-bit/2-bit | low-rank | 3-bit module 分布 | PPL 记录 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `custom_56_swap_l16w2_l19w2` | 2.4002 | 189.0 MiB | 56/104 | 104 | `w2:16, w3:8, wo:32` | W2-512 22.6492<br>W2-256 23.7345 |
| `custom_56_swap_l16w2_l22w2` | 2.4002 | 189.0 MiB | 56/104 | 104 | `w2:16, w3:8, wo:32` | W2-256 23.8176 |
| `custom_56_swap_l16w2_l20w2` | 2.4002 | 189.0 MiB | 56/104 | 104 | `w2:16, w3:8, wo:32` | W2-256 23.8434 |
| `custom_57_swap_l16w2_l22w2` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-256 23.5287 |
| `custom_57_swap_l16w2_l20w2` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-256 23.5711 |
| `custom_57_swap_l17w2_l22w2` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-512 22.4349<br>W2-256 23.5193 |
| `custom_57_swap_l18w2_l22w2` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-256 23.5858 |
| `custom_57_swap_l19w2_l22w2` | 2.4073 | 194.9 MiB | 57/103 | 103 | `w2:17, w3:8, wo:32` | W2-256 23.6791 |
| `custom_58_swap_l16w2_l22w2` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-512 22.2984<br>W2-256 23.3534 |
| `custom_58_swap_l20w2_l22w2` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-256 23.3953 |
| `custom_58_swap_l19w2_l22w2` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-256 23.4971 |
| `custom_58_swap_l20w2_l24w3` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:17, w3:9, wo:32` | W2-256 23.4586 |
| `custom_58_swap_l20w2_l26w3` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:17, w3:9, wo:32` | W2-256 23.4620 |
| `custom_58_best_swap_l17w2_l16w2` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-256 23.3555 |
| `custom_58_best_swap_l18w2_l16w2` | 2.4144 | 200.8 MiB | 58/102 | 102 | `w2:18, w3:8, wo:32` | W2-256 23.4084 |

#### 59-62 精炼候选实际 bit

| 实验/cache | 实际等效 bit | 相对 `mixed_0.0` 实际增量 | 3-bit/2-bit | low-rank | 3-bit module 分布 | PPL 记录 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `custom_59_from_best58_add_l23w2` | 2.4214 | 206.6 MiB | 59/101 | 101 | `w2:19, w3:8, wo:32` | W2-512 22.1449<br>W2-256 23.1974 |
| `custom_59_from_best58_add_l2w2` | 2.4214 | 206.6 MiB | 59/101 | 101 | `w2:19, w3:8, wo:32` | W2-256 23.2036 |
| `custom_59_from_best58_add_l24w3` | 2.4214 | 206.6 MiB | 59/101 | 101 | `w2:18, w3:9, wo:32` | W2-256 23.2320 |
| `custom_59_from_best58_add_l24w2` | 2.4214 | 206.6 MiB | 59/101 | 101 | `w2:19, w3:8, wo:32` | W2-256 23.2338 |
| `custom_60_from_best59_l23_l2w2` | 2.4285 | 212.5 MiB | 60/100 | 100 | `w2:20, w3:8, wo:32` | W2-512 21.9463<br>W2-256 23.0317 |
| `custom_60_from_best59_l23_l16w2` | 2.4285 | 212.5 MiB | 60/100 | 100 | `w2:20, w3:8, wo:32` | W2-256 23.0660 |
| `custom_60_from_best59_l23_l24w3` | 2.4285 | 212.5 MiB | 60/100 | 100 | `w2:19, w3:9, wo:32` | W2-256 23.0669 |
| `custom_60_from_best59_l23_l24w2` | 2.4285 | 212.5 MiB | 60/100 | 100 | `w2:20, w3:8, wo:32` | W2-256 23.0746 |
| `custom_61_from_best60_l23w3` | 2.4355 | 218.4 MiB | 61/99 | 99 | `w2:20, w3:9, wo:32` | W2-512 21.8344<br>W2-256 22.9020 |
| `custom_61_from_best60_l24w3` | 2.4355 | 218.4 MiB | 61/99 | 99 | `w2:20, w3:9, wo:32` | W2-256 22.9156 |
| `custom_61_from_best60_l9w3` | 2.4355 | 218.4 MiB | 61/99 | 99 | `w2:20, w3:9, wo:32` | W2-256 22.9928 |
| `custom_62_from_best61_l16w2` | 2.4426 | 224.2 MiB | 62/98 | 98 | `w2:21, w3:9, wo:32` | W2-512 21.7138<br>W2-256 22.7795 |
| `custom_62_from_best61_l24w2` | 2.4426 | 224.2 MiB | 62/98 | 98 | `w2:21, w3:9, wo:32` | W2-512 21.6932<br>W2-256 22.7876 |
| `custom_62_from_best61_l24w3` | 2.4426 | 224.2 MiB | 62/98 | 98 | `w2:20, w3:10, wo:32` | W2-512 21.7074<br>W2-256 22.7880 |
| `custom_62_from_best61_l9w3` | 2.4426 | 224.2 MiB | 62/98 | 98 | `w2:20, w3:10, wo:32` | 尚未产生有效 PPL 记录 |

#### 按实际 bit 重新理解成本

1. `mixed_0.0` 已经不是 2.0000 bit，而是 2.1731 bit，因为所有 160 个目标 linear 都保留了 4-bit SVD residual。
2. `mixed_0.3` 的实际等效 bit 为 2.3963，`custom_55` 为 2.3932；因此 55 层在实际 bit 口径下确实略低于 `mixed_0.3`，但差距只有约 2.6 MiB，结合 PTB 结果仍然不够稳。
3. `custom_56_swap_l16w2_l19w2` 为 2.4002 bit，略高于 `mixed_0.3`，但 Wikitext2-512 PPL 低约 0.36；这是小幅增存换明显质量收益。
4. `custom_57_swap_l17w2_l22w2` 为 2.4073 bit，相比 `mixed_0.3` 多约 9.1 MiB，但 Wikitext2-512 PPL 低约 0.57，是更稳的低成本甜点。
5. `custom_58_swap_l16w2_l22w2` 为 2.4144 bit，相比 `mixed_0.3` 多约 15.0 MiB，而 Wikitext2-512 PPL 低约 0.71；它不再是严格同成本，但仍是很强的单位成本收益点。
6. `custom_wo_all_w2_w3_62` 为 2.4426 bit，比 `mixed_0.4` 的 2.5041 bit 少约 51.2 MiB，同时 Wikitext2-512 PPL 只高约 0.0925；这是当前最强的高性能省成本候选。
7. 旧的“净存储变化”仍可用于解释搜索期为什么某些策略看起来接近同成本，但正式报告成本时应使用本节的实际等效 bit。

### 每日更新模板

后续记录可以使用以下模板。

```markdown
## YYYY-MM-DD

### 主题

### 新证据

### 决策

### 风险或开放问题

### 下一步行动
```

## 2026-06-10

### 主题

补充验证 `feed_forward.w1` 提升 bit 的真实效果，判断它是否应该进入当前敏感层保护策略。

目前已有结论认为 `w1` 不是低成本优先保护层，但这个判断主要来自间接证据：`w1` 在 `linear_score` 中进入较晚，且当前不含 `w1` 的自定义 Pareto 候选已经很强。为了避免遗漏 `w1` 的真实 PPL 贡献，今天需要做一组更直接的同成本实验。

### 实验目标

本轮实验要回答三个问题：

1. 最高分的 `w1`，尤其 `L3:w1` 和 `L2:w1`，是否能在同成本下替代边界 `w2` 或 `w3`。
2. `w1` 是完全不值得保护，还是只是在低成本区间优先级低、到了中高预算才开始有价值。
3. 如果 `w1` 有收益，它应该替代哪类边界层：后段 `w2`、后段 `w3`，还是只适合作为额外新增层。

### 基础证据

从 `linear_score_map` 看，最高分 `w1` 的排序为：

```text
rank 64: L3:w1
rank 65: L2:w1
rank 82: L4:w1
rank 85: L1:w1
rank 87: L5:w1
```

对照其他 module 的进入时机：

```text
第一个 wo: rank 1
第一个 w3: rank 9
第一个 w2: rank 33
第一个 w1: rank 64
第一个 wqkv: rank 98
```

因此本轮优先测试 `L3:w1` 和 `L2:w1`。`L4:w1` 可以作为补充实验，但不应在第一批占用太多 GPU 时间。

为了放大 `w1` 的测试效果，第一批不从 62 层高预算点开始。62 层附近已经保护了大量 `wo/w2/w3`，上下文更接近饱和，`w1` 的边际作用可能被稀释。更合理的做法是先在 32-48 层低中预算区间做压力测试：如果 `w1` 在这个区间能替代边界 `w2/w3`，说明它应该更早进入候选池；如果在这个区间也不能赢，再去 62 层只做确认即可。

当前几个低中预算母版的组成如下：

```text
custom_wo_w2_32:       wo 17, w2 15, w3 0,  w1 0
custom_wo_w2_w3_40:    wo 17, w2 15, w3 8,  w1 0
custom_wo_w2_w3_44:    wo 17, w2 19, w3 8,  w1 0
custom_wo_all_w2_w3_48: wo 32, w2 8,  w3 8,  w1 0
```

当前 62 层首选 `custom_wo_all_w2_w3_62` 的组成是：

```text
wo: 32
w2: 22
w3: 8
w1: 0
```

其中比较适合作为同成本替换对象的边界层包括：

```text
边界 w2: L24:w2, L20:w2, L16:w2, L17:w2
边界 w3: L25:w3, 以及必要时测试 L2:w3 / L8:w3
```

### 第一批实验：32-48 层低中预算压力测试

第一批固定总 3-bit 层数在 32-48 之间，优先测试 `w1` 是否能提前替代边界 `w2/w3`。这一批比 62 层更适合放大 `w1` 效果，因为此时许多后段敏感层还没有被保护，单个替换的 PPL 影响会更清楚。

优先生成以下 cache：

| 新 cache 名称 | 母版 | 变更 | 目的 |
| --- | --- | --- | --- |
| `custom_32_swap_l18w2_l3w1` | `custom_wo_w2_32` | 去掉 `L18:w2`，加入 `L3:w1` | 测试最高分 `w1` 能否替代 32 层预算中最边界的 `w2`。 |
| `custom_32_swap_l18w2_l2w1` | `custom_wo_w2_32` | 去掉 `L18:w2`，加入 `L2:w1` | 测试第二高分 `w1` 在低预算下是否有竞争力。 |
| `custom_40_swap_l25w3_l3w1` | `custom_wo_w2_w3_40` | 去掉 `L25:w3`，加入 `L3:w1` | 测试 `w1` 是否比 40 层预算中边界 `w3` 更值得保护。 |
| `custom_40_swap_l25w3_l2w1` | `custom_wo_w2_w3_40` | 去掉 `L25:w3`，加入 `L2:w1` | 验证 `w1` 替代边界 `w3` 是否稳定。 |
| `custom_44_swap_l22w2_l3w1` | `custom_wo_w2_w3_44` | 去掉 `L22:w2`，加入 `L3:w1` | 用更强的边界 `w2` 做压力测试；若能赢，说明 `w1` 很可能被低估。 |
| `custom_44_swap_l25w3_l3w1` | `custom_wo_w2_w3_44` | 去掉 `L25:w3`，加入 `L3:w1` | 测试 `w1` 是否至少能替代同预算中的边界 `w3`。 |
| `custom_48_swap_l12w2_l3w1` | `custom_wo_all_w2_w3_48` | 去掉 `L12:w2`，加入 `L3:w1` | 在全 `wo` 底座下测试 `w1` 是否能替代低预算边界 `w2`。 |
| `custom_48_swap_l25w3_l3w1` | `custom_wo_all_w2_w3_48` | 去掉 `L25:w3`，加入 `L3:w1` | 在全 `wo` 底座下测试 `w1` 是否能替代边界 `w3`。 |

判断标准：

1. 如果 `w1` 在 32/40/44/48 任一同成本替换中明显降低 PPL，说明 `w1` 不是单纯高预算候选，应纳入后续低中预算搜索。
2. 如果 `w1` 只能替代边界 `w3`，不能替代边界 `w2`，说明 `w1` 的优先级可能介于后段 `w3` 和关键 `w2` 之间。
3. 如果 `w1` 连边界 `w3` 都替代不了，说明它在低中预算区间确实不是优先保护对象。
4. 如果 32-48 层全部失败，则 62 层实验只作为补充确认，不必大规模展开。

优先评估口径：

```text
DATASET=wikitext2
N_SAMPLES=256
```

如果 256 samples 下某个 `w1` 替换候选比对应母版至少低 0.03 PPL，再用 512 samples 复核。若差距小于 0.01 PPL，先视为噪声区，不急于下结论。

### 第二批实验：62 层高预算复核

如果第一批显示 `w1` 有潜力，第二批再固定 62 个 3-bit 层，使用当前 75.0 MiB / 2.4426 actual-bit 档位做高预算复核。母版建议使用：

```text
internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_wo_all_w2_w3_62.pt
```

候选 cache：

| 新 cache 名称 | 变更 | 目的 |
| --- | --- | --- |
| `custom_62_swap_l24w2_l3w1` | 去掉 `L24:w2`，加入 `L3:w1` | 测试最高分 `w1` 能否替代最边界的 `w2`。 |
| `custom_62_swap_l24w2_l2w1` | 去掉 `L24:w2`，加入 `L2:w1` | 测试第二高分 `w1` 是否接近或超过边界 `w2`。 |
| `custom_62_swap_l25w3_l3w1` | 去掉 `L25:w3`，加入 `L3:w1` | 测试 `w1` 是否比最低优先级已选 `w3` 更有价值。 |
| `custom_62_swap_l25w3_l2w1` | 去掉 `L25:w3`，加入 `L2:w1` | 验证 `w1` 替代边界 `w3` 的稳定性。 |

判断标准：

1. 如果低中预算失败但 62 层成功，说明 `w1` 更像高预算补充层，而不是早期敏感层。
2. 如果低中预算和 62 层都成功，说明 `w1` 被当前策略系统性低估，应加入后续 Pareto 搜索。
3. 如果低中预算成功但 62 层失败，说明 `w1` 的价值可能依赖预算和上下文，需要围绕成功预算点继续局部搜索。

### 第三批实验：60 层中预算补充替换

如果第一批显示 `w1` 有潜力，但 62 层结果不清楚，可以继续在 60 层预算验证它是否只在某些中预算上下文有效。

母版建议使用：

```text
internvl2_8b_w2g32_scale_reweight_true_svd_1.0_custom_60_from_best59_l23_l2w2.pt
```

候选 cache：

| 新 cache 名称 | 变更 | 目的 |
| --- | --- | --- |
| `custom_60_swap_l16w2_l3w1` | 去掉 `L16:w2`，加入 `L3:w1` | 测试 `w1` 能否替代 60 层附近的边界 `w2`。 |
| `custom_60_swap_l24w3_l3w1` | 如果候选里存在 `L24:w3` 路线，则去掉 `L24:w3`，加入 `L3:w1` | 测试 `w1` 与中预算后段 `w3` 的优先级。 |

如果第一批中 `w1` 没有表现出任何优势，第二批和第三批都可以暂缓，不必消耗 GPU。

### 第四批实验：新增层上限测试

同成本替换能回答“`w1` 是否值得挤掉别人”，但不能回答“额外增加 `w1` 的绝对收益上限”。因此可以做少量新增层实验，只用于估计上限，不作为成本公平比较的主证据。

候选 cache：

| 新 cache 名称 | 变更 | 目的 |
| --- | --- | --- |
| `custom_63_from_62_add_l3w1` | 在 `custom_wo_all_w2_w3_62` 基础上加入 `L3:w1` | 估计最高分 `w1` 的绝对收益上限。 |
| `custom_64_from_62_add_l3_l2w1` | 在 `custom_wo_all_w2_w3_62` 基础上加入 `L3:w1` 和 `L2:w1` | 估计两个最高分 `w1` 的合计收益上限。 |

这组实验不能直接和 62 层同成本 swap 混为一谈，因为它增加了实际 bit 成本。它的作用只是判断：如果 `w1` 额外加入都几乎没有收益，那么后续就不值得为 `w1` 做更多替换实验。

### cache 生成方式

所有自定义 cache 仍然必须从 `mixed_0.0.pt` 生成，不能从已有非零 keep-ratio cache 反向修改。生成后必须检查：

```text
linear_bit_map 中 3-bit 层数量符合预期
所有新增的 w1 层已经从 low_rank 删除
被 drop 的层恢复为 2-bit，并保留 low_rank residual
low_rank_count = 160 - 3bit_layer_count
```

可以复用 `tools/build_custom_mbq_cache.py` 的 `layer_list` 或 `--drop-layer/--add-layer` 方式。推荐先基于原始策略生成候选，再用 drop/add 做同成本替换。

### PPL 执行规范

每次运行前先查 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
```

统一环境：

```bash
cd /home/users/Mayinyi/jikuixie/new-MBQ
export PYTHONPATH=.
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export HF_ENDPOINT=https://hf-mirror.com
```

256 samples 快速筛选命令模板：

```bash
GPU_ID=<显存最空的GPU编号> DATASET=wikitext2 N_SAMPLES=256 SCALE_FILE_NAME=<目标cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

512 samples 复核命令模板：

```bash
GPU_ID=<显存最空的GPU编号> DATASET=wikitext2 N_SAMPLES=512 SCALE_FILE_NAME=<目标cache文件名> bash tests/test_PPL/2_compute_quant_ppl.sh
```

### 预期结论形式

本轮实验结束后，应把 `w1` 归入以下三类之一：

```text
A. w1 明显有效: 至少一个同成本 w1 替换稳定优于原始候选，需要纳入后续 Pareto 搜索。
B. w1 只在新增层时有效: 不适合挤掉当前边界层，但高预算时可以作为候选。
C. w1 暂不值得保护: 同成本替换和新增层收益都很弱，继续维持低优先级。
```

当前先验判断是 C 或 B，但实验应允许 A 出现；尤其 `L3:w1` 和 `L2:w1` 在 rank 64/65 附近，已经接近 `mixed_0.4` 的进入边界，值得用少量 GPU 时间确认。

### 第一批实验执行记录

本轮只执行第一批 32-48 层低中预算压力测试，评估口径均为：

```text
DATASET=wikitext2
N_SAMPLES=256
```

所有 cache 均从 `mixed_0.0.pt` 生成，并完成一致性检查：

```text
3-bit 层数量与目标预算一致
新增 w1 层已从 low_rank 删除
被替换掉的层恢复为 2-bit，并保留 low_rank residual
low_rank_count = 160 - 3bit_layer_count
```

#### PPL 结果

| 实验 | 母版 | 替换 | PPL | 相对母版 | bit/low-rank | 初步判断 |
| --- | --- | --- | ---: | ---: | --- | --- |
| `custom_32_swap_l18w2_l3w1` | `custom_wo_w2_32` 25.5038 | `L18:w2 -> L3:w1` | 25.5982 | +0.0944 | 32 个 3-bit, lr=128 | 明显弱于母版。 |
| `custom_32_swap_l18w2_l2w1` | `custom_wo_w2_32` 25.5038 | `L18:w2 -> L2:w1` | 25.3959 | -0.1079 | 32 个 3-bit, lr=128 | 明显优于母版，值得 512 复核。 |
| `custom_40_swap_l25w3_l3w1` | `custom_wo_w2_w3_40` 24.6909 | `L25:w3 -> L3:w1` | 24.6860 | -0.0049 | 40 个 3-bit, lr=120 | 基本打平，噪声区。 |
| `custom_40_swap_l25w3_l2w1` | `custom_wo_w2_w3_40` 24.6909 | `L25:w3 -> L2:w1` | 24.6760 | -0.0149 | 40 个 3-bit, lr=120 | 小幅优于母版，暂不单独下结论。 |
| `custom_44_swap_l22w2_l3w1` | `custom_wo_w2_w3_44` 23.9151 | `L22:w2 -> L3:w1` | 23.9815 | +0.0664 | 44 个 3-bit, lr=116 | 明显弱于母版；`w1` 不能替代强 `L22:w2`。 |
| `custom_44_swap_l25w3_l3w1` | `custom_wo_w2_w3_44` 23.9151 | `L25:w3 -> L3:w1` | 23.8839 | -0.0312 | 44 个 3-bit, lr=116 | 明显优于母版，值得 512 复核。 |
| `custom_48_swap_l12w2_l3w1` | `custom_wo_all_w2_w3_48` 24.8853 | `L12:w2 -> L3:w1` | 24.8502 | -0.0351 | 48 个 3-bit, lr=112 | 明显优于母版，值得 512 复核。 |
| `custom_48_swap_l25w3_l3w1` | `custom_wo_all_w2_w3_48` 24.8853 | `L25:w3 -> L3:w1` | 24.8410 | -0.0443 | 48 个 3-bit, lr=112 | 明显优于母版，值得 512 复核。 |

#### 初步观察

1. `w1` 不是整体无效。第一批 8 个同成本替换里，有 4 个点优于母版，其中 3 个超过了 0.03 PPL 的快筛阈值。
2. `L2:w1` 在 32 层预算下表现很强：替代 `L18:w2` 后 PPL 降低约 0.1079。这说明 `w1` 内部存在明显层差异，不能只看 module 大类。
3. `L3:w1` 在 40 层替代 `L25:w3` 基本打平，在 44/48 层替代 `L25:w3` 明显变好，说明 `L3:w1` 很可能优于部分边界 `w3`。
4. `L3:w1` 不能替代强 `L22:w2`，替换后 PPL 变差约 0.0664。因此当前仍不能把 `w1` 放在强敏 `w2` 之前。
5. `custom_48_swap_l12w2_l3w1` 优于母版这一点值得注意：在全 `wo` 底座、低 `w2` 数量的上下文中，某些早中层 `w2` 可能不如 `L3:w1`。但这个结论目前只基于 256 samples，需要复核。
6. 当前更准确的 `w1` 定位应从“低优先级”修正为：“不适合替代强 `w2`，但可能优于边界 `w3`，并且 `L2:w1/L3:w1` 需要进入低中预算候选池”。

#### 下一步建议

优先做 512 samples 复核以下三个明显正向点：

```text
custom_32_swap_l18w2_l2w1
custom_44_swap_l25w3_l3w1
custom_48_swap_l25w3_l3w1
```

`custom_48_swap_l12w2_l3w1` 也值得复核，但由于它是 `w1` 替代 `w2` 的反常正向结果，建议排在前三个之后，避免过早相信 256-sample 的单点波动。

第二批 62 层高预算复核可以暂缓，先确认低中预算的正向点是否在 512 samples 下保持。如果 512 复核仍然成立，再把 `L2:w1/L3:w1` 纳入 55-62 层 Pareto 搜索。
