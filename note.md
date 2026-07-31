# 相比原始 MBQ 的改进点

当前版本不是简单复现原始 MBQ，而是在 MBQ 的 scale reparameterization 基础上，围绕 VLM 的 W2 weight-only 量化加入了混合精度、SVD 残差补偿和 residual rank 压缩分析。核心改进可以概括为：

```text
原始 MBQ: scale search + weight-only / W-A quantization
当前版本: modality-aware scale search
        + activation-aware residual scoring
        + mixed-bit selection
        + quantization-residual SVD compensation
        + module-aware residual-rank compression
```

## 1. 更细粒度的多模态 scale search

原始 MBQ 已经考虑多模态 reweight，但当前版本进一步细化了 reweight 粒度。

当前支持：

```text
attn_in   -> attention.wqkv
attn_out  -> attention.wo
mlp_in    -> feed_forward.w1 / feed_forward.w3
mlp_out   -> feed_forward.w2
```

也就是说，不再只用粗粒度的 `attn/mlp` ratio，而是把 attention 输入、attention 输出、MLP 输入、MLP 输出分开处理。

同时，scale search 支持多种 activation statistic：

```text
global
token_weighted
modality_mean
auto
```

并支持不同 loss reduction：

```text
legacy
token_weighted
modality_mean
```

这使得 scale search 可以更明确地服务于 VLM 的 text/vision token 分布，而不是只优化全局 token 平均误差。

## 2. Activation-aware 的 SVD residual compensation

当前版本加入了基于量化残差的 SVD 补偿。对一个 linear 层：

```text
W_q = Q(W_fp)
R   = W_fp - W_q
R   ~= Up @ Down
```

推理时实际使用：

```text
y = x @ W_q^T + x @ (Up @ Down)^T
```

也就是：

```text
quantized base weight + low-rank residual
```

这个残差不是对原始权重直接做 SVD，而是对 `apply_scale` 之后、按当前 `w_bit / w_group / double_quant` 得到的量化残差做 SVD。因此 SVD 补偿和最终量化配置是一致的。

## 3. SVD 候选层使用 modality-reweighted activation-aware residual error

候选层不是随便选，也不是只按权重误差排序，而是看 activation-aware residual error：

```text
R = W_fp - W_q

text_score = ||X_text @ R^T||^2 / ||X_text @ W_fp^T||^2
vis_score  = ||X_vis  @ R^T||^2 / ||X_vis  @ W_fp^T||^2

score = text_score + reweight_ratio * vis_score
```

在当前配置中：

```yaml
reweight: true
reweight_group: true
```

所以 SVD 候选层排序实际使用的是 modality-reweighted activation-aware residual error。`reweight_group` 打开后，不同模块族使用不同的 ratio：

```text
attention.wqkv  -> attn_in
attention.wo    -> attn_out
feed_forward.w1 -> mlp_in
feed_forward.w3 -> mlp_in
feed_forward.w2 -> mlp_out
```

这点可以作为方法贡献的一部分：SVD residual 并不是均匀加在所有层上，而是由多模态激活误差驱动选择。

## 4. Mixed-bit selection 与 low-rank residual 互斥

当前版本支持 `linear_mixed_probe`，用同一套 activation-aware score 选择部分敏感 linear 提升到更高 bit，例如：

```text
默认层: 2-bit base weight + low-rank residual
敏感层: 3-bit base weight
```

关键设计是互斥：

```text
如果某层被提升到 3-bit，则从 low_rank residual 列表中删除。
```

这样避免同一层同时使用：

```text
3-bit base weight + SVD residual
```

否则会导致成本统计不公平，也会让机制解释变得混乱。

## 5. SVD factor 量化

当前版本不是把 SVD residual 因子按 fp16 成本计算，而是支持对 `Up/Down` 做 pseudo quantization。

当前配置是：

```yaml
svd_quant: true
svd_quant_config:
  up:
    weight_quant: per_channel
    quant_bit: 4
  down:
    weight_quant: per_channel
    quant_bit: 4
```

因此 actual-bit 统计中，SVD residual 按 4-bit factor 计入，而不是按 fp16 计入。这使得 SVD 补偿在 W2 setting 下仍有合理的存储成本。

## 6. Double quantization

当前版本在 `pseudo_quantize_tensor` 中加入了 double quantization，对 primary quantization scales 再做低比特模拟量化。

当前配置：

```yaml
double_quant: true
double_quant_config:
  scale_bit: 4
  scale_group: 32
  zero_point: true
```

这会同时影响：

```text
W_q 的数值
SVD residual R = W_fp - W_q
最终 actual-bit 成本估计
```

因此 SVD residual 是针对 double-quant 后的实际量化误差构建的。

## 7. Module-aware residual-rank compression

最近实验进一步验证：不同模块族的 SVD residual rank 可压缩性不同。

目前观察到的规律：

```text
wqkv: rank64 安全；rank32 视母版可用；rank16 普遍不安全
w1:   rank64 基本安全
w3:   rank64 安全，rank32 在部分设置下可用
wo:   在 42 线 rank32 可用
w2:   强依赖母版；55 线 rank64 可用，42 线不安全
```

这说明 residual rank 不应该统一设置，而应该按 module family 分配。

这部分更像实验分析贡献，可以表述为：

```text
module-aware residual-rank allocation / residual-rank compression
```

## 可写成论文贡献的版本

可以把当前工作相对 MBQ 的改进浓缩成三点：

1. **Modality-reweighted activation-aware residual scoring**  
   用 text/vision token 上的 residual output error 评估每个 linear 的量化敏感性，并根据模块族使用不同 reweight ratio。

2. **Mixed-bit and low-rank residual compensation framework**  
   对敏感层使用更高 bit，对非敏感层使用 2-bit base weight + quantized SVD residual，并保证 high-bit layer 与 low-rank residual 互斥。

3. **Module-aware SVD residual rank allocation**  
   系统分析不同模块族的 residual rank 下界，发现 `wqkv/w1/w3/wo/w2` 的可压缩性不同，并据此构造更优 actual-bit/OCRBench Pareto 点。

一句话总结：

```text
在 MBQ 的多模态 scale search 基础上，我们进一步引入 activation-aware 的量化残差 SVD 补偿、score-guided mixed-bit selection，以及 module-aware residual-rank compression，使 VLM W2 weight-only 量化在更低 actual-bit 下保持更高 OCRBench 精度。
```
