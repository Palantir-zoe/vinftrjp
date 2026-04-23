# Change-Point RJFlow-VINF 方法部分中文初稿

## 1. 模型定义

设观测到的事件时刻为
\[
y=\{t_1,\dots,t_n\}\subset [0,L].
\]
我们考虑带有 \(k\) 个 change points 的分段齐次 Poisson 过程模型，其中
\[
k\in\{0,1,\dots,k_{\max}\}.
\]

给定 \(k\) 后，区间 \([0,L]\) 被划分为 \(k+1\) 个子区间。记各段长度为
\[
l_1,\dots,l_{k+1},\qquad l_i>0,\qquad \sum_{i=1}^{k+1} l_i = L.
\]
进一步定义归一化段长
\[
w_i = \frac{l_i}{L},\qquad w=(w_1,\dots,w_{k+1})\in \Delta_k,
\]
其中 \(\Delta_k\) 表示 \((k+1)\)-维概率单纯形。于是 change points 可写为
\[
\tau_j = L \sum_{i=1}^{j} w_i,\qquad j=1,\dots,k.
\]

每一段对应一个常数强度参数 \(h_i>0\)，因此模型参数可记为
\[
(k, w, h),\qquad h=(h_1,\dots,h_{k+1}).
\]

## 2. 先验分布

### 2.1 模型指标先验

我们对 \(k\) 指定截断 Poisson 先验：
\[
p(k)=\frac{\operatorname{Poisson}(k;\lambda)}{\sum_{j=0}^{k_{\max}}\operatorname{Poisson}(j;\lambda)},
\qquad k=0,\dots,k_{\max}.
\]

### 2.2 段长先验

我们令归一化段长向量服从 Dirichlet 先验：
\[
w \mid k \sim \operatorname{Dirichlet}(2,\dots,2).
\]
该先验保证每一段长度严格为正，且总长度恰为 \(L\)。

### 2.3 段内强度先验

对每个区间强度独立指定 Gamma 先验：
\[
h_i \mid k \stackrel{\text{i.i.d.}}{\sim} \operatorname{Gamma}(\alpha,\beta),
\qquad i=1,\dots,k+1,
\]
其中本文采用 shape-rate 参数化，即
\[
p(h_i)=\frac{\beta^\alpha}{\Gamma(\alpha)}h_i^{\alpha-1}e^{-\beta h_i},\qquad h_i>0.
\]

## 3. 无约束重参数化

为了在欧氏空间上训练 normalizing flow 并构造可逆跳跃提议，我们将约束参数映射到无约束空间。

### 3.1 Stick-breaking 重参数化

记
\[
s=(s_1,\dots,s_k)\in\mathbb{R}^k.
\]
定义
\[
z_i = \sigma\!\bigl(s_i-\log(k+1-i)\bigr),\qquad i=1,\dots,k,
\]
其中 \(\sigma(x)=1/(1+e^{-x})\) 为 logistic 函数。随后定义
\[
w_1 = z_1,\qquad
w_i = z_i\prod_{j=1}^{i-1}(1-z_j)\ \ (i=2,\dots,k),\qquad
w_{k+1}=\prod_{j=1}^{k}(1-z_j).
\]
该映射将 \(\mathbb{R}^k\) 一一映射到 \(\Delta_k\) 的内部。

### 3.2 强度的对数重参数化

对每个强度参数令
\[
r_i=\log h_i,\qquad h_i=e^{r_i},\qquad r_i\in\mathbb{R}.
\]

于是，固定 \(k\) 时的无约束参数可写为
\[
\theta_k=(s_1,\dots,s_k,r_1,\dots,r_{k+1})\in\mathbb{R}^{2k+1}.
\]

## 4. 无约束空间中的先验密度

由变量变换公式，
\[
\log p(\theta_k\mid k)
=
\log p(w(s)\mid k)
+ \log \left|\det \frac{\partial w}{\partial s}\right|
+ \sum_{i=1}^{k+1}\Bigl[\log p(h_i=e^{r_i}) + r_i\Bigr].
\]

### 4.1 Dirichlet 项

由于
\[
w\mid k \sim \operatorname{Dirichlet}(2,\dots,2),
\]
故
\[
\log p(w\mid k)
=
\log \Gamma\!\Bigl(\sum_{i=1}^{k+1}\alpha_i\Bigr)
- \sum_{i=1}^{k+1}\log\Gamma(\alpha_i)
+ \sum_{i=1}^{k+1}(\alpha_i-1)\log w_i,
\]
其中 \(\alpha_i=2\)。

### 4.2 Stick-breaking Jacobian

令
\[
\eta_i=s_i-\log(k+1-i),\qquad z_i=\sigma(\eta_i).
\]
则 Jacobian 的对数绝对值为
\[
\log \left|\det \frac{\partial w}{\partial s}\right|
=
\sum_{i=1}^{k}
\left[
-\eta_i + \log \sigma(\eta_i) + \log w_i
\right].
\]

### 4.3 Gamma-log 变换项

对 \(h_i=e^{r_i}\) 使用变量变换公式，有
\[
\log p(r_i)
=
\log p(h_i=e^{r_i}) + r_i.
\]
因此
\[
\sum_{i=1}^{k+1}\log p(r_i)
=
\sum_{i=1}^{k+1}\left[
\alpha\log\beta - \log\Gamma(\alpha)
+ (\alpha-1)r_i - \beta e^{r_i} + r_i
\right].
\]

## 5. 似然函数

给定参数 \((k,w,h)\) 后，第 \(i\) 段长度为
\[
l_i = L w_i.
\]
记该段中观测到的事件数为 \(N_i\)。由于每段为齐次 Poisson 过程，联合似然可写为
\[
\log p(y\mid k,w,h)
=
\sum_{i=1}^{k+1}\left[
N_i \log h_i - l_i h_i
\right].
\]

于是目标后验为
\[
\pi(k,\theta_k\mid y)
\propto
p(k)\,p(\theta_k\mid k)\,p(y\mid k,\theta_k).
\]

## 6. 基于模型特异 flow 的 RJMCMC 提议

对于每个模型 \(k\)，我们训练一个 normalizing flow
\[
T_k : z_k \mapsto \theta_k,
\qquad z_k \in \mathbb{R}^{2k+1},
\]
以近似该模型下的后验分布。该 flow 仅用于构造高效提议，不改变目标后验分布。

### 6.1 模型内提议

令当前状态为 \(\theta_k\)，其对应的 base 坐标为
\[
z_k = T_k^{-1}(\theta_k).
\]
模型内提议在 base 空间采用对称高斯随机游走：
\[
z_k' = z_k + \varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,\sigma_w^2 I).
\]
再通过 flow 变换回参数空间：
\[
\theta_k' = T_k(z_k').
\]
由于 base 空间中提议核对称，Metropolis-Hastings 比值中的 proposal 修正仅由 flow 的 Jacobian 给出。

### 6.2 跨模型提议

跨模型提议仅允许 \(k\to k\pm1\)。从当前模型 \(k\) 跳向 \(k'\) 时，先将 \(\theta_k\) 映射到 base 空间，再对新增维度引入辅助变量
\[
u \sim \mathcal{N}(0,\sigma_u^2 I).
\]
对于需要关闭的维度，则在反向提议密度中补回其高斯密度项。随后将扩展后的 base 向量通过目标模型对应的 flow \(T_{k'}\) 映射回参数空间。这样可保证前向和反向提议在维度上匹配，并满足 RJMCMC 的可逆性要求。

### 6.3 接受率

记 \((k,\theta)\) 为当前状态，\((k',\theta')\) 为提议状态，则接受概率为
\[
\alpha\bigl((k,\theta),(k',\theta')\bigr)
=
\min\left\{
1,\,
\frac{\pi(k',\theta'\mid y)}{\pi(k,\theta\mid y)}
\frac{q(k,\theta\mid k',\theta')}{q(k',\theta'\mid k,\theta)}
\right\}.
\]
由于目标密度与提议比值均被显式计算，算法在理论上保持详细平衡。

## 7. 实现细节与当前代码中的严格一致性

### 7.1 当前实现所保证的严格一致性

当前实现位于 [src/problems/change_point.py](/e:/Desktop/vinftrjp/src/problems/change_point.py:1) 与 [src/algorithms/change_point_models/change_point_model.py](/e:/Desktop/vinftrjp/src/algorithms/change_point_models/change_point_model.py:1)。其中与理论一致性的关键点如下：

1. 固定 \(k\) 时，段长参数 \(w\)、Dirichlet 先验项和 stick-breaking Jacobian 现在都由同一套 log-space stick-breaking 表达式计算，而不是由两套不同的 simplex 数值表示拼接得到。
2. Dirichlet 对数密度不再通过数值验证器调用，而是直接由
   \[
   \log p(w)=\log \Gamma\!\Bigl(\sum_i \alpha_i\Bigr)-\sum_i \log\Gamma(\alpha_i)+\sum_i (\alpha_i-1)\log w_i
   \]
   在 log-space 中计算，因此与重参数化公式完全一致。
3. likelihood 中使用 `xlogy` 计算 \(N_i\log h_i\)，从而在 \(N_i=0\) 且 \(h_i\) 极小的极限下仍保持数学上正确的值，而不会产生数值上的 `nan`。
4. `compute_prior` 与 `compute_llh` 已统一在 `float64` 下评估，从而减少 simplex 边界与极端 rate 下的舍入误差。

### 7.2 数值稳定性与理论一致性的关系

本文实现中，stick-breaking 权重不是通过直接累乘得到，而是先在 log-space 中写成
\[
\log w_i
\]
的解析表达，再通过稳定的 `softmax(log_weights)` 恢复 \(w\)。由于理论上
\[
\sum_{i=1}^{k+1} e^{\log w_i} = 1,
\]
因此该计算是对同一理论变换的稳定求值，而不是对目标分布的额外修改。换言之，这一步属于数值实现上的稳定重写，而非模型定义上的近似替代。

### 7.3 当前实现可支持的理论表述

基于上述修改，当前代码已经可以支撑如下论文表述：

1. 目标后验由截断 Poisson 模型先验、Dirichlet 段长先验、Gamma 强度先验和分段 Poisson process 似然共同定义。
2. 固定模型维度 \(k\) 后，无约束参数空间中的目标密度通过严格的变量变换公式获得。
3. RJMCMC 接受率中使用的目标密度与提议密度比值均与实现中的数值计算保持一致。
4. 因此，在标准的可测性与遍历性条件下，该算法保持针对目标后验分布的正确不变性；normalizing flow 仅改变提议效率，而不改变目标分布本身。

## 8. 建议在论文中保留的说明

为了让理论部分更稳妥，建议在论文实现细节或附录中保留以下说明：

1. 所有涉及 simplex 的计算均在 log-space 中实现，以避免 stick-breaking 末端段长在有限精度下出现下溢。
2. 目标密度评估采用双精度浮点数，以降低 Jacobian 与先验评估中的数值误差。
3. normalizing flow 只用于提议构造，不参与目标密度定义，因此不会改变后验分布。

## 9. 仍可进一步加强的验证

如果论文需要更强的实现层证据，建议补充以下实验或附录材料：

1. 对若干固定 \(k\) 的 prior draws 验证 `draw -> compute_prior` 的数值一致性。
2. 对模型内 flow 提议验证 `transformToBase -> transformFromBase` 的往返误差处于机器精度量级。
3. 对小规模数据集进行长链采样，并检查不同初始化下的后验模型概率是否一致。
4. 若篇幅允许，可在附录中给出 stick-breaking Jacobian 的完整推导。
