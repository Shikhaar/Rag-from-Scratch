# Modern Neural Network Optimization & Attention Mechanics

## 1. AdamW Optimizer Formulation
Standard Adam couples weight decay with gradient updates, leading to suboptimal regularization when combined with adaptive momentum. Ilya Loshchilov and Frank Hutter introduced AdamW, which decouples weight decay from the gradient step:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Where $\lambda$ is the decoupled weight decay factor, avoiding the distortion of L2 penalty caused by second-moment scaling $\sqrt{\hat{v}_t}$.

## 2. FlashAttention Mechanics
Standard Multi-Head Attention incurs $O(N^2)$ memory IO overhead because it materializes full $N \times N$ attention matrix $S = Q K^T$ into high-bandwidth memory (HBM).

FlashAttention achieves significant speedup and exact mathematical equivalence through:
1. **Tiling / Block computation**: Loading blocks of Queries, Keys, and Values into fast SRAM (20TB/s) on the GPU chip.
2. **Online Softmax**: Computing running softmax normalizers $(\max, \sum \exp)$ incrementally across blocks without materializing intermediate scores in HBM.
3. **Recomputation during Backward Pass**: Storing only output activations and softmax statistics, recomputing attention matrix blocks on-the-fly during backprop to save GPU memory bandwidth.

## 3. Mixture of Experts (MoE) Routing
In Sparse Mixture-of-Experts architectures (e.g. Mixtral 8x7B), feed-forward layers are replaced by $E$ distinct expert subnetworks. A learnable router or gating network computes a softmax distribution over top-$K$ selected experts (typically $K=2$ out of 8). This increases total model capacity to tens of billions of parameters while keeping active computational FLOPs equivalent to a much smaller dense model.
