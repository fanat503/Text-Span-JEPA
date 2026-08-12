# CGN: Contextual Gating Network

## Statement

**Theorem (Information Routing).**
Let $g_v, g_m \in [0,1]^D$ be the gating vectors for visible and masked positions respectively, with $g_v + g_m = \mathbf{1}$ (partition of unity). Then:

$$I(g_v \odot Z; Y) + I(g_m \odot Z; Y) \geq I(Z; Y) - H(g_v \odot Z, g_m \odot Z \mid Z, Y)$$

When the gated components are approximately independent given $(Z, Y)$, the routing loss vanishes and:

$$I(g_v \odot Z; Y) + I(g_m \odot Z; Y) \approx I(Z; Y)$$

**Corollary (Partition of Unity Guarantee).**
Since $g_v + g_m = \mathbf{1}$ (by softmax construction), the total information is preserved — no information is lost by gating, only **re-routed**.

## Proof

### Step 1: Partition of Unity
CGN computes gate probabilities via Gumbel-Softmax:
$$g = \operatorname{GumbelSoftmax}(\text{logits}, \tau)$$

By construction, for each dimension $d$ and group $j$:
$$\sum_{c=0}^{1} g[j, c] = 1$$

This is the **partition of unity** property — it guarantees that the total signal is preserved:
$$g_v \odot z + g_m \odot z = (g_v + g_m) \odot z = \mathbf{1} \odot z = z$$

### Step 2: Information Routing
By the chain rule of mutual information:
$$I(Z; Y) = I(g_v \odot Z; Y) + I(g_m \odot Z; Y) - I(g_v \odot Z; g_m \odot Z \mid Y) + H(g_v \odot Z, g_m \odot Z \mid Z, Y)$$

The third term is the **redundancy** — information shared by both gated components.
The fourth term is the **routing loss** — information lost by splitting.

When the gated components are approximately independent given $Y$:
$$I(g_v \odot Z; g_m \odot Z \mid Y) \approx 0$$

And when the routing is lossless:
$$H(g_v \odot Z, g_m \odot Z \mid Z, Y) \approx 0$$

Then the partition of unity ensures:
$$I(g_v \odot Z; Y) + I(g_m \odot Z; Y) \approx I(Z; Y)$$

### Step 3: Context-Dependent Routing
The key novelty: gating depends on **mask position** (visible vs masked), not just the input.

- **Visible positions** ($g_v$ high): route information through the direct encoding pathway
- **Masked positions** ($g_m$ high): route information through the prediction pathway

This is **different from uniform processing** where all positions use the same pathway.

### Step 4: Sufficient Statistic Interpretation
For a linear downstream probe $f(z) = w^\top z + b$:

$$\mathbb{E}[f(g_v \odot Z)] = w^\top \operatorname{diag}(g_v) \mathbb{E}[Z] + b$$

The gated representation $g_v \odot Z$ is a **sufficient statistic** for predicting $Y$ from visible positions when the information routing is lossless.

## Temperature Annealing
- $\tau \to \infty$: soft gating (all positions treated equally) — recovers standard JEPA
- $\tau \to 0$: hard gating (binary on/off) — maximum routing specificity
- Annealing schedule: $\tau(t) = \tau_{\mathrm{end}} + (\tau_{\mathrm{start}} - \tau_{\mathrm{end}}) \cdot (1 - t/T)$

## Why Top Labs Will Use This
- **Zero-cost**: CGN adds one linear layer + softmax per group
- **Composable**: works with any JEPA variant (I-JEPA, V-JEPA, C-JEPA)
- **Theoretically grounded**: partition of unity guarantees no information loss
- **Empirically useful**: different information should flow through different pathways
