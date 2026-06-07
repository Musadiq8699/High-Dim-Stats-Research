# High-Dimensional Statistics & Sparsity Research Log

## Literature Synthesis Matrix

| Citation Key | Core Methodology | Main Advantage | Critical Limitation / Gap |
| :--- | :--- | :--- | :--- |
| `tibshirani1996regression` | **L1 Regularization Penalty:** Adds an absolute weight constraint [lambda * sum(abs(beta))] to the loss function. Enforces an axis-aligned hyper-diamond budget space. | **Simultaneous Variable Selection & Shrinkage:** Enforces true sparsity by catching expanding error contours right on its sharp multidimensional vertices, driving noisy coefficients to absolute zero. | **Collinearity Instability:** If features are highly correlated (e.g., adjacent pixel blocks), the geometric error ellipsoid becomes a flat tunnel, causing Lasso to select one feature arbitrarily. |

### Foundational Comparison of Regulators
* **Ordinary Least Squares (OLS):** Achieves zero training bias by minimizing raw Residual Sum of Squares (RSS). However, it suffers from massive variance (overfitting) and becomes completely mathematically indeterminate in high-dimensional settings where the number of features exceeds the observations ($p > n$).
* **Best-Subset Selection:** Highly interpretable due to strict feature deletion, but computationally explosive ($O(2^p)$ combinations), making it completely unscalable for high-dimensional setups and highly unstable under minor data perturbations.
* **Ridge Regression ($L_2$):** Highly stable under multi-collinearity. It relies on a quadratic penalty ($\lambda \sum \beta_j^2$) that shrinks weights smoothly toward zero like a rubber band. However, it is physically incapable of hitting absolute zero—leaving a heavy background noise fingerprint and full computational inference overhead.

---

## Deep-Dive Technical Insights

### 1. The Core Optimization Equation
The Lasso optimization framework unrolls the traditional conceptual definition of RSS into its full mathematical components inside the error-minimization loop:

$$\text{Minimize} \quad \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

By expanding the shorthand prediction parameter ($\hat{y}_i$) into its explicit feature-weight breakdown ($\beta_0 + \sum \beta_j x_{ij}$), the formula shows exactly how the negative sign distributes across the weights inside the squaring brackets, exposing the target variables that the $L_1$ penalty actively constraints.

### 2. Geometric Mechanics of Sparsity ($L_1$ vs. $L_2$ Boundaries)
The fundamental divergence between Lasso and Ridge properties comes down to the sharp contours of their budget boundaries ($t$) interacting with the data's loss ellipses:

* **The Loss Ellipses (RSS Contours):** The unconstrained OLS solution sits as a single point (the minimum error valley). The red rings expanding outward from this point represent growing training error (introducing bias). The optimization algorithm must find where these expanding error rings first make contact with the legal penalty boundary.
* **The Lasso Diamond ($L_1$):** The absolute value constraint ($|\beta_1| + |\beta_2| \le t$) draws a sharp diamond whose sharp vertices sit perfectly on the coordinate axes. As the smooth, rounded loss contour expands outward from the OLS center, it mathematically tends to collide with these sharp protruding corners first. Because these contact points lie directly on the grid lines, the overlapping coordinate weight drops to **absolute zero**, yielding automated feature selection.
* **The Ridge Circle ($L_2$):** The squared constraint ($\beta_1^2 + \beta_2^2 \le t$) generates a smooth, cornerless circle. Expanding loss contours make contact with this circular boundary along its rounded, slanted shoulders. Because this tangent intersection occurs in open space away from the axes, both features retain small, non-zero decimal values, preventing true sparsity.

### 3. Behavior in High-Dimensional Sparse Environments ($p = 45$, $2$ Signal Features)
When scaled to high dimensions, the Lasso budget constraint turns into a multi-dimensional hyper-diamond (cross-polytope). In a dataset where 43 features are pure random noise and only 2 hold true signal:
* The training error contour stretches out into long, flat, insensitive valleys along the 43 noise dimensions, while remaining tightly compressed and sensitive along the 2 signal dimensions.
* Increasing the penalty parameter ($\lambda$) shrinks the hyper-diamond toward the origin.
* The data's error contour rolls down from the OLS point, sweeping past the flat noise dimensions, and hits the sharp vertices belonging exclusively to the 2 high-signal axes. The geometry acts as a physical sorting machine, freezing the 43 noise weights flat against the axis wall at exactly zero.

---

## The Hyperparameter Optimization Pipeline

The penalty parameter ($\lambda$) balances a two-player game between data fit ($\text{RSS}$) and complexity ($\text{Penalty}$). We isolate the optimal balance point using a structured, leakage-free empirical engineering pipeline:

```text
[Raw Dataset] ──► Split (80% Train/Validation, 20% Holdout Test)
                     │
                     ▼
             [Train/Validation Set] ──► Centering & Scaling (Standardization)
                     │
                     ▼
             [10-Fold Cross-Validation Loop]
                     ├── Iterate over log-scale Lambda Grid [0.001 to 1000]
                     ├── Evaluate Validation MSE empirically across folds
                     └── Identify Lambda_Min or Lambda_1SE via CV Error Curve
                     │
                     ▼
             [Final Training Pass] ──► Train once on full 80% pool using optimal Lambda
                     │
                     ▼
             [Holdout Evaluation] ──► Unvault 20% Test Set for unbiased generalization check
