## Research Foundations & Core Milestones

Our current research vectors center on the deconstruction of regularized linear architectures, specifically evaluating the parameters introduced by Robert Tibshirani's foundational 1996 manuscript. 

For the comprehensive, academically rigorous synthesis of these proofs, equations, and optimization engines, see our dedicated **[Standalone Literature Review Matrix](./LITERATURE_REVIEW.md)**.

### Architectural Core at a Glance

* **Geometric Budget Constraints:** The $L_1$ penalty constraint forms a sharp diamond polytope centered at the coordinate origin. This sharp geometry forces expanding Residual Sum of Squares (RSS) error contours to collide directly along the feature axes, driving noise weights to absolute zero.
* **The Orthonormal Special Case ($X^T X = I$):** Under conditions of perfect feature independence and unit scaling, Lasso collapses into a one-step closed-form shortcut known as the **Soft Thresholding Operator**.
* **The Optimization Bottleneck:** Standard gradient descent is incapable of optimizing Lasso due to non-differentiable corners at $\beta_j = 0$. This repository implements iterative **Cyclic Coordinate Descent** to optimize partial residuals one variable at a time across successive epochs.

---



## Daily Research Insights

* **June 4, 2026:** Initiated high-dimensional statistics literature review focusing on Tibshirani (1996). Decoded the foundational trade-offs between Ordinary Least Squares (OLS), Best-Subset Selection, and Regularization methods, noting the structural limitations of non-zero weight footprints regarding background noise accumulation.
* **June 5, 2026:** Analyzed the algebraic mechanics of the Lasso objective function. Documented how the shorthand prediction variable ($\hat{y}_i$) unrolls into its explicit feature-weight matrix inside the residual brackets, mapping how the penalty acts directly on the parameters.
* **June 6, 2026:** Explored the geometric proofs of sparsity. Mapped the physical interaction between expanding data error contours ($\text{RSS}$) and the regularized budget boundaries ($t$), contrasting the sharp, axis-aligned vertices of the $L_1$ diamond against the smooth tangent shoulders of the $L_2$ circle.
* **June 7, 2026:** Scaled the geometric model to high-dimensional sparse spaces ($p=45$). Established the complete 5-step Machine Learning Engineering Pipeline, detailing how a leakage-free $K$-Fold Cross-Validation routine isolates optimal $\lambda$ parameters ($\lambda_{\min}$ and $\lambda_{1\text{SE}}$) empirically without needing analytical Degrees of Freedom (d).

* **June 8, 2026:** Conducted exhaustive mathematical deconstruction of Section 2.2 (Orthonormal Design Case). Mapped the analytical operations of Hard Thresholding, Proportional Scaling, and Soft Thresholding. Isolated the breakdown mechanics of standard gradient descent on non-differentiable corners and documented how the Coordinate Descent engine uses partial residuals to deploy closed-form shortcuts cyclically across optimization epochs.
