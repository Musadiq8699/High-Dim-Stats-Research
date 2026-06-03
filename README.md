# High-Dimensional Statistics & Sparsity Research Log

## Literature Synthesis Matrix

| Citation Key | Core Methodology | Main Advantage | Critical Limitation / Gap |
| :--- | :--- | :--- | :--- |
| `tibshirani1996regression` | L1 Regularization Penalty | Simultaneous variable selection and coefficient shrinkage. | Inherent estimation bias; unstable selection with highly correlated features. |

### Foundational Comparison of Regulators
* **Ordinary Least Squares (OLS):** Zero training bias, but suffers from massive variance (overfitting) and absolute zero interpretability in high dimensions ($p > n$).
* **Best-Subset Selection:** Highly interpretable due to strict feature deletion, but computationally explosive ($O(2^p)$ combinations) and incredibly unstable under minor data perturbations.
* **Ridge Regression ($L_2$):** Highly stable under multi-collinearity, but relies on a "rubber-band" penalty ($2\lambda\beta$) that shrinks weights smoothly without ever hitting absolute zero—leaving a heavy background noise fingerprint and full inference overhead.
## Daily Research Insights
* **June 4, 2026:** Decoded the mathematical mechanics of Lasso’s absolute force vs. Ridge’s variable force. Documented the structural breakdown of accumulated prediction noise and computational deployment overhead caused by non-zero weights.
