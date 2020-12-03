# intrinsic-dimension-projections
Jonathan Lam & Richard Lee

Based on [this paper][1] and [its associated codebase][2].

### General goals
- Rewrite/upgrade/modernize to Python3/TF2 (e.g., type hinting for better IDE support, do more linting)
- Include better documentation/explanations of terms than original code
- Apply other projection types (other than random linear mappings); in particular, try random fourier features (RFF)

### Current TODO
- Finish implementation of basic projection model
- Test, make sure results are similar to original paper
- Implement RFF projection
- Write report

[1]: https://arxiv.org/abs/1804.08838
[2]: https://github.com/uber-research/intrinsic-dimension
