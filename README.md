# intrinsic-dimension-projections
Jonathan Lam & Richard Lee

Based on [this paper][1] and [its associated codebase][2].

### General goals
- Apply other projection types (other than random linear mappings); in particular, try random fourier features (RFF)

### Completed goals
- Rewrite/upgrade/modernize to Python3/TF2 (e.g., type hinting for better IDE support, do more linting)
- Include better documentation/explanations of terms than original code

### TODO
- Implement Conv2D, DepthwiseConv2D (maybe? for MobileNet architecture)
- Implement RFF projection
- Write report
- Experiment with training epochs vs. intrinsic dimension (can this be related to double descent paper?)
- Experiment with initialization (glorot_uniform seems to work well) and regularization methods
- Experiment with different intrinsic weights for different layers?
- Implement slightly-modified for inference? (i.e., hardcode the weights matrix so it doesn't have to be recalculated each time)

[1]: https://arxiv.org/abs/1804.08838
[2]: https://github.com/uber-research/intrinsic-dimension
