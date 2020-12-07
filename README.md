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

### Style guide
Placing a little bit of extra emphasis here because the codebase from the previous project was a little hard to decipher:
- Attempt to adhere to PEP8 as best as possible (use IntelliJ linting/code refactor when possible)
- Type hints in parameter lists/other initializations when possible to allow for maximum automated inferencing
- Comment at beginning of each file to describe what the file is and defining relevant terms
- Clear variable names (clarity at the cost of longer variable names)
- Comments describing every function (except obvious cases and some overloaded methods) for maximum clarity

[1]: https://arxiv.org/abs/1804.08838
[2]: https://github.com/uber-research/intrinsic-dimension
