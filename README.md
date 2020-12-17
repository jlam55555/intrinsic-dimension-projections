# intrinsic-dimension-projections
Jonathan Lam & Richard Lee

See [the paper][3] and [the presentation][4] associated with this research.

Based on ["Measuring the intrinsic dimension of objective landscapes"][1] and [its associated codebase][2].

### Repository contents
- `keras_ext`: main TF2/keras low-level architectural files (projection implementation)
- `models`: model architectures
- `plots`: sample plots used in report (scripts to generate these plots are not included)
- `runs`: sample pickled runs (not a complete list of models used in the report)
- `scripts`: scripts to automate model building and training

### Usage
The primary model for training was MNIST, and the training loop is in [scripts/mnist.py][5]. There, you can adjust the model hyperparameters:
- `epochs`: maximum number of epochs before termination (ignoring early stopping)
- `intrinsic_dims`: iterable of intrinsic dimension sizes to train
- `initializers`: initializers for the initial weight matrices theta_0
- `lrs`: set of learning rates to train with
- `model_types`: list of model types to train; can choose from "linear", "power", and "rff"
- `normalize_p`: whether to normalize the output basis vectors in the projection matrix

### General project goals
- Apply other projection types (other than random linear mappings); in particular, try random fourier features (RFF) and power terms
- Rewrite/upgrade/modernize to Python3/TF2 (e.g., type hinting for better IDE support, do more linting)
- Include better documentation/explanations of terms than original code

### Style guide
Placing a little bit of extra emphasis here because the codebase from the previous project was a little hard to decipher:
- Attempt to adhere to PEP8 as best as possible (use IntelliJ linting/code refactor when possible)
- Type hints in parameter lists/other initializations when possible to allow for maximum automated inferencing
- Comment at beginning of each file to describe what the file is and defining relevant terms
- Clear variable names (clarity at the cost of longer variable names)
- Comments describing every function (except obvious cases and some overloaded methods) for maximum clarity

[1]: https://arxiv.org/abs/1804.08838
[2]: https://github.com/uber-research/intrinsic-dimension
[3]: http://files.lambdalambda.ninja/reports/20-21_fall/ece472_intrinsic_dimension_projections.pdf
[4]: http://files.lambdalambda.ninja/reports/20-21_fall/ece472_intrinsic_dimension_projections_presentation.pdf
[5]: ./scripts/mnist.py
