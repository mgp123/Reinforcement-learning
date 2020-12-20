import numpy as np


def pytorch_use_numpy_as_input(module_class):
    class Res(module_class):
        def forward(self, x):
            x = np.asarray(x)
            return module_class.forward(x)
    return Res



