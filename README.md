# fa_kit
Factor Analysis Kit version 0.1.1

I've re-written various factor analysis tools for my own use several times over the years, so I'm trying to come up with a single factor analysis package that does everything I want. See also, [this](https://bmcmenamin.github.io/2017/09/12/releasing-fa-kit.html).

This is compatible with Python 2 and 3 on my Mac test machine, but due to the use of TensorFlow for factor rotations it requireqs Python 3 on Windows machines.

Directory layout:
* `./fa_test` has the main package
* `./examples` has Jupyter notebooks that demonstrate how to use the package
* `./tests` contains pytest unit tests

# Installation
You can install it using pip like this: `pip install fa_kit`
