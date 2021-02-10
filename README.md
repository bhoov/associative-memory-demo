# Associative Memory Demo

Explore how thinking of AI as an Associative Memory task simplifies and generalizes many of the desirable aspects of modern Machine Learning.

A demo created with [Pluto.jl](https://github.com/fonsp/Pluto.jl).

Math and inspiration taken from the [Hopfield Networks is All You Need Blog](https://ml-jku.github.io/hopfield-layers/)


## Getting Started
To use this interactive notebook, you'll have to use [Julia](https://julialang.org/downloads/) >= 1.5. Thankfully, this language is rapidly growing in popularity and is designed to be simple to code in for scientists and mathematicians.

A 6 min video on how to do this [here](https://www.youtube.com/watch?v=OOjKEgbt8AI)

1. Download the [latest version of Julia](https://julialang.org/downloads/). Follow the default instructions for MacOS
2. Open the newly installed `julia-1.5.x`. This should open a terminal with a julia instance.
3. Install [Pluto.jl](https://github.com/fonsp/Pluto.jl). Follow the instructions on that repo, or below:
    - Press the `[` key in the terminal. You are now in the package environment.
    - Type `add Pluto`. This will take a moment to download.
    - Backspace out of the package manager
    - `import Pluto`
    - `Pluto.run()`

This will open a Jupyter-like interface that will allow you to browse to `notebook.jl`.

Julia has a longer start up time and is slow the first time you run a cell. This is because the code you write is immediately compiled, and this allows it to run at near C-speeds which is important for the interactivity.

The environment is self contained in the notebook. It will take a long time to startup.

## Troubleshooting

#### Hangs on Startup
If this is the first time you are running the notebook, it is possible that Julia is trying to download MNIST and is asking you for a prompt on the command line. Unfortunately, Pluto has an issue interpreting the STDIN in the workers. 

To fix this, exit Pluto, and run the following within Julia:

``` julia
import Pkg; Pkg.add("MLDatasets");
using MLDatasets
MNIST.traindata()
```

Start pluto again
