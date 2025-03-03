# Getting Started with Julia

Julia is a scientific programming language like Matlab or Python. It provides users with a simple and interpretable syntax but C-like runtimes. 

I use Julia for most of my daily research and algorithm development, and I have found that using VSCode is a great way to interact with Julia.

## 1. Install Visual Studio Code (VSCode)
If you have not already, install VSCode for your operating system from the [following link](https://code.visualstudio.com/download). 

## 2. Install Julia
Next, install the correct version of Julia for your operating system from the [following link]( https://julialang.org/downloads)

## 3. Configure VSCode for Julia
To do this, install the Julia extension in VSCode. This can be done by clicking `View >> Extensions`. You will have to restart VSCode after this.

After this, you are ready to go! For more details on configuring VSCode for Julia and running Julia code in VSCode, the following links are useful:

1. VSCode Documentation: https://code.visualstudio.com/docs/languages/julia
2. Julia VSCode extension: https://www.julia-vscode.org/docs/stable/userguide/runningcode/
3. Julia Documentation : https://docs.julialang.org/en/v1/

## Running Julia from the Command Line
You may run Julia from the command line. This can be done in one of two ways. The first uses the read-eval-print loop (REPL) that serves as an interactive Julia session. You may launch the REPL using the following command line hit 
```console
foo@bar:~$ julia
``` 
This works similar to Python in the sense that you can define variables, import packages, run code blocks, etc.

The other option is to execute a pre-written Julia file from the command line. This would look like 
```console
foo@bar:~$ julia myscript.jl
``` 
which will execute the code within `myscript.jl` and then exit. 

## Running These Notebooks
If it is your first time cloning this repo and running these notebooks, you first need to install the necessary Julia packages. To do so, first enter the Julia REPL
```console
foo@bar:~$ julia
``` 
Then, enter the package manager by typing `]` in the REPL
```console
julia> ] 
``` 
Activate the current environment using the following 
```console
(@v1.xx) pkg> activate .
``` 
Here the `@v1.xx` just indicates the base environment for whatever version of Julia you have installed. These notebooks were created using Julia 1.10.2, but later versions should probably work. Feel free to contact Jens Rataczak if it doesn't. Once the environment is activated, you should see something like
```console
(folder_name) pkg> 
``` 
where `folder_name` is the current folder into which you have cloned this repo. Finally, install the necessary packages using the following 
```console
(folder_name) pkg> instantiate
``` 
Julia should then install the exact version of all dependencies specified in `Manifest.toml`. Once this is complete, you should be ready to run all the code in these notebooks. The package manager in Julia can be exited by hitting backspace until the normal Julia REPL prompt shows.

## Overview of Tutorials
This repository contains three separate convex-optimization tutorials. It is recommended that they be studied in the following order:

1. `convex_tutorial` provides an introduction to `Convex.jl` and how it can be used to solve simple optimization problems. Examples of both unconstrained and constrained optimization problems are given.
2. `ptr_demo` provides an implementation of the Penalized Trust Region (PTR) sequential convex programming (SCP) algorithm. It applies the PTR algorithm to a simple two-dimensional, deterministic system.
3. `stochastic_demo` expands the PTR algorithm to consider uncertainty in the system directly. It solves for a sequence of feed-forward controls and feedback gain matrices for the same system in demo 2, however, the constraints are now modeled probabilistically using chance constraints.
