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