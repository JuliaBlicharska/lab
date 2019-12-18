# Installing miniconda

Download miniconda installer (~70Mb):

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

(this one is for 64bit linux)


Then run the script:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

this will install miniconda by default in `$HOME/miniconda3` location. Confirm
all questions (installer guides through the process; it is convenient to install
shell integration). Log-out and log-in so that shell configuration takes place.

Create venv using conda:
```
conda create -n cvenv pip
```

this will install pip in the virtual env. `cvenv` is name of virtual env.
Activate it using:

```
conda activate cvenv
```

Now you can use python and pip normally.

Deactivate using:

```
conda deactivate cvenv
```
