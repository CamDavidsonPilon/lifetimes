# On reproducing this locally

If you fork/clone the library locally, also install it locally so you can create your own alterations and then perhaps contribute to the main project:

```bash
pip install -e .
```

This will overwrite the current `lifetimes` installation also. Any changes you make you automatically occur to the Python import, so you don't need to install it everytime you change something.

# Internal Folders

The `images` folder is supposed to have the images produced by the several `benchmarks` routines. However, the images won't be tracked by `git`. If you wish to get the images, simply run the scripts. (You will be able to see the relevant ones in the docs pages eventually.)