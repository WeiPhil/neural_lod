### NumPy/SciPy ###

- Tested with Python 3.8.3, NumPy 1.19.0, and SciPy 1.5.0.
- `flip.py` computes the FLIP map between two images. It's called from the `main.py` function,
  where you may load the reference and test images. Input images are assumed to be in sRGB space and in the [0,1] range.
- `utils.py` includes functions for loading and saving images.
- The default test and reference images are found in the `images` folder.
- The FLIP output is saved to the `images` folder.