# expokoi

Yet another version of [`herakoi`](https://github.com/lucadimascolo/herakoi), this time optimized for public events and exhibitions

## Installation
The sound synthesis in `expokoi` makes use of a SoundFont2 specifications and requires an external synthesis driver, `fluidsynth`.
If you are running `expokoi` within a `conda` environment, you can install `fluidsynth` by running

```
conda install -c conda-forge fluidsynth
```

If you are on Windows, you can also try with the installer on the [`fluidsynth` release](https://github.com/FluidSynth/fluidsynth/releases) page.

Then, to install `expokoi`: 

```
git clone https://github.com/herakoi/expokoi.git
cd expokoi
python -m pip install -e .
```

or directly

```
python -m pip install git+https://github.com/herakoi/expokoi.git
```


### Common issues
We observed that the `opencv-python` library has some compatibility issues with older operating systems (e.g., earlier than MacOS 11 Big Sur in the case of Apple machines). In such a case, installing a version of `opencv-python` earlier than `4.0` seems to solve the issue:

```
python -m pip install --force-reinstall "opencv-python<4"
```

## License
Copyright 2022 Michele Ginolfi, Luca Di Mascolo, and contributors.

herakoi is a free software made available under the MIT License. For details see the LICENSE file.
