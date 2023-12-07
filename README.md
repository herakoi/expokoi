# expokoi

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

## License
Copyright 2022 Michele Ginolfi, Luca Di Mascolo, and contributors.

herakoi is a free software made available under the MIT License. For details see the LICENSE file.
