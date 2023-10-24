# expokoi 

Yet another version of [`herakoi`](https://github.com/lucadimascolo/herakoi), this time optimized for public events and exhibitions

## Installation
To install `expokoi`, you can simply do the following: 

```
git clone https://github.com/herakoi/expokoi.git
cd expokoi
python -m pip install -e .
```

### Common issues
We observed that the `opencv-python` library has some compatibility issues with older operating systems (e.g., earlier than MacOS 11 Big Sur in the case of Apple machines). In such a case, installing a version of `opencv-python` earlier than `4.0` seems to solve the issue:

```
python -m pip install --force-reinstall "opencv-python<4"
```