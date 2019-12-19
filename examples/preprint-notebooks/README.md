# Introducing AFQ-Insight

This directory contains notebooks and other supporting files to
reproduce figures in the AFQ-Insight preprint. It includes applications
in the study of amyotrophic lateral sclerosis (ALS) and aging.

## Prerequisites

### Docker-based installation

We recommend that you use the supplied docker image to reproduce the results of this paper.
To build the docker image, type
```bash
make docker-build
```

After that, you can run the jupyter notebooks in this directory by typing
```bash
make docker-lab
```
and then navigating to the URL supplied in the output of that command. When you're done,
type <kbd>Ctrl</kbd>+<kbd>C</kbd> (twice to confirm) to return to your host machine shell.

If you want to obtain an interactive shell in the Docker image, type
```bash
make docker-shell
```

When you're done, type `exit` to return to your host machine shell.

### Data

To download the source data, type
```bash
make data
```

## Running the notebooks

Running the notebooks from start to finish on the raw data can take some time.
You may wish to do this for reproducibility purposes. If you'd like to skip this
and start from intermediate results, the authors are happy to supply these
intermediate values via private correspondence.
