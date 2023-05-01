#!/bin/bash

pip install -r requirements.txt
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jax jaxlib