#!/bin/bash

# copy models and create a duplicate model folder

cp -r ./models ./models-copy

# remove recrusively any model.pth file from that copy 

find ./models-copy -name "*.pth" -type f -delete


# compress the model folder
tar -czvf models-copy.tar.gz models-copy

# remove the copied model folder
rm -rf models-copy