#!/bin/bash
# Find the DATA_FOLDER and navigate to it.
DATA_FOLDER_PATH=$(grep DATA_FOLDER config.py | cut -d '=' -f 2 | tr -d '[:space:]')
echo "$DATA_FOLDER_PATH"
eval "cd $DATA_FOLDER_PATH"
# Check if the cd command was successful, else exit.
if [ $? -eq 0 ]; then
    echo "Changed to DATA_FOLDER and starting downloading data files."
else
    echo "Failed to change directory to $DATA_FOLDER_PATH. Does it exist?"
    exit
fi
# Start downloading.
wget https://zenodo.org/record/8379988/files/embeddings.zip
unzip embeddings.zip
rm embeddings.zip
wget https://zenodo.org/record/8379988/files/docs.zip
unzip docs.zip
rm docs.zip
wget https://zenodo.org/record/8379988/files/overview.csv
echo "Finished downloading the data."
