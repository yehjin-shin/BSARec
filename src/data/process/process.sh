#!/bin/bash

available_datasets="Beauty Sports_and_Outdoors Toys_and_Games Yelp LastFM ML-1M"

echo "Available datasets: ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M']" # TODO: Yelp
echo "Enter the name of the dataset (or type 'all' to process all datasets):"
read dataname

sh _download.sh "$dataname"
python _transform.py "$dataname"