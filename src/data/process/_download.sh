#!/bin/bash

DATASET_NAME="$1"
mkdir -p raw
cd raw

if [ "$DATASET_NAME" = "all" ]; then
    cd ..
    sh _download.sh Beauty            
    sh _download.sh Sports_and_Outdoors            
    sh _download.sh Toys_and_Games            
    sh _download.sh LastFM            
    sh _download.sh ML-1M 
    # sh _download.sh Yelp           

elif [ "$DATASET_NAME" = "Beauty" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz -O Beauty.json.gz
    gzip -d Beauty.json.gz

elif [ "$DATASET_NAME" = "Sports_and_Outdoors" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz -O Sports_and_Outdoors.json.gz
    gzip -d Sports_and_Outdoors.json.gz

elif [ "$DATASET_NAME" = "Toys_and_Games" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz -O Toys_and_Games.json.gz
    gzip -d Toys_and_Games.json.gz

elif [ "$DATASET_NAME" = "LastFM" ]; then
    wget https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip -O ./LastFM.zip
    mkdir -p LastFM
    unzip LastFM.zip -d ./LastFM
    rm LastFM.zip

elif [ "$DATASET_NAME" = "ML-1M" ]; then
    wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O ./raw_data/ML-1M.zip
    unzip ./raw_data/ML-1M.zip
    rm ./raw_data/ML-1M.zip

else   
    echo "Invalid dataset name"
    echo "Available datasets: ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M']" # TODO: Yelp
    echo "To download all datasets at once, enter 'all' as the dataset name."
    echo ""
fi

