# Processing Dataset

This repository contains code for downloading raw data and performing transformations. Follow the instructions below to execute the processing script and input your desired dataset name. If you want to process all datasets, enter `all`.

## Instructions

1. Run the following command to start the data processing:

   ```sh
   sh process.sh
   ```

2. You will see the following message with a list of available datasets:

   ```sh
   Available datasets: ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M']
   Enter the name of the dataset (or type 'all' to process all datasets):
   ```

3. Enter the name of the dataset you want to process, or type all to process all datasets.

## Acknowledgement

The code base used for processing datasets is available at: [CIKM2020-S3Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master).

Please note that we could not reproduce the results as the Yelp raw data used in the code base could not be located. In our experiments, we utilized pre-processed data provided in the code base. We will update this document if we succeed in reproducing the Yelp data results.