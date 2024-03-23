#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: generate_parquet
Description: Writes .parquet feature files for DeepProfiler output

Usage: 
      Requires:
        Params:
            -a projectfolder in which plates are located in output/features folder
Author: Benjamin Frey
Created: Nov. 16th 2023
Last Updated: [Last updated date]
Version: 0.1
Dependencies: 
    Requires previous feature extraction with DeepProfiler 
"""
import os
import numpy as np
import pandas as pd
import polars as pl
import time
import tqdm
import click 

@click.command()
@click.option('--projectfolder', '-o', type=click.Path(), prompt=True,
              help='The path to the DeepProfiler project folder where features are located.')

def main(projectfolder):
    start_time = time.time()
    try:
        metadata = prep_metadata(projectfolder)

        with tqdm.tqdm(total = len(metadata["Metadata_Plate"].unique()), desc="Analysing plates") as progress:
            parquet_df = create_parquet(projectfolder, metadata)
            progress.update()

        end_time = time.time()
        execution_time = end_time - start_time
        minutes = execution_time // 60
        seconds = execution_time % 60
        print(f"The script executed in {int(minutes)} minutes and {seconds:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def create_parquet(projectfolder: str, metadata):
    plate_names = find_plate_names(projectfolder)
    
    for plate in plate_names:
        print("Merging plate ", plate)
        validation = metadata[metadata["Metadata_Plate"] == plate].reset_index()
        if len(validation) < 600:
            batch_size = len(validation)
        else:
            batch_size = 600
        master_df = pl.DataFrame()
# Process in batches and append to the master DataFrame
        for i in range(0, len(validation), batch_size):
            batch = [(validation.loc[j, "Metadata_Plate"],
                    validation.loc[j, "Metadata_Well"],
                    validation.loc[j, "Metadata_Site"],
                    f"{projectfolder}/outputs/results/features/{validation.loc[j, 'Metadata_Plate']}/{validation.loc[j, 'Metadata_Well']}/{validation.loc[j, 'Metadata_Site']}.npz")
                    for j in range(i, min(i + batch_size, len(validation)))]
            batch_df = process_batch(batch)
            master_df = pl.concat([master_df, batch_df])
        meta_pl = pl.DataFrame(metadata)
        merged_features = master_df.join(
                meta_pl,
                on=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site'],  # columns to join on
                how='inner'  # you can change this to 'left', 'right', 'outer' as per your need
            )
        output_folder = os.path.join(projectfolder, "outputs", "results", "parquets")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Write the DataFrame to Parquet
        master_df.write_parquet(os.path.join(output_folder, f"{plate}_sc_features.parquet"))


def process_batch(batch):
    rows = []
    for item in batch:
        plate, well, site, filename = item
        try:
            with open(filename, "rb") as data:
                info = np.load(data)
                features_array = info["features"]
                for features in features_array:
                    row = {"Metadata_Plate": plate, "Metadata_Well": well, "Metadata_Site": site}
                    for idx, feature in enumerate(features):
                        row[f"Feature_{idx}"] = feature
                    rows.append(row)
        except FileNotFoundError:
            continue
    return pl.DataFrame(rows)

def find_plate_names(folder_path):
    # Direct path to the 'outputs/features' subfolder
    features_folder = os.path.join(folder_path, "outputs", "results", "features")

    # Check if the 'features' folder exists
    if not os.path.exists(features_folder):
        return "The 'outputs/features' folder does not exist in the given path."

    # List to hold the names of subfolders
    subfolder_names = []

    # Iterate over the items in the 'features' folder
    for item in os.listdir(features_folder):
        item_path = os.path.join(features_folder, item)
        # Check if the item is a folder
        if os.path.isdir(item_path):
            subfolder_names.append(item)

    return subfolder_names


def prep_metadata(projectfolder):
    metadata_folder = os.path.join(projectfolder, "inputs", "metadata")
    if not os.path.exists(metadata_folder):
        return None, "The 'inputs/metadata' folder does not exist in the given path."

    # Find the CSV file in the 'metadata' folder
    for file in os.listdir(metadata_folder):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(metadata_folder, file)
            # Read the CSV file into a pandas DataFrame
            metadata = pd.read_csv(csv_file_path)
    meta = metadata.sort_values(by=['Metadata_Well', 'Metadata_Site'])
    meta['Metadata_cmpdName'] = meta['Metadata_cmpdName'].str.upper()
    meta["Metadata_cmpdNameConc"] = meta["Metadata_cmpdName"] +   " " + meta["Metadata_cmpdConc"].astype(str)    
    #meta.rename(columns={"Metadata_Plate": "Plate", "Metadata_Well": "Well", "Metadata_Site": "Site"}, inplace=True)
    return meta    
    




# Main execution
if __name__ == "__main__":
    main()