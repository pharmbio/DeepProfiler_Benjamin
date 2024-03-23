#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: deepprofiler_dataprep
Description: Execute this script with an experiment in mind to generate folder structure and metadata to run DeepProfiler

Usage: 
      Requires:
        Params:
            -a projectfolder (ideally already created)
            -a path to a .csv containing Plate information (Treatments etc.)
            -a path to a checkpoint file (if not given, uses default pre-trained)
            -projectname that matches those for CellProfiler (for Quality Control using pharmbio package)
        Works using .parquet files from CellProfiler analysis:
            PREVIOUS CELLPROFILER EXTRACTION NECESSARY FOR SCRIPT TO WORK!

Author: Benjamin Frey
Created: Nov. 8th 2023
Last Updated: [Last updated date]
Version: 0.1
Dependencies: 
    Requires installation of DeepProfiler with all its dependencies:
    If not in stalled, run:
    git clone https://github.com/broadinstitute/DeepProfiler.git
    pip install -e 
    pharmbio package (see imports)
"""

# Import statements
import os 
os.environ["DB_URI"] = "postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb"
import pandas as pd
import time
import numpy as np
import tqdm
import click
import shutil
import sys
import math
import subprocess
import polars as pl
import pharmbio
from pharmbio.data_processing.quality_control import get_qc_module, get_channels, flag_outlier_images
from pharmbio.dataset.image_quality import get_image_quality_ref, get_image_quality_data



@click.command()
@click.option('--projectfolder', '-o', type=click.Path(), prompt=True,
              help='The path to the DeepProfiler project folder that will be generated and all files written in.')
@click.option('--metadata', '-m', type=click.Path(exists=True, file_okay=True, dir_okay=False), prompt='Please provide the path to the CSV file',
              help='The path to the CSV file with compound and plate information.')
@click.option('--checkpoints', '-ck', type=click.Path(exists=True), default = None,
              help='The path to a checkpoint file. If not provided, the script will use a default parameter value.')
@click.option('--projectname', '-p', prompt=True,
              help='The name of the project.')
@click.option('--mode', '-c', type=click.Choice(['metadata', 'profile']), prompt=True,
              help='Select "metadata" to execute metadata processing, or "profile" to run profiling.')              

def main(projectname, projectfolder, metadata, mode, checkpoints):
    start_time = time.time()
    if mode == 'metadata':
        try:
            print("Starting the script...")
            metadata_in = pd.read_csv(metadata)
            metadata_in = metadata_in.dropna(subset=['barcode'])
            print(metadata_in.columns)
            print("Creating project folder...")
            create_project_folder(projectfolder)

            if checkpoints is None:
                # Use the default checkpoint parameter value here
                default_checkpoint = "/home/jovyan/share/data/analyses/benjamin/Single_cell_project/Cell_Painting_CNN_v1.hdf5"
                print(f"Using default checkpoint: {default_checkpoint}")
                copy_checkpoint_to_subfolder(projectfolder, default_checkpoint)
            else:
                print("Copying checkpoint to subfolder...")
                copy_checkpoint_to_subfolder(projectfolder, checkpoints)
            
            print("Running quality control...")
            with tqdm.tqdm(total=1, desc="Quality Control") as progress:
                
                qc_df = run_quality_control(str(projectname), metadata=metadata_in, qc_plates=metadata_in["barcode"].unique(), sd=3)
                progress.update()

            with tqdm.tqdm(total=1, desc="Reading and Combining Parquet Files") as progress:
                location_df = read_combine_parquets(metadata_in)
                progress.update()

            with tqdm.tqdm(total=1, desc="Generating Metadata File") as progress:
                generate_metadata_main(qc_df, location_df, projectfolder, projectname)
                progress.update()

            with tqdm.tqdm(total=1, desc="Generating Locations File for DeepProfiler") as progress:
                generate_locations_deepprofiler(location_df, projectfolder)
                progress.update()

            with tqdm.tqdm(total=1, desc="Creating Image Data Symlinks") as progress:
                create_image_data_symlinks(location_df, projectfolder)
                progress.update()

            print("Script execution completed successfully. Ready for profiling with --mode profile.")
            end_time = time.time()
            execution_time = end_time - start_time
            minutes = execution_time // 60
            seconds = execution_time % 60
            print(f"The script executed in {int(minutes)} minutes and {seconds:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            sys.exit(1)
    elif mode == 'profile':
        profile(projectfolder)
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = execution_time // 60
        seconds = execution_time % 60
        print(f"The script executed in {int(minutes)} minutes and {seconds:.2f} seconds.")

    else:
        print("Invalid command. Use either 'metadata' or 'profile'.")

# Functions
def profile(profile_directory):
    """
    Runs bash command to profile project plates using previously created project folder. 
    First goes into DeepProfiler executable folder, then executes DeeProfiler script

    :param root_dir: profile directory. Directory to DeepProfiler file project (created using the metadata mode)
    :return: A list of paths one level deeper for each string.
    """
    try:
        # Change to the specified profile directory
        deepprofiler_executable = ""
        print(f"Changing to directory: {deepprofiler_executable}")
        subprocess.run(["cd", deepprofiler_executable], shell=True, check=True)

        # Execute a Python script using projectfolder as an input parameter
        print("Executing Python script...")
        subprocess.run(["python", "your_script.py", "--projectfolder=", profile_directory, "--config deepprofiler_config.json", "--metadata metadata.csv", "profile"], check=True)

        print("Profile execution completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def run_quality_control(project_name,  metadata, qc_plates: list, sd: float):
    "Function runs quality control on plates for project and filters metadata based on selected"
    # Set the environment variable
    
    qc_ref_df = get_image_quality_ref(project_name, filter={"plate_barcode": qc_plates})
    qc_df = get_image_quality_data(qc_ref_df, force_merging_columns="drop")
    flagged_images = flag_outlier_images(qc_df, default_sd_step=(-sd, sd)).select(['image_id','Metadata_AcqID','Metadata_Barcode','Metadata_Well','Metadata_Site','ImageNumber','outlier_flag']).filter(pl.col('outlier_flag') == 0).to_pandas()
    flagged_images.rename(columns={'Metadata_Barcode': 'barcode', 'Metadata_Well': 'well'}, inplace=True)
    specs_meta_full_flags = pd.merge(flagged_images, metadata, on = ["barcode", "well"], how = "left")

    return specs_meta_full_flags
    
#import pandas as pd
#import os
#os.environ["DB_URI"] = "postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb"
#run_quality_control(projectname, metadata=metadata_in, qc_plates=metadata_in["barcode"].unique(), sd=3)

def find_deeper_paths(root_dir, string_list):
    """
    For each string in string_list, join it with root_dir to generate a path,
    and then find directories or files that are one step deeper within that path.

    :param root_dir: The root directory as a string.
    :param string_list: A list of strings which will be appended to the root_dir to form paths.
    :return: A list of paths one level deeper for each string.
    """
    deeper_paths = []
    subdirs = next(os.walk(root_dir))[1]
    for string in string_list:
        matching_dirs = [d for d in subdirs if string in d]
        full_path = os.path.join(root_dir, matching_dirs[0])
        #full_path = os.path.join(root_dir, string)
        # Check if the path exists and is a directory
        if os.path.isdir(full_path):
            # Get all entries in the directory
            entries = next(os.walk(full_path))[1] + next(os.walk(full_path))[2]
            # Form full paths to these entries and extend the deeper_paths list
            deeper_paths.extend([os.path.join(full_path, entry) for entry in entries])
    
    return deeper_paths

def find_latest_parquet(folders, filename= "featICF_nuclei.parquet"):
    "Takes list of parent folders for plates and finds most recent parquet files, set to nucleu features to extract Nuclei Locations"
    
    paths = []

    for folder in folders:
        # Initialize the highest subfolder number and path to the Parquet file
        highest_num = -1
        path_to_file = ""
        
        # Check if folder path exists
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue
        
        # List all subfolders in the current folder
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            
            # Check if it's a directory and the name is an integer
            if os.path.isdir(subfolder_path) and subfolder.isdigit():
                subfolder_num = int(subfolder)
                
                # Check if the subfolder number is greater than the current highest
                if subfolder_num > highest_num:
                    # Check if the specified file exists in this subfolder
                    potential_file_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(potential_file_path):
                        highest_num = subfolder_num
                        path_to_file = potential_file_path
        
        # If a valid path was found, add it to the lis
        if path_to_file:
            paths.append(path_to_file)
        else:
            print(f"No '{filename}' found in the folder {folder}.")
    
    return paths


def read_combine_parquets(metadata, root_dir = "/share/data/cellprofiler/automation/results/"):
    """
    Reads and merges Parquet files corresponding to unique plate metadata.

    This function takes metadata that includes unique plate identifiers and locates the 
    corresponding Parquet files within a specified root directory by leveraging the 
    'find_latest_parquets' function. It then reads these Parquet files and combines them into 
    a single DataFrame. Only selected columns are retained during the read operation.

    Parameters:
    :param metadata: A pandas DataFrame containing at least the 'Metadata_Plate' column to identify unique plates.
    :param root_dir : str, The root directory where Parquet files are located. The function will search for Parquet files within this directory. Defaults to "/share/data/cellprofiler/automation/results/".

    :return: combined_df : A pandas DataFrame containing combined data from all the Parquet files associated with the unique plates.

    Raises:
    - Exception: If a Parquet file cannot be read, it prints an error message with the file path and exception.
    """
    file_paths = find_deeper_paths(root_dir, list(metadata["barcode"].unique()))
    combined_df = pd.DataFrame()
    columns = ["Metadata_Barcode", "Metadata_Site", "Metadata_AcqID", "Metadata_Well", "FileName_CONC", "FileName_HOECHST", "FileName_PHAandWGA", "FileName_SYTO", "FileName_MITO", "PathName_MITO", "PathName_HOECHST", "PathName_PHAandWGA", "PathName_SYTO", "Location_Center_X", "Location_Center_Y", "AreaShape_MajorAxisLength"]
    parquet_paths = find_latest_parquet(file_paths)
    for file_path in parquet_paths:
        try:
            # Read the Parquet file with selected columns only
            df = pd.read_parquet(file_path, columns=columns)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return combined_df

def generate_metadata_main(metadata, location_df, project_folder: str, project_name: str):
    
    """
    Main function to generate structure of Metadata required by DeepProfiler
    IMPORTANT: Requires these specific columns, if not existent, check your input data!

    :param metadata: Initial metadata input df with compound informations
    :param location_df: Dataframe with location and plate information as generated by previous functions
    :return: Metadata file for DeepProfiler
    """
    if any('moa' in column.lower() for column in metadata.columns):
        metadata_filt = metadata[["barcode", "well", "Metadata_Site", "cbkid", "[moa]", "compound_name", "cmpd_conc"]]
    else: 
        metadata_filt = metadata[["barcode", "well", "Metadata_Site", "cbkid", "compound_name", "cmpd_conc"]]
    new_column_names_metadata = {
    'barcode': 'Metadata_Plate',
    'well': 'Metadata_Well',
    'cbkid': 'Metadata_cmpdName',
    'cmpd_conc': "Metadata_cmpdConc"}
    metadata_filt = metadata_filt.rename(columns=new_column_names_metadata)

    location_filt = location_df[["Metadata_Barcode", "Metadata_Site", "Metadata_Well", "FileName_CONC", "FileName_HOECHST", "FileName_PHAandWGA", "FileName_SYTO", "FileName_MITO"]]
    location_filt = location_filt.drop_duplicates().reset_index(drop = True)
    new_column_names_locations = {
    'Metadata_Barcode': 'Metadata_Plate',
    'FileName_CONC': 'ER',
    'FileName_HOECHST': 'DNA',
    'FileName_PHAandWGA': "AGP",
    "FileName_SYTO": "RNA", 
    "FileName_MITO": "Mito"}
    location_filt = location_filt.rename(columns=new_column_names_locations)
    result = pd.merge(metadata_filt, location_filt, on=['Metadata_Plate', 'Metadata_Well', "Metadata_Site"], how='left')
    result["DNA"] = result["Metadata_Plate"] + "/" + result["DNA"] 
    result["ER"] = result["Metadata_Plate"] + "/" + result["ER"] 
    result["RNA"] = result["Metadata_Plate"] + "/" + result["RNA"] 
    result["AGP"] = result["Metadata_Plate"] + "/" + result["AGP"] 
    result["Mito"] = result["Metadata_Plate"] + "/" + result["Mito"] 
    result["Metadata_Site"] = "s" + result["Metadata_Site"].astype(str)
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    path_to_metadata = project_folder + "/inputs/metadata/metadata_deepprofiler" + project_name + ".csv"
    result.to_csv(path_to_metadata)
    #return result

def generate_locations_deepprofiler(location_df, root_folder):

    """
    Function to generate and write .csv for each site as expected by DeepProfiler. 
    Automatically saves files to pre-defined root folder: Needs to be locations folder in project

    :param meta_centers: Dataframe containing locations of cells as generated by read_combine_parquets
    :param root_folder: Folder with DeepProfiler folder structure 
    :return: None
    """
    #if "locations" not in root_folder:
    #    sys.exit("Error: 'locations' is not part of the root_folder. Please include 'locations' in the path.")

    plates = location_df["Metadata_Barcode"].unique()
    for plate in tqdm.tqdm(plates):
        output_folder = root_folder +  "/inputs/locations/" + plate
        os.makedirs(output_folder, exist_ok=True)
        plate_data = location_df[location_df["Metadata_Barcode"] == plate]
        # Group the data by 'well' and 'site' and save each group as a separate CSV file
        grouped = plate_data.groupby(['Metadata_Well', 'Metadata_Site'])
        for group_name, group_data in grouped:
            well, site = group_name
            filename = f"{well}-s{site}-Nuclei.csv"
            file_path = os.path.join(output_folder, filename)
            if os.path.exists(file_path):
                print(f"File {filename} already exists. Skipping to the next.")
                continue
            group_data['Nuclei_Location_Center_X'] = group_data['Location_Center_X'].astype(int)
            group_data['Nuclei_Location_Center_Y'] = group_data['Location_Center_Y'].astype(int)
            group_data[['Nuclei_Location_Center_X','Nuclei_Location_Center_Y']].to_csv(file_path, index=False)


def create_image_data_symlinks(location_df, output_root_folder: str):
    """
    Creates symbolic links for image files in their respective original directories to reduce run/ copy time.
    Images in folder required by DeepProfiler

    :param feat_df: A pandas DataFrame containing the columns 'PathName_HOECHST' and 'Metadata_Barcode',
                    which include the paths to the source image files and the barcode metadata, respectively.
    :param output_root_folder: The root directory path where the destination folders will be created
                               and where the symbolic links will point to.
    :return: None. The function performs file operations and does not return any value.
    """
    source_folders = location_df["PathName_HOECHST"].unique()
    for folder in tqdm.tqdm(source_folders):
        output_folder = output_root_folder + "/inputs/images"
        destination_folder = os.path.join(output_folder, location_df[location_df["PathName_HOECHST"] == folder]["Metadata_Barcode"].unique()[0])
        os.makedirs(destination_folder, exist_ok=True)
        print("Linking plate:", location_df[location_df["PathName_HOECHST"] == folder]["Metadata_Barcode"].unique()[0])
        #if "mikro" in folder:
        #    folder = folder.replace("mikro", "mikro2")
        for filename in os.listdir(folder):

            if filename.lower().endswith('.tiff') and 'thumb' not in filename.lower():
                source_file_path = os.path.join(folder, filename)
                destination_file_path = os.path.join(destination_folder, filename)
                if os.path.islink(destination_file_path) or os.path.exists(destination_file_path):
                    print(f"Symlink for {filename} already exists. Skipping to the next.")
                    continue
                if not os.path.islink(destination_file_path) and not os.path.exists(destination_file_path):
                    os.symlink(source_file_path, destination_file_path)


def create_project_folder(folder_path):
    """
    Creates a main folder and subfolders as specified.

    The structure will be:
    - <folder_path>/
      - inputs/
        - config/
        - images/
        - locations/
        - metadata/
      - outputs/
        - checkpoint/

    :param folder_path: The main directory path where the folder structure will be created.
    """
    # Main folder
    os.makedirs(folder_path, exist_ok=True)
    print(f"Main folder created at: {folder_path}")

    # Subfolders for Inputs
    inputs_subfolders = ['config', 'images', 'locations', 'metadata']
    for subfolder in inputs_subfolders:
        os.makedirs(os.path.join(folder_path, 'inputs', subfolder), exist_ok=True)
    
    # Subfolder for Outputs
    os.makedirs(os.path.join(folder_path, 'outputs', 'checkpoint'), exist_ok=True)
    #Copy generic config file to folder
    shutil.copy("/home/jovyan/share/data/analyses/benjamin/Single_cell_project/deepprofiler_config_example.json", os.path.join(folder_path, 'inputs', 'config', 'deepprofiler_config_example.json'))

def copy_checkpoint_to_subfolder(output_folder, checkpoint_file_path ):
    """
    Copies the specified checkpoint file to the checkpoint subfolder in the given output folder structure.

    :param output_folder: The main output folder where the subfolders are located.
    :param checkpoint_file_path: The path to the checkpoint file to be copied.
    """
    # Path to the destination checkpoint subfolder    
    checkpoint_dest_folder = os.path.join(output_folder, 'outputs', 'checkpoint')

    # Copy the checkpoint file to the 'checkpoint' subfolder
    checkpoint_dest_path = os.path.join(checkpoint_dest_folder, os.path.basename(checkpoint_file_path))
    shutil.copy2(checkpoint_file_path, checkpoint_dest_path)
    

# def main(projectname, projectfolder, metadata, checkpoints):
#     """
#     Executes the script with the provided folder paths.
    
#     PROJECTNAME is the projectname for QC and file saving
#     PROJECTFOLDER is the path to the folder that will be generated.
#     METADATA is the path to the folder where the metadata CSV file is located (not used in copying).
#     CHECKPOINTS is the path to a checkpoint file.
#     """
#     metadata_in = pd.read_csv(metadata)
#     create_project_folder(projectfolder)
#     copy_checkpoint_to_subfolder(projectfolder, checkpoints)
#     qc_df = run_quality_control(projectname,  metadata = metadata_in, qc_plates = metadata_in["barcode"].unique(), sd = 3)
#     location_df = read_combine_parquets(metadata_in)
#     generate_metadata_main(qc_df, location_df)
#     generate_locations_deepprofiler(location_df, projectfolder)
#     create_image_data_symlinks(location_df, projectfolder)




# Main execution
if __name__ == "__main__":
    main()