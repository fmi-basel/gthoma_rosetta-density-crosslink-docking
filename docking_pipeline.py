# Copyright 2025 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Georg Kempf, Friedrich Miescher Institute for Biomedical Research

from contextlib import closing
import json
import logging
from multiprocessing import Pool
import os
import pickle
import re
import shutil
import sys
import tarfile
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import subprocess
import pandas as pd
from pyRMSD.utils.proteinReading import Reader
import pyRMSD.RMSDCalculator
import pyRMSD.calculators
from pyRMSD.availableCalculators import availableCalculators
from sklearn.cluster import AgglomerativeClustering
import scipy.spatial.distance
from Bio.PDB import PDBParser, PDBIO, Structure, Model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import combinations
import time

# Setup logging
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler('log_file.txt')
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# Pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)


class Utils:
    """ A class with common utility functions.
    
    Methods
    -------
    run_command(cmd: List[str]) -> Tuple[str, str]:
        Runs a shell command and returns the output and error message.
    
    save_pickle(pkl_name: str, object_: Any) -> None:
        Saves an object to a pickle file.
        
    parse_scorefile(score_file: str) -> pd.DataFrame:
        Parses a score file into a pandas DataFrame.
        
    get_from_pickle(pkl_file: str) -> Any:
        Retrieves an object from a pickle file.
        
    remove_files(pose_dir: str, file_list: List[str]) -> None:
        Removes files in a directory that are not listed in file_list.
        
    get_filename(type_: str, step: str, name: str) -> str:
        Generates an absolute file name based on the provided parameters.
        
    load_json(json_file: str) -> Any:
        Loads and returns the content of a JSON file.
        
    divide_list(lst: List[Any], size: int) -> List[List[Any]]:
        Divides a list into sublists of specified size.
        
    merge_df_valid_xlinks(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        Merges two DataFrames on the 'description' column and combines the 'valid_xlinks' columns.
    """

    def run_command(self, cmd: List[str]) -> Tuple[str, str]:
        try:
            cmd_str = ' '.join(cmd)
            logger.info(f"Running command {cmd_str}")
            process = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output, error = process.communicate()

            output_str = output.decode('utf-8').strip()
            error_str = error.decode('utf-8').strip()

            return output_str, error_str
        except Exception as e:
            logger.info(e)
            return None, str(e)

    def save_pickle(self, pkl_name: str, object_: Any) -> None:
        """Save an object to a pickle file."""
        with open(pkl_name, 'wb') as f:
            pickle.dump(object_, f)

    def parse_scorefile(self, score_file: str) -> pd.DataFrame:
        """Parse a score file into a DataFrame."""
        return pd.read_csv(score_file, header=1, delimiter="\s+")

    def get_from_pickle(self, pkl_file: str) -> Any:
        """Load an object from a pickle file."""
        return pickle.load(open(pkl_file, 'rb'))

    def remove_files(self, pose_dir: str, file_list: List[str]) -> None:
        """Remove all files not in files_list."""
        logger.info(f"Pose dir {pose_dir}")
        if pose_dir:
            if os.path.exists(pose_dir):
                len_pose_dir = len(os.listdir(pose_dir))
                logger.info(f"Files to keep {len(file_list)}")
                files_to_remove = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) if f not in [os.path.basename(x) for x in file_list if x is not None]]
                logger.info(f"{len(files_to_remove)} from {len_pose_dir} files to remove")
                for f in files_to_remove:
                    if os.path.exists(f):
                        os.remove(f)

    def get_filename(self, type_: str, step: str, name: str) -> str:
        """Generate an absolute file name."""
        return os.path.abspath(f"{name}_{step}.{type_}")

    def load_json(self, json_file: str) -> Any:
        """Load JSON file."""
        with open(json_file, "r") as f:
            data = json.load(f)
            return data

    def divide_list(self, lst: List[Any], size: int) -> List[List[Any]]:
        """Divide a list into sublists of a specified size."""
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    def merge_df_valid_xlinks(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Merge two dataframes on 'description' and combine 'valid_xlinks' columns."""
        merged_df = pd.merge(df1, df2, on='description', how='inner', suffixes=('_df1', '_df2'))
        merged_df['sum_valid_xlinks'] = merged_df['valid_xlinks_df1'] + merged_df['valid_xlinks_df2']
        merged_df.drop(['valid_xlinks_df1', 'valid_xlinks_df2'], axis=1, inplace=True)
        merged_df['pose_path'] = merged_df['pose_path_df1']
        return merged_df


class RosettaDocking(Utils):
    """Class representing the docking pipeline and related operations."""
    
    def __init__(self, pdb_file: Optional[str] = None, 
                 map_file: Optional[str] = None, 
                 xlink_file: Optional[str] = None, 
                 nproc: Optional[int] = None, 
                 nstruct_centroid: Optional[int] = None, 
                 nstruct_local: Optional[int] = None, 
                 cluster_rmsd: Optional[float] = None, 
                 centroid_score_weights: Optional[str] = None,
                 num_top_centroid_poses: Optional[str] = None, 
                 local_score_weights: Optional[str] = None, 
                 cluster_center: Optional[str] = None, 
                 chains1: Optional[str] = None, 
                 chains2: Optional[str] = None, 
                 xl_threshold_distance: Optional[float] = None, 
                 overwrite: Optional[bool] = None, 
                 resolution: Optional[float] = None, 
                 centroid_score_labels: Optional[str] = None, 
                 local_score_labels: Optional[str] = None, 
                 xlink_test: bool = False, 
                 test_pdb: Optional[str] = None, 
                 dist_measure: Optional[str] = None) -> None:
        """
        Initialize the RosettaDocking object with its parameters.

        Parameters
        ----------
        pdb_file: Optional[str]
            Path to the PDB file to use.
        map_file: Optional[str]
            Path to the density map file.
        xlink_file: Optional[str]
            Path to the crosslink (XL) file.
        nproc: Optional[int]
            Number of processors to use.
        nstruct_centroid: Optional[int]
            Number of poses to generate in centroid docking.
        nstruct_local: Optional[int]
            Number of poses to generate in local docking.
        cluster_rmsd: Optional[float]
            RMSD threshold for clustering.
        centroid_score_weights: Optional[str]
            Weights for centroid score labels (comma-separated string).
        local_score_weights: Optional[str]
            Weights for local score labels (comma-separated string).
        cluster_center: Optional[str]
            Score type for determining cluster center.
        chains1: Optional[str]
            Chain identifier(s) representing the receptor.
        chains2: Optional[str]
            Chain identifier(s) representing the ligand.
        xl_threshold_distance: Optional[float]
            Maximum crosslink threshold/cutoff distance.
        overwrite: Optional[bool]
            Whether to overwrite existing files.
        resolution: Optional[float]
            Resolution of the map.
        centroid_score_labels: Optional[str]
            Labels for scores used in centroid docking pipeline (comma-separated string).
        local_score_labels: Optional[str]
            Labels for scores used in local docking pipeline (comma-separated string).
        xlink_test: bool
            Whether to run a test for the crosslink scoring.
        test_pdb: Optional[str]
            Path to the PDB file to use for testing.
        dist_measure: Optional[str]
            Distance measure to use for scoring.
        """

        # Initialize base parameters
        self.utils = Utils()
        self.base_dir = os.getcwd()
        self.nproc = nproc
        self.nstruct_centroid = nstruct_centroid
        self.nstruct_local = nstruct_local
        self.cluster_rmsd = cluster_rmsd

        # Process score weights
        if centroid_score_weights:
            self.centroid_score_weights = [float(x) for x in centroid_score_weights.split(',')]
        else:
            self.centroid_score_weights = []
            
        if local_score_weights:
            self.local_score_weights = [float(x) for x in local_score_weights.split(',')]
        else:
            self.local_score_weights = []

        # Additional parameters
        self.cluster_center = cluster_center
        self.chains1 = chains1
        self.chains2 = chains2
        self.xl_threshold_distance = xl_threshold_distance
        self.overwrite = overwrite
        self.resolution = resolution

        # Initialize paths and flags
        if map_file:
            self.map_file = os.path.join(self.base_dir, map_file)
            self.density_scoring = True
        else:
            self.map_file = None
            self.density_scoring = False
            
        self.pdb_file = os.path.join(self.base_dir, pdb_file)
        self.xl_scoring = False
        self.dist_measure = dist_measure
        
        # Load crosslink files
        if xlink_file:
            self.xlink_list = self.utils.load_json(xlink_file)
            self.xl_scoring = True
        else:
            self.xlink_list = None
        
        # Process score labels
        self.centroid_score_labels = centroid_score_labels.split(',') if centroid_score_labels else []
        self.local_score_labels = local_score_labels.split(',') if local_score_labels else []

        # Logging crosslinks
        logger.info("Found xlinks:")
        logger.info(self.xlink_list)
        
        if xlink_test:
            logger.info("XL test")
            self.score_poses_xwalk(pdb_file, "test.csv")
            raise SystemExit
            
        # Run docking pipelines
        pose_list, pose_dir, silent_file_centroid = self.centroid_docking_pipeline()
        # Top 5 poses
        pose_list = pose_list[:num_top_centroid_poses]
        logger.info("Top 5 poses from centroid docking:")
        logger.info(pose_list)
        pose_dir, pose_list = self.extract_silent(silent_file_centroid, pose_list)
        all_dfs = self.local_docking_pipeline(pose_list, pose_dir)
        self.get_top_poses(all_dfs, self.local_score_labels, self.local_score_weights)

    def generate_plots(self, df: pd.DataFrame, score_labels: List[str], name: str, markers: List[str] = None) -> None:
        """Generate scatter plots for given score labels."""
        
        def unique_pairwise_combinations(input_list: List[str], num_combinations: int) -> List[Tuple[str, str]]:
            return [combo for combo in combinations(input_list, num_combinations)]
        

        if markers:
            df_markers = df[df['description'].isin(markers)]
            logger.info("DF marked")
            logger.info(df_markers)
            df_nomarkers = df[~df['description'].isin(markers)]

        if len(score_labels) > 1:
            for score_label_1, score_label_2 in unique_pairwise_combinations(score_labels, 2):
                if not re.search('normalized', score_label_1) and not re.search('normalized', score_label_2):
                    if score_label_1 in df.keys() and score_label_2 in df.keys():
                        df.plot.scatter(x=score_label_1, y=score_label_2)
                        if markers:
                            plt.scatter(df_markers[score_label_1], df_markers[score_label_2], color='red')
                        fig_file = f"{name}_{score_label_1}_{score_label_2}.png"
                        logger.info(f"Saving {fig_file}")
                        plt.savefig(fig_file)
                    else:
                        logger.error(f"{score_label_1} and/or {score_label_2} not found in DataFrame")

        if len(score_labels) > 2:
            for score_label_1, score_label_2, score_label_3 in unique_pairwise_combinations(score_labels, 3):
                if not re.search('normalized', score_label_1) and not re.search('normalized', score_label_2) and not re.search('normalized', score_label_3):
                    if score_label_1 in df.keys() and score_label_2 in df.keys() and score_label_3 in df.keys():
                        fig = plt.figure()
                        ax = fig.add_subplot(projection='3d')
                        
                        if markers:
                            x_min = df[score_label_1].max()
                            ax.scatter(df_nomarkers[score_label_1], df_nomarkers[score_label_2], df_nomarkers[score_label_3])
                            ax.scatter(df_markers[score_label_1], df_markers[score_label_2], df_markers[score_label_3], c='red')
                            for i, row in df_markers.iterrows():
                                ax.plot([row[score_label_1], row[score_label_1]], [row[score_label_2], row[score_label_2]], [row[score_label_3], x_min], 'k--')  # Line to XY plane
                                ax.plot([row[score_label_1], row[score_label_1]], [row[score_label_2], 0], [row[score_label_3], row[score_label_3]], 'k--')  # Line to YZ plane
                                ax.plot([row[score_label_1],0], [row[score_label_2], row[score_label_2]], [row[score_label_3], row[score_label_3]], 'k--')  # Line to ZX plane
    
                        else:
                            ax.scatter(df[score_label_2], df[score_label_2], df[score_label_3])
                        ax.view_init(elev=20, azim=-60)
                        ax.set_xlabel(score_label_1)
                        ax.set_ylabel(score_label_2)
                        ax.set_zlabel(score_label_3)
                        fig_file = f"{name}_{score_label_1}_{score_label_2}_{score_label_3}_3dplot.png"
                        logger.info(f"Saving {fig_file}")
                        plt.savefig(fig_file)
                    else:
                        logger.error(f"{score_label_1} and/or {score_label_2} and/or {score_label_3} not found in DataFrame")

    def get_top_poses(self, df_list: List[pd.DataFrame], score_labels: List[str], score_weights: List[float]) -> None:
        """Return a list of top poses based on scores.
        
        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of DataFrames containing scores from different sources.
        score_labels : List[str]
            List of score labels.
        score_weights : List[float]
            List of scoring weights.
        """
        logger.info("Get top poses")
        logger.info(score_labels)
        pd.set_option('display.max_columns', None)
        relevant_labels = ['cluster_size', 'combined_score', 'description']
        for score_label in score_labels:
            relevant_labels.append(f"{score_label}_normalized")
            relevant_labels.append(score_label)
        logger.debug("DF list:")
        logger.debug(df_list)
        concat_df = pd.concat(df_list, ignore_index=True)
        df, score_labels_ext = self.get_combined_score(concat_df, score_labels, score_weights)
        df = df.sort_values('combined_score')
        top5 = df[relevant_labels]
        top5_poses = top5['description'].to_list()[-5:]
        self.generate_plots(df, score_labels, 'local_clustered_combined', markers=top5_poses)
        logger.info("Top 5 poses")
        logger.info(top5)
        
        for i, pose in enumerate(top5_poses):
            local_docking = re.search(r'centroid_(\d+)_', pose).group(1)
            for dir in os.listdir(self.base_dir):
                if os.path.isdir(dir):
                    if re.search(local_docking, dir):
                        logger.info(f"{local_docking} matches {dir}")
                        source_file = os.path.join(dir, f"{pose}.pdb")
                        target_file = f"{pose}_top{i}.pdb"
                        if not os.path.exists(source_file):
                            for file in os.listdir(self.base_dir):
                                if file.endswith(".silent"):
                                    if re.search(local_docking, file):
                                        silent_file = file
                                        self.extract_silent(silent_file, [pose])
                        try:
                            shutil.copyfile(source_file, target_file)
                        except Exception as e:
                            logger.info(f"Could not copy due to: {e}")
                        break

    def split_list(self, input_list: List[Any], split_size: int) -> List[List[Any]]:
        """Split a list into batches of the specified size.
        
        Parameters
        ----------
        input_list : List[Any]
            The list to be split.
        split_size : int
            The size of each batch.
        
        Returns
        -------
        List[List[Any]]
            List of sublists where each sublist has a length of `split_size`.
        """
        
        batches = {}
        batch_num = -1
        for i, item in enumerate(input_list):
            if i % split_size == 0:
                batch_num += 1
                logger.info(f"New sublist at index {i}")
                batches[batch_num] = [item]
            else:
                batches[batch_num].append(item)
                
        new_list = [v for v in batches.values()]
        logger.info(f"Generated {len(new_list)} batches for clustering")
        return new_list

    def scoring_pipeline(self,
                        silent_file_docking: Optional[str] = None,
                        sc_file_docking: Optional[str] = None,
                        sc_file_density_scoring: Optional[str] = None,
                        csv_file_xl_scoring: Optional[str] = None,
                        csv_file_density_scoring: Optional[str] = None,
                        csv_file_clustering: Optional[str] = None,
                        pose_dir: Optional[str] = None,
                        docking_step: Optional[str] = None,
                        all_dfs: Optional[List[pd.DataFrame]] = None,
                        pose_name: Optional[str] = None) -> Union[Tuple[List[str], str, str], List[pd.DataFrame]]:
        """Run the scoring pipeline for docking.
        
        Parameters
        ----------
        silent_file_docking : Optional[str]
            Path to the silent file from docking.
        sc_file_docking : Optional[str]
            Path to the score file from docking.
        sc_file_density_scoring : Optional[str]
            Path to the score file from density scoring.
        csv_file_xl_scoring : Optional[str]
            Path to the CSV file for XL scoring results.
        csv_file_density_scoring : Optional[str]
            Path to the CSV file for density scoring results.
        csv_file_clustering : Optional[str]
            Path to the CSV file for clustering results.
        pose_dir : Optional[str]
            Directory where poses are stored.
        docking_step : Optional[str]
            Docking step, either 'centroid' or 'local'.
        all_dfs : Optional[List[pd.DataFrame]]
            List of DataFrames from previous steps.
        pose_name : Optional[str]
            Name of the pose.
        
        Returns
        -------
        Union[Tuple[List[str], str, str], List[pd.DataFrame]]
            If docking step is 'centroid', returns a tuple with a list of poses, pose directory, and silent file path.
            If docking step is 'local', returns a list of DataFrames.
        """
        
        pose_list = None
        logger.info("Scoring by crosslinks")
        
        ##################
        ### XL scoring ###
        ##################
        
        if not os.path.exists(csv_file_xl_scoring):
            pose_dir, pose_list = self.extract_silent(silent_file_docking)
        
        xl_df = self.score_poses_xwalk_batch(pose_dir, csv_file_xl_scoring)
        
        #######################
        ### Density scoring ###
        #######################
        
        if self.density_scoring:
            logger.info("Scoring by density")
            
            if not os.path.exists(csv_file_density_scoring):
                pose_dir, pose_list = self.extract_silent(silent_file_docking)
            
            density_score_df = self.score_density_rosetta(pose_dir, sc_file_density_scoring, csv_file_density_scoring)
        
        ###########################
        ### Clustering of poses ###
        ###########################
        
        logger.info("Clustering")
        if not os.path.exists(csv_file_clustering):
            pose_dir, pose_list = self.extract_silent(silent_file_docking, pose_list)
            logger.info("Last 5 elements from pose_list")
            logger.info(pose_list[-5])
            extracted_file_list = self.extract_docked_chains(pose_dir, pose_list)
            logger.info(f"Extracted file list {len(extracted_file_list)}")
            extracted_file_list_batches = self.split_list(extracted_file_list, 5000)
            ranked_df_list = []
            
            for i, batch in enumerate(extracted_file_list_batches):
                clustered_poses_df = self.cluster_poses(pose_dir, batch, csv_file_clustering.replace(".csv", f"_{i}.csv"))
                if docking_step == 'centroid':
                    _, ranked_df = self.select_best_cluster_poses(cluster_df=clustered_poses_df,
                                                    xl_df=xl_df,
                                                    density_df=density_score_df,
                                                    score_labels=self.centroid_score_labels,
                                                    score_weights=self.centroid_score_weights,
                                                    cluster_center=self.cluster_center,
                                                    pose_name=pose_name,
                                                    skip_plots=True)
                elif docking_step == 'local':
                    energy_score_df = self.utils.parse_scorefile(sc_file_docking)
                    _, ranked_df = self.select_best_cluster_poses(cluster_df=clustered_poses_df,
                                                    density_df=density_score_df,
                                                    energy_df=energy_score_df,
                                                    xl_df=xl_df,
                                                    score_labels=self.local_score_labels,
                                                    score_weights=self.local_score_weights,
                                                    pose_name=pose_name,
                                                    cluster_center=self.cluster_center,
                                                    skip_plots=True)
                ranked_df_list.append(ranked_df)
            
            clustered_poses_df_concat = pd.concat(ranked_df_list, ignore_index=True)
            pose_list = clustered_poses_df_concat['file_path'].tolist()
            logger.info(f"len pose_list for final clustering {len(pose_list)}")
            logger.info(pose_list)
            clustered_poses_df = self.cluster_poses(pose_dir, pose_list, csv_file_clustering)
            logger.info("merged clustered batches df")
            logger.info(clustered_poses_df)
        else:
            clustered_poses_df = pd.read_csv(csv_file_clustering)

        if docking_step == 'centroid':
            if not pose_name:
                pose_name = 'centroid'
                
            pose_list, ranked_df = self.select_best_cluster_poses(cluster_df=clustered_poses_df,
                                                    xl_df=xl_df,
                                                    density_df=density_score_df,
                                                    score_labels=self.centroid_score_labels,
                                                    score_weights=self.centroid_score_weights,
                                                    cluster_center=self.cluster_center,
                                                    pose_name=pose_name)
            self.generate_plots(ranked_df, self.centroid_score_labels, 'centroid_filtered_clustered')
            logger.info("Top 5 clustered poses")
            logger.info(pose_list[:5])
            ranked_df.to_csv(f'all_scores_centroid_filtered_clustered')
            self.utils.remove_files(pose_dir, pose_list)
            return pose_list, pose_dir, silent_file_docking
        
        elif docking_step == 'local':
            energy_score_df = self.utils.parse_scorefile(sc_file_docking)
            pose_list, ranked_df = self.select_best_cluster_poses(cluster_df=clustered_poses_df,
                                            density_df=density_score_df,
                                            energy_df=energy_score_df,
                                            xl_df=xl_df,
                                            score_labels=self.local_score_labels,
                                            score_weights=self.local_score_weights,
                                            pose_name=pose_name,
                                            cluster_center=self.cluster_center)
            logger.info("Clustered poses")
            logger.info(pose_list[:5])
            self.utils.remove_files(pose_dir, pose_list)
            ranked_df.to_csv(f'all_scores_{pose_name}')
            all_dfs.append(ranked_df)
            return all_dfs

    def centroid_docking_pipeline(self) -> Tuple[List[str], str, str]:
        """Pipeline for centroid docking.
        
        Returns
        -------
        Tuple[List[str], str, str]
            A tuple containing the list of poses, the directory of poses, and the silent file path.
        """
        
        prepack_dir = 'prepack'
        pkl_file_prepacking = self.utils.get_filename('pkl', 'prepack', '')
        silent_file_docking = self.utils.get_filename('silent', 'docking', 'centroid')
        sc_file_docking = self.utils.get_filename('sc', 'docking', 'centroid')
        sc_file_density_scoring = self.utils.get_filename('sc', 'density_scored', 'centroid')
        csv_file_clustering = self.utils.get_filename('csv', 'clustered', 'centroid')
        csv_file_density_scoring = self.utils.get_filename('csv', 'density_scored', 'centroid')
        csv_file_xl_scoring = self.utils.get_filename('csv', 'xl_scored', 'centroid')
        starting_pdb = self.pdb_file
        pose_dir = "centroid_extracted_poses"

        if self.overwrite:
            if os.path.exists("centroid_extracted_poses"):
                shutil.rmtree("centroid_extracted_poses")

        ### Prepack
        if not os.path.exists(pkl_file_prepacking):
            self.run_prepack(starting_pdb, prepack_dir)
            prepack_pose_path = self.evaluate_prepack(prepack_dir)
            logger.info(prepack_pose_path)
            self.utils.save_pickle(pkl_file_prepacking, prepack_pose_path)
        else:
            logger.info("Prepack pkl file found")
            prepack_pose_path = self.utils.get_from_pickle(pkl_file_prepacking)

        ### Run centroid docking
        if not os.path.exists(silent_file_docking):
            self.run_centroid_docking(prepack_pose_path, sc_file_docking, silent_file_docking)
        else:
            logger.info("Found centroid docking silent_file")

        ### Run scoring pipeline
        pose_list, pose_dir, silent_file_docking = self.scoring_pipeline(silent_file_docking=silent_file_docking,
                                                    sc_file_docking=sc_file_docking,
                                                    sc_file_density_scoring=sc_file_density_scoring,
                                                    csv_file_xl_scoring=csv_file_xl_scoring,
                                                    csv_file_density_scoring=csv_file_density_scoring,
                                                    csv_file_clustering=csv_file_clustering,
                                                    docking_step='centroid')
        return pose_list, pose_dir, silent_file_docking

    def local_docking_pipeline(self, pose_list: List[str], pose_dir_centroid: str) -> List[pd.DataFrame]:
        """Run local docking on 5 best centroid poses.
        
        Parameters
        ----------
        pose_list : List[str]
            List of poses to use for local docking.
        pose_dir_centroid : str
            Directory containing centroid poses.
        
        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames from scoring steps.
        """
        
        all_dfs = []
        for pose in pose_list:
            if not pose.endswith(".pdb"):
                pose = f"{pose}.pdb"
            logger.info(f"Running local docking pipeline for {pose}")
            pose_name = os.path.splitext(os.path.basename(pose))[0]
            sc_file_docking = self.utils.get_filename('sc', 'local_docking', pose_name)
            sc_file_density_scoring = self.utils.get_filename('sc', 'density_scoring', pose_name)
            silent_file_docking = self.utils.get_filename('silent', 'local_docking', pose_name)
            csv_file_clustering = self.utils.get_filename('csv', 'clustering', pose_name)
            csv_file_xl_scoring = self.utils.get_filename('csv', 'xl_scored', pose_name)
            csv_file_density_scoring = self.utils.get_filename('csv', 'density_scored', pose_name)
            pose_dir = f"{pose_name}_local_docking_extracted_poses"
            if not os.path.exists(sc_file_docking):
                self.run_local_docking(os.path.join(pose_dir_centroid, pose), sc_file_docking, silent_file_docking)
            else:
                logger.info(f"{sc_file_docking} found. Skipping local docking")
            
            energy_score_df = self.evaluate_local_docking(sc_file_docking)
            
            all_dfs = self.scoring_pipeline(silent_file_docking=silent_file_docking,
                                            sc_file_docking=sc_file_docking,
                                            sc_file_density_scoring=sc_file_density_scoring,
                                            csv_file_xl_scoring=csv_file_xl_scoring,
                                            csv_file_density_scoring=csv_file_density_scoring,
                                            csv_file_clustering=csv_file_clustering,
                                            pose_dir=pose_dir,
                                            all_dfs=all_dfs,
                                            pose_name=pose_name,
                                            docking_step='local')
        return all_dfs

    def generate_options_file(self, step: str) -> str:
        """Generates and returns the file path of the options file.

        Parameters
        ----------
        step : str
            The docking step (e.g., 'centroid' or 'local').

        Returns
        -------
        str
            Path to the generated options file.
        """
        options_list = [
            f"-parser:protocol {step}.xml",
            "-docking",
            f"\t-partners {self.chains1}_{self.chains2}",
            "\t-dock_pert 3 8",
            "\t-dock_mcm_trans_magnitude 0.1",
            "\t-dock_mcm_rot_magnitude 5.0",
            "-run:max_retry_job 10",
            "-use_input_sc",
            "-ex1",
            "-ex2aro",
            "-beta"
        ]
        if step == 'centroid':
            options_list.insert(2, "\t-randomize2")
            options_list.insert(3, "\t-spin")
        out_file = f"options_{step}"
        with open(out_file, "w") as f:
            for line in options_list:
                f.write(f"{line}\n")
        return out_file

    def generate_centroid_protocol(self) -> str:
        """Generates and returns the file path of the centroid docking protocol XML.

        Returns
        -------
        str
            Path to the generated centroid docking protocol XML file.
        """
        rosettascripts = ET.Element("ROSETTASCRIPTS")
        scorefxns = ET.SubElement(rosettascripts, "SCOREFXNS")
        score_function_high = ET.SubElement(scorefxns, "ScoreFunction", name="high", weights="beta")
        movers = ET.SubElement(rosettascripts, "MOVERS")
        docking1 = ET.SubElement(movers, "Docking", name="dock_low", score_low="score_docking_low", score_high="high", fullatom="0", local_refine="0", optimize_fold_tree="1", conserve_foldtree="0", ignore_default_docking_task="0", design="0", jumps="1")
        save_and_retrieve_sidechains = ET.SubElement(movers, "SaveAndRetrieveSidechains", name="srsc", allsc="1")
        apply_to_pose = ET.SubElement(rosettascripts, "APPLY_TO_POSE")
        protocols = ET.SubElement(rosettascripts, "PROTOCOLS")
        add_srsc_mover = ET.SubElement(protocols, "Add", mover="srsc")
        add_dock_low_mover = ET.SubElement(protocols, "Add", mover="dock_low")
        xml_string = ET.tostring(rosettascripts, encoding='unicode')
        out_file = "centroid_docking.xml"
        with open(out_file, "w") as f:
            f.write(xml_string)
        return out_file

    def generate_local_docking_protocol(self) -> str:
        """Generates and returns the file path of the local docking protocol XML.

        Returns
        -------
        str
            Path to the generated local docking protocol XML file.
        """
        rosettascripts = ET.Element("ROSETTASCRIPTS")
        scorefxns = ET.SubElement(rosettascripts, "SCOREFXNS")
        score_function_high = ET.SubElement(scorefxns, "ScoreFunction", name="high", weights="beta")
        movers = ET.SubElement(rosettascripts, "MOVERS")
        docking2 = ET.SubElement(movers, "Docking", name="dock_high", score_low="score_docking_low", score_high="high", fullatom="1", local_refine="1", optimize_fold_tree="1", conserve_foldtree="0", design="0", jumps="1")
        protocols = ET.SubElement(rosettascripts, "PROTOCOLS")
        add_dock_high_mover = ET.SubElement(protocols, "Add", mover="dock_high")
        xml_string = ET.tostring(rosettascripts, encoding='unicode')
        out_file = "local_docking.xml"
        with open(out_file, "w") as f:
            f.write(xml_string)
        return out_file

    def generate_density_scoring_protocol(self, map_file: str) -> str:
        """Generates and returns the file path of the density scoring protocol XML.
        
        Parameters
        ----------
        map_file : str
            Path to the density map file.
        
        Returns
        -------
        str
            Path to the generated density scoring protocol XML file.
        """
        rosettascripts = ET.Element("ROSETTASCRIPTS")
        scorefxns = ET.SubElement(rosettascripts, "SCOREFXNS")
        score_function_dens = ET.SubElement(scorefxns, "ScoreFunction", name="dens", weights="beta")
        reweight = ET.SubElement(score_function_dens, "Reweight", scoretype="elec_dens_fast", weight="35")
        set_scale_sc_dens_byres = ET.SubElement(score_function_dens, "Set", scale_sc_dens_byres="R:2.0,K:2.0,E:2.0,D:2.0,M:0.76,C:0.81,Q:1.5,H:0.81,N:0.81,T:0.81,S:0.81,Y:0.88,W:0.88,A:0.88,F:0.88,P:0.88,I:0.88,L:0.88,V:0.88")
        movers = ET.SubElement(rosettascripts, "MOVERS")
        setup_for_density_scoring = ET.SubElement(movers, "SetupForDensityScoring", name="setupdens")
        load_density_map = ET.SubElement(movers, "LoadDensityMap", mapfile=map_file, name="loaddens")
        protocols = ET.SubElement(rosettascripts, "PROTOCOLS")
        add_mover_setupdens = ET.SubElement(protocols, "Add", mover="setupdens")
        add_mover_loaddens = ET.SubElement(protocols, "Add", mover="loaddens")
        output = ET.SubElement(rosettascripts, "OUTPUT", scorefxn="dens")
        xml_string = ET.tostring(rosettascripts, encoding='unicode')
        out_file = "density_scoring.xml"
        with open(out_file, "w") as f:
            f.write(xml_string)
        return out_file
        
    def run_prepack(self, input_pdb: str, prepack_dir: str) -> None:
        """Runs the prepacking step.
        
        Parameters
        ----------
        input_pdb : str
            Path to the input PDB file.
        prepack_dir : str
            Directory for prepacking output.
        """
        logger.info("Running prepack")
        output_path = prepack_dir
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        os.chdir(output_path)
        cmd = [f"mpirun -np 10 docking_prepack_protocol.mpi.linuxgccrelease",
            f"-s {input_pdb}",
                f"-docking:partners {self.chains1}_{self.chains2}",
                "-nstruct 10",
                "-ignore_zero_occupancy false",
                "-scorefile prepack.sc"]
        self.utils.run_command(cmd)
        os.chdir(self.base_dir)

    def evaluate_prepack(self, prepack_dir: str) -> str:
        """Evaluates the prepacking step and returns the path to the lowest energy pose.
        
        Parameters
        ----------
        prepack_dir : str
            Directory for prepacking output.
        
        Returns
        -------
        str
            Path to the lowest energy pose PDB file.
        """
        df = self.utils.parse_scorefile(os.path.join(prepack_dir, "prepack.sc"))
        lowest_energy = df['total_score'].min()
        lowest_energy_pose_name = df.loc[df['total_score'] == lowest_energy, 'description'].iloc[0]
        lowest_energy_pose_path = os.path.join(prepack_dir, f"{lowest_energy_pose_name}.pdb")
        return lowest_energy_pose_path

    def run_centroid_docking(self, input_pdb: str, sc_file: str, silent_file: str) -> None:
        """Runs the centroid docking step.
        
        Parameters
        ----------
        input_pdb : str
            Path to the input PDB file.
        sc_file : str
            Path to the score file.
        silent_file : str
            Path to the silent file.
        """
        logger.info("Running centroid docking")
        docking_protocol = self.generate_centroid_protocol()
        docking_options = self.generate_options_file("centroid")
        cmd = [f"mpirun --oversubscribe -np {self.nproc} rosetta_scripts.mpi.linuxgccrelease",
            f"@{docking_options}",
                f"-parser:protocol {docking_protocol}",
                f"-s {input_pdb}",
                "-out:suffix _centroid",
                f"-nstruct {self.nstruct_centroid}",
                f"-scorefile {sc_file}",
                f"-out:file:silent {silent_file}"]
        self.utils.run_command(cmd)

    def evaluate_centroid_docking(self, sc_file: str) -> pd.DataFrame:
        """Evaluates the centroid docking results and returns them as a DataFrame.
        
        Parameters
        ----------
        sc_file : str
            Path to the score file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing docking results.
        """
        return self.utils.parse_scorefile(sc_file)

    def run_local_docking(self, input_pdb: str, sc_file: str, silent_file: str) -> None:
        """Runs the local docking step.
        
        Parameters
        ----------
        input_pdb : str
            Path to the input PDB file.
        sc_file : str
            Path to the score file.
        silent_file : str
            Path to the silent file.
        """
        docking_protocol = self.generate_local_docking_protocol()
        docking_options = self.generate_options_file("local")
        centroid_pose = os.path.splitext(os.path.basename(input_pdb))[0]
        cmd = ["mpirun --oversubscribe -np 50 rosetta_scripts.mpi.linuxgccrelease",
            f"@{docking_options}",
                f"-parser:protocol {docking_protocol}",
                "-out:suffix _full",
                f"-s {input_pdb}",
                f"-nstruct {self.nstruct_local}",
                f"-scorefile {sc_file}",
                f"-out:file:silent {silent_file}"]
        self.utils.run_command(cmd)

    def evaluate_local_docking(self, sc_file: str) -> pd.DataFrame:
        """Evaluates the local docking results and returns them as a DataFrame.
        
        Parameters
        ----------
        sc_file : str
            Path to the score file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing local docking results.
        """
        return self.utils.parse_scorefile(sc_file)

    def score_density_rosetta(self, pose_dir: str, sc_file: str, csv_density_scoring: str) -> pd.DataFrame:
        """Runs density scoring using Rosetta and returns the results as a DataFrame.
        
        Parameters
        ----------
        pose_dir : str
            Directory containing the poses.
        sc_file : str
            Path to the score file.
        csv_density_scoring : str
            Path to the CSV file for density scoring results.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing density scoring results.
        """
        logger.info("score density")

        if not os.path.exists(csv_density_scoring):
            if not os.path.exists(sc_file):
                filelist = "files_after_xl_filtering.txt"
                with open(filelist, 'w') as f:
                    files = [os.path.join(pose_dir, x) for x in os.listdir(pose_dir)]
                    for file in files:
                        f.write(f"{file}\n")
                cmd = [f"mpirun --oversubscribe -np {self.nproc} score_jd2.mpi.linuxgccrelease",
                        f"-in::file::l {filelist}",
                        "-ignore_unrecognized_res",
                        f"-edensity::mapfile {self.map_file}",
                        f"-edensity::mapreso {self.resolution}",
                        f"-edensity::grid_spacing {self.resolution/3}",
                        "-edensity::fastdens_wt 35.0",
                        "-edensity::cryoem_scatterers",
                        "-crystal_refine",
                        f"-scorefile {sc_file}"]
                self.utils.run_command(cmd)
            df = self.utils.parse_scorefile(sc_file)
            df['description'] = df['description'].str.replace("_0001$", "", regex=True)
            df.to_csv(csv_density_scoring)
        else:
            logger.info(f"{csv_density_scoring} already exists")
            df = pd.read_csv(csv_density_scoring)

        df_sorted = df.sort_values('elec_dens_fast')
        logger.info("Sorted DF after density scoring")
        logger.info(df_sorted[['elec_dens_fast', 'description']])

        return  df_sorted

    def get_tags_from_silent(self, silent_file: str) -> List[str]:
        """Extracts and returns a list of tags from a silent file.
        
        Parameters
        ----------
        silent_file : str
            Path to the silent file.
        
        Returns
        -------
        List[str]
            List of tags extracted from the silent file.
        """
        tag_list = []

        with open(silent_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if not line:
                continue
            if line.startswith("SCORE"):
                fields = line.split()
                if not fields[-1] == 'description':
                    tag_list.append(fields[-1])

        tag_list = list(set(tag_list))

        return tag_list

    def extract_silent_by_tag(self, silent_file: str, tag: str) -> None:
        """Extracts poses from a silent file by tag.
        
        Parameters
        ----------
        silent_file : str
            Path to the silent file.
        tag : str
            Tag of the pose to extract.
        """
        cmd = ["extract_pdbs.static.linuxgccrelease",
        f"-in:file:silent {silent_file}",
        f'-in:file:tags {tag.replace(".pdb", "")}']
        if not os.path.exists(tag):
            output, error = self.utils.run_command(cmd)

    def extract_silent(self, silent_file: str, pose_list: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """Extracts poses from a silent file and returns the directory and list of pose names.
        
        Parameters
        ----------
        silent_file : str
            Path to the silent file.
        pose_list : Optional[List[str]]
            List of pose names to extract.
        
        Returns
        -------
        Tuple[str, List[str]]
            Tuple containing the directory of poses and the list of pose names.
        """
        skip = False
        silent_file = os.path.abspath(silent_file)
        logger.info("extract silent")
        docking_name = os.path.splitext(os.path.basename(silent_file))[0]
        pose_dir = f"{docking_name}_extracted_poses"

        if os.path.exists(pose_dir):
            pose_dir_files = [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(pose_dir)]
        else:
            pose_dir_files = []

        pose_list_initial = pose_list

        if os.path.exists(pose_dir) and not pose_list is None:
            pose_list = [os.path.splitext(os.path.basename(x))[0] for x in pose_list]
            logger.info(f"pose dir exsits. Checking if {len(pose_list)} poses already extracted")
            initial_pose_list = pose_list
            pose_list = [x for x in pose_list if not x in pose_dir_files]
            logger.info(f"{len(pose_list)}/{len(initial_pose_list)} still need to be extracted")
            if len(pose_list) == 0:
                skip = True
            else:
                logger.info("Pose list is empty.")

        if pose_list:
            logger.info(f"Len pose list: {len(pose_list)}")
        else:
            logger.info("pose list is None")

        if os.path.exists(pose_dir) and pose_list is None:
            pose_list = []
            tag_list = self.get_tags_from_silent(silent_file)
            for tag in tag_list:
                if not tag in pose_dir_files:
                    pose_list.append(tag)
            logger.info(f"{len(pose_list)}/{len(pose_dir_files)} still need to be extracted")
        else:
            if not os.path.exists(pose_dir):
                os.mkdir(pose_dir)
            if (pose_list is None or len(pose_list) == 0) and not skip:
                pose_list = self.get_tags_from_silent(silent_file)
                logger.info(f"Found {len(pose_list)} poses in {silent_file}")

        logger.info("pose dir")
        logger.info(pose_dir)

        if len(pose_list) > 0:
            if len(pose_list) > int(self.nproc):
                logger.info(f"len of pose list {len(pose_list)} larger than number of cpus {self.nproc}. Distributing the extraction over all cpus.")
                chunk_size = int(len(pose_list) / self.nproc)
                devided_pose_list = self.utils.divide_list(pose_list, chunk_size)
            else:
                logger.info(f"len of pose list {len(pose_list)} smaller than number of cpus {self.nproc}. Not chunking extraction.")
                devided_pose_list = [[x] for x in pose_list]

            cmd_list = []
            for i, pose_subset in enumerate(devided_pose_list):
                pose_list_file = os.path.abspath(f'pose_list_{i}.txt')
                with open(pose_list_file, 'w') as f:
                    for pose in pose_subset:
                        f.write(f"{pose} ")
                cmd = ["extract_pdbs.static.linuxgccrelease",
                    f"-in:file:silent {silent_file}",
                    f"-in:file:tagfile {pose_list_file}"]
                cmd_list.append(cmd)
            os.chdir(pose_dir)
            with closing(Pool(self.nproc)) as pool:
                results = pool.map(self.utils.run_command, cmd_list)
            for result in results:
                output, error = result
                logger.info(output)
                logger.info(error)
            os.chdir(self.base_dir)
        else:
            logger.info("pose list is empty. Using all poses found in provided dir.")
            if pose_list_initial:
                pose_list = pose_list_initial
            else:
                pose_list = [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(pose_dir)]

        return pose_dir, pose_list

    def run_xwalk(self, args: Tuple[str, str, str, Dict[int, int]]) -> Optional[List[Dict[str, Union[str, int, float]]]]:
        """Runs Xwalk for a given set of arguments and returns a list of results dictionaries.
        
        Parameters
        ----------
        args : Tuple[str, str, str, Dict[int, int]]
            A tuple containing the Xwalk input file, pose path, Xwalk results file, and mapping of homomer IDs.
        
        Returns
        -------
        Optional[List[Dict[str, Union[str, int, float]]]]
            List of dictionaries containing the Xwalk results, or None if an error occurred.
        """
        xwalk_in_file, pose_path, xwalk_results_file, homomer_id_mapping = args
        cmd = ["java",
            "-Xmx1024m",
            f"-cp {os.environ['XWALK_BIN']}",
                "Xwalk",
                f"-infile {pose_path}",
                f"-dist {xwalk_in_file}",
                    "-xSC",
                    f"> {xwalk_results_file}"]
        self.utils.run_command(cmd)
        results = self.parse_xwalk_results(xwalk_results_file, homomer_id_mapping)

        if results is not None:
            if len(results) != len(self.xlink_list):
                logger.error(f"Result list does not match input. Some records must be wrong in {pose_path}.")
                results = None
            if None not in [x['pose_path'] if 'pose_path' in x else None for x in results]:
                os.remove(xwalk_in_file)
                os.remove(xwalk_results_file)
            else:
                logger.error(f"Error with pose {pose_path}")
                results = None
        else:
            logger.error(f"Error with pose {pose_path}. No result was returned")
            logger.info(f"Command for debugging: {cmd}")
            results = None

        return results

    def parse_xwalk_results(self, xwalk_results_file: str, homomer_id_mapping: Dict[int, int]) -> Optional[List[Dict[str, Union[str, int, float]]]]:
        """Parses Xwalk results and returns a list of dictionaries containing the parsed data.
        
        Parameters
        ----------
        xwalk_results_file : str
            Path to the Xwalk results file.
        homomer_id_mapping : Dict[int, int]
            Mapping of homomer IDs for the crosslinks.
        
        Returns
        -------
        Optional[List[Dict[str, Union[str, int, float]]]]
            List of dictionaries containing the parsed Xwalk results, or None if an error occurred.
        """
        try:
            results = []
            if os.path.exists(xwalk_results_file):
                with open(xwalk_results_file, 'r') as f:
                    lines = f.readlines()
                logger.info(f"Parsing xwalk results. {len(lines)} records found.")
                for line in lines:
                    items = line.split('\t')
                    seq_dist = int(items[4])
                    euk_dist = float(items[5])
                    sas_dist = float(items[6])
                    xl_id = int(items[0])
                    pose_path = items[1]
                    xl1 = items[2]
                    xl2 = items[3]
                    xl1_split = xl1.split('-')
                    xl2_split = xl2.split('-')
                    res1, chain1 = xl1_split[1], xl1_split[2]
                    res2, chain2 = xl2_split[1], xl2_split[2]
                    homomer_id = homomer_id_mapping[xl_id]
                    if homomer_id == 0:
                        homomer_id = None
                    result = {'res1': res1,
                            'chain1': chain1,
                            'res2': res2,
                            'chain2': chain2,
                            'seq_dist': seq_dist,
                            'euk_dist': euk_dist,
                            'sas_dist': sas_dist,
                            'xl_id': xl_id,
                            'pose_path': pose_path,
                            'homomer_id': homomer_id}
                    results.append(result)
                return results
            else:
                logger.error(f"{xwalk_results_file} not found")
                return None
        except Exception as e:
            logger.info(f"Error in parsing {xwalk_results_file}")
            traceback.print_exc()
            return None

    def score_poses_xwalk_batch(self, pose_dir: str, csv_xl_scoring: str) -> pd.DataFrame:
        """Scores poses using Xwalk in batch mode and returns the results as a DataFrame.
        
        Parameters
        ----------
        pose_dir : str
            Directory containing the poses.
        csv_xl_scoring : str
            Path to the CSV file for XL scoring results.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing XL scoring results.
        """
        def get_index(pose_path: str, score_list: List[Dict[str, Union[str, int]]]) -> int:
            for i, score_dict in enumerate(score_list):
                if pose_path == score_dict['pose_path']:
                    return i

        if not os.path.exists(csv_xl_scoring):
            score_list = []
            pose_dir_name = os.path.basename(pose_dir)
            tar_filename_base = f'poses_xwalk_{pose_dir_name}'
            homomer_id_mapping = {}
            xwalk_in_file = os.path.join(self.base_dir, f"xwalk_in_{pose_dir_name}")

            tar_files = []
            homomer_id_valid = {}

            matching_files = [file for file in os.listdir(pose_dir) if file.endswith('.pdb')]
            files_sublists = [matching_files[i:i + 100] for i in range(0, len(matching_files), 100)]

            for i, sublist in enumerate(files_sublists):
                start = time.time()
                poses_with_result = []
                expected_poses = [os.path.basename(x) for x in sublist]
                tar_filename = f"{tar_filename_base}_{i}.tar"
                logger.info(f"Creating tarfile for xwalk {tar_filename}")
                with tarfile.open(tar_filename, 'w') as tar:
                    for file in sublist:
                        file_path = os.path.join(pose_dir, file)
                        tar.add(file_path, arcname=file)
                tar_files.append(tar_filename)

                with open(xwalk_in_file, 'w') as f:
                    for o, (res1, chain1, res2, chain2, homomer_id) in enumerate(self.xlink_list):
                        f.write(f"{o}\tx\tLYS-{res1}-{chain1}-CB\tLYS-{res2}-{chain2}-CB\n")
                        if not o in homomer_id_mapping:
                            homomer_id_mapping[o] = int(homomer_id)


                xwalk_results_file = os.path.join(self.base_dir, f"xwalk_out_{pose_dir_name}_{i}")
                cmd = ["java",
                    "-Xmx16384m",
                    f"-cp {os.environ['XWALK_BIN']}",
                        "Xwalk",
                        f"-infile {tar_filename}",
                        f"-dist {xwalk_in_file}",
                            "-xSC",
                            f"> {xwalk_results_file}"]
                logger.info(f"Running {cmd}")
                if not os.path.exists(xwalk_results_file) and not self.overwrite:
                    output, error = self.utils.run_command(cmd)
                    if 'OutOfMemoryError' in output:
                        raise RuntimeError("Xwalk ran out of memory. Increase java heap size. Make sure to remove empty output.")
                    elif 'error' in output:
                        raise RuntimeError("Error in Xwalk run. Make sure to remove empty output.")
                    logger.info("Xwalk output:")
                    logger.info(output)
                    logger.info("Xwalk error:")
                    logger.info(error)
                else:
                    logging.info(f"{xwalk_results_file} already exists. Reusing.")
                results = self.parse_xwalk_results(xwalk_results_file, homomer_id_mapping)
                logger.info("Results")
                logger.info(results)
                if results is not None:
                    if None in [x['pose_path'] if 'pose_path' in x else None for x in results]:
                        error = f"Error in parsing xwalk outfile. Pose path is None."
                        logger.error(error)
                        raise ValueError(error)
                else:
                    error = f"Error in parsing xwalk outfile"
                    logger.error(error)
                    raise ValueError(error)
                    logger.info(f"Command for debugging: {cmd}")


                for result_dict in results:
                    pose_path = result_dict['pose_path']
                    poses_with_result.append(pose_path)
                    homomer_id_result = result_dict['homomer_id']
                    if not pose_path in homomer_id_valid:
                        homomer_id_valid[pose_path] = []
                    logger.info("Assessing the following xl result:")
                    logger.info(result_dict)
                    if not result_dict['pose_path'] in [x['pose_path'] for x in score_list]:
                        score_list.append({'pose_path': result_dict['pose_path'], 'valid_xlinks': 0})
                        logger.info(f"Adding initial entry for {pose_path}")
                        if homomer_id_result:
                            homomer_id_valid[pose_path].append(homomer_id_result)
                    pose_path_index = get_index(result_dict['pose_path'], score_list)
                    if result_dict[self.dist_measure] > 0 and result_dict[self.dist_measure] < self.xl_threshold_distance:
                        logger.info(f"Distance {result_dict[self.dist_measure]} below threshold of {self.xl_threshold_distance}")
                        if homomer_id_result:
                            if homomer_id_result not in homomer_id_valid[pose_path]:
                                logger.info("homomer_id_valid dict")
                                logger.info(homomer_id_valid)
                                logger.info(f"{homomer_id_result} not seen before. Counting as a valid xlink")
                                score_list[pose_path_index]['valid_xlinks'] += 1
                                homomer_id_valid[pose_path].append(homomer_id_result)
                                logger.info(f"Adding {homomer_id_result} to dict")
                            else:
                                logger.info(f"{homomer_id_result} already seen before. Not counting again")
                        else:
                            logger.info(f"Counting a normal valid crosslink for")
                            score_list[pose_path_index]['valid_xlinks'] += 1
                        logger.info(f"Final number of counted xlinks for {pose_path}: {score_list[pose_path_index]['valid_xlinks']}")


                poses_with_result = list(set(poses_with_result))
                end = time.time()
                elapsed = end -start
                logger.info(f"Got {len(poses_with_result)} results for {len(sublist)} poses. Time elapsed: {elapsed}")
                for expected_pose in expected_poses:
                    if not expected_pose in poses_with_result:
                        logging.error(f"No result for {expected_pose}")

            logger.info(f"Got {len(score_list)} results for {len(matching_files)} poses. ")

            df = pd.DataFrame(score_list).sort_values('valid_xlinks')

            df['description'] = df['pose_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            df.to_csv(csv_xl_scoring)
        else:
            df = pd.read_csv(csv_xl_scoring)

        df['description'] = df['pose_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        df = df.sort_values('valid_xlinks', ascending=False)
        logger.info("Sorted DF after XL scoring")
        logger.info(df[['valid_xlinks', 'description']])
        df.to_csv(csv_xl_scoring)

        return df

    def score_poses_xwalk(self, pose_dir: str, csv_xl_scoring: str) -> pd.DataFrame:
        """Scores poses using Xwalk and returns the results as a DataFrame.
        
        Parameters
        ----------
        pose_dir : str
            Directory containing the poses.
        csv_xl_scoring : str
            Path to the CSV file for XL scoring results.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing XL scoring results.
        """
        def get_index(pose_path: str, score_list: List[Dict[str, Union[str, int]]]) -> int:
            for i, score_dict in enumerate(score_list):
                if pose_path == score_dict['pose_path']:
                    return i
        score_list = []
        xwalk_jobs = []
        homomer_id_mapping = {}
        if not os.path.exists(csv_xl_scoring):
            for i, pose in enumerate(os.listdir(pose_dir)):
                if pose.endswith(".pdb"):
                    pose_path = os.path.join(self.base_dir, pose_dir, pose)
                    pose_name = os.path.splitext(os.path.basename(pose_path))[0]
                    xwalk_in_file = os.path.join(self.base_dir, pose_dir, f"xwalk_in_{pose_name}")
                    xwalk_results_file = os.path.join(self.base_dir, pose_dir, f"xwalk_results_{pose_name}")
                    with open(xwalk_in_file, 'w') as f:
                        for i, (res1, chain1, res2, chain2, homomer_id) in enumerate(self.xlink_list):
                            f.write(f"{i}\t{pose_name}\tLYS-{res1}-{chain1}-CB\tLYS-{res2}-{chain2}-CB\n")
                            if not i in homomer_id_mapping:
                                homomer_id_mapping[i] = int(homomer_id)

                    xwalk_jobs.append((xwalk_in_file, pose_path, xwalk_results_file, homomer_id_mapping))

            logger.info(f"xwalk jobs to run: {len(xwalk_jobs)}")

            with closing(Pool(self.nproc)) as p:
                results = p.map_async(self.run_xwalk, xwalk_jobs)

            homomer_id_valid = {}
            for result_dicts in results.get():
                logger.info(result_dicts)
                if not result_dicts is None:
                    for result_dict in result_dicts:
                        pose_path = result_dict['pose_path']
                        homomer_id_result = result_dict['homomer_id']
                        if not pose_path in homomer_id_valid:
                            homomer_id_valid[pose_path] = []
                        logger.info("Assessing the following xl result:")
                        logger.info(result_dict)
                        if result_dict[self.dist_measure] > 0 and result_dict[self.dist_measure] < self.xl_threshold_distance:
                            logger.info(f"Distance below threshold of {self.xl_threshold_distance}")
                            if not result_dict['pose_path'] in [x['pose_path'] for x in score_list]:
                                score_list.append({'pose_path': result_dict['pose_path'], 'valid_xlinks': 1})
                                logger.info(f"Adding initial entry for {pose_path}")
                                if homomer_id_result:
                                    homomer_id_valid[pose_path].append(homomer_id_result)
                            else:
                                if homomer_id_result:
                                    if not homomer_id_result in homomer_id_valid[pose_path]:
                                        logger.info("homomer_id_valid dict")
                                        logger.info(homomer_id_valid)
                                        logger.info(f"{homomer_id_result} not seen before. Counting as a valid xlink")
                                        pose_path_index = get_index(result_dict['pose_path'], score_list)
                                        score_list[pose_path_index]['valid_xlinks'] += 1
                                        homomer_id_valid[pose_path].append(homomer_id_result)
                                        logger.info(f"Adding {homomer_id_result} to dict")
                                    else:
                                        logger.info(f"{homomer_id_result} already seen before. Not counting again")
                                else:
                                    logger.info(f"Counting a normal valid crosslink for")

                                    pose_path_index = get_index(result_dict['pose_path'], score_list)
                                    score_list[pose_path_index]['valid_xlinks'] += 1
                                logger.info(f"Final number of counted xlinks for {pose_path}: {score_list[pose_path_index]['valid_xlinks']}")

            logger.info(f"Got {len(score_list)} results form {len(xwalk_jobs)} jobs.")

            df = pd.DataFrame(score_list).sort_values('valid_xlinks')

            df['description'] = df['pose_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            df.to_csv(csv_xl_scoring)
        else:
            df = pd.read_csv(csv_xl_scoring)
        df['description'] = df['pose_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        df = df.sort_values('valid_xlinks', ascending=False)
        logger.info("Sorted DF after XL scoring")
        logger.info(df[['valid_xlinks', 'description']])
        df.to_csv(csv_xl_scoring)

        return df

    def run_extract_chains(self, pdb_file: str) -> Optional[str]:
        """Extracts chains from a given PDB file and returns the path to the new file.
        
        Parameters
        ----------
        pdb_file : str
            Path to the PDB file.
        
        Returns
        -------
        Optional[str]
            Path to the new file with extracted chains, or None if an error occurred.
        """
        logger.info(f"Extracting chains from {pdb_file}")
        output_file_path = os.path.splitext(os.path.basename(pdb_file))[0]
        output_file_path = f"{output_file_path}_docked_chains.pdb"
        output_file_path = os.path.join(os.path.dirname(pdb_file), output_file_path)
        if not os.path.exists(output_file_path):
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("input", pdb_file)
                writer = PDBIO()
                new_model = Model.Model(0)
                output_structure = Structure.Structure("output")
                output_structure.add(new_model)
                for chain in structure.get_chains():
                    if chain.id in self.chains2:
                        output_structure[0].add(chain)
                io = PDBIO()
                output_file = open(output_file_path, "w")
                for model in output_structure.get_models():
                    output_file.write("MODEL \n")
                    io.set_structure(model)
                    io.save(output_file)
                    output_file.write("ENDMDL\n")
                output_file.close()
                return output_file_path
            except Exception:
                logger.error(f"Error extracting chains for {pdb_file}")
                return None
        else:
            return output_file_path


    def extract_docked_chains(self, pose_dir: str, pose_list: List[str]) -> List[str]:
        """Extracts chains for clustering and returns a list of paths to the extracted files.
        
        Parameters
        ----------
        pose_dir : str
            Directory containing the poses.
        pose_list : List[str]
            List of pose names to process.
        
        Returns
        -------
        List[str]
            List of paths to the extracted files.
        """
        logger.info("Extracting chains for clustering")
        pdb_files = [
            os.path.join(self.base_dir, pose_dir, f)
            for f in os.listdir(pose_dir)
            if f.endswith(".pdb") and os.path.splitext(os.path.basename(f))[0] in pose_list
        ]
        logger.info(f"Extracting chains from {len(pdb_files)} files")
        if len(pdb_files) > 0:
            with closing(Pool(self.nproc)) as p:
                results = p.map_async(self.run_extract_chains, pdb_files)
            output_files = [os.path.basename(f) for f in results.get() if f is not None]
            return output_files
        else:
            logger.info(f"Could not find any files in {pose_dir}")
            sys.exit()

    def cluster_poses(self, pose_dir: str, file_list: List[str], csv_clusters: str) -> pd.DataFrame:
        """Clusters poses based on RMSD and returns the results as a DataFrame.

        Parameters
        ----------
        pose_dir : str
            Directory containing the poses.
        file_list : List[str]
            List of pose files to be clustered.
        csv_clusters : str
            Path to the CSV file for clustering results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing clustering results.
        """
        pdb_reader = Reader()
        logger.info(f"Pose_dir has {len(os.listdir(pose_dir))} files. File list has {len(file_list)} entries.")
        file_list = [os.path.basename(f) for f in file_list]
        pdb_files = [os.path.join(self.base_dir, pose_dir, f) for f in os.listdir(pose_dir) if f in file_list]
        logger.info(f"Number of files to cluster: {len(pdb_files)}")

        for pdb in pdb_files:
            pdb_reader.readThisFile(pdb)

        coordinates = pdb_reader.read(verbose=True)
        calculator = pyRMSD.RMSDCalculator.RMSDCalculator("NOSUP_OMP_CALCULATOR", coordinates)
        matrix = calculator.pairwiseRMSDMatrix()
        pairwise_matrix = scipy.spatial.distance.squareform(matrix)
        clustering = AgglomerativeClustering(metric='precomputed', n_clusters=None, distance_threshold=5.0, linkage='average').fit(pairwise_matrix)
        results = []

        for i, pdb in enumerate(pdb_files):
            result_dict = {
                'description': os.path.splitext(os.path.basename(pdb.replace("_docked_chains", "")))[0],
                'file_path': pdb,
                'cluster': clustering.labels_[i]
            }
            results.append(result_dict)

        df = pd.DataFrame(results)
        df.to_csv(csv_clusters, index=False)

        return df

    def get_combined_score(self, df: pd.DataFrame, score_labels: List[str], score_weights: List[float]) -> Tuple[pd.DataFrame, List[str]]:
        """Calculates combined scores for poses and returns the updated DataFrame and list of score labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the scores.
        score_labels : List[str]
            List of score labels to consider.
        score_weights : List[float]
            List of weights corresponding to the score labels.

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Updated DataFrame with combined scores and extended list of score labels.
        """
        extended_score_labels = score_labels.copy()

        for i, score_label in enumerate(score_labels):
            score_label_norm = f"{score_label}_normalized"
            extended_score_labels.append(score_label_norm)
            scaler = MinMaxScaler()
            column_values = df[score_label].values.reshape(-1, 1)
            column_scaled = scaler.fit_transform(column_values)

            if score_label in ['cluster_size', 'valid_xlinks']:
                df[score_label_norm] = column_scaled
            elif score_label in ['I_sc', 'elec_dens_fast']:
                df[score_label_norm] = 1 - column_scaled
            else:
                logger.info(f"{score_label} not found")
                sys.exit()

            if (df[score_label_norm] == 0.0).all():
                df.loc[:, score_label_norm] = 1.0

            if i == 0:
                df['combined_score'] = float(score_weights[i]) * df[score_label_norm]
            else:
                df['combined_score'] += float(score_weights[i]) * df[score_label_norm]
            logger.info(f"Score weight for {score_label_norm}: {score_weights[i]}")

        return df, extended_score_labels

    def select_best_cluster_poses(self, cluster_df: Optional[pd.DataFrame] = None, density_df: Optional[pd.DataFrame] = None, energy_df: Optional[pd.DataFrame] = None, xl_df: Optional[pd.DataFrame] = None, score_labels: Optional[List[str]] = None, score_weights: Optional[List[float]] = None, cluster_center: Optional[str] = None, pose_name: Optional[str] = None, skip_plots: bool = False) -> Tuple[List[str], pd.DataFrame]:
        """Selects best poses from each cluster and returns the list of pose names and a DataFrame.
        
        Parameters
        ----------
        cluster_df : Optional[pd.DataFrame]
            DataFrame containing clustering results.
        density_df : Optional[pd.DataFrame]
            DataFrame containing density scores.
        energy_df : Optional[pd.DataFrame]
            DataFrame containing energy scores.
        xl_df : Optional[pd.DataFrame]
            DataFrame containing XL scores.
        score_labels : Optional[List[str]]
            List of score labels to consider.
        score_weights : Optional[List[float]]
            List of weights corresponding to the score labels.
        cluster_center : Optional[str]
            Center of clusters for choosing representative poses.
        pose_name : Optional[str]
            Name of the pose.
        skip_plots : bool
            Whether to skip generating plots.

        Returns
        -------
        Tuple[List[str], pd.DataFrame]
            List of selected poses and DataFrame with filtered scores.
        """
        if cluster_df is not None:
            merged_df = cluster_df
            score_labels_found = ['description', 'file_path', 'cluster_size']
            if density_df is not None:
                logger.info("Merging cluster df and density df")
                merged_df = pd.merge(merged_df, density_df, on='description', how='inner')
                score_labels_found.append('elec_dens_fast')
                logger.info(f"Size of merged DF: {len(merged_df)}")
            else:
                logger.info("No density df provided")

            if energy_df is not None:
                logger.info("Merging with energy df")
                merged_df = pd.merge(merged_df, energy_df, on='description', how='inner')
                score_labels_found.append('I_sc')
                logger.info(f"Size of merged DF: {len(merged_df)}")
            else:
                logger.info("No energy df provided")

            if xl_df is not None:
                logger.info("Merging with xl df")
                merged_df = pd.merge(merged_df, xl_df, on='description', how='inner')
                score_labels_found.append('valid_xlinks')
                logger.info(f"Size of merged DF: {len(merged_df)}")
            else:
                logger.info("No xl df provided")

        else:
            logger.info("No df found")
            sys.exit()

        unique_counts = merged_df['cluster'].value_counts()
        merged_df['cluster_size'] = merged_df['cluster'].map(unique_counts)
        logger.info(merged_df[score_labels_found])

        for score_label in score_labels:
            if score_label not in score_labels_found:
                print(f"Cannot score by {score_label} since it was not found in the output files")
                sys.exit()

        merged_df, score_labels = self.get_combined_score(merged_df, score_labels, score_weights)
        labels = score_labels.copy()
        score_labels.append("cluster_size")
        labels.append("combined_score")
        labels.append("description")
        labels.append("cluster_size")
        labels.append("file_path")
        logger.info(merged_df[labels])
        logger.info(f"Cluster center: {cluster_center}")
        if not skip_plots:
            self.generate_plots(merged_df, score_labels, f'{pose_name}_unclustered')

        if cluster_center in ['cluster_size', 'valid_xlinks']:
            # Get best pose form cluster by max value for cluster_size and valid_xlinks
            grouped_df = merged_df.groupby('cluster').apply(lambda x: x.loc[x[cluster_center].idxmax(), labels]).sort_values('combined_score', ascending=False)
        else:
            # Get best pose form cluster by min value for density_score and energy_score
            grouped_df = merged_df.groupby('cluster').apply(lambda x: x.loc[x[cluster_center].idxmin(), labels]).sort_values('combined_score', ascending=False)
        if not skip_plots:
            self.generate_plots(merged_df, score_labels, f'{pose_name}_clustered')

        logger.info("Best poses")
        logger.info(grouped_df)
        pose_list = [f"{x}.pdb" for x in grouped_df['description'].tolist()]
        return pose_list, grouped_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rosetta docking pipeline with scoring by energy, crosslinks and density')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to PDB file')
    parser.add_argument('--map_file', type=str, default=None, help='Path to map file')
    parser.add_argument('--xlink_file', type=str, default=None, help='Path to xlink file in json format')
    parser.add_argument('--chains1', type=str, required=True, help='Chains 1')
    parser.add_argument('--chains2', type=str, required=True, help='Chains 2')
    parser.add_argument('--resolution', type=float, default=None, help='Effective resolution of the 3DEM map in Angstrom')
    parser.add_argument('--nproc', type=int, default=10, help='Number of processors')
    parser.add_argument('--nstruct_centroid', type=int, default=50000, help='Number of centroid structures. Default 50,000')
    parser.add_argument('--nstruct_local', type=int, default=2000, help='Number of local structures. Default 2,000 per selected centroid pose')
    parser.add_argument('--cluster_rmsd', type=float, default=9, help='Cluster RMSD in Angstrom')
    parser.add_argument('--centroid_score_labels', default=None, type=str, help='Rosetta scoring labels to be used in pose ranking, e.g. elec_dens_fast,valid_xlinks')
    parser.add_argument('--centroid_score_weights', type=str, default=None, help='List of weights corresponding to centroid score label list')
    parser.add_argument('--num_top_centroid_poses', type=int, default=5, help='Number of clustered and ranked centroid poses to select for local docking. Default top 5.')
    parser.add_argument('--local_score_labels', default=None, type=str, help='Rosetta scoring labels to be used in pose ranking. e.g. I_sc,elec_dens_fast,valid_xlinks')
    parser.add_argument('--local_score_weights', type=str, default=None, help='List of weights corresponding to local score label list')
    parser.add_argument('--cluster_center', type=str, default='elec_dens_fast', help='Which score to use for finding the cluster center. e.g. one of elec_dens_fast,valid_xlinks or combined_score to use the average score based on --centroid_score_labels or --local_score_labels')
    parser.add_argument('--xl_threshold_distance', type=float, default=30.0, help='XL threshold distance in Angstrom')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite files')
    parser.add_argument('--xlink_test', action='store_true', default=False, help='Test of crosslink scoring for debugging')
    parser.add_argument('--dist_measure', default='dist', type=str, choices=['euk_dist', 'sas_dist'], help='One of [euk_dist, sas_dist]. Sets the distance measure used by Xwalk to euclidean distance or SAS distance (see Xwalk documentation)')
    args = parser.parse_args()

    RosettaDocking(pdb_file=args.pdb_file, map_file=args.map_file, xlink_file=args.xlink_file,
                   nproc=args.nproc, nstruct_centroid=args.nstruct_centroid, nstruct_local=args.nstruct_local,
                   cluster_rmsd=args.cluster_rmsd, centroid_score_weights=args.centroid_score_weights, num_top_centroid_poses=args.num_top_centroid_poses, local_score_weights=args.local_score_weights,
                   cluster_center=args.cluster_center, chains1=args.chains1, chains2=args.chains2,
                   xl_threshold_distance=args.xl_threshold_distance, overwrite=args.overwrite,
                   resolution=args.resolution, centroid_score_labels=args.centroid_score_labels, local_score_labels=args.local_score_labels, xlink_test=args.xlink_test, dist_measure=args.dist_measure)


