import codecs
import csv
import math
import random
import sys
import copy
import os
import pickle

import numpy as np
from gym import spaces, Env
import mzutils
from mzutils import get_things_in_loc
from pensimpy.peni_env_setup import PenSimEnv

from smpl.envs.pensimenv import PenSimEnvGym, get_observation_data_reformed
from smpl.envs.utils import normalize_spaces


csv.field_size_limit(sys.maxsize)
MINUTES_PER_HOUR = 60
BATCH_LENGTH_IN_MINUTES = 230 * MINUTES_PER_HOUR
BATCH_LENGTH_IN_HOURS = 230
STEP_IN_MINUTES = 12
STEP_IN_HOURS = STEP_IN_MINUTES / MINUTES_PER_HOUR
NUM_STEPS = int(BATCH_LENGTH_IN_MINUTES / STEP_IN_MINUTES)
WAVENUMBER_LENGTH = 2200


class PeniControlTemporalData:
    """
    dataset class helper, mainly aims to mimic d4rl's qlearning_dataset format (which returns a dictionary).
    produced from PenSimPy generated csvs.
    """

    def __init__(self, load_just_a_file='', dataset_folder='examples/example_batches', delimiter=',', time_length=20, observation_dim=9,
                 action_dim=6, normalize=True, np_dtype=np.float32) -> None:
        """
        :param dataset_folder: where all dataset csv files are living in
        """
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.normalize = normalize
        self.np_dtype = np_dtype
        self.max_observations = [276.0, 8.052615, 362.8414, 6.858637, 270.0, 1800.0001, 946.03937, 126920.055,
                                 23.949417]
        self.min_observations = [0.16000001, 4.5955915, 237.97954, 0.0, 0.0, 0.0, 0.0, 50006.516, 2.3598127]
        self.max_actions = [4100.0, 151.0, 36.0, 76.0, 1.2, 510.0]
        self.min_actions = [0.0, 7.0, 21.0, 29.0, 0.5, 0.0]
        self.max_observations = np.array(self.max_observations, dtype=self.np_dtype)
        self.min_observations = np.array(self.min_observations, dtype=self.np_dtype)
        self.max_actions = np.array(self.max_actions, dtype=self.np_dtype)
        self.min_actions = np.array(self.min_actions, dtype=self.np_dtype)

        self.time_length = time_length

        if load_just_a_file != '':
            file_list = [load_just_a_file]
        else:
            file_list = get_things_in_loc(dataset_folder, just_files=True)
        self.file_list = file_list

    def load_file_list_to_dict(self, file_list, shuffle=True):
        file_list = file_list.copy()
        random.shuffle(file_list)
        dataset = {}
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []

        observations_list = []
        actions_list = []
        next_observations_list = []
        rewards_list = []
        terminals_list = []

        for file_path in file_list:
            tmp_observations = []
            tmp_actions = []
            tmp_next_observations = []
            tmp_rewards = []
            tmp_terminals = []
            with codecs.open(file_path, 'r', encoding='utf-8') as fp:
                csv_reader = csv.reader(fp, delimiter=self.delimiter)
                next(csv_reader)
                # get rid of the first line containing only titles
                for row in csv_reader:
                    observation = [row[0]] + row[7:-1]
                    # there are 9 items: Time Step, pH,Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration
                    assert len(observation) == self.observation_dim
                    action = [row[1], row[2], row[3], row[4], row[5], row[6]]
                    # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
                    assert len(action) == self.action_dim
                    reward = row[-1]
                    terminal = False
                    # print(observation[0])
                    tmp_observations.append(observation)
                    tmp_actions.append(action)
                    tmp_rewards.append(reward)
                    tmp_terminals.append(terminal)
                    

            tmp_terminals[-1] = True
            tmp_next_observations = tmp_observations[1:] + [tmp_observations[-1]]
            observations += tmp_observations
            actions += tmp_actions
            next_observations += tmp_next_observations
            rewards += tmp_rewards
            terminals += tmp_terminals
        for i in range(len(observations) - self.time_length + 1):
            observations_list.append(observations[i:i+self.time_length])
            actions_list.append(actions[i:i+self.time_length])
            next_observations_list.append(next_observations[i:i+self.time_length])
            rewards_list.append(rewards[i:i+self.time_length])
            terminals_list.append(terminals[i:i+self.time_length])
        observations = observations_list
        actions = actions_list
        next_observations = next_observations_list
        rewards = rewards_list
        terminals = terminals_list

        dataset['observations'] = np.array(observations, dtype=np.float32)
        dataset['actions'] = np.array(actions, dtype=np.float32)
        dataset['next_observations'] = np.array(next_observations, dtype=np.float32)
        dataset['rewards'] = np.array(rewards, dtype=np.float32)
        dataset['terminals'] = np.array(terminals, dtype=bool)
        print(f'observations shape: {dataset["observations"].shape}')
        print(f'actions shape: {dataset["actions"].shape}')
        print(f'next_observations shape: {dataset["next_observations"].shape}')
        print(f'rewards shape: {dataset["rewards"].shape}')
        print(f'terminals shape: {dataset["terminals"].shape}')

        # self.dataset_max_observations = dataset['observations'].max(axis=0)
        # self.dataset_min_observations = dataset['observations'].min(axis=0)
        # self.dataset_max_actions = dataset['actions'].max(axis=0)
        # self.dataset_min_actions = dataset['actions'].min(axis=0)
        # print("max observations:", self.max_observations)
        # print("min observations:", self.min_observations)
        # print("dataset max observations:", self.dataset_max_observations)
        # print("dataset min observations:", self.dataset_min_observations)
        # print("max actions:", self.max_actions)
        # print("min actions:", self.min_actions)
        # print("dataset max actions:", self.dataset_max_actions)
        # print("dataset min actions:", self.dataset_min_actions)
        # print("normalize:", self.normalize)
        # print("using max/min observations and actions.")
        # if self.normalize:
        #     dataset['observations'], _, _ = normalize_spaces(dataset['observations'], self.max_observations,
        #                                                      self.min_observations)
        #     dataset['next_observations'], _, _ = normalize_spaces(dataset['next_observations'], self.max_observations,
        #                                                           self.min_observations)
        #     dataset['actions'], _, _ = normalize_spaces(dataset['actions'], self.max_actions,
        #                                                 self.min_actions)  # passed in a normalized version.
        # # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        return dataset

    def get_dataset(self):
        return self.load_file_list_to_dict(self.file_list)

if __name__ == "__main__":
    env_name = "pensimenv"
    dataset_location = "offline_temporal_datasets"
    normalize = False
    dataset_obj = PeniControlTemporalData(
        dataset_folder="pensimpy_1010_samples", normalize=normalize, time_length=20
    )
    if dataset_obj.file_list:
        print("Temporal Penicillin_Control_Challenge data correctly initialized.")
    else:
        raise ValueError("Penicillin_Control_Challenge data initialization failed.")
    file_slice = 3
    file_list = dataset_obj.file_list[:file_slice]
    number_of_training_set = 2
    training_items = random.sample(file_list, number_of_training_set)
    evaluating_items = copy.deepcopy(file_list)
    for i in training_items:
        evaluating_items.remove(i)
    dataset_obj.file_list = training_items
    dataset_d4rl_training = dataset_obj.get_dataset()
    dataset_obj.file_list = evaluating_items
    dataset_d4rl_evaluating = dataset_obj.get_dataset()
    dataset_loc = os.path.join(dataset_location, f"{env_name}")
    mzutils.mkdir_p(dataset_loc)

    dataset = dataset_d4rl_training
    tmp_dataset_loc = os.path.join(
        dataset_loc, f"{number_of_training_set}_normalize={normalize}.pkl"
    )
    with open(tmp_dataset_loc, "wb") as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {tmp_dataset_loc}")

    dataset = dataset_d4rl_evaluating
    tmp_dataset_loc = os.path.join(
        dataset_loc, f"{file_slice-number_of_training_set}_normalize={normalize}.pkl"
    )
    with open(tmp_dataset_loc, "wb") as fp:
        pickle.dump(dataset, fp)
    print(f"saved dataset {tmp_dataset_loc}")

