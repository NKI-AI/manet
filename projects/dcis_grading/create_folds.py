"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from sklearn.model_selection import train_test_split, KFold
from fexp.utils.io import read_json, write_list


def main():
    parser = argparse.ArgumentParser(description='Create folds.')
    parser.add_argument('description', type=pathlib.Path, help='Path to dataset_description.json')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds.')
    parser.add_argument('--testing-proportions', type=float,
                        help='Percentage as number between 0 and 1 of testing proportion. Can be 0.')
    args = parser.parse_args()
    directory_path = args.description.parent

    train = list(read_json(args.description).keys())

    if args.testing_proportions > 0.0:
        train, test = train_test_split(train, test_size=args.testing_proportions)
        write_list(directory_path / 'testing_set.txt', test)

    kf = KFold(args.num_folds)
    splits = kf.split(train)
    splits = [_ for _ in splits]
    for idx, (train_idxs, test_idxs) in enumerate(splits):
        path_to_lists_folder = directory_path / f'fold_{idx}'
        path_to_lists_folder.mkdir(exist_ok=True)
        train_set = [train[idx] for idx in train_idxs]
        write_list(path_to_lists_folder / 'training_set.txt', train_set)
        validation_set = [train[idx] for idx in test_idxs]
        write_list(path_to_lists_folder / 'validation_set.txt', validation_set)

    print('Done.')

if __name__ == '__main__':
    main()
