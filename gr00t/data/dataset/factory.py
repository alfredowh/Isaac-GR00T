# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
import warnings

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor, ShardedDataset
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        eval_enabled = self.config.training.eval_strategy != "no"
        eval_ratio = self.config.training.eval_set_split_ratio
        if eval_enabled and not (0.0 < eval_ratio < 1.0):
            raise ValueError(
                f"eval_set_split_ratio must be in (0, 1), got {eval_ratio}"
            )

        train_datasets = []
        train_weights = []
        eval_datasets = []
        eval_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            per_spec_train_datasets = []
            per_spec_eval_datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    decoder_kwargs=self.config.data.decoder_kwargs,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )

                split_train_dataset, split_eval_dataset = self._split_train_eval_dataset(
                    dataset=dataset,
                    eval_enabled=eval_enabled,
                    eval_ratio=eval_ratio,
                    split_seed=self.config.data.seed,
                )
                self._log_dataset_split_summary(
                    dataset_path=str(dataset_path),
                    train_dataset=split_train_dataset,
                    eval_dataset=split_eval_dataset,
                    eval_enabled=eval_enabled,
                )
                per_spec_train_datasets.append(split_train_dataset)
                if split_eval_dataset is not None:
                    per_spec_eval_datasets.append(split_eval_dataset)

            train_lengths = np.array([len(dataset) for dataset in per_spec_train_datasets])
            train_relative_lengths = train_lengths / train_lengths.sum()
            for dataset, relative_length in zip(per_spec_train_datasets, train_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                train_datasets.append(dataset)
                train_weights.append(weight)

            if per_spec_eval_datasets:
                eval_lengths = np.array([len(dataset) for dataset in per_spec_eval_datasets])
                eval_relative_lengths = eval_lengths / eval_lengths.sum()
                for dataset, relative_length in zip(per_spec_eval_datasets, eval_relative_lengths):
                    weight = relative_length * dataset_spec.mix_ratio
                    eval_datasets.append(dataset)
                    eval_weights.append(weight)

        if eval_enabled and not eval_datasets:
            raise ValueError(
                "Evaluation is enabled but no validation shards are available. "
                "Increase dataset size or lower eval_set_split_ratio."
            )

        total_train_shards = sum(len(ds) for ds in train_datasets)
        total_train_steps = sum(
            sum(ds.get_shard_length(i) for i in range(len(ds))) for ds in train_datasets
        )
        if eval_enabled:
            total_eval_shards = sum(len(ds) for ds in eval_datasets)
            total_eval_steps = sum(
                sum(ds.get_shard_length(i) for i in range(len(ds))) for ds in eval_datasets
            )
            self._log_info_once(
                "Dataset totals - train: %d shards (%d steps), eval: %d shards (%d steps)",
                total_train_shards,
                total_train_steps,
                total_eval_shards,
                total_eval_steps,
            )
        else:
            self._log_info_once(
                "Dataset totals - train: %d shards (%d steps)",
                total_train_shards,
                total_train_steps,
            )

        eval_mixture = None
        if eval_datasets:
            eval_mixture = ShardedMixtureDataset(
                datasets=eval_datasets, # train_datasets
                weights=eval_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )

        return (
            ShardedMixtureDataset(
                datasets=train_datasets,
                weights=train_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            eval_mixture,
        )

    @staticmethod
    def _split_train_eval_dataset(
        dataset: ShardedSingleStepDataset,
        eval_enabled: bool,
        eval_ratio: float,
        split_seed: int,
    ) -> tuple[ShardedDataset, ShardedDataset | None]:
        if not eval_enabled:
            return dataset, None

        num_shards = len(dataset)
        if num_shards < 2:
            warnings.warn(
                f"Dataset {dataset.dataset_path} has {num_shards} shard(s); "
                "cannot create a non-empty validation split."
            )
            return dataset, None

        rng = np.random.default_rng(split_seed)
        permuted_indices = rng.permutation(num_shards)

        eval_count = int(round(num_shards * eval_ratio))
        eval_count = max(1, min(eval_count, num_shards - 1))

        eval_indices = sorted(int(i) for i in permuted_indices[:eval_count])
        train_indices = sorted(int(i) for i in permuted_indices[eval_count:])

        return ShardSubsetDataset(dataset, train_indices), ShardSubsetDataset(
            dataset,
            eval_indices,
            extra_inputs={"return_loss": True},
        )

    @staticmethod
    def _is_rank0_or_single_process() -> bool:
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def _dataset_num_steps(dataset: ShardedDataset) -> int:
        return sum(dataset.get_shard_length(i) for i in range(len(dataset)))

    @staticmethod
    def _log_info_once(message: str, *args) -> None:
        if DatasetFactory._is_rank0_or_single_process():
            logging.info(message, *args)

    @staticmethod
    def _log_dataset_split_summary(
        dataset_path: str,
        train_dataset: ShardedDataset,
        eval_dataset: ShardedDataset | None,
        eval_enabled: bool,
    ) -> None:
        train_shards = len(train_dataset)
        train_steps = DatasetFactory._dataset_num_steps(train_dataset)
        if eval_enabled:
            eval_shards = 0 if eval_dataset is None else len(eval_dataset)
            eval_steps = 0 if eval_dataset is None else DatasetFactory._dataset_num_steps(eval_dataset)
            DatasetFactory._log_info_once(
                "Dataset split %s - train: %d shards (%d steps), eval: %d shards (%d steps)",
                dataset_path,
                train_shards,
                train_steps,
                eval_shards,
                eval_steps,
            )
        else:
            DatasetFactory._log_info_once(
                "Dataset %s - train: %d shards (%d steps)",
                dataset_path,
                train_shards,
                train_steps,
            )

class ShardSubsetDataset(ShardedDataset):
    """View over a subset of shard indices from a base sharded dataset."""

    def __init__(
        self,
        base_dataset: ShardedSingleStepDataset,
        shard_indices: list[int],
        extra_inputs: dict | None = None,
    ):
        super().__init__(base_dataset.dataset_path)
        self.base_dataset = base_dataset
        self.shard_indices = shard_indices
        self.extra_inputs = extra_inputs or {}
        self.embodiment_tag = base_dataset.embodiment_tag

    def __len__(self) -> int:
        return len(self.shard_indices)

    def get_shard_length(self, idx: int) -> int:
        return self.base_dataset.get_shard_length(self.shard_indices[idx])

    def get_shard(self, idx: int) -> list:
        datapoints = self.base_dataset.get_shard(self.shard_indices[idx])
        if not self.extra_inputs:
            return datapoints

        updated_datapoints = []
        for datapoint in datapoints:
            updated_datapoint = dict(datapoint)
            updated_datapoint.update(self.extra_inputs)
            updated_datapoints.append(updated_datapoint)
        return updated_datapoints

    def set_processor(self, processor: BaseProcessor):
        self.base_dataset.set_processor(processor)

    def get_dataset_statistics(self) -> dict:
        return self.base_dataset.get_dataset_statistics()

    def get_initial_actions(self):
        if hasattr(self.base_dataset, "get_initial_actions"):
            return self.base_dataset.get_initial_actions()
        return []
