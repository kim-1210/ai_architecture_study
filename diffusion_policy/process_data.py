from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("diffusion_policy/datas/reach_bc.npz")
DEFAULT_OUTPUT = Path("diffusion_policy/datas/reach_bc_imitation_h15.npz")


@dataclass(frozen=True)
class ProcessedDataset:
    observations: np.ndarray
    action_chunks: np.ndarray
    action_chunks_t: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray


def load_episodes(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return data["observations"], data["actions"], data["episode_ids"]


def build_action_chunk(actions: np.ndarray, start: int, horizon: int) -> np.ndarray:
    end = start + horizon
    chunk = actions[start:end]
    if chunk.shape[0] == horizon:
        return chunk.astype(np.float32, copy=False)

    pad_count = horizon - chunk.shape[0]
    pad = np.repeat(actions[-1][None, :], pad_count, axis=0)
    return np.concatenate([chunk, pad], axis=0).astype(np.float32, copy=False)


def make_imitation_dataset(
    episode_observations: np.ndarray,
    episode_actions: np.ndarray,
    raw_episode_ids: np.ndarray,
    horizon: int,
) -> ProcessedDataset:
    observations = []
    action_chunks = []
    episode_ids = []
    timesteps = []

    for fallback_episode_id, (obs_seq, act_seq) in enumerate(
        zip(episode_observations, episode_actions, strict=True)
    ):
        if len(obs_seq) != len(act_seq):
            raise ValueError(
                f"Episode {fallback_episode_id} has mismatched lengths: "
                f"{len(obs_seq)=}, {len(act_seq)=}"
            )

        episode_id = int(raw_episode_ids[fallback_episode_id])
        for timestep, obs in enumerate(obs_seq):
            observations.append(np.asarray(obs, dtype=np.float32))
            action_chunks.append(build_action_chunk(act_seq, timestep, horizon))
            episode_ids.append(episode_id)
            timesteps.append(timestep)

    observations_np = np.stack(observations, axis=0)
    action_chunks_np = np.stack(action_chunks, axis=0)

    return ProcessedDataset(
        observations=observations_np,
        action_chunks=action_chunks_np,
        action_chunks_t=np.transpose(action_chunks_np, (0, 2, 1)),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        timesteps=np.asarray(timesteps, dtype=np.int32),
    )


def save_dataset(dataset: ProcessedDataset, output_path: Path, horizon: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=dataset.observations,
        action_chunks=dataset.action_chunks,
        action_chunks_t=dataset.action_chunks_t,
        episode_ids=dataset.episode_ids,
        timesteps=dataset.timesteps,
        horizon=np.int32(horizon),
        obs_dim=np.int32(dataset.observations.shape[-1]),
        action_dim=np.int32(dataset.action_chunks.shape[-1]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reach_bc.npz into an imitation-learning dataset with fixed action chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to source rollout npz. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to processed dataset npz. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Number of future actions per training sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")

    episode_observations, episode_actions, raw_episode_ids = load_episodes(args.input)
    dataset = make_imitation_dataset(
        episode_observations=episode_observations,
        episode_actions=episode_actions,
        raw_episode_ids=raw_episode_ids,
        horizon=args.horizon,
    )
    save_dataset(dataset, args.output, args.horizon)

    print(f"Saved imitation dataset to {args.output}")
    print(f"observations shape: {dataset.observations.shape}")
    print(f"action_chunks shape: {dataset.action_chunks.shape}")
    print(f"action_chunks_t shape: {dataset.action_chunks_t.shape}")


if __name__ == "__main__":
    main()
