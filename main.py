import os
from argparse import ArgumentParser
import numpy as np
import torch
import pybullet_envs
import gym
import warnings

warnings.filterwarnings("ignore")

from ddpg import DDPG
from ddpg.replay_buffer import MemoryBuffer
from ddpg.replay_buffer.transition import Transition


def evaluate_policy(env: gym.Env, policy: DDPG, eval_episodes: int = 10) -> float:
    """Function to evaluate current policy during training. It averages all rewards over episodes.

    Args:
        env (gym.Env): Environment
        policy (DDPG): DDPG agent.
        eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 10.

    Returns:
        float: Average reward in episodes.
    """
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.take_action(np.array(obs))
            obs, reward, done, *_ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("\t" + "-" * 50)
    print(f"\tAverage Reward over the Evaluation Step: {avg_reward:.2f}")
    print("\t" + "-" * 50)
    return avg_reward


if __name__ == "__main__":
    print("Setting up hyperparameters...")
    parser = ArgumentParser(
        description="Input parameters for RL with DDPG in a continuous environment."
    )
    parser.add_argument(
        "--checkpoint-folder",
        metavar="N",
        type=str,
        help="Folder to store checkpoints and results.",
        default="results",
    )
    parser.add_argument(
        "--env-name",
        metavar="N",
        type=str,
        help="Name of environment.",
        default="Walker2DBulletEnv-v0",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        help="Random seed to replicate results.",
        default=0,
    )
    parser.add_argument(
        "--models-hidden-layers",
        metavar="N",
        type=int,
        nargs="+",
        help="Number of units for each hidden layer.",
        default=[300],
    )
    parser.add_argument(
        "--max-time-steps",
        metavar="N",
        type=int,
        help="Number of total iterations for expeirence.",
        default=500_000,
    )
    parser.add_argument(
        "--start-timesteps",
        metavar="N",
        type=int,
        help="Number of iterations before which model randomly performs actions.",
        default=1e4,
    )
    parser.add_argument(
        "--exploration-gaussian-noise",
        metavar="N",
        type=float,
        help="Standard deviation of Gaussian Noise when exploring.",
        default=0.10,
    )
    parser.add_argument(
        "--batch_size",
        metavar="N",
        type=int,
        help="Size of the batch during training.",
        default=32,
    )
    parser.add_argument(
        "--discount-factor",
        metavar="N",
        type=float,
        help="Discount factor used in Bellman Equation.",
        default=0.99,
    )
    parser.add_argument(
        "--tau",
        metavar="N",
        type=float,
        help="Target network update rate.",
        default=0.995,
    )
    parser.add_argument(
        "--policy-noise",
        metavar="N",
        type=float,
        help="Standard deviation of Gaussian Noise when following policy.",
        default=0.10,
    )
    parser.add_argument(
        "--noise-clip",
        metavar="N",
        type=float,
        help="Maximum size of Gaussian Noise added to actions.",
        default=0.50,
    )
    parser.add_argument(
        "--policy-freq",
        metavar="N",
        type=int,
        help="Frequency with which we updated target network weights.",
        default=2,
    )
    parser.add_argument(
        "--evaluation-freq",
        metavar="N",
        type=int,
        help="Frequency with which we evaluate policy.",
        default=5e3,
    )
    args = parser.parse_args()
    print("Hyperaparameters set up!")

    print("Instantiating environment, TD3 Agent and Replay Experience...")
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPG(
        state_space_dim=state_dim,
        action_space_dim=action_dim,
        max_action_value=max_action,
        hidden_layers=args.models_hidden_layers,
        discount_factor=args.discount_factor,
        policy_update_freq=args.policy_freq,
    )
    replay_experience = MemoryBuffer()
    print("Agent and environment ready!")

    print("Creating folders and filenames for checkpoints/artifacts...")
    CHECKPOINT_FOLDER = os.path.join(os.curdir, args.checkpoint_folder)
    MODELS_FOLDER = os.path.join(CHECKPOINT_FOLDER, "models")
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)
        os.makedirs(MODELS_FOLDER)
    print("Everything ready!")

    print("Starting training process...")
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    done = True

    evaluations = [evaluate_policy(env, agent)]
    while total_timesteps < args.max_time_steps:

        if done:

            if total_timesteps != 0 and len(replay_experience) > args.batch_size:
                print(
                    f"\tTotal time steps: {total_timesteps} Episode Num: {episode_num} Reward {episode_reward:.2f}"
                )
                agent.train(
                    replay_buffer=replay_experience,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    episode_length=episode_timesteps,
                )

            if timesteps_since_eval >= args.evaluation_freq:
                timesteps_since_eval %= args.evaluation_freq
                evaluations.append(evaluate_policy(env, agent))
                agent.save(os.path.join(MODELS_FOLDER, "model"))
                np.save(os.path.join(CHECKPOINT_FOLDER, "results"), evaluations)

            obs = env.reset()
            done = False

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.take_action(np.array(obs))
            if args.exploration_gaussian_noise > 0:
                action = (
                    action
                    + np.random.normal(
                        loc=0.0, scale=args.exploration_gaussian_noise, size=action_dim
                    )
                ).clip(env.action_space.low, env.action_space.high)

        new_obs, reward, done, _ = env.step(action)
        done_transition = (
            False
            if (episode_timesteps + 1) == env._max_episode_steps and not done
            else done
        )

        transition = Transition(
            state=obs, action=action, reward=reward, next_state=new_obs, done=done
        )

        replay_experience.add(transition)
        episode_reward += reward

        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    evaluations.append(evaluate_policy(env, agent))
    agent.save(os.path.join(MODELS_FOLDER, "model"))
    np.save(os.path.join(CHECKPOINT_FOLDER, "results"), evaluations)

    print("Training finished!")
