# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlSacActorCriticCfg,
    RslRlSacAlgorithmCfg,
)


@configclass
class HandoverBlockSACRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "dual_block_handover_sac"
    empirical_normalization = False
    policy = RslRlSacActorCriticCfg(
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = RslRlSacAlgorithmCfg(
        learning_rate=3e-4,
        gamma=0.98,
        tau=0.005,
        alpha=0.2,  # entropy regularization coefficient
        automatic_entropy_tuning=True,
        batch_size=256,
        replay_buffer_size=int(1e6),
        start_steps=10000,
        update_after=1000,
        update_every=50,
        policy_update_delay=2,
        target_update_interval=1,
        max_grad_norm=1.0,
    )
