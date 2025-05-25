
  
```bash

###########################

HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco_plane_no_domain_rand \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003

###########################

HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco_plane_domain_rand \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003

###########################
>>>>>
HYDRA_FULL_ERROR=1 \
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco_no_domain_rand \
headless=True \
rewards.reward_penalty_curriculum=False \
rewards.reward_scales.tracking_lin_vel=5 \
rewards.reward_scales.tracking_ang_vel=2.5


###########################

## 133.14

python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco_domain_rand \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003 \
rewards.reward_scales.tracking_lin_vel=10


```