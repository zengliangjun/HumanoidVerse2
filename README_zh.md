# HumanoidVerse2 : 模块化多模拟器学习框架.

基于 [HumanoidVerse ](README_HumanoidVerse.md) 开发,环境设置及初始运行请参考 [HumanoidVerse ](README_HumanoidVerse.md)

本项目基于HumanoidVerse 开发，为机器人算法开发更高效。

# task
将task抽像出不同的子模块，所有子模块生命周期均如 [BaseManager](humanoidverse/envs/base_task/term/base.py)
所示：
## BaseManager

<details>
<summary>BaseManager代码 .</summary>
```python
class BaseManager():
    def __init__(self, _task):
        self.task = _task                 # 任务/环境实例对像 （BaseTask 子类）
        self.num_envs = _task.num_envs    # 环境实例数
        self.device = _task.device        # 仿真环境运行设备
        self.config = _task.config        #

    # 第一阶段，初始化阶段
    def pre_init(self):
        '''
        子模块在初始化时被其它模块依赖信息，提前定义和初始化。
        '''
        pass

    def init(self):
        '''
        子模块初始化信息。
        '''
        pass

    def post_init(self):
        '''
        子模块初始化依赖于其它子模块，在此初始化。
        '''
        pass

    # 第二阶段，仿真环境运行阶段
    def pre_physics_step(self, actions):
    	'''
        子模块在仿真环境运行阶段时被其它模块依赖信息需提前更新设定时，将相关操作实现于此。
        '''
        pass

    def physics_step(self):
    	'''
        子模块在仿真环境运行阶段相关操作实现于此。
        '''
        pass

    def post_physics_step(self):
    	'''
        子模块在仿真环境运行阶段化依赖于其它子模块仿真环境运行阶段运行结果，将相关操作实现于此。
        '''
        pass

    # 第三阶段，运行计算阶段
    def pre_compute(self):
    	'''
        子模块在运行计算阶段被其它模块依赖信息需提前更新设定时，将相关操作实现于此。
        '''
        pass

    def check_termination(self):
    	'''
        目前仅被 episode模块实现，用于计算环境是否终止
        '''
        pass

    def compute_reward(self):
    	'''
        目前仅被 rewards模块实现，用于计算reward
        '''
        pass

    def reset(self, env_ids):
    	'''
        用于重置终止环境
        '''
        pass

    def compute(self):
    	'''
        子模块在运行计算阶段相关操作实现于此。
        '''
        pass

    def post_compute(self):
    	'''
        子模块在运行计算阶段依赖于其它子模块运行计算结果，将相关操作实现于此。
        '''
        pass

    # 第四阶段，调试显示阶段
    def draw_debug_vis(self):
    	'''
        子模块绘制debug显示实现
        '''
        pass
'''

</details>

<details>
<summary>生命周期不同阶段</summary>

- init 初始化阶段
- physics_step 仿真运行阶段
- compute 计算阶段

</details>

<details>
<summary>依赖情况分别处理</summary>

- pre 被依赖，提前处理
-  不依赖(依赖)
- post 依赖,最后处理

将模块互相关系错开，尽可能不产生冲突
</details>

<details>
<summary>特殊接口</summary>

- check_termination 仅被 episode模块实现，用于计算环境是否终止
- compute_reward 目前仅被 rewards模块实现，用于计算reward
</details>


## register

所有子模块均通过注册的方式被 [BaseTask](humanoidverse/envs/base_task/base_task.py)创建
如 [register](humanoidverse/envs/locomotion/term/register.py)

<details>
<summary>子模块管理</summary>

- foundation 核心类子模块 最高级别
- status 状态类子模块
- assistant 输助类子模块
- mdp mdp类子模块
- sim2real sim2real类子模块
- extras extras类子模块
</details>


# agent
将agent抽像出不同的子模块，所有子模块生命周期均如 [BaseComponent](humanoidverse/agents/base_algo/base.py)
所示：


## BaseComponent

<details>

<summary>BaseComponent代码 .</summary>

```python
class BaseComponent():
    def __init__(self, _algo: base_algo.BaseAlgo):
        self.algo = _algo                     # 算法实例对像 （BaseAlgo 子类）
        self.num_envs = _algo.env.num_envs    # 环境实例数
        self.device = _algo.device            # 算法训练/评估环境运行设备
        self.config = _algo.config            # 配置

    # 第一阶段，初始化阶段
    def pre_init(self):
        '''
        子模块在初始化时被其它模块依赖信息，提前定义和初始化。
        '''
        pass

    def init(self):
        '''
        子模块初始化信息。
        '''
        pass

    def post_init(self):
        '''
        子模块初始化依赖于其它子模块，在此初始化。
        '''
        pass

    # 第二阶段
    # level 0
    def pre_loop(self, _inputs = None):
    	'''
        循环前处理调用
        '''
        pass

    def post_loop(self, _inputs = None):
    	'''
        循环后处理调用
        '''
        pass

    # level 1
    def pre_epoch(self, _inputs = None):
    	'''
        epoch 前处理调用
        '''
        pass

    def post_epoch(self, _inputs = None):
    	'''
        epoch 后处理调用
        '''
        pass

    # level 2
    def pre_step(self, _inputs = None):
    	'''
        子模块在step被其它模块依赖信息需提前更新设定时，将相关操作实现于此。
        '''
        pass

    def step(self, _inputs = None):
    	'''
        子模块在step相关操作实现于此。
        '''
        pass

    def post_step(self, _inputs = None):
    	'''
        子模块在step依赖于其它子模块时，将相关操作实现于此。
        '''
        pass

    # 辅助函数
    def load(self, _loaded_dict: dict):
    	'''
        加载初始化词典
        '''
        pass

    '''
    为保存信息
    '''
    def update(self, _state_dict: dict):
    	'''
        保存信息到词典
        '''
        pass

    '''
    输出训练信息到日志
    '''
    def write(self, writer, _it):
        pass


    '''
    仅适用于需要输出日志的组件
    def log_epoch(self, width, pad):
        pass

    '''

```
</details>

<details>

<summary>生命周期不同阶段</summary>

- init 初始化阶段
- loop 第一级循环
- epoch 第二级循环（一个训练周期)
- step  第三级（一个训练/数据收集步)
</details>


## register

所有子模块均通过注册的方式被 [BaseAlgo](humanoidverse/agents/base_algo/base_algo.py)创建
如 [register](humanoidverse/agents/ppo/register.py)

目前分为以下类

<details>
<summary>logic</summary>

- evaluater 评估子模块
- rollout 回放子模块
- statistics 输助(统计)子模块
- trainer 训练子模块
</details>


<details>
<summary>component</summary>

- envwarp task封装子模块
- modules 模型子模块
- optimizer 优化器子模块
- storage 回放封装子模块
</details>

h1 locomotion train  

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_h1_locomotion \
+robot=h1/h1_10dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=HumanoidLocomotion \
experiment_name=H110dof_loco_IsaacGym \
headless=True

HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+checkpoint=logs/HumanoidLocomotion/isaacgym/20250411_214128-H110dof_loco_IsaacGym-locomotion-h1_10dof/model_19700.pt

```
  

g1 locomotion train  

  
```bash
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
experiment_name=G123dof_loco_plane \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003
```

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003
+checkpoint=logs/Locomotion/isaacgym/20250517_124306-G123dof_loco_plane-locomotion-g1_29dof_anneal_23dof/model_26000.pt
```

g1 asap train  
  

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
project_name=ASAP \
experiment_name=MotionTracking_CR7 \
robot.motion.motion_file=data/motions/asap/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0


HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+checkpoint=logs/

```

g1 motion file play  
  

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/play_agent.py \
+simulator=isaacgym \
+exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
project_name=ASAP \
experiment_name=MotionTracking_CR7 \
robot.motion.motion_file=data/motions/asap/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl

```


h1 asap train  

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacsim45 \
+exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=h1/OmniH2Oh1 \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/motion_tracking \
project_name=ASAP \
experiment_name=OmniH2Oh1 \
robot.motion.motion_file=data/motions/OmniH2O/amass_phc_filtered.pkl \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0

```

h1 motion file play  

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/play_agent.py \
+simulator=isaacgym \
+exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=h1/OmniH2Oh1 \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
project_name=ASAP \
experiment_name=OmniH2Oh1 \
robot.motion.motion_file=data/motions/OmniH2O/stable_punch.pkl

```



# Citation
Please use the following bibtex if you find this repo helpful and would like to cite:

```bibtex
@misc{HumanoidVerse2,
  author = {liangjun},
  title = {HumanoidVerse2: A Multi-Simulator Framework with Modular Design for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zengliangjun/HumanoidVerse2}},
}


bibtex
@article{he2025asap,
  title={ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills},
  author={He, Tairan and Gao, Jiawei and Xiao, Wenli and Zhang, Yuanhang and Wang, Zi and Wang, Jiashun and Luo, Zhengyi and He, Guanqi and Sobanbabu, Nikhil and Pan, Chaoyi and Yi, Zeji and Qu, Guannan and Kitani, Kris and Hodgins, Jessica and Fan, Linxi "Jim" and Zhu, Yuke and Liu, Changliu and Shi, Guanya},
  journal={arXiv preprint arXiv:2502.01143},
  year={2025}
}


@misc{HumanoidVerse,
  author = {CMU LeCAR Lab},
  title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

