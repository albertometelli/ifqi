from spmi.envs.teacher_student import TeacherStudentEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spi_exact import SPI
from spmi.utils.uniform_policy import UniformPolicy
import numpy as np
from gym.utils import seeding

mdp = TeacherStudentEnv(n_literals=2,
                        max_value=1,
                        max_update=1,
                        max_literals_in_examples=2,
                        horizon=10)

uniform_policy = UniformPolicy(mdp)


dataset = collect_episodes(mdp, n_episodes=10)


eps = 0.00005
delta = 0.1
spi = SPI(mdp, eps, delta)
policy = spi.safe_policy_iteration_target_trick(uniform_policy)

