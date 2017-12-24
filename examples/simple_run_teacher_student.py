from spmi.envs.teacher_student import TeacherStudentEnv
from spmi.evaluation.evaluation import collect_episodes

mdp = TeacherStudentEnv(n_literals=5,
                        max_value=2,
                        max_update=1,
                        max_literals_in_examples=2,
                        horizon=10)

dataset = collect_episodes(mdp, n_episodes=10)

