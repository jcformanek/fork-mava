import dm_env
import numpy as np

from mava.adders.reverb import base
from mava.utils.wrapper_utils import parameterized_restart, parameterized_termination

# TODO Clean this up, when using newer versions of acme.
try:
    from acme.adders.reverb.sequence import EndBehavior
except ImportError:
    from acme.adders.reverb.sequence import EndOfEpisodeBehavior as EndBehavior

agents = {"agent_0", "agent_1", "agent_2"}
reward_step1 = {"agent_0": 0.0, "agent_1": 0.0, "agent_2": 1.0}
reward_step2 = {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0}
reward_step3 = {"agent_0": 0.0, "agent_1": 1.0, "agent_2": 0.0}
reward_step4 = {"agent_0": 1.0, "agent_1": 1.0, "agent_2": 1.0}
reward_step5 = {"agent_0": -1.0, "agent_1": -1.0, "agent_2": -1.0}
reward_step6 = {"agent_0": 0.5, "agent_1": -5.0, "agent_2": 1.0}
reward_step7 = {"agent_0": 1.0, "agent_1": 3.0, "agent_2": 1.0}

obs_first = {agent: np.array([0.0, 1.0]) for agent in agents}
obs_step1 = {agent: np.array([1.0, 2.0]) for agent in agents}
obs_step2 = {agent: np.array([2.0, 3.0]) for agent in agents}
obs_step3 = {agent: np.array([3.0, 4.0]) for agent in agents}
obs_step4 = {agent: np.array([4.0, 5.0]) for agent in agents}
obs_step5 = {agent: np.array([5.0, 6.0]) for agent in agents}
obs_step6 = {agent: np.array([6.0, 7.0]) for agent in agents}
obs_step7 = {agent: np.array([7.0, 8.0]) for agent in agents}

default_discount = {agent: 1.0 for agent in agents}
default_action = {agent: 0.0 for agent in agents}
env_restart = parameterized_restart(
    reward={agent: 0.0 for agent in agents},
    discount=default_discount,
    observation=obs_first,
)

final_step_discount = {agent: 0.0 for agent in agents}


TEST_CASES = [
    dict(
        testcase_name="ShortEpsPeriodOne",
        sequence_length=3,
        period=1,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {},
                ),
            ],
            [
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step4,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ShortEpsPeriodOneWithExtras",
        sequence_length=3,
        period=1,
        first=(env_restart, {"state": -1}),
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
                {"state": 0},
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
                {"state": 1},
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
                {"state": 2},
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=final_step_discount,
                ),
                {"state": 3},
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first,
                    default_action,
                    reward_step1,
                    default_discount,
                    True,
                    {"state": -1},
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    reward_step2,
                    default_discount,
                    False,
                    {"state": 0},
                ),
                base.Trajectory(
                    obs_step2,
                    default_action,
                    reward_step3,
                    default_discount,
                    False,
                    {"state": 1},
                ),
            ],
            [
                base.Trajectory(
                    obs_step1,
                    default_action,
                    reward_step2,
                    default_discount,
                    False,
                    {"state": 0},
                ),
                base.Trajectory(
                    obs_step2,
                    default_action,
                    reward_step3,
                    default_discount,
                    False,
                    {"state": 1},
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {"state": 2},
                ),
            ],
            [
                base.Trajectory(
                    obs_step2,
                    default_action,
                    reward_step3,
                    default_discount,
                    False,
                    {"state": 1},
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {"state": 2},
                ),
                base.Trajectory(
                    obs_step4,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {"state": 3},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ShortEpsPeriodOneEarlyTermination",
        sequence_length=3,
        period=1,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    reward_step2,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step2,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ShortEpsPeriodOneEarlyTerminationWithPadding",
        sequence_length=4,
        period=1,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    reward_step2,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step2,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    {
                        agent: np.zeros_like(obs_step2[list(agents)[0]])
                        for agent in agents
                    },
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="ShortEpsPeriodOneEarlyTerminationNoPadding",
        sequence_length=4,
        period=1,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1,
                    default_action,
                    reward_step2,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step2,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
        end_behavior=EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name="ShortEpsPeriodTwo",
        sequence_length=3,
        period=2,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3,
                    default_action,
                    reward_step4,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step4,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    final_step_discount,
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="LongEpsPadding",
        sequence_length=3,
        period=3,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step5,
                    observation=obs_step5,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step6,
                    observation=obs_step6,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step7,
                    observation=obs_step7,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step5, default_action, reward_step6, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step6,
                    default_action,
                    reward_step7,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step7,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
                base.Trajectory(
                    {
                        agent: np.zeros_like(obs_step7[list(agents)[0]])
                        for agent in agents
                    },
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
    ),
    dict(
        testcase_name="LongEpsNoPadding",
        sequence_length=3,
        period=3,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step5,
                    observation=obs_step5,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step6,
                    observation=obs_step6,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step7,
                    observation=obs_step7,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step5, default_action, reward_step6, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step6,
                    default_action,
                    reward_step7,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step7,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
        end_behavior=EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name="LongEpsContinuePeriodTwo",
        sequence_length=3,
        period=2,
        first=env_restart,
        steps=(
            (
                default_action,
                dm_env.transition(
                    reward=reward_step1,
                    observation=obs_step1,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step2,
                    observation=obs_step2,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step3,
                    observation=obs_step3,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step4,
                    observation=obs_step4,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                dm_env.transition(
                    reward=reward_step5,
                    observation=obs_step5,
                    discount=default_discount,
                ),
            ),
            (
                default_action,
                parameterized_termination(
                    reward=reward_step6,
                    observation=obs_step6,
                    discount=final_step_discount,
                ),
            ),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode,  next_extras)
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step5,
                    default_action,
                    reward_step6,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step6,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
            [
                base.Trajectory(
                    obs_step6,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step5,
                    default_action,
                    reward_step6,
                    final_step_discount,
                    False,
                    {},
                ),
            ],
            [
                base.Trajectory(
                    obs_step5,
                    default_action,
                    reward_step6,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step6,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_first, default_action, reward_step1, default_discount, True, {}
                ),
                base.Trajectory(
                    obs_step1, default_action, reward_step2, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step2, default_action, reward_step3, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step3, default_action, reward_step4, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
            ],
            [
                base.Trajectory(
                    obs_step4, default_action, reward_step5, default_discount, False, {}
                ),
                base.Trajectory(
                    obs_step5,
                    default_action,
                    reward_step6,
                    final_step_discount,
                    False,
                    {},
                ),
                base.Trajectory(
                    obs_step6,
                    default_action,
                    {agent: 0.0 for agent in agents},
                    {agent: 0.0 for agent in agents},
                    False,
                    {},
                ),
            ],
        ),
        agents=agents,
        end_behavior=EndBehavior.CONTINUE,
        repeat_episode_times=3,
    ),
]
