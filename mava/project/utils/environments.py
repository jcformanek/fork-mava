import gym
import dm_env
import numpy as np
from acme import specs
from mava import types

class IndependentCartPole:

    def __init__(self, num_agents):
        self._agents = [f"agent_{id}" for id in range(num_agents)]

        self._envs = {}
        for agent in self._agents:
            self._envs[agent] = gym.make("CartPole-v0")

    def reset(self):
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        observations = {}
        legal_actions = {}
        for agent, env in self._envs.items():
            observations[agent] = env.reset()
            legal_actions[agent] = np.ones(2, "int64")

        OLT = self._convert_observations(
            observations, legal_actions, self._done
        )

        self._discounts = {
            agent: np.ones((), "float32")
            for agent in self._agents
        }

        rewards = {
            agent: np.zeros((), "float32")
            for agent in self._agents
        }

        extras = {"s_t": np.concatenate(list(observations.values()), dtype="float32")}

        timestep = dm_env.TimeStep(self._step_type, rewards, self._discounts, OLT)

        return timestep, extras

    def step(self, actions):
        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        next_observations = {}
        rewards = {}
        dones = {}
        legal_actions = {}
        for agent in self._agents:
            next_observations[agent], rewards[agent], dones[agent], _ = self._envs[agent].step(actions[agent])
            legal_actions[agent] = np.ones(2, "int64")

        for agent in self._agents:
            rewards[agent] = np.asarray(rewards[agent], "float32")

        self._done = all(list(dones.values()))

        OLT = self._convert_observations(
            next_observations, legal_actions, self._done
        )

        next_extras = {"s_t": np.concatenate(list(next_observations.values()), dtype="float32")}

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
            self._done = True

            # Discount on last timestep set to zero
            self._discounts = {
                agent: np.zeros((), "float32")
                for agent in self._agents
            }
        else:
            self._step_type = dm_env.StepType.MID

        next_timestep = dm_env.TimeStep(self._step_type, rewards, self._discounts, OLT)

        return next_timestep, next_extras

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

    def _convert_observations(
        self, observations, legal_actions, done
    ):
        olt_observations = {}
        for agent in self._agents:

            olt_observations[agent] = types.OLT(
                observation=np.asarray(observations[agent], "float32"),
                legal_actions=np.asarray(legal_actions[agent], "int64"),
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def extra_spec(self):
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        return {"s_t": np.ones(4*len(self._agents), "float32")}

    def observation_spec(self):
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            observation_specs[agent] = types.OLT(
                observation=np.ones(4, "float32"),
                legal_actions=np.ones(2, "int64"),
                terminal=np.asarray(True, dtype=np.float32),
            )

        return observation_specs

    def action_spec(
        self,
    ):
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=2, dtype="int64"
            )
        return action_specs

    def reward_spec(self):
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self._agents:
            reward_specs[agent] = np.ones((), "float32")
        return reward_specs

    def discount_spec(self):
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self._agents:
            discount_specs[agent] = np.ones((), "float32")
        return discount_specs

    @property
    def agents(self):
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._agents

    @property
    def possible_agents(self):
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._agents