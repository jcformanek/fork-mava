# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type: ignore
import time
from typing import Dict, NamedTuple

import dm_env
import gym
import numpy as np
from acme import specs, types
from mava.utils.sort_utils import sort_str_num


def rad_rot_to_xy(rad_rot):
    return np.cos(rad_rot), np.sin(rad_rot)


def deg_rot_to_xy(deg_rot):
    return np.cos(deg_rot * np.pi / 180), np.sin(deg_rot * np.pi / 180)


def should_wait(wait_list):
    should_wait = False
    for entity in wait_list:
        if not entity.wm.new_data:
            should_wait = True
            break
    return should_wait


def wait_for_next_observations(obs_to_wait_for):
    """
    Wait for next observation before agents should think again.
    """
    # Wait for new data
    waiting = should_wait(obs_to_wait_for)

    # start = time.time()
    while waiting:
        waiting = should_wait(obs_to_wait_for)
        time.sleep(0.0001)
    # end = time.time()
    # print("Wait time: ", end-start)

    # Set new data false
    for entity in obs_to_wait_for:
        entity.wm.new_data = False


# Dummy class to get the observation in the correct format
class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


class SpecWrapper(dm_env.Environment):
    """Spec wrapper for 2D RoboCup environment."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self, num_players: int):
        self._reset_next_step = True

        self.scaling = 200.0

        # Chose action
        act_min = [0.0] * 7  # 6 + No action
        act_max = [1.0] * 7  # 6 + No action

        # Action continuous component
        # All directions are in x, y format
        act_min.extend(
            [
                -100 / self.scaling,
                -1,
                -1,  # dash (power, direction)
                0,
                -1,
                -1,  # kick (power, direction)
                0,
                0,  # change_view (width, quality)
                -1,
                -1,
                0,  # tackle (direction, foul)
                -1,
                -1,  # turn (direction)
                -1,
                -1,
            ]
        )  # turn_neck(direction)

        act_max.extend([100 / self.scaling, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        assert len(act_min) == len(act_max)
        action_spec = specs.BoundedArray(
            shape=(len(act_min),),
            dtype="float32",
            name="action",
            minimum=act_min,
            maximum=act_max,
        )

        self.action_size = action_spec.shape[0]

        # obs_dict = {"time_left": 0, "side": 1, "sense_self": 2,
        #  "coords": (3, 5), "body_dir": (5, 7),
        #             "head_dir": (7, 9), "width": (9, 12),
        # "quality": 13, "stamina": 14, "effort": 15,
        #             "speed_amount": 16, "speed_direction": (17, 19),
        # "neck_direction": (19, 21),
        #             "see_ball": 21, "ball_dist": 22,
        # "ball_dir": (23, 25), "ball_dist_change": 25,
        #             "ball_dir_change": 26, "ball_speed": 27,
        # "last_action": (28, 28 + self.action_size),
        #             }

        # TODO: Check if all bounds are correct
        obs_min = [
            0.0,  # time_left
            0.0,  # sense_self
            -100 / self.scaling,
            -50 / self.scaling,  # coords
            -1,
            -1,  # body_dir
            -1,
            -1,  # head_dir
            0,
            0,
            0,  # width
            0,  # quality
            0,  # stamina
            0,  # effort
            0,  # speed_amount
            -1,
            -1,  # speed_direction
            -1,
            -1,  # neck_direction
            0,  # see_ball
            0,  # ball_dist
            -1,
            -1,  # ball_dir
            -100 / self.scaling,  # ball_dist_change
            -180 / self.scaling,  # ball_dir_change
            0,  # ball_speed
        ]

        obs_max = [
            1.0,  # time_left
            1.0,  # sense_self
            100 / self.scaling,
            50 / self.scaling,  # coords
            1,
            1,  # body_dir
            1,
            1,  # head_dir
            1,
            1,
            1,  # width
            1,  # quality
            1,  # stamina
            1,  # effort
            100 / self.scaling,  # speed_amount
            1,
            1,  # speed_direction
            1,
            1,  # neck_direction
            1,  # see_ball
            100 / self.scaling,  # ball_dist
            1,
            1,  # ball_dir
            100 / self.scaling,  # ball_dist_change
            180 / self.scaling,  # ball_dir_change
            100 / self.scaling,  # ball_speed
        ]

        # Last action
        obs_min.extend(action_spec.minimum)
        obs_max.extend(action_spec.maximum)
        assert len(obs_min) == len(obs_max)
        self.player_obs_size = len(obs_min)
        player_obs_spec = specs.BoundedArray(
            shape=(len(obs_min),),
            dtype="float32",
            name="ff_observation",
            minimum=obs_min,
            maximum=obs_max,
        )
        self.num_agents = num_players

        # See player
        # player.distance / self.scaling,
        # player.direction / self.scaling,
        # player.dist_change / self.scaling,
        # player.dir_change / self.scaling,
        # player.speed / self.scaling,
        # b_dir_x,
        # b_dir_y,
        # h_dir_x,
        # h_dir_y,
        obs_min = [0, -200 / self.scaling] + [-1]*8
        obs_max = [1, +200 / self.scaling] + [1]*8
        assert len(obs_min) == len(obs_max)
        self.other_agent_obs_size = len(obs_min)
        agent_obs_spec = specs.BoundedArray(
            shape=(len(obs_min),),
            dtype="float32",
            name="ff_observation",
            minimum=obs_min,
            maximum=obs_max,
        )
        # First (num_players/2)-1 is team and second num_players/2 is enemies
        obs_spec = [player_obs_spec] + [agent_obs_spec] * (num_players - 1)

        self.agent_keys = ["player_" + str(r) for r in range(num_players)]

        self._observation_specs = {}
        self._action_specs = {}

        # Time_left, ball coords, ball delta_coords
        state_min = [0, -100 / self.scaling, -100 / self.scaling, -10, -10]
        state_max = [1, 100 / self.scaling, 100 / self.scaling, 10, 10]
        assert len(state_min) == len(state_max)
        self._time_ball_state_size = len(state_min)
        time_ball_state_spec = specs.BoundedArray(
            shape=(len(state_min),),
            dtype="float32",
            name="state",
            minimum=state_min,
            maximum=state_max,
        )


        # First player is the critic player
        # Coords, delta_coords, body_angle (x, y format),
        # head_angle (x, y format)
        state_min = [
            -100 / self.scaling,
            -100 / self.scaling,
            -10,
            -10,
            -1,
            -1,
            -1,
            -1,
        ]

        state_max = [
            +100 / self.scaling,
            +100 / self.scaling,
            +10,
            +10,
            1,
            1,
            1,
            1,
        ]
        assert len(state_min) == len(state_max)
        self._agent_state_size = len(state_min)
        agent_state_spec = specs.BoundedArray(
            shape=(len(state_min),),
            dtype="float32",
            name="state",
            minimum=state_min,
            maximum=state_max,
        )
        per_agent_state_spec = [time_ball_state_spec] + [agent_state_spec] * num_players

        self._state_spec = {}
        for agent in self.agent_keys:
            # TODO: Why is the action spec in two places?
            self._observation_specs[agent] = OLT(
                observation=obs_spec,
                legal_actions=action_spec,
                terminal=specs.Array((1,), np.float32),
            )

            self._state_spec[agent] = per_agent_state_spec

            self._action_specs[agent] = action_spec

        self._discount = dict(zip(self.agent_keys, [np.float32(1.0)] * len(self.agent_keys)))
    def reset(self):
        pass

    def step(self):
        pass

    def observation_spec(self) -> types.NestedSpec:
        return self._observation_specs

    def action_spec(self) -> types.NestedSpec:
        return self._action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent_key in self.agent_keys:
            reward_specs[agent_key] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self.agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {"env_states": self._state_spec}

    def _proc_robocup_obs(
        self, observations: Dict, done: bool, nn_actions: Dict = None
    ) -> Dict:
        # TODO: Try to automatically normalise by min max boundries
        processed_obs_dict = {}
        for agent_key in self.agent_keys:
            env_agent_obs = observations[agent_key]

            if nn_actions:
                last_action = nn_actions[agent_key]
            else:
                last_action = None

            proc_agent_obs = self.proc_agent_env_obs(env_agent_obs, last_action)

            observation = OLT(
                observation=proc_agent_obs,
                legal_actions=np.ones(self.action_size, dtype=np.float32),
                terminal=np.asarray([done], dtype=np.float32),
            )
            processed_obs_dict[agent_key] = observation
        return processed_obs_dict

    def proc_agent_env_obs(self, env_agent_obs, last_action):  # noqa: C901
        # All angles is in x,y format
        proc_agent_obs = [np.zeros(self.player_obs_size, dtype=np.float32)] + [
            np.zeros(self.other_agent_obs_size, dtype=np.float32)
        ] * (self.num_agents - 1)
        obs_dict = {
            "time_left": 0,
            "sense_self": 1,
            "coords": (2, 4),
            "body_dir": (4, 6),
            "head_dir": (6, 8),
            "width": (8, 11),
            "quality": 11,
            "stamina": 12,
            "effort": 13,
            "speed_amount": 14,
            "speed_direction": (15, 17),
            "neck_direction": (17, 19),
            "see_ball": 19,
            "ball_dist": 20,
            "ball_dir": (21, 23),
            "ball_dist_change": 23,
            "ball_dir_change": 24,
            "ball_speed": 25,
            "last_action": (26, 26 + self.action_size),
        }

        assert self.player_obs_size == 26 + self.action_size

        # Time left obs
        if "game_step" in env_agent_obs and "game_length" in env_agent_obs:
            # Time left
            proc_agent_obs[0][obs_dict["time_left"]] = (
                1 - env_agent_obs["game_step"] / env_agent_obs["game_length"]
            )

        # Team side obs
        focus_agent_side = env_agent_obs["side"]

        if (
            "estimated_abs_coords" in env_agent_obs
            and env_agent_obs["estimated_abs_coords"][0] is not None
        ):
            # see_own_stats
            proc_agent_obs[0][obs_dict["sense_self"]] = 1.0

            # coords
            coords = env_agent_obs["estimated_abs_coords"]
            s, e = obs_dict["coords"]
            proc_agent_obs[0][s] = float(coords[0]) / self.scaling
            proc_agent_obs[0][e - 1] = float(coords[1]) / self.scaling

            # body_angle
            s, e = obs_dict["body_dir"]
            proc_agent_obs[0][s:e] = deg_rot_to_xy(env_agent_obs["estimated_abs_body_dir"])

            # head_angle
            s, e = obs_dict["head_dir"]
            proc_agent_obs[0][s:e] = deg_rot_to_xy(env_agent_obs["estimated_abs_neck_dir"])

            # view_width
            w_to_int = {"narrow": 0, "normal": 1, "wide": 2}
            onehot = np.zeros(3)
            onehot[w_to_int[env_agent_obs["view_width"]]] = 1
            s, e = obs_dict["width"]
            proc_agent_obs[0][s:e] = onehot

            # view_quality
            q_to_int = {"high": 0, "low": 1}
            proc_agent_obs[0][obs_dict["quality"]] = q_to_int[
                env_agent_obs["view_quality"]
            ]

            # stamina
            proc_agent_obs[0][obs_dict["stamina"]] = env_agent_obs["stamina"] / 8000

            # effort
            proc_agent_obs[0][obs_dict["effort"]] = env_agent_obs["effort"]

            # speed_amount
            proc_agent_obs[0][obs_dict["speed_amount"]] = (
                env_agent_obs["speed_amount"] / self.scaling
            )

            # speed_dir
            s, e = obs_dict["speed_direction"]
            proc_agent_obs[0][s:e] = deg_rot_to_xy(env_agent_obs["speed_direction"])

            # Relative neck dir
            s, e = obs_dict["neck_direction"]
            proc_agent_obs[0][s:e] = deg_rot_to_xy(env_agent_obs["neck_direction"])

        # See_ball, ball_distance, ball_direction, dist_change, dir_change, speed
        if "ball" in env_agent_obs and env_agent_obs["ball"] is not None:
            # Has ball flag
            proc_agent_obs[0][obs_dict["see_ball"]] = 1.0

            if env_agent_obs["ball"].distance is not None:
                proc_agent_obs[0][obs_dict["ball_dist"]] = (
                    env_agent_obs["ball"].distance / self.scaling
                )

            if env_agent_obs["ball"].direction is not None:
                s, e = obs_dict["ball_dir"]
                proc_agent_obs[0][s:e] = deg_rot_to_xy(env_agent_obs["ball"].direction)

            if env_agent_obs["ball"].dist_change is not None:
                proc_agent_obs[0][obs_dict["ball_dist_change"]] = (
                    env_agent_obs["ball"].dist_change / self.scaling
                )

            if env_agent_obs["ball"].dir_change is not None:
                proc_agent_obs[0][obs_dict["ball_dir_change"]] = (
                    env_agent_obs["ball"].dir_change / self.scaling
                )

            if env_agent_obs["ball"].speed is not None:
                proc_agent_obs[0][obs_dict["ball_speed"]] = (
                    env_agent_obs["ball"].speed / self.scaling
                )

        # Last player actions
        if last_action is not None:
            s, e = obs_dict["last_action"]
            proc_agent_obs[0][s:e] = last_action

        # TODO: Why am I not using the other atributes as well?
        # [see_player, is_on_team, distance, direction, dist_change,
        # dir_change, speed, body_direction, neck_direction]
        if "players" in env_agent_obs and len(env_agent_obs["players"]) > 0:
            # ["opponent", player.distance, player.direction (x, y format)]
            players = env_agent_obs["players"]

            # num_see_players = len(players)
            # player_type_dict = {"opponent": 0.0, "team": 1.0}

            last_team_i = 1
            last_opp_i = int(self.num_agents/2)
            for p_i, player in enumerate(players):
                # distance,
                # direction,
                # dist_change,
                # dir_change,
                # speed,
                # team,
                # side,
                # body_direction,
                # neck_direction,

                if focus_agent_side is not None and player.side is not None:
                    # Get use i
                    if player.side == focus_agent_side:
                        use_i = last_team_i
                        last_team_i += 1
                    else:
                        use_i = last_opp_i
                        last_opp_i += 1

                    # player_direction (x, y format)
                    if player.body_direction is not None:
                        b_dir_x, b_dir_y = deg_rot_to_xy(player.body_direction)
                    else:
                        b_dir_x = 0.0
                        b_dir_y = 0.0

                    if player.neck_direction is not None:
                        h_dir_x, h_dir_y = deg_rot_to_xy(player.neck_direction)
                    else:
                        h_dir_x = 0.0
                        h_dir_y = 0.0

                    assert use_i < self.num_agents
                    proc_agent_obs[use_i] = [
                        1.0,
                        player.distance / self.scaling if player.distance is not None else 0.0,
                        player.direction / self.scaling if player.direction is not None else 0.0,
                        player.dist_change / self.scaling if player.dist_change is not None else 0.0,
                        player.dir_change / self.scaling if player.dir_change is not None else 0.0,
                        player.speed / self.scaling if player.speed is not None else 0.0,
                        b_dir_x,
                        b_dir_y,
                        h_dir_x,
                        h_dir_y,
                        ]

        # if not env_agent_obs["obs_updated"]:
        #     proc_agent_obs[2] = 0.0
        #
        return proc_agent_obs

    def _proc_robocup_state(self, state) -> Dict:
        # TODO: Try to automatically normalise by min max boundries
        processed_state_dict = {}
        for state_i, agent_key in enumerate(sort_str_num(self.agent_keys)):

            sign = 1
            if state["players"][state_i]["side"] == 1:
                sign = -1

            processed_state_dict[agent_key] = self._proc_agent_state(state, sign)
        return processed_state_dict

    def _proc_agent_state(self, state: Dict, sign) -> np.array:
        state_dict = {
            "time_left": 0,
            "ball_coords": (1, 3),
            "ball_delta": (3, 5),
            "p_coords": (0, 2),
            "p_delta": (2, 4),
            "p_body_dir": (4,6),
            "p_neck_dir": (6, 8),
        }
        assert self._time_ball_state_size == 5
        assert self._agent_state_size == 8

        proc_agent_state = [np.zeros(self._time_ball_state_size, dtype=np.float32)] + [np.zeros(self._agent_state_size, dtype=np.float32)] * self.num_agents

        # Time left
        proc_agent_state[0][state_dict["time_left"]] = (
            1 - state["game_step"] / state["game_length"]
        )

        # Ball:
        ball = state["ball"]
        s, e = state_dict["ball_coords"]
        proc_agent_state[0][s:e] = [
            sign * float(ball["coords"][0]) / self.scaling,
            sign * float(ball["coords"][1]) / self.scaling,
        ]
        s, e = state_dict["ball_delta"]
        proc_agent_state[0][s:e] = [
            sign * float(ball["delta_coords"][0]),
            sign * float(ball["delta_coords"][1]),
        ]  # TODO: Should delta coords have a normaliser?

        # TODO: Add check to see if players are in the correct order
        players = state["players"]
        if players:
            for i in range(self.num_agents):
                if len(players) > i:
                    player = players[i]
                    # 'side': 0, 'coords': (52.6498, 0.54963),
                    # 'delta_coords': (0.000227909, 0.00371977), 'body_angle': 163,
                    # 'neck_angle': 0}
                    s, e = state_dict["p_coords"]
                    proc_agent_state[i+1][s:e] = [
                        sign * player["coords"][0] / self.scaling,
                        sign * player["coords"][1] / self.scaling,
                    ]
                    s, e = state_dict["p_delta"]
                    # TODO: Should delta coords have a normaliser?
                    x, y = player["delta_coords"]
                    proc_agent_state[i+1][s:e] = [sign * x, sign * y]
                    s, e = state_dict["p_body_dir"]
                    x, y = deg_rot_to_xy(player["body_angle"])
                    proc_agent_state[i+1][s:e] = [sign * x, sign * y]
                    s, e = state_dict["p_neck_dir"]
                    x, y = deg_rot_to_xy(player["neck_angle"])
                    proc_agent_state[i+1][s:e] = [sign * x, sign * y]

                else:
                    break

        # # Add agent observations
        # start_i = offset + self.num_agents * p_size
        # for agent_i, agent_key in enumerate(self.agents):
        #     obs = proc_obs[agent_key].observation
        #     proc_agent_state[
        #         start_i
        #         + agent_i * self.obs_size : start_i
        #         + (agent_i + 1) * self.obs_size
        #     ] = obs

        return proc_agent_state

    def _proc_robocup_actions(self, actions: Dict) -> Dict:
        # dash (speed), turn (direction), kick (power, direction)
        processed_action_dict = {}

        for agent_key in actions.keys():
            action = actions[agent_key]
            assert len(action) == self.action_size
            processed_action_dict[agent_key] = self.proc_agent_action(action)

        return processed_action_dict

    def proc_agent_action(self, action):
        # TODO: Add catch, tackle, move, pointto, attentionto and say commands as well.
        int_to_command = [
            "dash",
            "kick",
            "change_view",
            "tackle",
            "turn",
            "turn_neck",
            "none",
        ]

        # Remove change_view and turn_neck action
        # TODONE: Remove this
        # action[2] = 0
        # action[5] = 0

        command = int_to_command[np.argmax(action[0 : len(int_to_command)])]

        # Do the command
        assert len(int_to_command) == 7

        act_dict = {
            "dash_pow": 7,
            "dash_dir": (8, 10),
            "kick_pow": 10,
            "kick_dir": (11, 13),
            "width": 13,
            "quality": 14,
            "tackle_dir": (15, 17),
            "tackle_foul": 17,
            "turn": (18, 20),
            "neck": (20, 22),
        }

        if command == "dash":
            power = action[act_dict["dash_pow"]] * self.scaling
            s, e = act_dict["dash_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(dash " + str(power) + " " + str(dir) + ")"
        elif command == "kick":
            power = action[act_dict["kick_pow"]] * self.scaling
            s, e = act_dict["kick_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(kick " + str(power) + " " + str(dir) + ")"
        elif command == "change_view":
            w_to_text = ["narrow", "normal", "wide"]
            width = w_to_text[int(action[act_dict["width"]] * 2.99)]
            q_to_text = ["high", "low"]
            quality = q_to_text[int(action[act_dict["quality"]] * 1.99)]
            robocup_action = "(change_view " + width + " " + quality + ")"
        elif command == "tackle":
            s, e = act_dict["tackle_dir"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            f_to_text = ["true", "false"]
            foul = f_to_text[int(action[act_dict["tackle_foul"]] * 1.99)]
            robocup_action = "(tackle " + str(dir) + " " + foul + ")"
        elif command == "turn":
            s, e = act_dict["turn"]
            dir_x, dir_y = action[s:e]
            dir = np.arctan2(dir_y, dir_x) * 180 / np.pi
            robocup_action = "(turn " + str(dir) + ")"
        elif command == "turn_neck":
            s, e = act_dict["neck"]
            x, y = action[s:e]
            turn_neck_dir = np.arctan2(y, x) * 180 / np.pi
            robocup_action = "(turn_neck " + str(turn_neck_dir) + ")"
        elif command == "none":
            robocup_action = "(done)"
        else:
            raise NotImplementedError("Command not implemented: ", command)
        return robocup_action

    @property
    def possible_agents(self) -> gym.Env:
        """Returns the number of possible agents."""
        return self.agent_keys
