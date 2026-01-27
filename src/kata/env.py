import gymnasium as gym
import numpy as np
from collections import deque

from kata.funcs import step_prod_line


class KataEnv(gym.Env):
    metadata = {"render_modes": ["numpy", "human"], "render_fps": 4}

    def __init__(self, config: dict):
        super(KataEnv, self).__init__()
        self.config = config

        # Assigned tickets vectors
        self.assigned_tickets = {
            "ids": np.fill(len(config["technicians"]), -1),
            "priorities": np.fill(len(config["technicians"]), -1),
            "time_to_complete": np.fill(len(config["technicians"]), -1),
            "machine_ids": np.fill(len(config["technicians"]), -1),
            "components": np.fill(len(config["technicians"]), ""),
            "type_of_failure": np.fill(len(config["technicians"]), ""),
        }

        # Pending tickets dequeue
        self.pending_tickets = deque()

        # Global step counter
        self.episode_step = 0

    def _get_next_obs(self):
        pass

    def _get_info(self):
        pass

    def _get_reward(self):
        pass

    def _get_done(self):
        pass

    def _get_truncated(self):
        pass

    def reset(self, seed=None):
        pass

    def step(self, action):
        # The action is the index of the technician to assign the next ticket to
        action = int(action)
        # Check if the action is valid (i.e. if the technician does not have an assigned ticket)
        if self.assigned_tickets["ids"][action] != -1:
            invalid = True

        # Assign the ticket to the technician
        # Update the assigned tickets vectors
        self.assigned_tickets["ids"][action] = self.current_ticket["id"]
        self.assigned_tickets["priorities"][action] = self.current_ticket["priority"]
        self.assigned_tickets["time_to_complete"][action] = self.current_ticket[
            "time_to_complete"
        ]
        self.assigned_tickets["machine_ids"][action] = self.current_ticket["machine_id"]
        self.assigned_tickets["components"][action] = self.current_ticket["components"]
        self.assigned_tickets["type_of_failure"][action] = self.current_ticket[
            "type_of_failure"
        ]

        # Compute the reward
        reward = self._get_reward()

        # Compute if the episode is done
        done = self._get_done() or invalid

        # Compute the info
        info = self._get_info()

        ### Transition to the next state
        while True:
            ### Get the next ticket
            # 1. If there are pending tickets, pick the next one
            while len(self.pending_tickets) == 0:
                for i, production_line in enumerate(self.production_lines):
                    # Step the production line
                    (
                        self.status[i],
                        self.in_buff[i],
                        self.out_buff[i],
                        self.prod_completions[i],
                        self.absolute_times[i],
                        self.s_since_repairs[i],
                        self.virtual_ages[i],
                    ) = step_prod_line(
                        self.status[i],
                        self.in_buff[i],
                        self.out_buff[i],
                        self.prod_completions[i],
                        self.absolute_times[i],
                        self.s_since_repairs[i],
                        self.virtual_ages[i],
                        production_line["prod_rates"],
                        production_line["prod_costs"],
                        production_line["in_max_cap"],
                        production_line["out_max_cap"],
                        production_line["weibull_ks"],
                        production_line["weibull_inv_lambdas"],
                    )
                    self.episode_step += 1

                    # Check if a machine broke, if so, create a ticket and add it to the pending tickets
                    idxes_broken = np.where(self.status[i] == -1)[0]

                    # Check if any of the broken machines already have a ticket assigned
                    for idx in idxes_broken:
                        if idx not in self.assigned_tickets["machine_ids"]:
                            # Create a ticket for the broken machine
                            ticket = create_ticket(args)
                            self.pending_tickets.append(ticket)

                # 2. Check if a technician is currently available, if no technician is available, skip to the next step
            if any(self.assigned_tickets["ids"] == -1):
                # If there's at least one avaialble technician, finish the step
                break
            else:
                # If no technician is available, continue stepping the production lines
                continue
        self.current_ticket = self.pending_tickets.popleft()

        next_obs = self._get_next_obs(self.current_ticket)

        return next_obs, reward, done, self.episode_step < self.max_episode_steps, info

