import gymnasium as gym
import numpy as np
from collections import deque

from kata.funcs import step_prod_line


def create_ticket(
    ticket_id: int,
    priority: int,
    time_to_complete: int,
    machine_id: int,
    component: str,
    type_of_failure: str,
) -> dict:
    """
    Create a ticket dictionary for a machine repair request.
    
    Args:
        ticket_id: Unique identifier for the ticket
        priority: Priority level of the ticket (higher is more urgent)
        time_to_complete: Estimated time to complete the repair
        machine_id: ID of the machine that needs repair
        component: Component that failed
        type_of_failure: Type of failure that occurred
        
    Returns:
        Dictionary containing ticket information
    """
    return {
        "id": ticket_id,
        "priority": priority,
        "time_to_complete": time_to_complete,
        "machine_id": machine_id,
        "component": component,
        "type_of_failure": type_of_failure,
    }


class KataEnv(gym.Env):
    """
    KataEnv - Knowledge-Aware Technician Assignment Environment
    
    A Gymnasium-compatible reinforcement learning environment for simulating
    technician assignment in manufacturing production lines with machine breakdowns.
    
    The environment models:
    - Multiple production lines with machines that can fail
    - Technicians who can be assigned to repair broken machines
    - Tickets representing repair requests with priorities and completion times
    - Machine degradation using Weibull failure distributions and Kijima repair models
    
    The agent's goal is to assign repair tickets to technicians efficiently to
    minimize downtime and maximize production throughput.
    
    Observation Space:
        A Box space containing:
        - Current ticket features (id, priority, time_to_complete, machine_id)
        - Technician states (assigned tickets and their attributes)
        - Production line states (machine statuses, buffers, production)
    
    Action Space:
        Discrete space representing which technician to assign the current ticket to.
        The action is an integer in [0, n_technicians-1].
    
    Reward:
        The reward considers:
        - Penalty for pending tickets (encourages fast assignment)
        - Penalty for machines in maintenance state (encourages uptime)
        - Small time penalty (encourages episode efficiency)
    
    Episode Termination:
        - Terminated: All machines are broken (catastrophic failure)
        - Truncated: Max episode steps reached
    
    Config Dictionary:
        technicians: List of technician configurations
        production_lines: List of production line configurations, each containing:
            - prod_rates: Production rates per machine
            - prod_costs: Steps required to complete a product
            - in_max_cap: Input buffer max capacities
            - out_max_cap: Output buffer max capacities
            - weibull_ks: Weibull shape parameters (failure distribution)
            - weibull_inv_lambdas: Weibull scale parameters (1/lambda)
            - initial_in_buff: Initial input buffer level
        max_episode_steps: Maximum steps before truncation
    
    Example:
        >>> config = {
        ...     "technicians": [{"id": 0}, {"id": 1}, {"id": 2}],
        ...     "production_lines": [{
        ...         "prod_rates": np.array([10, 15, 20]),
        ...         "prod_costs": np.array([100, 150, 200]),
        ...         "in_max_cap": np.array([50, 50, 50]),
        ...         "out_max_cap": np.array([50, 50, 50]),
        ...         "weibull_ks": np.array([2.0, 2.5, 3.0]),
        ...         "weibull_inv_lambdas": np.array([0.01, 0.01, 0.01]),
        ...         "initial_in_buff": 10,
        ...     }],
        ...     "max_episode_steps": 1000,
        ... }
        >>> env = KataEnv(config)
        >>> obs, info = env.reset(seed=42)
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    metadata = {"render_modes": ["numpy", "human"], "render_fps": 4}

    def __init__(self, config: dict):
        super(KataEnv, self).__init__()
        self.config = config

        # Extract config parameters
        self.n_technicians = len(config["technicians"])
        self.n_production_lines = len(config["production_lines"])
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.production_lines = config["production_lines"]
        
        # Define observation and action spaces
        # Observation: [ticket features, technician states, production line states]
        # For simplicity, we'll define a flattened observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._get_obs_dim(),), dtype=np.float32
        )
        
        # Action space: select which technician to assign the ticket to
        self.action_space = gym.spaces.Discrete(self.n_technicians)

        # Assigned tickets vectors
        self.assigned_tickets = {
            "ids": np.full(self.n_technicians, -1, dtype=np.int32),
            "priorities": np.full(self.n_technicians, -1, dtype=np.int32),
            "time_to_complete": np.full(self.n_technicians, -1, dtype=np.int32),
            "machine_ids": np.full(self.n_technicians, -1, dtype=np.int32),
            "components": np.full(self.n_technicians, "", dtype=object),
            "type_of_failure": np.full(self.n_technicians, "", dtype=object),
        }

        # Pending tickets dequeue
        self.pending_tickets = deque()

        # Global step counter
        self.episode_step = 0
        
        # Ticket counter for generating unique IDs
        self.ticket_counter = 0
        
        # Current ticket to be assigned
        self.current_ticket = None
        
        # Production line states (will be initialized in reset)
        self.status = None
        self.in_buff = None
        self.out_buff = None
        self.prod_completions = None
        self.absolute_times = None
        self.s_since_repairs = None
        self.virtual_ages = None
        
    def _get_obs_dim(self) -> int:
        """Calculate the dimension of the observation space."""
        # Current ticket features: id, priority, time_to_complete, machine_id (4 numeric)
        ticket_dim = 4
        # Technician states: assigned ticket info (6 per technician)
        technician_dim = 6 * self.n_technicians
        # Production line states: simplified (status, buffers) per machine
        machines_per_line = max(
            len(pl.get("prod_rates", [])) for pl in self.production_lines
        )
        prod_line_dim = self.n_production_lines * machines_per_line * 4  # status, in_buff, out_buff, prod_completion
        
        return ticket_dim + technician_dim + prod_line_dim

    def _get_next_obs(self, current_ticket=None):
        """
        Construct the observation for the current state.
        
        Args:
            current_ticket: The current ticket to be assigned (optional)
            
        Returns:
            numpy array containing the observation
        """
        obs_parts = []
        
        # Current ticket features (4 values)
        if current_ticket is not None:
            obs_parts.extend([
                float(current_ticket["id"]),
                float(current_ticket["priority"]),
                float(current_ticket["time_to_complete"]),
                float(current_ticket["machine_id"]),
            ])
        else:
            obs_parts.extend([0.0, 0.0, 0.0, 0.0])
        
        # Technician states (6 values per technician)
        for i in range(self.n_technicians):
            obs_parts.extend([
                float(self.assigned_tickets["ids"][i]),
                float(self.assigned_tickets["priorities"][i]),
                float(self.assigned_tickets["time_to_complete"][i]),
                float(self.assigned_tickets["machine_ids"][i]),
                # For string features, we use a simple hash or numeric representation
                float(hash(str(self.assigned_tickets["components"][i])) % 1000),
                float(hash(str(self.assigned_tickets["type_of_failure"][i])) % 1000),
            ])
        
        # Production line states (simplified)
        for i in range(self.n_production_lines):
            n_machines = len(self.status[i])
            for j in range(n_machines):
                obs_parts.extend([
                    float(self.status[i][j]),
                    float(self.in_buff[i][j]),
                    float(self.out_buff[i][j]),
                    float(self.prod_completions[i][j]),
                ])
            # Pad if necessary to match max machines per line
            max_machines = max(len(pl.get("prod_rates", [])) for pl in self.production_lines)
            for _ in range(max_machines - n_machines):
                obs_parts.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(obs_parts, dtype=np.float32)

    def _get_info(self):
        """
        Return auxiliary information about the environment state.
        
        Returns:
            Dictionary containing diagnostic information
        """
        return {
            "episode_step": self.episode_step,
            "pending_tickets": len(self.pending_tickets),
            "assigned_tickets_count": np.sum(self.assigned_tickets["ids"] != -1),
            "available_technicians": np.sum(self.assigned_tickets["ids"] == -1),
        }

    def _get_reward(self):
        """
        Calculate the reward for the current state/action.
        
        The reward considers:
        - Negative reward for waiting time (pending tickets)
        - Negative reward for machines being down
        - Positive reward for completing repairs
        
        Returns:
            float: The reward value
        """
        reward = 0.0
        
        # Penalty for pending tickets (encourages fast assignment)
        reward -= len(self.pending_tickets) * 0.1
        
        # Penalty for machines in maintenance state
        for i in range(self.n_production_lines):
            n_broken = np.sum(self.status[i] == -1)
            reward -= n_broken * 1.0
        
        # Small penalty for each time step (encourages efficiency)
        reward -= 0.01
        
        return reward

    def _get_done(self):
        """
        Check if the episode should terminate.
        
        Returns:
            bool: True if episode is done, False otherwise
        """
        # Episode ends if max steps reached or all machines are broken
        all_broken = all(
            np.all(self.status[i] == -1) for i in range(self.n_production_lines)
        )
        return all_broken

    def _get_truncated(self):
        """
        Check if the episode should be truncated (time limit).
        
        Returns:
            bool: True if episode is truncated, False otherwise
        """
        return self.episode_step >= self.max_episode_steps

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            tuple: (observation, info)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode counter
        self.episode_step = 0
        self.ticket_counter = 0
        
        # Reset assigned tickets
        self.assigned_tickets = {
            "ids": np.full(self.n_technicians, -1, dtype=np.int32),
            "priorities": np.full(self.n_technicians, -1, dtype=np.int32),
            "time_to_complete": np.full(self.n_technicians, -1, dtype=np.int32),
            "machine_ids": np.full(self.n_technicians, -1, dtype=np.int32),
            "components": np.full(self.n_technicians, "", dtype=object),
            "type_of_failure": np.full(self.n_technicians, "", dtype=object),
        }
        
        # Clear pending tickets
        self.pending_tickets.clear()
        
        # Initialize production line states for each production line
        self.status = []
        self.in_buff = []
        self.out_buff = []
        self.prod_completions = []
        self.absolute_times = []
        self.s_since_repairs = []
        self.virtual_ages = []
        
        for prod_line in self.production_lines:
            n_machines = len(prod_line["prod_rates"])
            self.status.append(np.zeros(n_machines, dtype=np.int32))
            self.in_buff.append(np.full(n_machines, prod_line.get("initial_in_buff", 10), dtype=np.int32))
            self.out_buff.append(np.zeros(n_machines, dtype=np.int32))
            self.prod_completions.append(np.full(n_machines, -1, dtype=np.int32))
            self.absolute_times.append(np.zeros(n_machines, dtype=np.int32))
            self.s_since_repairs.append(np.zeros(n_machines, dtype=np.int32))
            self.virtual_ages.append(np.zeros(n_machines, dtype=np.float32))
        
        # Generate initial ticket if needed or wait for first breakdown
        self.current_ticket = None
        
        # Step production lines until we get a ticket or a technician can be assigned
        while self.current_ticket is None:
            # Step all production lines to generate tickets
            for i, production_line in enumerate(self.production_lines):
                self.status[i], self.prod_completions[i], self.in_buff[i], self.out_buff[i], \
                self.absolute_times[i], self.s_since_repairs[i], self.virtual_ages[i] = step_prod_line(
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
                
                # Check if a machine broke, if so, create a ticket
                idxes_broken = np.where(self.status[i] == -1)[0]
                for idx in idxes_broken:
                    machine_id = int(i * 100 + idx)  # Encode production line and machine index
                    if machine_id not in self.assigned_tickets["machine_ids"]:
                        # Create a ticket for the broken machine
                        ticket = create_ticket(
                            ticket_id=self.ticket_counter,
                            priority=np.random.randint(1, 6),  # Random priority 1-5
                            time_to_complete=np.random.randint(5, 20),  # Random repair time
                            machine_id=machine_id,
                            component="component_" + str(np.random.randint(1, 5)),
                            type_of_failure="failure_type_" + str(np.random.randint(1, 3)),
                        )
                        self.ticket_counter += 1
                        self.pending_tickets.append(ticket)
            
            # If we have pending tickets, get one
            if len(self.pending_tickets) > 0:
                self.current_ticket = self.pending_tickets.popleft()
                break
            
            # Prevent infinite loop - if too many steps, create a synthetic ticket
            if self.episode_step > 100:
                # Create a synthetic ticket to start the episode
                ticket = create_ticket(
                    ticket_id=self.ticket_counter,
                    priority=3,
                    time_to_complete=10,
                    machine_id=0,
                    component="component_1",
                    type_of_failure="failure_type_1",
                )
                self.ticket_counter += 1
                self.current_ticket = ticket
                break
        
        obs = self._get_next_obs(self.current_ticket)
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: Index of the technician to assign the ticket to
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # The action is the index of the technician to assign the next ticket to
        action = int(action)
        invalid = False
        
        # Check if the action is valid (i.e. if the technician does not have an assigned ticket)
        if self.assigned_tickets["ids"][action] != -1:
            invalid = True
            # Apply penalty for invalid action
            reward = -10.0
            terminated = True
            truncated = False
            info = self._get_info()
            info["invalid_action"] = True
            obs = self._get_next_obs(self.current_ticket)
            return obs, reward, terminated, truncated, info

        # Assign the ticket to the technician
        # Update the assigned tickets vectors
        self.assigned_tickets["ids"][action] = self.current_ticket["id"]
        self.assigned_tickets["priorities"][action] = self.current_ticket["priority"]
        self.assigned_tickets["time_to_complete"][action] = self.current_ticket[
            "time_to_complete"
        ]
        self.assigned_tickets["machine_ids"][action] = self.current_ticket["machine_id"]
        self.assigned_tickets["components"][action] = self.current_ticket["component"]
        self.assigned_tickets["type_of_failure"][action] = self.current_ticket[
            "type_of_failure"
        ]

        # Compute the reward
        reward = self._get_reward()

        # Compute if the episode is done
        terminated = self._get_done()

        # Compute the info
        info = self._get_info()

        ### Transition to the next state
        # Decrement time to complete for all assigned tickets
        for i in range(self.n_technicians):
            if self.assigned_tickets["ids"][i] != -1:
                self.assigned_tickets["time_to_complete"][i] -= 1
                # If repair is complete, free the technician and repair the machine
                if self.assigned_tickets["time_to_complete"][i] <= 0:
                    # Extract machine info from machine_id
                    machine_id = self.assigned_tickets["machine_ids"][i]
                    prod_line_idx = machine_id // 100
                    machine_idx = machine_id % 100
                    
                    # Repair the machine if it's in valid range
                    if prod_line_idx < self.n_production_lines and machine_idx < len(self.status[prod_line_idx]):
                        self.status[prod_line_idx][machine_idx] = 0  # Set to idle
                        self.s_since_repairs[prod_line_idx][machine_idx] = 0
                        # Keep virtual age for imperfect repair (Kijima model)
                    
                    # Free the technician
                    self.assigned_tickets["ids"][i] = -1
                    self.assigned_tickets["priorities"][i] = -1
                    self.assigned_tickets["time_to_complete"][i] = -1
                    self.assigned_tickets["machine_ids"][i] = -1
                    self.assigned_tickets["components"][i] = ""
                    self.assigned_tickets["type_of_failure"][i] = ""
        
        # Step production lines until we get a new ticket and have an available technician
        while True:
            ### Get the next ticket
            # 1. If there are pending tickets and we have an available technician, use one
            if len(self.pending_tickets) > 0 and np.any(self.assigned_tickets["ids"] == -1):
                break
            
            # 2. Step production lines to generate new tickets or wait for technician to become available
            for i, production_line in enumerate(self.production_lines):
                # Step the production line
                self.status[i], self.prod_completions[i], self.in_buff[i], self.out_buff[i], \
                self.absolute_times[i], self.s_since_repairs[i], self.virtual_ages[i] = step_prod_line(
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
                    machine_id = i * 100 + idx
                    if machine_id not in self.assigned_tickets["machine_ids"]:
                        # Create a ticket for the broken machine
                        ticket = create_ticket(
                            ticket_id=self.ticket_counter,
                            priority=np.random.randint(1, 6),
                            time_to_complete=np.random.randint(5, 20),
                            machine_id=machine_id,
                            component="component_" + str(np.random.randint(1, 5)),
                            type_of_failure="failure_type_" + str(np.random.randint(1, 3)),
                        )
                        self.ticket_counter += 1
                        self.pending_tickets.append(ticket)
            
            # Decrement time for ongoing repairs during waiting
            for i in range(self.n_technicians):
                if self.assigned_tickets["ids"][i] != -1:
                    self.assigned_tickets["time_to_complete"][i] -= 1
                    # If repair is complete, free the technician and repair the machine
                    if self.assigned_tickets["time_to_complete"][i] <= 0:
                        machine_id = self.assigned_tickets["machine_ids"][i]
                        prod_line_idx = machine_id // 100
                        machine_idx = machine_id % 100
                        
                        if prod_line_idx < self.n_production_lines and machine_idx < len(self.status[prod_line_idx]):
                            self.status[prod_line_idx][machine_idx] = 0
                            self.s_since_repairs[prod_line_idx][machine_idx] = 0
                        
                        self.assigned_tickets["ids"][i] = -1
                        self.assigned_tickets["priorities"][i] = -1
                        self.assigned_tickets["time_to_complete"][i] = -1
                        self.assigned_tickets["machine_ids"][i] = -1
                        self.assigned_tickets["components"][i] = ""
                        self.assigned_tickets["type_of_failure"][i] = ""
                        
            # Check if we can break the loop
            if len(self.pending_tickets) > 0 and np.any(self.assigned_tickets["ids"] == -1):
                break
            
            # Check termination conditions
            if self._get_done() or self._get_truncated():
                terminated = self._get_done()
                truncated = self._get_truncated()
                # Return with no next ticket
                obs = self._get_next_obs(None)
                return obs, reward, terminated, truncated, info
                
        # Get the next ticket from pending queue
        self.current_ticket = self.pending_tickets.popleft()

        next_obs = self._get_next_obs(self.current_ticket)
        truncated = self._get_truncated()

        return next_obs, reward, terminated, truncated, info

