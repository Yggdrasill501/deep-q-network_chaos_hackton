import gym
from gym import spaces


class SystemEnv(gym.Env):
    def __init__(self, target_state=None):
        super(SystemEnv, self).__init__()

        # Possible click coordinates
        self.possible_click_coordinates = [(500, 500), (1000, 500), (1500, 500), (500, 800), (1000, 800), (1500, 800)]

        # Define the swipe action coordinates
        self.swipe_down_coordinates = ((100, 200), (100, 400))
        self.swipe_up_coordinates = ((100, 400), (100, 200))

        # Action space: 0 for swipe down, 1 for swipe up, index of the possible_click_coordinates list for click
        self.action_space = spaces.MultiDiscrete([3, len(self.possible_click_coordinates) + 1])

        # Dummy observation space, since we don't have real observations
        self.observation_space = spaces.Discrete(1)

        # Initial state and previous state
        self.state = self.reset()
        self.previous_state = None

        # Set of visited states
        self.visited_states = set()

        # List of predefined states
        self.predefined_states = ["State1", "State2", "State3", "State4"]

        # Swipe used flag
        self.swipe_used = False

        # Target state for production mode
        self.target_state = target_state

    def click(self, coords_index):
        # Get the actual coordinates from the index
        x, y = self.possible_click_coordinates[coords_index]
        click_command = f"execute sending the command trough adb with {x}, {y}"
        return click_command

    def swipe(self, swipe_type):
        # Perform a swipe action
        if swipe_type == 0:  # Swipe down
            swipe_command = f"execute sending the command trough adb with {self.swipe_down_coordinates}"
        else:  # Swipe up
            swipe_command = f"execute sending the command trough adb with {self.swipe_up_coordinates}"
        self.swipe_used = True
        return swipe_command

    def calculate_reward(self):
        if self.state not in self.visited_states:
            reward = 1.0
            if self.swipe_used:
                reward += 0.2
        elif self.state == self.previous_state:
            reward = -1.0
            if self.swipe_used:
                reward -= 0.5
        else:  # state is among visited states but not the same as the previous state
            reward = -1.0
            if not self.swipe_used:
                reward += 0.2

        return reward

    def check_state(self):
        # Check the state of the system and updates self.state
        self.state = "state"

    def episode_is_done(self):
        # Check whether the episode is done
        if self.target_state is not None:  # Production mode
            return self.state == self.target_state
        else:  # Training mode
            return set(self.predefined_states).issubset(self.visited_states)

    def step(self, action):
        # Check if action is a list or tuple
        if not isinstance(action, (list, tuple)):
            raise TypeError("Action must be a list or tuple.")

        # Execute the action
        if action[0] == 2:  # Click
            self.click(action[1])
        else:  # Swipe (0 for down, 1 for up)
            self.swipe(action[0])

        # Save the previous state
        self.previous_state = self.state

        # Check the state of the system
        self.check_state()

        # Calculate the reward
        reward = self.calculate_reward()

        # Reset the swipe used flag after state check
        self.swipe_used = False

        # Check if the episode is done
        done = self.episode_is_done()

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state to the initial state
        self.state = "AppGrid"
        self.visited_states.clear()
        self.swipe_used = False
        return self.state