"""Base game for minesweeper."""

import random
from string import ascii_lowercase

import numpy as np


class Game:
    """Class containing a playable Minesweeper game."""

    _UNOPENED = -3
    _MINE = -1
    _FLAG = -2
    _OPENED = 0

    def __init__(self, size_y, size_x, mine_count):
        self._size_y = size_y
        self._size_x = size_x
        self._mine_count = mine_count
        self._top_labels = [str(label) for label in list(range(size_x))]
        self.reset()

    def reset(self):
        self._create_minefield(self._size_y, self._size_x, self._mine_count)
        return self.state()

    def step(self, n_action):
        """Action specified as linear index of a field to be opened."""
        y, x = np.unravel_index(n_action, (self._size_y, self._size_x))
        return self.action(y, x, False)

    def action(self, y, x, flag):
        """Take an action by either flagging or selecting a field.

        Return the reward.
        :rtype: (new_state, reward, done)
        :param y: y coordinate
        :param x: x coordinate
        :param flag: True if action is flag/remove flag
        """
        if not self._valid_position(y, x):
            raise Exception("({},{}) is not a valid position.".format(y, x))

        penalty_useless_action = -20
        if flag:
            if self._overlay[y, x, 1] == 0:
                self._overlay[y, x, 0] = self._FLAG
                self._overlay[y, x, 1] = 1
                reward = -3
            elif self._overlay[y, x, 0] == self._FLAG:
                self._overlay[y, x, 0] = self._UNOPENED
                self._overlay[y, x, 1] = 0
                reward = -1
            else:  # Try to flag an already open area
                reward = penalty_useless_action
        else:
            if self._overlay[y, x, 1] == 0:
                self._open_overlay(y, x)
                reward = 1
            else:  # Try to open an already open area
                reward = penalty_useless_action

        won = self.won()
        if won == 0:
            done = False
        elif won == 1:
            reward = 100
            done = True
        else:
            reward = -100
            done = True

        return self.state(), reward, done

    def open(self, y, x):
        """Open a field.

        :rtype: True if move is accepted
        :param y: y coordinate
        :param x: x coordinate
        """
        if self._valid_position(y, x):
            self._open_overlay(y, x)
        else:
            raise Exception("({},{}) is not a valid position.".format(y, x))

    def flag(self, y, x):
        """Flag or remove a flag from a field.

        :rtype: True if flag can be set or removed.
        :param y: y coordinate
        :param x: x coordinate
        """
        if self._valid_position(y, x):
            if self._overlay[y, x] == self._UNOPENED:
                self._overlay[y, x] = self._FLAG
            elif self._overlay[y, x] == self._FLAG:
                self._overlay[y, x] = self._UNOPENED
        else:
            raise Exception("({},{}) is not a valid position.".format(y, x))

    def state(self):
        """Return current state of the game."""
        return np.copy(self._overlay)

    def won(self):
        """Return -1, 0 or 1 if the game is lost, undecided or won."""
        overlay = np.copy(self._overlay)
        for mine in self._mines:
            # If mine is unopened or flagged open it
            if overlay[mine[0], mine[1], 1] == 0 or overlay[mine[0], mine[1], 0] == self._FLAG:
                overlay[mine[0], mine[1], 1] = 1
        if np.any(overlay[:, :, 0] == self._MINE):
            return -1
        elif np.any(overlay[:, :, 0] == self._FLAG):  # Check that only mines are flagged
            return 0
        elif np.all(overlay[:, :, 1] != 0):  # All fields are open and no fields are flags or mines
            return 1
        else:
            return 0

    def _create_minefield(self, size_y, size_x, mine_count):
        """Create minefield with random mine distribution.

        First dimension is y and second dimension is x."""
        self._field = np.zeros((size_y, size_x), np.int8)
        self._overlay = np.zeros((size_y, size_x, 2), np.int8)
        self._overlay[:, :, 0] = self._UNOPENED

        possible_fields = [(y, x) for y in range(size_y) for x in range(size_x)]
        self._mines = random.sample(possible_fields, mine_count)

        self._mines = [(4, 0), (6, 0), (0, 4), (7, 3), (4, 5), (0, 6), (1, 6), (4, 6), (1, 7), (6, 7)]

        # Set up values to define near mines
        for mine in self._mines:
            self._set_field_around_mine(mine)

        # Set mines into field
        for mine in self._mines:
            self._field[mine[0], mine[1]] = self._MINE

    def _set_field_around_mine(self, mine):
        """Set surrounding values around mine."""
        y = mine[0]
        x = mine[1]
        y_from = y-1 if y > 0 else 0
        y_to = y+2
        x_from = x-1 if x > 0 else 0
        x_to = x+2
        self._field[y_from:y_to, x_from:x_to] += 1

    def _valid_position(self, y, x):
        """Check if passed position is still on the board."""
        return (y >= 0) and (y < self._size_y) and (x >= 0) and (x < self._size_x)

    def _open_overlay(self, y, x):
        """Open overlay around the opened position."""
        if self._overlay[y, x, 1] == 0:
            self._overlay[y, x, 0] = self._field[y, x]
            self._overlay[y, x, 1] = 1
            if self._overlay[y, x, 0] == 0:
                [self._open_overlay(ny, nx) for ny, nx in self._neighbours(y, x)]

    def _neighbours(self, y, x):
        """Get neighbours of given position."""
        neighbours = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                else:
                    if self._valid_position(y+dy, x+dx):
                        neighbours.append((y+dy, x+dx))
        return neighbours

    def show_field(self):
        field = np.array2string(self._overlay[:, :, 0], sign=' ')
        # Print top row containing labels
        print('     ' + '  '.join(self._top_labels))
        for label, row in zip(ascii_lowercase[:self._size_y], field.splitlines()):
            print(label + ' ' + row)
