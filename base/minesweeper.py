"""Base game for minesweeper."""

import random
from string import ascii_lowercase

class Game:
    """Class containing a playable Minesweeper game."""

    def __init__(self, size_y, size_x, mine_count):
        self._size_y = size_y
        self._size_x = size_x
        self._create_minefield(size_x, size_y, mine_count)
        self._top_labels = [str(label) for label in list(range(size_x))]

    def open(self, y, x):
        """Open a field.

        Returns True if the move is accepted."""
        if self._valid_position(y, x):
            self._open_overlay(y, x)
        else:
            raise Exception("({},{}) is not a valid position.".format(y,x))

    def flag(self, y, x):
        """Flag a field.

        Returns True if the move is accepted.note"""
        if self._valid_position(y, x):
            if self._overlay[y][x] == '.':
                self._overlay[y][x] = 'F'
            elif self._overlay[y][x] == 'F':
                self._overlay[y][x] = '.'
        else:
            raise Exception("({},{}) is not a valid position.".format(y,x))

    def won(self):
        """Return -1, 0 or 1 if the game is lost, undecided or won."""
        # flatten = lambda l: [item for sublist in l for item in sublist]
        fields = [field for row in self._overlay for field in row]
        if any(field == 'X' for field in fields):
            return -1
        elif all(field != '.' for field in fields):
            return 1
        else:
            return 0

    def _create_minefield(self, size_x, size_y, mine_count):
        """Create minefield with random mine distribution.

        First dimension is y and second dimension is x."""
        self._field = [['0' for y in range(size_y)] for x in range(size_x)]
        self._overlay = [['.' for y in range(size_y)] for x in range(size_x)]

        possible_fields = [ (y,x) for y in range(size_y) for x in range(size_x) ]
        self._mines = random.sample(possible_fields, mine_count)
        for mine in self._mines:
            self._add_mine(mine)

    def _add_mine(self, mine):
        """Add mine to field and set surrounding values."""
        y = mine[0]
        x = mine[1]
        self._field[y][x] = 'X'
        for ny, nx in self._neighbours(y, x):
            if (self._field[ny][nx] != 'X'):
                self._field[ny][nx] = str(int(self._field[ny][nx]) + 1)

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

    def _valid_position(self, y, x):
        """Check if passed position is still on the board."""
        return (y >= 0) and (y < self._size_y) and (x >= 0) and (x < self._size_x)

    def _open_overlay(self, y, x):
        """Open overlay around the opened position."""
        if self._overlay[y][x] == '.':
            self._overlay[y][x] = self._field[y][x]
            if self._overlay[y][x] == '0':
                [self._open_overlay(ny, nx) for ny,nx in self._neighbours(y,x)]

    def show_field(self):
        # Print top row containing labels
        print('  ' + ' '.join(self._top_labels))
        for label, row in zip(ascii_lowercase[:self._size_y], self._overlay):
            print(label + ' ' + ' '.join(row))
