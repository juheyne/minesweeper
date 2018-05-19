from string import ascii_lowercase

from minesweeper import Game


if __name__ == '__main__':
    size_x = 8
    size_y = 8
    mines = 10
    letters = {letter: index for index, letter in enumerate(ascii_lowercase)}
    game = Game(size_y, size_x, mines)

    print('Minesweper ({},{}) with {} mines.'.format(size_y, size_x, mines))
    print('Input your moves in row column format, e.g. a 4.')
    print('If you want to add a flag, add f after the position, e.g. b 6 f.')
    print('Flags are {} and unopened fields are {}.'.format(Game._FLAG, Game._UNOPENED))

    while game.won() == 0:
        game.show_field()
        print()
        move = input('Your input: ')
        move = move.split()
        try:
            y = letters[move[0]]
            x = int(move[1])
            if len(move) == 2:
                game.open(y, x)
            else:
                game.flag(y, x)
        except Exception:
            print('Invalid move. Please try again.')
    if game.won() == 1:
        print('You won. Congratulations')
    else:
        print('You lost.')
