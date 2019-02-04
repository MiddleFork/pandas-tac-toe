"""MiddleFork/pandas_tac_toe

A pandas/numpy implementation of tic-tac-toe proof of concept

"""
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


class TTT:
    """The main tic-tac-toe class,  consisting of the 3x3 board implemented as a pandas DataFrame,
    as well as methods to play the game, interact with the board, check for victory, and declare a winner.

    """

    def __init__(self):
        """We create the TTT board as a 3x3 DF whose values may only be 'X' or 'O' (or NaN)"""
        self.gameOver = False
        self.lastToken = None
        self.play_order = []
        cat_type = CategoricalDtype(categories=['X', 'O'], ordered=True)
        self.board = pd.DataFrame(np.array([np.nan] * 9).reshape(3, 3)).astype(cat_type)
        self.df_cellindex = pd.DataFrame(np.arange(1, 10).reshape(3, 3), dtype='int')
        self.print_board()

    def print_board(self):
        print(self.board.to_string(header=False, index=False, na_rep='-', justify='left'))

    @property
    def next_token(self):
        """
        X always plays first in TTT, so odd moves are 'X', even moves are 'O'
        A simple modulo against the number of unplayed cells gives us what we need
        """
        return 'X' if self.n_open_cells % 2 == 1 else 'O'


    @property
    def played_cells(self):
        """Boolean DF of played cells"""
        return self.board.notnull()

    @property
    def n_played_cells(self):
        return self.board.stack().count()

    @property
    def n_open_cells(self):
        return self.board.size - self.n_played_cells

    @property
    def last_cell_played(self):
        return self.play_order[-1]

    def cell_pos(self, i):
        df = self.df_cellindex
        return (df[df[df == i].count(axis=1) == 1].index.values[0],
                df[df[df == i].count(axis=0) == 1].index.values[0])

    @property
    def diags(self):
        """Generate arrays of the 2 board diagonals. These are used to test wins by diagonal"""
        diags = (
            np.diag(self.board),
            np.diag(self.board[self.board.columns[::-1]])
        )
        return diags

    @property
    def arr_open_cells(self):
        """Generate an array of the open (unplayed) cells.
        This is used to a) prompt the user to select a cell,
        and b) reject user input when the cell has already been played.
        """
        return self.df_cellindex.mask(self.played_cells).stack().astype(int).values

    def is_winnable(self):
        """Don't bother testing for a win if it isn't even possible.
        Here, at least so far, that means if there are more than five unplayed cells
        """
        if self.n_open_cells > 5:
            return False
        return True

    def choose_cell(self):
        """
        Prompt the user to identify the cell they wish to play,
        using the array of open cells (arr_open_cells) as allowed choices.
        """
        print("Available cells are: {0}".format(self.arr_open_cells))
        while True:
            x = input('Cell? ')
            if (x.isdigit() and int(x) in self.arr_open_cells):  # The choice must be the integer index representation of the desired cell.
                return int(x)
            else:
                continue

    def play_token(self):
        """Play the token (X or O) on the desired cell.
        """
        tkn = self.next_token
        cell_ind = self.choose_cell()
        cell_pos = self.cell_pos(cell_ind)
        self.board = self.board.where(self.df_cellindex[self.df_cellindex == cell_ind].isnull(), other=tkn)  # we play the cell
        self.play_order.append(cell_ind)  # we append the cell to the game's play order
        self.lastToken = tkn  # identify the game's last-played token, so when we have a winner, this is the player who won.
        self.lastPos = cell_pos  # identify the game's last-played cell, so we can limit our test for victory to the row, col, or diag
        self.print_board()  # re-print the game board

    def check_for_win(self):
        """See if there is a victory
        We do this by seeing if any row, column, or diagonal has 3 of the same tokens.
        These are defined in the list 'win_conditions'.
        """
        if not self.is_winnable():  # Don't bother.
            return

        i = self.lastPos[0]  # row of last cell played
        j = self.lastPos[1]  # col of last cell played

        win_conditions = [  # TTT is won in one of these for ways: by row, by col, by first diagonal, by second diagonal, when there are 3 identical tokens
            self.board.iloc[i].value_counts().max() == 3,  # win by row
            self.board.iloc[:, j].value_counts().max() == 3,  # win by col
            pd.Series(np.append(self.diags[0], -1)).value_counts().values[0] == 3,  # win by diag 0 - append a -1 to the np array in case we have all NaN
            pd.Series(np.append(self.diags[1], -1)).value_counts().values[0] == 3,  # win by diag 1
        ]
        for i in win_conditions:
            if i:  # we have a winner if x is true
                self.game_is_won()

    def game_is_won(self):
        """The game has been won.
        We announce the winner and end the game.
        """
        print("{0} Wins ".format(self.lastToken))
        self.gameOver = True

    def play_game(self):
        """Play the game.
        As long as there is one open cell and no winner, keep playing.
        End the game when we have a winner.

        TODO:  end the game when we reach a situation where neither side can win, even if there are empty cells remaining
        """
        while self.n_open_cells > 0 and not self.gameOver:
            self.play_token()
            self.check_for_win()


def main():
    """Play the game when launched directly"""
    ttt = TTT()
    ttt.play_game()


if __name__ == "__main__":
    # execute only if run as a script
    main()
