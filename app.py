import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


class TTT:
    def __init__(self):
        self.gameOver = False
        self.lastToken = None
        self.play_order = []
        cat_type = CategoricalDtype(categories=['X', 'O'], ordered=True)
        self.board = pd.DataFrame(np.array([np.nan] * 9).reshape(3, 3)).astype(cat_type)
        self.df_cellindex = pd.DataFrame(np.arange(1, 10).reshape(3, 3), dtype='int')
        print("\n\nWelcome to Pandas-Tac-Toe.  Warning:  Diagonal wins are not yet implemented.")
        self.print_board()

    def print_board(self):
        print(self.board.to_string(header=False, index=False, na_rep='-', justify='left'))

    @property
    def next_token(self):
        """
        X always plays first in TTT, so odd moves are 'X', even moves are 'O'
        """
        return ('X' if self.n_open_cells % 2 == 1 else 'O')

        # token = lambda n: 'X' if n % 2 == 1 else 'O'
        # return token(self.n_open_cells)

    @property
    def played_cells(self):
        """Boolean DF of played cells"""
        return self.board.notnull()

    @property
    def df_open_cells(self):
        return self.board.isnull()

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
        """Board diagonals"""
        diags = (
            np.diag(self.board),
            np.diag(self.board[self.board.columns[::-1]])
        )
        return diags

    @property
    def arr_open_cells(self):
        return self.df_cellindex.mask(self.played_cells).stack().astype(int).values

    def is_winnable(self):
        """Don't bother testing for a win if it isn't even possibile"""
        if self.n_open_cells > 5:
            return False
        return True

    def choose_cell(self):
        print("Available cells are: {0}".format(self.arr_open_cells))
        while True:
            x = input('Cell? ')
            if (x.isdigit() and int(x) in self.arr_open_cells):
                return int(x)
            else:
                continue

    def play_token(self):
        tkn = self.next_token
        cell_ind = self.choose_cell()
        cell_pos = self.cell_pos(cell_ind)
        print('\nPlaying token {0} at cell index #{1}, position {2}'.format(tkn, cell_ind, cell_pos))
        self.board = self.board.where(self.df_cellindex[self.df_cellindex == cell_ind].isnull(), other=tkn)
        self.play_order.append(cell_ind)
        self.lastToken = tkn
        self.lastPos = cell_pos
        self.print_board()

    def check_for_win(self):
        """See if there is a victory
        We do this by seeing if any row, column, or diagonal has 3 of the same tokens.
        """
        if not self.is_winnable():
            return

        i = self.lastPos[0]
        j = self.lastPos[1]
        if self.board.iloc[i].value_counts().max() == 3 or self.board.iloc[:, j].value_counts().max() == 3:
            print('Game Over')
            self.gameOver = True

    def play_game(self):
        while self.n_open_cells > 0 and not self.gameOver:
            self.play_token()
            self.check_for_win()


def main():
    ttt = TTT()
    ttt.play_game()


if __name__ == "__main__":
    # execute only if run as a script
    main()