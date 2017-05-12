from anytree import Node, RenderTree, PreOrderIter
import random as rnd
from matplotlib import pyplot as plt
import datetime

# Tic Tac Toe Stats and Game simulator.
# Allows us to estimate for arbitrary grid sizes whether there is a first mover advantage or not.
# Also allows us to estimate which is the best grid reference to start the game with.
# Copyright 2017 Azahar Machwe, Bristol, UK
class Grid:
    """Tic Tac Toe Grid"""

    def __init__(self, x_by_x=3):
        self.GRID, self.GRID_REF = self.build_grid(x_by_x)

        self.__GRID_LEN__ = x_by_x
        self.__TOTAL_GRIDS__ = x_by_x * x_by_x

        # Constants for 'x', 'o' and 'draw'
        self.X = "x"
        self.O = "o"
        self.D = "draw"

    def build_grid(self, x_by_x=3):
        """
        :param x_by_x: Size of grid
        :return: Grid and Grid Reference
        
        Build the Tic Tac Toe Grid - provide size of the square grid.
        
        Grid references for 3x3 tic tac toe grid:
            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9
            
        
        """

        # base grid (e.g. 3x3) each space can be 'x', 'o' or ''
        base = []

        # base_ref is a list of grid locations (e.g. for 3x3 grid - list will have location numbers from 1 to 9)
        base_ref = [i for i in range(1, (x_by_x * x_by_x + 1))]
        for x in range(0, x_by_x):
            row = []
            for y in range(0, x_by_x):
                row.append('')

            base.append(row)

        return base, base_ref

    def clean(self):
        """Clean the grid - remove x or o"""
        self.GRID = self.build_grid(x_by_x=self.__GRID_LEN__)[0]

    def build_graph(self):
        """
            Build graph of possible moves this will NOT work for grid size more than 3x3 as possible number of moves
            will be greater than 20 trillion! 
            
            Use this to do brute-force solving of a tic-tac-toe grid.
            The graph is a set of moves - each node is a grid reference and one walk from root to a leaf is a full game
            with each grid reference present once.
            
            
            It can be thought of as a set of move sequences, some examples include:
            [1, 3, 4, 6, 5, 7, 9, 8, 2]
            [1, 2, 7, 5, 6, 4, 9, 8, 3]
            Where each entry is a grid reference (or a grid entry) 
            
            Grid references for 3x3 tic tac toe grid:
            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9
            
            We always assume 'o' is the first move and 'x' the next.
            For move sequence: [1, 3, 4, 6, 5, 7, 9, 8, 2]
            The final grid becomes:
            o | o | x
            ---------
            o | o | x
            ---------
            x | x | o
            
            Here 'o' is the winner (i.e. first mover is the winner)
            
            
        """
        root = Node("Root")
        if len(self.GRID_REF) > 9:
            raise Exception("Cannot create a graph for Grid size greater than 3 x 3. Use 'sample_moves' instead.")

        self.populate_graph(self.GRID_REF, root, 0, "r")

        return root

    def sample_moves(self, no_of_samples=100000):
        """
        :param no_of_samples:  take number of unique samples
        :return:  set of move sequences
        
        This is the alternative to brute force - we sample unique move sequences instead of trying to create a full set of 
        moves. This is required for grid sizes greater than 3x3.
        
        
        Grid size - Number of Grid Slots - total number of moves
        2x2 = 4 = 24 move sequences
        3x3 = 9 = 362880 move sequences
        4x4 = 16 =  >20 trillion move sequences
        
        In general total number of moves = Factorial(number of grid items) 
        
        """

        move_samples = {}
        current_state = [str(x) for x in self.GRID_REF]
        while len(move_samples) < no_of_samples:
            moves = ':'.join(current_state)
            move_samples[moves] = True

            current_state = self.switch(current_state)

        return move_samples.keys()

    def populate_graph(self, nodes_to_be_added, parent, step, root):
        """ populate the moves graph to create all possible move sequences """
        step += 1
        for i in range(0, len(nodes_to_be_added)):
            _nodes_to_be_added = [x for x in nodes_to_be_added]
            tmp = _nodes_to_be_added.pop(len(nodes_to_be_added) - 1 - i)
            child = parent.name + ":" + str(tmp)
            node = Node(child, parent=parent)

            if root is None:
                self.populate_graph(_nodes_to_be_added, node, step, tmp)
            else:
                self.populate_graph(_nodes_to_be_added, node, step, root)

    def switch(self, state):
        """
        Randomly switch moves in a move set 
        :param state: current move sequence
        :return: switched move sequence
            
          
            Example:
            Input > [1, 3, 4, 6, 5, 7, 9, 8, 2]
            Output > [2, 3, 4, 6, 5, 7, 9, 8, 1] (first and last grid ref are switched)
        """
        pick_A = rnd.randint(0, self.__TOTAL_GRIDS__ - 1)
        pick_Z = rnd.randint(0, self.__TOTAL_GRIDS__ - 1)

        while pick_A == pick_Z:
            pick_Z = rnd.randint(0, self.__TOTAL_GRIDS__ - 1)

        tmp = state[pick_A]
        state[pick_A] = state[pick_Z]
        state[pick_Z] = tmp

        return state

    def pretty_print(self):
        """ pretty print the grid """

        separator = ''
        for i in range(0, self.__GRID_LEN__ * 16):
            separator = separator + '-'

        for i in range(0, self.__GRID_LEN__):
            line = ''
            for j in range(0, self.__GRID_LEN__):

                if j > 0:
                    line = line + "\t|\t{}".format(self.GRID[i][j])
                else:
                    line = line + "\t" + self.GRID[i][j]

            if i != 0:
                print(separator)

            print(line)

    def move(self, row, col, value):
        """
        Mark a particular grid (given by row - column reference) with a value of 'x' or 'o'
        :param row: 
        :param col: 
        :param value:  ('x' or 'o')
        :return: 
        """
        if self.GRID[row][col] == '':
            self.GRID[row][col] = value
            return True

        return False

    def set_o(self, row, col):
        """
        Convenience method - set 'o' at a grid entry
        :param row: grid row
        :param col: grid col
        :return: 
        """
        self.move(row, col, self.O)

    def set_o_slot(self, slot):
        """
        Convenience method for 'o', using slot (e.g. in a 3x3 grid slots are numbered from 1 to 9 (left to right)
        :param slot: the slot number for a grid 
        :return: 
        """
        idx = self.slot_to_index(slot)

        self.set_o(idx[0], idx[1])

    def set_x(self, row, col):
        """
        Convenience method - set 'x' at a grid entry
        :param row: grid row
        :param col: grid col
        :return: 
        """
        self.move(row, col, self.X)

    def slot_to_index(self, slot):
        """
        Convert a slot to row, col value
        :param slot: integer value - 1 to max grid ref (e.g. 3x3 max grid ref = 9)
        :return: 
        
           Grid references (slots) for 3x3 tic tac toe grid:
            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9
            
            so slot 1 = 0, 0 and slot 8 = 2, 1
        """
        slot = int(slot)
        for i in range(1, self.__GRID_LEN__ + 1):
            slot_a = (i - 1) * self.__GRID_LEN__
            slot_z = i * self.__GRID_LEN__
            if slot > slot_a and slot <= slot_z:
                return (i - 1, slot - (1 + slot_a))

    def set_x_slot(self, slot):
        """
        Convenience method for 'x', using slot (e.g. in a 3x3 grid slots are numbered from 1 to 9 (left to right)
        :param slot: the slot number for a grid 
        :return: 
        """
        idx = self.slot_to_index(slot)
        self.set_x(idx[0], idx[1])

    def clear(self, row, col):
        """
        Clear a particular grid entry
        :param row: 
        :param col: 
        :return: 
        """
        self.GRID[row][col] = ''

    def game_over(self):
        """
        Check if the game is over (a straight line with same symbols)
        :return: Winning mark ('x' or 'o', None otherwise)
        """

        # Check Vertical
        for i in range(0, self.__GRID_LEN__):
            x_count = 0
            o_count = 0
            for j in range(0, self.__GRID_LEN__):
                if self.GRID[i][j] == self.X:
                    x_count += 1
                elif self.GRID[i][j] == self.O:
                    o_count += 1

            if x_count == self.__GRID_LEN__:
                return self.X
            if o_count == self.__GRID_LEN__:
                return self.O

        # Check Horizontal

            x_count = 0
            o_count = 0
            for j in range(0, self.__GRID_LEN__):
                if self.GRID[j][i] == self.X:
                    x_count += 1
                elif self.GRID[j][i] == self.O:
                    o_count += 1

                if x_count == self.__GRID_LEN__:
                    return self.X
                if o_count == self.__GRID_LEN__:
                    return self.O


        # Check Cross
        matches = 1
        for i in range(0, self.__GRID_LEN__ - 1):
            if self.GRID[i][i] != '' and self.GRID[i][i] == self.GRID[i + 1][i + 1]:
                matches += 1

        if matches == self.__GRID_LEN__:
            return self.GRID[0][0]


        # Check Cross
        matches = 1
        for i in range(0, self.__GRID_LEN__ - 1):
            if self.GRID[i][self.__GRID_LEN__ - 1 - i] != '' and self.GRID[i][self.__GRID_LEN__ - 1 - i] == \
                    self.GRID[i + 1][self.__GRID_LEN__ - 2 - i]:
                matches += 1

        if matches == self.__GRID_LEN__:
            return self.GRID[0][self.__GRID_LEN__ - 1]


        return None

    def evaluate(self, moves):
        """
        Evaluate a complete move sequence
        
        Assume 'o' moves first
        :param moves: sequence of legal moves (length = number of grid entries)
        :return: 
        """
        if len(moves) != self.__TOTAL_GRIDS__:
            raise Exception("Not a complete game.")

        set_o = True
        for enum, move in enumerate(moves):
            if set_o:
                set_o = False
                self.set_o_slot(move)
            else:
                set_o = True
                self.set_x_slot(move)
            if enum >= self.__GRID_LEN__+self.__GRID_LEN__-2:
                result = self.game_over()
                if result is not None:
                    return result

        return self.D


###### Main #######


def main(brute_force=False, grid_width=3, sample_size=10000):
    """
    
    :param brute_force: boolean - if True then grid width must be less than or equal to 3
    :param grid_width: One dimension of a square grid - e.g. value of 3 = grid of 3x3
    :param sample_size: sample size if using sampling - number of unique move sequences to use for calculating stats
    :return: 
    
    """

    # Build grid
    g = Grid(x_by_x=grid_width)

    # Prepare based on whether we are using brute force or not
    if brute_force:
        if grid_width > 3:
            raise Exception("Using brute force - won't work for Grid greater than 3 x 3.")

        # Build brute force graph - all possible move sequences
        root = g.build_graph()
        # Use this to render the tree:
        #   for pre, fill, node in RenderTree(root):
        #         print("%s%s" % (pre, node.name))

        # Convert graph to individual move sequences
        n = [x for x in PreOrderIter(root)]
        count = 0
        move_sequences = []
        for _n in n:
            if _n.is_leaf:
                count += 1
                move_sequences.append(_n.name)

    else:
        # Do not use brute force - instead just create a set of move sequences by sampling uniformly
        print("Using samples")
        move_sequences = g.sample_moves(no_of_samples=sample_size)

    # Counts of winners, we assume 'o' moves first.
    count_first_move_win = 0
    count_second_move_win = 0
    count_draw = 0
    total_games = 0

    # First grid references
    first_grid_ref = {}

    for move_sequence in move_sequences:
        """Process each move sequence"""
        moves = move_sequence.split(":")

        if len(moves) == g.__TOTAL_GRIDS__ + 1:
            # Required to trim the root node when using move sequences from the brute force graph
            moves = moves[1:]

        # Clean the grid
        g.clean()

        # Get output from particular move sequence
        output = g.evaluate(moves)

        total_games += 1

        # Index 0 = First move won, Index 1 = Second move won, Index 2 = Draw
        idx = None

        if output == g.O:
            count_first_move_win += 1
            idx = 0
        elif output == g.X:
            count_second_move_win += 1
            idx = 1
        elif output == g.D:
            count_draw += 1
            idx = 2

        # Process first grid reference counts
        first = moves[0]
        if idx is not None:
            if first_grid_ref.get(first) is None:

                update = [0, 0, 0]
                update[idx] = 1
                first_grid_ref[first] = update
            else:
                update = first_grid_ref[first]
                update[idx] += 1
    # Print game estimates for first mover, second movers and draws
    print("Total: {}\t\tDraw: {}\tPlay first:{}\tPlay Second:{}".format(total_games, count_draw * 100. / total_games, count_first_move_win * 100. / total_games, count_second_move_win * 100. / total_games))
    print(first_grid_ref)

    # Return game count % for draws, first move wins and second move wins
    return (total_games, count_draw * 100. / total_games, count_first_move_win * 100. / total_games, count_second_move_win * 100. / total_games)


#### Run ####

# Use Brute force or not.
BRUTE_FORCE = False

# This is important only if using Brute Force = False - this decides how many rounds of estimations to do.
NUMBER_OF_ROUNDS = 100

# Grid Width
GRID_WIDTH = 4

# Sample size only if using estimates (Brute force = False)
SAMPLE_SIZE = 100000

# To store estimates
result = [[], [], [], []]

for i in range(0, NUMBER_OF_ROUNDS):
    out = main(grid_width=GRID_WIDTH, brute_force=BRUTE_FORCE, sample_size=SAMPLE_SIZE)

    print(i)

    #Collate the results
    for j in range(0, len(out)):
        result[j].append(out[j])

    # Break if brute force - as we have exact counts - no need for further rounds
    if BRUTE_FORCE:
        break

# Plot a histogram of first win result counts

# Second Move Win
plt.hist(result[3], bins=20, label="Second Move Win", alpha = 0.7)

# First Move Win
plt.hist(result[2], bins=20, label="First Move Win", alpha = 0.7)

# Draw
plt.hist(result[1], bins=20, label="Draw", alpha = 0.7)

plt.legend(loc='upper right')
plt.show()

