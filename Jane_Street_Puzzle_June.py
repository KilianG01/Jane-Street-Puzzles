# import necessary libraries and modules
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy as dcopy
from skimage.morphology import label
import scipy.ndimage as scp
import numba as nb
from z3 import *

# Define the labels at the top and left side of the grid
top_labels = [5, 1, 6, 1, 8, 1, 22, 7, 8]
left_labels = [55, 1, 6, 1, 24, 3, 6, 7, 2]

# Define a function to print the solved matrix with a nice formatting
def sol_print(solved, matrix):
    # setup plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # preprocess solved matrix
    x = np.array((solved * matrix).astype("int").astype("str"))
    x[x == "0"] = "-"
    # plot heatmap
    ax = sns.heatmap(
        matrix, annot=x, cbar=False, cmap="Set3_r", fmt="", linewidths=0.25
    )
    ax.axis("off")
    # print result
    print(x)
    print("Function to print grid called.")

# Define a function to print the grid to the console
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

# Define the Matrix class, which is used to solve the puzzle
class Matrix:
    def __init__(self, top_labels, left_labels):
        # initialize class variables
        self.top_labels = top_labels
        self.left_labels = left_labels
        # a list of all grids not yet ruled out.[Level,grid,coordinates of the grid yet to be filled]
        self.potential_grids = [
            [-9, np.ones((9, 9), dtype=int) * -1, [0, 0, 9, 9], [*range(9, 0, -1)]]
        ]
        self.solution = []
        self.splits = [list(i) for i in itertools.product([0, 1], repeat=9)]
        self.end_flag = 0
        self.grid_counter = 0

    # the function that adds a layer to the grid
    def add_layer(self, grid, coords, lvl, alignment):
        row_start, col_start, row_end, col_end = coords
        if alignment == 0:
            grid[row_start:row_end, col_start] = lvl
            grid[row_start, col_start:col_end] = lvl
            row_start += 1
            col_start += 1

        if alignment == 1:
            grid[row_start:row_end, col_start] = lvl
            grid[row_end - 1, col_start:col_end] = lvl
            row_end -= 1
            col_start += 1

        if alignment == 2:
            grid[row_start:row_end, col_end - 1] = lvl
            grid[row_start, col_start:col_end] = lvl
            row_start += 1
            col_end -= 1

        if alignment == 3:
            grid[row_start:row_end, col_end - 1] = lvl
            grid[row_end - 1, col_start:col_end] = lvl
            row_end -= 1
            col_end -= 1

        coords = [row_start, col_start, row_end, col_end]

        return grid, coords

    def check_grid(self, grid):
        # print(grid)
        self.grid_counter += 1
        # make sure there is enough space in the grid for all the numbers needed
        if not (self.count(grid)):
            return False
        for i in range(9):
            row = grid[i, :]
            col = grid[:, i]
            if -1 not in row:
                if not self.check_line(row, self.left_labels[i]):
                    return False
            if -1 not in col:
                if not self.check_line(col, self.top_labels[i]):
                    return False
        return True

    @staticmethod
    @nb.njit(fastmath=True)
    def count(grid):
        for num in range(2, 10):
            if np.any(grid == num) and (np.sum(grid == num) < num):
                return False
            if np.any(grid == 1) and (np.sum(grid == 1) != 1):
                return False
        return True

    # Function to check line, change to adhere to new ruleset
    def check_line(self, line, param_start):
        for split in self.splits:
            grid = line * split
            if self.valid_line(grid, param_start):
                return 1
        return 0

    def valid_line(self, pos, param):
        if param == "":
            return True

        blocks = []
        block = 0
        inblock = 0
        for p in pos:
            if p != 0:
                if inblock == 0:
                    block += p
                    inblock = 1
                else:
                    block *= 10
                    block += p

            else:
                if inblock == 1:
                    blocks.append(block)
                    block = 0
                    inblock = 0

        if inblock == 1:
            blocks.append(block)

        if len(blocks) < 1:
            return False

        elif np.gcd.reduce(blocks, dtype=int) != param:
            return False

        return True


    def forced_cells(self, hook):
        row_force = np.ones((9, 9), dtype=int) * -1
        col_force = dcopy(row_force)

        # loop through the params to determine forced cells
        for i in range(9):
            # print("Row:", i)
            row_force[i, :] = self.forced_line(hook[i, :], self.left_labels[i])
        for i in range(9):
            # print("Col:", i)
            col_force[:, i] = self.forced_line(hook[:, i], self.top_labels[i])

        final = np.ones((9, 9), dtype=int) * -1

        # look at the 2 different versions of the matrices and combine
        for i, j in itertools.product(range(9), range(9)):
            options = np.array([row_force[i, j], col_force[i, j]])
            if (np.any(options == 1)) & (np.all(options != 0)):
                final[i, j] = 1
            if (np.any(options == 0)) & (np.all(options != 1)):
                final[i, j] = 0
            # flag inconsistent forced matrices
            if (np.any(options == 1)) & (np.any(options == 0)):
                return 0, final

        final[hook == 1] = 1
        grid = final * hook
        for z in range(1, 10):
            if np.sum(grid == z) > z:
                return 0, final
        # check that the forced cells do not violate 2x2
        if self.twobytwo(grid):
            return 0, final

        return 1, final

    def forced_line(self, line, param):
        poss_line = []
        for split in self.splits:
            grid = line * split
            if self.valid_line(grid, param):
                poss_line.append(split)
        poss_array = np.array(poss_line)
        forced = np.ones(9, dtype=int) * -1
        for i in range(9):
            if np.all(poss_array[:, i] == 1):
                forced[i] = 1
            elif np.all(poss_array[:, i] == 0):
                forced[i] = 0
        return forced

    #########################################################
    # Fill the final cells by backtracking
    # Fill the final cells using z3
    def neighbours(self, i, j):
        l = []
        if i - 1 >= 0:
            l.append((i - 1, j))
        if i + 1 < 9:
            l.append((i + 1, j))
        if j - 1 >= 0:
            l.append((i, j - 1))
        if j + 1 < 9:
            l.append((i, j + 1))
        return l

    def fill_rest(self, grid, hooks):
        # set up solver and variables
        s = Tactic("pqffd").solver()
        N = 9
        X = np.array(IntVector("x", N * N), dtype=object).reshape((N, N))

        # force fixed and set to either 0/1
        s += [
            e == int(grid[i, j]) for (i, j), e in np.ndenumerate(X) if grid[i, j] != -1
        ]
        s += [And(e >= 0, e <= 1) for (i, j), e in np.ndenumerate(X)]

        # no 2x2
        s += [
            Or(e == 0, X[i + 1][j] == 0, X[i][j + 1] == 0, X[i + 1][j + 1] == 0)
            for (i, j), e in np.ndenumerate(X[:-1, :-1])
        ]

        # cells have at least one neighbour to help connectivity
        s += [
            Implies(e != 0, Or([X[k, l] != 0 for (k, l) in self.neighbours(i, j)]))
            for (i, j), e in np.ndenumerate(X)
        ]

        # count of numbers in hook
        s += And(
            [
                sum([e for (i, j), e in np.ndenumerate(X) if int(hooks[i, j]) == h])
                == h
                for h in range(1, 10)
            ]
        )

        count = 0
        # function to evaluate
        evalu = np.vectorize(lambda x: m.evaluate(x).as_long())
        while s.check() == sat:  # Run as long as a solution can be found
            m = s.model()
            result = evalu(X)
            grid = result * hooks

            valid_solution = True  # Assume the solution is valid until proven otherwise
            for i in range(9):
                for col, row in itertools.product(range(9), range(9)):
                    row = grid[i, :]
                    col = grid[:, i]
                    if not self.valid_line(row, self.left_labels[i]):
                        s += Or([X[i, j] != int(result[i, j]) for j in range(9)])
                        valid_solution = False  # The solution is not valid
                        break  # Break from the loop as soon as a problem is found
                    if not self.valid_line(col, self.top_labels[i]):
                        s += Or([X[j, i] != int(result[j, i]) for j in range(9)])
                        valid_solution = False  # The solution is not valid
                        break  # Break from the loop as soon as a problem is found
                if not valid_solution:
                    break

            if valid_solution:  # If no problem was found, the solution is valid
                # Check connectivity
                while np.max(label(result != 0, connectivity=1)) > 1:
                    s += Or([e != int(result[i, j]) for (i, j), e in np.ndenumerate(X)])
                    if s.check() != sat:
                        valid_solution = False  # The solution is not valid
                        break
                    m = s.model()
                    result = evalu(X)

                if (
                    valid_solution
                ):  # If no problem was found in connectivity, the solution is valid
                    # Do something with the valid solution
                    grid = result
                    sol_print(grid, hooks)
                    print(
                        "\nThe product of the areas is : {:,.0f}".format(
                            self.areas(grid)
                        )
                    )
                    self.end_flag = 1
                    break  # Exit the loop as we have a valid solution

            # Exclude the current model from future solutions
            s += Or([e != int(result[i, j]) for (i, j), e in np.ndenumerate(X)])

    def possible(self, digit, row, col, grid, hooks):
        grid = grid * hooks
        grid[row, col] = hooks[row, col] * digit
        hook_number = hooks[row][col]

        # check the placement doesn't break connectivity
        if np.max(label(grid != 0, connectivity=1)) > 1:
            return False

        # check the placement doesn't break 2 by 2
        if self.twobytwo(grid):
            return False

        for num in range(2, 10):
            if np.sum(grid == num) > num:
                return False

        # check rows and columns
        for i in range(9):
            row = grid[i, :]
            col = grid[:, i]
            if np.all(row > -1):
                if not self.valid_line(row, self.left_labels[i]):
                    return False
            if np.all(col > -1):
                if not self.valid_line(col, self.top_labels[i]):
                    return False

        # final checks if the placement completes the grid
        if np.sum(grid == -1) == 1:
            # check counts again
            for i in range(2, 10):
                if np.sum(grid == i) != i:
                    return False
        return True

    def twobytwo(self, grid):
        for i, j in itertools.product(range(9), range(9)):
            if (
                i > 0
                and j > 0
                and grid[i, j] > 0
                and grid[i - 1, j - 1] > 0
                and grid[i - 1, j] > 0
                and grid[i, j - 1] > 0
            ):
                return True

            if (
                i > 0
                and j < 7
                and grid[i, j] > 0
                and grid[i - 1, j + 1] > 0
                and grid[i - 1, j] > 0
                and grid[i, j + 1] > 0
            ):
                return True

            if (
                i < 7
                and j > 0
                and grid[i, j] > 0
                and grid[i + 1, j - 1] > 0
                and grid[i + 1, j] > 0
                and grid[i, j - 1] > 0
            ):
                return True

            if (
                i < 7
                and j < 7
                and grid[i, j] > 0
                and grid[i + 1, j + 1] > 0
                and grid[i + 1, j] > 0
                and grid[i, j + 1] > 0
            ):
                return True

        return False

    ##############################################
    # Get the products of the areas of the connected cells

    def areas(self, grid):
        labels, num = scp.label(np.logical_not(grid != 0))
        areas = scp.sum(np.logical_not(grid != 0), labels, index=range(1, num + 1))
        return np.prod(areas)

    ###############################################
    # Main solver.

    def solve(self):
        while len(self.potential_grids) > 0:
            temp_grid = self.potential_grids.pop(0)
            # create the potential rotations at the given level
            rotations = []
            l, g, c, nums = temp_grid
            for num in nums:
                for alignment in range(4):
                    lvl = dcopy(l)
                    grid = dcopy(g)
                    coords = dcopy(c)
                    grid, coords = self.add_layer(grid, coords, num, alignment)
                    if lvl != -1:
                        rotations.append(
                            [lvl + 1, grid, coords, [n for n in nums if n != num]]
                        )
                    else:
                        rotations = [
                            [lvl + 1, grid, coords, [n for n in nums if n != num]]
                        ]

            # check valid grids (where the sum can be made from available digits) and save the ones that work
            for i in range(len(rotations)):
                lvl, g, coords, nums = rotations[i]
                if self.check_grid(g):
                    if lvl != 0:
                        self.potential_grids.append([lvl, g, coords, nums])
                    else:
                        self.solution.append(g)

        print("There are {} valid hook placements".format(len(self.solution)))

        # solve each grid in the cut down list
        forced_grids = []
        for i in range(len(self.solution)):
            if self.end_flag == 0:
                hooks = self.solution[i]
                flag, forced_grid = self.forced_cells(hooks)
                # print("Flag:", flag)
                print("forced grid:", forced_grid)
                print("self.solution[", i, "]:", self.solution[i])
                if flag:
                    # for valid forced grids solve the final matrix
                    print("Grid #", i + 1, "still to solve", np.sum(forced_grid == -1))
                    self.fill_rest(forced_grid, hooks)

#---------------------------------------------------------------------------------------

# start time
start = time.perf_counter()
# create variable grid as an instance of the class Matrix
grid = Matrix(top_labels, left_labels)
# call solve function
grid.solve()
# stop time once solution has been found and print time
stop = time.perf_counter()
print("\n Solution took {:0.4f} seconds\n".format((stop - start)))