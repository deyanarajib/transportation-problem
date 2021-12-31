import numpy as np
from transportation import Transportation

class NorthWestCorner:
    """
    North-West Corner Method (NWCM) Steps (Rule)
    Step-1:	Select the upper left corner cell of the transportation matrix and allocate min(s1, d1).
    Step-2:
        a. Subtract this value from supply and demand of respective row and column.
        b. If the supply is 0, then cross (strike) that row and move down to the next cell.
        c. If the demand is 0, then cross (strike) that column and move right to the next cell.
        d. If supply and demand both are 0, then cross (strike) both row & column and move diagonally to the next cell.
    Step-3:	Repeact this steps until all supply and demand values are 0.

    Source: https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=nwcm&ex=0
    """

    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])
        
        if self.table[x, -1] < self.table[-1, y]:
            #delete row and supply x then change value of demand y
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins
            
        elif self.table[x, -1] > self.table[-1, y]:
            #delete column and demand y then change value of supply x
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins
            
        else:
            #delete row and supply x, column and demand y
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):

            #pick north west corner cell
            x, y = 0, 0

            #allocated row x to column y or vice versa
            self.allocate(x + 1, y + 1)

            #print table
            if show_iter:
                self.trans.print_frame(self.table)
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    #example 1 balance problem
    cost = np.array([[19, 30, 50, 10],
                    [70, 30, 40, 60],
                    [40, 8,  70, 20]])
    supply = np.array([7, 9, 18])
    demand = np.array([5, 8, 7, 14])

    #example 2 unbalance problem
    cost = np.array([[4,   8,  8],
                    [16, 24, 16],
                    [ 8, 16, 24]])
    supply = np.array([76, 82, 77])
    demand = np.array([72, 102, 41])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize NWC method with table that has been prepared before.
    NWC = NorthWestCorner(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = NWC.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
           C0     C1     C2      C3 Supply
R0      19(5)  30(2)     50      10      7
R1         70  30(6)  40(3)      60      9
R2         40      8  70(4)  20(14)     18
Demand      5      8      7      14     34

TOTAL COST: 1015

example 2 unbalance problem
           C0      C1      C2  Dummy Supply
R0      4(72)    8(4)       8      0     76
R1         16  24(82)      16      0     82
R2          8  16(16)  24(41)  0(20)     77
Demand     72     102      41     20    235

TOTAL COST: 3528
'''
