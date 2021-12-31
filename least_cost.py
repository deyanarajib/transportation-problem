import numpy as np
from transportation import Transportation

class LeastCost:
    """
    Least Cost Method (LCM) Steps (Rule)
    Step-1:	Select the cell having minimum unit cost cij and allocate as much as possible, i.e. min(si,dj).
    Step-2:
        a. Subtract this min value from supply si and demand dj.
        b. If the supply si is 0, then cross (strike) that row and If the demand dj is 0 then cross (strike) that column.
        c. If min unit cost cell is not unique, then select the cell where maximum allocation can be possible
    Step-3:	Repeact this steps for all uncrossed (unstriked) rows and columns until all supply and demand values are 0.

    Source: https://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=lcm
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
            cost = self.table[1:-1, 1:-1]
            supply = self.table[1:-1, -1]
            demand = self.table[-1, 1:-1]

            #find index of minimum cost
            mins = np.argwhere(cost == np.min(cost))
            alloc = []
            for i, j in mins:
                alloc.append(min([supply[i], demand[j]]))
            x, y = mins[np.argmax(alloc)]

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
                    [40,  8, 70, 20]])
    supply = np.array([7, 9, 18])
    demand = np.array([5, 8, 7, 14])

    #example 2 unbalance problem
    cost = np.array([[ 4,  8,  8],
                    [16, 24, 16],
                    [8, 16, 24]])
    supply = np.array([76, 82, 77])
    demand = np.array([72, 102, 41])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize least cost method with table that has been prepared before.
    least_cost = LeastCost(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = least_cost.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
           C0    C1     C2     C3 Supply
R0         19    30     50  10(7)      7
R1      70(2)    30  40(7)     60      9
R2      40(3)  8(8)     70  20(7)     18
Demand      5     8      7     14     34

TOTAL COST: 814

example 2 unbalance problem
           C0      C1      C2  Dummy Supply
R0      4(56)       8       8  0(20)     76
R1         16  24(41)  16(41)      0     82
R2      8(16)  16(61)      24      0     77
Demand     72     102      41     20    235

TOTAL COST: 2968
'''
