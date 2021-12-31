import numpy as np
from transportation import Transportation

class HeuristicMethod2:
    """
    Heuristic Method 2 Algorithm
    Step 1: Determine the penalty i.e. the difference between the lowest and highest cost element of thet row/column.
    Step 2: Identify the row/column having highest penalty and choose the variable having lowest cost in this selected row/column.
    Step 3: Allocate as much as possible to this variable.
    Step 4: Cross out the row or column which ever is statisfied and adjust the variable and required quantities.
    Step 5: Compute the penalties and repeat procedure till all rows and columns are satisfied.

    Source: http://cbom.atozmath.com/example/CBOM/Transportation.aspx?he=e&q=h2&ex=0
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

    def penalty(self, cost):
        gaps = np.zeros(cost.shape[0])
        for i, c in enumerate(cost):
            gaps[i] = max(c) - min(c)
        return gaps
    
    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):

            if show_iter:
                self.trans.print_frame(self.table)

            cost = self.table[1:-1, 1:-1]

            row_P = self.penalty(cost)
            col_P = self.penalty(cost.T)

            if max(row_P) > max(col_P):
                x = np.argmax(row_P)
                y = np.argmin(cost[x])
            else:
                y = np.argmax(col_P)
                x = np.argmin(cost[:, y])

            self.allocate(x + 1, y + 1)
            
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
                     [ 8, 16, 24]])
    supply = np.array([76, 82, 77])
    demand = np.array([72, 102, 41])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize HM1 with table that has been prepared before.
    HM2 = HeuristicMethod2(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = HM2.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
           C0    C1     C2      C3 Supply
R0      19(5)    30     50   10(2)      7
R1         70    30  40(7)   60(2)      9
R2         40  8(8)     70  20(10)     18
Demand      5     8      7      14     34

TOTAL COST: 779

example 2 unbalance problem
           C0      C1      C2  Dummy Supply
R0          4   8(76)       8      0     76
R1         16  24(21)  16(41)  0(20)     82
R2      8(72)   16(5)      24      0     77
Demand     72     102      41     20    235

TOTAL COST: 2424
'''
