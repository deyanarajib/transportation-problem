import numpy as np
from transportation import Transportation

class TheAdvanceMethod:
    """
    The Advance Method Algorithm
    Step 1: Select row/column index having minimum value in supply and demand as i (if it's row) or j (if it's column).
    Step 2: Select index of minimum cost in row/column has minimum supply.or demand as i (if it's row) or j (if it's column)
    Step 3: Let value = Xij.
    Step 4:
        a. Subtract this value from supply si and demand dj.
        b. If the supply si is 0, then cross (strike) that row and If the demand dj is 0 then cross (strike) that column.
        c. If min unit cost cell is not unique, then select the cell where supply/demand has minimum value.
    Step-5:	Repeact this steps for all uncrossed (unstriked) rows and columns until all supply and demand values are 0.
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

        cost = self.table[1:-1, 1:-1].copy()
        cost = np.where(cost % 2 == 1, cost, np.inf)
        mins = np.min(cost)

        cost = self.table[1:-1, 1:-1].copy()
        cost = np.where(cost % 2 == 1, cost - mins, cost)
            
        self.table[1:-1, 1:-1] = cost.copy()

        if show_iter:
            self.trans.print_frame(self.table)

        x, y = np.argwhere(self.table[1:-1, 1:-1] == 0)[0]
        self.allocate(x + 1, y + 1)

        while self.table.shape != (2, 2):

            cost = self.table[1:-1, 1:-1].copy()
            supply = self.table[1:-1, -1].copy()
            demand = self.table[-1, 1:-1].copy()

            n, m = cost.shape

            if min(supply) < min(demand):
                x = np.argmin(supply)
                mins = min(cost[x])
                if list(cost[x]).count(mins) == 1:
                    y = np.argmin(cost[x])
                else:
                    i = np.arange(m)[cost[x] == mins]
                    y = i[np.argmin(demand[i])]
            else:
                y = np.argmin(demand)
                mins = min(cost[:, y])
                if list(cost[:, y]).count(mins) == 1:
                    x = np.argmin(cost[:, y])
                else:
                    i = np.arange(n)[cost[:, y] == mins]
                    x = i[np.argmin(supply[i])]

            self.allocate(x + 1, y + 1)

            if show_iter:
                self.trans.print_frame(self.table)
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    #example 1 balance problem
    cost = np.array([[489, 350, 142, 365, 424, 272, 272],
                    [272, 410, 350, 489, 365, 489, 253],
                    [424, 489, 365, 253, 410, 410, 142],
                    [365, 257, 472, 272, 350, 410, 142],
                    [350, 272, 365, 472, 410, 257, 272],])
    supply = np.array([2314, 2628, 2493, 2268, 2398])
    demand = np.array([1900, 1778, 1694, 1851, 1959, 1838, 1081])

    #example 2 unbalance problem
    cost = np.array([[60, 120, 75, 180],
                    [58, 100, 60, 165],
                    [62, 110, 65, 170],
                    [65, 115, 80, 175],
                    [70, 135, 85, 195],])
    supply = np.array([8000, 9200, 6250, 4900, 6100])
    demand = np.array([5000, 2000, 10000, 6000])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize TAM method with table that has been prepared before.
    TAM = TheAdvanceMethod(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = TAM.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
               C0         C1         C2         C3         C4         C5         C6 Supply
R0            489        350        142  365(1180)        424  272(1134)        272   2314
R1            272        410        350        489  365(1547)        489  253(1081)   2628
R2      424(1822)        489        365   253(671)        410        410        142   2493
R3        365(78)  257(1778)        472        272   350(412)        410        142   2268
R4            350        272  365(1694)        472        410   257(704)        272   2398
Demand       1900       1778       1694       1851       1959       1838       1081  12101

TOTAL COST: 3948441

example 2 unbalance problem
              C0         C1        C2         C3    Dummy Supply
R0            60        120        75        180  0(8000)   8000
R1      58(5000)        100  60(4200)        165        0   9200
R2            62        110   65(250)  170(6000)        0   6250
R3            65  115(2000)        80        175  0(2900)   4900
R4            70        135  85(5550)        195   0(550)   6100
Demand      5000       2000     10000       6000    11450  34450

TOTAL COST: 2280000
'''
