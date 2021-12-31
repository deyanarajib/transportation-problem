import numpy as np
from transportation import Transportation

class HarmonicMeanApproach:
    """
    Harmonic Mean Approach Algorithm
    1. Check wheter the given transportation problem is balanced or not. If not, balance or by adding dummy row or column. Then go to next step.
    2. Find the harmonic mean for each row and each column. Then find the maximum value among that.
    3. Allocate the minimum supply or demand at the place of minimum value of the related row or column.
    4. Repeat the step 2 and 3 until all the demand are satisfied and all the the supplies are exhausted.
    5. Total minimum cost = sum of the product of the cost and it's corresponding allocated values of supply or demand.

    Source: https://medium.com/@ETE/a-new-method-to-solve-transportation-problem-harmonic-mean-approach-juniper-publishers-9b3d956276e2
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

    def hmean(self, cost):
        hm = []
        n, m = cost.shape
        for i in range(n):
            try:
                hm.append(m / sum(1/cost[i]))
            except ZeroDivisionError:
                hm.append(0)
        return hm

    def solve(self, show_iter=False):

        while self.table.shape != (2, 2):

            cost = self.table[1:-1, 1:-1].copy()

            hmrow = self.hmean(cost)
            hmcol = self.hmean(cost.T)

            if max(hmrow) > max(hmcol):
                x = np.argmax(hmrow)
                y = np.argmin(cost[x])
            else:
                y = np.argmax(hmcol)
                x = np.argmin(cost[:, y])

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

    #initialize HMA method with table that has been prepared before.
    HMA = HarmonicMeanApproach(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = HMA.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
               C0        C1         C2         C3         C4         C5        C6 Supply
R0            489  350(909)  142(1405)        365        424        272       272   2314
R1      272(1900)       410        350        489        365        489  253(728)   2628
R2            424       489   365(289)  253(1851)        410        410  142(353)   2493
R3            365  257(309)        472        272  350(1959)        410       142   2268
R4            350  272(560)        365        472        410  257(1838)       272   2398
Demand       1900      1778       1694       1851       1959       1838      1081  12101

TOTAL COST: 3232307

example 2 unbalance problem
              C0         C1        C2         C3    Dummy Supply
R0      60(5000)        120  75(2550)        180   0(450)   8000
R1            58  100(2000)  60(1200)  165(6000)        0   9200
R2            62        110  65(6250)        170        0   6250
R3            65        115        80        175  0(4900)   4900
R4            70        135        85        195  0(6100)   6100
Demand      5000       2000     10000       6000    11450  34450

TOTAL COST: 2159500
'''
