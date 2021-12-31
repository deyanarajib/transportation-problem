import numpy as np
from transportation import Transportation

class KaragulSahinApproximation:
    """
    Karagul-Sahin's Algorithm
    1. Calculate the rij (pdm) and rji (psm) values for matrix A (wcd) and B (wcs)
    2. Calculate the weighted transportation cost matrix by multiplying the rates and the cost values and form A (wcd) and B (wcs) matrices.
    3. To start with the smallest weighted costs in the matrices wcd and wcs, make assignments taking into account the demand and supply constraints.
    4. If all demand are met, finish the algorithm. Otherwise, go back to Step 3.
    5. Compare the solution values of assignments matrices. Set the smaller solution as the initial solution.

    Source: K. Karagul and Y. Sahin, "A novel approximation method to obtain initial basic feasible solution of transportation problem", J. King Saud Univ. 2019.
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

    def solve_part(self, show_iter=False):

        while self.table.shape != (2, 2):

            if show_iter:
                self.trans.print_frame(self.table)

            #finding index of minimum cost
            mins = np.min(self.table[1:-1, 1:-1])
            x, y = np.argwhere(self.table[1:-1, 1:-1] == mins)[0]

            #allocated row x to column y or vice versa
            self.allocate(x + 1, y + 1)

        return self.alloc

    def find_cost(self, alloc, table):

        #finding total cost given (Ri, Cj, v)
        total_cost = 0
        for i, j, v in alloc:
            try:
                i = list(table[1:-1, 0]).index(i)
            except ValueError:
                i = list(table[1:-1, 0]).index("Dummy")

            try:
                j = list(table[0, 1:-1]).index(j)
            except ValueError:
                j = list(table[0, 1:-1]).index("Dummy")

            total_cost += v * table[i + 1, j + 1]

        return total_cost

    def solve(self, show_iter=False):

        supply = self.table[1:-1, -1]
        demand = self.table[-1, 1:-1]

        n = len(supply)
        m = len(demand)

        #compute Rij and Rji
        Rij, Rji = np.zeros((2, n, m), dtype=object)
        for i, s in enumerate(supply):
            for j, d in enumerate(demand):
                Rij[i, j] = d/s
                Rji[i, j] = s/d

        #solve for WCD and WCS
        min_cost = np.inf
        for R, title in zip([Rij, Rji], ["WCD", "WCS"]):
            
            if show_iter:
                print("{} SOLUSTION\n".format(title))

            #make a copy of table then multiply with Rij/Rji (WCD/WCS)
            cost = self.table[1:-1, 1:-1] * R
            supply = self.table[1:-1, -1]
            demand = self.table[-1, 1:-1]

            trans = Transportation(cost, supply, demand)
            trans.setup_table()

            ks = KaragulSahinApproximation(trans)

            alloc = ks.solve_part(show_iter=show_iter)
            total_cost = self.find_cost(alloc, self.table)

            if show_iter:
                print("{} TOTAL COST = {}\n".format(title, total_cost))

            #save allocation if it has minimum cost
            if total_cost < min_cost:
                min_cost = total_cost
                self.alloc = alloc[:]
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    #example 1 balance problem
    cost = np.array([[6,  8, 10],
                    [7, 11, 11],
                    [7,  5, 12]])
    supply = np.array([150, 175, 275])
    demand = np.array([200, 100, 300])

    #example 2 unbalance problem
    cost = np.array([[390, 380, 500],
                    [290, 280, 400],
                    [240, 230, 350]])
    supply = np.array([30000, 40000, 60000])
    demand = np.array([20000, 30000, 30000])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize Karagul-Sahin method with table that has been prepared before.
    KS = KaragulSahinApproximation(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = KS.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
            C0      C1       C2 Supply
R0       6(25)       8  10(125)    150
R1           7      11  11(175)    175
R2      7(175)  5(100)       12    275
Demand     200     100      300    600

TOTAL COST: 5050

example 2 unbalance problem
                C0          C1          C2     Dummy  Supply
R0             390         380         500  0(30000)   30000
R1             290         280  400(20000)  0(20000)   40000
R2      240(20000)  230(30000)  350(10000)         0   60000
Demand       20000       30000       30000     50000  130000

TOTAL COST: 23200000
'''
