import numpy as np
from transportation import Transportation

class MaximumDevideMinimumAllotment:
    """
    MDMA Algorithm
    Step 1: Construct the Transportation Table (TT) for the given Pay Off Matrix (POM).
    Step 2: Choose the maximum element(ME) from POM and divide all elements by the ME in the Constructed Transportation Table (CTT).
    Step 3: Supply the demand for the minimum element newly CTT.
    Step 4: Select the next maximum element in CTT and repeat the same procedure for remaining allotments

    Source: A. Amaravathy, K. Thiagarajan and S. Vimala, "MDMA Method- An Optimal Solution for Transportation Problem", Middle-East Journal of Scientific Research 24 (12): 3706-3710, 2016.
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

            cost = self.table[1:-1, 1:-1].copy()
            self.table[1:-1, 1:-1] /= np.max(cost)
            x, y = np.argwhere(cost == np.min(cost))[0]

            #allocated row x to column y or vice versa
            self.allocate(x + 1, y + 1)

            if show_iter:
                self.trans.print_frame(self.table)
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    #example 1 balance problem
    cost = np.array([[12,  4, 9, 5, 9],
                    [ 8,  1, 6, 6, 7],
                    [ 1, 12, 4, 7, 7],
                    [10, 15, 6, 9, 1]])

    supply = np.array([55, 45, 30, 50])
    demand = np.array([40, 20, 50, 30, 40])

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

    #initialize MDMA method with table that has been prepared before.
    MDMA = MaximumDevideMinimumAllotment(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = MDMA.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
            C0     C1     C2     C3     C4 Supply
R0      12(10)      4  9(15)  5(30)      9     55
R1           8  1(20)  6(25)      6      7     45
R2       1(30)     12      4      7      7     30
R3          10     15  6(10)      9  1(40)     50
Demand      40     20     50     30     40    180

TOTAL COST: 705

example 2 unbalance problem
           C0      C1      C2  Dummy Supply
R0      4(56)       8       8  0(20)     76
R1         16  24(41)  16(41)      0     82
R2      8(16)  16(61)      24      0     77
Demand     72     102      41     20    235

TOTAL COST: 2968
'''
