import numpy as np
from transportation import Transportation

class AssigningShortestMinimax:
    """
    ASM Method Algorithm
    Step 1: Construct the transportation table from given transportation problem.
    Step 2: Subtract each row entries of the transportation table from the respective row minimum and then subtract each column entries of the resulting transportation table from respective column minimum.
    Step 3: Now there will be at least one zero in each row and in each column in the reduced cost matrix. Select the first zero (row-wise) occurring in the cost matrix. Suppose (i, j)th zero is selected, count the total number of zeros (excluding the selected one) in the ith row and jth column. Now select the next zero and count the total number of zeros in the corresponding row and column in the same manner. Continue it for all zeros in the cost matrix.
    Step 4: Now choose a zero for which the number of zeros counted in step 3 is minimum and supply maximum possible amount to that cell. If tie occurs for some zeros in step 3 then choose a (k.l)th zero breaking tie such that the total sum of all the elements in the kth row and lth column is maximum. Allocate maximum possible amount to that cell.
    Step 5: After performing step 4, delete the row or column for further calculation where the supply from a given source is depleted or the demand for a given destination is satisfied.
    Step 6: Check whether the resultant matrix possesses at least one zero in each row and in each column. If not, repeat step 2, otherwise go to step 7.
    Step 7: Repeat step 3 to step 6 until and unless all the demands are satisfied and all the supplies are exhausted.
    
    Source: B. Satheesh Kumara,*, R. Nandhinib and T. Nanthinic: "A comparative study of ASM and NWCR method in transportation problem", Malaya J. Mat. 5(2)(2017) 321–327.

    Algorithm of the revised version of ASM-Method
    Step 1 : Construct the transportation tableau from given TP. Check whether the problem is balanced or not. If the problem is balanced, go to Step 4, otherwise go to Step 2.
    Step 2 : If the problem is not balanced, then any one of the following two cases may arise:
        a) If total supply exceeds total demand, introduce an additional dummy column to the transportation table to absorb the excess supply. The unit transportation cost for the cells in this dummy column is set to ‘M’, where M > 0 is a very large but finite positive quantity. or
        b) If total demand exceeds total supply, introduce an additional dummy row to the transportation table to satisfy the excess demand. The unit transportation cost for the cells in this dummy row is set to ‘M’, where M>0 is a very large but finite positive quantity.
    Step 3 : 
        a) In case (a) of Step 2, identify the lowest element of each row and subtract it from each element of the respective row and then, in the resulting tableau, identify the lowest element of each column and subtract it from each element of the respective column and go to Step 5. or
        b) In case (b) of Step 2, identify the lowest element of each column and subtract it from each element of the respective column and then, in the resulting tableau, identify the lowest element of each row and subtract it from each element of the respective row and go to Step 5.
    Step 4 : Identify the lowest element of each row and subtract it from each element of the respective row and then, in the resulting tableau, identify the lowest element of each column and subtract it from each element of the respective column.
    Step 5 : In the reduced tableau, each row and each column contains at least one zero. Now, select the first zero (say zero) and count the number of zeros (excluding the selected one) in the row and column and record as a subscript of selected zero. Repeat this process for all zeros in the transportation tableau.
    Step 6 : Now, choose the cell containing zero for which the value of subscript is minimum and supply maximum possible amount to that cell. If tie occurs for 268 Abdul Quddoos et al. some zeros in Step 5, choose the cell of that zero for breaking tie such that the sum of all the elements in the row and column is maximum. Supply maximum possible amount to that cell.
    Step 7 : Delete that row (or column) for further consideration for which the supply from a given source is exhausted (or the demand for a given destination is satisfied). If, at any stage, the column demand is completely satisfied and row supply is completely exhausted simultaneously, then delete only one column (or row) and the remaining row (or column) is assigned a zero supply (or demand) in further calculation.
    Step 8 : Now, check whether the reduced tableau contains at least one zero in each row and each column. If this does not happens, repeat Step 4 otherwise go to Step 9.
    Step 9 : Repeat Step 5 to Step 8 till all the demands are satisfied and all the supplies are exhausted.
    
    Source: Abdul Quddoos, Shakeel Javaid* and M. M. Khalid: "A Revised Version of ASM-Method for Solving Transportation Problem", Int. J. Agricult. Stat. Sci. Vol. 12, Supplement 1, pp. 267-272, 2016.
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

    def reduce_rows(self):
        mins = np.min(self.table[1:-1, 1:-1], 1).reshape(-1, 1)
        self.table[1:-1, 1:-1] -= mins

    def reduce_cols(self):
        mins = np.min(self.table[1:-1, 1:-1], 0)
        self.table[1:-1, 1:-1] -= mins

    def select_index(self):
        zeros = np.argwhere(self.table[1:-1, 1:-1] == 0)
        n = zeros.shape[0]

        a, b, c = np.zeros((3, n))
        for i, (x, y) in enumerate(zeros):
            xx = list(self.table[x + 1, 1:-1])
            yy = list(self.table[1:-1, y + 1])
            
            a[i] = (xx.count(0) - 1) + (yy.count(0) -1)
            b[i] = sum(xx) + sum(yy)
            c[i] = (self.table[x + 1, -1] + self.table[-1, y + 1]) / 2

        mask = a == min(a)
        if len(a[mask]) > 1:
            select = np.zeros(n)
            select[mask] = b[mask]

            mask = np.all([mask, b == max(b)], 0)
            if len(select[mask]) > 1:
                
                select = np.array([np.inf] * n)
                select[mask] = c[mask]
                return zeros[np.argmin(select)]
            else:
                return zeros[np.argmax(select)]
        else:
            return zeros[np.argmin(a)]

    def revision(self):

        if self.table[-2, 1:-1].sum() == 0:
            #table has dummy row
            mins = np.min(self.table[1:-2, 1:-1], 0)
            self.table[1:-2, 1:-1] -= mins
            self.table[-2, 1:-1] = mins.copy()
            self.reduce_rows()
            self.table[-2, 1:-1] = max(self.table[-2, 1:-1]) - self.table[-2, 1:-1]

        elif self.table[1:-1, -2].sum() == 0:
            #table has dummy column
            mins = np.min(self.table[1:-1, 1:-2], 1)
            self.table[1:-1, 1:-2] -= mins.reshape(-1, 1)
            self.table[1:-1, -2] = mins.copy()
            self.reduce_cols()
            self.table[1:-1, -2] = max(self.table[1:-1, -2]) - self.table[1:-1, -2]
            
    def solve(self, show_iter=False, revision=False):

        if revision:
            #use ASM revision algorithm
            self.revision()
            if show_iter:
                self.trans.print_frame(self.table)

        while self.table.shape != (2, 2):

            self.reduce_rows()
            self.reduce_cols()
            x, y = self.select_index()
            self.allocate(x + 1, y + 1)

            if show_iter:
                self.trans.print_frame(self.table)
            
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    #example 1 balance problem
    cost = np.array([[11, 13, 17, 14],
                    [16, 18, 14, 10],
                    [21, 24, 13, 10]])
    supply = np.array([250, 300, 400])
    demand = np.array([200, 225, 275, 250])

    #example 2 unbalance problem
    cost = np.array([[2, 7, 14],
                     [3, 3,  1],
                     [5, 4,  7],
                     [1, 6,  2]])
    supply = np.array([5, 8, 7, 15])
    demand = np.array([7, 9, 18])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize ASM method with table that has been prepared before.
    ASM = AssigningShortestMinimax(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    #revision=True will using ASM Revision algorithm for unbalance problem, default=False.
    allocation = ASM.solve(show_iter=True, revision=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
             C0       C1       C2       C3 Supply
R0       11(25)  13(225)       17       14    250
R1      16(175)       18       14  10(125)    300
R2           21       24  13(275)  10(125)    400
Demand      200      225      275      250    950

TOTAL COST: 12075

example 2 unbalance problem
          C0    C1    C2 Dummy Supply
R0         2  7(4)    14  0(1)      5
R1         3     3  1(8)     0      8
R2         5  4(5)  7(2)     0      7
R3      1(7)     6  2(8)     0     15
Demand     7     9    18     1     35

TOTAL COST: 93

example 2 unbalance problem (revision)
          C0    C1     C2 Dummy Supply
R0      2(2)  7(2)     14  0(1)      5
R1         3     3   1(8)     0      8
R2         5  4(7)      7     0      7
R3      1(5)     6  2(10)     0     15
Demand     7     9     18     1     35

TOTAL COST: 79
'''
