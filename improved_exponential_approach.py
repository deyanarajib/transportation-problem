import numpy as np
import itertools
from transportation import Transportation

class ImprovedExponentialApproach:
    """
    Algoritma Improved Exponential Approach (IND)
    Langkah 1: Membentuk model transportasi (Tabel) dari masalah transportasi yang diberikan. Apabila tabel transportasi belum seimbang ke langkah 2, jika sudah seimbang langsung ke langkah 3.
    Langkah 2: Jika kolom (baris) dummy ditambahkan, kurangi setiap entri kolom (baris) dari minimum kolom (baris) masing-masing. Mengganti biaya dummy dengan biaya yang terbesar dari tabel yang sudah direduksi sebelumnya. Jika kolom dummy yang ditambahkan maka ke step 3a lalu 3b dan jika baris dummy yang ditambahkan maka ke step 3b lalu 3a.
    Langkeh 3:
        a. Mengurangi setiap entri baris dari tabel transportasi dari minimum baris masing-masing.
        b. Mengurangi setiap entri kolom tabel transportasi dari kolom minimum masing-masing. Sehingga setiap baris dan kolom akan memiliki setidaknya satu nol.
    Langkah 4: Mengecek apakah setiap kolom permintaan kurang dari atau sama dengan jumlah persediaan dalam baris dengan melihat pada kolom yang biaya tereduksinya bernilai nol. Mengecek apakah setiap baris persediaan kurang dari atau sama dengan jumlah permintaan dalam kolom dengan melihat pada baris yang biaya tereduksinya bernilai nol. Apabila syarat tersebut terpenuhi langsung ke langkah 7. Jika tidak, lanjut ke langkah 5.
    Langkeh 5: Menarik garis horisontal dan vertikal pada semua baris dan kolom yang memiliki angka nol dengan jumlah garis minimum, sedemikian hingga biaya yang tidak memenuhi pada langkah 4 tidak tertutup.
    Langkeh 6: Memilih biaya terkecil pada sel yang tidak terkena garis, kemudian mengurangkan sebesar biaya terpilih ke semua biaya yang tidak terkena garis. Menambahkan sebesar biaya terpilih ke semua biaya yang terletak pada perpotongan dua garis. Kembali ke langkah 4.
    Langkeh 7: Memilih nol yang terdapat dalam tabel. Menghitung jumlah total angka nol (tidak termasuk yang dipilih) dalam baris dan kolom yang bersesuaian. Menetapkan penalti eksponen (jumlah nol berturut-turut masing-masing baris dan kolom). Mengulangi prosedur diatas untuk semua nol dalam tabel.
    Langkeh 8: Mengalokasikan nilai sel dengan jumlah maksimum yang mungkin dengan memperhatikan prioritas pengalokasian sebagai berikut:
        a. Nol yang memiliki penalti eksponen bernilai 0.
        b. Nol yang memiliki penalti eksponen bernilai 1.
        c. Memilih sel yang memiliki biaya tereduksi terbesar dan dinamakan ( ). Jika terdapat lebih dari satu sel, maka memilih sel lain dengan biaya tereduksi terbesar berikutnya. Mengalokasikan pada nol yang terdapat pada baris i atau kolom j dengan penalti eksponen yang minimum hingga persediaan baris i atau permintaan kolom j terpenuhi.
        d. Memilih nol dengan penalti eksponen minimum pada tabel. Jika terjadi nilai penalti eksponen sama untuk setiap sel maka pertama memeriksa nilai permintaan dan persediaan, menghitung nilai rata-ratanya dan menetapkan alokasi untuk nilai rata-rata terendah. Apabila tetap sama maka mengalokasikan pada sel dengan biaya yang terendah sebelum direduksi.
    Langkeh 9: Menandai baris atau kolom (di mana persediaan atau permintaan menjadi nol) untuk tidak dimasukan dalam perhitungan selanjutnya, kemudian kembali ke langkah 4 hingga semua permintaan dan persediaan terpenuhi.
    Langkeh 10: Menghitung biaya optimumnya.

    Sumber: Dimas Alfan Hidayat, Siti Khabibah, dan Suryoto, "Metode Improved Exponential Approach dalam Menentukan Solusi Optimum pada Masalah Transportasi", Universitas Diponegoro.
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

    def minimum_line(self, cost):
        X = cost.copy()

        count = 0
        while 0 in X:
            R, C = np.argwhere(X == 0).T.tolist()

            temp = []
            for i in sorted(set(R)):
                temp.append(["R", i, R.count(i)])
            for j in sorted(set(C)):
                temp.append(["C", j, C.count(j)])
            temp = np.array(temp, dtype=object)

            x, y, _ = temp[np.argmax(temp[:, 2])]
            if x == "R":
                X[y] = -1
            else:
                X[:, y] = -1
            count += 1
        return min([max(X.shape), count])


    def get_score(self, cost):
        R, C = np.argwhere(cost == 0).T
        supply = self.table[1:-1, -1]
        demand = self.table[-1, 1:-1]

        score = 0
        for i in sorted(set(R)):
            if supply[i] <= demand[C[R == i]].sum():
                score += 1
        for j in sorted(set(C)):
            if demand[j] <= supply[R[C == j]].sum():
                score += 1
        return score


    def exponential_approach(self, show_iter=False):

        n, m = [i -2 for i in self.table.shape]

        self.reduce_rows()
        if show_iter:
            self.trans.print_frame(self.table)

        self.reduce_cols()
        if show_iter:
            self.trans.print_frame(self.table)

        min_line = self.minimum_line(self.table[1:-1, 1:-1])
        
        tried = []
        while True:

            score = self.get_score(self.table[1:-1, 1:-1])
            if score == n + m:
                break

            R, C = np.argwhere(self.table[1:-1, 1:-1] == 0).T
            setR = ["R{}".format(i) for i in sorted(set(R))]
            setC = ["C{}".format(j) for j in sorted(set(C))]

            maxscore = -np.inf
            for comb in itertools.combinations(setR + setC, min_line):
                r = [int(i[1:]) for i in comb if i.startswith("R")]
                c = [int(j[1:]) for j in comb if j.startswith("C")]

                if len(r) in [0, n] or len(c) in [0, m] or len(r + c) > max([n, m]):
                    continue

                cost = self.table[1:-1, 1:-1].copy()
                dels = np.delete(np.delete(cost, c, 1), r, 0)
                minK = np.min(dels)
                
                if minK == 0:
                    continue

                for i in range(n):
                    for j in range(m):
                        if i in r and j in c:
                            cost[i, j] += minK
                        elif i not in r and j not in c:
                            cost[i, j] -= minK

                score_iter = self.get_score(cost)
                if score_iter > maxscore:
                    maxscore = score_iter
                    pick = cost.copy()
                    rc = [r, c]

            if rc in tried:
                min_line += 1
                continue
            else:
                tried.append(rc)
            
            self.table[1:-1, 1:-1] = pick.copy()
            
            if show_iter:
                self.trans.print_frame(self.table)

    def solve(self, show_iter=False):
        
        self.exponential_approach(show_iter=show_iter)

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
    cost = np.array([[73, 40, 9, 79, 20],
                    [62, 93, 96, 8,  13],
                    [96, 65, 80, 50, 65],
                    [57, 58, 29, 12, 87],
                    [56, 23, 87, 18, 12]])
    supply = np.array([8, 7, 9, 3, 5])
    demand = np.array([6, 8, 10, 4, 4])

    #example 2 unbalance problem
    cost = np.array([[10,  2, 16, 14, 10],
                    [ 6, 18, 12, 13, 16],
                    [ 8,  4, 14, 12, 10],
                    [14, 22, 20,  8, 18]])
    supply = np.array([300, 500, 825, 375])
    demand = np.array([350, 400, 250, 150, 400])

    #initialize transportation problem
    trans = Transportation(cost, supply, demand)

    #setup transportation table.
    #minimize=True for minimization problem, change to False for maximization, default=True.
    #ignore this if problem is minimization and already balance
    trans.setup_table(minimize=True)

    #initialize IEA with table that has been prepared before.
    IEA = ImprovedExponentialApproach(trans)

    #solve problem and return allocation lists which consist n of (Ri, Cj, v)
    #Ri and Cj is table index where cost is allocated and v it's allocated value.
    #(R0, C1, 3) means 3 cost is allocated at Row 0 and Column 1.
    #show_iter=True will showing table changes per iteration, default=False.
    allocation = IEA.solve(show_iter=False)

    #print out allocation table in the form of pandas DataFrame.
    #(doesn't work well if problem has large dimension).
    trans.print_table(allocation)

#Result from example problem above
'''
example 1 balance problem
           C0     C1     C2    C3     C4 Supply
R0         73     40   9(8)    79     20      8
R1         62     93     96  8(4)  13(3)      7
R2      96(5)  65(4)     80    50     65      9
R3      57(1)     58  29(2)    12     87      3
R4         56  23(4)     87    18  12(1)      5
Demand      6      8     10     4      4     32

TOTAL COST: 1102

example 2 unbalance problem
            C0      C1       C2      C3       C4   Dummy Supply
R0          10  2(300)       16      14       10       0    300
R1      6(250)      18  12(250)      13       16       0    500
R2      8(100)  4(100)       14      12  10(400)  0(225)    825
R3          14      22       20  8(150)       18  0(225)    375
Demand     350     400      250     150      400     450   2000

TOTAL COST: 11500
'''
