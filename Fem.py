import numpy as np
import matplotlib.pyplot as plt
from math import floor,ceil,sqrt
global a,b,c,d
global m,n,h,k,area,numberNode,numberTriangle
global listCoordinateX,listCoordinateY
global listIndexNodeTrong,listIndexNodeCanhMot,listIndexNodeCanhHai,listIndexNodeCanhBa,listIndexNodeCanhBon
global listIndexNodeGocMot,listIndexNodeGocHai,listIndexNodeGocBa,listIndexNodeGocBon
from timeit import default_timer as timer


class Mesh():
    def __init__(self,khoangTheoChieuNgang,khoangTheoChieuCao,soDoanTheoChieuNgang,soDoanTheoChieuCao):
        self.a=khoangTheoChieuNgang[0]
        self.b=khoangTheoChieuNgang[1]
        self.c=khoangTheoChieuCao[0]
        self.d=khoangTheoChieuCao[1]
        self.m=soDoanTheoChieuNgang
        self.n=soDoanTheoChieuCao
        global a, b, c, d
        global m, n, h, k, area, numberNode, numberTriangle
        global listCoordinateX, listCoordinateY
        global listIndexNodeTrong, listIndexNodeCanhMot, listIndexNodeCanhHai, listIndexNodeCanhBa, listIndexNodeCanhBon
        global listIndexNodeGocMot, listIndexNodeGocHai, listIndexNodeGocBa, listIndexNodeGocBon
        (a, b, c, d, m, n) = (self.a, self.b, self.c, self.d, self.m, self.n)
        h = (b - a) / m
        k = (d - c) / n
        area = 1 / 2 * h * k
        numberNode = (m + 1) * (n + 1)
        numberTriangle = 2 * m * n
        listCoordinateX = np.zeros((1, m + 1)).flatten()
        listCoordinateY = np.zeros((1, n + 1)).flatten()
        for i in range(m + 1):
            listCoordinateX[i] = a + i * (b - a) / m
        for i in range(n + 1):
            listCoordinateY[i] = d - i * (d - c) / n
        listIndexNodeTrong, listIndexNodeCanhMot, listIndexNodeCanhHai, listIndexNodeCanhBa, listIndexNodeCanhBon = [], [], [], [], []
        listIndexNodeGocMot, listIndexNodeGocHai, listIndexNodeGocBa, listIndexNodeGocBon = [], [], [], []
        for i in range(numberNode):
            if (Node(i).getTypeOfNode() == 1):
                listIndexNodeTrong.append(i)
                continue
            if (Node(i).getTypeOfNode() == 2):
                listIndexNodeCanhMot.append(i)
                continue
            if (Node(i).getTypeOfNode() == 3):
                listIndexNodeCanhHai.append(i)
                continue
            if (Node(i).getTypeOfNode() == 4):
                listIndexNodeCanhBa.append(i)
                continue
            if (Node(i).getTypeOfNode() == 5):
                listIndexNodeCanhBon.append(i)
                continue
            if (Node(i).getTypeOfNode() == 6):
                listIndexNodeGocMot.append(i)
                continue
            if (Node(i).getTypeOfNode() == 7):
                listIndexNodeGocHai.append(i)
                continue
            if (Node(i).getTypeOfNode() == 8):
                listIndexNodeGocBa.append(i)
                continue
            if (Node(i).getTypeOfNode() == 9):
                listIndexNodeGocBon.append(i)
                continue

    def getInfo(self):
        print("Khoảng theo trục Ox là: [",a,',',b,']')
        print("Khoảng theo trục Oy là: [",c,',',d,']')
        print("Tổng số Node là: ",numberNode)
        print("Tập các node là điểm trong: ",listIndexNodeTrong)
        print("Tập các điểm biên trên cạnh một: ",listIndexNodeCanhMot)
        print("Tập các điểm biên trên cạnh hai: ",listIndexNodeCanhHai)
        print("Tập các điểm biên trên cạnh ba: ",listIndexNodeCanhBa)
        print("Tập các điểm biên trên cạnh bốn: ",listIndexNodeCanhBon)
        print("Điểm góc một: ",listIndexNodeGocMot)
        print("Điểm góc hai: ",listIndexNodeGocHai)
        print("Điểm góc ba: ",listIndexNodeGocBa)
        print("Điểm góc bốn: ",listIndexNodeGocBon)
        print("Độ dài đoạn chia theo chiều ngang: ",h)
        print("Độ dài đoạn chia theo chiều cao :",k)
        print("Tổng số tam giác là: ",numberTriangle)
        print("Diện tích một tam giác được chia là: ",area)

    def show(self):

        for i in range(numberNode):
            for j in range(numberNode):
                dieukien1 = abs(Node(j).getX_WithIndex() - Node(i).getX_WithIndex()) >= 2
                dieukien2 = abs(Node(j).getY_WithIndex() - Node(i).getY_WithIndex()) >= 2
                dieukien3 = (Node(j).getY_WithIndex() - Node(i).getY_WithIndex()) == 1 and (
                        Node(j).getX_WithIndex() - Node(i).getX_WithIndex()) == 1
                dieukien4 = (Node(j).getY_WithIndex() - Node(i).getY_WithIndex()) == -1 and (
                        Node(j).getX_WithIndex() - Node(i).getX_WithIndex()) == -1
                if (dieukien1 or dieukien2 or dieukien3 or dieukien4):
                    continue
                else:
                    drawLine(i, j)
        plt.plot([1.1, 1.1], 'w')
        plt.plot([-0.1, -0.1], [0, 1], 'w')
        for i in range(m + 1):
            print(d)
            plt.text(Node(i).getX(), self.d + 0.07, str(i), fontsize=12, horizontalalignment='center',
                     verticalalignment='center')
        for i in listIndexNodeCanhBon:
            plt.text(self.a - 0.07, Node(i).getY(), str(Node(i).getY_WithIndex()), fontsize=15,
                     horizontalalignment='center',
                     verticalalignment='center')
        for i in listIndexNodeGocBon:
            plt.text(self.a - 0.07, Node(i).getY(), str(Node(i).getY_WithIndex()), fontsize=14,
                     horizontalalignment='center',
                     verticalalignment='center')
        for i in listIndexNodeGocMot:
            plt.text(self.a - 0.07, Node(i).getY(), str(Node(i).getY_WithIndex()), fontsize=14,
                     horizontalalignment='center',
                     verticalalignment='center')
        for i in range(numberNode):
            plt.text(Node(i).getX() + 0.02, Node(i).getY() - 0.05, str(i), fontsize=14,
                     horizontalalignment='center',
                     verticalalignment='center', color='blue')
        for i in range(1, numberTriangle + 1):
            (x1, x2, x3) = Triangle(i).getCoordinatesWithIndex()
            x = (Node(x1).getX() + Node(x2).getX()) / 2
            y = (Node(x1).getY() + Node(x3).getY()) / 2
            if (int(i) % 2 == 0):
                x = x + 0.06
                y = y - 0.08
            else:
                x = x - 0.06
                y = y - 0.01
            print(x, y)
            plt.text(x, y, str(i), fontsize=14,
                     horizontalalignment='center',
                     verticalalignment='center', color='red')
        plt.show()
class Node():
    'Hàm tạo các node của bài toán'
    def __init__(self,id):
        self.id=id
        self.m=m
        self.n=n
    def getCoordinatesWithIndex(self):
        return np.array([self.getX_WithIndex(),self.getY_WithIndex()])
    def getX_WithIndex(self):
        return int(self.id%(self.m+1))
    def getY_WithIndex(self):
        return int(floor(self.id/(self.m+1)))
    def getCoordinates(self):
        return np.array([listCoordinateX[self.getX_WithIndex()], listCoordinateY[self.getY_WithIndex()]])
    def getX(self):
        return listCoordinateX[self.getX_WithIndex()]
    def getY(self):
        return listCoordinateY[self.getY_WithIndex()]
    def getTypeOfNode(self):
        if (self.getX_WithIndex() not in [0,self.m] and self.getY_WithIndex() not in [0,self.n]):
            return 1
        if(self.getY_WithIndex()==0 and self.getX_WithIndex() not in [0,self.m]):
            return 2
        if(self.getX_WithIndex()==m and self.getY_WithIndex() not in [0,self.n]):
            return 3
        if(self.getY_WithIndex()==n and self.getX_WithIndex() not in [0,self.m]):
            return 4
        if(self.getX_WithIndex()==0 and self.getY_WithIndex() not in [0,self.n]):
            return 5
        if(self.getX_WithIndex()==0 and self.getY_WithIndex()==0):
            return 6
        if(self.getX_WithIndex()==m and self.getY_WithIndex()==0):
            return 7
        if(self.getX_WithIndex()==m and self.getY_WithIndex()==n):
            return 8
        if(self.getX_WithIndex()==0 and self.getY_WithIndex()==n):
            return 9

    def getTriangleAdjacent(self):
        if(Node(self.id).getTypeOfNode()==1):
            x1 = floor((2 * self.m * (self.id - self.m - 1) + 1) / (self.m + 1)) + 2 - ceil(
                ((self.id + 2) % (m + 1)) / (m + 1))
            for i in [x1, x1 + 1, x1 + 2, x1 - 1, x1 - 2, x1 - 3, x1 + 3]:
                if (Triangle(i).getCoordinatesWithIndex()[0] == self.id):
                    x1 = i
                    break
            x2 = x1 + 1
            x3 = x1 + 2
            x4 = x1 + 2 * m - 1
            x5 = x4 + 1
            x6 = x4 + 2
            return [(x1,x2,x3,x4,x5,x6),(1,2,3,4,5,6)]
        if(Node(self.id).getTypeOfNode()==2):
            return [(self.id * 2 - 1, self.id * 2, self.id * 2 + 1),(4,5,6)]
        if(Node(self.id).getTypeOfNode()==3):
            return [(2*self.m*self.getY_WithIndex(),2*self.m*self.getY_WithIndex()-1+2*self.m,2*self.m*self.getY_WithIndex()+2*self.m),(1,4,5)]
        if(Node(self.id).getTypeOfNode()==4):
            x3 = 2 * self.m * self.n - 2 * ((self.m + 1) * (self.n + 1) - 1 - self.id)
            return [(x3, x3 + 1, x3 + 2),(1,2,3)]
        if(Node(self.id).getTypeOfNode()==5):
            return [(2*self.m*(self.getY_WithIndex()-1)+1,2*self.m*(self.getY_WithIndex()-1)+2,2*self.m*(self.getY_WithIndex()-1)+1+2*self.m),(2,3,6)]
        if(Node(self.id).getTypeOfNode()==6):
            return [[1],[6]]
        if(Node(self.id).getTypeOfNode()==7):
            return [(2*self.m-1,2*self.m),(4,5)]
        if(Node(self.id).getTypeOfNode()==8):
            return [[2*self.m*self.n],[1]]
        if(Node(self.id).getTypeOfNode()==9):
            return [(2*self.m*(self.n-1)+1,2*self.m*(self.n-1)+2),(2,3)]
class Triangle():
    'Hàm xử lí các tam giác'
    def __init__(self, id):
        self.id = id
        self.m = m
        self.n = n
    def getCoordinatesWithIndex(self):
        if(self.id%2==1):
            x1=(self.id+1)/2+floor((self.id-1)/(2*self.m))-1
            x2=(self.id+1)/2+floor((self.id-1)/(2*self.m))
            x3=(self.id+1)/2+floor((self.id-1)/(2*self.m))+self.m
            return (int(x1),int(x2),int(x3))
        if(self.id%2==0):
            x4 = self.id / 2 + floor((self.id - 2) / (2 * self.m)) + m + 1
            x5=self.id/2+floor((self.id-2)/(2*self.m))+m
            x6 = self.id / 2 + floor((self.id - 2) / (2 * self.m))
            return (int(x4),int(x5),int(x6))
    def getCoordinates(self):
        (x1,x2,x3)=Triangle(self.id).getCoordinatesWithIndex()
        return (Node(x1).getCoordinates(),Node(x2).getCoordinates(),Node(x3).getCoordinates())

def PhiTriangleType1(x,indexOfNode):
    return 1+1/h*(x[0]-Node(int(indexOfNode)).getX())-1/k*(x[1]-Node(int(indexOfNode)).getY())
def PhiTriangleType2(x,indexOfNode):
    return 1-1/k*(x[1]-Node(int(indexOfNode)).getY())
def PhiTriangleType3(x,indexOfNode):
    return 1-1/h*(x[0]-Node(int(indexOfNode)).getX())
def PhiTriangleType4(x,indexOfNode):
    return 1+1/h*(x[0]-Node(int(indexOfNode)).getX())
def PhiTriangleType5(x,indexOfNode):
    return  1+1/k*(x[1]-Node(int(indexOfNode)).getY())
def PhiTriangleType6(x,indexOfNode):
    return 1-(1/h)*(x[0]-Node(int(indexOfNode)).getX())+(1/k)*(x[1]-Node(int(indexOfNode)).getY())

def GradientPhiTriangleType1(x,indexOfNode):
    return np.array([1/h,-1/k])
def GradientPhiTriangleType2(x,indexOfNode):
    return np.array([0,-1/k])
def GradientPhiTriangleType3(x,indexOfNode):
    return np.array([-1/h,0])
def GradientPhiTriangleType4(x,indexOfNode):
    return np.array([1/h,0])
def GradientPhiTriangleType5(x,indexOfNode):
    return np.array([0,1/k])
def GradientPhiTriangleType6(x,indexOfNode):
    return np.array([-1/h,1/k])

def drawLine(p1, p2):
    aa=np.array([Node(p1).getX(),Node(p2).getX()])
    bb=np.array([Node(p1).getY(),Node(p2).getY()])
    plt.plot(aa,bb,'b',marker = 'o')

Phi=[None,PhiTriangleType1,PhiTriangleType2,PhiTriangleType3,PhiTriangleType4,PhiTriangleType5,PhiTriangleType6]
GradientPhi=[None,GradientPhiTriangleType1,GradientPhiTriangleType2,GradientPhiTriangleType3,GradientPhiTriangleType4,GradientPhiTriangleType5,GradientPhiTriangleType6]


can15=sqrt(15)
z=1/21
W=[9/40]+[(155-can15)/1200]*3+[(155+can15)/1200]*3
Epsilon=[0]*7
Epsilon[0]=np.array([1/3,1/3]).flatten()
Epsilon[1]=np.array([z*(6-can15),z*(6+can15)]).flatten()
Epsilon[2]=np.array([z*(9+2*can15),z*(6-can15)]).flatten()
Epsilon[3]=np.array([z*(6-can15),z*(9+2*can15)]).flatten()
Epsilon[4]=np.array([z*(6+can15),z*(6+can15)]).flatten()
Epsilon[5]=np.array([z*(6+can15),z*(9-2*can15)]).flatten()
Epsilon[6]=np.array([z*(9-2*can15),z*(6+can15)]).flatten()



f=lambda x:x[0]+x[1]
alpha=lambda x:x[0]*2+x[1]

Mesh1 = Mesh([0,2],[0,2],100,100)
#Mesh1.show()
VectoVePhai=np.zeros((numberNode,1)).flatten()

# # Tính vecto f neu chua tinh tich phan duong

for i in range(numberNode):
    listIdOfTriangleAdjacent = Node(i).getTriangleAdjacent()[0]  # List các chỉ số tam giác kề với đỉnh i
    listIdOfPhil=Node(i).getTriangleAdjacent()[1]                # List các chỉ số của hàm Phi  với tam giác tương ứng
    # Ví dụ với i=30 thì listIdOfTriangleAdjacent=(37, 38, 55) cho biết có 3 tam giác có chỉ số thế kia cần tính tích phân
                                                          # Tương ứng listIdOfPhil=(2,3,6) cho biết với tam giác 37 dùng hàm phi2, với tam giác 38 dùng phi3,..
    VectoVePhai[i]=0

    # Tính tích phân trên từng tam giác kề với đỉnh i:
    for j in range(len(listIdOfTriangleAdjacent)):
        idOfMomentTriangle = int(listIdOfTriangleAdjacent[j])  # Id của tam giác đang xét
        idOfPhi=int(listIdOfPhil[j])
        resultWithThisTriangle = 0
        triangle = Triangle(idOfMomentTriangle)
        (x1, x2, x3) = triangle.getCoordinates()  # Tọa độ 3 đỉnh của tam giác đang xét
        sum=0
        for t in range(7):  # Bắt đầu dùng 7 điểm Gauss để tính tích phần trên tam giác đang xét
            epsilon_t=x1+Epsilon[t][0]*(x2-x1)+Epsilon[t][1]*(x3-x1)
            sum+=0.5*f(epsilon_t)*Phi[idOfPhi](epsilon_t,i)*W[t]
        resultWithThisTriangle=2*area*sum
        VectoVePhai[i]+= resultWithThisTriangle

A=np.zeros((numberNode,numberNode))             # Tạo ma trận A kích thước numberNode x NumberNode
## Tính ma trận A chưa kể tích phân đường:
for i in range(numberNode):
    for j in range(numberNode):
        start = timer()
        dieukien1=abs(Node(j).getX_WithIndex()-Node(i).getX_WithIndex())>=2
        dieukien2=abs(Node(j).getY_WithIndex()-Node(i).getY_WithIndex())>=2
        dieukien3=(Node(j).getY_WithIndex()-Node(i).getY_WithIndex())==1 and (Node(j).getX_WithIndex()-Node(i).getX_WithIndex())==1
        dieukien4=(Node(j).getY_WithIndex()-Node(i).getY_WithIndex())==-1 and (Node(j).getX_WithIndex()-Node(i).getX_WithIndex())==-1
        if ( dieukien1 or dieukien2 or dieukien3 or dieukien4 ):
            print("skipA[{0}][{1}]".format(i,j))
            print(str(timer()-start))
            input()
            continue
        else:
            start = timer()
            listIdOfTriangleAdjacent_i = Node(i).getTriangleAdjacent()[0]  # List các chỉ số tam giác kề với đỉnh i
            listIdOfPhil_i = Node(i).getTriangleAdjacent()[1]             # List các chỉ số của hàm Phi  với tam giác tương ứng
            listIdOfTriangleAdjacent_j = Node(j).getTriangleAdjacent()[0]  # List các chỉ số tam giác kề với đỉnh j
            listIdOfPhil_j = Node(j).getTriangleAdjacent()[1]
            giao = []                                                     # List các tam giác kề giao của 2 đỉnh i , j
            for t in listIdOfTriangleAdjacent_i:
                if t in listIdOfTriangleAdjacent_j:
                    giao.append(t)
            for zz in giao:                                                 # Hiện tại zz là id của tam giác đang xét
                p = listIdOfTriangleAdjacent_i.index(zz)
                q = listIdOfTriangleAdjacent_j.index(zz)
                idOfPhi_i_WithThisTriangle=listIdOfPhil_i[p]
                idOfPhi_j_WithThisTriangle=listIdOfPhil_j[q]
                (x1,x2,x3)=Triangle(zz).getCoordinates()                    # Tọa độ 3 đỉnh của tam giác đang xét
                sum=0
                resultWithThisTriangle=0
                for t in range(7):                                          # Bắt đầu dùng tích phân Gauss để tính tích phân trên tam giác zz
                    epsilon_t = x1 + Epsilon[t][0] * (x2 - x1) + Epsilon[t][1] * (x3 - x1)
                    gradient_i=GradientPhi[idOfPhi_i_WithThisTriangle](epsilon_t,i)
                    gradient_j=GradientPhi[idOfPhi_j_WithThisTriangle](epsilon_t,j)
                    sum+=(np.dot(gradient_i,gradient_j)+alpha(epsilon_t)*Phi[idOfPhi_i_WithThisTriangle](epsilon_t,i)*Phi[idOfPhi_j_WithThisTriangle](epsilon_t,j))*0.5*W[t]
                resultWithThisTriangle = 2 * area * sum
                A[i][j]+=resultWithThisTriangle 
            print("thoi gian tinh A[{0}][{1}]".format(i,j), "la:",str(timer()-start))
            input()

print("Véc tơ vế phải là: ",VectoVePhai)
print("Ma trận A là:")
#print(A)
#U=np.dot(np.linalg.inv(A),VectoVePhai)
#print("Véc tơ nghiệm u là: ",U)

