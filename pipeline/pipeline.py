LOAD, RAND = 1,0
TEST = 0
CSLOOP = (not TEST) and 0
MLOOP = (not CSLOOP)

class SymNorm():
    def __init__(self, m,n,w, LOAD=True,TEST =True):
        self.c = 2**int(np.log(w*10))

    if TEST:
        mList ,cList, rList= [100], [10],[2]
        suffix = 'test'
        path = TESTSET
        MLOOP = 1
        cr = np.log2(cList[0]*rList[0])
        colName = ['n','m','w','sRate','c','r', 'cr', 'ex', 'un','cs','errCs','errUn']
    else:
        path = PCKSET
        if CSLOOP:
            mList=[10000]
            cList = [2**6, 2**7, 2**8, 2**9, 2**10]
            rList = [2**2]
            suffix = f'_csL_m{mList[-1]}_'
            colName = ['n','m','w','c','r','cr', 'ex','cs','errCs']
        elif MLOOP:
            mList = [100,300]
            # mList = [100, 500, 1000, 5000, 10000, 50000]
            # mList = [5000, 10000, 50000, 100000, 500000]        
            cList =[2**10]
            # rList = [2**3]
            rList = [None]
            if rList[0] == None:
                suffix = f'_mL_c{cList[0]}_rNone'
            else:
                cr = np.log2(cList[0]*rList[0])
                suffix = f'_mL_t{cr}_'
            colName = ['n','m','w','sRate','c','r', 'cr', 'ex', 'un','cs','errCs','errUn']
        else:
            pass