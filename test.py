import emlp
from emlp.reps import V,T,Rep
from emlp.groups import Z,S,SO,Group
import numpy as np
from emlp.reps.linear_operators import LazyKron

class ProductSubRep(Rep):
    def __init__(self,G,subgroup_id,size):
        """   Produces the representation of the subgroup of G = G1 x G2
              with the index subgroup_id in {0,1} specifying G1 or G2.
              Also requires specifying the size of the representation given by G1.d or G2.d """
        self.G = G
        self.index = subgroup_id
        self._size = size
    def __str__(self):
        return "V_"+str(self.G).split('x')[self.index]
    def size(self):
        return self._size
    def rho(self,M): 
        # Given that M is a LazyKron object, we can just get the argument
        return M.Ms[self.index]
    def drho(self,A):
        return A.Ms[self.index]
    def __call__(self,G):
        # adding this will probably not be necessary in a future release,
        # necessary now because rep is __call__ed in nn.EMLP constructor
        assert self.G==G
        return self
    

G1,G2 = SO(3),S(8)
G = G1 * G2

VSO3 = ProductSubRep(G,0,G1.d)
VS8 = ProductSubRep(G,1,G2.d)

Vin = VSO3 + V(G)
Vout = VS8
print(f"Vin: {Vin} of size {Vin.size()}")
print(f"Vout: {Vout} of size {Vout.size()}")
# make model
model = emlp.nn.EMLP(Vin, Vout, group=G)
input_point = np.random.randn(Vin.size())*10
print("input_point:",input_point)
print(f"Output shape: {model(input_point).shape}")

# Test the equivariance
def rel_err(a,b):
    return np.sqrt(((a-b)**2).sum())/(np.sqrt((a**2).sum())+np.sqrt((b**2).sum()))

lazy_G_sample = LazyKron([G1.sample(),G2.sample()])


out1 = model(Vin.rho(lazy_G_sample)@input_point)


out2 = Vout.rho(lazy_G_sample)@model(input_point)
print("out1 = ",out1)
print("out2 = ",out2)
print(f"Equivariance Error: {rel_err(out1,out2)}")