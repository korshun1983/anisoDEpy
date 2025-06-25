class AdvancedParams:
    def __init__(self):
        self.EigSearchStart = 1.0
        self.num_eig_max = 50

class Mesh:
    def __init__(self):
        self.dhmax = 0.25
        self.ext_boundary_shape = 'cir'
        self.hmax = 0.05
        self.output = 'no'

class InputParams:
    def __init__(self):
        self.ABC_account_r = 'yes'
        self.ABC_degree = 1.0
        self.ABC_factor = 0.1
        self.AddDomainL = 1.0
        self.AddDomainLoc = 'ext'
        self.AddDomainType = 'abc'
        self.BCtype = ['FS','rigid']
        self.DomainEcc = [0, 0]
        self.DomainNth = [12, 12]
        self.DomainParam = ([1.0, 2.25],[2.23, 40.9, 8.5, 26.9, 10.5, 15.3, 0, 0])
        self.DomainRx = [0.1, 2.0]
        self.DomainRy = [0.1, 2.0]
        self.DomainTheta = [0, 0]
        self.DomainType = ['fluid', 'HTTI']
        self.LDomain_in_LSH = 'yes'
        self.ModelName = ''
        self.PML_degree = 2.0
        self.PML_factor = 10.0
        self.PML_method = 2.0
        self.RefDomainParam = ([1.0, 2.25],[2.23, 40.9, 8.5, 26.9, 10.5, 15.3, 0, 0])
        self.RefDomainType = 'HTTI'
        self.Advanced = AdvancedParams()
        self.f_array = []
        self.Mesh = Mesh()