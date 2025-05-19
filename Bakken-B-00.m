Name = Bakken-B-00

DomainRx = [0.1 , 2.0]
DomainRy = [0.1 , 2.0]
DomainTheta = [0 , 0 ]
DomainEcc = [0 , 0]
DomainEccAngle = [0 , 0]

LDomain_in_LSH = yes

DomainType = fluid,  HTTI 

AddDomainLoc = ext
AddDomainType = abc
AddDomainL = 1.

PML_factor = 10 
PML_degree = 2.0 
PML_method = 2.

ABC_factor = 0.1 
ABC_degree = 1.0
ABC_account_r = yes 

DomainParam = ( [1.0, 2.25],  [ 2.23 , 40.9 , 8.5 , 26.9 , 10.5 , 15.3 , 0 , 0] )

RefDomainType = HTTI
RefDomainParam = same

BCType = FS, rigid

f_array = [0.5:0.25:15]

DomainNth = [12 , 12] 

mud_domain = 1

Advanced.num_eig_max = 50 

Advanced.EigSearchStart = 1.0 

Mesh.hmax = 0.05    
Mesh.dhmax = 0.25   
Mesh.output = no  
Mesh.ext_boundary_shape = cir 
