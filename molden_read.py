import pyscf
from pprint import pprint

import Code.pyscf_tools as pyscfTools

def main():
    molden = pyscf.tools.molden.load("MP2_natural_orbitals.molden")
    
    molecule = molden[0]
    molecule.build()
    molecule.verbose=4
    molecule.charge=-1
    molecule.spin=0
    pprint(vars(molecule))

    cas = molecule.CASCI(2,2).kernel()
    
    pprint(vars(cas))


if __name__ == "__main__":
    main()
