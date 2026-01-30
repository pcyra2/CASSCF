# Nottingham  CASSCF Code

## Requirements

- python==3.13
- pyscf
- qiskit
- qiskit-nature
- openfermion
- numpy
- libmsym
- pyberny
- ase

## Running

Run in a directory with the molecule.xyz in. 
Preferably if it also has the MP2_natural_orbitals.json this will read in the MP2 natural orbitals from a previous calculation.

To run: 
CASSCF -c=NET_CHARGE -s=SPIN -ci=MAX_CASCI[1] -scf=MAX_SCF[2] -m=MAX_MEMORY -size=MAX_HAMILTONIAN_GENERATED[3] 

[1] - The maximum size of CASCI that is actually executed (not the maximum hamiltonian size)
[2] - The maximum size of CASSCF that is actually executed 
[3] - The maximum hamiltonian size to be generated

There are also extra flags to toggle extra functionality:
-CCSD (performs CCSD)
-noRDM (Doesnt generate the RDM files associated with CCSD)
-nevpt (performs NEVPT2 on the CASCI calculations)




## Orca compatability

This is designed to re-produce CASCI calculations performed in orca with unrelaxed rHF MP2 Natural orbitals, that were generated with the frozen core approximation.