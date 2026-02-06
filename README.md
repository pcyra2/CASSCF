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

## Installation

git clone https://github.com/pcyra2/CASSCF.git

cd CASSCF
conda create -n CASSCF
conda activate CASSCF
conda install python==3.13
pip install -e . 

## Running

Run in a directory with the molecule.xyz in. 
Preferably if it also has the MP2_natural_orbitals.json this will read in the MP2 natural orbitals from a previous calculation.

To run: 
```CASSCF -c=NET_CHARGE -s=SPIN -ci=MAX_CASCI[1] -scf=MAX_SCF[2] -m=MAX_MEMORY[3] -size=MAX_HAMILTONIAN_GENERATED[4] ```

[1] - The maximum size of CASCI that is actually executed (not the maximum hamiltonian size)
[2] - The maximum size of CASSCF that is actually executed 
[3] - The maximum ammount of RAM pySCF can use in MB (i.e. 1000 = 1GB) WARNING: This should be less than the total ammount of RAM available! (I go with 60% available RAM but play around with this...)
[4] - The maximum hamiltonian size to be generated

There are also extra flags to toggle extra functionality:
-CCSD (performs CCSD)
-noRDM (Doesnt generate the RDM files associated with CCSD)
-nevpt (performs NEVPT2 on the CASCI calculations)
-nelec=NUMBER_ACTIVE_ELECTRONS (Only used in the chemically informed/manually selected active space environment)

For Chemically informed active space calculations, Run the code in the following procedure:

```CASSCF -c=NET_CHARGE -s=SPIN -m=MAX_MEMORY -size=0 -CCSD ```

This will take a while but should generate the CCSD data, RDM's, and MP2 Natural orbitals. 

You can then visualise the "MP2_natural_orbitals.molden" using JMol to decide the chemically interesting orbitals. 
Both JMol and my orbital selection are 1 indexed (Not pythonic 0 indexing so take care and check orbital occupations!)

Once orbitals have been selected, run the following command (say you want to simulate orbitals 5 and 6 in water):

```CASSCF -c=NET_CHARGE -s=SPIN -m=MAX_MEMORY -size=0 -nelec=2 -nevpt  5 6```

The output/hamiltonian naming convention is JOBTYPE_NELEC-NORB, if you want multiple hamiltonians with matching NELEC-NORB, you will have to implement this mannually... Sorry! You could always just run in seperate directories!

You can find the output of this calculation in outputs/CASCI_2-2.out

A summary of the useful information should also be in summary.json



## Orca compatability

This is designed to re-produce CASCI calculations performed in orca with unrelaxed rHF MP2 Natural orbitals, that were generated with the frozen core approximation.