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

Execute the CASSCF.py script from the work-directory with the .xyz file. Make sure to change the system_info dictionary at the top of the script. Importantly, nFrozen, charge, and spin.

## Orca compatability

This is designed to re-produce CASCI calculations performed in orca with unrelaxed rHF MP2 Natural orbitals, that were generated with the frozen core approximation.