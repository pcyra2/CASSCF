import json
from ase import build
from pyscf import gto, dft
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf


def construct_pyscf_molecule(path_to_spin: str, molecule: str):
    with open(path_to_spin + "/hund_spins.txt") as f:
        g2_spins = json.load(f)

    assert molecule in list(g2_spins.keys())

    atoms = build.molecule(molecule)
    spin = g2_spins[molecule]

    # spins obtains from dummy gpaw calcualtion to generate spin from hunds rule (need to see if PySCF has such a feature)

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = "out_" + molecule
    mol.atom = ase_atoms_to_pyscf(atoms)
    mol.basis = "sto6g"  # this is a small basis set we would wan tot explore the use of larger basis sets
    mol.spin = spin
    # mol.symmetry = True
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = "b3lyp"  # this a old funcitonal, but is a hybrid,
    if len(atoms) > 1:
        mf = mf.newton()
    mf.kernel()

    # geometric
    if len(atoms) > 1:
        from pyscf.geomopt.geometric_solver import optimize

        mol_eq = optimize(mf, maxsteps=100)
        print(mol_eq.atom_coords())
        mol = mol_eq

    return mol
