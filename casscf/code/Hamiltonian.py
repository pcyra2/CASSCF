import scipy.sparse
import scipy.sparse.linalg
import tqdm as tqdm

import numpy
import os
import pyscf
from pyscf import ao2mo
import scipy
from sympy import *
from sympy.physics.quantum import TensorProduct
from scipy import sparse
import re   
import openfermion
import time
from openfermion.ops import FermionOperator
from openfermion import get_sparse_operator, jw_get_ground_state_at_particle_number
from qiskit_nature.second_q.operators import  PolynomialTensor
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
import qiskit_nature.second_q.operators.tensor_ordering as order

from casscf.code.helpers.extract_integrals import construct_spin_one_body, construct_spin_two_body, build_molecular_integrals_uhf
from casscf.code.helpers.extract_integrals import extract_integral_tensors, build_molecular_integrals, construct_spin_one_body, construct_spin_two_body
from casscf.code.helpers.storage_conversion import dense_to_sparse
import casscf.code.io as io
import casscf.code.pyscf_tools as pyscf_tools

def getInts(Mol: pyscf.M , HF: pyscf.scf.uhf.UHF|pyscf.scf.rhf.RHF, BASISSET:str, BASISTYPE: str, UHF:bool, path:str, MOLECULE:str, nFrozen:int)->(dict,dict):
    """Obtains the one and two electron integrals, then stores them as dictionaries in a pipecraft format. Also generates a stage_data.json file
    Stores them in the path: IntegralPath/BASIS/points[index]/integrals.json

    Args:
        Mol (pyscf.M): pySCF Molecule description
        HF (pyscf.scf.UHF|pyscf.scf.RHF): pySCF Meanfield calc (Usually UHF)
        BASISSET (str): The basis set used
        BASISTYPE (str): atomic or molecular basis
        UHF (bool): If UHF, set as True, else if RHF, set as False
        path (str): Path to save the integrals (usually set as WorkDir/Integrals)
        MOLECULE (str): "Name" of molecule. (For pipecraft)
        nFrozen (int): Whether to freeze the core or not. (Defaults to False

    Returns:
        one_body (dict): one electron integrals in a dictionary.
        two_body (dict): two electron integrals in a dictionary.
    """
    if UHF == True and BASISTYPE == "molecular":
        BASISSET = BASISSET.replace("**","dp").replace("*","p")
        MOLECULE = MOLECULE.replace(".xyz","").replace(".","").split("/")
        MOLECULE = MOLECULE[-1]
        if path != None:
            try:
                os.mkdir(f"{path}/integrals")
            except FileExistsError:
                pass
            try:
                os.mkdir(f"{path}/integrals/{MOLECULE}")
            except FileExistsError:
                pass
            try:
                os.mkdir(f"{path}./integrals/{MOLECULE}/{BASISSET}")
            except FileExistsError:
                pass
            try:
                os.mkdir(f"{path}./integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}")
            except FileExistsError:
                pass
        
        one_body, two_body, constant_energy, ints= build_molecular_integrals_uhf(HF, 0)
        mol = HF.mol
        n_modes = len(one_body)

        n_electrons = mol.tot_electrons()
        result = {
            "constant_energy": constant_energy,
            "one_body": dense_to_sparse(one_body),
            "two_body": dense_to_sparse(two_body),
            "n_electrons": n_electrons,
            "n_modes": n_modes,
            "molecule_name": MOLECULE,
            "molecule_basis": BASISSET,
            "molecule_uhf": UHF,
        }
        if path != None:
            io.jsonDump(result, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/integrals.json")
            stage_data = {"stage":"molecule_generation"}
            io.jsonDump(stage_data, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/stage_data.json")
        if nFrozen > 0:
            print("Performing Frozen Core integral generation")
            if path != None:
                try:
                    os.mkdir(f"{path}./integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE")
                except FileExistsError:
                    pass
            one_body_FC, two_body_FC, constant_energy, ints = build_molecular_integrals_uhf(HF, nFrozen)
            mol = HF.mol
            n_modes = len(one_body_FC)

            n_electrons = mol.tot_electrons() - 2 * nFrozen
            result_fc = {
                "constant_energy": constant_energy,
                "one_body": dense_to_sparse(one_body_FC),
                "two_body": dense_to_sparse(two_body_FC),
                "n_electrons": n_electrons,
                "n_modes": n_modes,
                "molecule_name": MOLECULE,
                "molecule_basis": BASISSET,
                "molecule_uhf": UHF,
                "frozen_orbitals": nFrozen,
            }
            if path != None:
                io.jsonDump(result_fc, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE/integrals.json")
                stage_data = {"stage":"molecule_generation"}
                io.jsonDump(stage_data, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE/stage_data.json")
        else:
            result_fc = None
        return result, result_fc
    else:
        raise NotImplementedError("Only UHF in the molecular basis is supported at the moment")
    # MOLECULE = MOLECULE.replace(".xyz","").replace(".","").split("/")
    # MOLECULE = MOLECULE[-1]
    # one_body, two_body, integrals = extract_integral_tensors(Mol, HF, BASISTYPE, UHF)
    # n_electrons = Mol.tot_electrons()
    # n_modes = len(one_body)
    # constant_energy = Mol.energy_nuc()
    # result = {
    #             "constant_energy": constant_energy,
    #             "one_body": dense_to_sparse(one_body),
    #             "two_body": dense_to_sparse(two_body),
    #             "n_electrons": n_electrons,
    #             "n_modes": n_modes,
    #             "molecule_name": MOLECULE,
    #             "molecule_basis": BASISTYPE,
    #             "molecule_uhf": UHF,
    #         }
    
    
    # if nFrozen > 0:
    #     pass
    # if nFrozen > 0:
    #     try:
    #         os.mkdir(f"{path}./integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE")
    #     except FileExistsError:
    #         pass
    #     io.jsonDump(result, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE/integrals.json")
    #     stage_data = {"stage":"molecule_generation"}
    #     io.jsonDump(stage_data, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/FROZEN_CORE/stage_data.json")
    # else:
    #     io.jsonDump(result, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/integrals.json")
    #     stage_data = {"stage":"molecule_generation"}
    #     io.jsonDump(stage_data, f"{path}/integrals/{MOLECULE}/{BASISSET}/{BASISTYPE}/stage_data.json")
    return one_body, two_body, result, integrals

def getInts_raw(mol: pyscf.M, mf:pyscf.scf.uhf.UHF|pyscf.scf.rhf.RHF, UHF:bool, frozen=0):
    one, two, nuc, integrals = build_molecular_integrals_uhf(mf, frozen, mol)
    n_electrons = mol.tot_electrons() - 2 * frozen
    print(f"{n_electrons=}")
    data = {
                "constant_energy": nuc,
                "one_body": dense_to_sparse(one),
                "two_body": dense_to_sparse(two),
                "n_electrons": n_electrons,
                "n_modes": len(one),
                "molecule_uhf": UHF,
            }
    print(len(integrals["molecular"]["one_body"]))
    # if thresh > 0:

    return integrals, data

def T(*args):
    """Tensor Product
    """
    if len(args) == 1:
        return args[0]
    elif len(args) == 0:
        return sparse.csr_matrix([[1]])
    else:
        tp = TensorProduct(args[0], T(*args[1:]))  
        return tp
    
I = numpy.array([[1, 0], [0, 1]])
X = numpy.array([[0, 1], [1, 0]])
Y = numpy.array([[0, 0 - 1j], [0 + 1j, 0]])
Z = numpy.array([[1, 0], [0, -1]])

pauli_map = {
    "I": sparse.csr_matrix(I),
    "X": sparse.csr_matrix(X),
    "Y": sparse.csr_matrix(Y),
    "Z": sparse.csr_matrix(Z),
}

def pauli_to_matrix(pauli: str, nqubits: int) -> numpy.array:
    """Used to convert the pauli hamiltonian into a matrix. Niam wrote this  :) 
    This code takes ages... so the code stores them once generated.

    Args:
        pauli (str): The pauli string
        nqubits (int): The number of cubits

    Returns:
        numpy.array: The matrix of the given pauli
    """
    pauli_ops = [pauli_map[x] for x in re.split(r"\d", pauli)[:-1] if x != ""] 
    qubits = [int(x) for x in re.split(r"\D", pauli)[1:]]
    parsed_pauli = dict(zip(qubits, pauli_ops))
    for i in range(nqubits):
        if i not in parsed_pauli:
            parsed_pauli[i] = pauli_map["I"]
    full_pauli_op = [parsed_pauli[i] for i in range(nqubits)]
    return T(*full_pauli_op)

def qubitised_to_matrix(Hamiltonian: dict, nqubits: int) -> sparse.csr_matrix:
    """Takes a pauli hamiltonian(dict) and converts to a matrix. Also does some data management to speed calculations up

    Args:
        Hamiltonian (dict): Dictionary of hamiltonian with form ham[key]=coeff
        nqubits (int): Number of qubits

    Returns:
        sparse.csr_matrix: the Matrix hamiltonian with shape [2**nqubits, 2**nqubits]
    """
    start = time.perf_counter()
    matrix = sparse.csr_matrix(numpy.zeros(shape=(2**nqubits, 2**nqubits), dtype=numpy.complex128))
    for i in Hamiltonian.keys():
        coeff = Hamiltonian[i]
        # print(i, coeff)
        # # for x in re.split(r"\d", i):
        # #     if x != "":
        # #         print(x)
        # pauli_ops = [pauli_map[x] for x in re.split(r"\d", i)[:-1] if x != ""]
        # print(pauli_ops)
        try:
            relPath = os.path.normpath(os.path.dirname(__file__)+f"/../Parameters/QubitBlanks/{nqubits}/{i}.npz")
            mat = scipy.sparse.load_npz(relPath)
        except FileNotFoundError:
            tmp = pauli_to_matrix(i, nqubits)
            mat = sparse.csr_matrix(tmp.toarray())
            scipy.sparse.save_npz( relPath, mat)
        matrix += mat * coeff
    stop = time.perf_counter()
    print(f"INFO: Time taken to convert hamiltonian is {stop - start}")
    return matrix

def diagonalise_matrix(matrix: sparse.csr_matrix, nelec: int)->(float, list):
    """Performs ED on the matrix hamiltonian. Uses openfermion as a backend

    Args:
        matrix (sparse.csr_matrix): Hamiltonian in matrix form
        nelec (int, optional): Used to find the segment of the matrix to diagonalise. (Defaults to 5)

    Returns:
        val (float):  Energy of the system (Hatree)
        Vec (list): The ground state of the system.
    """
    start = time.perf_counter()
    #val, vec = scipy.sparse.linalg.eigs(matrix, k=6, )
    # val, vec = openfermion.linalg.get_ground_state(matrix)
    qubits = int(numpy.log(matrix.shape[0])/numpy.log(2)) # gets the number of qubits from the matrix size. as matrix = 2^qubits x 2^qubits
    print(f"qubits: {qubits}")
    indices = [i for i in range(int(2**qubits)) if sum([int(j) for j in bin(i)[2:]]) == int(nelec)] # Gets the correct sector of the matrix. 
    newMat = numpy.zeros([len(indices),len(indices)], dtype=complex)
    for i in range(len(indices)):
        for j in range(len(indices)):
            newMat[i,j] = matrix[indices[i], indices[j]]
    # newMat = matrix[indices,indices]
    print("Start")
    # try:
    val, vec = openfermion.linalg.get_ground_state(newMat)
    # except ValueError:
    #     val, vec = openfermion.linalg.get_ground_state(matrix)
    print("Stop")
    newvec = numpy.zeros(2**qubits, dtype=complex)
    for i, index in enumerate(indices):
        newvec[index] = vec[i]
    vec = newvec
    stop = time.perf_counter()
    print(f"INFO: ED took {stop - start} seconds")
    return val, vec

def pauli_to_openfermion(hamiltonian:dict, const: float ,nqubits: int)->dict:
    terms_dict={"": const, **hamiltonian}
    ham_dict = {
        "terms": terms_dict,
        "num_qubits": nqubits,
        "num_terms": len(terms_dict),
    }
    return ham_dict

def integrals_to_openfermion(Data: dict):
    nelec = int(Data["n_electrons"])
    constant_energy = float(Data["constant_energy"])
    one_body=Data["one_body"]
    two_body=Data["two_body"]
    openfermion_hamiltonian = FermionOperator()
    for index, coefficient in one_body.items():
        index = index.split()
        i = str(index[0].replace(",","").replace("(", "")).replace("np.int64","").replace(")","")
        j = str(index[1].replace(")","")).replace("np.int64(","")
        # print(f"{i=} {j=}")
        # (i, j) = index
        openfermion_hamiltonian += complex(coefficient) * FermionOperator(
            str(i) + "^ " + str(j)
        )
    if two_body != None:
        for index, coefficient in two_body.items():
            index = index.split()
            i = str(index[0].replace(",","").replace("(", "")).replace("np.int64","").replace(")","")
            j = str(index[1].replace(",","")).replace("np.int64","").replace(")","").replace("(", "")
            k = str(index[2].replace(",","")).replace("np.int64","").replace(")","").replace("(", "")
            l = str(index[3].replace(")","")).replace("np.int64","").replace("(", "")
            # (i, j, k, l) = index
            openfermion_hamiltonian += complex(coefficient) * FermionOperator(
                str(i) + "^ " + str(j) + "^ " + str(k) + " " + str(l)
            )
    openfermion_hamiltonian += constant_energy * FermionOperator("")
    return openfermion_hamiltonian, nelec, constant_energy

def integrals_to_qiskit(Data: dict, integrals: dict):
    # a = {"+-": integrals["molecular"]["alpha"]}
    # alpha = PolynomialTensor({"+-": integrals["molecular"]["alpha"], 
    #                           "++--": integrals["molecular"]["alpha_alpha"]})
    # beta = PolynomialTensor({"+-": integrals["molecular"]["beta"],
    #                          "++--": integrals["molecular"]["beta_beta"]})
    # alpha_beta = PolynomialTensor({"++--": integrals["molecular"]["alpha_beta"]})
    # print(alpha)
    # print(beta)
    # qiskit_integrals = ElectronicIntegrals(alpha, beta, alpha_beta)
    print(order.find_index_order(integrals["molecular"]["alpha_beta"]))
    qiskit_ham = ElectronicEnergy.from_raw_integrals(h1_a=integrals["molecular"]["alpha"],
                                                              h2_aa=order.to_physicist_ordering(integrals["molecular"]["alpha_alpha"], index_order=order.IndexType("chemist")),
                                                              h1_b=integrals["molecular"]["beta"],
                                                              h2_bb=order.to_physicist_ordering(integrals["molecular"]["beta_beta"], index_order= order.IndexType("chemist")),
                                                              h2_ba=order.to_physicist_ordering(integrals["molecular"]["alpha_beta"], index_order=order.IndexType("chemist")),
                                                              auto_index_order=False)
    qiskit_ham.nuclear_repulsion_energy = Data["constant_energy"]
    # problem = ElectronicStructureProblem(qiskit_ham)
    # problem.num_particles = (int(nelec/2), int(nelec/2))
    return qiskit_ham

def qiskit_to_integrals(integrals:ElectronicEnergy):
    # pprint(vars(integrals))
    mode_count = len(integrals.alpha["+-"][0])
    one_body_block = [integrals.alpha["+-"], integrals.beta["+-"]]
    print(order.find_index_order(integrals.alpha["++--"]))
    two_body_block = [order.to_chemist_ordering(integrals.alpha["++--"], index_order= order.IndexType("physicist")), 
                      order.to_chemist_ordering(integrals.beta_alpha["++--"], index_order= order.IndexType("physicist")),
                      order.to_chemist_ordering(integrals.alpha_beta["++--"],index_order= order.IndexType("physicist")),
                      order.to_chemist_ordering(integrals.beta["++--"],index_order= order.IndexType("physicist"))]
    # two_body_block = [integrals.alpha["++--"], 
    #                 integrals.beta_alpha["++--"],
    #                 integrals.beta_alpha["++--"],
    #                 integrals.beta["++--"],]
    one_body = construct_spin_one_body(one_body_block, mode_count)
    two_body = construct_spin_two_body(two_body_block, mode_count)
    return one_body, two_body

def diagonalise_integrals(Data,
    # one_body,
    # two_body,
    ) -> float:
    """
    Compute exact energy of a fermionic hamiltonian using an eigendecomposition

    Args:
        Data (dict): Core system data (nelec, nuclear energy, one and two electron integrals)

    Returns:
        ED_energy (float): The minimum energy of the system
        gs (list): the state of the ground state 
    """
    openfermion_hamiltonian, nelec, constant_energy = integrals_to_openfermion(Data)
   
    try:
        ED_energy, gs = diagonalise_OpenFermion(openfermion_hamiltonian, nelec)
    except IndexError:
        ED_energy = "matrix too small for sparse eigensolver"
        gs = None

    return ED_energy, gs

def diagonalise_OpenFermion(fermionic_hamiltonian: FermionOperator,
    nelec: int,
    )->(float, list):
    """Compute exact energy of a fermionic hamiltonian using openfermion

    Args:
        fermionic_hamiltonian (FermionOperator): hamiltonian as an openfermion hamiltonian
        nelec (int): number of electrons
    Returns:
        gs_energy (float): The minimum energy of the system
        gs (list): the state of the ground state
    """

    sparse_fermionic_hamiltonian = get_sparse_operator(fermionic_hamiltonian)
    gs_energy, gs = jw_get_ground_state_at_particle_number(
        sparse_fermionic_hamiltonian, nelec
    )
    return gs_energy, gs

def ikjl_to_ijkl(two_body):
    """
    Function to reorder the two body integrals from the ikjl format to the ijkl format. Qiskit chemists notation is ikjl which is why it is needed. 

    Args:
        two_body (dict): ikjl ordered two-body integrals

    Returns:
        reordered_two_body (dict): ijkl ordered two-body integrals
    """
    key_list = list(two_body.keys())
    reordered_two_body = {}
    for key in key_list:
        key_tmp = key.replace("(", "").replace(")", "").split(",")
        new_key = f"({key_tmp[0]}, {key_tmp[2]}, {key_tmp[1]}, {key_tmp[3]})"
        reordered_two_body[new_key] = two_body[key]
    return reordered_two_body

def Threshold_Integrals(Data: dict, Threshold: float, type="absolute"):
    if "one_body" in Data:
        New_integrals = {}
        New_integrals["one_body"] = {}
        New_integrals["two_body"] = {}
        for key in Data["one_body"].keys():
            if numpy.absolute(Data["one_body"][key]) > Threshold:
                New_integrals["one_body"][key] = Data["one_body"][key]
        for key in Data["two_body"].keys():
            if numpy.absolute(Data["two_body"][key]) > Threshold:
                New_integrals["two_body"][key] = Data["two_body"][key]
        Data["one_body"] = New_integrals["one_body"]
        Data["two_body"] = New_integrals["two_body"]
    elif "molecular" in Data:
        for key in Data["molecular"].keys():
            Data["molecular"][key] = numpy.where(numpy.abs(Data["molecular"][key])> Threshold, Data["molecular"][key], 0)
    else:
        raise Exception("I dont know how to threshold this data. Sorry. Either give me a stage_data dictionary, or a raw integrals dictionary. ")
    return Data

def parse_orca1e(data:dict):
    """Converts the 1 electron integrals obtained by orca into the dictionary format that is provided to phasecraft. 

    Args:
        data (dict): orca 1 electron integral dictionary (data = orca.json["Molecule"]["HMO"])

    Returns:
        new_data (dict): Dictionary containing integrals and their keys. indices are in tuple format.
    """
    norbs = len(data[0])
    new_data = {}
    for i in range(norbs):
        for j in range(norbs):
            new_data[(i, j)] = float(data[0][i][j])
            new_data[(i+norbs, j+norbs)] = float(data[0][i][j])
    return new_data

def parse_orca2e(data:dict, norbs: int):
    """Converts the 2 electron integrals obtained by orca into the dictionary format that is provided to phasecraft. 

    Args:
        data (dict): orca 2 electron integral dictionary (data = orca.json["Molecule"]["2elIntegrals"])
        norbs (int): Number of spatial orbitals (nqubits/2)

    Returns:
        new_data (dict): Dictionary containing integrals and their keys. indices are in tuple format in physisist's notation. 
    """
    new_data = {}
    locs = ["MO_PQRS", "MO_PRQS"]
    blocks = ["alpha", "beta",]
    for loc in locs:
        for block1 in blocks:
            for block2 in blocks:
                if f"{block1}/{block2}" in data[loc].keys():
                    blockstr = f"{block1}/{block2}"
                else:
                    blockstr = f"{block2}/{block1}"
                if blockstr not in data[loc]:
                    break
                for a, b, c, d, val in data[loc][blockstr]:
                    if loc == "MO_PQRS":
                        i = a
                        k = b
                        j = c
                        l = d
                    elif loc == "MO_PRQS":
                        i = a 
                        j = b 
                        k = c 
                        l = d 
                    if block1 == "beta":
                        i += norbs
                        k += norbs
                    if block2 == "beta": 
                        j += norbs
                        l += norbs
                    new_val = val * -0.5 # Converts the value to be the same as Phasecrafts. (see construct 2 body blocks in extract integrals. )
                    if (i,j,k,l) in new_data.keys():
                        assert numpy.isclose(new_data[(i,j,k,l)], new_val), f"{(i,j,k,l)} has gone wrong... {new_data[(i,j,k,l)]} != {new_val}"
                    else:
                        keys =[(i,j,k,l), (k,j,i,l), (i,l,k,j), (k,l,i,j),
                                   (j,i,l,k), (j,k,l,i), (l,i,j,k), (l,k,j,i)] # implements permutation symmetry
                        for key in keys:
                            if key in new_data.keys():
                                if numpy.isclose(new_data[key], new_val) == False:
                                    print(f"data {key} exists but is different: old={new_data[key]}, new={val * -0.5}")
                            else:    
                                new_data[key] = float(new_val)
    return new_data

def CAS_to_Hamiltonian(CASSCF, mo_list, orbs, active_orbs=None, active_elecs = None):
    if active_elecs is None and active_orbs is not None:
        active_elecs = active_orbs
    elif active_orbs is None and active_elecs is not None: 
        active_orbs = active_elecs
    elif active_orbs is not None and active_elecs is not None:
        pass
    else:
        raise ValueError("Active orbitals and active electrons are not set.")
    orbitals = CASSCF.sort_mo(mo_list, orbs)
    one_body_cas, constant_energy = CASSCF.get_h1cas(mo_coeff=orbitals)
    two_body_cas = CASSCF.get_h2eff(mo_coeff=orbitals)
    h1e = construct_spin_one_body([one_body_cas]*2,active_orbs)
    h2e = construct_spin_two_body([ao2mo.restore(1,two_body_cas,active_orbs)]*4,active_orbs)
    h2e = numpy.swapaxes(h2e,1,2)
    try:
        e_tot = CASSCF.e_tot
    except:
        e_tot = None
    result = {
            "pySCF_energy": e_tot,
            "constant_energy": constant_energy,
            "one_body": dense_to_sparse(h1e),
            "two_body": dense_to_sparse(h2e),
            "n_electrons": active_elecs,
            "n_modes": active_orbs*2,
            "molecule_name": 'Inhibitor',
            "molecule_basis": 'sto3g',
            "molecule_uhf": False,
        }
    if active_orbs < 10:
        ED,_ = diagonalise_integrals(result)
        result["exact_diagonalisation"] = ED
    return result