import json
import numpy
import os
import h5py
import casscf.code.pyscf_tools as pyscfTools
import pyscf

def jsonDump(dict:dict, path:str,):
    """
    The function `jsonDump` takes a dictionary and a file path as input, then writes the dictionary to a
    JSON file at the specified path.
    
    Args:
        dict (dict): A dictionary containing the data that you want to dump into a JSON file.
        path (str): The `path` parameter in the `jsonDump` function is a string that represents the file
    path where the JSON data will be dumped or saved. It should be the location where you want to store
    the JSON data.
    """
    try:
        with open(path, "w") as f:
            json.dump(dict, f, indent="\t", sort_keys=True, )
    except TypeError:
        with open(path, "w") as f:
            json.dump(dict, f, indent="\t",)

def jsonRead(path:str)->dict:
    """
    The function `jsonRead` reads and returns the content of a JSON file located at the specified path.
    
    Args:
        path (str): The `path` parameter in the `jsonRead` function is a string that represents the file
    path to the JSON file that you want to read and load.
    
    Returns:
        text (dict): The function `jsonRead` reads a JSON file located at the specified `path` and returns the contents
    of the file as a Python dictionary.
    """
    with open(path, "r") as f:
        text = json.load(f, object_hook=parse_float_keys)
        f.close()
    return text

def textDump(text, path:str):
    """Prints a text file, 

    Args:
        text: accepts either array of text or string.
        path (str): file path to write to. 
    """
    with open(path, "w")as f:
        if type(text) == str:
            print(text, file=f)
        else:
            for i in text:
                print(i, file=f)
        f.close()

def textRead(path:str)->list:
    """Reads in a text file as a list of lines.

    Args:
        path (str): path to file.
    Returns:
        text (list): list of lines in the file.
    """
    with open(path, "r") as f:
        text = f.readlines()
        f.close()
    clean = [line.replace("\n","") for line in text]
    return clean

def MakeDir(path: str):
    """Makes directory in path if it doesn't already exist

    Args:
        path (str): Path to directory. 
    """
    print(f"INFO: Generating {path}")
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"INFO: {path} already exists")
        pass
    
# def StateRead(path: str) -> list:
#     """Reads in a state vector from a text file (Basically converts it to complex)

#     Args:
#         path (str): path to file

#     Returns:
#         state (list): Complex state vector as list
#     """
#     text = textRead(path)
#     state = numpy.zeros(len(text), dtype=complex)
#     for i in range(len(text)):
#         state[i] = complex(text[i])
#     return state



def parse_float_keys(dct)->float|str:
    rval = dict()
    for key, val in dct.items():
        try:
            # Convert the key to an integer
            int_key = float(key)
            # Assign value to the integer key in the new dict
            rval[int_key] = val
        except ValueError:
            # Couldn't convert key to an integer; Use original key
            rval[key] = val
    return rval

def TrajIn(path:str, molFile:str, charge: int, spin:int, basis: str, symmetry:bool)-> list:
    """Reads in a trajectory file of .xyz coordinates. 

    Args:
        path (str): path to file
        charge (int): Net charge of system
        spin (int): number of unpaired electrons
        basis (str): basis set to use
        
    Returns:
        list (str): List of Molecules
    """
    text = textRead(path+molFile)
    state = numpy.zeros(len(text)) # initiate a line type array. number is number of items per line
    starts = [0]
    OrigNat = int(text[0])
    for line in range(1,len(text)):
        state[line] = len(text[line].split())
        if state[line] == 1:
            try:
                nat = int(text[line])
                assert nat == OrigNat, "Number of atoms seems to change in the trajectory file at line "+str(line)
                starts.append(line)
            except ValueError:
                if line -1 in starts: # Then this is a comment line. so allowed to be a single item. 
                    pass
                else:
                    raise Exception("There seems to be incorrect formatting on line " + str(line)+", There is only one item in the line, therefore expecting NAT, but instead got a string...")
    Molecules = [None]*len(starts)
    starts.append(len(text))
    Name=molFile.replace(".trj","")
    for i in range(len(starts)-1):
        molecule_lines = [text[line].replace("\n","") for line in range(starts[i], starts[i+1])]
        try:
            os.mkdir(path+str(i))
        except FileExistsError:
            pass
        if i != len(starts)-2:
            assert len(molecule_lines) == OrigNat + 2, "There seems to be incorrect formatting in the trajectory file, number of atoms != NAT"
        else: 
            assert len(molecule_lines) >= OrigNat + 2, "There seems to be incorrect formatting in the trajectory file, number of atoms != NAT"
        
        textDump(molecule_lines, path+str(i)+"/"+str(Name)+"_"+str(i)+".xyz")
        Molecules[i] = pyscfTools.genMol(path+str(i)+"/"+str(Name)+"_"+str(i)+".xyz", charge, spin, basis, symmetry)
    return Molecules

# def StateWrite(state: list, path: str):
#     hf = h5py.File(path, "w")
#     hf.create_dataset("StateVector", data=state)
#     hf.close()

# def StateRead(path: str)->list:
#     hf = h5py.File(path, "r")
#     state=hf.get("StateVector")
#     state = numpy.array(state)
#     return state

# def rdmWrite(rdm1: numpy.array, rdm2: numpy.array, path:str):
#     hf = h5py.File(path, "w")
#     hf.create_dataset("RDM1", data=rdm1)
#     hf.create_dataset("RDM2", data=rdm2)
#     hf.close()

# def rdmRead(path:str) -> (numpy.array, numpy.array):
#     hf = h5py.File(path, "r")
#     rdm1 = hf.get("RDM1")
#     rdm1 = numpy.array(rdm1)
#     rdm2 = hf.get("RDM2")
#     rdm2 = numpy.array(rdm2)
#     return rdm1, rdm2


# def h5Read(keys: list, path:str)->list:
#     hf = h5py.File(path, "r")
#     if len(keys) == 0:
#         keys = list(hf.keys())
#         Data = [None]*len(keys)
#         for i in range(len(keys)):
#             Data[i] = hf.get(keys[i])
#         hf.close()
#         return Data, keys
#     else:            
#         Data = [None]*len(keys)
#         for i in range(len(keys)):
#             Data[i] = numpy.array(hf.get(keys[i]))
#         hf.close()
#         return Data

def h5Read(keys:list, path:str)->dict:
    hf = h5py.File(path, "r")
    if len(keys) == 0:
        keys = list(hf.keys())
    Data = {}
    for key in keys:
        Data[str(key)] = numpy.array(hf.get(key))
    hf.close()
    return Data


# def h5Write(Data: list, keys: list, path:str):
    # print(len(Data), len(keys))
    # hf = h5py.File(path, "w")
    # for i in range(len(keys)):
    #     hf.create_dataset(keys[i], data=Data[i])
    # hf.close()
def h5Write(Data: dict, path:str):
    hf = h5py.File(path, "w")
    keys = list(Data.keys())
    for key in keys:
        print(key)
        hf.create_dataset(str(key), data=Data[key])
    hf.close()
    

def MolWrite(mol: pyscf.M, path:str, filetype="xyz"):
    if filetype == "xyz":
        nat = len(mol._atom)
        lines = [None]*(nat+2)
        lines[0] = str(nat)
        lines[1] = "pySCF generated molecule file"
        for i in range(nat):
            lines[2+i] = str(mol._atom[i][0])+" "+str(round(mol._atom[i][1][0]/1.88972612, 8))+" "+str(round(mol._atom[i][1][1]/1.88972612,8)) +" "+str(round(mol._atom[i][1][2]/1.88972612,8)) # Internal atomic coordinates are stored as Bohr for pySCF
        textDump(lines, path)

