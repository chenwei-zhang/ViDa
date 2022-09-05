"""
Reads from DNA23 pickle cache and creates hdf cache. Each CTMC is stored with:
    K: rate matrix
    Si: initial state
    Sf: final state
Arrhenius model, trained with IS and BBVI prior, are used to determine each K
"""

import numpy as np
import csv
import h5py
import pickle

PATH = '.'

def open_csv(document) :
    """open a csv file"""
    with open(document, 'rt') as f:
        my_CSV = list(csv.reader(f))
    return my_CSV

class ArrheniusModels:
    def __init__(self, params) -> None:
        self.params = np.atleast_2d(params)
        assert self.check_types(self.params)

        theta = self.params[0]
        self.kinetic_parameters = {
            "stack": (theta[0], theta[1]),
            "loop": (theta[2], theta[3]),
            "end": (theta[4], theta[5]),
            "stack+loop": (theta[6], theta[7]),
            "stack+end": (theta[8], theta[9]),
            "loop+end": (theta[10], theta[11]),
            "stack+stack": (theta[12], theta[13]),
            "alpha": theta[14]}

    @staticmethod
    def check_types(params: np.ndarray) -> bool:
        return params.ndim == 2 and params.shape[1] == 15

    def transition_rate(self, concentration, T, DeltaG, n_complexdiff, left, right):
        R = 0.001987
        RT = R * (T + 273.15)

        """
        R is the molar gas constant in kcal/(mol K).
        """

        """
        Uses the Arrhenius kinetic model to calculate the transition rate for
        state1->state and state2->state1. Only returns the transition rate for
        state1->state2.
        """

        DeltaG2 = -DeltaG

        lnA_left, E_left = self.kinetic_parameters[left]
        lnA_right, E_right = self.kinetic_parameters[right]
        lnA = lnA_left + lnA_right
        E = E_left + E_right

        if n_complexdiff == 0:
            """Using plus 2 instead of plus 1 since were calculating we're calculating the transition \
                rate from state1 to state2 and from state2 to state1 simultaneously. """
            if DeltaG > 0.0:
                rate1 = np.e **(lnA - (DeltaG + E) / RT)
                rate2 = np.e ** (lnA - E / RT)
            else:
                rate1 = np.e ** (lnA - E / RT)
                rate2 = np.e **(lnA - (DeltaG2 + E) / RT)
        elif n_complexdiff == 1:
            rate1 = (self.kinetic_parameters["alpha"] * concentration) * np.e  ** (lnA - E / RT)
            rate2 = self.kinetic_parameters["alpha"] * np.e ** (lnA - (DeltaG2 + E) / RT)
        elif n_complexdiff == -1:
            rate1 = self.kinetic_parameters["alpha"] * np.e ** (lnA - (DeltaG + E) / RT)
            rate2 = (self.kinetic_parameters["alpha"] * concentration) * np.e  ** (lnA - E / RT)
        else:
            raise ValueError(
                'Exception in Arrhenius_rate function.  Check transition rate calculations!')

        return rate1, rate2

def read_cache(dataset_name, docID):
    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "statespace" + "/" + "statespace" + str(docID), "rb") as state_file:
        statespace = pickle.load(state_file, encoding='latin1')

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "energy" + "/" + "energy" + str(docID), "rb") as energy_file:
        energies = pickle.load(energy_file, encoding='latin1')

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "transition_structure" + "/" + "transition_structure" + str(docID), "rb") as context_file:
        transition_structure= pickle.load(context_file, encoding='latin1')

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "fast_access" + "/" + "fast_access" + str(docID), "rb") as fast_file:
        # fast_access = pickle.load(fast_file, encoding='latin1')
        fast_access = None

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "ncomplex" + "/" + "ncomplex" + str(docID), "rb") as n_file:
        ncomplex = pickle.load(n_file, encoding='latin1')

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "initial" + "/" + "initial" + str(docID), "rb") as initial_file:
        initial = pickle.load(initial_file, encoding='latin1')

    with open(PATH + "/pickle_cache/" +dataset_name+ "/" + "final" + "/" + "final" + str(docID), "rb") as final_file:
        final = pickle.load(final_file, encoding='latin1')

    return statespace, energies, transition_structure, ncomplex, initial, final


def get_docID(row, name, _zip=None) :
    if _zip==None:
        docID = name + str(row)
    else:
        docID = name + str(_zip) + str(row)
    return docID


def add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics):
    n = len(statespace)
    state_map = np.zeros((n, len(statespace[0])))
    energy = np.zeros(n)
    for s in statespace:
        state_map[statespace.index(s), :] = s
        energy[statespace.index(s)] = energies[s]

    K = np.zeros((n,n))
    for s1, s2 in transition_structure:
        i = statespace.index(s1)
        j = statespace.index(s2)
        DeltaG = energies[s2] - energies[s1]
        n_complexdiff = ncomplex[s1] - ncomplex[s2]
        left, right = transition_structure[(s1,s2)]
        rate1, rate2 = kinetics.transition_rate(concentration, T, DeltaG, n_complexdiff, left, right)
        K[i,j] = rate1
        K[j,i] = rate2
    Si = np.int32(statespace.index(initial))
    Sf = np.int32(statespace.index(final))

    g1 = h5f.create_group(docID)

    # NOTE: When using HDF5 Julia package and h5py in Python3,
    #   Multi-dimensional arrays have reversed order.
    # Therefore using transpose of matrices.
    g1.create_dataset('K',data=K.T)
    g1.create_dataset('states',data=state_map.T)

    g1.create_dataset('energies',data=energy)

    # one-indexed for Julia
    g1.create_dataset('Si',data=Si+1)
    g1.create_dataset('Sf',data=Sf+1)

    # concentration and bimol to calculate rate in Julia
    g1.create_dataset('conc',data=concentration)
    g1.create_dataset('bimol',data=bimol)

def create_hf5(h5f, kinetics, datasets):

    for reaction_type in datasets:
        if reaction_type == "bubble":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Altanbonnet")
                    lenrow = len(file[row])
                    concentration = float (file[row][lenrow-3])
                    Kinv = float (file[row][4])
                    T = 1000/Kinv - 273.15
                    bimol = False
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "four_waystrandexchange":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Dabby")
                    lenrow = len(file[row])
                    concentration = np.max ( ( float (file[row][lenrow-2] ), float (file[row][lenrow-1]  ) ) )
                    # TODO: inverse
                    Kinv = float (file[row][6])
                    T = 1000/Kinv - 273.15
                    bimol = True
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "hairpin":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                j = reaction_dataset[-3]
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Bonnet"+j, _zip)
                    lenrow = len(file[row])
                    if _zip==0:
                        concentration = float (file[row][lenrow-2])
                    else:
                        concentration = float (file[row][lenrow-3])
                    Kinv = float (file[row][3])
                    T = 1000/Kinv - 273.15
                    bimol = False
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "hairpin1":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "GoddardT", _zip)
                    lenrow = len(file[row])
                    concentration = float (file[row][lenrow-3])
                    # TODO: inv temp
                    Kinv = float (file[row][3])
                    T = 1000/Kinv - 273.15
                    bimol = False
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "hairpin4":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Kim", _zip)
                    lenrow = len(file[row])
                    concentration = float (file[row][lenrow-3])
                    Kinv= float (file[row][3])
                    T = 1000/Kinv - 273.15
                    bimol = False
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "helix":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Morrison", _zip)
                    lenrow = len(file[row])
                    concentration = np.max ( ( float (file[row][lenrow-2] ), float (file[row][lenrow-1]  ) ) )
                    Kinv = float (file[row][3])
                    T = 1000/Kinv - 273.15
                    bimol = _zip
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "helix1":
            for reaction_dataset in datasets[reaction_type]:
                _zip = False
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "ReynaldoDissociate", _zip)
                    lenrow = len(file[row])
                    concentration = np.max ( ( float (file[row][lenrow-2] ), float (file[row][lenrow-1]  ) ) )
                    T = float (file[row][3])
                    bimol = _zip
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "three_waystranddisplacement":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "Zhang")
                    lenrow = len(file[row])
                    concentration = np.max ( ( float (file[row][lenrow-2] ), float (file[row][lenrow-1]  ) ) )
                    Kinv = float (file[row][5])
                    T = 1000/Kinv - 273.15
                    bimol = True
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1
        if reaction_type == "three_waystranddisplacement1":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/raw" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    docID = get_docID(row, "ReynaldoSequential")
                    lenrow = len(file[row])
                    concentration = np.max ( ( float (file[row][lenrow-2] ), float (file[row][lenrow-1]  ) ) )
                    T = float (file[row][3])
                    bimol = True
                    statespace, energies, transition_structure, ncomplex, initial, final \
                         = read_cache(reaction_id, docID)
                    add_dataset(statespace, transition_structure, energies, ncomplex, initial, final, concentration, T, bimol, docID, h5f, kinetics)
                    row+=1

def main():

    h5f = h5py.File("dna23.h5", "w")

    theta = [12.5805935,   4.186373,   13.36014277,  2.84494644, 12.27265568,  3.07707082,\
        12.90134082,  3.53452839, 12.95720823,  3.21707749, 12.2951224,   2.18171263,\
        13.5356558,   2.83848218,  0.06588867]
    kinetics = ArrheniusModels([theta])

    datasets = { "bubble": ["Fig4"],
                 "four_waystrandexchange": ["Table5.2"],
                 "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"],
                 "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
                 "hairpin4" : ["Table1_0", "Table1_1"],
                 "helix" : ["Fig6_0", "Fig6_1"],
                 "helix1" : ["Fig6a"],
                 "three_waystranddisplacement" : ["Fig3b"],
                 "three_waystranddisplacement1" : ["Fig6b"]
    }

    create_hf5(h5f, kinetics, datasets)

if __name__ == "__main__":
    main()