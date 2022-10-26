import numpy as np
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def split_data(adjs, coeffs, energies):

    split_data = [adjs, coeffs,  energies ]
    tr_adjs, te_adjs, tr_coeffs, te_coeffs, tr_energies, te_energies = train_test_split(*split_data,
                                                                    train_size=0.70,
                                                                    random_state=42)    


    return [tr_adjs, tr_coeffs, tr_energies], [te_adjs, te_coeffs, te_energies]