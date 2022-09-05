# contour plot of the test function
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import h5py


class MetropolisModels:

    def __init__(self, params) -> None:
        self.params = np.atleast_2d(params)
        assert self.check_types(self.params)

        theta = self.params[0]
        self.kinetic_parameters ={ "k_uni" : theta[0] , "k_bi" : theta[1] }

    @staticmethod
    def check_types(params: np.ndarray) -> bool:
        return params.ndim == 2 and params.shape[1] == 2

    def transition_rate(self, concentration, T, DeltaG, n_complexdiff):

        if n_complexdiff == None:
            n_complexdiff = 0

        R = 0.001987
        RT = R * (T + 273.15)

        DeltaG2 = -DeltaG

        k_uni = self.kinetic_parameters["k_uni"]
        k_bi = self.kinetic_parameters["k_bi"]

        if  n_complexdiff == 1    :
            rate1 = k_bi * concentration
            rate2 = k_bi * np.e ** ( - DeltaG2 / RT)

        elif n_complexdiff ==  -1   :
            rate1 = k_bi * np.e ** ( - DeltaG / RT)
            rate2 = k_bi * concentration

        elif  n_complexdiff == 0   :

            if DeltaG > 0.0:
                rate1 = k_uni  * np.e **(-DeltaG  / RT)
                rate2 = k_uni
            else:
                rate1 = k_uni
                rate2 = k_uni  * np.e **(-DeltaG2  / RT)

        else :
            raise ValueError(
                'Exception, fix this in Metropolis_rate function. \
                Check transition rate calculations!')

        assert rate1 != None and rate2 != None

        return rate1, rate2

##################################################

def rescale_mean(A,n):
    return np.multiply(A,n/2) + n/2*np.ones(2)

def rescale_cov(S,n):
    return np.multiply(S,n**2/4)

def rescale_SiSf(percentiles_init, percentiles_final, n):
    initials = [ (x1*n//100, y1*n//100) for x1,y1 in percentiles_init]
    finals = [ (x1*n//100, y1*n//100) for x1,y1 in percentiles_final]

    return initials, finals

##################################################

def plot_contour(obj,x,y,initials,finals):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, obj, levels=50,cmap="inferno")

    for i in range(len(initials)):
        plt.plot([initials[i][0],finals[i][0]], [initials[i][1],finals[i][1]])

    # plt.show()

def plot_energies(energies):
    fig = plt.figure()
    ax = sns.heatmap(energies.T, cmap="inferno")
    ax.invert_yaxis()
    # plt.show()

##################################################

def initial_final_conitions(initials, finals, n):
    # one-indexed for Julia
    initials = [ coord_to_index(x1, y1, n)+1 for x1,y1 in initials]
    finals = [ coord_to_index(x1, y1, n)+1 for x1,y1 in finals]

    return np.array(initials), np.array(finals)

def convert_energy(n, obj, min, max):
    max_obj = np.max(obj)
    energies = obj - (max_obj)*np.ones((n,n))

    min_E = np.min(energies)
    energies = np.multiply(energies, (min-max)/min_E) + max*np.ones((n,n))

    return energies

def coord_to_index(x,y, n):
    # starts at bottom, goes left to right.
    return int(y*n + x)

def index_to_cood(i,n):
    return (i%n, i//n)

##################################################

def transition_strucutre(n):
    transitions = []
    for i in range(n**2):
        for j in range(i+1,n**2):
            if (j-i== 1 and j%n!=0) or abs(i-j) == n:
                transitions.append((i,j))
    return transitions

def statespace(n):
    # map from state index to its coordinate representation
    states = np.zeros((n**2,2))      # n^2 states, represented by a tuple.
    for x in range(n):
        for y in range(n):
            i = coord_to_index(x,y,n)
            states[i, :] = [x,y]
    return states

##################################################

def side_basin(pos,n,rev=False):
    # side_basin -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([.8,-.8], n),
        rescale_cov([[0.1, 0], [0, .1]], n))
    rv2 = multivariate_normal(
        rescale_mean([-.8,0], n),
        rescale_cov([[.1, 0], [0, .1]], n))
    rv3 = multivariate_normal(
        rescale_mean([0,.8], n),
        rescale_cov([[20.0, 0], [0, 0.1]], n))
    rv4 = multivariate_normal(
        rescale_mean([0,.8], n),
        rescale_cov([[0.1, 0], [0, 10.0]], n))
    rv5 = multivariate_normal(
        rescale_mean([0,-.8], n),
        rescale_cov([[0.1, 0], [0, 10]], n))

    obj = n*(-.8*rv1.pdf(pos) - \
            rv2.pdf(pos) + \
            rv3.pdf(pos) + \
            rv4.pdf(pos) + \
            rv5.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(50,90),(50,90),(10,50),(10,50),(90,10),(90,10)]
    percentiles_final = [(90,10),(10,50),(90,10),(50,90),(10,50),(50,90)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def side_basin2(pos,n,rev=False):
    # side_basin2 -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([0,1], n),
        rescale_cov([[2.0, 0], [0, 2.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([-1,-1], n),
        rescale_cov([[.2, 0], [0, .2]], n))
    rv3 = multivariate_normal(
        rescale_mean([1,-1], n),
        rescale_cov([[.2, 0], [0, .5]], n))
    rv4 = multivariate_normal(
        rescale_mean([0,-1], n),
        rescale_cov([[.3, .8], [.3, 1]], n))

    obj = n*(rv1.pdf(pos) - \
            1.2*rv2.pdf(pos) - \
            rv3.pdf(pos) + \
            rv4.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(50,99),(40,50),(99,50),(0,0),(0,0),(99,0),(0,99),(99,0),(0,0),(99,99)]
    percentiles_final = [(99,0),(99,0),(0,0),(99,50),(99,0),(0,0),(99,0),(0,99),(99,99),(0,0)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def central_peak(pos,n,rev=False):
    # central_peak -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([.5,.5], n),
        rescale_cov([[1.0, 0], [0, 1.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([.5,-.5], n),
        rescale_cov([[1.0, 0], [0, 1.0]], n))
    rv3 = multivariate_normal(
        rescale_mean([-.5,.5], n),
        rescale_cov([[1.0, 0], [0, 1.0]], n))
    rv4 = multivariate_normal(
        rescale_mean([-.5,-.5], n),
        rescale_cov([[1.0, 0], [0, 1.0]], n))
    rv5 = multivariate_normal(
        rescale_mean([0,0], n),
        rescale_cov([[10.0, 0], [0, 0.1]], n))
    rv6 = multivariate_normal(
        rescale_mean([0,0], n),
        rescale_cov([[0.1, 0], [0, 10.0]], n))

    obj = n*(-rv1.pdf(pos) - \
            rv2.pdf(pos) - \
            rv3.pdf(pos) - \
            rv4.pdf(pos) + \
            rv5.pdf(pos) + \
            rv6.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(50,50),(50,50),(0,99),(0,99),(50,0),(50,0),(20,20)]
    percentiles_final = [(20,80),(0,99),(50,50),(99,0),(50,99),(99,50),(80,80)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def parallel_mountains(pos,n,rev=False):
    # parallel_mountains -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([0,0], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([.5,.5], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv3 = multivariate_normal(
        rescale_mean([-.5,-.5], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv4 = multivariate_normal(
        rescale_mean([0,-1], n),
        rescale_cov([[1.0, 0], [0, 0.01]], n))
    rv5 = multivariate_normal(
        rescale_mean([-1,0], n),
        rescale_cov([[0.01, 0], [0, 1.0]], n))
    rv6 = multivariate_normal(
        rescale_mean([0,1], n),
        rescale_cov([[1.0, 0], [0, 0.01]], n))
    rv7 = multivariate_normal(
        rescale_mean([1,0], n),
        rescale_cov([[0.01, 0], [0, 1.0]], n))

    obj = n*(rv1.pdf(pos)+rv2.pdf(pos)+rv3.pdf(pos)) -   \
            .2*n*(rv4.pdf(pos)+rv5.pdf(pos)+rv6.pdf(pos)+rv7.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(0,0),(50,50),(50,50),(30,30),(30,30),(30,30),(99,0)]
    percentiles_final = [(99,99),(0,0),(99,0),(0,0),(99,0),(99,99),(0,99)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def parallel_mountains2(pos,n,rev=False):
    # parallel_mountains2 -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([0,0], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([.5,.5], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv3 = multivariate_normal(
        rescale_mean([-.5,-.5], n),
        rescale_cov([[1.0, -.95], [-.95, 1.0]], n))
    rv4 = multivariate_normal(
        rescale_mean([0,-1], n),
        rescale_cov([[1.0, 0], [0, 0.01]], n))
    rv5 = multivariate_normal(
        rescale_mean([-1,0], n),
        rescale_cov([[0.01, 0], [0, 1.0]], n))
    rv6 = multivariate_normal(
        rescale_mean([0,1], n),
        rescale_cov([[1.0, 0], [0, 0.01]], n))
    rv7 = multivariate_normal(
        rescale_mean([1,0], n),
        rescale_cov([[0.01, 0], [0, 1.0]], n))
    rv8 = multivariate_normal(
        rescale_mean([1,-1], n),
        rescale_cov([[0.1, 0], [0, 0.1]], n))

    obj = n*(rv1.pdf(pos)+rv2.pdf(pos)+rv3.pdf(pos)) -   \
            .2*n*(rv4.pdf(pos)+rv5.pdf(pos)+rv6.pdf(pos)+rv7.pdf(pos)+2*rv8.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(0,0),(50,50),(50,50),(30,30),(30,30),(30,30),(99,0)]
    percentiles_final = [(99,99),(0,0),(99,0),(0,0),(99,0),(99,99),(0,99)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def mountain_detour(pos,n,rev=False):
    # mountain_detour -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([0,0], n),
        rescale_cov([[1.0, -.9], [-.9, 1.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([.5,.5], n),
        rescale_cov([[1.0, -.9], [-.9, 1.0]], n))
    rv3 = multivariate_normal(
        rescale_mean([-.5,-.5], n),
        rescale_cov([[1.0, -.9], [-.9, 1.0]], n))
    rv4 = multivariate_normal(
        rescale_mean([0,-1], n),
        rescale_cov([[2.0, 0], [0, 0.001]], n))
    rv5 = multivariate_normal(
        rescale_mean([-1,0], n),
        rescale_cov([[0.001, 0], [0, 2.0]], n))
    rv6 = multivariate_normal(
        rescale_mean([0,1], n),
        rescale_cov([[2.0, 0], [0, 0.001]], n))
    rv7 = multivariate_normal(
        rescale_mean([1,0], n),
        rescale_cov([[0.001, 0], [0, 2.0]], n))

    obj = n*(
        rv1.pdf(pos) + \
        rv2.pdf(pos) + \
        rv3.pdf(pos) - \
        .1*rv4.pdf(pos) - \
        .1*rv5.pdf(pos) - \
        .2*rv6.pdf(pos) - \
        .2*rv7.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(0,0),(50,50),(50,50),(30,30),(30,30),(30,30),(99,0)]
    percentiles_final = [(99,99),(0,0),(99,0),(0,0),(99,0),(99,99),(0,99)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def bridge(pos,n,rev=False):
    # bridge -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([.5,-.5], n),
        rescale_cov([[1.0, -.8], [-.8, 1.0]], n))
    rv2 = multivariate_normal(
        rescale_mean([-.5,.5], n),
        rescale_cov([[1.0, .8], [.8, 1.0]], n))
    rv3 = multivariate_normal(
        rescale_mean([-.5,-1], n),
        rescale_cov([[.1, 0], [0, .1]], n))
    rv4 = multivariate_normal(
        rescale_mean([1,.5], n),
        rescale_cov([[.1, 0], [0, .1]], n))

    obj = n*(rv1.pdf(pos)+rv2.pdf(pos)-rv3.pdf(pos)-rv4.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(25,75),(80,20),(0,99),(80,20),(25,0),(25,0),(25,0),(25,99)]
    percentiles_final = [(80,20),(25,75),(80,20),(0,99),(99,75),(25,75),(25,99),(25,0)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def three_valleys1(pos,n,rev=False):
    # three_valleys1 -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([.8,-.8], n),
        rescale_cov([[1, 0], [0, 1]], n))
    rv2 = multivariate_normal(
        rescale_mean([-.8,.8], n),
        rescale_cov([[.1, 0], [0, .1]], n))
    rv3 = multivariate_normal(
        rescale_mean([.8,.8], n),
        rescale_cov([[.2, 0], [0, .1]], n))
    rv4 = multivariate_normal(
        rescale_mean([-.8,-.8], n),
        rescale_cov([[.1, 0], [0, .2]], n))

    obj = n*(rv1.pdf(pos)-rv2.pdf(pos)-rv3.pdf(pos)-rv4.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(90,10),(10,90),(90,10),(90,90),(10,90),(90,90),(10,50),(10,50),
                         (10,10),(50,50),(90,90),(50,50),(10,90),(90,50),(50,90)]
    percentiles_final = [(10,90),(90,10),(90,90),(90,10),(90,90),(10,90),(90,10),(90,90),
                         (90,90),(90,90),(50,50),(10,90),(50,50),(0,50),(50,0)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals

def three_valleys2(pos,n,rev=False):
    # three_valleys2 -> done
    # rescales and computes unnormalized objective for any nxn square grid

    assert(pos.shape == (n,n,2))

    rv1 = multivariate_normal(
        rescale_mean([.8,-.8], n),
        rescale_cov([[1, 0], [0, 1]], n))
    rv2 = multivariate_normal(
        rescale_mean([-.8,.8], n),
        rescale_cov([[.1, 0], [0, .1]], n))
    rv3 = multivariate_normal(
        rescale_mean([.8,.8], n),
        rescale_cov([[.1, 0], [0, .1]], n))
    rv4 = multivariate_normal(
        rescale_mean([-.8,-.8], n),
        rescale_cov([[.1, 0], [0, .1]], n))

    obj = n*(rv1.pdf(pos)-.5*rv2.pdf(pos)-rv3.pdf(pos)-rv4.pdf(pos))

    if rev:
        obj*=-1

    percentiles_init =  [(90,10),(10,90),(90,10),(90,90),(10,90),(90,90),(10,50),(10,50),
                         (10,10),(50,50),(90,90),(50,50),(10,90),(90,50),(50,90)]
    percentiles_final = [(10,90),(90,10),(90,90),(90,10),(90,90),(10,90),(90,10),(90,90),
                         (90,90),(90,90),(50,50),(10,90),(50,50),(0,50),(50,0)]

    initials, finals = rescale_SiSf(percentiles_init, percentiles_final, n)

    return obj, initials, finals


##################################################

def generate_dataset(n, x, y, pos, states, transitions, min_energy, max_energy, T, landscape, plot, kinetics, h5f, rev=False):

    # states -> n^2 x 2 numpy array. (index -> coodinate)
    # transitions -> list of tuples. Each tuple is (index,index) of neighbouring states.
    #   (reverse transitions not included)

    obj, initials, finals = eval(landscape)(pos, n, rev)
    energies = convert_energy(n, obj, min_energy, max_energy)

    energy = np.zeros(n**2)
    for s in states:
        ind = coord_to_index(s[0], s[1], n)
        energy[ind] = energies[int(s[0]), int(s[1])]

    K = np.zeros((n**2,n**2))
    for i,j in transitions:
        DeltaG = energy[j] - energy[i]
        rate1, rate2 = kinetics.transition_rate(None, T, DeltaG, None)
        K[i,j] = rate1
        K[j,i] = rate2

    Si_indexes, Sf_indexes = initial_final_conitions(initials, finals ,n)

    g1 = h5f.create_group(landscape)
    g1.create_dataset('K',data=K.T)
    g1.create_dataset('energies',data=energy)
    g1.create_dataset('Si',data=Si_indexes)
    g1.create_dataset('Sf',data=Sf_indexes)

    if plot:
        plot_contour(obj,x,y,initials,finals)
        plot_energies(energies)

def main():

    theta = [2.41e6,None]             # kbi is not used
    kinetics = MetropolisModels([theta])

    landscapes = [
                  "side_basin",
                  "side_basin2",
                  "central_peak",
                  "parallel_mountains",
                  "parallel_mountains2",
                  "mountain_detour",
                  "bridge",
                  "three_valleys1",
                  "three_valleys2"]

    ####### options ###########
    n=20                    # grid size
    rev=True                # reversed landscape
    plot=True               # view plots?
    min_energy  = -15       # minimum energy on landscape
    max_energy = -.01        # maximum energy on landscape
    T = 32.0                # tempurature (required for metropolis model)

    x, y = np.mgrid[0:n, 0:n]
    pos = np.dstack((x, y))

    # Note: statespace structure (dictionnary) is the same for all grid CTMCs of the same size
    #       same for transition structure
    state_map = statespace(n)
    transitions = transition_strucutre(n)

    filename = "data/synthetic_rev"+str(n)+".h5"
    # filename = "temp.h5"
    h5f = h5py.File(filename, "w")

    h5f.create_dataset('states',data=state_map.T)

    for f in landscapes:
        generate_dataset(n, x, y, pos, state_map, transitions, min_energy, max_energy, T, f, plot, kinetics, h5f, rev)

    plt.show()

if __name__ == "__main__":
    main()


# reasonable min-max energies ->
#       -10,-0.01 gives MFPT between 0.0002 and 2600
#       -15,-0.01 gives MFPT between 0.0002 6.8e6
