
'''

Spectral analysis of multi-agent topology adjacency matrices via the Successor Representation (SR) formulation.

Spectral features extracted from the SR matrix include:

- Spectral Radius: error propagation

- Spectral Gap: convergence speed

- Condition Number: topological fragility


'''


import numpy as np
from dataclasses import dataclass

#-----------
## Adjacency Matrix Constructors
#-----------

def _row_normalize(A):
    '''
    Ensures the max eigenvalue is 1 for SR stability.

    Since we are using M = (I - γA)^{-1} for SR there is a mathematical limit of γ x ρ(A) < 1

    This could produce nonsensical negative eigenvalues if not normalized.

    The following row-normalizes the adjacency matrices so they are proper transition matrices.
    
    '''
    row_sums = A.sum(axis=1)
    #Prevents division by zero for terminal nodes
    row_sums[row_sums==0] = 1.0

    return A/row_sums[:, np.newaxis]




def chain_adjacency(n):
    '''

    "The Telephone"

    A causal implementation of a chain of agents.
    That is information only flows from one agent to the next.
    
    Agent 1 --> Agent 2 --> Agent N
    
    '''
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    return _row_normalize(A)



def star_adjacency(n_leaves):
    '''

    "The Judge"

    That is information only flows from leaves to judge in a single step.
    But the judge state does flow back to the leafs in the next step so technically its bidirectional. 

                Agent 2
                   |
                   V
    Agent 1 --> Judge <-- Agent 3
    
    '''
    n = n_leaves + 1
    A = np.zeros((n, n))
    for leaf in range(1, n):
        #Leaf outputs to judge
        A[leaf, 0] = 1.0
        #Judge outputs back to leaf for next step
        A[0, leaf] = 1.0
    return _row_normalize(A)


def mesh_adjacency(n):
    '''
    "The Debator"

    Agent-to-agent topology.
    Every agent makes a decision from every agents opinions.

    Agent 1 <--> Agent 2

    '''
    if n < 2:
        raise ValueError('Mesh topology requires at least 2 nodes.')
    A = np.full((n, n), 1.0)
    np.fill_diagonal(A, 0.0)
    return _row_normalize(A)



#-----------
## Results
#-----------


@dataclass
class SpectralResult:
    '''
    Holds all spectral features for one (topology, gamma) combination.

    '''
    gamma:           float
    sr_matrix:       np.ndarray
    eigenvalues:     np.ndarray
    spectral_radius: float
    spectral_gap:    float
    condition_number: float

    def summary(self):
        return (
            f'Gamma={self.gamma:.3f}\n'
            f'SpRad={self.spectral_radius:.4f}\n'
            f'SpGap={self.spectral_gap:.4f}\n'
            f'CndNbr={self.condition_number:.4f}'
        )



#-----------
## Spectral Core
#-----------



class Spectral:

    '''
    
    Sucessor Representation analyzer. 

    Input Params:

    - A: adjacency matrix
    - Gamma: discount factor; can be a single value or list
    
    '''

    def __init__(self, A, gamma):
        A = np.array(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(
                f'Adjacency matrix must be square 2D, but got shape {A.shape}.'
            )
        self.A = A
        self.gamma = gamma


    #-----------
    ## Compute Functions
    #-----------

    def _sr_compute(self, gamma):
        '''
        Performs the computations of the SR Formulation:

        M = (I - γA)^{-1}


        '''
        I = np.eye(self.A.shape[0])
        inner  = I - gamma * self.A
        M = np.linalg.inv(inner)
        eigvals = np.linalg.eigvals(M)

        return M, eigvals
    

    def _require_scalar_gamma(self):
        '''
        To ensure valid gamma input before using single gamma methods.

        '''
        if self.gamma is None:
            raise ValueError('Gamma is not set. Pass gamma= when instantiating.')
        if isinstance(self.gamma, (list, np.ndarray)):
            raise ValueError(
                'Gamma is a list. Use multi_gamma() method or set singular gamma value.'
            )
        return float(self.gamma)
    

    #-----------
    ## Single Gamma Methods
    #-----------

    def sr_matrix(self):
        '''
        High level SR matrix method.

        Returns: (M, eigenvalues)

        '''
        gamma = self._require_scalar_gamma()

        return self._sr_compute(gamma)


    def spectral_radius(self):
        '''
        High level spectral radius method.

        ρ(M) = max |λ_i|

        '''
        _, eigv = self._sr_compute(self._require_scalar_gamma())

        return float(np.max(np.abs(eigv)))


    def spectral_gap(self):
        '''
        High level spectral gap method.

        Δ = |λ_1| - |λ_2|

        '''
        _, eigv = self._sr_compute(self._require_scalar_gamma())
        sorted_eigv = np.sort(np.abs(eigv))[::-1]

        #Need at least 2 since its the difference between the two largest
        if len(sorted_eigv) < 2:
            return 0.0
        
        return sorted_eigv[0] - sorted_eigv[1]


    def condition_nbr(self):
        '''
        Computes the condition through the standard formulation.

        κ(M) = |M| * |M^{-1}|

        '''
        M, _ = self._sr_compute(self._require_scalar_gamma())

        return float(np.linalg.cond(M))
    

    def full_single_analysis(self):
        '''
        Runs all three metrics at once for current scalar gamma.
        
        '''
        gamma = self._require_scalar_gamma()
        M, eigvals = self._sr_compute(gamma)
        sorted_eigs = np.sort(np.abs(eigvals))[::-1]
        gap = float(sorted_eigs[0] - sorted_eigs[1]) if len(sorted_eigs) >= 2 else 0.0

        return SpectralResult(
            gamma=gamma,
            sr_matrix=M,
            eigenvalues=eigvals,
            spectral_radius=float(np.max(np.abs(eigvals))),
            spectral_gap=gap,
            condition_number=float(np.linalg.cond(M))
        )
    


    #-----------
    ## Multi Gamma Methods
    #-----------


    def multi_gamma(self, gammas, print_results=True):
        '''
        Iterative analysis for multiple gamma levels.

        Inputs Include:
        - gammas: list of gamma values to evaluate
        - print_results: prints a summary per gamma if true

        Returns a list of the SpectralResult dataclasses per gamma
         
        '''
        #Gamma list handling
        if gammas is None:
            if isinstance(self.gamma, (list,np.array)):
                gammas = list(self.gamma)
            elif self.gamma is not None:
                gammas = [float(self.gamma)]
            else:
                raise ValueError('Provide gammas as argument or set gamma= when instantiating.')
            
        results = []

        #Loops through gammas and provides full analysis for each
        for g in gammas:
            tmp = Spectral(self.A, gamma=g)
            result = tmp.full_single_analysis()
            results.append(result)

            if print_results:
                print(result.summary())
                print('SR Matrix: \n', np.round(result.sr_matrix, 3))
                print('Eigenvalues:', np.round(result.eigenvalues, 3))

        return results



