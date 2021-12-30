# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 09:30:03 2021

@author: Ambrosio Valencia
"""

import numpy as np
import quantecon.game_theory as gt

from itertools import chain, combinations, product
# from more_itertools import powerset
from operator import xor
from numbers import Number


#%% Function `powergset`
def powergset(iterable, return_complement = True, exclude_empty = True):
    """
    Generate the powerset from the elements in set `iterable`
    with the option to return the complement of each subset
    with respect to `iterable` and exclude the empty set.

    Parameters
    ----------
    iterable : array_like
        The set of element. It can be an object of type ``set``, ``list``,
        ``tuple``, or ``numpy.ndarray``.
    return_complement : bool, optional
        Whether to return each subset's complement with respect
        to `iterable`. The default is True.
    exclude_empty : bool, optional
        Whether to exclude the empty set from the powerset. The default is True.

    Returns
    -------
    list
        The powerset.

    """
    
    # Convert iterable to set
    iter_set = set(iterable)
    
    # If `exclude_empty` is True, start counting from index 1
    # thus excluding the empty set from the resulting powerset `pset`
    if not isinstance(exclude_empty, bool): start = 1
    elif exclude_empty: start = 1
    else: start = 0
        
    power_set = chain.from_iterable(combinations(iter_set, r)
                                    for r in range(len(iter_set) + 1)[start:])
    
    # Assemble powerset
    pset = [ set(s) for s in power_set ]
    
    # If `return_complement` is True, return each set
    # difference between the `iter_set` and each of
    # the elements in `pset` and return both `pset`
    # and the set of such differences `rset`
    if return_complement:        
        rset = [ iter_set-s for s in pset ]        
        return pset, rset
    # otherwise, just return `pset`
    else:
        return pset
    

#%% Function `gameset`
def nx2gameset(n, i, start_from = 0, return_JuK = False, return_N_i = False):
    """
    Generate set of all player-reduced normal-form games
    observed by player :math:`i` within the grand :math:`n`-player game.

    Parameters
    ----------
    n : int
        The number of players. It must be greater than 0.
        Any number that is not an integer will be transformed into one.
    i : int
        Player :math:`i`'s numerical index. It must be equal or greater than 0.
        Any number that is not an integer will be transformed into one.
    start_from : int, optional
        Numerical index of the first player. The default is 0.
    return_JuK : bool, optional
        Whether to return the sets :math:`\\mathcal{J}` (third entry), 
        :math:`\\mathcal{N}_{\\varphi}\\setminus{i}` (fourth entry),
        and :math:`\\mathcal{K}` (fifth entry). The default is False.
    return_N_i : bool, optional
        Whether to return the set :math:`\\mathcal{N}\\setminus{i}`.
        The default is False.

    Returns
    -------
    list, tuple
        A list of lists, each with two elements,
        :math:`\\mathcal{N}_{\\psi}\\setminus\\mathcal{J}` and
        :math:`\\mathcal{N}_{\\psi}`. Three more elementes,
        :math:`\\mathcal{J}`, :math:`\\mathcal{N}_{\\varphi}\\setminus{i}`,
        and :math:`\\mathcal{K}`, are included if `return_JuK` is `True`.
        If `return_N_i` is `True`

    """    
    n = int(n)
    i = int(i)
    
    #%% Create the cell array 'gset' to allocate the subgames
    gset = []
        
    # This is the set of players N\{i} faced by i
    N__i = { j+start_from for j in range(n) if j+start_from != i }
    
    #%% Organize the sets of players in N\{i} whom i will face...
    #   'Js_and_Phis': |J| + |Nphi\J| > 0, with , where...
    #                  J is the set of players j whose s_j are unknown
    #                  Nphi\J are the players k not in J that play s_k = phi
    #     'Only_Psis': Npsi\J are the players k not in J that play s_k = psi
    # 
    #   Keep in mind that |J| > 0 and |Nphi\J| >= 0.
    #   So 'Js_and_Phis' must not include the empty set.
    #   To do that, set the second argument in function 'powerset' to 'true'.
    #   The first output will give you 'Js_and_Phis'.
    #   The second output will give you the differences between 'N__i' and each
    #   set in 'Js_and_Phis'... That is, 'Only_Psis'.
    Js_and_Phis, Only_Psis = powergset(N__i)

    #   Loop across the set of 'Js_and_Phis'
    for j in range(len(Js_and_Phis)):
        
        # Create subsets of possible J from each subset in 'Js_and_Phis'.
        # Keep in mind that |J| > 0; so 'powerset(...,true)'.
        J = powergset( Js_and_Phis[j], return_complement = False )
        
        # Now loop across each J
        for idx in range(len(J)):
        
            gset.append([])
            
            # First subset (0) includes the possible Npsi\J.
            # This subset is used for the scenario where s_j = phi is assumed.
            gset[-1].append(Only_Psis[j])
            
            # Second subset (1) includes the unions of Npsi\J and J.
            # This subset is used for the scenario where s_j = psi is assumed.
            gset[-1].append( Only_Psis[j] | J[idx] );
            
            if return_JuK:
                # Third subset (2) are the players j in J.
                gset[-1].append( J[idx] );
                
                # Fourth subset (3) includes the possible Nphi\J.
                gset[-1].append( xor(N__i, gset[-1][1]) );
                
                # Fifth subset (4) includes the unions of Nphi\J and Npsi\J.
                # These are all the players k in K, those whose s_k are given.
                gset[-1].append( Only_Psis[j] | gset[-1][3] );
                
    if return_N_i:
        # * Benny, bring me everyone.
        # - What do you mean "everyone"?
        # * EVERYONE!!!
        return gset, N__i
    else:
        return gset


#%% Function `nsubgames`
def nx2subgames(n):
    """
    Calculate the number of player-reduced (sub) games observed by
    each player in an :math:`n`-player 2-strategy normal-form game.

    Parameters
    ----------
    n : int
        The number of players. It must be greater than 0.
        Any number that is not an integer will be transformed into one.

    Returns
    -------
    int
        The number of player-reduced (sub) games. It returns zero
        if :math:`n < 2`.
        
    Notes
    -----
    For any number of players :math:`n \\geq 2`, the number of distinct
    values of structural fear and greed sum up to :math:`n \\cdot {g}_{(n)}`,
    where :math:`{g}_{(n)}` is the number of all possible player-reduced games
    :math:`\\mathcal{G}_{\\mathcal{N} \\setminus \\mathcal{K}}` observed by
    :math:`i \\in \\mathcal{N}` and it amounts to:
    
    .. math::
        {g}_{(n)}
        = \\sum_{k=0}^{n-2} (2^{n-k-1} - 1) \\cdot {}_{n-1}C_{k},
        
    which by means of the binomial identity simplifies to
    
    .. math::
        {g}_{(n)} = 3^{n-1} - 2^{n-1}.
        
    .. [1] A. Valencia-Romero, "Strategy Dynamics in Collective Systems
       Design," PhD dissertation, Stevens Institute of Technology, 2021.

    """
    if n > 1:
        return int(3**(int(n)-1) - 2**(int(n)-1))
    else:
        return 0


#%% Function `nx2gamma` to obtain the harmony gamma values
def nx2gamma(n, npsi):
    """
    Calculate the harmony gamma value given a number of players
    :math:`n_{\\psi\\setminus i}` (excluding :math:`i`) who play pure strategy
    :math:`\\psi_k` in an :math:`n`-player 2-strategy normal-form game.

    Parameters
    ----------
    n : int
        The number of players. It must be greater than 0.
        Any number that is not an integer will be transformed into one.
    npsi : int
        The number of players other than :math:`i` who play one of two
        pure strategies. Any number that is not an integer will be
        transformed into one.

    Returns
    -------
    int
        The gamma value.

    """
    
    nphi = int(n) - int(npsi) - 1
    
    return int(2**int(npsi) + 2**nphi - 2)

#%% #%% Function `nx2feargreed` to calculate several fear and greed spaces
def nx2feargreed(nf, c = None, surplus_only = False, who = 'all',
                 u_amplitude = None):
    
    # Check if this is a tuple of NumPy arrays 
    # or a QuantEcon normal-form game.
    # In either case, check the number of arrays input and
    # the number of strategy combinations in each of them;
    # the former must coincide with the number of players
    # resulting from the logarithm base-2 of the latter.
    if isinstance(nf, gt.normal_form_game.NormalFormGame):
        n_ = len(nf.players)
        s_comb = set([ np.size(ui.payoff_array) for ui in nf.players ])
                
    elif np.all([ type(ui) == np.ndarray for ui in nf ]):
        n_ = len(nf)
        s_comb = set([ np.size(ui) for ui in nf ])
    
    # If neither, report it.
    else:
        raise ValueError("This is neither a tuple of NumPy arrays or a "+
                         "QuantEcon normal-form game.")
    
    # Check if `ci` has been provided anf if it is a valid input.
    if c == None:
        c = np.zeros(n_)
    elif isinstance(who, (tuple, list, np.ndarray)):
        if (len(c) == n_) and np.all([ isinstance(ci, Number) for ci in c ]):
            c = np.array(c)
        else:
            raise ValueError("c is not a numerical array or some of "+
                             "its elements are not numbers.")   
    else:
        raise ValueError("c is not a valid input.")            
    
    
    # Check if the validity of the player indices provided.
    if who == 'all':
        who = list(range(n_))
    elif isinstance(who, (tuple, list, set, np.ndarray)):
        for i in list(range(n_)):
            if i not in who:
                raise ValueError("Some of the player indices provided are "+
                                 "larger than the number of arrays input.")
    else:
        raise ValueError("The list of player indices provided "+
                         "is not an array object.")
    
    
    # Check if the validity of the player indices provided.
    if (len(s_comb) == 1) and (np.log2(next(iter(s_comb))) % 1 == 0):
        n = int(np.log2(next(iter(s_comb))))
        if n != n_:
            raise ValueError("This is not an nx2 normal-form game. "+
                             "Number of players is inconsistent.")
    else:
        if len(s_comb) != 1:
            raise ValueError("This is not an nx2 normal-form game "+
                             "Number of strategy combinations per player does not match.")
        else:
            raise ValueError("This is not an nx2 normal-form game "+
                             "At least one player has more than two strategies.")


#%% Function `nx2feargreed_i` to calculate individual fear and greed spaces
def nx2feargreed_i(i, payoff_array, ci = 0., return_harmony = True,
                   shift = True, surplus_only = False, ui_amplitude = None):
    """
    Calculate the structural fear and greed value space
    :math:`{\\langle F_i, G_i \\rangle}^\\text{T}` of player :math:`i` given
    their payoff array.

    Parameters
    ----------
    i : int
        Player :math:`i`'s numerical index. It must be equal or greater than 0.
        Any number that is not an integer will be transformed into one.
    payoff_array : numpy.ndarray
        The payoff array with :math:`2^n` elements.
    ci : array_like, optional
        Individual cost of selecting the individual alternative strategy (i.e.
        :math:`\\langle \\psi_i, s_{\\text{-}i} \\rangle` for all
        :math:`s_{\\text{-}i} \\in \\mathcal{S}_{i}^{n-1}`).
        The default is 0.
    return_harmony : bool, optional
        Whether to return player :math:`i`'s harmony index.
        The default is True.
    shift : bool, optional
        Whether to shift (rearrange) the indices (assumed to be assigned from
        the point of view of player :math:`i` so they match the (first-player
        centric) reference system. Required when the `payoff_array` parameter
        is taken from a `quantecon.game_theory` (`gt`) `Player` object.
        The default is True.
    surplus_only : bool, optional
        Whether to assign the cost to the collective individual strategy
        only (i.e. :math:`\\langle \\psi_i, \\psi_{\\text{-}i} \\rangle`).
        The default is False.
    ui_amplitude : Number, optional
        The amplitude of player :math:`i`'s payoffs. The default is None.

    Returns
    -------
    tuple
        A tuple with two numerical arrays of size :math:`1 \\times {g}_{(n)}`
        (check method `nx2subgames`). If `return_harmony`, a third element
        (`float` number) is output

    """
    
    # Check if array will be rewritten.
    # if rewrite:
    #     arr = payoff_array
    # else:
    arr = payoff_array.copy()
    
    # Obtain number of players
    n = int(np.log2(np.size(payoff_array)))
    
    # Reshape if array not input in 
    if np.shape(payoff_array) != n*(2,):
        payoff_array = np.reshape(payoff_array, n*(2,))
    
    # Create indices for ci assignements.
    # If surplus only, assign ci to the very
    # last strategy combination.
    if surplus_only:
        indices = tuple(np.ones(n, dtype=int))
    else:
        indices = n*[slice(None)]
        indices[i] = 1
        indices = tuple(indices)
    
    arr[indices] = arr[indices] - ci
    
    # Get the payoff amplitude of this player.
    if (ui_amplitude is None) or isinstance(ui_amplitude, Number):
        ui_amplitude = np.ptp(arr)
    
    #%% Get player-reduced game set and its size
    gset = nx2gameset(n, i)
    nx2g = nx2subgames(n)
    
    # Create Fi and Gi arrays
    Fi = np.zeros(nx2g)
    Gi = np.zeros(nx2g)
    
    # Iterate over each player-reduced game
    # to calculate Fi and Gi.
    for g in range(nx2g):
        
        #%% Create the indices for Fi
        phi_F = np.zeros(n, dtype=int)
        psi_F = np.zeros(n, dtype=int)
        for j in gset[g][0]:
            phi_F[j] = 1
            psi_F[j] = 1
        psi_F[i] = 1
        if shift:
            phi_F = nx2shiftindices(phi_F, i)
            psi_F = nx2shiftindices(psi_F, i)
        else:
            phi_F = tuple(phi_F)
            psi_F = tuple(psi_F)
                    
        Fi[g] = (arr[phi_F] - arr[psi_F])/ui_amplitude
        
        
        #%% Create the indices for Gi
        phi_G = np.zeros(n, dtype=int)
        psi_G = np.zeros(n, dtype=int)
        for k in gset[g][1]:
            phi_G[k] = 1
            psi_G[k] = 1
        psi_G[i] = 1
        if shift:
            phi_G = nx2shiftindices(phi_G, i)
            psi_G = nx2shiftindices(psi_G, i)
        else:
            phi_G = tuple(phi_G)
            psi_G = tuple(psi_G)
        
        Gi[g] = (arr[phi_G] - arr[psi_G])/ui_amplitude
    
    #%% Return
    if return_harmony:
        # return arr, Fi, Gi, -np.mean(Fi+Gi)/2
        return Fi, Gi, -np.mean(Fi+Gi)/2
    else:
        # return arr, Fi, Gi
        return Fi, Gi


#%% Shift indices
def nx2shiftindices(indices, i):
    """
    Shift the given strategy indices (i.e. the strategy combination) to the
    point of view of player :math:`i` in an :math:`n`-player 2-strategy
    normal-form game.

    Parameters
    ----------
    indices : array_like
        The input indices from a first-player perspective.
        Each entry must be either 0 or 1.
    i : int
        Player :math:`i`'s numerical index. It must be equal or greater than 0.
        Any number that is not an integer will be transformed into one.

    Returns
    -------
    tuple
        The shifted indices.

    """
    
    if np.all((np.array(indices)==0) + (np.array(indices)==1)):    
    
        n = len(indices)
        noroll_idx = np.array(list( product(*n*(range(2),)) ))
        roll_idx = np.roll(noroll_idx, i, axis = 1)
        
        x = np.where((roll_idx == tuple(indices)).all(axis=1))[0][0]
    
    else:
        raise ValueError("The indices must be either 0 or 1.")
        
    
    return tuple(noroll_idx[x,:])