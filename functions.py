import numpy as np
from Bio import AlignIO
from scipy import spatial
USE_W = 0
if USE_W:
    W = np.loadtxt("wfunc2.dat")
else:
    W = np.ones((21,21))

def Corr(g, distances_A, distances_B):
    x = distances_A
    y = distances_B
    num = 0
    den_r = 0
    den_s = 0
    R_av = np.average(distances_A)
    S_av = np.average(distances_B)
    for i, i_b in enumerate(g):
        for j, j_b in enumerate(g):
            num += (x[i,j] - R_av) * (y[int(i_b),int(j_b)] - S_av)
            den_r += (x[i,j] - R_av)**2
            den_s += (y[int(i_b),int(j_b)] - S_av)**2
    r = num / (np.sqrt(den_r) * np.sqrt(den_s))
    return(r)

def local_fields(i, j, coupling_matrix, sitefreq, q):
    couplings=np.ones([q,q])
    for A in range(q-1):
        for B in range(q-1):
            couplings[A,B] = coupling_matrix[i*(q-1)+A, j*(q-1)+B]
    exp_local_i = np.ones((q))/q
    exp_local_j = np.ones((q))/q
    Pi = sitefreq[i,:]
    Pj = sitefreq[j,:]

    epsilon=.0001
    diff=1.0
    while(diff>epsilon):

         scra1 = np.dot(exp_local_i, np.transpose(couplings))
         scra2 = np.dot(exp_local_j, couplings)

         new1 = np.divide(Pi, scra1)
         new1 = new1/(np.sum(new1))
         new2 = np.divide(Pj, scra2)
         new2 = new2/np.sum(new2)

         abs_diff = [max(abs(new1-exp_local_i)), max(abs(new2-exp_local_j))]

         diff = max(abs_diff)

         exp_local_i = new1
         exp_local_j = new2

    return couplings, exp_local_i, exp_local_j, Pi, Pj

def Codemsa(g, encoded_msa0, seqs_b, OFFSET, theta):
    g = np.array(g).astype(int)
    encoded_msa = encoded_msa0.copy()
    encoded_msa[:,OFFSET:] = seqs_b[g]

    #WEIGHT SEQUENCES
    hammdist = spatial.distance.pdist(encoded_msa, 'hamming')
    weight_matrix = spatial.distance.squareform(hammdist < (1.0- theta))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    Meff = np.sum(weight)
    return encoded_msa, Meff

def Sitefreq(encoded_msa, Meff, ics, nA, q, LAMBDA):
    sitefreq = np.empty((nA,q),dtype=float)
    for i in range(nA):
        for aa in range(q):
            sitefreq[i,aa] = np.sum(np.equal(encoded_msa[:,i],aa))/Meff
    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/q
    return sitefreq

def Entropy(sitefreq,nA,nB):
    ent = np.zeros((nA+nB),dtype=float)
    for i in range(nA+nB):
        ent[i] = -np.sum(sitefreq[i,:]*np.log(sitefreq[i,:]))
    return ent

def cantor(x, y):
    return (x + y) * (x + y + 1) / 2 + y

def Pairfreq(encoded_msa, Meff, ics, nP, idx_pairs, sitefreq, q, LAMBDA):
    pairfreq = np.zeros((nP,q,nP,q),dtype=float)
    for i, col_i in enumerate(ics):
        for j, col_j in enumerate(ics):
            c = cantor(encoded_msa[:,col_i],encoded_msa[:,col_j])
            unique,aaIdx = np.unique(c,True)
            for x,item in enumerate(unique):
                pairfreq[i,encoded_msa[aaIdx[x],col_i],j,encoded_msa[aaIdx[x],col_j]] = np.sum(np.equal(c,item))

    pairfreq /= Meff
#    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(q*q)

    for i,(phi,psi) in enumerate(idx_pairs):
        for am_i in range(q):
            for am_j in range(q):
                if (am_i==am_j):
                    pairfreq[i,am_i,i,am_j] = sitefreq[phi,am_i]
                else:
                    pairfreq[i,am_i,i,am_j] = 0.0
    return pairfreq

def Coupling(sitefreq, pairfreq, nP, q):
    corr_matrix = np.empty(((nP)*(q-1), (nP)*(q-1)),dtype=float)
    for i in range(nP):
        for j in range(nP):
            for am_i in range(q-1):
                for am_j in range(q-1):
                    corr_matrix[i*(q-1)+am_i, j*(q-1)+am_j] = pairfreq[i,am_i,j,am_j] - sitefreq[i,am_i]*sitefreq[j,am_j]

    inv_corr = np.linalg.inv(corr_matrix)
    coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix

def SP(coupling_matrix, sitefreq, h1, h2, idx_pairs, nP, q):
    data_h1 = np.empty((nP,q-1),dtype=float)
    data_h2 = np.empty((nP,q-1),dtype=float)
    data_c_pp = np.empty((nP,q-1,q-1),dtype=float)
    tmp = q-1

    for k in range(nP):
        i = idx_pairs[k][0]
        j = idx_pairs[k][1]
        data_h1[k,:] = np.log(h1[i,j,:tmp])
        data_h2[k,:] = np.log(h2[i,j,:tmp])
        data_c_pp[k,:,:] = np.log(coupling_matrix[i*tmp:(i+1)*tmp,j*tmp:(j+1)*tmp])
    d0 = np.reshape(data_c_pp,(nP*tmp*tmp))
    d1 = np.reshape(data_h1, (nP*tmp))
    d2 = np.reshape(data_h2, (nP*tmp))
    sp_norm_pp = np.linalg.norm(np.concatenate((d0, d1, d2))[:nP*tmp*tmp])

    return sp_norm_pp

def information(sitefreq, pairfreq, nP, pairs, idx_pairs, q):
    H_xy_matrix_pp = np.empty((nP),dtype=float); t = 10e-10
    mi_matrix_pp = np.empty((nP),dtype=float)
    for k,(i,j) in enumerate(idx_pairs):
        pij = pairfreq[i,:,j,:]
        pi = np.transpose(np.broadcast_to(sitefreq[i,:],(q,q)))
        pj = np.transpose(np.broadcast_to(sitefreq[j,:],(q,q)))
        mi_matrix_pp[k] = np.sum(pij*np.log(pij/(pi*pj+t)+t))
        H_xy_matrix_pp[k] = -np.sum(pij*np.log(pij+t))

    return mi_matrix_pp, np.sum(H_xy_matrix_pp)

def local_fields_vectors(coupling_matrix,sitefreq,nA, nB, q):
    h1 = np.empty((nA+nB,nA+nB,q),dtype=float)
    h2 = np.empty((nA+nB,nA+nB,q),dtype=float)
    for i in range(nA+nB):
        for j in range(nA+nB):
            eij, h_i, h_j, Pi, Pj = local_fields(i, j, coupling_matrix, sitefreq, q)
            h1[i,j,:] = h_i
            h2[i,j,:] = h_j
    return h1, h2

def direct_information(coupling_matrix,sitefreq, nP, q):
    tiny = 1.0e-100
    H_xy_di = np.zeros((nP,nP),dtype=float)
    DI = np.zeros((nP,nP),dtype=float)

    ent = np.zeros(nP,dtype=float)
    for i in range(nP):
        ent[i] -= np.sum(sitefreq[i,:]*np.log(sitefreq[i,:]))


    for i in range(nP-1):
        for j in range(i+1,nP):
            eij, h_i, h_j, Pi, Pj = local_fields(i, j, coupling_matrix, sitefreq, q)
            x = np.multiply(eij, np.outer(h_i, h_j))
            Pij = x/sum(sum(x))
            H_xy_di[i,j] = -(np.sum(Pij[:,:]*np.log(Pij[:,:])*W))
            Pfac = np.outer(Pi, Pj)
            z = np.outer(Pij*W, np.log((Pij+tiny)/(Pfac+tiny))*W)
            DI[i,j] = np.trace(z)
            DI[j,i] = DI[i,j]

    average_DI = np.average(DI)
    for i in range(nP-1):
        average_i = np.average(DI[i,:])
        for j in range(i+1,nP):
            correction = (np.average(DI[:,j])*average_i)/average_DI
            DI[i,j] = DI[i,j] - correction
            DI[j,i] = DI[i,j]

    for i in range(nP):
        if ent[i] > (np.average(ent)+np.std(ent)):
            DI[i,:]=0
            DI[:,i]=0

    return DI, H_xy_di

def calc_delta_f(pairfreq1, pairfreq2, ics, pairs):
    for i,col_i in enumerate(ics):
        for j,col_j in enumerate(ics):
            if (col_i,col_j) not in pairs:
                pairfreq1[i,:,j,:] = 0
                pairfreq2[i,:,j,:] = 0
    delta_f = pairfreq1 - pairfreq2
    delta_f *= delta_f
    delta_f = np.sqrt(np.sum(delta_f))

    return delta_f
