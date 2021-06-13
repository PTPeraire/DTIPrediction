import os
import pickle
import h5py
from functools import partial
from multiprocessing import Pool
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from scipy.spatial import distance
import numpy as np

def get_bio():
    with open("/aloy/home/ptorren/parse_chembl/biology_metadata_v27_onlypchembl.pkl", "rb") as f:
        bio_md27 = pickle.load(f)
    return bio_md27

def _create_target_profile(opath):
    with h5py.File("/aloy/web_checker/current/full/B/B4/B4.001/sign0/sign0.h5", "r") as f:
        B4 = f["V"][:]
        features = f["features"][:]
        keys = f["keys"][:]
    
    with h5py.File("/aloy/web_checker/current/full/B/B4/B4.001/sign3/sign3.h5", "r") as f:
        keysS3 = f["keys"][:]
        
    B4T = B4.transpose()
    
    compound_inchis = {}
    for tgt, up  in tqdm(zip(B4T, features)):
        if up.startswith("Class"):
            continue
        cpd_idx = tgt.nonzero()[0]
        if len(cpd_idx) < 2:
            continue
        compound_inchis[up] = [inchis[i] for i in np.isin(list(keysS3), list(keys[cpd_idx])).nonzero()[0]]
    
    del keysS3
    del B4
    del keys
    del inchis
    
    sign_GLOBAL = Signaturizer('GLOBAL')
    compound_sign_GLOBAL = {}
    for tgt, up  in tqdm(zip(B4T, features)):
        if up.startswith("Class"):
            continue
        cpd_idx = tgt.nonzero()[0]
        if len(cpd_idx) < 2:
            continue
        cpd_inch = compound_inchis[up]
        compound_sign_GLOBAL[up] = sign_GLOBAL.predict(cpd_inch, keytype='InChI').signature
    with open(opath, "wb") as f:
        pickle.dump(compound_sign_GLOBAL,f)

def get_targetprofiles(opath):
    if os.path.exists(opath):
        with open(opath, "rb") as f:
            target_profile = pickle.load(f)
        return target_profile
    else:
        print("This option is currently not available: please use the target profile file provided")
        return None
        # TODO: Test creating target profiles
#         _create_target_profile()
#         with open(opath, "rb") as f:
#             target_profile = pickle.load(f)
#         return target_profile

def get_NN(up, query_sign, compound_signs):
    results = []
    tgt_sign = compound_signs[up]
    for QS in query_sign:
        results.append(min([distance.cosine(QS, s) for s in tgt_sign]))
    return results

def map_D2T(query_sign, compound_signs, uniprot=None, n_cpus=None):
    if uniprot is None:
        uniprot = sorted(list(compound_signs.keys()))
    elif type(uniprot) == str:
        if uniprot.lower() == 'human':
            bio_md27 = get_bio()
            uniprot = np.asarray(list(compound_signs.keys()))[np.isin(list(compound_signs.keys()), bio_md27.uniprot_id.values.tolist())].tolist()
        else:
            print("Target universe not contemplated")
            exit()
    else:
        uniprot = uniprot
    if n_cpus is None:
        res = []
        for up in tqdm(uniprot):
            if up is None:
                res.append(np.ones(len(query_sign)).tolist())
                continue
            res.append(get_NN(up, query_sign = query_sign, compound_signs = compound_signs))
    else:
        d2t_mapping = partial(get_NN, query_sign = query_sign, compound_signs = compound_signs)
        res = Pool(n_cpus).map(d2t_mapping, uniprot)
    return res, uniprot

def run(query_sign, compound_sign, uniprot=None, n_cpus=None):
    results, targets = map_D2T(query_sign, compound_sign, uniprot)
    rank = stats.rankdata(results, method='ordinal')
    scal = RobustScaler().fit_transform(results)
    df = pd.DataFrame([[tgt, r, dist, s ] for tgt, r, dist, s in zip(targets, rank.flatten(), np.asarray(results).flatten(), scal.flatten())], columns = ['Target', 'Rank', 'Score', 'Scale'])
    return df
