import scipy.io
import pandas as pd

def load_MP_data(data_file):
    MP_data = scipy.io.loadmat(data_file)
    return MP_data

def get_friends_A(MP_data):
    A = MP_data['friends_A']
    A = (A + A.T).astype('bool')
    A = A.astype('int')
    return A

def get_followers_A(MP_data):
    A = MP_data['followers_A']

    return A

def get_all_votes(MP_data):
    attr_M = pd.DataFrame(MP_data['labeled_ys'])
    attr_M.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I7 V1', 'I7 V2', 'I7 V3',\
    'I7 V4', 'I7 V5', 'I7 V6', 'I7 V7', 'I7 V8', 'I7 V9', 'I7 V10', 'I8 V1', 'I8 V2', 'I8 V3',\
    'I8 V4', 'I8 V5', 'I8 V6', 'I8 V7', 'I8 V8', 'V9', 'I10 V1', 'I10 V2', 'I10 V3', 'I10 V4', \
    'I10 V5', 'I10 V6', 'I10 V7', 'I10 V8', 'V11', 'I12 V1', 'I12 V2', 'I12 V3', 'I12 V4', 'V13']

    return attr_M

def get_all_parties(MP_data):
    parties = MP_data['label_party']
    idx_party_dict = {1:'Not applicable', 2:'Conservative', 3:'Democratic Unionist Party', 4:'Green', \
    5:'Independent', 6:'Liberal Democrat', 7:'Labour', 8:'Plaid Cymru', 9:'Sinn Fein', 10:'Scottish National Party'}
    MP_party_dict = {MP_id: idx_party_dict[parties[MP_id][0]] for MP_id in range(len(parties))}

    return parties, MP_party_dict

def get_MP_names(MP_data):
    allVotes = MP_data['allVotes']
    idx_name_dict = {}
    for i in range(len(allVotes)):
        idx_name_dict[i] = allVotes[i][1][0]

    return idx_name_dict
