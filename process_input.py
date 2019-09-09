from collections import defaultdict
from datetime import datetime
import pickle

hf_list = ['40201','40211','40291','40401','40403','40411','40413','40491','40493','4280','4281','42820','42821','42822','42823','42830','42831','42832','42833','42840','42841','42842','42843','4289']

# def process_input(admission_f, ICD_c_f):
#     pidadid, adiddate = defaultdict(list), dict()
#     f = open(admission_f)
#     for line in f.readlines()[1:]:
#         line = line.strip().split(',')
#         pid, adid = int(line[1]), int(line[2])
#         date = datetime.strptime(line[3], '%Y-%m-%d %H:%M:%S')
#         pidadid[pid].append(adid)
#         adiddate[adid] = date
#     f.close()

#     diagnosis_code = defaultdict(list)
#     f = open(ICD_c_f)
#     for line in f.readlines()[1:]:
#         line = line.strip().split(',')
#         adid, code = line[2], line[4]
#         diagnosis_code[adid].append(code[1:-1])
#     f.close()

#     code_l = []
#     for _, j in diagnosis_code.items():
#         for code in j:
#             if j not in code_l:
#                 code_l.append(j)

#     print(len(code_l))

# def get_all_icd(ICD_D, ICD_P):
#     D = []
#     P = []
#     f = open(ICD_D)
#     for line in f.readlines()[1:]:
#         code = line.split(',')[1][1:-1]
#         D.append(code)
#     f.close()

#     f = open(ICD_P)
#     for line in f.readlines()[1:]:
#         code = line.split(',')[1][1:-1]
#         P.append(code)
#     f.close()
#     return D, P

# def count(ICD_c_f):
#     pid_icd = defaultdict(list)
#     icd_count = defaultdict(int)
#     f = open(ICD_c_f)
#     for line in f.readlines()[1:]:
#         line = line.strip().split(',')
#         # print(line)
#         # print(line[3], type(line[3]))
#         pid = int(line[1])
#         admission_id = int(line[2])
#         seq = int(line[3]) if line[3] != '' else -1
#         # print(pid, seq, type(pid), type(seq))
#         icd = line[-1][1:-1]
#         if icd.startswith('E'):
#             if len(icd)>4:
#                 icd = icd[:4]
#         # elif icd.startswith('V'):
#         #     if len(icd)>3:
#         #         icd = icd[:3]
#         else:
#             if len(icd)>3:
#                 icd = icd[:3]
#         if icd != '':
#             pid_icd[pid].append((admission_id, seq, icd))
#             icd_count[icd] += 1
#     f.close()
#     for pid in pid_icd.keys():
#         pid_icd[pid].sort(key = lambda x:(x[0], x[1]))
#     return pid_icd, icd_count
        
def main(admissions, diagnosis):
    pid_adid = defaultdict(list)
    adid_date = dict()

    f = open(admissions)
    for line in f.readlines()[1:]:
        line = line.strip().split(',')
        pid, adid = int(line[1]), int(line[2])
        date = datetime.strptime(line[3], '%Y-%m-%d %H:%M:%S')
        pid_adid[pid].append(adid)
        adid_date[adid] = date
    f.close()

    adid_icd = defaultdict(list)
    f = open(diagnosis)
    for line in f.readlines()[1:]:
        line = line.strip().split(',')
        adid, code = int(line[2]), line[4][1:-1]
        # if code.startswith('E'):
        #     if len(code)>4:
        #         code = code[:4]
        # else:
        #     if len(code)>3:
        #         code = code[:3]
        if code != '':
            adid_icd[adid].append(code)
    f.close()

    patient_visit_order = dict()
    for pid, adids in pid_adid.items():
        if len(adids) >= 2:
            s = sorted([(adid_date[adid], adid_icd[adid]) for adid in adids])
            patient_visit_order[pid] = s

    pids = []
    dates = []
    icds = []
    for pid, date_icd in patient_visit_order.items():
        pids.append(pid)
        dates.append([i[0] for i in date_icd])
        icds.append([i[1] for i in date_icd])
    
    icd_types = dict()
    encoded = []
    for p in icds:
        encoded_p = []
        for v in p:
            encoded_v = []
            for icd in v:
                if icd not in icd_types.keys():
                    icd_types[icd] = len(icd_types)
                encoded_v.append(icd_types[icd])
                # else:
                #     encoded_v.append(icd_types[icd])
            encoded_p.append(encoded_v)
        encoded.append(encoded_p)
    print(len(icd_types))

    date_code = dict()
    time = []
    for p in dates:
        tmp = []
        for date in p[:-1]:
            if date not in date_code.keys():
                date_code[date] = len(date_code)
            tmp.append(date_code[date])
        time.append(tmp)



    sequences, labels = [], []
    for patient in encoded:
        sequences.append(patient[:-1])
        labels.append(patient[-1])
        # for visit in patient:
        #     if len(visit) > 1:
        #         sequences.append([visit[:-1]])
        #         labels.append(visit[-1])
    print(sequences[:10])
    # print(labels[:10])

    converted_hf = []
    for code in hf_list:
        if code in icd_types.keys():
            converted_hf.append(icd_types[code])
    # print(converted_hf)


    for i in range(len(labels)):
        if any([code in converted_hf for code in labels[i]]):
            labels[i] = 0
        else:
            labels[i] = 1

    ## Put in one list
    for i in range(len(sequences)):
        tmp = []
        for v in sequences[i]:
            tmp += v
        sequences[i] = tmp

    print(sequences[:10])
    print(time[:10])
    print(max([len(i) for i in sequences]))


    pickle.dump(sequences, open('sequences', 'wb'), protocol=1)
    pickle.dump(labels, open('labels', 'wb'),protocol=1)
    pickle.dump(time, open('time', 'wb'), protocol=1)

    # pickle.dump(pids, open('PIDs', 'wb'), -1)
    # pickle.dump(dates, open('DATEs', 'wb'), -1)
    # pickle.dump(encoded, open('encoded', 'wb'), -1)
    # pickle.dump(icd_types, open('ICD_types', 'wb'), -1)

    return pid_adid, adid_date, adid_icd, patient_visit_order

# def choose_hf(admissions, diagnosis):
    pid_adid = defaultdict(list)
    adid_date = dict()

    

    adid_icd = defaultdict(list)
    f = open(diagnosis)
    for line in f.readlines()[1:]:
        line = line.strip().split(',')
        adid, code = int(line[2]), line[4][1:-1]
        if code.startswith('E'):
            if len(code)>4:
                code = code[:4]
        else:
            if len(code)>3:
                code = code[:3]
        if code != '':
            adid_icd[adid].append(code)
    f.close()

    f = open(admissions)
    for line in f.readlines()[1:]:
        line = line.strip().split(',')
        pid, adid = int(line[1]), int(line[2])
        date = datetime.strptime(line[3], '%Y-%m-%d %H:%M:%S')
        pid_adid[pid].append(adid)
        adid_date[adid] = date
    f.close()

    new = dict()
    for key in pid_adid.keys():
        if any([adid in pid_adid[key] for adid in adid_icd.keys()]):
            new[key] = pid_adid[key]
    pid_adid = new

    patient_visit_order = dict()
    for pid, adids in pid_adid.items():
        if len(adids) >= 2:
            s = sorted([(adid_date[adid], adid_icd[adid]) for adid in adids])
            patient_visit_order[pid] = s

    pids = []
    dates = []
    icds = []
    for pid, date_icd in patient_visit_order.items():
        pids.append(pid)
        dates.append([i[0] for i in date_icd])
        icds.append([i[1] for i in date_icd])
    
    icd_types = dict()
    encoded = []
    for p in icds:
        encoded_p = []
        for v in p:
            encoded_v = []
            for icd in v:
                if icd not in icd_types.keys():
                    icd_types[icd] = len(icd_types)
                encoded_v.append(icd_types[icd])
                # else:
                #     encoded_v.append(icd_types[icd])
            encoded_p.append(encoded_v)
        encoded.append(encoded_p)

    sequence, label = [], []
    
    return encoded




if __name__ == '__main__':

    pid_adid, adid_date, adid_icd, patient_visit_order = main('MIMIC-III/ADMISSIONS.csv', 'MIMIC-III/DIAGNOSES_ICD.csv')

    # a = choose_hf('MIMIC-III/ADMISSIONS.csv', 'MIMIC-III/DIAGNOSES_ICD.csv')
