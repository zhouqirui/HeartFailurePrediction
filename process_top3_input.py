from collections import defaultdict
from datetime import datetime
import pickle

hf_list = ['40201','40211','40291','40401','40403','40411','40413','40491','40493','4280','4281','42820','42821','42822','42823','42830','42831','42832','42833','42840','42841','42842','42843','4289']

        
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
        priority = int(line[3]) if line[3]!='' else 0
        if code != '':
            adid_icd[adid].append((priority, code))
    for key in adid_icd.keys():
        adid_icd[key].sort(key = lambda x: x[0])
        adid_icd[key] = [i[1] for i in adid_icd[key]]
    
    f.close()

    patient_visit_order = dict()
    for pid, adids in pid_adid.items():
        if len(adids) >= 2:
            s = sorted([(adid_date[adid], adid_icd[adid]) for adid in adids])
            patient_visit_order[pid] = s

    pids = []
    dates = []
    icds = []
    c=0
    d=0
    for pid, date_icd in patient_visit_order.items():
        if [] not in [i[1] for i in date_icd]:
            c +=1
            pids.append(pid)
            dates.append([i[0] for i in date_icd])
            icds.append([i[1] for i in date_icd])
        else:
            d+=1
    print(c, d)
        
    
    icd_types = dict()
    encoded = []
    for p in icds:
        encoded_p = []
        # if p[-1] == []:
        #     print(p)
        #     print(icds.index(p))
        for v in p:
            encoded_v = []
            for icd in v:
                if icd not in icd_types.keys():
                    icd_types[icd] = len(icd_types)
                encoded_v.append(icd_types[icd])
            # if encoded_v==[]:
            # print(p)
            encoded_p.append(encoded_v)
        encoded.append(encoded_p)
    # print(len(icd_types))

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
        # print(patient)
        sequences.append(patient[:-1])
        labels.append(patient[-1][0])
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


    # for i in range(len(labels)):
    #     if any([code in converted_hf for code in labels[i]]):
    #         labels[i] = 0
    #     else:
    #         labels[i] = 1

    ## Put in one list
    for i in range(len(sequences)):
        tmp = []
        for v in sequences[i]:
            tmp += v
        sequences[i] = tmp

    print(sequences[:10])
    print(labels[:10])
    # print(time[:10])
    print(max([len(i) for i in sequences]))
    print(max(labels))


    # pickle.dump(sequences, open('sequences', 'wb'), protocol=1)
    # pickle.dump(labels, open('labels', 'wb'),protocol=1)
    # pickle.dump(time, open('time', 'wb'), protocol=1)

    return pid_adid, adid_date, adid_icd, patient_visit_order


if __name__ == '__main__':
    pid_adid, adid_date, adid_icd, patient_visit_order = main('MIMIC-III/ADMISSIONS.csv', 'MIMIC-III/DIAGNOSES_ICD.csv')
