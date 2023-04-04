import joblib
import numpy as np
import pandas as pd

def seeHerb(val):
    print(val)
    symptom = ['Immunity', 'Weight loss', 'Weight Gain',
                'Teeth and bone Strength', 'Knee pain/arthritis',
                'Blood detoxification', 'Infection and wounds', 'Body Pain',
                'Skin disease', 'Inactiveness', 'Gastric&Intestine troubles',
                'Hyperthermia or Fever', 'Diabetes', 'Cold,cough,throat infections',
                'Vision concerns', 'Nerve disorder', 'PCOS', 'Headache',
                'Respiratory concerns', 'Kidney & urinary problems', 'Heart problems',
                'Constipation', 'Diarrhoea', 'Liver', 'Male Reproductive concerns']
    herbs = ['Arugampul', 'Ashwagandha', 'Athimathuram', 'Arapu Leaves', 'Ammal Pacharisi', 'Athi Leaves', 'Arasu Leaves', 'Avuri Leaves', 'Arasu Seeds', 'Arasu pattai', 'Athi pattai', 'Amukura', 'Agraharam', 'Ashoka pattai', 'Aada Thodai', 'Aavaram Flowers', 'Aadu Theenda Palai', 'Avarai Panchangam', 'Aalamaram Seeds', 'Aalamaram pattai', 'Orange Peel', 'Aali Seeds', 'Aavarai Leaves', 'Lavangam pattai', 'Indhupu Flowers', 'Impural Flowers', 'Usilai', 'Oridhal Thamarai', 'Lemon Peel', 'Omam', 'Kandangkathiri', 'Kadukaai', 'Kasthuri Manjal', 'Curry Leaves', 'Karunjeeragam', 'Kabasura', 'Karungaali pattai', 'Karuvelam pattai', 'Kandanthipilli', 'Kalarchikaai', 'Kaasini Keerai', 'Kaarboga Arisi', 'Krambu', 'Keelanelli', 'Kuppaimeni', 'Kurunthotti Root', 'Koovai Kilangu', 'Kodi Veli', 'Kottai Karanthai', 'Guava Leaves', 'Korai Kilangu', 'Sadhakuppai', 'Srpagandha root', 'Sarkarai Kolli', 'Siru Thekkku', 'Sirukurinjaan', 'Siru Nerunjil', 'Siriyaanangai', 'Sitharathai', 'Siruthumbai leaves', 'Siruseruppadai', 'Sirupeelai', 'Sirupayaru', 'Sivanaar Vembu', 'Sivakaranthai', 'Seeragam', 'Seenthal kodi', 'Sukku', 'Surathu Nila aavarai', 'Surathu Nilaivaagai', 'Sembaruthi Flowers', 'Sabja seeds', 'Sevviyam', 'Aloe Vera', 'Thaneervittan Kilangu', 'Thaandri Kaai', 'Thaalisapathiri leaves', 'Thippli', 'Thiruneetru Pachilai', 'Thuthi Leaves', 'Tulsi Leaves', 'Thuthi Seeds', 'Thudhuvalai', 'Thottal Sivungi', 'Devatharu', 'Thetran Kottai', 'Nannari', 'Nathai Soori', 'Naaval Seeds', 'Nayuruvi Leaves', 'Naval pattai', 'Naalpamaraathi', 'Nilavembu ', 'Nithya kalyani roots', 'Nithya kalyani leaves', 'Nilavaagai', 'Nilapanai kizhangu', 'Neermulli', 'Neermulli seeds', 'Neeli auri leaves', 'Nochi leaves', 'Nellikai', 'Parpadagam', 'Paal mudhingan', 'Parangi pattai', 'Pakarkai', 'Badham Pisin', 'Pirandai', 'Pudhina leaves', 'Boomi chakkarai kilangu', 'Poovasaram pattai', 'Poolangilangu', 'Poonai kaali vidhai', 'Ponkorandi root', 'Ponnagkanni', 'Manjal karisalaangkanni', 'Mara manjal ', 'Magilam flowers', 'Marutham pattai', 'Marudhani leaves', 'Manathakkali ', 'Maavilangapattai', 'Maathulai peel', 'Maasikkaai', 'Maamparupu', 'Milagu', 'Mudakkathan', 'Musumusukkai leaf', 'Murungai Vidhai', 'Murungai Poo', 'Murungai pisin', 'Murungai leaves', 'Multhani matti', 'Mookaratai saaranai', 'Yaanai nerinjil', 'Roja Idhazh ', 'Vallarai', 'Vasambu', 'Vatta thiruppi', 'Vaazhai thandu', 'Vaaivilangam', 'Vadhanarayanan', 'Vaadhamadakki', 'Vishnugranthi', 'Virali manjal', 'Vilvam leaves', 'Vellai karisalaangkanni', 'Vettiver', 'Vendhayam', 'Venthamarai flowers', 'Vellarugu', 'Vellari seeds', 'Veppam leaves', 'Veppam flowers', 'Veliparuthi', 'Veppam pattai', 'Jadhikkai', 'Tulsi seeds', 'Sembaruthi leaves', 'Shataveri']

    tr = pd.read_csv("D:\\Python\\Herbal-Medicine-Predictor\\data.csv")
    tr.replace({'Herb Name':{'Arugampul': 0, 'Ashwagandha': 1, 'Athimathuram': 2, 'Arapu Leaves': 3, 'Ammal Pacharisi': 4, 'Athi Leaves': 5, 'Arasu Leaves': 6, 'Avuri Leaves': 7, 'Arasu Seeds': 8, 'Arasu pattai': 9, 'Athi pattai': 10, 'Amukura': 11, 'Agraharam': 12, 'Ashoka pattai': 13, 'Aada Thodai': 14, 'Aavaram Flowers': 15, 'Aadu Theenda Palai': 16, 'Avarai Panchangam': 17, 'Aalamaram Seeds': 18, 'Aalamaram pattai': 19, 'Orange Peel': 20, 'Aali Seeds': 21, 'Aavarai Leaves': 22, 'Lavangam pattai': 23, 'Indhupu Flowers': 24, 'Impural Flowers': 25, 'Usilai': 26, 'Oridhal Thamarai': 27, 'Lemon Peel': 28, 'Omam': 29, 'Kandangkathiri': 30, 'Kadukaai': 31, 'Kasthuri Manjal': 32, 'Curry Leaves': 33, 'Karunjeeragam': 34, 'Kabasura': 35, 'Karungaali pattai': 36, 'Karuvelam pattai': 37, 'Kandanthipilli': 38, 'Kalarchikaai': 39, 'Kaasini Keerai': 40, 'Kaarboga Arisi': 41, 'Krambu': 42, 'Keelanelli': 43, 'Kuppaimeni': 44, 'Kurunthotti Root': 45, 'Koovai Kilangu': 46, 'Kodi Veli': 47, 'Kottai Karanthai': 48, 'Guava Leaves': 49, 'Korai Kilangu': 50, 'Sadhakuppai': 51, 'Srpagandha root': 52, 'Sarkarai Kolli': 53, 'Siru Thekkku': 54, 'Sirukurinjaan': 55, 'Siru Nerunjil': 56, 'Siriyaanangai': 57, 'Sitharathai': 58, 'Siruthumbai leaves': 59, 'Siruseruppadai': 60, 'Sirupeelai': 61, 'Sirupayaru': 62, 'Sivanaar Vembu': 63, 'Sivakaranthai': 64, 'Seeragam': 65, 'Seenthal kodi': 66, 'Sukku': 67, 'Surathu Nila aavarai': 68, 'Surathu Nilaivaagai': 69, 'Sembaruthi Flowers': 70, 'Sabja seeds': 71, 'Sevviyam': 72, 'Aloe Vera': 73, 'Thaneervittan Kilangu': 74, 'Thaandri Kaai': 75, 'Thaalisapathiri leaves': 76, 'Thippli': 77, 'Thiruneetru Pachilai': 78, 'Thuthi Leaves': 79, 'Tulsi Leaves': 80, 'Thuthi Seeds': 81, 'Thudhuvalai': 82, 'Thottal Sivungi': 83, 'Devatharu': 84, 'Thetran Kottai': 85, 'Nannari': 86, 'Nathai Soori': 87, 'Naaval Seeds': 88, 'Nayuruvi Leaves': 89, 'Naval pattai': 90, 'Naalpamaraathi': 91, 'Nilavembu ': 92, 'Nithya kalyani roots': 93, 'Nithya kalyani leaves': 94, 'Nilavaagai': 95, 'Nilapanai kizhangu': 96, 'Neermulli': 97, 'Neermulli seeds': 98, 'Neeli auri leaves': 99, 'Nochi leaves': 100, 'Nellikai': 101, 'Parpadagam': 102, 'Paal mudhingan': 103, 'Parangi pattai': 104, 'Pakarkai': 105, 'Badham Pisin': 106, 'Pirandai': 107, 'Pudhina leaves': 108, 'Boomi chakkarai kilangu': 109, 'Poovasaram pattai': 110, 'Poolangilangu': 111, 'Poonai kaali vidhai': 112, 'Ponkorandi root': 113, 'Ponnagkanni': 114, 'Manjal karisalaangkanni': 115, 'Mara manjal ': 116, 'Magilam flowers': 117, 'Marutham pattai': 118, 'Marudhani leaves': 119, 'Manathakkali ': 120, 'Maavilangapattai': 121, 'Maathulai peel': 122, 'Maasikkaai': 123, 'Maamparupu': 124, 'Milagu': 125, 'Mudakkathan': 126, 'Musumusukkai leaf': 127, 'Murungai Vidhai': 128, 'Murungai Poo': 129, 'Murungai pisin': 130, 'Murungai leaves': 131, 'Multhani matti': 132, 'Mookaratai saaranai': 133, 'Yaanai nerinjil': 134, 'Roja Idhazh ': 135, 'Vallarai': 136, 'Vasambu': 137, 'Vatta thiruppi': 138, 'Vaazhai thandu': 139, 'Vaaivilangam': 140, 'Vadhanarayanan': 141, 'Vaadhamadakki': 142, 'Vishnugranthi': 143, 'Virali manjal': 144, 'Vilvam leaves': 145, 'Vellai karisalaangkanni': 146, 'Vettiver': 147, 'Vendhayam': 148, 'Venthamarai flowers': 149, 'Vellarugu': 150, 'Vellari seeds': 151, 'Veppam leaves': 152, 'Veppam flowers': 153, 'Veliparuthi': 154, 'Veppam pattai': 155, 'Jadhikkai': 156, 'Tulsi seeds': 157, 'Sembaruthi leaves': 158, 'Shataveri': 159}},inplace=True) 

    X_test = tr[symptom]
    y_test = tr[["Herb Name"]]
    y_test = np.ravel(y_test)

    cls = joblib.load('decision_tree_classifier.sav')
    clf = joblib.load('random_forest_classifier.sav')
    knn = joblib.load('knn_classifier.sav')
    log_reg = joblib.load('logistic_reg.sav')
    gnb = joblib.load('naive_bayes_classifier.sav')

    lst=[]
    for i in range(0,len(symptom)):
        lst.append(0)
    
   
    for j in range(0,len(val)):
        for k in range(0,len(symptom)):
            if(val[j] == symptom[k]):
                lst[k] = 1
    
    input_test = [lst]
    
    pred = cls.predict(input_test)
    acc = int(cls.score(X_test,y_test)*100)
    k = np.array(pred).tolist()
    acc = str(acc)+"%"

    pred1 = clf.predict(input_test)
    acc1 = int(clf.score(X_test,y_test)*100)
    k1 = np.array(pred1).tolist()
    acc1 = str(acc1)+"%"

    pred2 = knn.predict(input_test)
    acc2 = int(knn.score(X_test,y_test)*100)
    k2 = np.array(pred2).tolist()
    acc2 = str(acc2)+"%"

    pred3 = log_reg.predict(input_test)
    acc3 = int(log_reg.score(X_test,y_test)*100)
    k3 = np.array(pred3).tolist()
    acc3 = str(acc3)+"%"

    pred4 = gnb.predict(input_test)
    acc4 = int(gnb.score(X_test,y_test)*100)
    k4 = np.array(pred3).tolist()
    acc4 = str(acc4)+"%"

    res = [[herbs[k[0]],acc],[herbs[k1[0]],acc1],[herbs[k2[0]],acc2],[herbs[k3[0]],acc3],[herbs[k4[0]],acc4]]
    return res



