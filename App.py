from msilib.schema import Font
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def core():
    global NamesClassifier,AccClassifier,dtc,knn,nb,svm,rfc,logreg
    df=pd.read_csv('Autism_July_18.csv')

    pb['value']+=20
    mainl.config(text="Performing Preprocessing")
    main.update()

    df.drop(['Case_No', 'Who completed the test','Qchat-10-Score','Family_mem_with_ASD','Jaundice','Ethnicity','Sex','Age_Mons'], axis = 1, inplace = True)
    from sklearn.model_selection import train_test_split
    x = df.drop(['Class/ASD Traits '], axis = 1)
    y = df['Class/ASD Traits ']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    pb['value']+=20
    mainl.config(text="Training Models")
    main.update()

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    dtcpred = dtc.predict(x_test)
    dtcScore=accuracy_score(y_test,dtcpred)

    pb['value']+=10
    main.update()

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfcpred = rfc.predict(x_test)
    rfcScore=accuracy_score(y_test,rfcpred)

    pb['value']+=10
    main.update()

    svm = SVC()
    svm.fit(x_train, y_train)
    svmpred = svm.predict(x_test)
    svmScore=accuracy_score(y_test,svmpred)

    pb['value']+=10
    main.update()

    error_rate = []
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        pred_i = knn.predict(x_test)
        error_rate.append(np.mean(pred_i != y_test))
    knnn=error_rate.index(min(error_rate))
    knn = KNeighborsClassifier(n_neighbors=knnn+1)
    knn.fit(x_train, y_train)
    knnpred = knn.predict(x_test)
    knnScore=accuracy_score(y_test,knnpred)

    pb['value']+=10
    main.update()

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logregpred = logreg.predict(x_test)
    logregScore=logreg.score(x_train, y_train)

    pb['value']+=10
    main.update()

    nb = GaussianNB()
    nb.fit(x_train, y_train)
    nbpred = nb.predict(x_test)
    nbScore=accuracy_score(y_test,nbpred)

    pb['value']+=10
    mainl.config(text="Initialising")
    main.update()

    AccClassifier=[i*100 for i in [dtcScore,rfcScore,svmScore,knnScore,logregScore,nbScore]]
    NamesClassifier=['Decision Tree Classifier','Random Forest Classifier','Support Vector Machine','K Nearest Neighbor','Logistic Regression','Naive Bayes']

def task():
    core()
    main.destroy()


main = Tk()
main.title("Loading ... Please Wait !")
mainl = Label(main, text="Loading Dataset")
mainl.grid(row=2,column=1,sticky="news",padx=10,pady=10)
pb = Progressbar(
    main,
    orient='horizontal',
    mode='determinate',
    length=280
)
pb.grid(row=1,column=1,sticky="news",padx=10,pady=10)
main.after(200, task)
main.mainloop()

def endall():
    root.destroy()
    sys.exit(0)

def predictaut():
    global v1,v2,v3,v4,v5,v6,v7,v8,v9
    v=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10]
    for i in range(0,9):
        if int(v[i].get())>=3:
            v[i]=1
        else:
            v[i]=0
    if int(v[9].get())<=3:
        v[9]=1
    else:
        v[9]=0
    v=np.array(v)
    v=v.reshape(1,-1)
    x=str(menu1.get())[0:str(menu1.get()).find(" (")]
    if x=='Decision Tree Classifier':
        ans=dtc.predict(v)
    elif x=='Random Forest Classifier':
        ans=rfc.predict(v)
    elif x=='K Nearest Neighbor':
        ans=knn.predict(v)
    elif x=='Support Vector Machine':
        ans=svm.predict(v)
    elif x=='Logistic Regression':
        ans=logreg.predict(v)
    else:
        ans=nb.predict(v)
    if ans[0]=='Yes':
        messagebox.showerror("Prediction","Your child might be suffering from Autistic Spectrum Disorder. A multi-disciplinary assessment should be performed.")
    else:
        messagebox.showinfo("Prediction","Autism Spectrum Disorder was NOT diagnosed. Your child is healthy")
    
root=Tk()
root.title("Autism Prediction")
BoldFont=("Comic Sans MS", 11, "bold italic")
lx=Label(root,text="For each item, please select the response which best applies to your child",font=BoldFont).grid(row=0,column=1,columnspan=6,padx=10,pady=10)
l1=Label(root,text="Does your child look at you when you call his/her name ?").grid(row=1,column=1,sticky="news",padx=10,pady=10)
l2=Label(root,text="Does your child point to indicate that he/she wants something ?").grid(row=2,column=1,sticky="news",padx=10,pady=10)
l3=Label(root,text="Does your child point to share interest with you ?").grid(row=3,column=1,sticky="news",padx=10,pady=10)
l4=Label(root,text="How easy is it for you to get eye contact with your child ?").grid(row=4,column=1,sticky="news",padx=10,pady=10)
l5=Label(root,text="Does your child pretend ?").grid(row=5,column=1,sticky="news",padx=10,pady=10)
l6=Label(root,text="Does your child follow where you are looking ?").grid(row=6,column=1,sticky="news",padx=10,pady=10)
l7=Label(root,text="If someone is upset, Does your child try to comfort them ?").grid(row=7,column=1,sticky="news",padx=10,pady=10)
l8=Label(root,text="Would you describe your child\'s first words as :").grid(row=8,column=1,sticky="news",padx=10,pady=10)
l9=Label(root,text="Does your child use simple gestures ?").grid(row=9,column=1,sticky="news",padx=10,pady=10)
l10=Label(root,text="Does your child stare at nothing with no apparant purpose ?").grid(row=10,column=1,sticky="news",padx=10,pady=10)
v1 = StringVar(root, "1")
v2 = StringVar(root, "1")
v3 = StringVar(root, "1")
v4 = StringVar(root, "1")
v5 = StringVar(root, "1")
v6 = StringVar(root, "1")
v7 = StringVar(root, "1")
v8 = StringVar(root, "1")
v9 = StringVar(root, "1")
v10 = StringVar(root, "1")
val1 = {"Always" : "1",
        "Usually" : "2",
        "Sometimes" : "3",
        "Rarely" : "4",
        "Never" : "5"}
val2 = {"Very Easy" : "1",
        "Quite Easy" : "2",
        "Quite Difficult" : "3",
        "Very Difficult" : "4",
        "Impossible" : "5"}
val3 = {"Many times a day" : "1",
        "A few times a day" : "2",
        "A few times a week" : "3",
        "Less than once a week" : "4",
        "Never" : "5"}
val4 = {"Very typical" : "1",
        "Quite typical" : "2",
        "Slightly unusual" : "3",
        "Very unusual" : "4",
        "Child doesn't speak" : "5"}
rb=[]
j=0
for (val,v) in [(val1,v1),(val2,v2),(val3,v3),(val3,v4),(val3,v5),(val3,v6),(val1,v7),(val4,v8),(val3,v9),(val3,v10)]:
    i1=0
    for (text, value) in val.items():
        rbt=Radiobutton(root, text = text, variable = v,value = value)
        rbt.grid(row=1+j,column=2+i1,sticky="new",padx=5,pady=10)
        rb.append(rbt)
        i1+=1
    j+=1

separator = Separator(root, orient='horizontal').grid(row=11,column=1,columnspan=6,padx=10,pady=5,sticky="ew")
lx=Label(root,text="Choose Classifier").grid(row=12,column=1,columnspan=1,padx=10,pady=10)
menu1= StringVar()
NamesClassifiers=[]
for p in range(6):
    global NamesClassifier,AccClassifier
    x=NamesClassifier[p]+" ( "+str(round(AccClassifier[p],2))+"% )"
    NamesClassifiers.append(x)
menu1.set(NamesClassifiers[0])
drop1 = OptionMenu(root,menu1,NamesClassifiers[0],*NamesClassifiers).grid(row=12,column=2,columnspan=2,padx=10,pady=10)
b1=Button(root,text="Predict",width=20,command=predictaut)
b1.grid(row=12,column=4,columnspan=2,padx=10,pady=10,sticky="w")
b2=Button(root,text="Exit",width=20,command=endall)
b2.grid(row=12,column=5,columnspan=2,padx=10,pady=20)
root.bind('<Return>', predictaut)
root.mainloop()