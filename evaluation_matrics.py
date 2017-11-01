import numpy as np
#labels=     [1,2,1,3,1,4,5,3,6]
#predictions=[1,2,3,4,5,6,7,8,9]


#classes=[1,2,3,4,5,6,7,8,9]

from sklearn.metrics import f1_score



def find_pos_and_neg(labels,predictions):
    num_classes=len(set(labels))
    batch_size=len(predictions)
    tp = {}
    tn = {}
    fp = {}
    fn = {}
    for i in range(1,num_classes+1):
        for j in range(0,batch_size):
            if predictions[j]==i:
                if labels[j]==i:
                    if i in tp.keys():
                        tp[i] = tp[i] + 1
                    else:
                        tp[i] = 1
                else:
                    if i in fp.keys():
                        fp[i] = fp[i] + 1
                    else:
                        fp[i] = 1
            else:
                if labels[j]==i:
                    if i in fn.keys():
                        fn[i] = fn[i] + 1
                    else:
                        fn[i] = 1
                else:
                    if i in tn.keys():
                        tn[i] = tn[i] + 1
                    else:
                        tn[i] = 1
    return tp,tn,fp,fn



def get_precision_and_recall_and_f1(labels,predictions):
    num_classes = len(set(predictions))
    precision={}
    recall={}
    f1={}
    tp,tn,fp,fn=find_pos_and_neg(labels,predictions)

    for i in range(1,num_classes+1):
        if i in tp.keys():
            pass
        else:
            tp[i]=0


        if i in fp.keys():
            pass
        else:
            fp[i]=0
        
        if i in fn.keys():
            pass
        else:
            fn[i]=0

        if tp[i] + fp[i] != 0:
            precision[i]=tp[i]/(tp[i]+fp[i])
        else:
            precision[i]=0


        if tp[i] + fn[i] != 0:
            recall[i]=tp[i]/(tp[i]+fn[i])
        else:
            recall[i]=0

        if precision[i]+recall[i]!=0:
            f1[i]=(2*precision[i]*recall[i])/(precision[i]+recall[i])
        else:
            f1[i]=0

    return precision,recall,f1
    
    
def get_f1_macro(f1,classes):

    sc=0
    for cls,score in f1.items():
        sc=sc+score
        #classes=classes+1

    f1_net=sc/classes

    return f1_net
    



"""""
a={1:3,2:5,3:6}
b={1:4,2:6}
c={}
r={}
for i in range(1,num_classes+1):
    if i in a.keys():
        continue
    else:
        a[i]=0

    if i in b.keys():
        continue
    else:
        b[i]=0

    c[i]=tp[i]/(tp[i]+fp[i])

    r[i]=tp[i]/(tp[i]+fn[i])

"""""



    

