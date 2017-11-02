import pickle
from evaluation_matrics import get_precision_and_recall_and_f1,get_f1_macro
import matplotlib.pyplot as plt


def retrieve_contents(filename):
    content=pickle.load(open(filename,"rb"))
    return content


def next_batch(batch_number,batch_size,num_epoch,task):
    if task=="train":
        NUMBER_OF_SENTENCES=7000
        vectors = retrieve_contents("dataset/padded_train_vectors.p")
        seq_lengths=retrieve_contents("dataset/train_seq_length.p")
        labels=retrieve_contents("dataset/one_hot_labels.p")
    if task=="dev":
        NUMBER_OF_SENTENCES=1000

        vectors = retrieve_contents("dataset/dev_vectors.p")
        labels = retrieve_contents("dataset/dev_labels.p")
        seq_lengths = retrieve_contents("dataset/dev_sequence_length.p")


    if NUMBER_OF_SENTENCES-(batch_number*batch_size) <batch_size:
        batch_number=0
        num_epoch=num_epoch+1

    sentence_batch=vectors[batch_number*batch_size:batch_number*batch_size+batch_size]

    seq_lengths_batch=seq_lengths[batch_number*batch_size:batch_number*batch_size+batch_size]
    label_batch=labels[batch_number*batch_size:batch_number*batch_size+batch_size]

    batch_number=batch_number+1
    return sentence_batch,label_batch,seq_lengths_batch,batch_number,num_epoch



def test_f1_score(model,sess):
    test_batch_number=0
    test_num_epoch=0
    x_test, y_test, seq_len_test, test_batch_number,test_num_epoch = next_batch(test_batch_number, 950,test_num_epoch, "dev")
    pr = sess.run(model.pred, feed_dict={model.sentence_vectors: x_test, model.label_vector: y_test, model.seq_lengths: seq_len_test})
    lb = sess.run(model.max_indices,
                  feed_dict={model.sentence_vectors: x_test, model.label_vector: y_test, model.seq_lengths: seq_len_test})

    precision, recall, f1 = get_precision_and_recall_and_f1(lb, pr)

    return get_f1_macro(f1,len(set(lb)))

def log_f1_score(path,log_text):
    f=open(path,'a')
    f.write(log_text+"\n")

def plot_test_score(test_scores,iterations):
    plt.plot(test_scores,iterations)
    