import tensorflow as tf
from utils import next_batch,test_f1_score,log_f1_score,plot_test_score
from evaluation_matrics import get_precision_and_recall_and_f1,get_f1_macro
#import your Model
from class_model import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(resume_training=False,total_epochs=5,test_after_n_iter=0,disp_loss_after_n_iter=0,merge_summary_after_n_iter=0,hyperparameters=None):
    tf.reset_default_graph()
    #initialize the model
    model=Model(num_units=hyperparameters.num_units,embedding_size=hyperparameters.embedding_size,
                learning_rate=hyperparameters.learning_rate,batch_size=hyperparameters.batch_size,
                )



    #saver
    saver=tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        if resume_training==True:
            saver.restore(sess,"saved_model/model.ckpt")
        else:
            sess.run(init)


        merge_summary = tf.summary.merge_all()
        tensorboard_path = "tensorboard_analysis/" + str(hyperparameters.__dict__)
        writer = tf.summary.FileWriter(tensorboard_path)
        writer.add_graph(sess.graph)
        batch_number = 0
        iter = 1
        num_epoch=0
        test_score=[]
        iterations=[]
        while num_epoch < total_epochs:

            ######################
            ##train logic goes here
            prev_epoch = num_epoch
            x, y, seq, batch_number, num_epoch = next_batch(batch_number, model.batch_size, num_epoch, "train")
            """""
            if num_epoch != 0 and prev_epoch != num_epoch and num_epoch % test_after_n_epoch == 0:
                print("The test f1 score after {} epoch is {}".format(num_epoch, test_f1_score(model, sess)))
            """""
            lb = sess.run(model.max_indices,
                          feed_dict={model.sentence_vectors: x, model.label_vector: y, model.seq_lengths: seq})
            pr = sess.run(model.pred,
                          feed_dict={model.sentence_vectors: x, model.label_vector: y, model.seq_lengths: seq})
            precision, recall, f1 = get_precision_and_recall_and_f1(lb, pr)
            f1_macro = get_f1_macro(f1, len(set(lb)))
            sess.run(model.opt, feed_dict={model.sentence_vectors: x, model.label_vector: y, model.seq_lengths: seq})

            ###############

            if iter % disp_loss_after_n_iter == 0:
                los = sess.run(model.loss, feed_dict={model.sentence_vectors: x, model.label_vector: y, model.seq_lengths: seq})


                print("For iter ", iter)
                print("F1 score ",f1_macro)
                print("Loss ", los)
                print("__________________")

            if iter % test_after_n_iter==0 or iter==1:

                f1_score_test=test_f1_score(model,sess)
                test_score.append(f1_score_test)
                iterations.append(iter)
                plot_test_score(test_score,iterations)


            if iter % merge_summary_after_n_iter:

                s = sess.run(merge_summary,
                             feed_dict={model.sentence_vectors: x, model.label_vector: y, model.seq_lengths: seq})
                writer.add_summary(s, iter)




            iter = iter + 1


        final_test_score=test_f1_score(model,sess)
        log_text="The final f1 score for configuration {} is {} after {} epochs and {} iterations".format(str(hyperparameters.__dict__),final_test_score,total_epochs,iter)
        logging_path="tensorboard_analysis/"+str(hyperparameters.__dict__)+"/logfile.txt"
        plot_path="tensorboard_analysis/"+str(hyperparameters.__dict__)+"/test_plot.png"
        log_f1_score(logging_path,log_text)


        save_path="saved_model/"+str(hyperparameters.__dict__)+"/model.ckpt"
        saver.save(sess,save_path)
        tf.Session.close(sess)
        plt.savefig(plot_path)









