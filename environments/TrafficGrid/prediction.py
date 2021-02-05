
import numpy as np
import tensorflow as tf
from environments.TrafficGrid.networks import rnn
from scipy.special import softmax
from environments.TrafficGrid.utility import one_hot, cross_entr, norm_p
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score


class model (object):

    def __init__(self, parameters):

        self.parameters=parameters
        self.seq_len= self.parameters['hor']
        self.learning_rate= self.parameters['learning_rate']
        self.dim_d_set=self.parameters['dim_d_set']
        self.n_classes= self.parameters['n_classes']
        #self.loss_burning=self.parameters['loss_burning']
        self.n_outputs=self.parameters['n_outputs']

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.step = tf.Variable(0, name="global_step",
                                trainable=False, dtype=tf.int32)
            self.increment = tf.assign(self.step,
                                   tf.add(self.step, 1))

            self.build_model()
            self.build_optimizer()
            init=tf.global_variables_initializer()
            self.sess.run(init)


    def build_model(self):

        self.observations = tf.placeholder(shape=[None, self.parameters['hor'], self.dim_d_set],
                                          dtype=tf.float32, name='observations')
        hidden=self.observations
        self.logits=rnn(hidden,self.parameters['n_units'], self.n_classes, self.n_outputs)
        
        self.pred = tf.nn.softmax(self.logits, axis=-1)


    def build_optimizer(self):

        self.targets = tf.placeholder(shape=[None, self.parameters['hor'],self.n_outputs],
                                     dtype=tf.int32, name='inf_source')

        onehot_target = tf.one_hot(self.targets,self.n_classes,axis=-1)
        onehot_target = tf.reshape(onehot_target,[tf.shape(onehot_target)[0]*tf.shape(onehot_target)[1],self.n_outputs, self.n_classes])
        logits=tf.reshape(self.logits,[tf.shape(self.targets)[0]*tf.shape(self.targets)[1],self.n_outputs, self.n_classes])
        self.loss_0=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_target)
        self.loss_1=tf.reduce_mean(self.loss_0,axis=0)
        self.loss=tf.reduce_sum(self.loss_1)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)        
        self.train_op = optimizer.minimize(self.loss)


    def update(self, batch):

        feed_dict = {self.observations: batch['batch_x'],
                     self.targets: batch['labels']}
     

        run_dict = {'loss': self.loss,
                    'loss_1':self.loss_1,
                    'logits': self.pred,
                    'update': self.train_op
                    }
        
        loss_value, loss_1, pred,_ = self.sess.run(list(run_dict.values()),
                                feed_dict=feed_dict)

        
        return loss_value, loss_1, pred


    def predictions(self, dset):

        feed_dict={self.observations: dset}
        run_dict={'logits':self.pred,
                'log':self.logits
                 }

        pred_lab,logits = self.sess.run(list(run_dict.values()),
                                feed_dict=feed_dict)

        pred_lab=np.reshape(pred_lab,[np.shape(dset)[0],self.parameters['hor'],2,2])
        return pred_lab


    def evaluate(self, dset, label):

        feed_dict = {self.observations: dset}

        run_dict = {'logits': self.pred}

        pred_lab = self.sess.run(list(run_dict.values()),
                                feed_dict=feed_dict)

        #label (n_iter, hor) the values are classes
        #pred_lab (1, n_iter x hor, n_classes) the values are the probabilities.

        label=np.array(label)
        one_hot_lab = one_hot(label, self.n_classes)
        
        #one_hot_lab, pred_lab = (n_iter x hor, n_outputs, n_classes) are probabilities.
        one_hot_lab=np.reshape(one_hot_lab, [np.shape(one_hot_lab)[0]*np.shape(one_hot_lab)[1],np.shape(one_hot_lab)[2],np.shape(one_hot_lab)[3]])
        pred_lab = np.reshape(pred_lab, [np.shape(pred_lab)[0]*np.shape(pred_lab)[1],np.shape(pred_lab)[2],np.shape(pred_lab)[3]])
        

        cross_entropy=cross_entr(one_hot_lab,pred_lab)
        norm_1=norm_p(one_hot_lab,pred_lab,1)
        

        return cross_entropy, norm_1

        



