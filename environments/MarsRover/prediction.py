
import tensorflow as tf
from environments.MarsRover.networks import rnn
from scipy.special import softmax
from environments.MarsRover.utility import one_hot, cross_entr, norm_p
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score
import numpy as np

class model (object):

    def __init__(self, parameters):

        self.parameters=parameters
        self.seq_len= self.parameters['hor']
        self.learning_rate= self.parameters['learning_rate']
        self.dim_d_set=self.parameters['dim_d_set']
        self.n_classes= self.parameters['n_classes']
        self.loss_burning=self.parameters['loss_burning']

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
        self.logits=rnn(hidden,self.parameters['n_units'], self.n_classes)
        
        self.pred = tf.nn.softmax(self.logits, axis=-1)


    def build_optimizer(self):

        self.targets = tf.placeholder(shape=[None, self.parameters['hor']],
                                     dtype=tf.int32, name='inf_source')

        onehot_target = tf.one_hot(self.targets,self.n_classes,axis=-1)
        onehot_target = tf.reshape(onehot_target,[-1,self.n_classes])
        
        pred_1=tf.reshape(self.logits,[tf.shape(self.targets)[0],self.parameters['hor'],self.n_classes])
        pred_2=pred_1[:, self.loss_burning:,:]
        self.burnt_pred=tf.reshape(pred_2,[tf.shape(self.targets)[0]*tf.shape(pred_2)[1],self.n_classes])

        onehot_target_1=tf.reshape(onehot_target,[tf.shape(self.targets)[0],self.parameters['hor'],self.n_classes])
        onehot_target_2=onehot_target_1[:, self.loss_burning:,:]
        self.burnt_onehot_target=tf.reshape(onehot_target_2,[tf.shape(self.targets)[0]*tf.shape(onehot_target_2)[1],self.n_classes])
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.burnt_pred, labels=self.burnt_onehot_target))
        
        #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.burnt_onehot_target* tf.log(self.pred), reduction_indices=[1]))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        self.train_op = optimizer.minimize(self.loss)


    def update(self, batch):

        feed_dict = {self.observations: batch['batch_x'],
                     self.targets: batch['labels']}
        
        run_dict = {'loss': self.loss,
                    'logits': self.pred,
                    'update': self.train_op
                    }

        loss_value, pred,_ = self.sess.run(list(run_dict.values()),
                                feed_dict=feed_dict)
        
        return loss_value, pred


    def predictions(self, dset):

        feed_dict={self.observations: dset}
        run_dict={'logits':self.pred}

        pred_lab = self.sess.run(list(run_dict.values()),
                                feed_dict=feed_dict)
        
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
        pred_lab = np.reshape(pred_lab, [np.shape(one_hot_lab)[0], np.shape(one_hot_lab)[1], np.shape(pred_lab)[-1]])
        #one_hot_lab (n_iter,hor, n_classes) the values are probability (either 0 or 1)
        #pred_lab is reshaped (n_iter, hor, n_classes) the values are the probabilities.


        ''' 
        #Excluding the steps that are not used to train.. why?
        one_hot_lab = one_hot_lab[:,self.loss_burning:,:]
        pred_lab = pred_lab[:,self.loss_burning:,:]
        '''

        #one_hot_lab, pred_lab = (n_iter x hor, n_classes) are probabilities.
        one_hot_lab=np.reshape(one_hot_lab, [np.shape(one_hot_lab)[0]*np.shape(one_hot_lab)[1],np.shape(one_hot_lab)[-1]])
        pred_lab = np.reshape(pred_lab, [np.shape(pred_lab)[0]*np.shape(pred_lab)[1],np.shape(pred_lab)[-1]])

        cross_entropy=cross_entr(one_hot_lab,pred_lab)
        norm_1=norm_p(one_hot_lab,pred_lab,1)


        return cross_entropy, norm_1

        



