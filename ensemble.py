import pickle
import numpy as np
import os

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.alphas = np.zeros(self.n_weakers_limit)

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        num = X.shape[0]    # 样本数量
        w = np.ones(num)/num
        if not os.path.exists('base_classifier'):
            os.mkdir('base_classifier')
        for i in range(self.n_weakers_limit):
            print('training base classifier %d/%d' % (i+1,self.n_weakers_limit))
            # Configure Decision tree
            b_learner = self.weak_classifier(random_state=0, max_depth=2)
            # Build Decision tree according training data
            b_learner.fit(X, y, sample_weight=w)
            # Save Model to a file in order to use it next time without build model step
            self.save(b_learner,'base_classifier/base_classifier_%d.pkl' % i)

            pre_y = b_learner.predict(X)

            # 计算分类误差率
            errorVector = np.zeros(num)
            errorVector[pre_y != y] = 1
            errorRate = np.dot(errorVector,w)
            if errorRate > 0.5:
                break
            alpha = 0.5 * np.log(( 1 - errorRate ) / errorRate)
            self.alphas[i] = alpha
            w = w * np.exp( - alpha * y * pre_y )
            w = w / w.sum()

        print('training finish')

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pre_ys = []
        for i in range(self.n_weakers_limit):
            b_learner = self.load('base_classifier/base_classifier_%d.pkl' % i)
            pre_y = b_learner.predict(X)
            pre_ys.append(pre_y)
        pre_ys = np.array(pre_ys)
        final_pre_y = np.dot(self.alphas, pre_ys)
        return final_pre_y

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        print('predicting validation datasets...')
        final_pre_y = self.predict_scores(X)
        final_pre_y = np.array(list(map((lambda x: 1 if x >= threshold else -1), final_pre_y)))
        print('predicted results:', final_pre_y)
        return final_pre_y

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
