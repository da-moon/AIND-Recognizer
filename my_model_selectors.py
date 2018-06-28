import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """
    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scores = []
        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                d = model.n_features
                p = num_states ** 2 + 2 * d * num_states - 1
                result = -2.0 * logL + p * logN
                scores.append([result, model])
            except:
                pass
        if scores :
            minimum = min(scores, key = lambda x : x[0])
            return minimum[1] 
        else :
            return self.n_constant        

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scores = []
        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_states)
                scores_temp = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        scores_temp.append(model.score(X, lengths))
                result = [model.score(self.X, self.lengths) - np.mean(scores_temp), model]
                scores.append(result)
            except:
                pass
        if scores :
            maximum = max(scores, key = lambda x : x[0])
            return maximum[1] 
        else :
            return self.n_constant

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split_method = KFold()
        likelihoods = []
        scores = []
        for num_states in range(self.min_n_components, self.max_n_components+1):        
            try:
                if len(self.sequences)> 2:
                    for train_index, test_index in split_method.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)
                        model = self.base_model(num_states)
                        likelihood = model.score(X_test, lengths_test)
                        likelihoods.append(likelihood)
                else:
                    model = self.base_model(num_states)
                    likelihoods = model.score(self.X, self.lengths)
                scores.append([np.mean(likelihoods), model])
            except:
                pass
        if scores :
            maximum = max(scores, key = lambda x : x[0])
            return maximum[1] 
        else :
            return self.n_constant