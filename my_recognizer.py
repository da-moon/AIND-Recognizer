import warnings
from asl_data import SinglesData
def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
        :param models:  dict of trained models
                        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
        :param test_set: SinglesData object
        :return:(list, list)    as probabilities, guesses
                                both lists are ordered by the test set word_id
                                probabilities is a list of dictionaries where each key a word and value is Log Liklihood
                                [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
                                {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
                                ]
                                guesses is a list of the best guess words ordered by the test set word_id
                                ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for X, lengths in test_set.get_all_Xlengths().values():
        best_score = float('-Inf') 
        best_guess = None  
        likelihoods = {}  
        for word, model in models.items():
            try:
                likelihoods[word] = model.score(X, lengths)
                if likelihoods[word] > best_score:
                    best_guess, best_score = word, likelihoods[word]
            except:
                likelihoods[word] = float('-Inf')
        probabilities.append(likelihoods)
        guesses.append(best_guess)
    return probabilities, guesses
