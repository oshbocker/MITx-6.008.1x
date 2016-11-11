import sys
import os.path
import numpy as np
from collections import Counter
from scipy.misc import logsumexp

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    word_count = Counter()
    for file in file_list:
        for word in set(util.get_words_in_file(file)):
            word_count[word] += 1
            
    return word_count

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    word_counts = get_counts(file_list)
    log_probs = Counter()
    num_files = len(file_list)
    
    for word in word_counts.keys():
        log_probs[word] = np.log((word_counts[word]+1)/(num_files+2))

    return log_probs


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    spam_list = file_lists_by_category[0]
    ham_list = file_lists_by_category[1]
    spam_probs = get_log_probabilities(spam_list)
    ham_probs = get_log_probabilities(ham_list)
    log_probabilities_by_category = [spam_probs, ham_probs]
    
    num_spam = len(spam_list)
    num_ham = len(ham_list)
    num_total = num_spam + num_ham
    
    prob_spam = np.log(num_spam/num_total)
    prob_ham = np.log(num_ham/num_total)
    
    log_prior = [prob_spam, prob_ham]
    return (log_probabilities_by_category, log_prior)


def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    prob_spam = log_prior_by_category[0]
    prob_ham = log_prior_by_category[1]
    spam_probs = log_probabilities_by_category[0]
    ham_probs = log_probabilities_by_category[1]
    
    spam_probs_cond = 0
    ham_probs_cond = 0
    
    # For each word in the email get the prob of it appearing in spam or ham
    # If the word was not in the training dataset, assign it the 1/(# of words in relevant dictionary + 2)
    email_words = util.get_words_in_file(email_filename)
    for word in email_words:
        if word not in spam_probs.keys():
            spam_probs_cond += -np.log(len(spam_probs.keys())+2)
        else:
            spam_probs_cond += spam_probs[word]
        if word not in ham_probs.keys():
            ham_probs_cond += -np.log(len(ham_probs.keys())+2)
        else:
            ham_probs_cond += ham_probs[word]
    
    # Using Bayes' Theorem calculate the log probability of spam and ham        
    spam = (prob_spam+spam_probs_cond)/logsumexp([prob_spam+spam_probs_cond,prob_ham+ham_probs_cond])
    ham = (prob_ham+ham_probs_cond)/logsumexp([prob_spam+spam_probs_cond,prob_ham+ham_probs_cond])
     
    # If the prob of spam is higher return spam o.w. ham 
    if spam/ham > 1:
        return 'ham'
    else:
        return 'spam'



def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
