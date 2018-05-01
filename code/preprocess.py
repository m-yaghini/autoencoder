import numpy as np


def preprocess_documents_matrix(docs_mat):
    '''
    preprocess raw articles, in the form of a numpy array of strings (documents) or a list of strings.
    # partially adopted from gensim documentation
    :param docs_mat: numpy vector of texts.
    :return: cleaned and tokenized list of texts
    '''

    from string import punctuation
    punctuation = set(punctuation)

    docs_list = docs_mat.tolist()

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())

    texts = [[''.join(ch for ch in word if ch not in punctuation)
              for word in (document.lower().split())
              if word not in stoplist]
             for document in docs_list]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    return texts


def docs_vocab_freq_matrix(docs, vocab):
    '''
    processes the documents into frequent distributions over the words of the vocabulary.
    :param docs: (list) list of tokenized documents.
    :param vocab: (list) list of vocabulary words
    :return: a matrix of distribution of word frequencies for documents
    '''
    from collections import Counter

    docs_freq_distribution = []
    for doc in docs:
        word_freq = Counter()
        word_freq.update(doc)
        doc_freq_dist = [word_freq[word] for word in vocab]
        docs_freq_distribution.append(doc_freq_dist)
    return docs_freq_distribution


def docs_embeddings_mat(en_model, docs, labels):
    '''
    processes the document into mean feature vectors based on word embeddings.
    Handles the exceptions when there is no mean of feature vectors (happens when no word in the article is found in the
    word embeddings).
    :param en_model: a loaded Gensim set of word embeddings for the language.
    :param docs: a list of tokenized documents.
    :param labels: a list of labels of docs.
    :return: a matrix of mean feature vectors for documents.
    '''
    docs_embeddings_list = []
    docs_labels_list = []
    for doc_ind, doc in enumerate(docs):
        word_embeddings_list = []
        for word in doc:
            try:
                word_embedding = en_model.get_vector(word)
                word_embeddings_list.append(word_embedding)
            except KeyError:
                continue
        try:
            doc_embedding = np.mean(word_embeddings_list, axis=0)
            docs_embeddings_list.append(doc_embedding)
            docs_labels_list.append(labels[doc_ind])
        except FloatingPointError:
            continue
    docs_embeddings_mat = np.vstack(docs_embeddings_list)
    docs_labels = np.hstack(docs_labels_list)
    return docs_embeddings_mat, docs_labels
