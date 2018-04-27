
class ArtcileClassifier(object):

    @staticmethod
    def preprocess_documents_matrix(docs_mat):
        # paritally adopted from gensim documentation

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


