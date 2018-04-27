from wikidata_scrapper import WikipediaArticleSet
from numpy import savez_compressed

#
# categories = ['Culture', 'Geography', 'Health', 'History', 'Mathematics', 'Science', 'Philosophy',
#               'Technology', 'Society', 'Religion']

categories2depth = {'Culture': 1, 'Geography': 2, 'Health': 1, 'History': 1, 'Mathematics': 1, 'Science': 1,
                    'Philosophy': 2,
                    'Technology': 1, 'Society': 1, 'Religion': 2}

article_set = WikipediaArticleSet(categories2depth)
X, y = article_set(1000)
savez_compressed('./data_10class_1000perClass_2', X=X, y=y)