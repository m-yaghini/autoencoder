import pywikibot
from pywikibot import pagegenerators
import mwparserfromhell as mw

site = pywikibot.Site()


class WikipediaArticleSet(object):
    '''
    An article set is a collection of Wikipedia articles collected on categories given to the
    constructor. After creation, calling a WikipediaArticleSet with an integer n will give a set of n pages
    with features (content of pages) and corresponding integer labels (category ids, with a object
    dictionary label2cat for converstion to the original category).
    '''

    def __init__(self, category2search_depth_mapping, language='en'):
        self.categories = category2search_depth_mapping.keys()
        self.label_set = self._assign_code_ids(self.categories)
        self.cat2label = {key: val for key, val in zip(self.categories, self.label_set)}
        self.label2cat = {val: key for key, val in self.cat2label.items()}
        self.language = language
        self.categories2depth = category2search_depth_mapping

    @staticmethod
    def _assign_code_ids(categories):
        return range(len(categories))

    def _get_pages(self, cat, num=10):
        titles = []
        texts = []
        py_cat = pywikibot.Category(site, 'Category:' + cat)
        gen = pagegenerators.CategorizedPageGenerator(py_cat, recurse=self.categories2depth[cat])

        for page in gen:
            if len(titles) < num:
                _title = page.title()
                if ('Portal' in _title or 'Outline' in _title
                        or 'Category' in _title or 'List' in _title or 'Index' in _title):  # some filter words
                    continue
                else:
                    parsed_text = mw.parse(page.text)
                    texts.append(parsed_text.strip_code())
                    titles.append(page.title())
                    print(_title)
            else:
                labels = [self.cat2label[cat]] * num
                return texts, labels, titles

    def extract_labels(self, numof_articles_per_class):
        '''
        Gives back article features (X) and category labels (y) for len(self.categories) classes of data. Each with
        numof_articles_per_class articles.
        :param numof_articles_per_class: self-explanatory
        :return: features, labels
        '''
        labels = []
        features = []
        for ind, cat in enumerate(self.categories):
            print('=== Progress: {0:.2f}%'.format(100 * ind / float(len(self.categories))))
            print('Extracting articles for category "{}"'.format(cat))
            try:
                _texts, _labels, _ = self._get_pages(cat, num=numof_articles_per_class)
                labels += _labels
                features += _texts
            except TypeError:
                print("=== Error in {}".format(cat))
                continue
        print('=== Progress: 100% \n Done!')
        return features, labels

    def __call__(self, numof_articles_per_class):
        return self.extract_labels(numof_articles_per_class)
