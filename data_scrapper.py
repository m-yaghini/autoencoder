import wikipediaapi


class WikipediaArticleSet(object):
    '''
    An article set is a collection of Wikipedia articles collected on categories given to the
    constructor. After creation, calling a WikipediaArticleSet with an integer n will give a set of n pages
    with features (content of pages) and corresponding integer labels (category ids, with a object
    dictionary label2cat for converstion to the original category).
    '''

    def __init__(self, categories, language='en'):
        self.categories = categories
        self.label_set = self._assign_code_ids(categories)
        self.cat2label = {key: val for key, val in zip(self.categories, self.label_set)}
        self.label2cat = {val: key for key, val in self.cat2label.items()}
        self.language = language

    @staticmethod
    def _assign_code_ids(categories):
        return range(len(categories))

    def _get_pages(self, category, num=10):

        def _get_enough_articles(categorymembers, pages_content_list, pages_title_list, valid_article_count):
            _valid_article_count = valid_article_count
            _pages_content_list = pages_content_list
            _pages_title_list = pages_title_list
            for c in categorymembers.values():
                if _valid_article_count < num:
                    if (c.ns == wikipediaapi.Namespace.CATEGORY):
                        _pages_content_list, _pages_title_list, _valid_article_count = \
                            _get_enough_articles(c.categorymembers, _pages_content_list, _pages_title_list,
                                                 _valid_article_count)
                    else:
                        if c.text:  # non-empty text
                            _pages_content_list.append(c.text)
                            _pages_title_list.append(c.title)
                            _valid_article_count += 1
                            print('{}. {}'.format(_valid_article_count, c.title))
                        else:
                            continue
                else:
                    return _pages_content_list, _pages_title_list, _valid_article_count

        wp = wikipediaapi.Wikipedia(self.language)
        cat = wp.page("Category:" + category)
        pages_content_list, pages_title_list, _ = _get_enough_articles(cat.categorymembers, [], [], 0)
        pages_label_list = [self.cat2label[category]] * num
        return pages_content_list, pages_label_list, pages_title_list


    def extract_labels(self, numof_articles_per_class):
        labels = []
        features = []
        for ind, cat in enumerate(self.categories):
            print('=== Progress: {0:.2f}%'.format(100 * ind / float(len(self.categories))))
            print('Extracting articles for category "{}"'.format(cat))
            _content, _label, _ = self._get_pages(cat, num=numof_articles_per_class)
            labels += _label
            features += _content
        print('=== Progress: 100% \n Done!')
        return features, labels


    def __call__(self, numof_articles_per_class):
        return self.extract_labels(numof_articles_per_class)
