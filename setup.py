from setuptools import setup

setup(name='arxivtools',
      version='0.1',
      description='tools to get word embedding from arxiv.org abstracts',
      url='https://github.com/quynhneo/detm-arxiv',
      author='Quynh M. Nguyen',
      author_email='qmn203@nyu.edu',
      license='MIT',
      packages=['arxivtools'],
      install_requires=['nltk', 'gensim'],
      zip_safe=False)
