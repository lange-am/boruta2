from setuptools import setup

setup(name='Boruta2',
      version='0.0.1',
      description='Python Implementation of Boruta2 (improved Boruta) all-relevant Feature Selection',
      url='https://github.com/lange-am/boruta2',
      author='Daniel Homola, Andrey Lange',
      author_email='lange_am@mail.ru',
      license='BSD 3 clause',
      packages=['boruta2', 'boruta2/test'],
      package_dir={'boruta2': 'boruta2'},
      package_data={'boruta2/examples/*csv': ['boruta2/examples/*.csv']},
      include_package_data=True,
      keywords=['feature selection', 'machine learning', 'random forest'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'scipy>=0.17.0'
                        ])
