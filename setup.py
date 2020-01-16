from distutils.core import setup
setup(
  name='probability',
  packages=['probability'],
  version='0.0.2',
  license='MIT',
  description='Probability in Python 3',
  author='Vahndi Minah',
  url='https://github.com/vahndi/probability',
  download_url='https://github.com/vahndi/probability/archive/v_0.0.1.tar.gz',
  keywords=['python', 'probability'],
  install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)