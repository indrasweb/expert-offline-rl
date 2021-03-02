from setuptools import setup, find_packages

setup(
  name = 'eorl',
  packages = find_packages(exclude=['dataset-collection']),
  version = '0.0.1',
  license='MIT',
  description = 'Expert Offline RL Datasets',
  author = 'Harry Songhurst',
  author_email = 'harrysonghurst@gmail.com',
  url = 'https://github.com/indrasweb/expert-offline-rl',
  keywords = ['self-supervised learning', 'contrastive learning', 'imitation learning', 'reinforcement learning'],
  include_package_data=True,
  install_requires=[
      'gsutil',
      'numpy'
  ],
  classifiers=[
      'Development Status :: Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.7.9',
  ],
)