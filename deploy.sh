# m2r README.md
rm -rf grecx.egg-info
rm -rf dist
python setup.py sdist
twine upload dist/* --verbose
