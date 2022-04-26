# m2r README.md
rm -rf grecx.egg-info
rm -rf dist
python setup.py sdist
twine check dist/*
twine upload dist/* --verbose
