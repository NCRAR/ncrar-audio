- [ ] Build using `python -m build .`
- [ ] Upload to test server via twine `twine upload -r testpypi-consult dist/*`
- [ ] Create test env: `conda create -n test python` 
- [ ] Install in test env: 
```
conda activate test
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ncrar_audio
python -m pip install pytest pytest-benchmark
cd <src folder>/tests
pytest
```
- [ ] Verify downloaded version was used: `python -c "import ncrar_audio; print(ncrar_audio.__version__)"`
- [ ] Verify downloaded package was used: `python -c "import ncrar_audio; print(ncrar_audio.__file__)"`
- [ ] Upload to actual server via twine `twine upload -r pypi-consult dist/*`
