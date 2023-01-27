import pkgutil
import importlib
import inspect

import gluonts.model

SUBMODULES = [pkg.name for pkg in pkgutil.walk_packages(gluonts.model.__path__, gluonts.model.__name__+'.') if pkg.ispkg]

ESTIMATORS = {}
PREDICTORS = {}

for subm in SUBMODULES:
    try:
        imported = importlib.import_module(subm)
    except ModuleNotFoundError:
        print('Could not load methods from', subm)
    members = inspect.getmembers(imported, inspect.isclass)
    for est, est_cls in members:
        if 'estimator' in est.lower():
            name = est.lower().replace('estimator', '')
            if name in ESTIMATORS:
                print('Already found a method for', est)
            ESTIMATORS[name] = est_cls
        if 'predictor' in est.lower():
            name = est.lower().replace('predictor', '')
            if name in PREDICTORS:
                print('Already found a method for', est)
            PREDICTORS[name] = est_cls

print('\nESTIMATORS')
for key in ESTIMATORS.keys():
    print(key)

print('\nPREDICTORS')
for key in PREDICTORS.keys():
    print(key)

