REM python nist_base_partial.py rotate
REM python nist_drift_classifier_2.py flip
python nist_drift_classifier_2.py appear
python nist_drift_classifier_2.py remap
python nist_drift_classifier_2.py transfer
python nist_drift_classifier_2.py rotate

python nist_base.py

python nist_model_freezing.py flip 3
python nist_model_freezing.py flip 2
python nist_model_freezing.py flip 1

python nist_model_freezing.py rotate 3
python nist_model_freezing.py rotate 2
python nist_model_freezing.py rotate 1

python nist_model_freezing.py appear 3
python nist_model_freezing.py appear 2
python nist_model_freezing.py appear 1

python nist_model_freezing.py remap 3
python nist_model_freezing.py remap 2
python nist_model_freezing.py remap 1

python nist_model_freezing.py transfer 3
python nist_model_freezing.py transfer 2
python nist_model_freezing.py transfer 1