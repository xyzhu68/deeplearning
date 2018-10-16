REM python nist_drift_classifier_2.py flip
REM python nist_drift_classifier_2.py rotate
REM python nist_drift_classifier_2.py appear
REM python nist_drift_classifier_2.py remap
REM python nist_drift_classifier_2.py transfer

REM python nist_drift_classifier_2.py flip 5
REM python nist_drift_classifier_2.py flip 2
REM python nist_drift_classifier_2.py flip 1

REM python nist_drift_classifier_2.py rotate 5
REM python nist_drift_classifier_2.py rotate 2
REM python nist_drift_classifier_2.py rotate 1

REM python nist_drift_classifier_2.py appear 5
REM python nist_drift_classifier_2.py appear 2
REM python nist_drift_classifier_2.py appear 1

python nist_drift_classifier_2.py remap 5
python nist_drift_classifier_2.py remap 2
python nist_drift_classifier_2.py remap 1

REM python nist_drift_classifier_2.py transfer 5
REM python nist_drift_classifier_2.py transfer 2
REM python nist_drift_classifier_2.py transfer 1

python nist_model_freezing_no_patching.py flip 5
python nist_model_freezing_no_patching.py flip 2
python nist_model_freezing_no_patching.py flip 1
python nist_model_freezing_no_patching.py flip 0

python nist_model_freezing_no_patching.py rotate 5
python nist_model_freezing_no_patching.py rotate 2
python nist_model_freezing_no_patching.py rotate 1
python nist_model_freezing_no_patching.py rotate 0

python nist_model_freezing_no_patching.py appear 5
python nist_model_freezing_no_patching.py appear 2
python nist_model_freezing_no_patching.py appear 1
python nist_model_freezing_no_patching.py appear 0

python nist_model_freezing_no_patching.py remap 5
python nist_model_freezing_no_patching.py remap 2
python nist_model_freezing_no_patching.py remap 1
python nist_model_freezing_no_patching.py remap 0

python nist_model_freezing_no_patching.py transfer 5
python nist_model_freezing_no_patching.py transfer 2
python nist_model_freezing_no_patching.py transfer 1
python nist_model_freezing_no_patching.py transfer 0


setlocal
cd ..\mnist
python mnist_drift_classifier_2.py flip
python mnist_drift_classifier_2.py rotate
python mnist_drift_classifier_2.py appear
python mnist_drift_classifier_2.py remap
python mnist_drift_classifier_2.py transfer
endlocal