#!/usr/bin/python3

import os

feats = ['CCA10_AGE3_6_9_12', 'CCA10_12a_AGE6_9_12','CCA10_12a_AGE9_12','CCA10_12a_AGE12','CCA10_12a_AGE9','CCA10_12a_AGE6','CCA10_12a_AGE3',
         'CCA10_AGE3_6_9_12_Cs', 'CCA10_12a_AGE6_9_12_Cs','CCA10_12a_AGE9_12_Cs','CCA10_12a_AGE12_Cs','CCA10_12a_AGE9_Cs','CCA10_12a_AGE6_Cs','CCA10_12a_AGE3_Cs',
         'constants']

outcome_controls = [
    #depression
    ('dep_a12status_constants', 'AnyDepressiveDisorder_includingNOS_15', 'AnyDepressiveDisorder_includingNOS_12 Race_updated_white Race_updated_black Race_updated_asian Race_updated_native_american Ethnicity_updated_hispanic White_nonhispanic_updated_white_non_hispanic Sex_Male CDI_total_age9_child'),
    ('dep_constants_only', 'AnyDepressiveDisorder_includingNOS_15', 'Race_updated_white Race_updated_black Race_updated_asian Race_updated_native_american Ethnicity_updated_hispanic White_nonhispanic_updated_white_non_hispanic Sex_Male CDI_total_age9_child'),
    ('dep_a12status_only', 'AnyDepressiveDisorder_includingNOS_15', 'AnyDepressiveDisorder_includingNOS_12'),

    #anxiety
    ('anx_a12status_constants', 'AnyAnxiety_15', 'AnyAnxiety_12 Race_updated_white Race_updated_black Race_updated_asian Race_updated_native_american Ethnicity_updated_hispanic White_nonhispanic_updated_white_non_hispanic Sex_Male CDI_total_age9_child'),
    ('anx_constants_only', 'AnyAnxiety_15', 'Race_updated_white Race_updated_black Race_updated_asian Race_updated_native_american Ethnicity_updated_hispanic White_nonhispanic_updated_white_non_hispanic Sex_Male CDI_total_age9_child'),
    ('anx_a12status_only', 'AnyAnxiety_15', 'AnyAnxiety_12')]

models = ['lr', 'lrnone', 'etc', 'mlp']

for model in models:
    print("\n\nMODEL:", model, "\n-----------------------\n\n")
    for tup in outcome_controls:
        label, outcome, controls = tup
        for feat in feats:
            command = "./dlatkInterface.py -d klein -c ID -f 'feat$"+feat+"$input_spss_intersect$ID$16to16' --group_freq_thresh 0 --outcome_table total_outcomes --outcomes "+outcome+' --controls '+controls+' --all_controls_only --combo_test_classif --model '+model+' --folds 10 --stratify --csv --prob_csv --pred_csv --output_name /data/klein_adol_dep/FALL_2021/'+model+'.'+label+'.'+feat
            print("\nRUNNING:", command)
            os.system(command)

