from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
from dlatk.featureGetter import FeatureGetter
import time
import csv





JSON = "AimAheadTests.json"
MODEL_NAME = 'ridgecv'
DATABASE = 'ai_fairness'
TABLE = 'msgs_100u'
CORREL_FIELD = 'who'
OUTCOME_TABLE = "aim_ahead_binary_outcomes"
OUTCOME_FIELDS = ["Ab_krupitskyA_2011"]
OUTCOME_CONTROLS = []#["Ab_krupitskyA_2011"]
GROUP_FREQ_THRESH = 0
FEATURE_TABLES = ['aim_ahead_master_feats']





def main():

    fgs = [FeatureGetter(corpdb = DATABASE, corptable = TABLE, correl_field=CORREL_FIELD, featureTable=featTable, featNames="", wordTable = None) for featTable in FEATURE_TABLES]
    
    available_outcomes = [
        'ctn0094_relapse_event', 'Ab_krupitskyA_2011', 'Ab_ling_1998',
        'Rs_johnson_1992', 'Rs_krupitsky_2004', 'Rd_kostenB_1993'
    ]
    

    for outcome in available_outcomes:
        OUTCOME_FIELDS = [outcome]
        og = OutcomeGetter(corpdb = DATABASE, corptable = TABLE, correl_field=CORREL_FIELD, outcome_table=OUTCOME_TABLE, outcome_value_fields=OUTCOME_FIELDS, outcome_controls=OUTCOME_CONTROLS, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh = GROUP_FREQ_THRESH)
        cps = []
        cps.append(ClassifyPredictor(og, fgs, 'lr', None, None)) # {'C':[0.01], 'dual':[False]}
        cps.append(ClassifyPredictor(og, fgs, 'lrl1', None, None)) # {'C':[0.01, 0.1, 0.001, 1, .0001, 10], 'penalty':['l1'], 'dual':[False]},
        cps.append(ClassifyPredictor(og, fgs, 'linear-svc', None, None)) # {'C':[0.01, 0.1, 0.001, 1, .0001, 10], 'penalty':['l1'], 'dual':[False], 'class_weight':['balanced']}
        cps.append(ClassifyPredictor(og, fgs, 'linear-svcl2', None, None)) # {'C':[0.01, 0.1, 0.001, 1, .0001, 10], 'penalty':['l2'], 'dual':[False], 'class_weight':['balanced']}
        
        for cp in cps:
            start_time = time.time()
            scoresRaw = cp.testControlCombos(comboSizes = [], nFolds = 10, allControlsOnly=True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            outputStream = open("original_print" + JSON.replace(".json", ".csv"), 'a')
            csv_writer = csv.writer(outputStream)
            csv_writer.writerow(["Aim_Ahead_Test " + cp.modelName, "Execution Time (seconds)", elapsed_time])
            RegressionPredictor.printComboControlScoresToCSV(scoresRaw, outputStream)
            outputStream.close()





if __name__ == "__main__":
    main()