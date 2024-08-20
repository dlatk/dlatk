from dlatk.featureGetter import FeatureGetter
from dlatk.database.dataEngine import DataEngine
from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
import dlatk.dlaConstants as dlac
#import ResultHandler as rh
import csv
import subprocess
import multiprocessing
from abc import ABC, abstractmethod








#python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$1gram$msgs_100u$cnty$16to16' 'feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$16to16' --fit_reducer --model pca --transform_to_feats cnty_1gram_topic_reduced100 --n_components 100


#python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$1gram$msgs_100u$cnty$16to16' --fit_reducer --model pca --transform_to_feats cnty_1gram_topic_reduced100 --n_components 100
#python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$16to16' --fit_reducer --model pca --transform_to_feats topic_reduced100 --n_components 100


'''
CREATE TABLE `feat$1gram_topics_combined$cnty$16to16` AS
SELECT * FROM `feat$1gram$msgs_100u$cnty$16to16`
UNION ALL
SELECT group_id, feat, value, group_norm FROM `feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$16to16`;


python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$1gram$msgs_100u$cnty$16to16' --feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 500


FA test
python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$dr_pca_cnty_1gram_reduced100$msgs_100u$cnty' 'feat$dr_pca_topic_reduced100$msgs_100u$cnty' --outcome_table combined_county_outcomes  --group_freq_thresh 0 --outcomes heart_disease suicide life_satisfaction perc_fair_poor_health --output_name ai_fairness_output  --nfold_test_regression --factor_adaptation --adaptation_factor logincomeHC01_VC85ACS3yr$10 hsgradHC03_VC93ACS3yr$10 --model ridgecv --folds 10 --csv


RFA test
python3.5 dlatkInterface.py -d ai_fairness -t msgs_100u -c cnty -f 'feat$dr_pca_cnty_1gram_reduced100$msgs_100u$cnty' 'feat$dr_pca_topic_reduced100$msgs_100u$cnty' --outcome_table combined_county_outcomes  --group_freq_thresh 0 --outcomes heart_disease suicide life_satisfaction perc_fair_poor_health --output_name ai_fairness_RFA_output  --nfold_test_regression --factor_adaptation --res_control --control 'logincomeHC01_VC85ACS3yr$10' 'hsgradHC03_VC93ACS3yr$10' --adaptation_factor logincomeHC01_VC85ACS3yr$10 hsgradHC03_VC93ACS3yr$10 --model ridgecv --folds 10 --csv


'''








JSON = "PostStratFactorAdaptTests.json"
MODEL_NAME = 'ridgecv'
DATABASE = 'ai_fairness'
TABLE = 'msgs_100u'
CORREL_FIELD = 'cnty'
OUTCOME_TABLE = "combined_county_outcomes"
OUTCOME_FIELDS = ["heart_disease", "suicide", "life_satisfaction", "perc_fair_poor_health"]
OUTCOME_CONTROLS = ["total_pop10", "femalePOP165210D$10", "hispanicPOP405210D$10", "blackPOP255210D$10", "forgnbornHC03_VC134ACS3yr$10", "bachdegHC03_VC94ACS3yr$10", "marriedaveHC03_AC3yr$10", "logincomeHC01_VC85ACS3yr$10", "unemployAve_BLSLAUS$0910", "perc_less_than_18_chr14_2012", "perc_65_and_over_chr14_2012"]
GROUP_FREQ_THRESH = 1
FEATURE_TABLES = ["feat$1gram$msgs_100u$cnty$16to16", "feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$16to16"]






def main():


    adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]
    adaptationFactorsToRemove = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10", "bachdegHC03_VC94ACS3yr$10"]


    #RegressionTest().run()


    #RegressionTest(featTables=["feat$1gram$msgs_100u$cnty$16to16$k10bin50IE", "feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$k10bin50IE"]).run()


    #ResidualControlRegressionTest(outcomeControls = OUTCOME_CONTROLS, outcomeFields=OUTCOME_FIELDS).run()


    #RegClassTests.ResidualFactorAdaptationRegressionTest(outcomeControls = list(set(OUTCOME_CONTROLS) - set(adaptationFactorsToRemove)), outcomeFields=OUTCOME_FIELDS + adaptationFactors).run(adaptationFactors=adaptationFactors)
    print(list(set(OUTCOME_CONTROLS)))
    RegClassTests.FactorAdaptationRegressionTest(outcomeControls = list(set(OUTCOME_CONTROLS)), outcomeFields=OUTCOME_FIELDS + adaptationFactors).run(adaptationFactors=adaptationFactors)


    #RegClassTests.ResidualFactorAdaptationRegressionTest(featTables=["feat$1gram$msgs_100u$cnty$16to16$k10bin50IE", "feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$k10bin50IE"], outcomeControls = list(set(OUTCOME_CONTROLS) - set(adaptationFactorsToRemove)), outcomeFields=OUTCOME_FIELDS + adaptationFactors).run(adaptationFactors=adaptationFactors)


   
    #RegressionTest(featTables=["feat$cat_2000fb_w$msgs30u$cnty$1gra$ie$raking$s0$b0"]).run()






    # Test2 = lambda: (lambda: [RegressionTest(featTables=["feat$1gram$msgs_100u$cnty$16to16$k10bin50IE", "feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$k10bin50IE"]).run()])()
    # process2 = multiprocessing.Process(target=Test2)
    # Test3 = lambda: (lambda: [ResidualControlRegressionTest().run()])()
    # process3 = multiprocessing.Process(target=Test3)


    #Test regular regression on n-grams
    #regReg = RegressionTest()
    #regReg.run()


    #Test Robust PostStrat
    #regReg = RegularRegressionTest()
    #regReg.run(featTables=["feat$1gram$msgs_100u$cnty$16to16$k10bin50IE", "feat$cat_met_a30_2000_cp_w$msgs_100u$cnty$k10bin50IE"])


    #resReg = ResidualControlRegressionTest()
    #resReg.run()






class RegClassTests():


    def __init__(self, JSON, MODEL_NAME, DATABASE, TABLES, TABLE, CORREL_FIELD, OUTCOME_TABLE, OUTCOME_FIELDS, OUTCOME_CONTROLS, GROUP_FREQ_THRESH, FEATURE_TABLES):
        self.JSON = JSON
        self.MODEL_NAME = MODEL_NAME
        self.DATABASE = DATABASE
        self.TABLES = TABLES
        self.TABLE = TABLE
        self.CORREL_FIELD = CORREL_FIELD
        self.OUTCOME_TABLE = OUTCOME_TABLE
        self.OUTCOME_FIELDS = OUTCOME_FIELDS
        self.OUTCOME_CONTROLS = OUTCOME_CONTROLS
        self.GROUP_FREQ_THRESH = GROUP_FREQ_THRESH
        self.FEATURE_TABLES = FEATURE_TABLES




    class Test(ABC):
        @abstractmethod
        def __init__(self):
            self.scores = {}
        @abstractmethod
        def run(self):
            pass


    class ClassificationTest(Test):


        name = "Classification Test"


        def __init__(self, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH):
            og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None)
            fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]


            self.rp = ClassifyPredictor(og, fgs, 'lr', None, None)
            self.result = {
                "name": self.name,
                "tables": {
                    "table": table,
                    "feat": featTables,
                    "outcome": outcomeTable,
                    "outcomeFields": outcomeFields,
                    "outcomeControls": outcomeControls
                },
                "scores": {}
            }


        def run(self):
            scoresRaw = self.rp.testControlCombos(comboSizes = [], nFolds = 10, allControlsOnly=True)
            self._saveResults(scoresRaw)


        def _saveResults(self, scoresRaw):
            for score in scoresRaw:
                self.result["scores"][score] = scoresRaw[score][tuple()][1]






    class FactorAdaptationClassificationTest(ClassificationTest):
        name = "Factor Adaptation Classification Test"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


        def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
            self.result["adaptationFactors"] = adaptationFactors
            scoresRaw = self.rp.testControlCombos(nFolds = 10, adaptColumns = adaptationFactors, allControlsOnly=True)
            self._saveResults(scoresRaw)






    class RegressionTest(Test):


        name = "Regression Test"


        def __init__(self, db=DATABASE, table=TABLE, outcomeTable=OUTCOME_TABLE, correlField = CORREL_FIELD, featTables = FEATURE_TABLES, outcomeFields = OUTCOME_FIELDS, outcomeControls = OUTCOME_CONTROLS, groupFreqThresh = GROUP_FREQ_THRESH):
            og = OutcomeGetter(corpdb = db, corptable = table, correl_field=correlField, outcome_table=outcomeTable, outcome_value_fields=outcomeFields, outcome_controls=outcomeControls, outcome_categories = [], multiclass_outcome = [], featureMappingTable='', featureMappingLex='', wordTable = None, fold_column = None, group_freq_thresh = 0)
            fgs = [FeatureGetter(corpdb = db, corptable = table, correl_field=correlField, featureTable=featTable, featNames="", wordTable = None) for featTable in featTables]
            RegressionPredictor.featureSelectionString = dlac.DEF_RP_FEATURE_SELECTION_MAPPING['magic_sauce']


            self.rp = RegressionPredictor(og, fgs, MODEL_NAME, None, None)
            self.result = {
                "name": self.name,
                "tables": {
                    "table": table,
                    "feat": featTables,
                    "outcome": outcomeTable,
                    "outcomeFields": outcomeFields,
                    "outcomeControls": outcomeControls
                },
                "scores": {}
            }


        def run(self):
            scoresRaw = self.rp.testControlCombos(comboSizes = [], nFolds = 10, allControlsOnly=True, report = False)
            self._saveResults(scoresRaw)


        def _saveResults(self, scoresRaw):
            '''for score in scoresRaw:
                self.result["scores"][score] = scoresRaw[score][tuple()][1]


            rh.SaveResult(self.result, "PoststratFactoradaptResults.json")'''
            outputStream = open("original_print" + JSON.replace(".json", ".csv"), 'a')
            csv_writer = csv.writer(outputStream)
            csv_writer.writerow([self.name, self.result["tables"]["outcomeFields"], self.result["tables"]["outcomeControls"], self.result["tables"]["feat"]])
            RegressionPredictor.printComboControlScoresToCSV(scoresRaw, outputStream)
            outputStream.close()






    class ResidualControlRegressionTest(RegressionTest):


        name = "Residualized Controls Regression Test"


        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


        def run(self):
            scoresRaw = self.rp.testControlCombos(nFolds = 10, residualizedControls =True, allControlsOnly=True, report = False)
            self._saveResults(scoresRaw)






    class FactorAdaptationRegressionTest(RegressionTest):
        name = "Factor Adaptation Regression Test"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


        def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
            self.result["adaptationFactors"] = adaptationFactors
            scoresRaw = self.rp.testControlCombos(nFolds = 10, adaptationFactorsName = adaptationFactors, integrationMethod="fa", allControlsOnly=True, report = False)
            self._saveResults(scoresRaw)






    class ResidualFactorAdaptationRegressionTest(RegressionTest):
        name = "Residualized Factor Adaptation Regression Test"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


        def run(self, adaptationFactors = ["logincomeHC01_VC85ACS3yr$10", "hsgradHC03_VC93ACS3yr$10"]):
            self.result["adaptationFactors"] = adaptationFactors
            scoresRaw = self.rp.testControlCombos(nFolds = 10, residualizedControls=True, adaptationFactorsName = adaptationFactors, integrationMethod="rfa", allControlsOnly=True, report = False)
            self._saveResults(scoresRaw)
























if __name__ == "__main__":
    main()
