#########################################
# Mediation Analysis
#
# Interfaces with FeatureWorker and Statsmodels
#
# example: 
#
# example usage: ./fwInterface.py -d DATABASE -t TABLE -c GROUP_BY --mediation \
#                --outcome_table OUTCOME_TABLE_NAME
#                --path_starts PATH_START_1 ... PATH_START_J \
#                --mediators
#                --outcomes OUTCOME_1 ... OUTCOME_K \
#                -f FEATURE_TABLE [-features FEATURE_1 ... FEATURE_I] \
#                [-controls CONTROL_1 ... CONTROLS_L] \
#                [--feat_as_path_start or --feat_as_outcome or --feat_as_control or --no_features] \
#                [--mediation_boot] \
#                [--group_freq_thresh GROUP_THRESH] \
#                [--output_name OUTPUT_NAME]

import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
import statsmodels.api as sm
import scipy.stats as st
import csv
import collections
import patsy
from math import sqrt
from scipy.stats import zscore
from scipy.stats.stats import pearsonr, spearmanr
from fwConstants import pCorrection, DEF_P
from operator import itemgetter
import itertools
import sys

MAX_SUMMARY_SIZE = 10 # maximum number of results to print in summary for each path start / outcome pair

"""
Mediation analysis
Implements algorithm 1 ('parametric inference') and algorithm 2
('nonparametric inference') from:
Imai, Keele, Tingley (2010).  A general approach to causal mediation
analysis. Psychological Methods 15:4, 309-334.
http://imai.princeton.edu/research/files/BaronKenny.pdf
The algorithms are described on page 317 of the paper.
In the case of linear models with no interactions involving the
mediator, the results should be similar or identical to the earlier
Barron-Kenny approach.
"""

class Mediation(object):
	"""
	Conduct a mediation analysis.
	Parameters
	----------
	outcome_model : statsmodels model
		Regression model for the outcome.  Predictor variables include
		the treatment/exposure, the mediator, and any other variables
		of interest.
	mediator_model : statsmodels model
		Regression model for the mediator variable.  Predictor
		variables include the treatment/exposure and any other
		variables of interest.
	exposure : string or (int, int) tuple
		The name or column position of the treatment/exposure
		variable.  If positions are given, the first integer is the
		column position of the exposure variable in the outcome model
		and the second integer is the position of the exposure variable
		in the mediator model.  If a string is given, it must be the name
		of the exposure variable in both regression models.
	mediator : string or int
		The name or column position of the mediator variable in the
		outcome regression model.  If None, infer the name from the
		mediator model formula (if present).
	moderators : dict
		Map from variable names or index positions to values of
		moderator variables that are held fixed when calculating
		mediation effects.  If the keys are index position they must
		be tuples `(i, j)` where `i` is the index in the outcome model
		and `j` is the index in the mediator model.  Otherwise the
		keys must be variable names.
	outcome_fit_kwargs : dict-like
		Keyword arguments to use when fitting the outcome model.
	mediator_fit_kwargs : dict-like
		Keyword arguments to use when fitting the mediator model.
	Returns a ``MediationResults`` object.
	Notes
	-----
	The mediator model class must implement ``get_distribution``.
	Examples
	--------
	A basic mediation analysis using formulas:
	>>> probit = sm.families.links.probit
	>>> outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat + age + educ + gender + income",
										data, family=sm.families.Binomial(link=probit))
	>>> mediator_model = sm.OLS.from_formula("emo ~ treat + age + educ + gender + income", data)
	>>> med = Mediation(outcome_model, mediator_model, "treat", "emo").fit()
	>>> med.summary()
	A basic mediation analysis without formulas.  This may be slightly
	faster than the approach using formulas.  If there are any
	interactions involving the treatment or mediator variables this
	approach will not work, you must use formulas.
	>>> outcome = np.asarray(data["cong_mesg"])
	>>> outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", data,
								  return_type='dataframe')
	>>> probit = sm.families.links.probit
	>>> outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit))
	>>> mediator = np.asarray(data["emo"])
	>>> mediator_exog = patsy.dmatrix("treat + age + educ + gender + income", data,
								 return_type='dataframe')
	>>> mediator_model = sm.OLS(mediator, mediator_exog)
	>>> tx_pos = [outcome_exog.columns.tolist().index("treat"),
	>>>           mediator_exog.columns.tolist().index("treat")]
	>>> med_pos = outcome_exog.columns.tolist().index("emo")
	>>> med = Mediation(outcome_model, mediator_model, tx_pos, med_pos).fit()
	>>> med.summary()
	A moderated mediation analysis.  The mediation effect is computed
	for people of age 20.
	>>> fml = "cong_mesg ~ emo + treat*age + emo*age + educ + gender + income",
	>>> outcome_model = sm.GLM.from_formula(fml, data,
	>>>                                     family=sm.families.Binomial())
	>>> mediator_model = sm.OLS.from_formula("emo ~ treat*age + educ + gender + income", data)
	>>> moderators = {"age" : 20}
	>>> med = Mediation(outcome_model, mediator_model, "treat", "emo",
						moderators=moderators).fit()
	References
	----------
	Imai, Keele, Tingley (2010).  A general approach to causal mediation
	analysis. Psychological Methods 15:4, 309-334.
	http://imai.princeton.edu/research/files/BaronKenny.pdf
	Tingley, Yamamoto, Hirose, Keele, Imai (2014).  mediation : R
	package for causal mediation analysis.  Journal of Statistical
	Software 59:5.  http://www.jstatsoft.org/v59/i05/paper
	"""

	def __init__(self, outcome_model, mediator_model, exposure, mediator=None, moderators=None, outcome_fit_kwargs=None, mediator_fit_kwargs=None):

		self.outcome_model = outcome_model
		self.mediator_model = mediator_model
		self.exposure = exposure
		self.moderators = moderators if moderators is not None else {}

		if mediator is None:
			self.mediator = self._guess_endog_name(mediator_model, 'mediator')
		else:
			self.mediator = mediator

		self._outcome_fit_kwargs = (outcome_fit_kwargs if outcome_fit_kwargs
									is not None else {})
		self._mediator_fit_kwargs = (mediator_fit_kwargs if mediator_fit_kwargs
									 is not None else {})

		# We will be changing these so need to copy.
		self._outcome_exog = outcome_model.exog.copy()
		self._mediator_exog = mediator_model.exog.copy()

		# Position of the exposure variable in the mediator model.
		self._exp_pos_mediator = self._variable_pos('exposure', 'mediator')

		# Position of the exposure variable in the outcome model.
		self._exp_pos_outcome = self._variable_pos('exposure', 'outcome')

		# Position of the mediator variable in the outcome model.
		self._med_pos_outcome = self._variable_pos('mediator', 'outcome')


	def _variable_pos(self, var, model):
		if model == 'mediator':
			mod = self.mediator_model
		else:
			mod = self.outcome_model

		if var == 'mediator':
			return maybe_name_or_idx(self.mediator, mod)[1]

		exp = self.exposure
		exp_is_2 = ((len(exp) == 2) and (type(exp) != type('')))

		if exp_is_2:
			if model == 'outcome':
				return exp[0]
			elif model == 'mediator':
				return exp[1]
		else:
			return maybe_name_or_idx(exp, mod)[1]


	def _guess_endog_name(self, model, typ):
		if hasattr(model, 'formula'):
			return model.formula.split("~")[0].strip()
		else:
			raise ValueError('cannot infer %s name without formula' % typ)


	def _simulate_params(self, result):
		"""
		Simulate model parameters from fitted sampling distribution.
		"""
		mn = result.params
		cov = result.cov_params()
		return np.random.multivariate_normal(mn, cov)


	def _get_mediator_exog(self, exposure):
		"""
		Return the mediator exog matrix with exposure set to the given
		value.  Set values of moderated variables as needed.
		"""
		mediator_exog = self._mediator_exog
		if not hasattr(self.mediator_model, 'formula'):
			mediator_exog[:, self._exp_pos_mediator] = exposure
			for ix in self.moderators:
				v = self.moderators[ix]
				mediator_exog[:, ix[1]] = v
		else:
			# Need to regenerate the model exog
			df = self.mediator_model.data.frame.copy()
			df.loc[:, self.exposure] = exposure
			for vname in self.moderators:
				v = self.moderators[vname]
				df.loc[:, vname] = v
			klass = self.mediator_model.__class__
			init_kwargs = self.mediator_model._get_init_kwds()
			model = klass.from_formula(data=df, **init_kwargs)
			mediator_exog = model.exog

		return mediator_exog


	def _get_outcome_exog(self, exposure, mediator):
		"""
		Retun the exog design matrix with mediator and exposure set to
		the given values.  Set values of moderated variables as
		needed.
		"""
		outcome_exog = self._outcome_exog
		if not hasattr(self.outcome_model, 'formula'):
			outcome_exog[:, self._med_pos_outcome] = mediator
			outcome_exog[:, self._exp_pos_outcome] = exposure
			for ix in self.moderators:
				v = self.moderators[ix]
				outcome_exog[:, ix[0]] = v
		else:
			# Need to regenerate the model exog
			df = self.outcome_model.data.frame.copy()
			df.loc[:, self.exposure] = exposure
			df.loc[:, self.mediator] = mediator
			for vname in self.moderators:
				v = self.moderators[vname]
				df.loc[:, vname] = v
			klass = self.outcome_model.__class__
			init_kwargs = self.outcome_model._get_init_kwds()
			model = klass.from_formula(data=df, **init_kwargs)
			outcome_exog = model.exog

		return outcome_exog


	def _fit_model(self, model, fit_kwargs, boot=False):
		klass = model.__class__
		init_kwargs = model._get_init_kwds()
		endog = model.endog
		exog = model.exog
		if boot:
			ii = np.random.randint(0, len(endog), len(endog))
			endog = endog[ii]
			exog = exog[ii, :]
		outcome_model = klass(endog, exog, **init_kwargs)
		return outcome_model.fit(**fit_kwargs)


	def fit(self, method="parametric", n_rep=1000):
		"""
		Fit a regression model to assess mediation.

		Arguments
		---------
		method : string
			Either 'parametric' or 'bootstrap'.
		n_rep : integer
			The number of simulation replications.

		Returns a MediationResults object.
		"""

		if method.startswith("para"):
			# Initial fit to unperturbed data.
			outcome_result = self._fit_model(self.outcome_model, self._outcome_fit_kwargs)
			mediator_result = self._fit_model(self.mediator_model, self._mediator_fit_kwargs)
		elif not method.startswith("boot"):
			raise("method must be either 'parametric' or 'bootstrap'")

		indirect_effects = [[], []]
		direct_effects = [[], []]

		for iter in range(n_rep):

			if method == "parametric":
				# Realization of outcome model parameters from sampling distribution
				outcome_params = self._simulate_params(outcome_result)

				# Realization of mediation model parameters from sampling distribution
				mediation_params = self._simulate_params(mediator_result)
			else:
				outcome_result = self._fit_model(self.outcome_model,
												 self._outcome_fit_kwargs, boot=True)
				outcome_params = outcome_result.params
				mediator_result = self._fit_model(self.mediator_model,
												  self._mediator_fit_kwargs, boot=True)
				mediation_params = mediator_result.params

			# predicted outcomes[tm][te] is the outcome when the
			# mediator is set to tm and the outcome/exposure is set to
			# te.
			predicted_outcomes = [[None, None], [None, None]]
			for tm in 0, 1:
				mex = self._get_mediator_exog(tm)
				gen = self.mediator_model.get_distribution(mediation_params,
														   mediator_result.scale,
														   exog=mex)
				potential_mediator = gen.rvs(mex.shape[0])

				for te in 0, 1:
					oex = self._get_outcome_exog(te, potential_mediator)
					po = self.outcome_model.predict(outcome_params, oex)
					predicted_outcomes[tm][te] = po

			for t in 0, 1:
				indirect_effects[t].append(predicted_outcomes[1][t] - predicted_outcomes[0][t])
				direct_effects[t].append(predicted_outcomes[t][1] - predicted_outcomes[t][0])

		for t in 0, 1:
			indirect_effects[t] = np.asarray(indirect_effects[t]).T
			direct_effects[t] = np.asarray(direct_effects[t]).T

		self.indirect_effects = indirect_effects
		self.direct_effects = direct_effects

		rslt = MediationResults(self.indirect_effects, self.direct_effects)
		rslt.method = method
		return rslt


def _pvalue(vec):
	return 2 * min(sum(vec > 0), sum(vec < 0)) / float(len(vec))

def get_generalized_distribution(self, params, scale=1, exog=None, exposure=None, offset=None):
	"""
	Returns a random number generator for the predictive distribution.
	Parameters
	----------
	params : array-like
		The model parameters.
	scale : scalar
		The scale parameter.
	exog : array-like
		The predictor variable matrix.
	Returns a frozen random number generator object.  Use the
	``rvs`` method to generate random values.
	Notes
	-----
	Due to the behavior of ``scipy.stats.distributions objects``,
	the returned random number generator must be called with
	``gen.rvs(n)`` where ``n`` is the number of observations in
	the data set used to fit the model.  If any other value is
	used for ``n``, misleading results will be produced.
	"""

	fit = self.predict(params, exog, exposure, offset, linear=False)

	import scipy.stats.distributions as dist

	if isinstance(self.family, families.Gaussian):
		return dist.norm(loc=fit, scale=np.sqrt(scale))

	elif isinstance(self.family, families.Binomial):
		return dist.binom(n=1, p=fit)

	elif isinstance(self.family, families.Poisson):
		return dist.poisson(mu=fit)

	elif isinstance(self.family, families.Gamma):
		alpha = fit / float(scale)
		return dist.gamma(alpha, scale=scale)

	else:
		raise ValueError("get_generalized_distribution not implemented for %s" % self.family.name)


#linear_model
def get_linear_distribution(model, params, scale, exog=None, dist_class=None):
	"""
	Returns a random number generator for the predictive distribution.
	Parameters
	----------
	params : array-like
		The model parameters (regression coefficients).
	scale : scalar
		The variance parameter.
	exog : array-like
		The predictor variable matrix.
	dist_class : class
		A random number generator class.  Must take 'loc' and
		'scale' as arguments and return a random number generator
		implementing an `rvs` method for simulating random values.
		Defaults to Gaussian.
	Returns a frozen random number generator object with mean and
	variance determined by the fitted linear model.  Use the
	``rvs`` method to generate random values.
	Notes
	-----
	Due to the behavior of ``scipy.stats.distributions objects``,
	the returned random number generator must be called with
	``gen.rvs(n)`` where ``n`` is the number of observations in
	the data set used to fit the model.  If any other value is
	used for ``n``, misleading results will be produced.
	"""
	fit = model.predict(params, exog)
	if dist_class is None:
		from scipy.stats.distributions import norm
		dist_class = norm
	gen = dist_class(loc=fit, scale=np.sqrt(scale))
	return gen


class MediationResults(object):

	def __init__(self, indirect_effects, direct_effects):

		self.indirect_effects = indirect_effects
		self.direct_effects = direct_effects

		indirect_effects_avg = [None, None]
		direct_effects_avg = [None, None]
		for t in 0, 1:
			indirect_effects_avg[t] = indirect_effects[t].mean(0)
			direct_effects_avg[t] = direct_effects[t].mean(0)

		self.ACME_ctrl = indirect_effects_avg[0]
		self.ACME_tx = indirect_effects_avg[1]
		self.ADE_ctrl = direct_effects_avg[0]
		self.ADE_tx = direct_effects_avg[1]
		self.total_effect = (self.ACME_ctrl + self.ACME_tx + self.ADE_ctrl + self.ADE_tx) / 2

		self.prop_med_ctrl = self.ACME_ctrl / self.total_effect
		self.prop_med_tx = self.ACME_tx / self.total_effect
		self.prop_med_avg = (self.prop_med_ctrl + self.prop_med_tx) / 2

		self.ACME_avg = (self.ACME_ctrl + self.ACME_tx) / 2
		self.ADE_avg = (self.ADE_ctrl + self.ADE_tx) / 2


	def summary(self, alpha=0.05, p_correction_method='', numMeds=1):
		"""
		Provide a summary of a mediation analysis.
		"""

		columns = ["Estimate", "Lower CI bound", "Upper CI bound", "P-value"]
		index = ["ACME (control)", "ACME (treated)", "ADE (control)", "ADE (treated)",
				 "Total effect", "Prop. mediated (control)", "Prop. mediated (treated)",
				 "ACME (average)", "ADE (average)", "Prop. mediated (average)"]
		smry = pd.DataFrame(columns=columns, index=index)

		for i, vec in enumerate([self.ACME_ctrl, self.ACME_tx, self.ADE_ctrl, self.ADE_tx,
								 self.total_effect, self.prop_med_ctrl,
								 self.prop_med_tx, self.ACME_avg, self.ADE_avg,
								 self.prop_med_avg]):

			if ((vec is self.prop_med_ctrl) or (vec is self.prop_med_tx) or
				(vec is self.prop_med_avg)):
				smry.iloc[i, 0] = np.median(vec)
			else:
				smry.iloc[i, 0] = vec.mean()
			smry.iloc[i, 1] = np.percentile(vec, 100 * alpha / 2)
			smry.iloc[i, 2] = np.percentile(vec, 100 * (1 - alpha / 2))
			if p_correction_method.startswith("bonf"):
				smry.iloc[i, 3] = _pvalue(vec)*numMeds
			else:
				smry.iloc[i, 3] = _pvalue(vec)

		smry = smry.convert_objects(convert_numeric=True)

		return smry

class MediationAnalysis:
	"""
	Interface between Mediation class in Statsmodels and FeatureWorker with the addition of standard Baron and Kenny approach. 

	Attributes:
		outcomeGetter: OutcomeGetter object
		featureGetter: FeatureGetter object
		pathStartNames (list): 
		mediatorNames (list): 
		outcomeNames (list): 
		controlNames (list): 

		mediation_method (str): "parametric" or "bootstrap" 
		boot_number (int): number of bootstrap iterations
		sig_level (float): significane level for reporting results in summary 
		output (dict): 
		output_sobel (dict): 
		output_p (dict): 

		baron_and_kenny (boolean): if True runs Baron and Kenny method
		imai_and_keele (boolean): if True runs Imai, Keele, and Tingley method

	References
	----------
	Imai, Keele, Tingley (2010).  A general approach to causal mediation
	analysis. Psychological Methods 15:4, 309-334.
	http://imai.princeton.edu/research/files/BaronKenny.pdf
	Tingley, Yamamoto, Hirose, Keele, Imai (2014).  mediation : R
	package for causal mediation analysis.  Journal of Statistical
	Software 59:5.  http://www.jstatsoft.org/v59/i05/paper
	"""

	def __init__(self, fg, og, path_starts, mediators, outcomes, controls, method="parametric", boot_number=1000, sig_level=DEF_P, style='baron'):
		
		self.outcomeGetter = og
		self.featureGetter = fg

		self.pathStartNames = path_starts  
		self.mediatorNames = mediators
		self.outcomeNames = outcomes  
		self.controlNames = controls

		self.mediation_method = method
		self.boot_number = boot_number
		self.sig_level = sig_level

		self.output = dict()
		self.output_sobel = dict()
		self.output_p = dict() # [c_p, c'_p, alpha_p, beta_p, sobel_p, ...]

		if style == 'baron':
			self.baron_and_kenny = True
			self.imai_and_keele = False
		elif style == 'imai':
			self.baron_and_kenny = False
			self.imai_and_keele = True
		elif style == 'both':
			self.baron_and_kenny = True
			self.imai_and_keele = True

	def _truncate_groups(seq, max_group_size, key):
		"""yield only the first `max_group_size` elements from each sub-group of `seq`"""
		for key, group in itertools.groupby(seq, key):
			for item in list(group)[:max_group_size]:
				yield item

	def print_summary(self, output_name=''):
		summary_results = []
		if output_name:
			if ".csv" in output_name:
				csv_name = output_name.replace(".csv", "_summary.csv")
			else:
				csv_name = output_name + "_summary.csv"
		else:
			csv_name = "mediation_summary.csv"
				
		header = ["Path Start", "Outcome", "Mediator"]
		if self.baron_and_kenny:
			header = header + ["c-c'", "sobel_P", "alpha", "beta", "c'"]
		if self.imai_and_keele: 
			header = header + ["Prop_mediated_average_Estimate", "Prop_mediated_average_P_value", 
				"ACME_average_Estimate", "ACME_average_P_value"] 

		for path_start in self.output:
			for outcome in self.output[path_start]:
				count = 0
				for mediator in self.output[path_start][outcome]:
					sobel_p = self.output_p[path_start][outcome][mediator][4]
					if sobel_p <= self.sig_level:
						results = [path_start, outcome, mediator]
						if self.baron_and_kenny:
							results = results + [self.output_sobel[path_start][outcome][mediator].tolist()[4] , sobel_p] + \
								[self.output_sobel[path_start][outcome][mediator].tolist()[6], 
								self.output_sobel[path_start][outcome][mediator].tolist()[9], 
								self.output_sobel[path_start][outcome][mediator].tolist()[2]]
						if self.imai_and_keele: 
							results = results + [self.output[path_start][outcome][mediator][0], 
								self.output_p[path_start][outcome][mediator][5],
								self.output[path_start][outcome][mediator][4],
								self.output_p[path_start][outcome][mediator][6]]
						summary_results.append(results)
							

		summary_results.sort(key=lambda x: (x[0].lower(), x[1].lower(), -abs(x[3])), reverse=False)
		if len(summary_results) > 0:
			print "Printing results to: %s" % csv_name
			with open(csv_name, 'wb') as csvfile:
				f = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
				f.writerow(header)
				for key, group in itertools.groupby(summary_results, key=lambda x: (x[0], x[1])):
					for item in list(group)[:MAX_SUMMARY_SIZE]:
						f.writerow(item)
		else:
			print "Summary: nothing passes significance threshold of %s." % (self.sig_level)
			

	def print_csv(self, output_name=''):
		if output_name:
			csv_name = output_name
		else:
			csv_name = "mediation.csv"
				
		header = ["Path Start", "Outcome", "Mediator"]
		if self.baron_and_kenny:
			header = header + ["c", "c_p", "c'", "c'_p", "c-c'", "alpha*beta", "alpha", "alpha_error", "alpha_p", 
				"beta", "beta_error", "beta_p", "sobel", "sobel_SE", "sobel_P"]
		if self.imai_and_keele: 
			header = header + ["Prop_mediated_average_Estimate", "Prop_mediated_average_P_value", 
				"ACME_average_Estimate", "ACME_average_P_value", 
				"ADE_average_Estimate", "ADE_average_P_value", 
				"Prop_mediated_average_Lower_CI_bound", "Prop_mediated_average_Upper_CI_bound", 
				"ACME_average_Lower_CI_bound", "ACME_average_Upper_CI_bound", 
				"ADE_average_Lower_CI_bound", "ADE_average_Upper_CI_bound", 
				"ACME_treated_Estimate", "ACME_treated_P_value", "ACME_treated_Lower_CI_bound", "ACME_treated_Upper_CI_bound", 
				"ACME_control_Estimate", "ACME_control_P_value", "ACME_control_Lower_CI_bound", "ACME_control_Upper_CI_bound", 
				"ADE_treated_Estimate", "ADE_treated_P_value","ADE_treated_Lower_CI_bound", "ADE_treated_Upper_CI_bound", 
				"ADE_control_Estimate", "ADE_control_P_value", "ADE_control_Lower_CI_bound", "ADE_control_Upper_CI_bound",
				"Total_effect_Estimate", "Total_effect_P_value", "Total_effect_Lower_CI_bound", "Total_effect_Upper_CI_bound", 
				"Prop_mediated_control_Estimate", "Prop_mediated_control_P_value", "Prop_mediated_control_Lower_CI_bound", "Prop_mediated_control_Upper_CI_bound", 
				"Prop_mediated_treated_Estimate", "Prop_mediated_treated_P_value", "Prop_mediated_treated_Lower_CI_bound", "Prop_mediated_treated_Upper_CI_bound"] 
				
		print "Printing results to: %s" % csv_name
		with open(csv_name, 'wb') as csvfile:
			f = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
			f.writerow(header)
			for path_start in self.output:
				for outcome in self.output[path_start]:
					for mediator in self.output[path_start][outcome]:
						bk_rearranged = []
						med_rearranged = []
						p_list = self.output_p[path_start][outcome][mediator]
						if self.baron_and_kenny:
							bk_rearranged = [self.output_sobel[path_start][outcome][mediator][0], p_list[0], self.output_sobel[path_start][outcome][mediator][2], p_list[1],
											self.output_sobel[path_start][outcome][mediator][4], self.output_sobel[path_start][outcome][mediator][5], 
											self.output_sobel[path_start][outcome][mediator][6], self.output_sobel[path_start][outcome][mediator][7], p_list[2],
											self.output_sobel[path_start][outcome][mediator][9], self.output_sobel[path_start][outcome][mediator][10], p_list[3], 
											self.output_sobel[path_start][outcome][mediator][12], self.output_sobel[path_start][outcome][mediator][13], p_list[4]]
						if self.imai_and_keele: 
							med_rearranged = [self.output[path_start][outcome][mediator][0], p_list[5], 
												self.output[path_start][outcome][mediator][4], p_list[6], 
												self.output[path_start][outcome][mediator][8], p_list[7]] \
												 + self.output[path_start][outcome][mediator][2:4] +  self.output[path_start][outcome][mediator][6:8] +  self.output[path_start][outcome][mediator][10:12] \
												 + [self.output[path_start][outcome][mediator][12], p_list[8]] + self.output[path_start][outcome][mediator][14:16] \
												 + [self.output[path_start][outcome][mediator][16], p_list[9]] + self.output[path_start][outcome][mediator][18:20] \
												 + [self.output[path_start][outcome][mediator][20], p_list[10]] + self.output[path_start][outcome][mediator][22:24] \
												 + [self.output[path_start][outcome][mediator][24], p_list[11]] + self.output[path_start][outcome][mediator][26:28] \
												 + [self.output[path_start][outcome][mediator][28], p_list[12]] + self.output[path_start][outcome][mediator][30:32] \
												 + [self.output[path_start][outcome][mediator][32], p_list[13]] + self.output[path_start][outcome][mediator][34:36] \
												 + [self.output[path_start][outcome][mediator][36], p_list[14]] + self.output[path_start][outcome][mediator][38:] 
						f.writerow([path_start, outcome, mediator] + bk_rearranged + med_rearranged)

		
	def prep_data(self, path_start, mediator, outcome, controlDict=None, controlNames=None, zscoreRegression=None):
		"""
		Take dictionary data and return a Pandas DataFrame indexed by group_id.
		Column names are 'path_start', 'mediator' and 'outcome'
		"""
		ps_df = pd.DataFrame(data=collections.OrderedDict(sorted(path_start.items())).items(), columns=['group_id', 'path_start']).set_index(['group_id']).fillna(0)
		m_df = pd.DataFrame(data=mediator.items(), columns=['group_id', 'mediator']).set_index(['group_id'])
		o_df = pd.DataFrame(data=outcome.items(), columns=['group_id', 'outcome']).set_index(['group_id'])

		data = ps_df.join(m_df)
		data = data.join(o_df)
		if controlDict and controlNames: 
			for control in controlNames:
				data = data.join(pd.DataFrame(data=controlDict[control].items(), columns=['group_id', control]).set_index(['group_id']))
		data = data.dropna()
		return data

	def get_data(self, switch, outcome_field, location, features):
		"""
		get data from outcomeGetter / featureGetter
		"""
		data = None
		if switch == "feat_as_path_start" and location == "path_start":
				data = features[outcome_field]
		elif switch == "feat_as_outcome" and location == "outcome":
				data = features[outcome_field]
		elif switch == "feat_as_control" and location == "control":
				data = features[outcome_field]
		elif switch == "default" and location == "mediator":
				data = features[outcome_field]
		else:
			data = dict((x, y) for x, y in self.outcomeGetter.getGroupAndOutcomeValues(outcomeField = outcome_field))
		return data

	def mediate(self, group_freq_thresh = 0, switch="default", spearman = False, p_correction_method = 'BH', 
				zscoreRegression = True, logisticReg = False):
		"""
		Runs the medition. 

		Args:
			group_freq_thresh (int): 
			switch (str): controls source (FeatureGetter or OutcomeGetter) of variables (path_starts, mediators, outcomes, controls)
			spearman (boolean): NOT BEING USED
			p_correction_method (str): Name of p correction method
			zscoreRegression (boolean): True if data is z-scored
			logisticReg (boolean): True if running logistic regression
		
		Data sources according to 'switch':
				"default": 
					FeatureGetter: mediators
					OutcomeGetter: path_starts and outcomes
				"feat_as_path_start":
					FeatureGetter: path_starts
					OutcomeGetter: mediators, outcomes and controls
				"feat_as_outcome":
					FeatureGetter: outcomes
					OutcomeGetter: path_starts, mediators and controls
				"feat_as_control":
					FeatureGetter: controls
					OutcomeGetter: path_starts, mediators and outcomes
				"no_features":
					OutcomeGetter: path_starts, mediators, outcomes and controls
		"""

		if "no_features" not in switch:
			(groups, allOutcomes, controls) = self.outcomeGetter.getGroupsAndOutcomes(group_freq_thresh)

		if self.featureGetter:
			(allFeatures, featureNames) = self.featureGetter.getGroupNormsWithZerosFeatsFirst(groups=groups)

		if switch == "feat_as_path_start":
			if len(self.pathStartNames) == 0:
				self.pathStartNames = allFeatures.keys()
			numMeds = len(self.pathStartNames)

		elif switch == "feat_as_outcome":
			if len(self.outcomeNames) == 0:
				self.outcomeNames = allFeatures.keys()
			numMeds = len(self.outcomeNames)

		elif switch == "feat_as_control":
			if len(self.controlNames) == 0:
				self.controlNames = allFeatures.keys()
			numMeds = len(self.pathStartNames)

		elif switch == "no_features":
			numMeds = len(self.pathStartNames)

		elif switch == "default":
			if len(self.mediatorNames) == 0:
				self.mediatorNames = allFeatures.keys()
			numMeds = len(self.mediatorNames)
			
		mediation_count = 0
		total_mediations = str(len(self.pathStartNames)*len(self.mediatorNames)*len(self.outcomeNames))

		for path_start in self.pathStartNames:
			self.output[path_start] = {}
			self.output_sobel[path_start] = {}
			self.output_p[path_start] = {}

			for outcome in self.outcomeNames:
				self.output[path_start][outcome] = {}
				self.output_sobel[path_start][outcome] = {}
				self.output_p[path_start][outcome] = {}

				for mediator in self.mediatorNames:
					mediation_count += 1
					self.output_p[path_start][outcome][mediator] = []
					self.output[path_start][outcome][mediator] = []
					
					if len(self.controlNames) > 0:
						data = self.prep_data(self.get_data(switch, path_start, "path_start", allFeatures), 
							self.get_data(switch, mediator, "mediator", allFeatures), 
							self.get_data(switch, outcome, "outcome", allFeatures), 
							controlsDict, self.controlNames, zscoreRegression=zscoreRegression)
						
						control_formula = " + " + " + ".join(self.controlNames)
					else: 
						data = self.prep_data(self.get_data(switch, path_start, "path_start", allFeatures), 
							self.get_data(switch, mediator, "mediator", allFeatures), 
							self.get_data(switch, outcome, "outcome", allFeatures), 
							zscoreRegression=zscoreRegression)
						
						control_formula = ""
					
					outcome_exog = patsy.dmatrix("mediator + path_start " + control_formula + " -1", data,
													  return_type='dataframe')
					mediator_exog = patsy.dmatrix("path_start " + control_formula + " -1", data,
											 return_type='dataframe')
					direct_exog = patsy.dmatrix("path_start " + control_formula + " -1", data,
											 return_type='dataframe')

					if zscoreRegression:
						outcome_array = np.asarray(data['outcome']) if logisticReg else np.asarray(zscore(data['outcome']))
						mediator_array = np.asarray(data['mediator']) if logisticReg else np.asarray(zscore(data['mediator']))
						outcome_exog['mediator'] = zscore(outcome_exog['mediator'])
						outcome_exog['path_start'] = zscore(outcome_exog['path_start'])
						mediator_exog['path_start'] = zscore(mediator_exog['path_start'])
						direct_exog['path_start'] = zscore(direct_exog['path_start'])
					else:
						outcome_array = np.asarray(data['outcome'])
						mediator_array = np.asarray(data['mediator'])

					outcome_model = sm.OLS(outcome_array, outcome_exog)
					mediator_model = sm.OLS(mediator_array, mediator_exog)
					direct_model = sm.OLS(outcome_array, direct_exog)

					# classic mediation with Sobel Test
					if self.baron_and_kenny:
						outcome_results = outcome_model.fit()
						mediator_results = mediator_model.fit()
						direct_results = direct_model.fit()

						c = direct_results.params.get('path_start')
						c_prime = outcome_results.params.get('path_start')
						alpha = mediator_results.params.get('path_start')
						alpha_error = mediator_results.bse.get('path_start')
						beta = outcome_results.params.get('mediator')
						beta_error = outcome_results.bse.get('mediator')
						sobel_SE = sqrt(beta*beta*alpha_error*alpha_error + alpha*alpha*beta_error*beta_error)
						sobel = (alpha*beta)/ sobel_SE

						if p_correction_method.startswith("bonf"):
							c_p = direct_results.pvalues.get('path_start')*numMeds
							c_prime_p = outcome_results.pvalues.get('path_start')*numMeds
							alpha_p = mediator_results.pvalues.get('path_start')*numMeds
							beta_p = outcome_results.pvalues.get('mediator')*numMeds
							sobel_p = st.norm.sf(abs(sobel))*2*numMeds
						else:
							c_p = direct_results.pvalues.get('path_start')
							c_prime_p = outcome_results.pvalues.get('path_start')
							alpha_p = mediator_results.pvalues.get('path_start')
							beta_p = outcome_results.pvalues.get('mediator')
							sobel_p = st.norm.sf(abs(sobel))*2

						self.output_sobel[path_start][outcome][mediator] = np.array([c, c_p, c_prime, c_prime_p, c-c_prime, alpha*beta, alpha, alpha_error, alpha_p, beta, beta_error, beta_p, sobel, sobel_SE, sobel_p])
						self.output_p[path_start][outcome][mediator] = self.output_p[path_start][outcome][mediator] + [c_p, c_prime_p, alpha_p, beta_p, sobel_p ]

					# imai_and_keele mediation method
					if self.imai_and_keele:
						tx_pos = [outcome_exog.columns.tolist().index("path_start"),
								  mediator_exog.columns.tolist().index("path_start")]
						med_pos = outcome_exog.columns.tolist().index("mediator")

						med = Mediation(outcome_model, mediator_model, tx_pos, med_pos)

						med_result = med.fit(method=self.mediation_method, n_rep=self.boot_number)
						summary = med_result.summary(p_correction_method = p_correction_method, numMeds=numMeds)
						summary_array = np.reshape(summary.values, 40).tolist()
						
						self.output[path_start][outcome][mediator] = summary_array
						self.output_p[path_start][outcome][mediator] = self.output_p[path_start][outcome][mediator] + summary["P-value"].tolist()

					print "Mediation number " + str(mediation_count) + " out of " + total_mediations
					
					if len(self.controlNames) > 0:
						print "Path Start: %s, Mediator: %s, Outcome: %s, Controls: %s" % (path_start, mediator, outcome, ", ".join(self.controlNames))
					else:
						print "Path Start: %s, Mediator: %s, Outcome: %s" % (path_start, mediator, outcome)
					if self.baron_and_kenny:
						print "C: %s, C_p: %s, C': %s, C'_p: %s" % (str(c), str(c_p), str(c_prime), str(c_prime_p))
						print "C-C': %s, alpha*beta: %s" % (str(c-c_prime), str(alpha*beta))
						print "alpha: %s, alpha_error: %s, alpha_p: %s" % (str(alpha), str(alpha_error), str(alpha_p))
						print "beta: %s, beta_error: %s, beta_p: %s" % (str(beta), str(beta_error), str(beta_p))
						print "Sobel z-score: %s, Sobel SE: %s, Sobel p: %s" % (str(sobel), str(sobel_SE), str(sobel_p))
					if self.imai_and_keele:
						print summary
					print ""

		if p_correction_method and not p_correction_method.startswith("bonf"):
			p_list = []
			if self.baron_and_kenny:
				p_list = p_list + ["C_p", "C'_p", "alpha_p", "beta_p", "sobel_p"]
			if self.imai_and_keele:
				p_list = p_list + ["Prop_mediated_average_P_value", "ACME_average_P_value", "ADE_average_P_value", "ACME_treated_P_value", 
					"ACME_control_P_value", "ADE_treated_P_value", "ADE_control_P_value", "Total_effect_P_value", 
					"Prop_mediated_control_P_value", "Prop_mediated_treated_P_value"]
			p_dict = dict()
			r_dict = dict()
			for path_start in self.pathStartNames:
				for outcome in self.outcomeNames:
					for i, p in enumerate(p_list):
						p_dict[p] = dict()
						for mediator in self.mediatorNames:
							p_dict[p][mediator] = self.output_p[path_start][outcome][mediator][i]
						p_dict[p] = pCorrection(p_dict[p], p_correction_method, [0.05, 0.01, 0.001], rDict = None)
					for mediator in self.mediatorNames:
						self.output_p[path_start][outcome][mediator] = [p_dict[p][mediator] for p in p_list]

