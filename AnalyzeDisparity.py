from itertools import count
import pandas as pd
from pydantic import constr
import us
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pdb

def getFolderPaths(external=True):
    if external:
        rawDataFolder = '/Volumes/One Touch/Data/'
    else:
        rawDataFolder = '/Users/hadielzayn/Documents/Data/voter/L2 Data/'
    cleanDataFolder = '/Users/hadielzayn/Dropbox/Research/RegLab/Projects/Ecological Inference/Data/'
    outputFolder = '/Users/hadielzayn/Dropbox/Research/RegLab/Projects/Ecological Inference/Output'
    return rawDataFolder,cleanDataFolder,outputFolder

def setGlobals():
    states_with_truth = ['AL','NC','SC']
    dem_varbs = ['Borough', 'Borough_Ward', 'City', 'City_Ward', 'County', 'CountyEthnic_Description',
             'CountyEthnic_LALEthnicCode', 'LALVOTERID', 'Precinct', 'Residence_Addresses_CensusBlock',
             'Residence_Addresses_CensusBlockGroup', 'Residence_Addresses_CensusTract', 'Voters_Age',
             'Voters_CalculatedRegDate', 'Voters_LastName', 'Voters_FirstName','Parties_Description','Voters_FIPS']
    return states_with_truth,dem_varbs

def chenEstimator(dataset,outcome='f_dem',f_black='f_black_cty',f_nonblack='f_nonblack_cty',weight=None):
    if weight is None:
        return (dataset[outcome]*dataset[f_black]).sum()/(dataset[f_black].sum())- (dataset[outcome]*dataset[f_nonblack]).sum()/(dataset[f_nonblack]).sum()
    else:
        return (dataset[outcome]*dataset[f_black]*dataset[weight]).sum()/((dataset[f_black]*dataset[weight]).sum())- (dataset[outcome]*dataset[f_nonblack]*dataset[weight]).sum()/((dataset[f_nonblack]*dataset[weight])).sum()
    
def regEstimator(dataset,outcome='f_dem',f_black='f_black_cty',weight=None,returnSE=False):
    if weight is None:
        model = smf.ols(outcome + ' ~ '+f_black,dataset)
    else:
        model = smf.wls(outcome+' ~ '+f_black,dataset,weights=dataset[weight])
    fitted = model.fit()
    if returnSE:
        return fitted.params[f_black],fitted.bse[f_black]
    else:
        return fitted.params[f_black]

def getTruth(dataset,posOutcomeAndBlack='demAndBlack',posOutcomeAndNonBlack='demAndNonBlack',total_b='blackTot',total_nb='nonblackTot'):
    return (dataset[posOutcomeAndBlack]).sum()/(dataset[total_b]).sum()-(dataset[posOutcomeAndNonBlack].sum())/(dataset[total_nb]).sum()

def methodOfBoundsByUnit(ds,blackTotVarb='blackTotKnown',nonblackTotVarb='nonblackTotKnown',totVarb='totalDemKnown',unit_varb='County'):
    ds['maxNonBlacks'] = (ds[[nonblackTotVarb,totVarb]]).min(axis=1)
    ds['minBlacks'] = (ds[totVarb]-ds['maxNonBlacks']).clip(lower=0)
    ds['maxBlacks'] =  (ds[[blackTotVarb,totVarb]]).min(axis=1)
    ds['minNonBlacks'] = (ds[totVarb]-ds['maxBlacks']).clip(lower=0)
    dif_lb = ds['minBlacks'].sum()/ds[blackTotVarb].sum() - ds['maxNonBlacks'].sum()/ds[nonblackTotVarb].sum()
    dif_ub = ds['maxBlacks'].sum()/ds[blackTotVarb].sum() - ds['minNonBlacks'].sum()/ds[nonblackTotVarb].sum()
    return dif_lb,dif_ub

def methodOfBoundsOverall(ds):
    summed = ds[['blackTot','nonblackTot','totalDem','totalNonDem']].sum()
    min_black_dem = max(summed['totalDem']-summed['nonblackTot'],0)
    nonblack_dem_if_minblack = summed['totalDem']-min_black_dem
    dif_lb = min_black_dem/summed['blackTot'] - nonblack_dem_if_minblack/summed['nonblackTot']
    min_nonblack_dem = max(summed['totalDem']-summed['blackTot'],0)
    black_dem_if_minnonblack = summed['totalDem']-min_nonblack_dem
    dif_ub = black_dem_if_minnonblack/summed['blackTot'] - min_nonblack_dem/summed['nonblackTot']
    return dif_lb,dif_ub

def getEstimate(aggedLevel,type='chen',outcomeVariable='f_dem',pBlack='f_black_cty',pNonblack='f_nonblack_cty',weight='voters'):
    if type=='chen':
        return chenEstimator(aggedLevel,outcome=outcomeVariable,f_black=pBlack,f_nonblack=pNonblack,weight=weight)
    elif type=='reg':
        return regEstimator(aggedLevel,outcome=outcomeVariable,f_black=pBlack,f_nonblack=pNonblack,weight=weight)

def getVarianceRatio(dataset,pBlackVarb='bisgBlackCBG'):
    varPBlack = dataset[pBlackVarb].var()
    ePBlack = dataset[pBlackVarb].mean()
    return varPBlack/((ePBlack)*(1-ePBlack))

def cleanUnitLevel(data,pblack_varb='f_black_cty',pnonblack_varb='f_nonblack_cty'):
    if pblack_varb=='f_black_CBG':
        data['f_black_cbg'] = data['f_black_CBG']
        data['f_nonblack_cbg'] = data['f_nonblack_CBG']
    data['voters'] = data['Voters_FIPS']
    data['blackTot'] = data[pblack_varb]*data['voters']
    data['nonblackTot'] = data[pnonblack_varb]*data['voters']
    data['turnoutRegPre'] =data['turnedOutAndRegPre']
    data['notTurnoutRegPre'] = data['notTurnedOutAndRegPre']
    data['totalTurnout'] = data['turnoutRegPre']
    data['totalNotTurnout'] = data['notTurnoutRegPre']
    data['preRegVoters'] = data['turnoutRegPre']+data['notTurnoutRegPre']
    data['blackTot_tn'] = data[pblack_varb]*data['preRegVoters']
    data['nonblackTot_tn'] = data[pnonblack_varb]*data['preRegVoters']
    data['f_dem'] = data['regDem']/(data['regDem']+data['regNonDem'])
    data['f_turnout'] = data['turnoutRegPre'] / (data['turnoutRegPre']+data['notTurnoutRegPre'])
    data['f_dem'] = data['regDem']/(data['regDem']+data['regNonDem'])
    data['f_turnout'] = data['turnoutRegPre'] / (data['turnoutRegPre']+data['notTurnoutRegPre']) 
    data['totalDem'] = data['regDem']
    data['totalNonDem'] = data['regNonDem']
    data['totalDem'] = data['regDem']
    data['totalNonDem'] = data['regNonDem']
    return data

def cleanUnitLevelWithTruth(data):
    data['blackTotKnown'] = data['demAndBlack']+data['nonDemAndBlack']
    data['nonblackTotKnown'] = data['demAndNonBlack']+data['nonDemAndNonBlack']
    data['totalDemKnown'] = data['demAndBlack']+data['demAndNonBlack']
    data['totalNonDemKnown'] = data['nonDemAndBlack']+data['nonDemAndNonBlack']
    data['f_dem'] = data['totalDemKnown']/(data['totalDemKnown']+data['totalNonDemKnown'])
    data['blackTotRegPre'] = data['turnedOutAndRegPreAndBlack']+data['notTurnedOutAndRegPreAndBlack']
    data['nonblackTotRegPre'] = data['turnedOutAndRegPreAndNonblack']+data['notTurnedOutAndRegPreAndNonblack']
    data['turnoutRegPre'] = data['turnedOutAndRegPreAndBlack'] + data['turnedOutAndRegPreAndNonblack']
    data['notTurnoutRegPre'] = data['notTurnedOutAndRegPreAndNonblack']+data['notTurnedOutAndRegPreAndBlack']
    data['preRegVoters'] =  data['turnoutRegPre']+data['notTurnoutRegPre']
    data['f_turnout'] = data['turnoutRegPre'] / (data['turnoutRegPre']+data['notTurnoutRegPre'])
    return data

def getEstimatePair(data,outcome_varb='f_dem',pblack_varb='f_black_cty',pnonblack_varb='f_nonblack_cty',weight_varb='voters',returnSE=False):
    estimate_chen = chenEstimator(data,outcome=outcome_varb,f_black=pblack_varb,f_nonblack=pnonblack_varb,weight=weight_varb)
    if returnSE:
        estimate_reg,estimate_reg_se = regEstimator(data,outcome=outcome_varb,f_black=pblack_varb,weight=weight_varb,returnSE=True)
    else:
        estimate_reg = regEstimator(data,outcome=outcome_varb,f_black=pblack_varb,weight=weight_varb,returnSE=False)
        estimate_reg_se = None
    return estimate_chen,estimate_reg,estimate_reg_se


def boundImprovementPlot(resultData,yoffset=0.1,geo_suffix='_cty',variable_suffix='_tn',addCI=False,xlabel='Turnout Difference'):
    fig, ax = plt.subplots(figsize=(10,8))
    data = resultData
    y_pos = np.arange(len(data))
    ax.scatter(data['mob_ub'+geo_suffix+variable_suffix],y_pos+yoffset ,color='red',marker='<')
    ax.scatter(data['mob_lb'+geo_suffix+variable_suffix],y_pos+yoffset ,color='red',marker='>')
    ax.scatter(data['ub'+geo_suffix+variable_suffix],y_pos-yoffset ,color='blue',marker='|',zorder=10)
    ax.scatter(data['lb'+geo_suffix+variable_suffix],y_pos-yoffset ,color='blue',marker='|',zorder=10)
    ax.axvline(0,color='black',linestyle='--')
    ax.scatter(data['truth'+geo_suffix+variable_suffix],y_pos ,color='green',marker='^',label='Ground Truth')
    plt.xlim((-1.01,1))
    for i in range(len(data)):
        if i==0:
            lab1 = 'Old Bounds'
            lab2 = 'Improved Bounds'
            lab3 = 'Improved Bound CI'
        else:
            lab1=None
            lab2=None
            lab3=None
        plt.plot([data.iloc[i]['mob_lb'+geo_suffix+variable_suffix],data.iloc[i]['mob_ub'+geo_suffix+variable_suffix]],[y_pos[i]+0.1,y_pos[i]+0.1],color='red',marker=None,label=lab1)
        plt.plot([data.iloc[i]['lb'+geo_suffix+variable_suffix],data.iloc[i]['ub'+geo_suffix+variable_suffix]],[y_pos[i]-0.1,y_pos[i]-0.1],color='blue',marker=None,label=lab2,zorder=10)
        if addCI:
            plt.plot([data.iloc[i]['lb_ci'+geo_suffix+variable_suffix],data.iloc[i]['ub_ci'+geo_suffix+variable_suffix]],[y_pos[i]-0.1,y_pos[i]-0.1],color='purple',linestyle=':',marker=None,label=lab3,zorder=0)
            ax.scatter(data['ub_ci'+geo_suffix+variable_suffix],y_pos-yoffset ,color='purple',marker='4')
            ax.scatter(data['lb_ci'+geo_suffix+variable_suffix],y_pos-yoffset ,color='purple',marker='3')
        x = (data.iloc[i]['lb'+geo_suffix+variable_suffix]+data.iloc[i]['ub'+geo_suffix+variable_suffix])/2
        y = y_pos[i]
        if addCI:
            ax.annotate('Improvement: '+str(data.iloc[i]['improvementpct'+geo_suffix+variable_suffix].round(3))+'%;'+'CI: '+str(data.iloc[i]['improvementpct_ci'+geo_suffix+variable_suffix].round(3))+'%',
                xy=(x, y), xycoords='data',
                xytext=(0, 20), textcoords='offset pixels',
                horizontalalignment='center',
                verticalalignment='bottom')
        else:
            ax.annotate('Improvement: '+str(data.iloc[i]['improvementpct'+geo_suffix+variable_suffix].round(3))+'%',
                xy=(x, y), xycoords='data',
                xytext=(0, 20), textcoords='offset pixels',
                horizontalalignment='center',
                verticalalignment='bottom')
    plt.legend()
    ax.set_yticks(y_pos, labels=data.State,fontsize=10)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel)
    plt.savefig(outputFolder+'bound_improvement'+geo_suffix+variable_suffix+'.png')
    return fig


def getBounds(dataset,geo_suffix='_cbg',varb_suffix=''):
    reg_varb='reg'+geo_suffix+varb_suffix
    chen_varb = 'chen'+geo_suffix+varb_suffix
    mob_ub_varb = 'mob_ub'+geo_suffix+varb_suffix
    mob_lb_varb = 'mob_lb'+geo_suffix+varb_suffix
    newDat = dataset.copy()
    newDat['ub_pos_case'+geo_suffix+varb_suffix] = newDat[[reg_varb,mob_ub_varb]].min(axis=1)
    newDat['lb_pos_case'+geo_suffix+varb_suffix] = newDat[[chen_varb,mob_lb_varb]].max(axis=1)
    newDat['ub_neg_case'+geo_suffix+varb_suffix] = newDat[[chen_varb,mob_ub_varb]].max(axis=1)
    newDat['lb_neg_case'+geo_suffix+varb_suffix] = newDat[[reg_varb,mob_lb_varb]].min(axis=1)
    newDat['reg_pos'] = np.sign(newDat[reg_varb])
    newDat['ub'+geo_suffix+varb_suffix] = newDat['ub_pos_case'+geo_suffix+varb_suffix]
    newDat.loc[newDat.reg_pos==False,'ub'+geo_suffix+varb_suffix] = newDat['ub_neg_case'+geo_suffix+varb_suffix]
    newDat['lb'+geo_suffix+varb_suffix] = newDat['lb_pos_case'+geo_suffix+varb_suffix]
    newDat.loc[newDat.reg_pos==False,'lb'+geo_suffix+varb_suffix] = newDat['lb_neg_case'+geo_suffix+varb_suffix]
    merged = pd.merge(dataset,newDat[['lb'+geo_suffix+varb_suffix,'ub'+geo_suffix+varb_suffix,'State']],on='State')
    return merged


def constructCI(regVal,chenVal,regSE, reg=True, mult=1, zscore=1.96):
    """constructs confidence interval based on regression SE
    for reg, increase magnitude, for chen, decrease magnitude"""
    if reg:
        return regVal + np.sign(regVal)*zscore*regSE
    else:
        return chenVal - np.sign(chenVal)*zscore*regSE*mult


def constructFinalBounds(data,geo_suffix='_cty',varb_suffix='_tn',zscore=1.96):
    reg_varb='reg'+geo_suffix+varb_suffix
    chen_varb = 'chen'+geo_suffix+varb_suffix
    se_varb = 'se'+geo_suffix+varb_suffix
    var_ratio_varb = 'varianceRatio'+geo_suffix
    pos_varb = 'reg_pos'+geo_suffix+varb_suffix
    reg_ci_varb = 'reg'+geo_suffix+varb_suffix+'_ci'
    chen_ci_varb = 'chen'+geo_suffix+varb_suffix+'_ci'
    data = getBounds(data,geo_suffix,varb_suffix)
    data[reg_ci_varb] = data[[reg_varb,chen_varb,se_varb,pos_varb,var_ratio_varb]].apply(lambda x: constructCI(x[reg_varb],x[chen_varb],x[se_varb],True,x[var_ratio_varb],zscore),axis=1)
    data[chen_ci_varb] = data[[reg_varb,chen_varb,se_varb,pos_varb,var_ratio_varb]].apply(lambda x: constructCI(x[reg_varb],x[chen_varb],x[se_varb],False,x[var_ratio_varb],zscore),axis=1)
    data['ub_ci'+geo_suffix+varb_suffix] = data[[reg_ci_varb,chen_ci_varb]].max(axis=1)
    data['lb_ci'+geo_suffix+varb_suffix] = data[[reg_ci_varb,chen_ci_varb]].min(axis=1)
    data['improvebw'+geo_suffix+varb_suffix] = np.abs(data['ub'+geo_suffix+varb_suffix].clip(upper=1,lower=-1)-data['lb'+geo_suffix+varb_suffix].clip(upper=1,lower=-1))
    data['improvebw_ci'+geo_suffix+varb_suffix] = np.abs(data['ub_ci'+geo_suffix+varb_suffix].clip(upper=1,lower=-1)-data['lb_ci'+geo_suffix+varb_suffix].clip(upper=1,lower=-1))
    data['mobbw'+geo_suffix+varb_suffix] = np.abs(data['mob_ub'+geo_suffix+varb_suffix]-data['mob_lb'+geo_suffix+varb_suffix])
    data['improvementpct'+geo_suffix+varb_suffix] = ((data['mobbw'+geo_suffix+varb_suffix]-data['improvebw'+geo_suffix+varb_suffix])/(data['mobbw'+geo_suffix+varb_suffix])).round(2)*100
    data['improvementpct_ci'+geo_suffix+varb_suffix] = ((data['mobbw'+geo_suffix+varb_suffix]-data['improvebw_ci'+geo_suffix+varb_suffix])/(data['mobbw'+geo_suffix+varb_suffix])).round(2)*100
    return data

def getVarianceRatios(inddata):
    return getVarianceRatio(inddata,'bisgBlackCBG'), getVarianceRatio(inddata,'f_black_cty'),getVarianceRatio(inddata,'f_black_CBG')

def getResults(data,geo_suffix,varb_suffix, bisg=False,weight_varb='voters',hasTruth=False):
    if varb_suffix=='_tn': 
        outcome_varb,totPosOutcomeVarb,posOutcomeBlackVarb ,posOutcomeNonblackVarb,truthBlackVarb,truthNonblackVarb ='f_turnout','totalTurnout','turnedOutAndRegPreAndBlack', 'turnedOutAndRegPreAndNonblack','blackTotRegPre','nonblackTotRegPre'
    elif varb_suffix in ['_dem','']:
        if hasTruth:
            outcome_varb,totPosOutcomeVarb,posOutcomeBlackVarb ,posOutcomeNonblackVarb,truthBlackVarb,truthNonblackVarb ='f_dem','totalDemKnown','demAndBlack', 'demAndNonBlack','blackTotKnown','nonblackTotKnown'
        else:
            outcome_varb,totPosOutcomeVarb,posOutcomeBlackVarb ,posOutcomeNonblackVarb,truthBlackVarb,truthNonblackVarb ='f_dem','totalDem','demAndBlack', 'demAndNonBlack','blackTot','nonblackTot'
    else: 
        if hasTruth:
            outcome_varb,totPosOutcomeVarb,posOutcomeBlackVarb ,posOutcomeNonblackVarb,truthBlackVarb,truthNonblackVarb ='f_dem','totalDemKnown','demAndBlack', 'demAndNonBlack','blackTotKnown','nonblackTotKnown'
        else:
            outcome_varb,totPosOutcomeVarb,posOutcomeBlackVarb ,posOutcomeNonblackVarb,truthBlackVarb,truthNonblackVarb ='f_dem','totalDem','demAndBlack', 'demAndNonBlack','blackTot','nonblackTot'
    if bisg is True:
        pblack_varb,pnonblack_varb='f_black_bisg_cbg','f_nonblack_bisg_cbg'
    else:
        pblack_varb,pnonblack_varb = 'f_black'+geo_suffix,'f_nonblack'+geo_suffix
    chen,reg,se = getEstimatePair(data,outcome_varb=outcome_varb,pblack_varb=pblack_varb,pnonblack_varb=pnonblack_varb,weight_varb=weight_varb,returnSE=True)
    result ={}
    if hasTruth:
        mob_lb, mob_ub = methodOfBoundsByUnit(data,blackTotVarb='blackTotKnown',nonblackTotVarb='nonblackTotKnown',totVarb=totPosOutcomeVarb)
        result['truth'+geo_suffix+varb_suffix] = getTruth(data,posOutcomeAndBlack=posOutcomeBlackVarb,posOutcomeAndNonBlack=posOutcomeNonblackVarb,total_b=truthBlackVarb,total_nb=truthNonblackVarb)
    else:
        mob_lb, mob_ub = methodOfBoundsByUnit(data,blackTotVarb='blackTot',nonblackTotVarb='nonblackTot',totVarb=totPosOutcomeVarb)
        result['truth'+geo_suffix+varb_suffix] = None
    result['chen'+geo_suffix+varb_suffix] = chen
    result['reg'+geo_suffix+varb_suffix] = reg
    result['reg_pos'+geo_suffix+varb_suffix] = reg>0
    result['se'+geo_suffix+varb_suffix] = se
    result['mob_ub'+geo_suffix+varb_suffix] = mob_ub
    result['mob_lb'+geo_suffix+varb_suffix] = mob_lb
    return result

if __name__=='__main__':
    rawDataFolder,cleanDataFolder,outputFolder = getFolderPaths()
    states_with_truth,dem_varbs = setGlobals()
    states_to_do=[state.abbr for state in us.STATES]
    #states_to_do=['AL','NC','SC']
    result_list = []
    county_level_agg = []
    cbg_level_agg = []
    for state in states_to_do:
        county_level = pd.read_csv(cleanDataFolder+'/Clean/'+state+'_agged_county.csv')
        county_level = cleanUnitLevel(county_level)
        cbg_level = pd.read_csv(cleanDataFolder+'/Clean/'+state+'_agged_CBG.csv')
        cbg_level = cleanUnitLevel(cbg_level,pblack_varb='f_black_CBG',pnonblack_varb='f_nonblack_CBG')
        hasTruth = state in states_with_truth
        if hasTruth:
            truth = getTruth(county_level)
            county_level = cleanUnitLevelWithTruth(county_level)
            cbg_level = cleanUnitLevelWithTruth(cbg_level)
            truth_tn = getTruth(county_level,posOutcomeAndBlack='turnedOutAndRegPreAndBlack',posOutcomeAndNonBlack='turnedOutAndRegPreAndNonblack',total_b='blackTotRegPre',total_nb='nonblackTotRegPre')
            truth_tn_cbg = getTruth(cbg_level,posOutcomeAndBlack='turnedOutAndRegPreAndBlack',posOutcomeAndNonBlack='turnedOutAndRegPreAndNonblack',total_b='blackTotRegPre',total_nb='nonblackTotRegPre')
            indlevel=pd.read_csv(cleanDataFolder+'Clean/'+state+'_individual.csv')
            varianceRatio,varianceRatioCty,varianceRatioCBG = getVarianceRatios(indlevel)
        result_cty = getResults(county_level,geo_suffix='_cty',varb_suffix='', bisg=False,weight_varb='voters',hasTruth=hasTruth)
        result_cty_tn = getResults(county_level,geo_suffix='_cty',varb_suffix='_tn', bisg=False,weight_varb='voters',hasTruth=hasTruth)
        result_cbg = getResults(cbg_level,geo_suffix='_cbg',varb_suffix='', bisg=False,weight_varb='voters',hasTruth=hasTruth)
        result_cbg_tn = getResults(cbg_level,geo_suffix='_cbg',varb_suffix='_tn', bisg=False,weight_varb='voters',hasTruth=hasTruth)
        result_bisg = getResults(cbg_level,geo_suffix='_bisg',varb_suffix='', bisg=True,weight_varb='voters',hasTruth=hasTruth)
        result_bisg_tn = getResults(cbg_level,geo_suffix='_bisg',varb_suffix='_tn', bisg=True,weight_varb='voters',hasTruth=hasTruth)
        results = {**result_cty,**result_cty_tn,**result_cbg,**result_cbg_tn,**result_bisg,**result_bisg_tn}
        results['varianceRatio_bisg'] = varianceRatio if state in states_with_truth else np.nan
        results['varianceRatio_cbg'] = varianceRatioCBG if state in states_with_truth else np.nan
        results['varianceRatio_cty'] = varianceRatioCty if state in states_with_truth else np.nan
        results['State'] = state
        result_list.append(results)
        county_level['State'] = state
        cbg_level['State'] = state
        county_level_agg.append(county_level)
        cbg_level_agg.append(cbg_level)


    national_level = pd.concat(county_level_agg)
    national_level_cbg = pd.concat(cbg_level_agg)
    hasTruth = False
    result_cty = getResults(national_level,geo_suffix='_cty',varb_suffix='', bisg=False,weight_varb='voters',hasTruth=hasTruth)
    result_cty_tn = getResults(national_level,geo_suffix='_cty',varb_suffix='_tn', bisg=False,weight_varb='voters',hasTruth=hasTruth)
    result_cbg = getResults(national_level_cbg,geo_suffix='_cbg',varb_suffix='', bisg=False,weight_varb='voters',hasTruth=hasTruth)
    result_cbg_tn = getResults(national_level_cbg,geo_suffix='_cbg',varb_suffix='_tn', bisg=False,weight_varb='voters',hasTruth=hasTruth)
    result_bisg = getResults(national_level_cbg,geo_suffix='_bisg',varb_suffix='', bisg=True,weight_varb='voters',hasTruth=hasTruth)
    result_bisg_tn = getResults(national_level_cbg,geo_suffix='_bisg',varb_suffix='_tn', bisg=True,weight_varb='voters',hasTruth=hasTruth)
    results = {**result_cty,**result_cty_tn,**result_cbg,**result_cbg_tn,**result_bisg,**result_bisg_tn}
    results['State'] = 'Natl'
    result_list.append(results)

    results=pd.DataFrame(result_list)

    data = results.sort_values('reg_cty',ascending=False)
    data.loc[data.State=='Natl',['varianceRatio_bisg','varianceRatio_cbg','varianceRatio_cty']]=1    
    data = constructFinalBounds(data,geo_suffix='_cty',varb_suffix='')
    data = constructFinalBounds(data,geo_suffix='_cbg',varb_suffix='')
    data = constructFinalBounds(data,geo_suffix='_cty',varb_suffix='_tn')
    data = constructFinalBounds(data,geo_suffix='_cbg',varb_suffix='_tn')
    important = data[data.State.isin(['AL','NC','SC','Natl'])].copy().set_index('State').reindex(['AL','NC','SC','Natl']).reset_index()
    important.to_csv(outputFolder+'result_table.csv')
    boundImprovementPlot(important,geo_suffix='_cty',variable_suffix='',addCI=True,xlabel='Dem. Registration Difference')
    boundImprovementPlot(important,geo_suffix='_cty',variable_suffix='_tn',addCI=True)
    boundImprovementPlot(important,geo_suffix='_cbg',variable_suffix='',addCI=True,xlabel='Dem. Registration Difference')
    boundImprovementPlot(important,geo_suffix='_cbg',variable_suffix='_tn',addCI=True)
    # pdb.set_trace()
    # boundImprovementPlot(important,geo_suffix='_bisg',variable_suffix='',addCI=True)
