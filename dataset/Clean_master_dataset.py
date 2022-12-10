#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../src/')

import data_io
import pandas as pd
import numpy as np


# In[ ]:


state_list = ["AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA",
              "HI","IA","ID", "IL","IN","KS","KY","LA","MA","MD","ME",
              "MI","MN","MO","MS","MT","NC","ND","NE","NH",
              "NJ","NM","NV","NY", "OH","OK","OR","PA",
              "RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]

MIN_YEAR = 2010
MAX_YEAR = 2019
EXCEL_OPTIONS = {'strings_to_urls': False,
                'strings_to_formulas': False}


# In[ ]:


states_abbr_to_name = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


# ### 1. Define cleaning functions

# In[ ]:


def isBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return False
    #myString is None OR myString is empty or blank
    return True

def remove_leading_whitespace(df, column):
    new_series = pd.Series(len(df))
    i = 0
    for i in range(0, len(df)):
        old_str = df.loc[i, column]
        new_str = old_str.lstrip()
        new_series[i] = new_str

    return new_series

def get_loc_string(test1):

    test1 = test1.split('\r\n\xa0 ')
    test1 = test1[1]

    i = 0
    for i in range(0, len(test1) - 1):
        if test1[i].isspace() and test1[i+1].isspace():
            test1 = test1[0:i]
            break
    return test1

def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def extract_year(x):
    if type(x) != str:
        return np.nan
    else:
        x = x.lower()
    if x == 'none' or 'invalid date' in x or x == 'Created':
        return np.nan
    else:
        #print(x[-4:])
        try:
            new_var = int(x[-4:])
        except:
            return np.nan
        return new_var


def extract_whole_date(x):
    if type(x) != str or 'none' in x:
        return np.nan
    else:
        return x[8:]


def get_social(x):
    if type(x) != str:
        return x
    x = x.lower()
    if 'none' in x or pd.isnull(x):
        return np.nan
    if ',' in x:
        x = x.replace(',', '')
    if ' shares' in x:
        x = x.replace(' shares', '')
    if 'total' in x:
        x = x[0:x.find('total')]
        
    if ' share' in x:
        x = x.replace(' share', '')
    if ' followers' in x:
        x = x.replace(' followers', '')
    if ' follower' in x:
        x = x.replace(' follower', '')
    if 'k' in x:
        new = x[0:x.find('k')]
        if '.' in new:
            trail = '00'
            new = new.replace('.', '')
        else:
            trail = '000'
            
        return new+trail
    if 'friend' in x or 'comment' in x:
        return np.nan
    return x



def get_other_loc(x):
    if type(x) != str or x == 'none':
        return np.nan
    if ',' in x:
        split_str = x.split(',')
        return split_str[0].lower()
    else:
        return x.lower()

def get_state_var(x):
    if x == 'none':
        return np.nan

    if type(x) == str:

        if ',' in x:
            new_list = x.split(',')
            state_str = new_list[1]
            if isBlank(state_str):
                return np.nan
            else:
                return state_str.strip()
        else:
            return x[-2:]
    else:
        return np.nan
    
    
def extract_deprecated_val(x):
    if 'k' in x:
        x = x[0:x.find('k')]
        if '.' in x:
            trail = '00'
            x = x.replace('.', '')
        else:
            trail = '000'

    elif 'm' in x:
        x = x[0:x.find('m')]
        if '.' in x:
            trail = '00000'
            x = x.replace('.', '')
        else:
            trail = '000000'
    elif 'b' in x:
        x = x[0:x.find('b')]
        if '.' in x:
            trail = '00000000'
            x = x.replace('.', '')
        else:
            trail = '000000000'
    goal = x+trail

    goal = int(only_numerics(goal))
    return goal


def get_money_raised(x):
    if type(x) != str or 'none' in x:
        return np.nan
    elif '%' in x:
        return np.nan
    elif '$' not in x:
        return 'NOT USD'
    else:
        x = x.lower()
        if 'of' in x:
            new_info = x.split('of')
            try:
                if 'k' in new_info[0] or 'm' in new_info[0]:
                    money_raised = extract_deprecated_val(new_info[0])
                else:
                    money_raised = int(only_numerics(new_info[0]))
            except:
                print('failed to get money raised: ', x)
                money_raised = np.nan

        elif 'raised' in x:
            if 'goal' in x:
                new = x.split('\n')
                this_str = new[0]
                this_str = this_str[this_str.find('$'):]
                if '.' in this_str:
                    new = this_str[0:this_str.find('.')]
                    if 'k' in new or 'm' in new:
                        money_raised = extract_deprecated_val(new)
                    else:
                        money_raised = int(only_numerics(new))
            else:
                try:
                    if 'k' in x or 'm' in x:
                        money_raised = extract_deprecated_val(x)
                    else:
                        money_raised = int(only_numerics(x))
                except:
                    print('failed to get money raised: ', x)
                    money_raised = np.nan

        else:
            return np.nan

        return money_raised

    
def get_goal(x):
    if type(x) != str:
        return np.nan
    if '%' in x:
        return np.nan
    if '$' not in x:
        return 'NOT USD'
    x = x.lower()
    if 'raised' in x and 'of' not in x:
        if 'goal' in x:
            new = x.split('\n')
            new = new[1]
            new = new[new.find('$'):]
            if 'k' in new or 'm' in new or 'b' in new:
                goal = extract_deprecated_val(new)

            else:
                if '.' in x:
                    new = new[0:new.find('.')]
                goal = int(only_numerics(new))
            return goal

        else:
            return np.nan
    else:
        if 'of' in x:
            new_info = x.split('of')
            new = new_info[1]
            if 'k' in new or 'm' in new or 'b' in new:
                goal = extract_deprecated_val(new)
            else:
                if '.' in new:
                    new = new[0:new.find('.')]
                try:
                    goal = int(only_numerics(new))
                except:
                    goal = 'failed'
            return goal

        elif 'goal' in x:
            return int(only_numerics(x))
        else:
            print('failed to parse goal: ', x)
            
def get_num_contributors(x):

    if type(x) == str and x != 'none':
        x = x.lower()
        if 'raised' in x and '$' not in x:
            new = x.split('in')
            if 'k' in new:
                new = extract_deprecated_val(new)
            else:
                new = int(only_numerics(new[0]))
            return new
        elif 'donor' in x and 'day' not in x and 'month' not in x:
            if 'k' in x:
                new = extract_deprecated_val(x)
            else:
                new = int(only_numerics(x))
            return new
        elif 'people' in x or 'person' in x:
            if 'by' in x:
                str_split1 = x.split('by')
                if 'in' in x:
                    str_split2 = str_split1[1].split('in')
                    new = str_split2[0]
                    if 'k' in new:
                        new = extract_deprecated_val(new)
                    else:
                        new = int(only_numerics(new))
                    return new
                else:
                    new = str_split1[1]
                    if 'k' in new:
                        new = extract_deprecated_val(new)
                    else:
                        new = int(only_numerics(new))
                    return new
            else:
                print(x)
                return x
            
            
        else:
            return np.nan
    else:
        return np.nan
    
def remove_non_loc_info(x):
    if type(x) == str:
        temp = x.lower()
        if 'donations' in temp and ' 1 donation' not in temp:
            #print('donations in x')
            loc = temp.find('donations')
            delete = loc+len('donations')
            temp = temp[delete:]
            #return new
        if '1 donation' in temp:
            #print('donation in x')
            loc = temp.find('donation')
            delete = loc+len('donation')
            temp = temp[delete:]
            #return new
        if 'organizer' in temp:
            #print('organizer in x')
            loc = temp.find('organizer')
            delete = loc+len('organizer')
            temp = temp[delete:]

        return temp


# In[ ]:


#regex cleaning functions
import re 
def contruct_goal_pattern():
    rtypes = [] # (type of values returned (raise,goal,both) , notation that money is recorded in (US vs foreign))
    rpatterns = []
    rpatterns.append(r'(.*)raised of(.*)goal')
    rpatterns.append(r'(.*)of(.*)goal')
    rpatterns.append(r'(.*)of(.*)')
    rpatterns.append(r'(.*)raised of(.*)target')
    rpatterns.append(r'Raised:(.*)Goal:(.*)')
    rtypes+=[['both','US']] * 5
    
    rpatterns.append(r'(.*)des Ziels von(.*)') # german
    rpatterns.append(
        r'(.*)sur un objectif de(.*)') # french
    rpatterns.append(r'(.*)del objetivo de(.*)') # spanish
    rpatterns.append(r'(.*)da meta de(.*)') # romanian
    rpatterns.append(r'(.*)su(.*)raccolti') # italian
    rpatterns.append(r'(.*)van het doel(.*)') # dutch
    rtypes+=[['both','foreign']] * 6 

    rpatterns.append(r'(.*)raised')
    rtypes+=[['raised','US']]
    
    rpatterns.append(r'(.*)réunis') # french
    rpatterns.append(r'(.*)gesammelt') # german
    rpatterns.append(r'(.*)recaudados') # spanish
    rpatterns.append(r'(.*)arrecadados') # portugese
    rpatterns.append(r'(.*)raccolti') # italian
    rtypes+=[['raised','foreign']]*5

    rpatterns.append(r'(.*)goal')
    rpatterns.append(r'(.*)target')
    rtypes+=[['goal','US']]*2 
    
    rpatterns.append(r'Objectif\s*:(.*)') # french
    rpatterns.append(r'Objetivo\s*:(.*)') #spanish
    rpatterns.append(r'(.*)Ziel') # german
    rpatterns.append(r'Meta de(.*)') #romanian
    rpatterns.append(r'(.*)obiettivo') # italian
    rtypes+=[['goal','foreign']]*5 
    patterns_collection = pd.Series(rtypes, index=rpatterns, name='rtype')
    return patterns_collection


GOAL_PATTERNS = contruct_goal_pattern()

_clean_whitespace = lambda x: re.sub(r'\s+', ' ', x).strip()

THOUNDSAND_PATTERN = re.compile(r'\d+[,.]*\d*.*[k]')
MILLION_PATTERN = re.compile(r'\d+[,.]*\d*.*[m]')
BILLION_PATTERN = re.compile(r'\d+[,.]*\d*.*[b]')
MONEY_PATTERN = re.compile(r"""( #start of group0, this is the desired output
                                \d+ #start digit of money amount, mustbe followed by abbr, number or marker, nonwords or end of string
                                ((?<=\d)[,.]\d+)*  #(group1) this is an optional group that only appears if markers are present
                                ((?<=\d)[kmbKMB](?=\W|$)){0,1} #(group2)match thousand,mill,bill abbreviation if present but only if theres one of them
                                )#close group0
                            """,re.VERBOSE)
_remove_whitespace_inside_money = lambda x: re.sub(r'(?<=\d|[,.])\s(?=\d|[,.]|[kmbKMB](?=\W|$))','',x)
_extract_money_amount = lambda x: MONEY_PATTERN.findall(_remove_whitespace_inside_money(x))
def _switch_markers_to_us_notation(amnt):
    chars = []
    for c in amnt:
        if c == ',':
            chars.append('.')
        elif c == '.':
            chars.append(',')
        else:
            chars.append(c)
    return ''.join(chars)

def parse_money_into_floats(x,us_notation=True,switch_retry=True):
    out = {'amount':np.nan,'currency':np.nan}
    if pd.isnull(x): return out
    old_x = x
    x = x.strip().lower()
    if len(x) == 0: return out
    try:
        amnt = _extract_money_amount(x)[0][0]
        curr = x.replace(amnt,'').strip()
        if not us_notation:
            # money amount written in foreign notation
            # need to swap , and . 
            amnt = _switch_markers_to_us_notation(amnt)
        numeric_amnt = ''.join(re.findall('\d*|[,.]*', amnt))
        numeric_amnt = float(numeric_amnt.replace(',', ''))
        trail = 1
        if THOUNDSAND_PATTERN.search(amnt):
            trail = 1000
        elif MILLION_PATTERN.search(amnt):
            trail = 1000000
        elif BILLION_PATTERN.search(amnt):
            trail = 1000000000
        out['amount']=numeric_amnt * trail
        out['currency'] = curr
        return out
    except:
        if switch_retry:
            print(f'[WARNING] failed to parse {old_x} but will retry by swapping , and .')
            # ~ doesnt work, have to be not 
            out = parse_money_into_floats(x,us_notation=not us_notation,switch_retry=False)
            if not pd.isna([*out.values()]).all():
                print('[WARNING] parsed results might be inaccurate, check below')
                print(f"[RETRY OUTPUT] original:{x}|parsed_amnt:{out['amount']}|parsed_currency:{out['currency']}")
        else:
            print(f'failed to parse original x:{old_x}|stripped:{x}')
        return out
    
def get_raised_and_goal_amount(x, USD_only=True):
    import re
    out = {"raised": np.nan, "goal": np.nan,"raised_amnt":np.nan,
           "raised_curr":np.nan,"goal_amnt":np.nan,"goal_curr":np.nan}
    if x == 'none': return out
    if USD_only:
        if '$' not in x: return out
    x = _clean_whitespace(x)
    for rpattern, rtype in GOAL_PATTERNS.iteritems():
        results = re.findall(rpattern, x)
        if len(results) > 0:
            results = results[0]  # pop out results
            rtype_value,rtype_notation = rtype[0],rtype[1]
            if rtype_value == 'both':
                out["raised"], out["goal"] = results[0], results[1]
                for k in ["raised","goal"]:
                    results = parse_money_into_floats(out[k],us_notation=rtype_notation=='US')
                    out[k+"_amnt"],out[k+"_curr"] = results["amount"],results["currency"]
            elif rtype_value == "raised":
                out["raised"] = results
                results = parse_money_into_floats(out["raised"],us_notation=rtype_notation=='US')
                out["raised_amnt"],out["raised_curr"] = results["amount"],results["currency"]
            elif rtype_value == "goal":
                out["goal"] = results
                results = parse_money_into_floats(out["goal"],us_notation=rtype_notation=='US')
                out["goal_amnt"],out["goal_curr"] = results["amount"],results["currency"]
            break
    if pd.isna([*out.values()]).all(): print(f'failed to parse {x}')
    return out



def standardize_MBk_in_number_str(x):
    if pd.isnull(x): return x
    old_x = x
    x = x.strip().lower()
    if len(x) == 0: return np.nan
    try:
        x_i = re.findall('\d+[,.]*\d*', x)[0]
        x_i = float(x_i.replace(',', ''))
        trail = 1
        if THOUNDSAND_PATTERN.search(x):
            trail = 1000
        elif MILLION_PATTERN.search(x):
            trail = 1000000
        elif BILLION_PATTERN.search(x):
            trail = 1000000000
        return x_i * trail
    except:
        print(f'original x:{old_x}|stripped:{x}')
        return np.nan

def contruct_date_pattern():
    rpatterns = []
    rtypes = []
    rpatterns.append(r'Created ([a-zA-Z]+) (\d+), (\d+)')
    rtypes.append(['month', 'day', 'year'])
    rpatterns.append(r'Created (\d+) ([a-zA-z]+) (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Created by .*?on ([a-zA-z]+) (\d+), (\d+)')
    rtypes.append(['month', 'day', 'year'])
    rpatterns.append(r'Erstellt am (\d+). (\S+) (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Date de création : (\d+) (\S+) (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Fecha de creación: (\d+) de (\S+) de (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Creata il (\d+) (\S+) (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Gemaakt op (\d+) (\S+) (\d+)')
    rtypes.append(['day', 'month', 'year'])
    rpatterns.append(r'Criada em (\d+) de (\S+) de (\d+)')
    rtypes.append(['day', 'month', 'year'])
    # special case: put this in day field for later processing with archive_timestamp
    # double parenthesis to match regex findall output as other patterns
    rpatterns.append(r'Created ((\d+ days ago))')
    rtypes.append(['day', 'day'])
    return pd.Series(rtypes, index=rpatterns, name='rtype')

DATE_PATTERNS = contruct_date_pattern()

def parse_created_date(x):
    out = {"day": np.nan, "month": np.nan, "year": np.nan}
    if x == 'none': return out
    x = _clean_whitespace(x)
    if x.find('Invalid date') > -1: return out
    if x == 'Created': return out
    for rpattern, rtype in DATE_PATTERNS.iteritems():
        results = re.findall(rpattern, x)
        if len(results) > 0:
            results = results[0]  # pop out results
            for k, v in zip(rtype, results):
                out[k] = v
            break
    if pd.isna([*out.values()]).all(): print(f'failed to parse {x}')
    return out

def construct_status_pattern():
    rpatterns = []
    rtypes = []
    rpatterns.append(r'^(\S+) donor$') # ^ and $ help make match the whole string
    rtypes.append(['ndonor'])
    rpatterns.append(r'raised by (\S+) donor in \S+? duration')
    rtypes.append(['ndonor'])
    rpatterns.append(r'\S+? raised by (\S+) donor in \S+? duration')
    rtypes.append(['ndonor'])

    rpatterns.append(r'campaign created .*?duration ago')
    rtypes.append([])
    rpatterns.append(r'^recent donor [(](\S+)[)]$')
    rtypes.append(['ndonor'])
    rpatterns.append(r'goal reached!')
    rtypes.append([])
    rpatterns.append(r'campaign ended')
    rtypes.append([])
    rpatterns.append(r'only \S+? duration left to reach goal!')
    rtypes.append([])
    rpatterns.append(r'be the first to like this donor \S+? duration ago')
    rtypes.append([])
    rpatterns.append(r'\S+? donor likes this donor \S+? duration ago')
    rtypes.append([])

    rpatterns.append(r'gesammelt von (\S+) donore{0,1}n{0,1} in \S+? tage{0,1}n{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'gesammelt von (\S+) donore{0,1}n{0,1} in \S+? monate{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'gesammelt von (\S+) donore{0,1}n{0,1} in \S+? stunde{0,1}n{0,1}')
    rtypes.append(['ndonor'])

    rpatterns.append(r'(\S+) donornes ont fait un don en \S+? mois{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'(\S+) donorne a fait un don en \S+? mois{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'(\S+) donornes ont fait un don en \S+? jours{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'(\S+) donorne a fait un don en \S+? jours{0,1}')
    rtypes.append(['ndonor'])

    rpatterns.append(r'recaudados de (\S+) donoras en \S+? mese{0,1}s{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'recaudados de (\S+) donoras en \S+? días{0,1}')
    rtypes.append(['ndonor'])

    rpatterns.append(r'recolectados de (\S+) donoras{0,1} en \S+? días{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'recolectados de (\S+) donoras{0,1} en \S+? mese{0,1}s{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'recolectados de (\S+) donoras{0,1} en \S+? horas{0,1}')
    rtypes.append(['ndonor'])


    rpatterns.append(r'donati da (\S+) donor[ae] in \S+? mesi')
    rtypes.append(['ndonor'])
    rpatterns.append(r'donati da (\S+) donor[ae] in \S+? ore')
    rtypes.append(['ndonor'])
    rpatterns.append(r'donati da (\S+) donor[ae] in \S+? giorni')
    rtypes.append(['ndonor'])

    rpatterns.append(r'arrecadados por (\S+) pessoas em \S+? meses')
    rtypes.append(['ndonor'])
    rpatterns.append(r'arrecadados por (\S+) pessoas em \S+? dias')
    rtypes.append(['ndonor'])


    rpatterns.append(r'not launched yet!')
    rtypes.append([])
    rpatterns.append(r'campagne créée depuis \S+? mois')
    rtypes.append([])
    rpatterns.append(r'kampagne vor \S+? monate erstellt')
    rtypes.append([])
    rpatterns.append(r'campagna creata \S+? giorni fa')
    rtypes.append([])
    rpatterns.append(r'la campaña se creó hace \S+? días')
    rtypes.append([])
    rpatterns.append(r'ingezameld door (\S+) donoren binnen \S+? maanden')
    rtypes.append(['ndonor'])


    rpatterns.append(r'la campaña se creó hace \S+? mese{0,1}s{0,1}')
    rtypes.append([])
    rpatterns.append(r'kampagne vor \S+? monate{0,1} erstellt')
    rtypes.append([])
    rpatterns.append(r'kampagne vor \S+? tage{0,1}n{0,1} erstellt')
    rtypes.append([])
    rpatterns.append(r'campanha criada \S+? dias atrás')
    rtypes.append([])
    rpatterns.append(r'ingezameld door (\S+) donoren binnen \S+? dagen')
    rtypes.append(['ndonor'])
    rpatterns.append(r'la campaña se creó hace \S+? horas{0,1}')
    rtypes.append([])
    rpatterns.append(r'(\S+) donorne a fait un don en \S+? mois{0,1}')
    rtypes.append(['ndonor'])
    rpatterns.append(r'campagne créée depuis \S+? jours{0,1}')
    rtypes.append([])


    return pd.Series(rtypes, index=rpatterns, name='rtype')



STATUS_PATTERNS = construct_status_pattern()


def parse_status(x):
    out = {'ndonor': np.nan}
    if x == 'none': return out
    parsed = False
    for rpattern, rtype in STATUS_PATTERNS.iteritems():
        results = re.findall(rpattern, x)
        if len(results) > 0:
            for k, v in zip(rtype, results):
                out[k] = v
            parsed = True
            break
    if (not parsed) & pd.isna([*out.values()]).all():
        print(f'failed to parse {x}')
    return out

# remove period at end, lowercase everything, clean whitespace characters
_reformat_status = lambda x: _clean_whitespace(x[:-1].lower() if x[-1]=='.' else x.lower())
# replace variation of units with standard words
_duration_regex = re.compile(r'months*|days*|mins*|hours*')
_donor_regex = re.compile(r'people|person|donors*|donations*|supporters*')
_standardize_nouns = lambda x: _donor_regex.sub('donor',_duration_regex.sub('duration',x))
# piece it all together
standardize_status = lambda x: _standardize_nouns(_reformat_status(x))


# In[ ]:


def construct_weird_title_type_pattern():
    rpatterns=[]
    rtypes=[]
    rpatterns.append(r'^Page Not Found$')
    rpatterns.append(r'^Unknown Error$')
    rpatterns.append(r'^502 Bad Gateway$')
    rpatterns.append(r'^404 Not Found$')
    rpatterns.append(r'^403 Forbidden$')
    rtypes += ['error']*5

    rpatterns.append(r'^none$')
    rtypes+= ['missing']

    # gfm logistic
    rpatterns.append(r'- Local Widget Builder$')
    rpatterns.append(r'(.*)GoFundMe Support$')
    rtypes += ['logistic']*2

    # general home pages
    rpatterns.append(r'^GoFundMe, le 1er site de crowdfunding pour créer une cagnotte en ligne$')
    rpatterns.append(r'^GoFundMe : la plateforme gratuite n°1 de la collecte de fonds$')
    rpatterns.append(r'^GoFundMe, le site n°1 de financement participatif et de collecte de fonds en ligne sans frais de plateforme$')
    rpatterns.append(r'^Donate Online [|] Make Online Donations to People You Know!$')
    rpatterns.append(r'^GoFundMe: Top-Website für Crowdfunding und Fundraising$')
    rpatterns.append(r'^GoFundMe – die weltgrößte Crowdfunding-Seite zum Spendensammeln$')
    rpatterns.append(r'^Funding(.*)[|] Fundraising - GoFundMe$')
    rpatterns.append(r'^Raise Money For (.*?)[|](.*?)Fundraising - GoFundMe$')
    rpatterns.append(r'^Personal & Charity Online Fundraising Websites that WORK!$')
    rpatterns.append(r'(.*?)Fundraising - Start a Free Fundraiser$')
    rpatterns.append(r'^Fundraising für (.*?)[|] Sammle Geld für(.*?)[|] GoFundMe$')
    rpatterns.append(r'^Top Crowdfunding-Seite zum Spendensammeln – GoFundMe$')
    rpatterns.append(r'^Personal Online Fundraising Websites that Work[!]$')
    rpatterns.append(r'^Raise Money for YOU!(.*)!')
    rpatterns.append(r'^GoFundMe:(.*)1')
    rpatterns.append(r'^Raise money for your(.*?)Ideas!$')
    rpatterns.append(r'^Raise Money for(.*)[|] GoFundMe$')
    rpatterns.append(r'^Fundraising Ideas for(.*)')
    rpatterns.append(r'(.*)Fundraising [|] Raise Money for(.*)[|] GoFundMe$')
    rpatterns.append(r'(.*)Fundraising: Raise Money for (.*)')
    rpatterns.append(r'(.*)Fundraising [|] Crowdfunding for(.*)– Free at GoFundMe$')
    rpatterns.append(r'^Fundraising Ideas for(.*)')
    rpatterns.append(r'^Find success with these Creative Fundraising Idea$')
    rpatterns.append(r'^(.*)Fundraising [|] Fundraiser - GoFundMe[!]$')
    rtypes+=['homepage']*24
    return pd.Series(rtypes, index=rpatterns, name='rtype')
WEIRD_TITLE_TYPE_PATTERNS = construct_weird_title_type_pattern()

def detect_weird_title_type(x):
    out = {'type':np.nan}
    for rpattern, rtype in WEIRD_TITLE_TYPE_PATTERNS.iteritems():
        if re.search(rpattern,x):
            out['type'] = rtype
            break
    return out

def construct_title_pattern():
    rpatterns =[]
    rtypes=[]
    rpatterns.append(r'^Fundraiser by(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'(.*)by(.*)- GoFundMe$')
    rtypes.append(['campaign_title','organizer'])
    rpatterns.append(r'^Fundraiser for(.*?)by(.*?):(.*)')
    rtypes.append(['benefiter','organizer','campaign_title'])
    rpatterns.append(r'^Collecte de fonds pour(.*?)organisée par(.*?):(.*)')
    rtypes.append(['benefiter','organizer','campaign_title'])
    rpatterns.append(r'^Spendenkampagne von(.*?)für(.*?):(.*)')
    rtypes.append(['organizer','benefiter','campaign_title'])
    rpatterns.append(r'^Campanha de arrecadação de fundos para(.*?)por(.*?):(.*)')
    rtypes.append(['organizer','benefiter','campaign_title'])
    rpatterns.append(r'^Campaña de(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Campanha de arrecadação de fundos de (.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Cagnotte organisée par(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Spendenkampagne von(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Collecte de fonds organisée par(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Inzamelingsactie van(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Raccolta fondi di(.*?):(.*)')
    rtypes.append(['organizer','campaign_title'])
    rpatterns.append(r'^Cagnotte pour(.*?)organisée par(.*?):(.*)')
    rtypes.append(['benefiter','organizer','campaign_title'])
    rpatterns.append(r'^Inzamelingsactie voor(.*?)van(.*?):(.*)')
    rtypes.append(['benefiter','organizer','campaign_title'])
    return pd.Series(rtypes, index=rpatterns, name='rtype')
TITLE_PATTERNS = construct_title_pattern()

_remove_newline = lambda x: ' '.join(x.split()).strip()
def parse_title(x):
    out = {'benefiter': np.nan,'organizer':np.nan,'campaign_title':np.nan,'campaign_title_type':np.nan}
    if x == 'none': return out
    parsed = False
    x = _remove_newline(x)
    out['campaign_title_type'] = detect_weird_title_type(x)['type']
    if not pd.isnull(out['campaign_title_type']): 
        return out
    else:
        out['campaign_title_type'] = 'campaign'
    for rpattern, rtype in TITLE_PATTERNS.iteritems():
        results = re.findall(rpattern, x)
        if len(results) > 0:
            results=results[0]
            for k, v in zip(rtype, results):
                out[k] = v.strip()
            parsed = True
            break
    if not parsed:
        out['campaign_title'] = x
    return out


# ### 2. Define validation functions

# In[ ]:


def state_is_valid(x):
    if type(x) != str:
        return False
    if x not in state_list:
        return False
    else:
        return True
#searches for determinig whether campaign mentions cancer
SEARCHES = pd.read_csv(data_io.input_cleaned/'gfm'/'cancer_search_terms.csv', encoding='utf-8')
CANCER_SEARCHES = SEARCHES['cancer_type'].to_list()


def find_cancer_story_title(story, title):
    
    story_truth = True if type(story) == str else False
    title_truth = True if type(title) == str else False

    if story_truth == False and title_truth == False:
        return False
    else:
        new_story = story.lower() if story_truth == True else 'bad'
        new_title = title.lower() if title_truth == True else 'bad'
        if any(i in new_story for i in CANCER_SEARCHES):
            return True
        if any(i in new_title for i in CANCER_SEARCHES):
            return True
        
    return False

def cancer_in_x(x):
    if type(x) == str:
        x = x.lower()
        if any(i in x for i in CANCER_SEARCHES):
            return True
    return False
    
def immunotherapy_mention(x):
    if type(x) == str:
        if 'immunotherap' in x or 'immuno therap' in x:
            return True
        else:
            return False
    else:
        return False
    
def is_english(x):
    if type(x) == str:
        x = x.lower()
        if 'des ziels' in x:
            return False
        elif 'gesammelt' in x:
            return False
        elif 'recolectados' in x:
            return False
        elif 'del objetivo' in x:
            return False
        elif 'da meta de' in x:
            return False
        else:
            return True
        
        
def tag_is_valid(x):
    tag = x.lower()
    if 'medical' in tag:
        return True
    elif 'emergenc' in tag:
        return True
    elif 'family' in tag:
        return True
    elif 'community' in tag:
        return True
    elif 'other' in tag:
        return True
    else:
        return False
    
def assign_new_tag(x):
    tag = x.lower()
    if 'medical' in tag:
        return 'medical'
    else:
        return np.nan
    

def medical_in_tag(x):
    if type(x) == str:
        if 'medical' in x:
            return True
        else:
            return False
    return False

def check_for_spain(x, currency):
    if type(currency)== str:
        if currency == 'NOT USD':
            if type(x) == str:
                x = x.lower()
                if 'ct, spain' in x or ('barcelona' in x and 'spain' in x):
                    return True
    return False


# ### 3. Read in master dataframe and set up exclusion tracker
# 
# Master dataframe is generated by Make master tables.ipynb

# In[ ]:


MASTER_DF = pd.read_csv(data_io.input_raw/'gfm'/'all_output_no_duplicate.csv', 
                        encoding='utf-8', sep='|', index_col=[0], dtype=str)
original_length = len(MASTER_DF)


# In[ ]:


MASTER_DF.columns


# In[ ]:


#make a copy to be double sure you don't change master
df = MASTER_DF.copy()


# ### 4a. Parse variables using regex

# In[ ]:


use_tqdm=True
if use_tqdm: from tqdm import trange, tqdm


# #### Parse out title for benefiter, organizer, and campaign_title

# In[ ]:


if use_tqdm: 
    tqdm.pandas(desc='Parsing title')
    title_parsed_dicts=df.title.progress_apply(parse_title)
else:
    title_parsed_dicts=df.title.apply(parse_title)
title_parsed_df=pd.DataFrame.from_records(title_parsed_dicts,index=title_parsed_dicts.index)
title_parsed_df.head()


# Join parsed results into df

# In[ ]:


parsed_cols = ['benefiter','organizer','campaign_title','campaign_title_type']
df.drop(columns=parsed_cols,errors='ignore',inplace=True)
df = df.merge(title_parsed_df[parsed_cols],on='campaign_id',how='left',indicator=True)
# See merge results
print(df._merge.value_counts())
df.drop(columns='_merge',inplace=True)


# In[ ]:


#Review parsed results by diff campaign_title_type
df.loc[:,['title','benefiter','organizer','campaign_title','campaign_title_type']].groupby('campaign_title_type').head(3)


# #### Parse day,month & year from created_date

# In[ ]:


if use_tqdm: 
    tqdm.pandas(desc='Parsing date')
    date_parsed_dicts = df.created_date.progress_apply(parse_created_date)
else:
    date_parsed_dicts = df.created_date.apply(parse_created_date)
date_parsed_df = pd.DataFrame.from_records(date_parsed_dicts,index=date_parsed_dicts.index)                                           


# Some created_date have values "Create X days ago" and was parsed into the "day" field, fix this using archive_timestamp. these values are usually for campaign in 2019

# In[ ]:


# Fix the "days ago rows"
days_ago_m = date_parsed_df.day.str.find('days ago') > -1
days_ago_rows = date_parsed_df[days_ago_m]
# Parse day month year from archived timestamp 
archived_ts= pd.DataFrame.from_records(df.loc[days_ago_rows.index, 'archive_timestamp'].apply(
    lambda x: {
        'day': x[6:8],
        'month': x[4:6],
        'year': x[0:4]
    }),index=days_ago_rows.index)
archived_ts['day'] = archived_ts['day'].astype(int)-days_ago_rows.day.apply(lambda x: int(x.replace('days ago','')))
archived_ts['day']= archived_ts['day'].astype(str)
# Update day month year fields 
date_parsed_df.update(archived_ts,overwrite=True)
date_parsed_df.head()


# Join parsed results into df

# In[ ]:


parsed_cols = ['day','month','year']
df.drop(columns=parsed_cols,errors='ignore',inplace=True) #drop if exists
df = df.merge(date_parsed_df[['day','month','year']],on='campaign_id',how='left',indicator=True)
# See merge results
print(df._merge.value_counts())
df.drop(columns='_merge',inplace=True)
df.loc[:,['created_date','day','month','year']].head()


# Map non-English months to standard English months

# In[ ]:


month_map_dict = {'février':'february','octobre':'october','juli':'july','junho':'june','09':'september','abril':'april',
'março':'march','mars':'march','februar':'february','januar':'january', 'avril':'april','juin':'june',
'juillet':'july','augustus':'august','mai':'may','mai':'may','märz':'march','juni':'June',
'settembre':'september','gennaio':'january','septiembre':'september','mayo':'may',
'décembre':'december','nisan':'April','maggio':'may','febbraio':'february',
'marzo':'march','janvier':'january','dezember':'december','novembro':'november',
'febrero':'february','aprile':'april','maio':'may','novembre':'november','mei':'may',
'septembre':'september','oktober':'october','junio':'june','enero':'january','februari':'february','januari':'january',
'fevereiro':'february','noviembre':'november','giugno':'june','agosto':'august'}


# In[ ]:


# map values
mapped = df.month.str.lower().map(month_map_dict).str.capitalize()
# if value was mapped keep the value, else keep original 
df.month=mapped.where(mapped.notna(),df.month)


# Check to make sure non-English month was mapped correctly to english month

# In[ ]:


df.month.unique() # only english months should show up


# #### Check for duplicates by campaign_title, organizer and parsed dates

# In[ ]:


gb=df.groupby(['campaign_title','organizer','day','month','year']).size().reset_index()
po_dup_groups = gb[gb[0]>1]
po_dups=df.merge(po_dup_groups,on=['campaign_title','organizer','day','month','year']).sort_values(['campaign_title','organizer','day','month','year'])


# Check why there's still duplicates even tho we already stratified by cleaned_title,created_date, and lowered-case location in Make_master_table.ipynb

# In[ ]:


po_dups[['created_date','location','campaign_title','organizer','day','month','year']].head(10)


# Duplicates happened because of different original location or created_date, replace 'Organizer' values in location with 'none' and we'll keep the best duplicate in step 5a

# In[ ]:


df.location = df.location.replace('Organizer','none')


# Find if any is in a non-English language

# In[ ]:


show_cols =['gfm_url','cleaned_title','created_date','location','campaign_title','organizer','day','month','year']
merge_cols  = ['campaign_title','organizer','day','month','year']
non_eng_m = (~po_dups.status.apply(is_english)) | (~po_dups.goal.apply(is_english))
po_dups_non_english=df.loc[:,show_cols].merge(po_dups.loc[non_eng_m,merge_cols],on=merge_cols).sort_values(by=merge_cols)


# In[ ]:


po_dups_non_english.head(10)


# #### Parse goal into raised and goal amount 

# Note: will fail for values where money was mentioned but not specified if it was raised or goal (e.g. '$8,130') 

# In[ ]:


if use_tqdm:
    tqdm.pandas(desc='Parsing goal')
    goal_parsed_dicts = df.goal.progress_apply(get_raised_and_goal_amount,**dict(USD_only=False)) 
else:
    goal_parsed_dicts = df.goal.apply(get_raised_and_goal_amount,**dict(USD_only=False)) 
# toogle USD_only to parse euro,pound or % also, else it'd return np.nan for these fields
goal_parsed_df = pd.DataFrame.from_records(goal_parsed_dicts,index=goal_parsed_dicts.index)
goal_parsed_df.head()


# Parse string inside 'raised' and 'goal' into amount and currency

# Check to see locations of campaigns that are not USD to see if they're based in the US

# In[ ]:


not_usd_m = (goal_parsed_df.raised_curr.str.find('$')==-1) | (goal_parsed_df.goal_curr.str.find('$')==-1)
# look at location of these campaigns
not_usd_locs=df.loc[not_usd_m,'location'].str.lower().value_counts()


# From manual checking, at least the first 180 locations are not but there are almost 20,000 locations.

# In[ ]:


not_usd_locs.head(20)


# Do a code check using get_state_var() & state_is_valid()

# In[ ]:


valid_US_state_m = not_usd_locs.index.to_series().apply(get_state_var).apply(state_is_valid)
# show any potential US location that have non_USD values
not_usd_locs[valid_US_state_m]


# Check on values such as '2.2k','3M','1.5B' to make sure they're parsed correctly

# In[ ]:


pd.concat([goal_parsed_df.loc[goal_parsed_df.goal.str.find('k')>-1,:].head(3),
goal_parsed_df.loc[goal_parsed_df.goal.str.lower().str.find('m')>-1,:].head(3),
goal_parsed_df.loc[goal_parsed_df.goal.str.lower().str.find('b')>-1,:].head(3)])


# Join parsed results into df

# In[ ]:


parsed_goal_cols = ['raised_amnt','raised_curr','goal_amnt','goal_curr']
df.drop(columns=parsed_goal_cols,errors='ignore',inplace=True) # drop if exist
df = df.merge(goal_parsed_df[parsed_goal_cols],on='campaign_id',how='left',indicator=True)
# See merge results
print(df._merge.value_counts())
df.drop(columns='_merge',inplace=True)
df.loc[:,['goal','raised_amnt','raised_curr','goal_amnt','goal_curr']].head()


# Make sure non-english goals got parsed correctly

# In[ ]:


goal_format_example = 'obiettivo'
goal_f_m=df.goal.str.find(goal_format_example)>-1
df.loc[goal_f_m,['goal','raised_amnt','goal_amnt','raised_curr','goal_curr']]


# #### Parse status to get number of contributors

# In[ ]:


if use_tqdm:
    tqdm.pandas(desc='Standardizing status')
    s_status = df.status.progress_apply(standardize_status)
    tqdm.pandas(desc='Parsing status')
    status_parsed_dicts = s_status.progress_apply(parse_status)
else:
    s_status = df.status.apply(standardize_status)
    status_parsed_dicts = s_status.apply(parse_status)
status_parsed_df = pd.DataFrame.from_records(status_parsed_dicts,index=status_parsed_dicts.index)
status_parsed_df['num_contributors'] = pd.to_numeric(status_parsed_df.ndonor.apply(standardize_MBk_in_number_str))
status_parsed_df.head()


# Check on values such as '2.2k','3M','1.5B' to make sure they're parsed correctly

# In[ ]:


pd.concat([status_parsed_df.loc[status_parsed_df.ndonor.str.find('k')>-1,:].head(3),
status_parsed_df.loc[status_parsed_df.ndonor.str.lower().str.find('m')>-1,:].head(3),
status_parsed_df.loc[status_parsed_df.ndonor.str.lower().str.find('b')>-1,:].head(3)])


# Join parsed results into df

# In[ ]:


df = df.merge(status_parsed_df[['num_contributors']],on='campaign_id',how='left',indicator=True)
# See merge results
print(df._merge.value_counts())
df.drop(columns='_merge',inplace=True)
df.loc[:,['status','num_contributors']].head()


# ### 4b. Clean variables

# Coarse cleaning for location

# In[ ]:


df['state'] = df['location'].apply(get_state_var)
df['old_loc_copy'] = df['location'].copy()
df['location'] = df['location'].apply(remove_non_loc_info)
df['other_loc'] = df['location'].apply(get_other_loc)


# Clean social media info

# In[ ]:


df['likes'] = df['num_likes'].apply(get_social)
df['shares'] = df['num_shares'].apply(get_social)


# Clean tag

# In[ ]:


df['tag'] = df['tag'].str.lower()
df['new_tag'] = df['tag'].apply(assign_new_tag)


# ### 5a. Drop duplicates based on parsed title and parsed date

# Keep track of number of campaigns we will exclude

# In[ ]:


exclusion_df = pd.read_csv(data_io.input_cleaned/'gfm'/'exclusion_tracker_rd_1.csv',
                          index_col = 0)


#these numbers come from Make master table notebook
exclusion_df.loc['total', 'original_campaign_count'] = 1856154
exclusion_df.loc['deleted', 'original_campaign_count'] = 0

exclusion_df.loc[0, 'original_campaign_num'] = 1856154
exclusion_df.loc[1, 'original_campaign_num'] = 1856154

exclusion_df.loc['total', 'duplicate_url'] = 1835822
exclusion_df.loc['deleted', 'duplicate_url'] = (exclusion_df.loc['total', 'original_campaign_count'] - 
                                                exclusion_df.loc['total', 'duplicate_url'])

exclusion_df.loc['total', 'poor_wayback_qual'] = 1809281
exclusion_df.loc['deleted', 'poor_wayback_qual'] = (exclusion_df.loc['total', 'duplicate_url'] - 
                                                    exclusion_df.loc['total', 'poor_wayback_qual'])


exclusion_df.loc['total', 'duplicate_title_organizer_date_loc'] = len(df)
exclusion_df.loc['deleted', 'duplicate_title_organizer_date_loc'] = (exclusion_df.loc['total', 'poor_wayback_qual'] - 
                                                                    exclusion_df.loc['total', 'duplicate_title_organizer_date_loc'])

exclusion_df


# Define functions to keep best duplicate

# In[ ]:


def keep_best_duplicate(df,subset=['title', 'location'],use_tqdm=False):
    # For processing, these columns will be dropped later
    df = df.assign(keep_this_duplicate=False,uid=range(df.shape[0]))
    # Get potentially duplicated campaigns
    mb_duplicates_m = df.duplicated(subset=subset, keep=False)
    mb_duplicates = df.loc[mb_duplicates_m, :]
    # Higher score means having this field != none is more important
    importance_score = pd.Series({
        'goal': 10,
        'created_date': 10,
        'status': 5,
        'num_likes': 5,
        'num_shares': 5,
        'story': 3,
        'location': 3
    })

    def get_index_of_best_duplicate(group):
        group = group.copy().replace('none',np.nan)
        # since we're edditing group, pandas will act all weird so need to copy
        # Calculate parsing quality
        
        for idx, row in group.iterrows():
            group.loc[idx, 'parsing_quality'] = (row[importance_score.index].notna() *
                                                 importance_score).sum()
        # Sort campaigns by timestamp and consequently quality
        # More recent timestamp and higher quality will be the last row
        # if any archive_timestamp is nan, just sort by quality
        if group.archive_timestamp.isna().any():
            return group.sort_values(by=['parsing_quality']).uid.iloc[-1]  # return uid of last row
        else:
            return group.sort_values(by=['archive_timestamp', 'parsing_quality'
                                     ],na_position='first').uid.iloc[-1]  # return uid of last row
    # Process each group of duplicate
    if use_tqdm:
        # use tqdm to make it pretty
        tqdm.pandas(desc='Processing duplicates')
        best_duplicate_uids = mb_duplicates.groupby(
            subset).progress_apply(get_index_of_best_duplicate)
    else:
        best_duplicate_uids = mb_duplicates.groupby(
            subset).apply(get_index_of_best_duplicate)

    # Signal the duplicate to keep based on uid
    df.loc[df.uid.isin(best_duplicate_uids), 'keep_this_duplicate'] = True
    # Return rows that are not duplicates OR is the best duplicate
    return df.loc[(df.keep_this_duplicate & mb_duplicates_m)
                  | ~mb_duplicates_m, :].drop(
                      columns=['keep_this_duplicate', 'uid'])


# Re-check duplicates

# In[ ]:


gb=df.groupby(['campaign_title','organizer','day','month','year']).size().reset_index()
po_dup_groups = gb[gb[0]>1]
po_dups=df.merge(po_dup_groups,on=['campaign_title','organizer','day','month','year']).sort_values(['campaign_title','organizer','day','month','year'])


# In[ ]:


po_dups[['created_date','location','campaign_title','organizer','day','month','year']].head(20)
temp = po_dups.drop_duplicates(subset=['campaign_title', 'organizer', 'day', 'month', 'year'])
print(len(temp))


# In[ ]:


old_df = df.copy()
print(len(old_df))
df = keep_best_duplicate(old_df, subset=['campaign_title','organizer', 'day','month', 'year'], use_tqdm=True)
print(len(df))


# Check to make sure all duplicates were resolved

# In[ ]:


gb=df.groupby(['campaign_title','organizer','day','month','year']).size().reset_index()
po_dup_groups = gb[gb[0]>1]
po_dups=df.merge(po_dup_groups,on=['campaign_title','organizer','day','month','year']).sort_values(['campaign_title','organizer','day','month','year'])


# In[ ]:


po_dups[['created_date','location','campaign_title','organizer','day','month','year']].head(20)


# In[ ]:



exclusion_df.loc['total', 'duplicate_title_organizer_date'] = len(df)
exclusion_df.loc['deleted', 'duplicate_title_organizer_date'] = (exclusion_df.loc['total', 'duplicate_title_organizer_date_loc'] -
                                                                    exclusion_df.loc['total', 'duplicate_title_organizer_date'])
print(exclusion_df)


# In[ ]:


save=False
if save:
    df.to_csv(data_io.input_cleaned/'gfm'/'all_campaigns.csv',sep='|',encoding='utf-8')


# ### 5b. Exclude cases and update exclusion df

# Set up columns for exclusions

# In[ ]:


df['cancer'] = df.apply(lambda x: find_cancer_story_title(x['story'], x['title']),axis=1)
df['cancer_in_story'] = df['story'].apply(cancer_in_x)
df['cancer_in_title'] = df['title'].apply(cancer_in_x)
#df['immunotherapy'] = df['story'].apply(immunotherapy_mention)


# In[ ]:


print(df['cancer'].value_counts())


# In[ ]:


df['state_in_us'] = df['state'].apply(state_is_valid)
                                      
df['status_is_english'] = df['status'].apply(is_english)
df['money_is_english'] = df['goal'].apply(is_english)


# In[ ]:


len(df)


# In[ ]:


exclusion_df


# In[ ]:


##EXCLUDE BASED ON STATE
df = df[df['state_in_us'] == True]
print(len(df))
exclusion_df.loc['deleted', 'state_not_us'] = (exclusion_df.loc['total', 'duplicate_title_organizer_date'] - 
                                              len(df))
exclusion_df.loc['total', 'state_not_us'] = len(df)

exclusion_df


# In[ ]:


1776317 - 1528918


# In[ ]:


# Exclude campaigns with goal not in USD since none would be from US (see test in step 4a)
last_col = 'state_not_us'
df = df.loc[df.goal.str.find('$')>-1,:]
exclusion_df.loc['deleted', 'not_USD'] = exclusion_df.loc['total', last_col] - len(df)
exclusion_df.loc['total', 'not_USD'] = len(df)

print(len(df))
print(exclusion_df.loc['total', last_col] - len(df))
print(exclusion_df)
last_col = 'not_USD'


# Make a copy as all US campaigns

# In[ ]:


df = df[df['cancer']==True]
exclusion_df.loc['deleted', 'not_cancer'] = exclusion_df.loc['total', last_col] - len(df)
exclusion_df.loc['total', 'not_cancer'] = len(df)
print(exclusion_df)
print(len(df))
print(exclusion_df.loc['total', last_col] - len(df))


# In[ ]:


last_col = 'not_cancer'
#Exclude on tag
df = df[pd.isnull(df['new_tag'])==False]
exclusion_df.loc['deleted', 'tag_not_medical'] = exclusion_df.loc['total', last_col] - len(df)
exclusion_df.loc['total', 'tag_not_medical'] = len(df)
print(exclusion_df)
print(len(df))
print(exclusion_df.loc['total', last_col] - len(df))
last_col = 'tag_not_medical'


# In[ ]:


print(df.year.isnull().sum())


# In[ ]:


#Exclude on year if null

df = df.dropna(subset=['year'])
exclusion_df.loc['deleted', 'year_is_null'] = exclusion_df.loc['total', last_col] - len(df)
exclusion_df.loc['total', 'year_is_null'] = len(df)
print(exclusion_df)
print(len(df))
print(exclusion_df.loc['total', last_col] - len(df))
exclusion_df.to_csv(data_io.input_cleaned/'gfm'/'exclusion_tracker_rd_2.csv')
last_col = 'year_is_null'


# In[ ]:


writer = pd.ExcelWriter(data_io.input_cleaned/'gfm'/'cancer_campaigns_no_locs.xlsx',
                        engine='xlsxwriter',
                        options=EXCEL_OPTIONS)
df.to_excel(writer, encoding='utf-8-sig')
writer.close()


# ### 6. Generate unique location spreadsheet and merge with existing (if exists)

# In[ ]:


def check_if_canada(x):
    #'ANA NAN'
    if type(x) != str:
        return False
    if x[-2:] == 'ca':
        if x[0].isalpha():
            if x[1].isdigit():
                if x[2].isalpha():
                    if x[3].isdigit:
                        if x[4].isalpha:
                            if x[5].isdigit:
                                return True
    return False


# In[ ]:


#NOTE: if none of the locations have been geocoded, you need to save this file and run it through the geocoder
new_unique_locs = df.drop_duplicates(subset=['location'], keep='first')
print(new_unique_locs['state_in_us'].value_counts())
new_unique_locs = new_unique_locs[['location','state', 'other_loc']]
new_unique_locs['possible_canada'] = new_unique_locs['location'].apply(check_if_canada)
len(new_unique_locs)
save = True
if save:
    writer = pd.ExcelWriter(data_io.input_cleaned/'geolocations'/'unique_locations_to_scrape_all_years.xlsx',
                            engine='xlsxwriter',
                            options={'strings_to_urls': False,
                                     'strings_to_formulas': False})
    new_unique_locs.to_excel(writer, encoding='utf-8-sig', index = False)
    writer.close()

