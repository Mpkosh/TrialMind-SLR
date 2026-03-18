import requests
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
#import pymupdf4llm
import re
import os

from openai import AsyncOpenAI,OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import httpx
from trialmind.llm_utils.openai_async import batch_call_openai

from markdown_pdf import MarkdownPdf
from markdown_pdf import Section
import ast
import time

openai_client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)


def info_from_doc(file_path, with_ru=False):
    loader = PyMuPDFLoader(file_path, mode='single')
    docs = loader.load()
    
    text = re.search('База клинических испытаний:(.*)', docs[0].page_content
         ).groups()[0].strip()
    # Colon cancer ?
    '''
    treat_table = re.search('(Ранг.{,10}Препарат.{,10}Активированные.{,10}мишени'+\
          '.{,10}Подавленные.{,10}мишени.{,10}Drug.{,10}score)'+\
          '(.*)(№.{,10}Ген.{,10}Транcкрипт.{,10}Замена.{,10}Тип)',
          docs[0].page_content.replace('\x0c',''), 
          flags=re.DOTALL).groups()[1]
    '''
    
    '''
    # тк в некоторых доках (216) после нужной таблицы идет не "полный... мутаций", а "полный список генов..."
    
    treat_table = re.search('(Ранг.{,10}Препарат.{,10}Активированные.{,10}мишени'+\
          '.{,10}Подавленные.{,10}мишени.{,10}Drug.{,10}score)'+\
          '(.*)(Полный)',
          docs[0].page_content.replace('\x0c',''), 
          flags=re.DOTALL).groups()[1]

    # между 1) номером строки в таблице и 2) след.колонкой в строке
    treatements = re.findall('\\n\\d+ (.*)\n', treat_table)
    # между 1) номером строки в таблице и 2) след.колонкой в строке
    # любой символ кроме минуса и цифры; минус, если есть; до точки; после точки.
    drug_scores = re.findall('\\n[^-\d]?(-?\\d+\.\\d+).?\n', treat_table)
    df = pd.DataFrame([treatements,drug_scores]).T
    df.columns = ['treat','score']
    '''
    
    # ищем след.раздел от нашей таблицы
    table_oc = re.search('(?:Содержание)[^\\n]*(\\n[\D]*)', 
          docs[0].page_content.replace('\x0c',''), 
          flags=re.DOTALL).groups()[0].split('\n')
    table_treats_id = table_oc.index('Полный список препаратов')
    next_title = table_oc[table_treats_id+1]
    # первая часть, тк при переносе строк символы,...
    xx = next_title[:25].replace(' ','[ \\n]')
    
    table_part = re.search('(Ранг.{,10}Препарат.{,10}Активированные.{,10}мишени'+\
                          '.{,10}Подавленные.{,10}мишени.{,10}Drug.{,10}score)'+\
                          f'(.*)(?:{xx})',
                          #'(.*)(Полный)',
                          docs[0].page_content.replace('\x0c',''), 
                          flags=re.DOTALL).group()
    
    treats=re.findall("(\s?\\n\d+([ а-яА-Я]*))"+\
              "((\\n[ a-zA-Zа\d,\n]*((?!\\n\d).)*)| )"+\
              "(\\n(- |(-?\d+\.\d+)))",
             table_part,
             flags=re.DOTALL)
    treats_df = pd.DataFrame(treats, 
                             columns=['name_','treat',
                                      'm_','m2_','_',
                                      'drugscore_','score','d_']
                            )
    df = treats_df[['treat','score']]
    # просто не указано число; меняем
    df.loc[df.score.isin(['- ','-',' -']),'score'] = '10.00'
    
    
    df.score = df.score.astype(float)
    treat001 = ' '.join(df[df.score>=0.01].treat.values)
    
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {text}"}]
    
    fin_condition = use_llm(os.getenv("MODEL_NAME"), messages)
    
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {treat001}. Separate each term with a comma. Answer only with translations, do not include any clarifications"}]
    treatements_eng = use_llm(os.getenv("MODEL_NAME"), messages).split(',')
    treatements_eng = [i.strip() for i in treatements_eng]
    
    if with_ru:
        return fin_condition,treatements_eng, df, text, df[df.score>=0.01].treat.values
    else:
        return fin_condition,treatements_eng, df


def use_llm(model='Qwen/Qwen3-32B', messages=[],openai_client=openai_client):
    
    #print(os.getenv("BASE_URL"))
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"reasoning_effort": "none"}
    )
    fin = response.choices[0].message.content
    fin = fin.strip('<think>\n\n</think>\n\n')
    return fin




def results_ct(all_studies):
    chosen = all_studies[all_studies.results==True]
    outcome_participants = [obj for item_main in chosen.outcomes.values for obj in item_main 
                        if(obj.get('unitOfMeasure','') == 'Participants')] 
    messages = []
    for i in outcome_participants:
        messages.append([{'role':'system', 'content':'You are a helpful medical assistant. Extract information about the efficiency statistics from each outcome measure. Give: 1) a concise description of results with percentages and 2) a short description of study objectives /no_think'},
                {'role':'user', 'content':f"{i}"}])
    
    results = batch_call_openai(messages, os.getenv('MODEL_NAME'), int(os.getenv('TEMPERATURE')), thinking=False)
    return results
    


def get_clinicaltrials(query_term, query_intr, max_studies=20):
    base = 'https://clinicaltrials.gov/api/v2/studies'  # new API

    # fields for CSV and JSON: https://clinicaltrials.gov/data-api/about-api/csv-download

    # new API fields - for JSON
    extract_fields = [
        'NCTId',
        'Condition',
        'BriefTitle',
        'HasResults',
        "Intervention",
        #'PrimaryOutcome',
        'OutcomeMeasuresModule',
        #'NumPhases',
        'Phase',
        'OverallStatus',
        'StudyType',
        'BriefSummary',
        #'LastKnownStatus',

    ]
    #OutcomeMeasuresModule 
    # new API fields
    params = {
        'fields': ",".join(extract_fields), 
        'query.term': query_term,
        'query.intr': query_intr,
        #'filter.overallStatus':'',
        'pageSize': max_studies, # maximum number of !!studies!! to return in response
        'aggFilters':'results:with',
        'format': 'json', 
        #'pageToken': None  # first page doesn't need it
    }    
    all_studies = []
    results_studies = []
    no_results_studies = []
    next_page_token='a'
    page_temp = 0
    while next_page_token and page_temp<max_studies:

        #print(f'--- page: {page} ---')

        response = requests.get(base, params=params)

        if not response.ok:
            print('response.text:', response.text)
            break

        data = response.json()
        #print(data.keys())
        data_responses = data['studies']
        #print(data_responses)

        all_studies = all_studies+data_responses
        '''
        for index, item_main in enumerate(data_responses, 1):
            #print(item)
            item = item_main['protocolSection']['identificationModule']
            #print(item)
            
            print(f'{page:2},{index:4}:', item_main['protocolSection']['statusModule']['overallStatus'], 
                  #item_main['protocolSection']['statusModule']['lastKnownStatus'],
                  item_main['protocolSection']['designModule']['phases'],
                  #item_main['protocolSection']['designModule']['numPhases'],
                  #item_main['resultsSection']['outcomeMeasuresModule']['outcomeMeasures'], 
                  item['nctId'], item['briefTitle'])
            '''

        try:
        # next page
        #next_page_token = response.headers.get('x-next-page')  # (probably) for `CSV`
            next_page_token = data['nextPageToken']                 # for `JSON`
            #print(data['nextPageToken'])
        #print('x-next-page', next_page_token)    
            params['pageToken'] = next_page_token
        except KeyError:
            next_page_token=''
            pass
        page_temp+=1
        
    all_studies = pd.json_normalize(all_studies)
    all_studies.columns = [i.split('.')[-1] for i in \
                           all_studies.columns]

    if 'outcomeMeasures' not in all_studies.columns:
        all_studies['outcomeMeasures']=None 
    # раскрываем препараты из описания клин.испытаний

    # https://clinicaltrials.gov/study/NCT03792568 
    # here intervention == Other: ALK Inhibitor
    # [{'type': 'BIOLOGICAL', 'name': 'cetuximab'}, ?
    # мб убрать terminated suspended withdrawn unknown ...
    # в https://clinicaltrials.gov/study/NCT02510001?cond=Colorectal%20Cancer&intr=Crizotinib&rank=1 in interventions crizotinib is not mentioned
    all_studies['interventions'] = all_studies['interventions'].apply(lambda d: d if isinstance(d, list) else [{}]
                                  ).apply(lambda x: \
                                          [obj['name'].lower() for obj in x \
                                               if obj.get('type','') in [#'BIOLOGICAL',
        'DRUG'] ])



    return all_studies


def treat_studies(treatement, all_studies):
    idx = [treatement in ' '.join(d) for d in all_studies.interventions]
    return all_studies[idx]


def ctrials_res(ec_pred, all_studies):
    import numpy as np
    from pydantic import BaseModel, validator, Field, conlist  # This is the new version
    from typing import Dict, Literal
    
    
    
    evals = [i.evaluations for i in ec_pred]
    word2int = {"YES": 1, "UNCERTAIN": 0,"NO": -1}
    new_evals = []
    for one_e in evals:
        new_evals.append([word2int.get(item, 0) for item in one_e ])
    new_evals = np.array(new_evals)    
    all_studies['screen_eval'] = new_evals.sum(axis=1)
    
    class Measure(BaseModel):
        measure_description: str = Field(description='Short description of what is being measured',
                                        max_length=200)
        measure_result: float = Field(description='Percent of participants')

    class GroupMeasures(BaseModel):
        group_description: str = Field(description='Short group description',
                                      max_length=200)
        measures: list[Measure]

    class Outcome(BaseModel):
        description: str = Field(description='Short description of an outcome',
                                max_length=200)
        population: int = Field(description='Total number of participants.')
        time_frame: str = Field(description='Time frame', max_length=200 )
        measures: list[GroupMeasures]

    class Outcomes(BaseModel):
        outcomes: list[Outcome]
    
    PROMPT_RES_EXTRACTION  = f'''
    You are a clinical specialist analyzing clinical trial study reports. 
    Your task is to to extract specific information as structured data.

    # Reply Format: 
    Return the information in the following JSON-format.
    ```
    {Outcomes.model_json_schema()}
    ```
    You MUST return ONLY valid JSON, Do NOT include any explanations, comments, or extra text.
    '''
    
    at_once = False
    openai_client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False)
    )

    chosen = all_studies[(all_studies.screen_eval>=0)&(all_studies.hasResults==True)]
    chosen['res_with_part'] = chosen['outcomeMeasures'].apply(lambda x: [obj for obj in x \
                                                              if(obj.get('unitOfMeasure','').lower() in ['percentage of participants','participants'])])
    
    to_work = chosen[chosen.res_with_part.str.len()>0]
    if to_work.shape[0]:
        messages = []
        for i in to_work.res_with_part.values:
            n_outcomes = len(i)
            messages.append([{'role':'system', 'content':PROMPT_RES_EXTRACTION+' \no_think'},
                    {'role':'user', 'content':f"{i}"}])
        if at_once:
            results = batch_call_openai(messages, os.getenv('MODEL_NAME'), 
                                        int(os.getenv('TEMPERATURE')), thinking=False)
        else:
            #print(messages)
            resultsct = []
            for message in messages:
                #$answer = extract.use_llm(os.getenv('MODEL_NAME'),i)

                response = openai_client.chat.completions.parse(
                    model=os.getenv('MODEL_NAME'),
                    messages=message,
                    temperature=0,
                    response_format=Outcomes,
                    extra_body={"reasoning_effort": "none"}
                )
                fin = response.choices[0].message.parsed
                #answer = fin.strip('<think>\n\n</think>\n\n')
                resultsct.append(fin)
                
    return resultsct



def combine_res(fin_condition, treatements_eng, extracted, pmid_list):
    def replacement_match(match):
        #print(match.groups()[0])
        return f'[[{pmid_list[int(match.groups()[0])]}]]'

    text_chunks = '\n\n'.join(
        [f"<source id=\"{i}\"><result>{block.fieldresult[0].value}</result>"+\
         f"<context>{'. /n/n '.join(block.fieldresult[0]._cited_blocks)}</context></source>" \
             for i,block in enumerate(extracted)
        ])
    
    prompt = '''You are a clinical specialist. You are conducting a clincial meta-analysis.

        # Task
        The user will provide a list of extracted results of treating {fin_condition} with {treatements_eng} from different sources along with context. 
        1. Combine extracted results in one coherent paragraph. Each result should appear only once and stay in a separate sentence.
        2. Provide a reference to the document ID from which this information was extracted. The citation id must be integers only and be in double square brackets -- [[citation]]. Citations should not be repeated!
        

        # User provided inputs

        text_chunks = \"\"\"{text_chunks}\"\"\"

        # Response format
        Answer with sentences and references in double square brackets.
        '''
    messages=[{'role':'user',
               'content':prompt.format(text_chunks=text_chunks,
                                       fin_condition=fin_condition,
                                       treatements_eng=treatements_eng)
              }]
    response = use_llm(os.getenv("MODEL_NAME"), messages)
    return re.sub(r'\[\[(\d+)\]\]', replacement_match, response)


def gen_report(treatements_eng, fin_condition):
    import time
    from pydantic import BaseModel, validator, Field, conlist  # This is the new version
    from typing import Dict, Literal

    class hhey:
        def __init__(self,papers):

            self.dict_h = {}
            self.h = 0
            self.papers=papers
        def replacement_match(self,match):
            found_id = match.groups()[0]
            self.h+=1
            #print(self.h)
            try:
                #print(found_id, papers[papers.PMID==int(found_id)].Title.values[0])
                self.dict_h[self.h]=[self.papers[self.papers.PMID==int(found_id)].Title.values[0],
                                    'https://pubmed.ncbi.nlm.nih.gov/'+str(found_id)]
            except IndexError:
                pass
            return f'[{self.h}]'
    
    
    class Measure(BaseModel):
        measure_description: str = Field(description='Short description of what is being measured',
                                        max_length=200)
        measure_result: float = Field(description='Percent of participants')

    class GroupMeasures(BaseModel):
        group_description: str = Field(description='Short group description',
                                      max_length=200)
        measures: list[Measure]

    class Outcome(BaseModel):
        description: str = Field(description='Short description of an outcome',
                                max_length=200)
        population: int = Field(description='Total number of participants.')
        time_frame: str = Field(description='Time frame', max_length=200 )
        measures: list[GroupMeasures]

    class Outcomes(BaseModel):
        outcomes: list[Outcome]
    
    
    class FieldResult(BaseModel):
        name: str
        value: str = Field(description='Extracted information from the text based on the field description.',
                          max_length=200)
        source_id: conlist(int,min_length=0, max_length=3) = Field(description='Cited document IDs.')
    class Results(BaseModel):
        fieldresult: list[FieldResult] = Field(min_length=1, max_length=1) 

        
    pdf = MarkdownPdf(toc_level=4, optimize=True)

    section = Section("# Title\n", toc=False)
    text2 = f"""# Diagnosis: {fin_condition}"""


    for treatement in treatements_eng[:1]:
        papers = pd.read_csv(f"res_files/papers_all_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv")
        papers_res = pd.read_csv(f"res_files/papers_res_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv")
        res_extracted = papers_res['class'].apply(
            lambda x: Results.model_validate_json(x)).values
        for idx, i in enumerate(res_extracted):
            i.fieldresult[0]._cited_blocks = ast.literal_eval(
                                                papers_res['citations'].values[idx])
        pmid_list = papers_res.id.values
        
        trials = pd.read_csv(f"res_files/ctrials_res_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv")
        trial_res = trials['res'].apply(lambda x: Outcomes.model_validate_json(x)).values
        
        add = f"""\n## Treatement 1: {treatement}\n\n### Chosen clinical trials:\n"""
        text2 = text2+add
        
        
        k = hhey(papers=papers)
        rr = combine_res(fin_condition, treatement, 
                        res_extracted, pmid_list) 
        add_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match, rr)

        func_r = ''

        num = 1
        for i,j in zip(trials.values,trial_res):

            func_r+=f"{num}. '{i[2]}'"+\
                           f" {i[3].lower()}"+\
                           f" {' '.join(ast.literal_eval(i[7])).lower()}"+\
                           f' [https://clinicaltrials.gov/study/{i[1]}](https://clinicaltrials.gov/study/{i[1]})'+\
                          "\n\n"

            func_r_1 = ""
            #for j_n, j in enumerate(trial_res):
            func_r_1 = func_r_1+ \
                          f"&emsp;&emsp;&emsp;Population: {j.population}"+ '\n\n'+\
                          f"&emsp;&emsp;&emsp;Time: {j.time_frame}"+ '\n\n'

            for out in j.outcomes:
                func_r_1 = func_r_1+ "&emsp;&emsp;&emsp;"+out.category_name+\
                                ': '+str(out.outcome)+ '\n\n'

            func_r+=func_r_1
            num+=1

        text2=text2+func_r    


        text3 = f"""\n### Chosen scientific papers:\n\n{add_summ}\n\n"""

        text2 = text2+text3
        lit_list = '\n\n'.join([f"[{i}]  {k.dict_h.get(i, 'hh')[0]} [{k.dict_h.get(i, 'hh')[1]}]({k.dict_h.get(i, 'hh')[1]})" for i in range(1,k.h+1)])
        textall = text2+lit_list

    pdf.add_section(Section(textall))

    pdf.save("test.pdf")
    
    # translating into Russian!
    # TODO: replace diagnosis and treatement names with names from THE report!

    prompt = '''
        You are a medical specialist. Translate the text into Russian. 


        1. Do not change the format.
        2. English text in single quotes \' \' must be left as is without translation.
        3. Use the following term for Colorectal cancer -> Колоректальный рак
        4. Do not add any new text.

        # User provided inputs

        text = \"\"\"{text}\"\"\"

        # Response format
        Answer only with translated text.
        '''

    t1 = time.time()
    messages=[{'role':'user',
               'content':prompt.format(text=text2)
              }]

    fint= use_llm(model=os.getenv("MODEL_NAME"), messages=messages)
    print(time.time()-t1)

    pdf = MarkdownPdf(toc_level=4, optimize=True)
    pdf.add_section(Section(fint+"\n\n"+lit_list))

    pdf.save(f"doc_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.pdf")