import requests
import pandas as pd

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
import pymupdf4llm
import re
import os

from openai import AsyncOpenAI,OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import httpx
from trialmind.llm_utils.openai_async import batch_call_openai


openai_client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)


def info_from_doc(file_path):
    loader = PyMuPDFLoader(file_path, mode='single')
    docs = loader.load()
    
    text = re.search('База клинических испытаний:(.*)', docs[0].page_content
         ).groups()[0].strip()
    # Colon cancer ?

    border_phrase = 'Полный рейтинг и подробное описание даны на стр.'
    # ищем между 2-мя фразами, заменяем знак новой страницы на '', точка -- любой символ И переносы строк
    treat_table = re.search(f'(?<={border_phrase})(.*)(?={border_phrase})', docs[0].page_content.replace('\x0c',''), 
              flags=re.DOTALL).groups()[0]
    treatement = re.findall('\\n\\d+ (.*)\n', treat_table)
    treatement = ', '.join(treatement)
    
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {text}"}]
    
    fin_condition = use_llm(os.getenv("MODEL_NAME"), messages)
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {treatement}"}]
    treatements_eng = use_llm(os.getenv("MODEL_NAME"), messages).split(',')
    treatements_eng = [i.strip() for i in treatements_eng]
    
    return fin_condition,treatements_eng


def use_llm(model='Qwen/Qwen3-32B', messages=[]):
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
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
        #'aggFilters':'results:with',
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