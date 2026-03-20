import logging
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*escape sequence.*")

import pandas as pd
import numpy as np
import re
import time
import os
from dotenv import load_dotenv, find_dotenv
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section
import ast

from pydantic import BaseModel, Field, conlist  # This is the new version
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# for importing trialmind, api_key should be set beforehand
from trialmind.pubmed import pmid2papers, PubmedAPIWrapper, parse_bioc_xml, pmid2fulltext
from trialmind.api import StudyCharacteristicsExtraction, LiteratureScreening, CTScreening
import extract


logging.getLogger().setLevel(logging.INFO)
load_dotenv(find_dotenv(usecwd=True))

#embeddings = OllamaEmbeddings(model="qwen3-embedding")
embeddings = OllamaEmbeddings(model="all-minilm")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    #persist_directory="./chroma_langchain_db",  
)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, 
                                             chunk_overlap=100)


def get_res(file_path = "docs/LuC_213_L00_DL_edited_oncobox_ru.pdf",
            model_translate='qwen3:8b', model_main='qwen3:8b',
            n_papers=5,ct_pages=1,
            ct_criteria = \
            ["Does the trial focus on patients with '{fin_condition}'?",
             "Does the trial examine the use or sensitivity of '{treatement}' among main treatments?"],
            papers_criteria=\
            ["Does the study focus on patients/models/cells with '{fin_condition}'?",
             "Does the study examine the use/effect/sensitivity of '{treatement}' among main treatments?", 
             "Does the study describe the effect of '{treatement}' treatment?"],
            ct_screen_thresh=1,
            paper_screen_thresh = [1,1,0],
            save_files=True,
           ):    
    start = time.time()
    start_all = start
    
    # translation (as a task) is simple => we use a smaller llm to get results quicker
    os.environ["MODEL_NAME"] = model_translate
    logging.info(os.getenv('MODEL_NAME'))
    # get main info from .pdf
    logging.info('GETTING INFO FROM FILE')
    
    fin_condition,treatements_eng,df2,fin_condition_ru,treatements_ru = extract.info_from_doc(file_path, with_ru=True)

    end = time.time()
    logging.info(end-start)
    start = end
    
    # using a larger llm for better results
    #os.environ["MODEL_NAME"] ='qwen3:14b'
    os.environ["MODEL_NAME"] = model_main
    for treatement in treatements_eng[:1]:

        # ________________Get clinical trials
        logging.info('GETTING CLINICAL TRIALS')
        all_studies = extract.get_clinicaltrials(f'''"{fin_condition}"''', 
                                              #' OR '.join(treatements_eng), 
                                               treatement,  
                                              max_studies=ct_pages)
        logging.info(all_studies.shape)
        all_text = all_studies.briefTitle.fillna("") + ": " + all_studies.briefSummary.fillna("")
        api = CTScreening()
        end = time.time()
        logging.info(end-start)
        start = end
        
        logging.info('SCREENING CLINICAL TRIALS')
        ec_pred = api.run(
            population = f"Patients with {fin_condition} undergoing treatment with {treatement}",
            intervention = f"{treatement}",
            comparator = "",
            outcome = "",
            llm = os.getenv("MODEL_NAME"),
            criteria = [i.format(fin_condition=fin_condition,
                                 treatement=treatement) for i in ct_criteria],
            papers = all_text.values.tolist(), # make for the top-100 for demo
        )
        all_studies[['s1','s2']] = [i.evaluations for i in ec_pred]
        all_studies[['r1','r2']] = [i.rationale for i in ec_pred]
        if save_files:
            all_studies.to_csv(f"res_files/ctrials_all_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv", 
                      index=False)
        end = time.time()
        logging.info(end-start)
        start = end
        
        logging.info('EXTR RESULTS FROM CLINICAL TRIALS')
        ctrials_fin = extract.ctrials_res(ec_pred, all_studies)
        chosen_st = all_studies[(all_studies.screen_eval>=ct_screen_thresh)&(all_studies.hasResults==True)]
        chosen_st['res_with_part'] = chosen_st['outcomeMeasures'].apply(lambda x: [obj for obj in x \
                                                                  if(obj.get('unitOfMeasure','').lower() in ['percentage of participants','participants'])])
        chosen_st = chosen_st[chosen_st.res_with_part.str.len()>0]
        chosen_st['res']= [i.model_dump_json() for i in ctrials_fin]
        if save_files:
            chosen_st.to_csv(f"res_files/ctrials_res_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv", 
                      index=False)
        logging.info(ctrials_fin)
        end = time.time()
        logging.info(end-start)
        start = end

        # ________________Get papers
        logging.info('GETTING PAPERS')
        search_api = PubmedAPIWrapper()
            # page_size is the max number of records to return!!!! not pages!
        tmp_inputs = {
                "page_size": n_papers,
                "keyword_map": {'conditions':[fin_condition], 
                                'treatments':[treatement]
                               },
                "keywords": {
                    "OPERATOR": 'AND'
                }
        }
        response = search_api.build_search_query_and_get_pmid(tmp_inputs, 
                                                              api_key=os.getenv("PUBMED_API_KEY"))
        logging.info(f'{response[0],len(response[0]), response[1]}')
        df_papers = pmid2papers(pmid_list=response[0], 
                                api_key=os.getenv("PUBMED_API_KEY"))
        papers = df_papers[0]["Title"] + ": " + df_papers[0]["Abstract"].fillna("")
        papers = papers.tolist()
        end = time.time()
        logging.info(end-start)
        start = end
            # screening
        logging.info('SCREENING PAPERS')
        api = LiteratureScreening()
        ec_predP = api.run(
            population = f"Patients with {fin_condition} undergoing treatment with {treatement}",
            intervention = f"{treatement}",
            comparator = "",
            outcome = "",
            llm = os.getenv("MODEL_NAME"),
            criteria = [i.format(fin_condition=fin_condition,
                                 treatement=treatement) for i in papers_criteria],
            papers = papers, 
        )

        evalsP = [i.evaluations for i in ec_predP]
        word2int = {"YES": 1, "UNCERTAIN": 0, "NO": -1}
        new_evalsP = []
        for one_e in evalsP:
            new_evalsP.append([word2int.get(item, 0) for item in one_e ])
        new_evalsP = np.array(new_evalsP)  
        df_p_e = df_papers[0]
        df_p_e[['s1','s2','s3']] = new_evalsP
        df_p_e[['r1','r2','r3']] = [i.rationale for i in ec_predP]
        if save_files:
            df_p_e.to_csv(f"res_files/papers_all_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv", 
                      index=False)

        #df_p_e = pd.read_csv(f"res_files/papers_all_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv")
        chosen_df = df_p_e[(df_p_e['s1']>=paper_screen_thresh[0]
                           )&(df_p_e['s2']>=paper_screen_thresh[1]
                             )&(df_p_e['s3']>=paper_screen_thresh[2])]
        logging.info(chosen_df.shape[0])
        end = time.time()
        logging.info(end-start)
        start = end

        #end6=time.time()
        # if there are papers left AFTER screening
        if chosen_df.shape[0]:
            # ________________To RAG
                # full texts
            pmid_list = chosen_df.PMID.values.tolist()
            #['41213063',#'26451310',]
            logging.info('GETTING FULL TEXT PAPERS')
            res = pmid2fulltext(pmid_list, api_key=os.getenv("PUBMED_API_KEY"))
            res = [parse_bioc_xml(r) for r in res]

            # transform the parsed xml into paper content
            papers0 = []
            for parsed in res:
                paper_content = []
                for parsed_ in parsed["passage"]:
                    paper_content.append(parsed_['content'])
                paper_content = "\n".join(paper_content)
                papers0.append(paper_content)

            chosen_df['FullText'] = ''
            chosen_df['FullText'] = papers0

            pmid_list = chosen_df.PMID.values.tolist()
            papers_ch = chosen_df.FullText.values
            docs =  [Document(page_content=i, 
                              metadata={"source": j}
                             ) for i,j in zip(papers_ch,pmid_list)]
            end = time.time()
            logging.info(end-start)
            start = end
            
            logging.info('EMBEDDING...')
            all_splits = text_splitter.split_documents(docs)
            vector_store.add_documents(documents=all_splits)
            logging.info('FIN EMBEDDING')
            end = time.time()
            logging.info(end-start)
            start = end

            ii =len(pmid_list) #4
            logging.info('EXTR RESULTS FROM PAPERS')
            api = StudyCharacteristicsExtraction()
            res_extracted = api.run(
                papers_inp=[pmid_list[:ii],papers_ch[:ii]],
                #fields=[f'The effectiveness of treating {fin_condition} with {treatements_eng[0]}',
                #       ],
                fields=[f'{treatement} effectiveness, string, The outcome of treating {fin_condition} with {treatement}'],
                llm=os.getenv("MODEL_NAME"),
                chunk_size=0,
                chunk_overlap=0,
                thinking=False,
                vector_store = vector_store,
            )
            papers_fin_df = pd.DataFrame([[i.fieldresult[0].value, 
                                   i.fieldresult[0].source_id,
                                  i.fieldresult[0]._cited_blocks,] for i in res_extracted 
                                 ],
                                columns = ['result','idxs','citations'])
            papers_fin_df['id'] = pmid_list
            papers_fin_df['class']= [i.model_dump_json() for i in res_extracted]
            if save_files:
                papers_fin_df.to_csv(f"res_files/papers_res_df_{treatement.replace(' ','_')}_{fin_condition.replace(' ','_')}.csv", 
                                     index=False)
            end = time.time()
            logging.info(end-start)
            start = end
            logging.info('COMBINING RESULTS FROM CLINICAL TRIALS')
            rr = extract.combine_res(fin_condition, treatement, 
                                 res_extracted, pmid_list)
            end = time.time()
            logging.info(end-start)
            start = end
            logging.info(f'{rr}')

    end = time.time()
    logging.info(f'FUll time: {end - start_all}')
    return treatements_eng, fin_condition


def fill_pdf(treatements_eng, fin_condition, model_translate='qwen3:8b'):
    

    class hhey:
        def __init__(self,papers):

            self.dict_h = {}
            self.h = 0
            self.papers=papers
        def replacement_match(self,match):
            found_id = match.groups()[0]
            self.h+=1
            #logging.info(self.h)
            try:
                #logging.info(found_id, papers[papers.PMID==int(found_id)].Title.values[0])
                self.dict_h[self.h]=[self.papers[self.papers.PMID==int(found_id)
                                                ].Title.values[0],
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
        rr = extract.combine_res(fin_condition, treatement, 
                        res_extracted, pmid_list) 
        add_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match, rr)

        func_r = ''

        num = 1
        for i,trial_mess in zip(trials.values,trial_res):

            func_r+=f"{num}. '{i[2]}'"+\
                           f" {i[3].lower()}"+\
                           f" {' '.join(ast.literal_eval(i[7])).lower()}"+\
                           f' [https://clinicaltrials.gov/study/{i[1]}](https://clinicaltrials.gov/study/{i[1]})'+\
                          "\n\n"

            func_r_1 = ""
            for j_n, trial_out in enumerate(trial_mess.outcomes): 
                func_r_1 = func_r_1+ \
                            f"&emsp;&emsp;&emsp;Description: {trial_out.description}"+ '\n\n'+\
                              f"&emsp;&emsp;&emsp;Population: {trial_out.population}"+ '\n\n'+\
                              f"&emsp;&emsp;&emsp;Time: {trial_out.time_frame}"+ '\n\n'
                for mes in trial_out.measures:
                    func_r_1 = func_r_1+ "&emsp;&emsp;&emsp;group description: "+mes.group_description
                    for one_mes in mes.measures:
                        func_r_1 = func_r_1+ "&emsp;&emsp;&emsp;measure: "+one_mes.measure_description+\
                                        ' ('+str(one_mes.measure_result)+ ')\n\n'

            func_r+=func_r_1
            num+=1

        text2=text2+func_r    


        text3 = f"""\n### Chosen scientific papers:\n\n{add_summ}\n\n"""

        text2 = text2+text3
        lit_list = '\n\n'.join([f"[{i}]  {k.dict_h.get(i, 'hh')[0]} [{k.dict_h.get(i, 'hh')[1]}]({k.dict_h.get(i, 'hh')[1]})" for i in range(1,k.h+1)])
        textall = text2+lit_list

    pdf.add_section(Section(textall))

    pdf.save(f"ENG_doc_{treatement.replace(' ','_')}_{len(treatements_eng)}_{fin_condition.replace(' ','_')}.pdf")
    
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

    fint= extract.use_llm(model=model_translate, messages=messages)
    logging.info(time.time()-t1)

    pdf = MarkdownPdf(toc_level=4, optimize=True)
    pdf.add_section(Section(fint+"\n\n"+lit_list))

    pdf.save(f"RU_doc_{treatement.replace(' ','_')}_{len(treatements_eng)}_{fin_condition.replace(' ','_')}.pdf")