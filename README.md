# TrialMind


## Pre-requisites

- `requirements.txt` contains the required packages for the code to run.


## API Keys

The code requires API keys for the

- OpenAI GPT API for its chat endpoint: https://platform.openai.com/docs/api-reference/chat/create 

- PubMed API key for PubMed search: https://www.ncbi.nlm.nih.gov/home/develop/api/. Help is here: https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us 

- E2B sandbox API key for executing generated code (optional for result extraction): https://e2b.dev/docs


## Examples

### literature_search.ipynb
Show how to use the code to search for literature on a given topic.

### literature_screen.ipynb
Show how to use the code to screen literature on a given topic.

### study_extraction.ipynb
Show how to use the code to extract study information from a given paper.

### result_extraction.ipynb
Show how to use the code to extract results from a given paper.

# Citations

```bib
@article{wang2025accelerating,
  title   = {Accelerating clinical evidence synthesis with large language models},
  author  = {Wang, Zifeng and Cao, Lang and Danek, Benjamin and Jin, Qiao and Lu, Zhiyong and Sun, Jimeng},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {509},
  year    = {2025},
  doi     = {10.1038/s41746-025-01840-7},
}
```
