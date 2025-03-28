import numpy as np
import pandas as pd
import torch
import sentence_transformers
import warnings
from dotenv import load_dotenv
import os
import re
from googletrans import Translator

class ESCOextractor:
    def __init__(self, type_codes="occupations", gpu=False):
        '''
        :param type_codes: which code to return: skill or occupation
        :param gpu: flag to enable cuda acceleration
        '''
        #Error handlings
        if type_codes is None:
            raise ValueError(f'[esco_extractor] type_codes is None')
        if not isinstance(type_codes, str):
            raise TypeError(f'[esco_extractor] type_codes should be a string')
        if type_codes == '':
            raise ValueError(f'[esco_extractor] type_codes is an empty string')
        type_codes = type_codes.lower() #convert to lower case
        if (type_codes!="skills") and (type_codes!="occupations"):
            raise ValueError(f'[esco_extractor] type_codes should be "skills" or "occupations"')

        #attribute assigning
        self.type_codes = type_codes
        load_dotenv()  #load environmental variables

        if gpu is None:
            raise ValueError(f'[esco_extractor] GPU flag is None')
        if gpu:
            if torch.cuda.is_available():
                self.hardware = "cuda"
            else:
                raise NotImplemented(f'[esco_extractor] GPU acceleration only supports cuda')
        else:
            self.hardware = "cpu" #run the transformer model in cpu instead of gpu

        with warnings.catch_warnings():  # ignore warnings when loading the model
            warnings.simplefilter("ignore")
            if os.path.exists(os.getenv("MODEL_PATH")):
                self.ext_model = SentenceTransformer.load(os.getenv("MODEL_PATH"))  #load the saved model
            else:
                self.ext_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2",
                                                                           device=self.hardware)  #define the text transformer module and how to run
                self.ext_model.save_pretrained(os.getenv("MODEL_PATH"))  #save the model into memory

        #Data structures to store all the ESCO information
        self.codes = None #variable to store the skills/occupations for string comparison
        self.codes_id = None #variable to store the ESCO ID of the skills/occupations
        self.codes_desc = None #variable to store the ESCO description of the skills/occupations
        self.codes_ESCO = None #variable to store the ESCO codes of the skills/occupations
        self.encodings = None #variable to store the model encodings for the reference skills/occupations data

        #Data Structures to store processing on the ESCO information
        self.codes_id_proc = None  #variable to store the processed ESCO ID of the skills/occupations
        self.codes_desc_proc = None  #variable to store the processed ESCO description of the skills/occupations
        self.codes_ESCO_proc = None  #variable to store the processed ESCO codes of the skills/occupations
        self.encodings_proc = None  #variable to store the processed model encodings for the reference skills/occupations data

    def format_codes(self):
        '''
        format IDs and Descriptors of the reference ESCO entries as numpy arrays
        '''
        if self.type_codes == "skills":
            self.codes = pd.read_csv(os.getenv("SKILLS_PATH"))
        else:
            self.codes = pd.read_csv(os.getenv("OCCUPATIONS_PATH"))
            self.codes = self.codes.loc[self.codes['code'].str.len()<=7] #limit the codes to filter from

        #Format to numpy array for faster processing
        self.codes_id = self.codes["conceptUri"].to_numpy() #convert the ids
        self.codes["description"] = self.codes["description"].str.lower() #apply lower case
        self.codes_desc = self.codes["description"].to_numpy() #convert the descriptors
        if self.type_codes == "occupations":
            self.codes_ESCO = self.codes["code"].to_numpy() #convert the ESCO codes
        else:
            self.codes_ESCO = self.codes["definition"].to_numpy() #convert the ESCO codes

    def format_encodings(self):
        '''
        Format the encodings of the codes in a file, or load if it exists
        '''
        if self.type_codes == "occupations":
            if os.path.exists(os.getenv("ENCODINGS_OCCUPATIONS_PATH")):
                self.encodings = np.load(os.getenv("ENCODINGS_OCCUPATIONS_PATH")) #load the encodings directly
                self.encodings = torch.from_numpy(self.encodings) #convert to torch tensor
            else:
                self.encodings = self.ext_model.encode(self.codes_desc, device=self.hardware,
                                                        normalize_encodings=True, convert_to_tensor=True) #format the encodings
                np.save(os.getenv("ENCODINGS_OCCUPATIONS_PATH"), self.encodings) #save the encodings for future use
        elif self.type_codes == "skills":
            if os.path.exists(os.getenv("ENCODINGS_SKILLS_PATH")):
                self.encodings = np.load(os.getenv("ENCODINGS_SKILLS_PATH")) #load the encodings directly
                self.encodings = torch.from_numpy(self.encodings) #convert to torch tensor
            else:
                self.encodings = self.ext_model.encode(self.codes_desc, device=self.hardware,
                                                        normalize_encodings=True, convert_to_tensor=True) #format the encodings
                np.save(os.getenv("ENCODINGS_SKILLS_PATH"), self.encodings) #save the encodings for future use
        else:
            raise ValueError(f'[esco_extractor.format_encoding] type_codes should be skills or occupations')

    def format_tokens(self, text_input):
        '''
        :param text_input: the skill/occupation input the user wants to extract the ESCO code from
        :return: a list with the tokens for each string of the text input
        '''
        text_input = text_input.strip() #strip the string
        return re.split(r"\r|\n|\t|\.|,|;| and | na | no | em | da | do | para ", text_input) #split based on characters and not strings -> context

    def filter_token_occ(self, tokens):
        '''
        :param tokens: the tokens of the text input
        :return the indexes of the descriptors that contain the words in the tokens
        '''
        idxs = []  # list to append the indexes of the occurrences
        tokens = [tks for sub_list in tokens for tks in sub_list.split()] #flatten the tokens list
        for desc in range(len(self.codes_desc)):
            curr_desc = self.codes_desc[desc].lower()
            occ = 0
            for tkn in tokens:
                if tkn.lower() in curr_desc:
                    if desc not in idxs:
                        idxs.append(desc)
                        occ += 1
        return idxs

    def extract_max_similarity(self, tokens):
        '''
        :param tokens: the tokens of the text input
        :param text_input: the skill/occupation input the user wants to extract the ESCO code from
        :return The index of the descriptor that maximizes the cosine similarity
        '''
        input_encodings = self.ext_model.encode(tokens, device=self.hardware,
                                                 normalize_encodings=True, convert_to_tensor=True) #encodings for the input
        dot_prod = sentence_transformers.util.dot_score(input_encodings, self.encodings_proc) #compute the cosine similarity (dot product)
        scores, idxs = torch.max(dot_prod.T, dim=0) #top scores and indexes
        max_score = np.where(scores.numpy() == np.max(scores.numpy())) #index of the max score

        return idxs[max_score]

    def code_from_input(self, text_input, translate=False):
        '''
        :param text_input: the skill/occupation input the user wants to extract the ESCO code from
        :param translate: flag to translate the text_input from portuguese to english
        :return: the ESCO code for the given input in string format and the URL
        '''
        #Error handlings
        if text_input is None:
            raise ValueError(f'[esco_extractor.extract_code_from_input] text_input is None')
        if text_input == '':
            raise ValueError(f'[esco_extractor.extract_code_from_input] text_input is an empty string')
        if translate is None:
            raise ValueError(f'[esco_extractor.extract_code_from_input] translate is None')

        #Translate the text_input if required
        if translate:
            translator = Translator()
            text_input = translator.translate(text_input, src="pt", dest="en").text #translate the input from portuguese to english

        self.format_codes() #format the strings for text comparison
        self.format_encodings() #encode the reference descriptors
        text_input = text_input.lower() #apply lower case
        text_input = self.format_tokens(text_input) #format the text inputs

        #Pre-process the input
        if len(text_input) == 0:
            return '','' #if the text_input is empty, return an empty string
        else:
            tokens = text_input

        tkn_idxs = self.filter_token_occ(tokens) #find the indexes of the descriptors that contain at least one word in the input

        #Select only the entries with the tokens occurrences to speed up the search
        self.codes_id_proc = self.codes_id[tkn_idxs]
        self.codes_desc_proc = self.codes_desc[tkn_idxs]
        self.codes_ESCO_proc = self.codes_ESCO[tkn_idxs]
        self.encodings_proc = self.encodings[tkn_idxs]

        #Find the code based on the cosine similarity (dot product)
        esco_idx = self.extract_max_similarity(tokens) #extract the index of the max cosine similarity
        esco_code = np.unique(self.codes_ESCO_proc[esco_idx])[0].item() #the code that maximizes de cosine similarity
        esco_URL = np.unique(self.codes_id_proc[esco_idx])[0].item() #the code that maximizes de cosine similarity

        return esco_code, esco_URL
