# Software description
The esco_extractor repo contains a solution for the task of assigning an ESCO code to a job title. The solution implements the "all-MiniLM-L6-v2" sentence transformer model to compare the input string with all the descriptors collected from the European Commission database for the ESCO codes regarding occupations. The solution builds upon another [repository](https://github.com/KonstantinosPetrakis/esco-skill-extractor). However, it is optimized for the Portuguese language instead of English. Also, the goal is only to yield the code for the descriptor with the highest similarity between the job title input. The optimization task is done based on the dot product of the input tokens with the occupations tokens

![Screenshot from 2025-03-25 05-29-40](https://github.com/user-attachments/assets/58cce0a1-1ac8-4b94-931a-5717fec44f79)

where d is the index that maximizes the dot product between the encoded arrays. The output is the ESCO code and the URI of the assigned occupation.

# Running the model
The esco_extractor repo software is recommended to run as an auxiliary tool for web scrappers. Given the software was built as a package, to integrate it into any other framework, it is only required to run the ESCOextractor class constructor and call the extractor function providing a job title in Portuguese
```
ext_obj = extractor_utils.ESCOextractor()
code, uri = ext_obj.code_from_input(<job_title_PT>)
```
If the job title is only available in English, the extractor function has a flag to trigger Google Translator's API to translate the input from English to Portuguese. But, it might take more time to compute the outputs.
```
code, uri = ext_obj.code_from_input(<job_title_EN>, translate=True)
```
