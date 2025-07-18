# Software description
The esco_extractor repo contains a solution for the task of assigning an ESCO code to a job title. The solution implements the "all-MiniLM-L6-v2" sentence transformer model to compare the input string with all the descriptors collected from the European Commission database for the ESCO codes regarding occupations. The solution builds upon another [repository](https://github.com/KonstantinosPetrakis/esco-skill-extractor). However, it is optimized for the Portuguese language instead of English. Also, the goal is only to yield the code for the descriptor with the highest similarity between the job title input. The optimization task is done based on the dot product of the input tokens with the occupations tokens
<p align="center">
 <img src="https://github.com/user-attachments/assets/bd38dbff-1966-4e56-89ae-07d2f7f114b8" width="400" />
</p>
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
