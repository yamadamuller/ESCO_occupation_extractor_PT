import extractor_utils
import time

job_title = 'Engenheiro eletricista'
t_init_1 = time.time()
ext_obj = extractor_utils.ESCOextractor()
t_init_2 = time.time()
code, uri = ext_obj.code_from_input(job_title)
print(f'Calling the constructor each time: t_proc = {time.time()-t_init_1}')
print(f'Calling only the extractor each time: t_proc = {time.time()-t_init_2}')
print(f'ESCO code = {code}')
print(f'ESCO URI = {uri}')

