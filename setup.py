from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT ='-e .' # This is the last line in requirements.txt file which triggers setup.py; we mush remove this from the list
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements needed for this project    
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name='ML_Project',
version='0.0.1',
author='Arindam C',
author_email='a.c.e@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)