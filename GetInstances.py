import requests, os, gzip, shutil

hard = set({})
openInst = set({})
infeasible = set({})

with open(r'problems_list/hardInstances.txt') as hf:
    for line in hf:
        line = line.rstrip('\n')
        hard = hard | {line}

with open(r'problems_list/infeasibleInstances.txt') as inff:
    for line in inff:
        line = line.rstrip('\n')
        infeasible = infeasible | {line}

with open(r'problems_list/openInstances.txt') as of:
    for line in of:
        line = line.rstrip('\n')
        openInst = openInst | {line}


with open(r'problems_list/binary.txt') as bf:
    for line in bf:
        line = line.rstrip('\n')
        url = "https://miplib.zib.de/WebData/instances/" + line

        response = requests.get(url, stream=True)

        hardFolder = r"problems/hard_problems/"
        easyFolder = r"problems/easy_problems/"
        infeasibleFolder = r"problems/infeasible_problems/"
        openFolder = r"problems/open_problems/"
        if not os.path.exists(hardFolder):
            os.makedirs(hardFolder)
        
        if not os.path.exists(infeasibleFolder):
            os.makedirs(infeasibleFolder)
        
        if not os.path.exists(easyFolder):
            os.makedirs(easyFolder)
        
        if not os.path.exists(openFolder):
            os.makedirs(openFolder)

        if line in hard:
            with open('problems/temp.gz', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open('problems/temp.gz', 'rb') as f_in:
                with open(hardFolder + line.split('.')[0] + '.mps', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif line in openInst:
            with open('problems/temp.gz', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open('problems/temp.gz', 'rb') as f_in:
                with open(openFolder + line.split('.')[0] + '.mps', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif line in infeasible:
            with open('problems/temp.gz', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open('problems/temp.gz', 'rb') as f_in:
                with open(infeasibleFolder + line.split('.')[0] + '.mps', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            with open('problems/temp.gz', 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open('problems/temp.gz', 'rb') as f_in:
                with open(easyFolder + line.split('.')[0] + '.mps', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
