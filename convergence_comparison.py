import subprocess
j=0
for i in range(1,101):
    if j %6==0:
        j=0
    layouts = ['trickyClassic','powerClassic','capsuleClassic','originalClassic','mediumGrid','smalllGrid']
    feat_extractors =['IdentityExtractor','SimpleExtractor','BetterExtractor','IdentityExtractor','BetterExtractor','BetterExtractor']
    ghostNum  = [2,3,4,4,3,3]
    command = 'python executor.py -folder ApproximateSarsaAgent -run_number '+ \
                    str(i) + ' -p ApproximateSarsaAgent -a extractor='+ feat_extractors[j] +' -x 1000 -n 1000 -l ' + \
                    layouts[j] +' -k '+ str(ghostNum[j])
    j+=1
    print(command)
    subprocess.call(command, shell=True)