import subprocess

j = 0
for i in range(1, 5):
    if j % 6 == 0:
        j = 0
    layouts = ['trickyClassic', 'powerClassic', 'capsuleClassic', 'originalClassic', 'mediumGrid', 'smallGrid']
    feat_extractors = ['IdentityExtractor', 'SimpleExtractor', 'ComplexExtractor', 'IdentityExtractor',
                       'ComplexExtractor', 'ComplexExtractor']
    ghostNum = [2, 3, 4, 4, 3, 3]
    command = 'python executor.py -folder run_layouts/ApproximateSarsaAgent_' + layouts[j] + ' -run_number ' + \
              str(i) + ' -p ApproximateSarsaAgent -a extractor=' + feat_extractors[j] + ' -x 100 -n 100 -l ' + \
              layouts[j] + ' -k ' + str(ghostNum[j])
    j += 1
    print(command)
    subprocess.call(command, shell=True)
