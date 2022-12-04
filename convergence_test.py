import subprocess

for i in range(1, 101):
    command = 'python executor.py -folder ApproximateSarsaAgent -run_number ' + str(
        i) + ' -p ApproximateSarsaAgent -a extractor=BetterExtractor -x 1000 -n 1000 -l mediumClassic'
    print(command)
    subprocess.call(command, shell=True)
