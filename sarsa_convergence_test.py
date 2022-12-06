import subprocess
import multiprocessing
import os


def execute(process):
    os.system(f'python {process}')


if __name__ == '__main__':
    commands = []
    for i in range(1, 51):
        command = 'executor.py -folder ApproximateSarsaAgent_new_alpha2 -run_number ' + str(
            i) + ' -p ApproximateSarsaAgent -a extractor=ComplexExtractor -x 1000 -n 1000 -l mediumClassic'
        commands.append(command)

    process_pool = multiprocessing.Pool(processes=4)
    process_pool.map(execute, commands)
