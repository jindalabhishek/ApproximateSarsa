import subprocess
import multiprocessing
import os


def execute(process):
    os.system(f'python {process}')


if __name__ == '__main__':
    commands = []
    for i in range(1, 101):
        command = 'executor.py -folder ApproximateSarsaAgent -run_number ' + str(
        i) + ' -p ApproximateSarsaAgent -a extractor=BetterExtractor -x 1000 -n 1000 -l mediumClassic'
        commands.append(command)

    process_pool = multiprocessing.Pool(processes=8)
    process_pool.map(execute, commands)
