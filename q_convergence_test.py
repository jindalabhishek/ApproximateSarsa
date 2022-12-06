import multiprocessing
import os


def execute(process):
    os.system(f'python {process}')


if __name__ == '__main__':
    commands = []
    for i in range(1, 51):
        command = 'executor.py -folder ApproximateQAgent_ComplexExtractor -run_number ' + str(
            i) + ' -p ApproximateQAgent -a extractor=ComplexExtractor -x 100 -n 100 -l mediumClassic'
        commands.append(command)

    process_pool = multiprocessing.Pool(processes=4)
    process_pool.map(execute, commands)
