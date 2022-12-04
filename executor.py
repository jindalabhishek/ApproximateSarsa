import sys
import qlearningAgents
import pacman

qlearningAgents.folder = sys.argv[2]
qlearningAgents.run_number = sys.argv[4]
args_list = sys.argv[5:]
args = pacman.readCommand(args_list)
pacman.runGames(**args)
