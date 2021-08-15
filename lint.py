'''
Find score of written code using pylint

This program will check the quality of code written using pylint

Final output is the score
'''
from pylint.lint import Run

results = Run(['main.py'], do_exit=False)

print(results.linter.stats['global_note'])