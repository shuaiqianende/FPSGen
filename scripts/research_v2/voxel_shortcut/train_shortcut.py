import sys
from train_core import main
if __name__=='__main__':
    stage='surface' if '--surface' in sys.argv else 'shortcut'
    if '--surface' in sys.argv:sys.argv.remove('--surface')
    main(stage)
