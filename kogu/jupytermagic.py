from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core import magic_arguments
from IPython.utils.capture import capture_output
import tempfile
import subprocess
import os


@magics_class
class KoguExecution(Magics):
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--name', type=str, default='', nargs='?',
                              help="""Name of the experiment""")
    @magic_arguments.argument('--path', type=str, default='kogu', nargs='?',
                              help="""Path to kogu excecutable""")
    @cell_magic
    def kogu(self, line, cell):
        """run the cell, capturing stdout, stderr, and IPython's rich display() calls."""
        args = magic_arguments.parse_argstring(self.kogu, line)
        with capture_output(stdout=True, stderr=True, display=False) as io:
            self.shell.run_cell(cell)

        f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        f.write(io.stdout)
        f.write(io.stderr)

        fname = f.name
        f.close()
        print(io.stdout)
        experiment_name = args.name
        run_args = [args.path, "run", "-e=cat", fname]
        if experiment_name is not None and len(experiment_name) > 1:
            run_args.append("-n=" + experiment_name)

        subprocess.check_call(run_args)
        os.remove(fname)
