import subprocess
from invoke import task


@task(default=True)
def run(context, path=None, expression=None, log=True):
    """
    Execute tests
    """
    params = ['pytest', '-x', '-s', '-v']

    if expression:
        params += ['-k', expression]

    if path is not None:
        params.append(path)
    else:
        params.append('tests/')
    subprocess.call(params, shell=True)

    if log:
        with open('_data/log/test.log', 'w') as f:
            subprocess.call(params, stdout=f)
