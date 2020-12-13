import subprocess
from invoke import task


@task(default=True)
def run(context, path=None, expression=None):
    """
    Execute tests
    """
    params = ['pytest', '-x', '-s', '--cov=super_resolution', '--cov-report=term-missing',
              '--ignore=env', '--confcutdir=tests']

    if expression:
        params += ['-k', expression]

    if path is not None:
        params.append(path)
    else:
        params.append('tests/')

    subprocess.run(params, check=True)
